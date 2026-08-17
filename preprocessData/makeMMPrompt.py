"""
将多模态信息与翻译句子融合，生成不同类型的 prompt。

功能：
1. 读取 extractMM.py 的输出文件（包含 MMInfo 字段）
2. 根据每条数据的 language 字段自动判断翻译方向（en->zh 或 zh->en）
3. 建立 ID 到自然语言（角色、物品类别+属性）的映射
4. 生成去代码化（De-identified）、更自然的 Prompt
5. 针对空信息的模态（如 pointing_gaze: []），生成空字符串 prompt
6. 将所有 prompt 存储到原始数据中，输出为单个 JSON 文件
"""

from __future__ import annotations

from typing import Any, Dict, List
import argparse
import json
import os
import logging

# All supported cue types
ALL_CUE_TYPES = ["baseline", "people", "objects", "actions", "ocr", "spatial_relations", "pointing_gaze", "all_cues"]

PROMPT_PREAMBLE = (
    "You are a professional translator.\n"
    "Translate the source sentence into the target language.\n"
    "Use the provided video cue ONLY as auxiliary context when it helps disambiguate meaning.\n"
    "Do NOT invent any details beyond the cue.\n"
    "Output ONLY the final translation, with no explanation.\n"
)

def parse_language_field(language_str: str) -> tuple:
    """
    解析 language 字段，返回 (src_lang_code, src_lang_name, tgt_lang_code, tgt_lang_name)
    
    language 字段格式: "en: English" 或 "zh: Chinese"
    """
    lang_code = language_str.split(":")[0].strip().lower() if language_str else "en"
    
    if lang_code == "en":
        return ("en", "English", "zh", "Chinese")
    elif lang_code == "zh":
        return ("zh", "Chinese", "en", "English")
    else:
        # 默认 en -> zh
        return ("en", "English", "zh", "Chinese")


def build_all_cue_type_prompts(
    extraction: Dict[str, Any],
    source_sentence: str,
    src_lang_name: str = "English",
    tgt_lang_name: str = "Chinese",
) -> Dict[str, str]:
    """
    为每种 cue 类型生成一个合并后的 prompt 文本。
    此处进行了去代码化处理，将 P1, O1 等代号转换为自然语言描述。
    
    返回: Dict[cue_type, prompt_text]
    """
    people = extraction.get("people", []) or []
    objects = extraction.get("objects", []) or []
    actions = extraction.get("actions", []) or []
    ocr = extraction.get("ocr", []) or []
    spatial = extraction.get("spatial_relations", []) or []
    gaze = extraction.get("pointing_gaze", []) or []
    
    # ---------------------------------------------------------
    # Step 1: 建立 ID 到自然语言描述的映射 (ID Mapping)
    # ---------------------------------------------------------
    id_map = {}

    # 1.1 映射 People (P1 -> "the cook")
    for p in people:
        if not isinstance(p, dict): continue
        pid = p.get("person_id", "unknown")
        role = (p.get("role_guess") or {}).get("role", "person")
        # 如果 role 是 unknown，就叫 "the person"
        desc = role if role != "unknown" else "the person"
        id_map[pid] = desc

    # 1.2 映射 Objects (O1 -> "green vegetable")
    for o in objects:
        if not isinstance(o, dict): continue
        oid = o.get("object_id", "unknown")
        cat = o.get("category", "object")
        attrs = o.get("attributes", {})
        
        # 构造自然的物品描述：优先使用 label，其次 color，最后 category
        # 格式示例: "black soy sauce bottle" 或 "green vegetable"
        label = attrs.get("label")
        color = attrs.get("color")
        
        desc_parts = []
        if color and color != "unknown":
            desc_parts.append(str(color))
        
        # 如果 label 存在且不是 unknown，通常是最关键的信息
        if label and label != "unknown":
            main_desc = f"{cat} ({label})"
        else:
            main_desc = cat
            
        desc_parts.append(main_desc)
        
        full_desc = " ".join(desc_parts)
        id_map[oid] = full_desc

    # 辅助函数：根据 ID 获取描述，如果找不到则返回 ID 原文
    def get_desc(ref_id):
        if ref_id == "unknown" or not ref_id:
            return "unknown"
        # 尝试直接匹配
        if ref_id in id_map:
            return id_map[ref_id]
        # 有些数据可能引用了 Stove 等不在 object 列表里的环境物体，直接把下划线换空格
        return ref_id.replace("_", " ")

    # ---------------------------------------------------------
    # Step 2: 定义各种 Verbalizer (自然语言生成器)
    # ---------------------------------------------------------

    def verb_people(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict): continue
            pid = it.get("person_id")
            name = id_map.get(pid, "Unknown person")
            lines.append(f"- {name}")
        return "\n".join(lines) if lines else ""
    
    def verb_objects(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict): continue
            oid = it.get("object_id")
            name = id_map.get(oid, "Unknown object")
            attrs = it.get("attributes", {})
            extras = []
            if attrs.get("state") and attrs["state"] != "unknown":
                extras.append(f"state: {attrs['state']}")
            line = f"- {name}"
            if extras:
                line += f" [{', '.join(extras)}]"
            lines.append(line)
        return "\n".join(lines) if lines else ""
    
    def verb_actions(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict): continue
            pred = it.get("predicate", "action").replace("_", " ")
            agent = get_desc(it.get("agent_id"))
            patient = get_desc(it.get("patient_id"))
            inst = get_desc(it.get("instrument_id"))
            sent = f"- {agent} {pred}"
            if patient and patient != "unknown":
                sent += f" {patient}"
            if inst and inst != "unknown":
                sent += f" using {inst}"
            lines.append(sent)
        return "\n".join(lines) if lines else ""
    
    def verb_ocr(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict): continue
            text = it.get("text", "")
            if text:
                lines.append(f"- \"{text}\"")
        return "\n".join(lines) if lines else ""
    
    def verb_spatial(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict): continue
            subj = get_desc(it.get("subject_id"))
            rel = it.get("relation", "near").replace("_", " ") 
            obj = get_desc(it.get("object_id"))
            lines.append(f"- {subj} is {rel} {obj}")
        return "\n".join(lines) if lines else ""
    
    def verb_gaze(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict): continue
            src = get_desc(it.get("source_id"))
            typ = it.get("type", "gaze")
            verb = "looks at" if typ == "gaze" else "points to"
            tgt_id = it.get("target_id")
            tgt_desc = it.get("target_description")
            if tgt_id and tgt_id in id_map:
                target = id_map[tgt_id]
            elif tgt_desc:
                target = tgt_desc
            else:
                target = tgt_id if tgt_id else "unknown"
            lines.append(f"- {src} {verb} {target}")
        return "\n".join(lines) if lines else ""
    
    # ---------------------------------------------------------
    # Step 3: 生成 Prompt 文本
    # ---------------------------------------------------------
    cue_texts = {}
    
    # Baseline 始终有值
    cue_texts["baseline"] = "No video cue is provided. Translate using text only."
    
    # 其他类别，如果没有内容则为空字符串
    cue_texts["people"] = f"People in video:\n{verb_people(people)}" if people else ""
    cue_texts["objects"] = f"Objects visible in video:\n{verb_objects(objects)}" if objects else ""
    cue_texts["actions"] = f"Actions observed:\n{verb_actions(actions)}" if actions else ""
    cue_texts["ocr"] = f"On-screen text (OCR):\n{verb_ocr(ocr)}" if ocr else ""
    cue_texts["spatial_relations"] = f"Spatial relations:\n{verb_spatial(spatial)}" if spatial else ""
    cue_texts["pointing_gaze"] = f"Attention (Gaze/Pointing):\n{verb_gaze(gaze)}" if gaze else ""
    
    # All cues logic
    all_cue_parts = [v for k, v in cue_texts.items() if k != "baseline" and v]
    # 注意：all_cues 这里如果为空，默认还是给了一句 "No video cues available." 
    # 如果你也希望 all_cues 在全空时返回空字符串，可以将 else 后面的内容改为 ""
    cue_texts["all_cues"] = "Video cues:\n" + "\n\n".join(all_cue_parts) if all_cue_parts else "No video cues available."
    
    # Compose full prompts
    prompts = {}
    for cue_type, cue_text in cue_texts.items():
        # baseline 必须生成；其他 cue 若为空字符串则直接返回空 prompt
        if cue_type == "baseline" or cue_text:
            user_prompt = (
                f"{PROMPT_PREAMBLE}\n"
                f"Task: Translate from {src_lang_name} to {tgt_lang_name}.\n"
                f"Source sentence:\n{source_sentence}\n"
                f"Video cue:\n{cue_text}\n"
            )
            prompts[cue_type] = user_prompt
        else:
            prompts[cue_type] = ""

    
    return prompts


# -----------------------------
# Main program for batch processing
# -----------------------------
if __name__ == "__main__":
    logger = logging.getLogger('makeMMPrompt')
    formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="将多模态信息与翻译句子融合生成不同类型的 prompt")
    parser.add_argument("--inputFilePath", type=str, required=True,
                        help="输入文件路径（extractMM.py 的输出，包含 MMInfo 字段）")
    parser.add_argument("--outputFilePath", type=str, default="./data/work3/MMPrompts/data_with_prompts.json",
                        help="输出文件路径")
    parser.add_argument("--cueTypes", type=str, default="all",
                        help="要生成的 cue 类型")
    parser.add_argument("--logDir", type=str, default="./log",
                        help="日志目录")
    parser.add_argument("--onlyValidMMInfo", action="store_true",
                        help="是否只处理 MMInfoValid=True 的数据")
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.logDir, exist_ok=True)
    fileHandler = logging.FileHandler(f'{args.logDir}/makeMMPrompt.log')
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # Determine cue types
    if args.cueTypes.lower() == "all":
        selected_cue_types = ALL_CUE_TYPES
    else:
        selected_cue_types = [t.strip() for t in args.cueTypes.split(",")]
    
    logger.info(f"输入: {args.inputFilePath}")
    logger.info(f"输出: {args.outputFilePath}")
    
    # Read input
    try:
        with open(args.inputFilePath, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"读取失败: {e}")
        raise
    
    # Stats
    total_count = 0
    processed_count = 0
    
    output_data = []
    
    for idx, item in enumerate(input_data):
        total_count += 1
        
        # Check validity
        if args.onlyValidMMInfo and not item.get("MMInfoValid", False):
            continue
        
        # Parse Language
        lang_field = item.get("language", "en: English")
        src_code, src_name, tgt_code, tgt_name = parse_language_field(lang_field)
        src_sent = item.get(f"{src_code.upper()}_sentence", "")
        
        if not src_sent:
            continue
        
        # Build Prompts
        prompts = build_all_cue_type_prompts(
            extraction=item.get("MMInfo", {}),
            source_sentence=src_sent,
            src_lang_name=src_name,
            tgt_lang_name=tgt_name,
        )
        
        # Save results
        out_item = item.copy()
        out_item["src_lang"] = src_code
        out_item["tgt_lang"] = tgt_code
        
        for c_type in selected_cue_types:
            # 如果在 prompts 中找不到，默认为 baseline。
            # 但现在的逻辑下，只要 c_type 在 ALL_CUE_TYPES 中，prompts 都会有 key (可能是空字符串)
            out_item[f"mm_prompt_{c_type}"] = prompts.get(c_type, prompts["baseline"])
        
        output_data.append(out_item)
        processed_count += 1
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processing {idx+1}...")
    
    # Save Output
    os.makedirs(os.path.dirname(args.outputFilePath), exist_ok=True)
    with open(args.outputFilePath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Done. Processed: {processed_count}/{total_count}")