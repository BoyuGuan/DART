# -*- coding: utf-8 -*-
# 同时保存SFT数据和对应的源数据，并生成支持多Cue并列逻辑的RL数据

import argparse
import json
import os
import random
import logging
from typing import Dict, Any, List, Tuple
import datasets

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./log/process_metrics.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 需要排除的字段
EXCLUDED_FIELDS = {"translation_baseline", "translation_all_cues"}

def get_clean_cue_content(item: Dict[str, Any], cue_type: str) -> str:
    """
    从 mm_prompt_{cue_type} 中提取纯净的视频信息内容。
    """
    prompt_key = f"mm_prompt_{cue_type.replace('translation_', '')}"
    full_prompt = item.get(prompt_key, "")
    
    if not full_prompt:
        return "No information provided."

    if "Video cue:" in full_prompt:
        content = full_prompt.split("Video cue:")[-1].strip()
        return content
    
    return full_prompt

def construct_video_data(video_rel_path: str) -> List[Dict[str, Any]]:
    """
    构造符合 Qwen2-VL/Verl 要求的视频数据格式。
    将相对路径转换为绝对路径，并包装为字典结构。
    """
    abs_path = os.path.abspath(video_rel_path)
    # 构造符合 qwen2-vl 要求的格式
    video_info = {
        "type": "video",
        "video": f"file://{abs_path}",
        "fps": 2.0,       # 默认推荐设置
        "min_frames": 4,  # 默认推荐设置
        "max_frames": 768 # 默认推荐设置
    }
    # 返回列表格式以支持 verl 的数据处理流程
    return [video_info]

def generate_sft_item(item: Dict[str, Any], 
                      sft_type: str, 
                      cue_key: str = None, 
                      cue_content: str = None) -> Dict[str, Any]:
    """
    构造符合要求的 SFT 数据格式 (CoT 风格)。
    """
    video_id = item['video_id']
    clip_id = item['clip_id']

    # 确定源语言和目标语言
    src_lang = item.get('src_lang', '').strip().lower()
    
    if not src_lang:
        lang_field = item.get('language', '').lower()
        if lang_field.startswith('en'):
            src_lang = 'en'
        elif lang_field.startswith('zh'):
            src_lang = 'zh'
        else:
            raise ValueError(f"Cannot determine source language for item with video_id: {video_id}")

    # 根据源语言决定 Source 和 Target(Ref)
    if src_lang == 'zh':
        src_sentence = item.get('ZH_sentence', '')
        target_ref = item.get('EN_sentence', '')
        source_language = "Chinese"
        target_language = "English"
    else:
        src_sentence = item.get('EN_sentence', '')
        target_ref = item.get('ZH_sentence', '')
        source_language = "English"
        target_language = "Chinese"

    # --- 1. 构造统一的 Human 输入 ---
    human_prompt = (
        f"<video>\n"
        f"Please translate the following input sentence from {source_language} to {target_language} according to the video. ONLY output the translated sentence.\n"
        f"Input sentence:\n"
        f"{src_sentence}\n"
    )

    # --- 2. 构造包含思维链（CoT）的 GPT 回答 ---
    if sft_type == "baseline":
        # Case: 不需要视频信息 (Baseline 最好)
        gpt_response = (
            f"This text can be translated **without** video information.\n"
            f"The translation is:\n{target_ref}"
        )

    else:
        # Case: 需要视频信息 (Visual 最好)
        cue_name = cue_key.replace("translation_", "")
        gpt_response = (
            f"To translate this text, video information [{cue_name}] is **required**.\n"
            f"It is:\n{cue_content}\n"
            f"So the translation is:\n{target_ref}"
        )

    # 构造 Video 路径并转换为字典格式
    video_path_rel = f"./data/TriFine/videoClips/{video_id}/{video_id}_{clip_id}.mp4"
    video_data = construct_video_data(video_path_rel)

    return {
        "video": video_data, # 修改此处：传入构造好的列表字典
        "conversations": [
            {
                "from": "human",
                "value": human_prompt
            },
            {
                "from": "gpt",
                "value": gpt_response
            }
        ]
    }

def prepare_rl_candidates(data: List[Dict[str, Any]], diff_threshold: float = 2.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    准备 RL 数据的候选集，分为两组：
    - Group A: visual enhanced (comet_diff > threshold). 
    - Group B: baseline best
    """
    group_a_candidates = []
    group_b_candidates = []
    
    for item in data:
        metrics = item.get("translation_metrics", {})
        if not metrics:
            continue

        baseline_metrics = metrics.get("translation_baseline")
        if not baseline_metrics:
            continue
        
        baseline_comet = baseline_metrics["COMET"]
        visual_keys = [k for k in metrics.keys() if k not in EXCLUDED_FIELDS]
        
        if not visual_keys:
            group_b_candidates.append(item)
            continue
        
        # 1. 计算该 item 所有 visual cue 的 diff
        cue_diffs = []
        for k in visual_keys:
            score = metrics[k]["COMET"]
            diff = score - baseline_comet
            cue_diffs.append((k, score, diff))
        
        if not cue_diffs:
            group_b_candidates.append(item)
            continue
            
        # 找出该 item 下最大的 diff
        max_diff = max(x[2] for x in cue_diffs)
        
        # 2. 判定归属 Group A 还是 Group B
        if max_diff > diff_threshold:
            # Group A: 找出所有等于 max_diff 的 cues (处理平局)
            epsilon = 1e-9
            best_candidates = [x for x in cue_diffs if abs(x[2] - max_diff) < epsilon]
            
            # 记录所有并列最佳的 cue name (去掉 "translation_" 前缀)
            tied_cue_names = [x[0].replace("translation_", "") for x in best_candidates]
            
            # 为每一个并列最佳的 cue 生成一个候选样本
            for cue_key, score, diff in best_candidates:
                group_a_candidates.append({
                    "item": item,
                    "cue_key": cue_key,
                    "best_visual_comet": score,
                    "comet_diff": diff,
                    "tied_cues": tied_cue_names 
                })
        else:
            # Group B
            group_b_candidates.append(item)
    
    # Group A 按 comet_diff 全局降序排序
    group_a_candidates.sort(key=lambda x: x["comet_diff"], reverse=True)
    
    return group_a_candidates, group_b_candidates

def generate_rl_data(item: Dict[str, Any], 
                      cue_key: str = None,
                      rl_type: str = "visual",
                      tied_cues: List[str] = None,
                      data_source: str = "DART_VMT") -> Dict[str, Any]:
    """
    生成 RL 格式的数据。
    """
    video_id = item['video_id']
    clip_id = item['clip_id']
    
    # 确定源语言和目标语言
    src_lang = item.get('src_lang', '').strip().lower()
    if not src_lang:
        lang_field = item.get('language', '').lower()
        if lang_field.startswith('en'):
            src_lang = 'en'
        elif lang_field.startswith('zh'):
            src_lang = 'zh'
        else:
            raise ValueError(f"Cannot determine source language for item with video_id: {video_id}")
    
    if src_lang == 'zh':
        src_sentence = item.get('ZH_sentence', '')
        target_ref = item.get('EN_sentence', '')
        source_language = "Chinese"
        target_language = "English"
    else:
        src_sentence = item.get('EN_sentence', '')
        target_ref = item.get('ZH_sentence', '')
        source_language = "English"
        target_language = "Chinese"
    
    # 构造 prompt
    prompt_text = (
        f"<video>\n"
        f"Please translate the following input sentence from {source_language} to {target_language} according to the video. ONLY output the translated sentence.\n"
        f"Input sentence:\n"
        f"{src_sentence}\n"
    )
    
    # 构造 Video 数据（修复报错的关键部分）
    video_path_rel = f"./data/TriFine/videoClips/{video_id}/{video_id}_{clip_id}.mp4"
    video_data = construct_video_data(video_path_rel)
    
    metrics = item.get("translation_metrics", {})
    baseline_comet = metrics.get("translation_baseline", {}).get("COMET", 0.0)
    
    extra_info = {
        "video_id": video_id,
        "clip_id": clip_id,
        "source_sentence": src_sentence,
        "target_reference": target_ref,
        "source_language": source_language,
        "target_language": target_language,
        "baseline_comet": baseline_comet,
        "rl_type": rl_type,
    }
    
    if rl_type == "visual" and cue_key:
        cue_content = get_clean_cue_content(item, cue_key)
        cue_name = cue_key.replace("translation_", "")
        cue_comet = metrics.get(cue_key, {}).get("COMET", 0.0)
        
        extra_info.update({
            "current_cue_type": cue_name,   
            "best_cue_types": tied_cues if tied_cues else [cue_name], 
            "cue_content": cue_content,
            "best_cue_comet": cue_comet,
            "comet_diff": cue_comet - baseline_comet,
        })
    else:
        extra_info.update({
            "current_cue_type": "baseline",
            "best_cue_types": ["baseline"],
            "cue_content": "No visual cue needed",
            "best_cue_comet": baseline_comet,
            "comet_diff": 0.0,
        })
    
    rl_data = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "video": video_data, # 修改此处：传入构造好的列表字典
        "ability": "video_translation",
        "reward_model": {
            "style": "metric",
            "ground_truth": target_ref,
            "metric": "COMET",
        },
        "extra_info": extra_info,
    }
    
    return rl_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="./data/work3/MMinfoAndTrans/promptsAndTransMetrics.json")
    parser.add_argument("--output_json", type=str, default="./data/work3/sftData/sft_MMInfo_train_data.json")
    parser.add_argument("--output_source_json", type=str, default="./data/work3/meta_train_data.json", help="Save corresponding source items from input_json")
    parser.add_argument("--output_rl_dir", type=str, default="./data/work3/rl_data", help="Directory to save RL data in parquet format")
    parser.add_argument("--alpha", type=float, default=0.6, help="Ratio of visual-enhanced samples (alpha) to baseline samples (1-alpha)")
    parser.add_argument("--comet_diff", type=float, default=2.0, help="Threshold for COMET improvement over baseline")
    parser.add_argument("--rl_train_size", type=int, default=10000, help="Number of RL training samples")
    parser.add_argument("--rl_val_size", type=int, default=300, help="Number of RL validation samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input file not found: {args.input_json}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} items from {args.input_json}")

    # ==========================
    # SFT Data Generation (Task 1-3)
    # ==========================
    
    count_diff_gt_1 = 0
    sft_group_a_candidates = [] 
    sft_group_b_candidates = []

    for item in data:
        metrics = item.get("translation_metrics", {})
        if not metrics:
            continue

        baseline_metrics = metrics.get("translation_baseline")
        if not baseline_metrics:
            continue
        
        baseline_comet = baseline_metrics["COMET"]
        visual_keys = [k for k in metrics.keys() if k not in EXCLUDED_FIELDS]
        
        # SFT 逻辑 1: 筛选 Group A
        better_cues = []
        for k in visual_keys:
            score = metrics[k]["COMET"]
            if score > baseline_comet + args.comet_diff:
                better_cues.append((k, score))

        if better_cues:
            count_diff_gt_1 += 1
            max_score = max(c[1] for c in better_cues)
            best_cues_for_item = [c for c in better_cues if c[1] == max_score]
            
            for cue_key, score in best_cues_for_item:
                sft_group_a_candidates.append({
                    "item": item,
                    "cue_key": cue_key,
                    "score": score
                })
        
        # SFT 逻辑 2: 筛选 Group B
        is_baseline_highest = True
        for k in visual_keys:
            if metrics[k]["COMET"] > baseline_comet:
                is_baseline_highest = False
                break
        
        if is_baseline_highest:
            sft_group_b_candidates.append({
                "item": item,
                "score": baseline_comet
            })

    logger.info(f"-" * 30)
    logger.info(f"SFT Candidates (Task 1): Samples with Visual > Base + {args.comet_diff}: {count_diff_gt_1}")
    
    # SFT 采样
    sft_len_a = len(sft_group_a_candidates)
    sft_len_b = len(sft_group_b_candidates)
    
    sft_target_total = min(sft_len_a / args.alpha, sft_len_b / (1.0 - args.alpha))
    sft_target_a = int(sft_target_total * args.alpha)
    sft_target_b = int(sft_target_total * (1.0 - args.alpha))
    
    sft_sampled_a = random.sample(sft_group_a_candidates, sft_target_a) if sft_len_a >= sft_target_a else sft_group_a_candidates
    sft_sampled_b = random.sample(sft_group_b_candidates, sft_target_b) if sft_len_b >= sft_target_b else sft_group_b_candidates
    
    # 构造 SFT 数据
    sft_pairs = []
    for obj in sft_sampled_a:
        item = obj['item']
        cue_key = obj['cue_key']
        cue_content = get_clean_cue_content(item, cue_key)
        sft_item = generate_sft_item(item, sft_type="visual", cue_key=cue_key, cue_content=cue_content)
        sft_pairs.append((sft_item, item))

    for obj in sft_sampled_b:
        item = obj['item']
        sft_item = generate_sft_item(item, sft_type="baseline")
        sft_pairs.append((sft_item, item))

    random.shuffle(sft_pairs)
    
    # 保存 SFT 数据
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump([p[0] for p in sft_pairs], f, ensure_ascii=False, indent=4)
        
    source_dir = os.path.dirname(args.output_source_json)
    if source_dir:
        os.makedirs(source_dir, exist_ok=True)
    with open(args.output_source_json, "w", encoding="utf-8") as f:
        json.dump([p[1] for p in sft_pairs], f, ensure_ascii=False, indent=4)
        
    logger.info(f"SFT data saved to: {args.output_json}")

    # ==========================
    # RL Data Generation (Task 4 - Updated)
    # ==========================
    logger.info(f"\n{'-' * 30}")
    logger.info("Generating RL data with updated logic (Tie-Aware Multi-Cue)...")
    
    # 1. 准备候选集
    group_a_candidates, group_b_candidates = prepare_rl_candidates(data, diff_threshold=args.comet_diff)
    
    len_a = len(group_a_candidates)
    len_b = len(group_b_candidates)
    
    logger.info(f"RL Candidates Pool -> Group A (Visual Diff > {args.comet_diff}): {len_a} samples")
    logger.info(f"RL Candidates Pool -> Group B (Baseline Best/Diff Low): {len_b} items")
    
    if len_a == 0 and len_b == 0:
        logger.warning("No valid samples found for RL data generation.")
        return

    # 2. 分别计算 train 和 val 的分组目标（都按 alpha 比例）
    train_target_a = int(args.rl_train_size * args.alpha)
    train_target_b = int(args.rl_train_size * (1.0 - args.alpha))
    val_target_a = int(args.rl_val_size * args.alpha)
    val_target_b = int(args.rl_val_size * (1.0 - args.alpha))
    
    total_target_a = train_target_a + val_target_a
    total_target_b = train_target_b + val_target_b
    
    logger.info(f"Target -> Train: {train_target_a} (A) + {train_target_b} (B) = {args.rl_train_size}")
    logger.info(f"Target -> Val: {val_target_a} (A) + {val_target_b} (B) = {args.rl_val_size}")
    
    # 3. 从候选池中采样
    # Group A: 按 comet_diff 排序后取 top
    sampled_a_all = group_a_candidates[:total_target_a]
    if len(sampled_a_all) < total_target_a:
        logger.warning(f"Not enough Group A candidates! Requested {total_target_a}, found {len(sampled_a_all)}.")
    
    # Group B: 随机采样
    sampled_b_all = random.sample(group_b_candidates, total_target_b) if len_b >= total_target_b else group_b_candidates
    if len(sampled_b_all) < total_target_b:
        logger.warning(f"Not enough Group B candidates! Requested {total_target_b}, found {len(sampled_b_all)}.")
    
    # 4. 将采样结果分割为 train 和 val（保持各自的 alpha 比例）
    # 打乱 Group A 和 Group B 内部顺序后分割
    random.shuffle(sampled_a_all)
    random.shuffle(sampled_b_all)
    
    train_sampled_a = sampled_a_all[:train_target_a]
    val_sampled_a = sampled_a_all[train_target_a:train_target_a + val_target_a]
    
    train_sampled_b = sampled_b_all[:train_target_b]
    val_sampled_b = sampled_b_all[train_target_b:train_target_b + val_target_b]
    
    logger.info(f"Final Selection -> Train: {len(train_sampled_a)} (A) + {len(train_sampled_b)} (B)")
    logger.info(f"Final Selection -> Val: {len(val_sampled_a)} (A) + {len(val_sampled_b)} (B)")
    
    # 5. 生成 RL 数据
    def build_rl_data_list(sampled_a, sampled_b):
        """根据采样结果构建 RL 数据列表"""
        rl_list = []
        # 处理 Group A
        for candidate in sampled_a:
            item = candidate['item']
            cue_key = candidate['cue_key']
            tied_cues = candidate['tied_cues']
            rl_item = generate_rl_data(
                item, 
                cue_key=cue_key, 
                rl_type="visual",
                tied_cues=tied_cues
            )
            rl_list.append(rl_item)
        # 处理 Group B
        for item in sampled_b:
            rl_item = generate_rl_data(item, rl_type="baseline")
            rl_list.append(rl_item)
        random.shuffle(rl_list)
        return rl_list
    
    rl_train_data = build_rl_data_list(train_sampled_a, train_sampled_b)
    rl_val_data = build_rl_data_list(val_sampled_a, val_sampled_b)
    
    # 6. 保存
    os.makedirs(args.output_rl_dir, exist_ok=True)
    
    if rl_train_data:
        rl_train_dataset = datasets.Dataset.from_list(rl_train_data)
        rl_train_path = os.path.join(args.output_rl_dir, "train.parquet")
        rl_train_dataset.to_parquet(rl_train_path)
        logger.info(f"RL training data saved to: {rl_train_path} ({len(rl_train_data)} samples)")
        
        vis_count = sum(1 for x in rl_train_data if x['extra_info']['rl_type'] == 'visual')
        multi_cue_count = sum(1 for x in rl_train_data if x['extra_info'].get('best_cue_types') and len(x['extra_info']['best_cue_types']) > 1)
        
        logger.info(f"  - Visual Enhanced: {vis_count}")
        logger.info(f"  - Baseline Best: {len(rl_train_data) - vis_count}")
        logger.info(f"  - Samples from Multi-Max-Cue Items: {multi_cue_count}")

    if rl_val_data:
        rl_val_dataset = datasets.Dataset.from_list(rl_val_data)
        rl_val_path = os.path.join(args.output_rl_dir, "val.parquet")
        rl_val_dataset.to_parquet(rl_val_path)
        logger.info(f"RL validation data saved to: {rl_val_path} ({len(rl_val_data)} samples)")

    logger.info(f"{'-' * 30}\n")

if __name__ == "__main__":
    main()