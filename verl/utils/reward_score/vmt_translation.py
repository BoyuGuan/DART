# Copyright 2024 Bytedance Ltd. and/or its affiliates
# VMT Translation Reward Function with Remote COMET & Logic Check (Multi-Cue Support)

import re
import requests
import math
import logging
import random
from typing import Optional, Dict, Any, Tuple, List, Union
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# -----------------------------
# COMET 服务地址（8 个端口随机分配）
# -----------------------------
COMET_HOST = "172.18.31.29"
COMET_PORTS = list(range(10080, 10088))  # 10080-10087
COMET_ENDPOINT = "/compute_score"
COMET_URLS = [f"http://{COMET_HOST}:{p}{COMET_ENDPOINT}" for p in COMET_PORTS]

# 复用连接，减少建连开销
_HTTP = requests.Session()


def _compute_remote_comet(src_list: list, pred_list: list, ref_list: list) -> list:
    """
    调用远程 API 计算 COMET 分数。
    - 每次调用会随机选择 10080-10087 中的一个端口；
    - 若所选端口失败，会在其余端口中继续重试（随机顺序），直到成功或全部失败。
    """
    if not src_list:
        return []

    payload = {
        "src": src_list,
        "preds": pred_list,
        "refs": ref_list
    }

    # 随机打散端口顺序，实现“随机分配 + 故障重试”
    urls = COMET_URLS[:]
    random.shuffle(urls)

    last_err = None
    for url in urls:
        try:
            response = _HTTP.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if isinstance(result, dict) and "scores" in result:
                return result["scores"]
            elif isinstance(result, list):
                return result
            else:
                return [0.0] * len(src_list)

        except Exception as e:
            last_err = e
            # 端口/实例可能暂时不可用，继续尝试其他端口
            continue

    logger.error(f"COMET API call failed on all ports {COMET_PORTS}: {last_err}")
    return [0.0] * len(src_list)


def _compute_text_similarity(text_a: str, text_b: str) -> float:
    """
    计算两个短文本的相似度。
    """
    if not text_a or not text_b:
        return 0.0
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


class CoTParser:
    """
    解析模型输出的思维链（Chain of Thought）。
    """
    # 捕获组 1: cue_name (如果需要)
    # 捕获组 2: required 关键词
    RE_REQUIREMENT = re.compile(
        r"video information (?:\[(.*?)\] )?is \*\*(required|not required)\*\*",
        re.IGNORECASE
    )
    RE_TRANSLATION = re.compile(r"So the translation is:\s*(.*)", re.DOTALL | re.IGNORECASE)

    @staticmethod
    def parse(predict_str: str) -> Dict[str, Any]:
        result = {
            "video_required_pred": None,  # bool
            "cue_type_pred": None,        # str
            "translation_pred": ""        # str
        }

        if not predict_str:
            return result

        req_match = CoTParser.RE_REQUIREMENT.search(predict_str)
        if req_match:
            cue_content_raw = req_match.group(1)
            status_str = req_match.group(2).lower()

            if "not required" in status_str:
                result["video_required_pred"] = False
                result["cue_type_pred"] = "baseline"
            else:
                result["video_required_pred"] = True
                result["cue_type_pred"] = cue_content_raw.strip() if cue_content_raw else ""

        trans_match = CoTParser.RE_TRANSLATION.search(predict_str)
        if trans_match:
            result["translation_pred"] = trans_match.group(1).strip()
        else:
            lines = predict_str.strip().split('\n')
            if len(lines) > 0:
                result["translation_pred"] = lines[-1].strip()

        return result


def length_penalty_reward(predict_str: str, limit: int = 500, alpha: float = 0.001) -> float:
    """
    对过长的输出施加指数级增长的惩罚。
    
    Args:
        predict_str: 模型生成的字符串
        limit: 长度限制阈值
        alpha: 指数增长的陡峭程度 (系数越大，惩罚上升越快)
        
    Returns:
        float: 惩罚值 (正数，需要在外部被减去，或者你可以直接返回负数)
    """
    length = len(predict_str)
    
    if length <= limit:
        return 0.0
    
    excess = length - limit
    
    # 公式: e^(alpha * excess) - 1
    # 减去 1 是为了保证 excess=0 时惩罚平滑地从 0 开始
    try:
        penalty = math.exp(alpha * excess) - 1
    except OverflowError:
        # 防止 excess 极大导致溢出，设置一个硬性上限
        penalty = 100.0 
        
    return penalty

def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
    weights: Dict[str, float] = {"comet": 30, "logic": 1},
    return_details: bool = True
) -> Union[float, Dict[str, float]]:
    """
    综合 Reward 计算函数 (支持多 Cue 并列的情况)。
    """
    if not predict_str or not predict_str.strip():
        if return_details:
            return {
                "score": 0.0, "comet_score": 0.0, "logic_score": 0.0, "length_penalty": 0.0,
                "parsed_required": -1, "cue_similarity": 0.0, "rl_type": extra_info.get("rl_type", "baseline"),
                "target_cues": "", "matched_cue": "", "error": "empty_prediction"
            }
        return 0.0

    # 1. 解析模型输出
    parsed = CoTParser.parse(predict_str)
    pred_trans = parsed["translation_pred"]

    if not pred_trans:
        if return_details:
            # parsed_required: -1=未解析, 0=不需要视频, 1=需要视频
            parsed_req_val = -1 if parsed["video_required_pred"] is None else (1 if parsed["video_required_pred"] else 0)
            return {
                "score": 0.0, "comet_score": 0.0, "logic_score": 0.0, "length_penalty": 0.0,
                "parsed_required": parsed_req_val, "cue_similarity": 0.0,
                "rl_type": extra_info.get("rl_type", "baseline"), "target_cues": "", "matched_cue": "",
                "error": "no_translation_parsed"
            }
        return 0.0

    # 2. 计算主要 Reward: 翻译质量 (COMET)
    src_sent = extra_info.get("source_sentence", "")
    comet_scores = _compute_remote_comet([src_sent], [pred_trans], [ground_truth])
    quality_score = comet_scores[0] if comet_scores else 0.0

    # 3. 计算次要 Reward: 逻辑判定 (Logic Check - Updated for Multi-Cue)
    logic_score = 0.0
    max_cue_similarity = 0.0
    matched_cue = None  # 记录匹配到了哪一个 cue

    rl_type = extra_info.get("rl_type", "baseline")
    pred_required = parsed["video_required_pred"]

    if rl_type == "baseline":
        # 期望：不需要视频
        if pred_required is False:
            logic_score = 1.0
        else:
            logic_score = 0.0

    elif rl_type == "visual":
        # 期望：需要视频
        if pred_required is True:
            # 第一步：判定需要视频 -> 得一半分 (0.5)
            logic_score += 0.5

            # 第二步：判定 Cue 类型是否正确 (支持列表)
            pred_cue_type = parsed["cue_type_pred"]

            # 获取所有允许的正确答案列表
            target_cue_types = extra_info.get("best_cue_types", [])
            # 兼容旧数据：如果列表为空，尝试获取单数字段
            if not target_cue_types:
                single_cue = extra_info.get("best_cue_type")
                if single_cue:
                    target_cue_types = [single_cue]

            # 遍历列表，计算最大相似度
            if target_cue_types and pred_cue_type:
                best_match_info = (0.0, None)  # (score, cue_name)

                for target_cue in target_cue_types:
                    sim = _compute_text_similarity(target_cue, pred_cue_type)
                    if sim > best_match_info[0]:
                        best_match_info = (sim, target_cue)

                max_cue_similarity = best_match_info[0]
                matched_cue = best_match_info[1]

                # 评分逻辑：只要有一个匹配度高，就给满分
                if max_cue_similarity > 0.8:
                    logic_score += 0.5
                else:
                    logic_score += 0.5 * max_cue_similarity
            else:
                # 如果没有 target cues 数据或者模型没输出 cue，无法得分
                max_cue_similarity = 0.0
        else:
            logic_score = 0.0

    # 4. 计算惩罚
    len_pen = length_penalty_reward(predict_str, limit=500)

    # 5. 综合加权
    total_score = (weights["comet"] * quality_score) + (weights["logic"] * logic_score) - len_pen

    if return_details:
        # 注意：target_cues 是列表，需要转为字符串以兼容 numpy 数组
        target_cues_list = extra_info.get("best_cue_types", [])
        target_cues_str = "|".join(target_cues_list) if target_cues_list else ""
        
        # parsed_required: -1=未解析, 0=不需要视频, 1=需要视频
        parsed_req_val = -1 if pred_required is None else (1 if pred_required else 0)
        
        return {
            "score": float(total_score),
            "comet_score": float(quality_score),
            "logic_score": float(logic_score),
            "length_penalty": float(len_pen),
            "parsed_required": parsed_req_val,
            "cue_similarity": float(max_cue_similarity),
            "rl_type": rl_type,
            "target_cues": target_cues_str,  # 转为字符串避免 numpy 报错
            "matched_cue": matched_cue if matched_cue else ""  # 方便 Debug 看到底匹配上了哪个
        }

    return float(total_score)


# 单元测试示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 模拟数据：Visual 样本，且有两个 Cue 并列最佳 (objects 和 actions)
    extra_info_visual_multi = {
        "source_sentence": "猫坐在垫子上",
        "rl_type": "visual",
        "best_cue_types": ["objects", "actions"],  # 列表包含多个允许的正确答案
        "best_cue_type": "objects"  # 旧字段兼容
    }

    gt = "The cat is on the mat."

    print("-" * 30)
    print("TEST 1: Multi-Cue Match (Option A: objects)")
    # 模型预测了 objects -> 应该满分
    pred_1 = "To translate this text, video information [objects] is **required**. It is cat. So the translation is: The cat is on the mat."
    res_1 = compute_score(pred_1, gt, extra_info_visual_multi, return_details=True)
    print(f"Logic Score: {res_1['logic_score']} (Expected: 1.0)")
    print(f"Matched: {res_1['matched_cue']} (Sim: {res_1['cue_similarity']})")

    print("\n" + "-" * 30)
    print("TEST 2: Multi-Cue Match (Option B: actions)")
    # 模型预测了 actions -> 应该也是满分 (尽管旧字段是 objects)
    pred_2 = "To translate this text, video information [actions] is **required**. It is sitting. So the translation is: The cat is on the mat."
    res_2 = compute_score(pred_2, gt, extra_info_visual_multi, return_details=True)
    print(f"Logic Score: {res_2['logic_score']} (Expected: 1.0)")
    print(f"Matched: {res_2['matched_cue']} (Sim: {res_2['cue_similarity']})")

    print("\n" + "-" * 30)
    print("TEST 3: Multi-Cue Match (Option C: Wrong Cue)")
    # 模型预测了 environment -> 应该低分
    pred_3 = "To translate this text, video information [environment] is **required**. It is indoor. So the translation is: The cat is on the mat."
    res_3 = compute_score(pred_3, gt, extra_info_visual_multi, return_details=True)
    print(f"Logic Score: {res_3['logic_score']} (Expected: ~0.5 + low sim)")
    print(f"Matched: {res_3['matched_cue']} (Sim: {res_3['cue_similarity']})")
