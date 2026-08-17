"""
将自动提取到的多模态细粒度信息检查格式合法性，并将其改为json存储。
仅保存 MMInfoValid 为 True 的数据。

处理流程：
1. 读取vllmServerInference.py的输出文件（包含videoInfoExtraction字段）
2. 检查JSON格式是否正确
3. 验证结构是否符合预期的多模态信息格式
4. 过滤乱码和无效内容
5. 仅保存格式合法的数据，并统计过滤掉的数据信息
"""

import argparse
import json
import logging
import re
import os
import unicodedata
from typing import Dict, Any, Optional, List, Tuple


logger = logging.getLogger('extractMM')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


def is_valid_json_string(text: str) -> bool:
    """检查字符串是否是有效的JSON格式"""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def contains_garbled_text(text: str) -> bool:
    """
    检测文本是否包含乱码
    """
    if not text or not isinstance(text, str):
        return False
    
    garbled_count = 0
    total_chars = len(text)
    
    if total_chars == 0:
        return False
    
    for char in text:
        code_point = ord(char)
        
        # 控制字符（除了常见的\n\r\t）
        if code_point < 32 and char not in '\n\r\t':
            garbled_count += 1
            continue
        
        # 替换字符
        if char == '\ufffd':
            garbled_count += 1
            continue
        
        # 私用区字符 (PUA)
        if 0xE000 <= code_point <= 0xF8FF:
            garbled_count += 1
            continue
        
        # 辅助私用区
        if 0xF0000 <= code_point <= 0xFFFFD or 0x100000 <= code_point <= 0x10FFFD:
            garbled_count += 1
            continue
        
        # 非字符 (Noncharacters)
        if code_point in [0xFFFE, 0xFFFF] or (0xFDD0 <= code_point <= 0xFDEF):
            garbled_count += 1
            continue
    
    # 如果乱码字符超过5%，认为包含乱码
    return garbled_count > total_chars * 0.05


def extract_json_from_text(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    从文本中提取JSON对象，处理可能的markdown代码块或其他格式
    返回: (提取的JSON对象, 提取状态描述)
    """
    if not text or not isinstance(text, str):
        return None, "empty_input"
    
    text = text.strip()
    
    if not text:
        return None, "empty_after_strip"
    
    # 检查是否包含错误信息
    if text.startswith("Error:"):
        return None, "api_error"
    
    # 尝试直接解析
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, "direct_parse"
        return None, "not_dict"
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 尝试提取markdown代码块中的JSON
    json_patterns = [
        (r'```json\s*(\{.*?\})\s*```', "markdown_json"),  # ```json {...} ```
        (r'```\s*(\{.*?\})\s*```', "markdown_plain"),      # ``` {...} ```
    ]
    
    for pattern, pattern_name in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result, pattern_name
            except (json.JSONDecodeError, ValueError):
                continue
    
    # 尝试找到第一个 { 和最后一个 } 之间的内容
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result, "bracket_extract"
        except (json.JSONDecodeError, ValueError):
            pass
    
    return None, "parse_failed"


def validate_mm_info_structure(mm_info: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证多模态信息结构是否符合预期格式
    返回: (是否有效, 状态描述)
    """
    if not isinstance(mm_info, dict):
        return False, "not_dict"
    
    # 预期的字段列表（允许为空列表）
    expected_fields = ['people', 'objects', 'actions', 'ocr', 'spatial_relations', 'pointing_gaze']
    
    # 检查是否至少包含一个预期字段
    has_expected_field = any(field in mm_info for field in expected_fields)
    
    if not has_expected_field:
        return False, "no_expected_fields"
    
    # 检查字段值类型（应该是列表）
    for field in expected_fields:
        if field in mm_info:
            if not isinstance(mm_info[field], list):
                return False, f"field_{field}_not_list"
    
    return True, "valid"


def clean_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    清理单个字典项，移除乱码
    返回: 清理后的字典，如果整体无效则返回None
    """
    if not isinstance(item, dict) or not item:
        return None
    
    cleaned_item = {}
    
    for key, value in item.items():
        # 清理键名
        if not isinstance(key, str):
            key = str(key)
        
        if contains_garbled_text(key):
            continue
        
        # 清理值
        if isinstance(value, str):
            if contains_garbled_text(value):
                # 如果值包含乱码，设为unknown
                cleaned_item[key] = "unknown"
            else:
                cleaned_item[key] = value
        elif isinstance(value, dict):
            # 递归清理嵌套字典
            cleaned_nested = clean_item(value)
            if cleaned_nested:
                cleaned_item[key] = cleaned_nested
            else:
                cleaned_item[key] = {}
        elif isinstance(value, list):
            # 清理列表中的字符串
            cleaned_list = []
            for v in value:
                if isinstance(v, str):
                    if not contains_garbled_text(v):
                        cleaned_list.append(v)
                elif isinstance(v, dict):
                    cleaned_v = clean_item(v)
                    if cleaned_v:
                        cleaned_list.append(cleaned_v)
                else:
                    cleaned_list.append(v)
            cleaned_item[key] = cleaned_list
        else:
            # 其他类型直接保留
            cleaned_item[key] = value
    
    return cleaned_item if cleaned_item else None


def clean_mm_info(mm_info: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    清理多模态信息，移除无效项
    返回: (清理后的数据, 每个字段的有效项数统计)
    """
    cleaned = {}
    field_counts = {}
    
    expected_fields = ['people', 'objects', 'actions', 'ocr', 'spatial_relations', 'pointing_gaze']
    
    # 清理每个字段
    for field in expected_fields:
        if field in mm_info:
            items = mm_info[field]
            if isinstance(items, list):
                cleaned_items = []
                for item in items:
                    if isinstance(item, dict) and item:  # 非空字典
                        cleaned_item = clean_item(item)
                        if cleaned_item:
                            cleaned_items.append(cleaned_item)
                cleaned[field] = cleaned_items
                field_counts[field] = len(cleaned_items)
            else:
                cleaned[field] = []
                field_counts[field] = 0
        else:
            cleaned[field] = []
            field_counts[field] = 0
    
    return cleaned, field_counts


def process_mm_extraction(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int]]:
    """
    处理多模态提取结果，返回清理后的JSON对象
    返回: (清理后的数据, 状态描述, 字段计数)
    """
    # 提取JSON
    mm_info, extract_status = extract_json_from_text(raw_text)
    
    if mm_info is None:
        return None, f"extract_failed:{extract_status}", {}
    
    # 验证结构
    is_valid, validate_status = validate_mm_info_structure(mm_info)
    if not is_valid:
        return None, f"validate_failed:{validate_status}", {}
    
    # 清理数据
    cleaned_mm_info, field_counts = clean_mm_info(mm_info)
    
    return cleaned_mm_info, "success", field_counts


def get_clip_id(item: Dict[str, Any]) -> str:
    """获取clip的唯一标识"""
    if 'video_id' in item and 'clip_id' in item:
        return f"{item['video_id']}_{item['clip_id']}"
    if 'clipID' in item:
        return item['clipID']
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理多模态信息提取结果，验证JSON格式并过滤乱码")
    parser.add_argument("--MMInfoFilePath", type=str, required=True, 
                        help="输入文件路径（vllmServerInference.py的输出结果，包含videoInfoExtraction字段）")
    parser.add_argument("--originDataFilePath", type=str, default="./data/TriFine/Train_clips.json",
                        help="原始数据文件路径（Train_clips.json）")
    parser.add_argument("--outputFilePath", type=str, default="./data/work3/MMinfoAndTrans/MMInfo.json",
                        help="输出文件路径")
    parser.add_argument("--logDir", type=str, default="./log",
                        help="日志目录")
    parser.add_argument("--verbose", action="store_true",
                        help="是否输出详细的每条数据处理日志")
    args = parser.parse_args()
    
    # 确保日志目录存在
    os.makedirs(args.logDir, exist_ok=True)
    fileHandler = logging.FileHandler(f'{args.logDir}/extractMM.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    logger.info(f"开始处理文件: {args.MMInfoFilePath}")
    logger.info(f"原始数据文件: {args.originDataFilePath}")
    
    # 读取文件
    try:
        with open(args.MMInfoFilePath, 'r', encoding='utf-8') as f:
            mm_extraction_data = json.load(f)
        logger.info(f"成功读取多模态提取结果 {len(mm_extraction_data)} 条")
    except Exception as e:
        logger.error(f"读取多模态提取结果文件失败: {e}")
        raise
    
    try:
        with open(args.originDataFilePath, 'r', encoding='utf-8') as f:
            origin_data = json.load(f)
        logger.info(f"成功读取原始数据 {len(origin_data)} 条")
    except Exception as e:
        logger.error(f"读取原始数据文件失败: {e}")
        raise
    
    # 构建索引
    origin_data_dict = {}
    for item in origin_data:
        clip_id = get_clip_id(item)
        origin_data_dict[clip_id] = item
    
    # 统计信息
    total_count = len(mm_extraction_data)
    valid_count = 0
    invalid_count = 0
    empty_count = 0
    api_error_count = 0
    parse_error_count = 0
    validate_error_count = 0
    not_found_in_origin_count = 0
    
    field_names = ['people', 'objects', 'actions', 'ocr', 'spatial_relations', 'pointing_gaze']
    total_field_counts = {field: 0 for field in field_names}
    non_empty_field_counts = {field: 0 for field in field_names}
    error_reasons = {}
    
    # 结果容器
    output_data = []
    
    # 处理每条数据
    for idx, item in enumerate(mm_extraction_data):
        clip_id = get_clip_id(item)
        
        # 基础数据准备
        if clip_id in origin_data_dict:
            output_item = origin_data_dict[clip_id].copy()
        else:
            output_item = item.copy()
            not_found_in_origin_count += 1
            if args.verbose:
                logger.warning(f"第 {idx+1} 条数据 (clipID: {clip_id}) 在原始数据中未找到")
        
        mm_text = item.get('videoInfoExtraction', '')
        should_save = False # 标记是否保存当前条目
        
        if not mm_text or not isinstance(mm_text, str) or not mm_text.strip():
            # 空数据 -> 无效 -> 不保存
            empty_count += 1
            invalid_count += 1
            if args.verbose:
                logger.warning(f"第 {idx+1} 条数据 (clipID: {clip_id}) 的videoInfoExtraction字段为空 -> 跳过")
        else:
            # 处理多模态信息
            cleaned_mm_info, status, field_counts = process_mm_extraction(mm_text)
            
            if cleaned_mm_info is not None:
                # 成功提取 -> 有效 -> 准备保存
                output_item['MMInfo'] = cleaned_mm_info
                output_item['MMInfoValid'] = True
                output_item['MMInfoStatus'] = status
                valid_count += 1
                should_save = True # 标记为保存
                
                # 更新字段统计
                for field, count in field_counts.items():
                    total_field_counts[field] += count
                    if count > 0:
                        non_empty_field_counts[field] += 1
            else:
                # 提取失败 -> 无效 -> 不保存
                invalid_count += 1
                
                error_reasons[status] = error_reasons.get(status, 0) + 1
                if "api_error" in status:
                    api_error_count += 1
                elif "extract_failed" in status or "parse_failed" in status:
                    parse_error_count += 1
                elif "validate_failed" in status:
                    validate_error_count += 1
                
                if args.verbose:
                    logger.warning(f"第 {idx+1} 条数据 (clipID: {clip_id}) 处理失败: {status} -> 跳过")
        
        # 关键修改：只有 flag 为 True 时才添加到输出列表
        if should_save:
            output_data.append(output_item)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"已处理 {idx+1}/{total_count} 条数据")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.outputFilePath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    try:
        with open(args.outputFilePath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {args.outputFilePath}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise
    
    def safe_percentage(count, total):
        return f"{count/total*100:.2f}%" if total > 0 else "N/A"
    
    separator = "=" * 70
    
    # 更新统计文案
    stats_output = f"""
{separator}
处理完成！(已过滤无效数据)
{separator}

【数据量统计】
  原始输入总数: {total_count}
  最终保存总数: {len(output_data)} (剔除无效数据: {invalid_count})
  原始数据中未找到CLIP: {not_found_in_origin_count}

【无效数据（已过滤，不保存）细分】
  - 空数据/无字段: {empty_count}
  - API调用错误: {api_error_count}
  - JSON解析错误: {parse_error_count}
  - 结构验证错误: {validate_error_count}

【字段统计（基于最终保存的 {valid_count} 条数据）】
"""
    
    for field in field_names:
        total_items = total_field_counts[field]
        non_empty_count = non_empty_field_counts[field]
        avg_items = total_items / valid_count if valid_count > 0 else 0
        stats_output += f"  - {field:20s}: 总项数={total_items:6d}, 非空数据数={non_empty_count:5d} ({safe_percentage(non_empty_count, valid_count)}), 平均项数={avg_items:.2f}\n"
    
    if error_reasons:
        stats_output += f"\n【过滤原因详情】\n"
        for reason, count in sorted(error_reasons.items(), key=lambda x: -x[1]):
            stats_output += f"  - {reason}: {count}\n"
    
    stats_output += f"\n{separator}\n"
    
    logger.info(stats_output)
    print(stats_output)
    
    stats_file = args.outputFilePath.replace('.json', '_stats.txt')
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats_output)
        logger.info(f"统计信息已保存到: {stats_file}")
    except Exception as e:
        logger.warning(f"保存统计信息失败: {e}")