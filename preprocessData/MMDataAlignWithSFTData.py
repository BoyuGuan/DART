import json
import os
import sys

def load_json(path):
    """读取JSON文件"""
    print(f"正在读取: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 文件 {path} 不是有效的 JSON 格式")
        sys.exit(1)

def save_json(data, path):
    """保存JSON文件"""
    print(f"正在保存合并后的结果到: {path}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("保存完成。")

def extract_id_from_sft_path(video_path):
    """
    从 SFT 数据的 video 路径中提取片段ID
    示例: ./data/TriFine/videoClips/H0_cl573IkU/H0_cl573IkU_17.mp4 -> H0_cl573IkU_17
    """
    if not video_path:
        return None
    filename = os.path.basename(video_path)  # 获取文件名: H0_cl573IkU_17.mp4
    clip_id = os.path.splitext(filename)[0]  # 去除后缀: H0_cl573IkU_17
    return clip_id

def construct_id_from_result_item(item):
    """
    从 results 数据中构造片段ID
    逻辑: {video_id}_{clip_id}
    """
    video_id = str(item.get('video_id', '')).strip()
    clip_id = str(item.get('clip_id', '')).strip()
    
    # 确保两个字段都存在
    if not video_id or not clip_id:
        return None
        
    return f"{video_id}_{clip_id}"

def main():
    # --- 1. 定义文件路径 ---
    # SFT 文件 (作为基准 ID 集合)
    sft_data_path = '/home/byguan/3vmt/data/work3/sftData/sftData_50000.json'
    
    # 两个待筛选的 Results 文件
    result_files = [
        '/home/byguan/3vmt/data/work3/MMinfoAndTrans/MMInfoRawData/eval-2025-12-14-22-04-53/results.json',
        '/home/byguan/3vmt/data/work3/MMinfoAndTrans/MMInfoRawData/eval-2025-12-14-22-05-14/results.json'
    ]
    
    # 输出文件路径 (保存在 MMinfoAndTrans 目录下)
    output_path = '/home/byguan/3vmt/data/work3/MMinfoAndTrans/results_filtered_by_sft_merged.json'

    # --- 2. 建立 SFT ID 白名单 (Allowlist) ---
    print("--- 步骤1: 从 SFT 数据建立 ID 白名单 ---")
    sft_data = load_json(sft_data_path)
    valid_ids = set()
    
    for item in sft_data:
        video_path = item.get('video', '')
        extracted_id = extract_id_from_sft_path(video_path)
        if extracted_id:
            valid_ids.add(extracted_id)
            
    print(f"SFT 数据包含 {len(sft_data)} 条记录，提取出 {len(valid_ids)} 个唯一片段ID。")

    # --- 3. 筛选并合并 Results 数据 ---
    print("--- 步骤2: 过滤并合并 Results 文件 ---")
    merged_results = []
    total_processed = 0
    
    for res_file in result_files:
        data = load_json(res_file)
        file_kept_count = 0
        
        for item in data:
            total_processed += 1
            # 构造 ID 用于比对
            constructed_id = construct_id_from_result_item(item)
            
            # 如果构造出的 ID 在 SFT 的 ID 集合中，则保留
            if constructed_id and constructed_id in valid_ids:
                merged_results.append(item)
                file_kept_count += 1
        
        print(f"文件 {os.path.basename(res_file)}: 处理 {len(data)} 条，保留 {file_kept_count} 条")

    # --- 4. 保存结果 ---
    print(f"--- 筛选完成 ---")
    print(f"共处理 Results 数据: {total_processed} 条")
    print(f"最终合并保留数据: {len(merged_results)} 条")
    
    if len(merged_results) > 0:
        save_json(merged_results, output_path)
    else:
        print("警告: 结果文件中没有数据匹配 SFT ID，未生成输出文件。")

if __name__ == "__main__":
    main()
    