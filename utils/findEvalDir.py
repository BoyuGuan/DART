""" 
检查 ./eval 文件夹下子文件夹中的 eval.log 是否包含指定关键字（必须全部存在）
输入参数：需要检查的关键字（多个关键字需全部存在），用空格分隔
"""

import os
import argparse

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="检查 ./eval 文件夹下子文件夹中的 eval.log 是否包含指定关键字（必须全部存在）"
    )
    parser.add_argument(
        'keywords',
        nargs='+',
        help="需要检查的关键字（多个关键字需全部存在）"
    )
    args = parser.parse_args()

    eval_dir = "./eval"
    
    allDirs = os.listdir(eval_dir)
    allDirs.sort()
    # 遍历 ./eval 目录下的所有子文件夹
    for subfolder in allDirs:
        subfolder_path = os.path.join(eval_dir, subfolder)
        if os.path.isdir(subfolder_path):
            log_file = os.path.join(subfolder_path, "eval.log")
            if not os.path.exists(log_file):
                continue  # 如果不存在 eval.log 文件，则跳过该子文件夹
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"读取文件 {log_file} 时发生错误：{e}")
                continue

            # 检查是否所有关键字都存在于文件内容中
            if all(keyword in content for keyword in args.keywords):
                print(f"\n{'='*80}")
                print(f"文件夹: {subfolder}")
                print(f"{'='*80}")
                
                # 打印 eval.log 的内容
                print(f"\n--- eval.log 内容 ---")
                print(content)
                
                # 打印 translationMetricScores.log 的内容
                metric_log_file = os.path.join(subfolder_path, "translationMetricScores.log")
                if os.path.exists(metric_log_file):
                    try:
                        with open(metric_log_file, 'r', encoding='utf-8') as f:
                            metric_content = f.read()
                        print(f"\n--- translationMetricScores.log 内容 ---")
                        print(metric_content)
                    except Exception as e:
                        print(f"读取文件 {metric_log_file} 时发生错误：{e}")
                else:
                    print(f"\n--- translationMetricScores.log 不存在 ---")