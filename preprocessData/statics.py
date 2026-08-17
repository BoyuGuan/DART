# -*- coding: utf-8 -*-
# 统计多模态cue对翻译的促进作用

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

# 需要排除的字段（与 getBetterTrainData.py 保持一致）
EXCLUDED_FIELDS = {"translation_baseline", "translation_all_cues"}


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """加载 JSON 数据"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def analyze_cue_effectiveness(data: List[Dict[str, Any]], delta: float = 4.0) -> Dict[str, Any]:
    """
    分析各个多模态 cue 对翻译的促进作用。
    
    Args:
        data: 包含翻译指标的数据列表
        delta: COMET 分数提升阈值，超过此值视为有促进作用
    
    Returns:
        包含统计结果的字典
    """
    # 统计变量
    total_samples = len(data)
    samples_with_metrics = 0
    samples_with_baseline = 0
    
    # 每个 cue 的统计
    cue_stats = defaultdict(lambda: {
        "total_count": 0,           # 该 cue 出现的总次数（所有样本）
        "non_empty_count": 0,       # 该 cue 不为空的样本数
        "improved_count": 0,        # 该 cue 超过 baseline + delta 的次数
        "improved_non_empty_count": 0,  # 在非空样本中超过阈值的次数
        "comet_diffs": [],          # 所有 COMET 差值
        "non_empty_diffs": [],      # 非空样本的 COMET 差值
        "improved_diffs": [],       # 超过阈值的 COMET 差值
    })
    
    # 样本级别统计
    samples_any_cue_improved = 0          # 至少有一个 cue 超过阈值的样本数
    samples_no_cue_improved = 0           # 所有 cue 都没有超过阈值的样本数
    samples_best_cue_info = []            # 每个样本的最佳 cue 信息
    no_improvement_samples = []           # 没有任何 cue 超过阈值的样本详情
    
    # 全局 COMET 分数统计
    all_baseline_comets = []
    all_visual_comets = defaultdict(list)
    
    for item in data:
        metrics = item.get("translation_metrics", {})
        if not metrics:
            continue
        samples_with_metrics += 1
        
        baseline_metrics = metrics.get("translation_baseline")
        if not baseline_metrics:
            continue
        samples_with_baseline += 1
        
        baseline_comet = baseline_metrics.get("COMET", 0.0)
        all_baseline_comets.append(baseline_comet)
        
        # 获取所有 visual cue keys
        visual_keys = [k for k in metrics.keys() if k not in EXCLUDED_FIELDS]
        
        if not visual_keys:
            continue
        
        # 分析每个 cue
        item_best_diff = float('-inf')
        item_best_cues = []
        item_any_improved = False
        
        for cue_key in visual_keys:
            cue_comet = metrics[cue_key].get("COMET", 0.0)
            diff = cue_comet - baseline_comet
            
            cue_name = cue_key.replace("translation_", "")
            
            # 检查该 cue 在原始数据中是否非空
            # 通过检查对应的 mm_prompt 字段是否有实际内容
            prompt_key = f"mm_prompt_{cue_name}"
            cue_content = item.get(prompt_key, "")
            is_cue_non_empty = bool(cue_content and cue_content.strip() and cue_content.strip() != "No information provided.")
            
            # 更新 cue 统计（所有样本）
            cue_stats[cue_name]["total_count"] += 1
            cue_stats[cue_name]["comet_diffs"].append(diff)
            all_visual_comets[cue_name].append(cue_comet)
            
            # 更新非空样本统计
            if is_cue_non_empty:
                cue_stats[cue_name]["non_empty_count"] += 1
                cue_stats[cue_name]["non_empty_diffs"].append(diff)
            
            if diff > delta:
                cue_stats[cue_name]["improved_count"] += 1
                cue_stats[cue_name]["improved_diffs"].append(diff)
                item_any_improved = True
                
                # 统计非空样本中超阈值的数量
                if is_cue_non_empty:
                    cue_stats[cue_name]["improved_non_empty_count"] += 1
            
            # 追踪该样本的最佳 cue
            if diff > item_best_diff:
                item_best_diff = diff
                item_best_cues = [cue_name]
            elif diff == item_best_diff:
                item_best_cues.append(cue_name)
        
        if item_any_improved:
            samples_any_cue_improved += 1
        else:
            # 所有 cue 都没有超过阈值
            samples_no_cue_improved += 1
            no_improvement_samples.append({
                "video_id": item.get("video_id", "unknown"),
                "clip_id": item.get("clip_id", "unknown"),
                "baseline_comet": baseline_comet,
                "best_cues": item_best_cues,
                "best_diff": item_best_diff,
            })
        
        samples_best_cue_info.append({
            "video_id": item.get("video_id", "unknown"),
            "clip_id": item.get("clip_id", "unknown"),
            "baseline_comet": baseline_comet,
            "best_cues": item_best_cues,
            "best_diff": item_best_diff,
            "improved": item_best_diff > delta
        })
    
    # 计算汇总统计
    result = {
        "overview": {
            "total_samples": total_samples,
            "samples_with_metrics": samples_with_metrics,
            "samples_with_baseline": samples_with_baseline,
            "samples_any_cue_improved": samples_any_cue_improved,
            "samples_no_cue_improved": samples_no_cue_improved,
            "improvement_rate": samples_any_cue_improved / samples_with_baseline * 100 if samples_with_baseline > 0 else 0,
            "no_improvement_rate": samples_no_cue_improved / samples_with_baseline * 100 if samples_with_baseline > 0 else 0,
            "delta_threshold": delta,
        },
        "baseline_stats": {
            "mean_comet": sum(all_baseline_comets) / len(all_baseline_comets) if all_baseline_comets else 0,
            "min_comet": min(all_baseline_comets) if all_baseline_comets else 0,
            "max_comet": max(all_baseline_comets) if all_baseline_comets else 0,
        },
        "cue_effectiveness": {},
        "cue_ranking": [],
    }
    
    # 计算每个 cue 的详细统计
    for cue_name, stats in cue_stats.items():
        total = stats["total_count"]
        non_empty = stats["non_empty_count"]
        improved = stats["improved_count"]
        improved_non_empty = stats["improved_non_empty_count"]
        diffs = stats["comet_diffs"]
        non_empty_diffs = stats["non_empty_diffs"]
        improved_diffs = stats["improved_diffs"]
        
        cue_result = {
            "total_count": total,
            "non_empty_count": non_empty,
            "improved_count": improved,
            "improved_non_empty_count": improved_non_empty,
            # 以所有样本为分母的促进率
            "improvement_rate_all": improved / total * 100 if total > 0 else 0,
            # 以非空样本为分母的促进率（更能反映该cue的真实效果）
            "improvement_rate_non_empty": improved_non_empty / non_empty * 100 if non_empty > 0 else 0,
            "mean_diff": sum(diffs) / len(diffs) if diffs else 0,
            "mean_diff_non_empty": sum(non_empty_diffs) / len(non_empty_diffs) if non_empty_diffs else 0,
            "max_diff": max(diffs) if diffs else 0,
            "min_diff": min(diffs) if diffs else 0,
            "mean_improved_diff": sum(improved_diffs) / len(improved_diffs) if improved_diffs else 0,
            "mean_comet": sum(all_visual_comets[cue_name]) / len(all_visual_comets[cue_name]) if all_visual_comets[cue_name] else 0,
        }
        result["cue_effectiveness"][cue_name] = cue_result
    
    # 按改进率排序（优先使用非空样本促进率）
    result["cue_ranking"] = sorted(
        result["cue_effectiveness"].items(),
        key=lambda x: (x[1]["improvement_rate_non_empty"], x[1]["improvement_rate_all"], x[1]["mean_diff"]),
        reverse=True
    )
    
    # 统计最佳 cue 分布
    best_cue_distribution = defaultdict(int)
    for info in samples_best_cue_info:
        if info["improved"]:
            for cue in info["best_cues"]:
                best_cue_distribution[cue] += 1
    
    result["best_cue_distribution"] = dict(sorted(
        best_cue_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    # 统计没有任何 cue 超过阈值的样本信息
    if no_improvement_samples:
        no_imp_diffs = [s["best_diff"] for s in no_improvement_samples]
        no_imp_baselines = [s["baseline_comet"] for s in no_improvement_samples]
        result["no_improvement_stats"] = {
            "count": len(no_improvement_samples),
            "rate": len(no_improvement_samples) / samples_with_baseline * 100 if samples_with_baseline > 0 else 0,
            "mean_best_diff": sum(no_imp_diffs) / len(no_imp_diffs) if no_imp_diffs else 0,
            "max_best_diff": max(no_imp_diffs) if no_imp_diffs else 0,
            "min_best_diff": min(no_imp_diffs) if no_imp_diffs else 0,
            "mean_baseline_comet": sum(no_imp_baselines) / len(no_imp_baselines) if no_imp_baselines else 0,
            "samples": no_improvement_samples[:20],  # 只保存前20个样本作为示例
        }
    else:
        result["no_improvement_stats"] = {
            "count": 0,
            "rate": 0,
            "mean_best_diff": 0,
            "max_best_diff": 0,
            "min_best_diff": 0,
            "mean_baseline_comet": 0,
            "samples": [],
        }
    
    return result


def print_report(stats: Dict[str, Any]):
    """打印统计报告"""
    print("=" * 70)
    print("多模态 Cue 对翻译促进作用统计报告")
    print("=" * 70)
    
    overview = stats["overview"]
    print(f"\n【概览】")
    print(f"  总样本数: {overview['total_samples']}")
    print(f"  有翻译指标的样本数: {overview['samples_with_metrics']}")
    print(f"  有 Baseline 指标的样本数: {overview['samples_with_baseline']}")
    print(f"  COMET 提升阈值 (delta): {overview['delta_threshold']}")
    print(f"  至少有一个 Cue 超过阈值的样本数: {overview['samples_any_cue_improved']}")
    print(f"  整体促进率: {overview['improvement_rate']:.2f}%")
    print(f"  所有 Cue 都未超过阈值的样本数: {overview['samples_no_cue_improved']}")
    print(f"  无促进率: {overview['no_improvement_rate']:.2f}%")
    
    baseline = stats["baseline_stats"]
    print(f"\n【Baseline 统计】")
    print(f"  平均 COMET: {baseline['mean_comet']:.4f}")
    print(f"  最小 COMET: {baseline['min_comet']:.4f}")
    print(f"  最大 COMET: {baseline['max_comet']:.4f}")
    
    print(f"\n【各 Cue 促进效果排名】(按非空样本促进率降序)")
    print("-" * 100)
    print(f"{'Cue 类型':<15} {'总样本':>8} {'非空样本':>10} {'超阈值数':>10} {'促进率(总)':>12} {'促进率(非空)':>14} {'平均Diff':>10}")
    print("-" * 100)
    
    for cue_name, cue_stats in stats["cue_ranking"]:
        print(f"{cue_name:<15} {cue_stats['total_count']:>8} {cue_stats['non_empty_count']:>10} "
              f"{cue_stats['improved_count']:>10} {cue_stats['improvement_rate_all']:>11.2f}% "
              f"{cue_stats['improvement_rate_non_empty']:>13.2f}% {cue_stats['mean_diff']:>10.4f}")
    
    print("-" * 100)
    
    print(f"\n【各 Cue 详细统计】")
    for cue_name, cue_stats in stats["cue_ranking"]:
        print(f"\n  {cue_name}:")
        print(f"    - 总样本数: {cue_stats['total_count']}")
        print(f"    - 非空样本数: {cue_stats['non_empty_count']} ({cue_stats['non_empty_count']/cue_stats['total_count']*100:.1f}%)")
        print(f"    - 超过阈值次数: {cue_stats['improved_count']} (非空中: {cue_stats['improved_non_empty_count']})")
        print(f"    - 促进率(以总样本为分母): {cue_stats['improvement_rate_all']:.2f}%")
        print(f"    - 促进率(以非空样本为分母): {cue_stats['improvement_rate_non_empty']:.2f}%")
        print(f"    - COMET Diff 范围: [{cue_stats['min_diff']:.4f}, {cue_stats['max_diff']:.4f}]")
        print(f"    - 平均 COMET Diff (全部): {cue_stats['mean_diff']:.4f}")
        print(f"    - 平均 COMET Diff (非空): {cue_stats['mean_diff_non_empty']:.4f}")
        print(f"    - 超阈值样本的平均 Diff: {cue_stats['mean_improved_diff']:.4f}")
        print(f"    - 平均 COMET 分数: {cue_stats['mean_comet']:.4f}")
    
    if stats["best_cue_distribution"]:
        print(f"\n【最佳 Cue 分布】(在超阈值样本中，各 Cue 成为最佳的次数)")
        print("-" * 40)
        for cue_name, count in stats["best_cue_distribution"].items():
            print(f"  {cue_name}: {count}")
    
    # 打印无促进样本统计
    no_imp = stats.get("no_improvement_stats", {})
    if no_imp.get("count", 0) > 0:
        print(f"\n【无促进样本统计】(所有多模态 Cue 相较 Baseline 都没有 {overview['delta_threshold']} 幅度提升)")
        print("-" * 70)
        print(f"  无促进样本数: {no_imp['count']}")
        print(f"  无促进率: {no_imp['rate']:.2f}%")
        print(f"  这些样本中最佳 Diff 的均值: {no_imp['mean_best_diff']:.4f}")
        print(f"  这些样本中最佳 Diff 的范围: [{no_imp['min_best_diff']:.4f}, {no_imp['max_best_diff']:.4f}]")
        print(f"  这些样本的 Baseline COMET 均值: {no_imp['mean_baseline_comet']:.4f}")
        
        if no_imp.get("samples"):
            print(f"\n  示例样本 (前 {len(no_imp['samples'])} 个):")
            for i, sample in enumerate(no_imp["samples"][:10], 1):
                print(f"    {i}. video_id={sample['video_id']}, clip_id={sample['clip_id']}, "
                      f"baseline={sample['baseline_comet']:.4f}, best_diff={sample['best_diff']:.4f}, "
                      f"best_cues={sample['best_cues']}")
    
    print("\n" + "=" * 70)


def save_report_to_file(stats: Dict[str, Any], log_dir: str = "./log"):
    """将统计报告保存到 log 文件"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cue_statistics.log")
    
    with open(log_file, "w", encoding="utf-8") as f:
        overview = stats["overview"]
        
        f.write("=" * 70 + "\n")
        f.write("多模态 Cue 对翻译促进作用统计报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        
        f.write(f"\n【概览】\n")
        f.write(f"  总样本数: {overview['total_samples']}\n")
        f.write(f"  有翻译指标的样本数: {overview['samples_with_metrics']}\n")
        f.write(f"  有 Baseline 指标的样本数: {overview['samples_with_baseline']}\n")
        f.write(f"  COMET 提升阈值 (delta): {overview['delta_threshold']}\n")
        f.write(f"  至少有一个 Cue 超过阈值的样本数: {overview['samples_any_cue_improved']}\n")
        f.write(f"  整体促进率: {overview['improvement_rate']:.2f}%\n")
        f.write(f"  所有 Cue 都未超过阈值的样本数: {overview['samples_no_cue_improved']}\n")
        f.write(f"  无促进率: {overview['no_improvement_rate']:.2f}%\n")
        
        baseline = stats["baseline_stats"]
        f.write(f"\n【Baseline 统计】\n")
        f.write(f"  平均 COMET: {baseline['mean_comet']:.4f}\n")
        f.write(f"  最小 COMET: {baseline['min_comet']:.4f}\n")
        f.write(f"  最大 COMET: {baseline['max_comet']:.4f}\n")
        
        f.write(f"\n【各 Cue 促进效果排名】(按非空样本促进率降序)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Cue 类型':<15} {'总样本':>8} {'非空样本':>10} {'超阈值数':>10} {'促进率(总)':>12} {'促进率(非空)':>14} {'平均Diff':>10}\n")
        f.write("-" * 100 + "\n")
        
        for cue_name, cue_stats in stats["cue_ranking"]:
            f.write(f"{cue_name:<15} {cue_stats['total_count']:>8} {cue_stats['non_empty_count']:>10} "
                  f"{cue_stats['improved_count']:>10} {cue_stats['improvement_rate_all']:>11.2f}% "
                  f"{cue_stats['improvement_rate_non_empty']:>13.2f}% {cue_stats['mean_diff']:>10.4f}\n")
        
        f.write("-" * 100 + "\n")
        
        f.write(f"\n【各 Cue 详细统计】\n")
        for cue_name, cue_stats in stats["cue_ranking"]:
            f.write(f"\n  {cue_name}:\n")
            f.write(f"    - 总样本数: {cue_stats['total_count']}\n")
            f.write(f"    - 非空样本数: {cue_stats['non_empty_count']} ({cue_stats['non_empty_count']/cue_stats['total_count']*100:.1f}%)\n")
            f.write(f"    - 超过阈值次数: {cue_stats['improved_count']} (非空中: {cue_stats['improved_non_empty_count']})\n")
            f.write(f"    - 促进率(以总样本为分母): {cue_stats['improvement_rate_all']:.2f}%\n")
            f.write(f"    - 促进率(以非空样本为分母): {cue_stats['improvement_rate_non_empty']:.2f}%\n")
            f.write(f"    - COMET Diff 范围: [{cue_stats['min_diff']:.4f}, {cue_stats['max_diff']:.4f}]\n")
            f.write(f"    - 平均 COMET Diff (全部): {cue_stats['mean_diff']:.4f}\n")
            f.write(f"    - 平均 COMET Diff (非空): {cue_stats['mean_diff_non_empty']:.4f}\n")
            f.write(f"    - 超阈值样本的平均 Diff: {cue_stats['mean_improved_diff']:.4f}\n")
            f.write(f"    - 平均 COMET 分数: {cue_stats['mean_comet']:.4f}\n")
        
        if stats["best_cue_distribution"]:
            f.write(f"\n【最佳 Cue 分布】(在超阈值样本中，各 Cue 成为最佳的次数)\n")
            f.write("-" * 40 + "\n")
            for cue_name, count in stats["best_cue_distribution"].items():
                f.write(f"  {cue_name}: {count}\n")
        
        # 无促进样本统计
        no_imp = stats.get("no_improvement_stats", {})
        if no_imp.get("count", 0) > 0:
            f.write(f"\n【无促进样本统计】(所有多模态 Cue 相较 Baseline 都没有 {overview['delta_threshold']} 幅度提升)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  无促进样本数: {no_imp['count']}\n")
            f.write(f"  无促进率: {no_imp['rate']:.2f}%\n")
            f.write(f"  这些样本中最佳 Diff 的均值: {no_imp['mean_best_diff']:.4f}\n")
            f.write(f"  这些样本中最佳 Diff 的范围: [{no_imp['min_best_diff']:.4f}, {no_imp['max_best_diff']:.4f}]\n")
            f.write(f"  这些样本的 Baseline COMET 均值: {no_imp['mean_baseline_comet']:.4f}\n")
            
            if no_imp.get("samples"):
                f.write(f"\n  示例样本 (前 {len(no_imp['samples'])} 个):\n")
                for i, sample in enumerate(no_imp["samples"], 1):
                    f.write(f"    {i}. video_id={sample['video_id']}, clip_id={sample['clip_id']}, "
                          f"baseline={sample['baseline_comet']:.4f}, best_diff={sample['best_diff']:.4f}, "
                          f"best_cues={sample['best_cues']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    return log_file


def main():
    parser = argparse.ArgumentParser(description="统计多模态 Cue 对翻译的促进作用")
    parser.add_argument(
        "--input_json", 
        type=str, 
        default="./data/work3/MMinfoAndTrans/promptsAndTransMetrics.json",
        help="输入 JSON 文件路径"
    )
    parser.add_argument(
        "--delta", 
        type=float, 
        default=2.0,
        help="COMET 分数提升阈值，超过此值视为有促进作用 (默认: 2.0)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="可选：将统计结果保存为 JSON 文件"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log",
        help="日志文件保存目录 (默认: ./log)"
    )
    args = parser.parse_args()
    
    print(f"加载数据: {args.input_json}")
    data = load_data(args.input_json)
    print(f"共加载 {len(data)} 条样本")
    
    print(f"\n开始分析 (delta={args.delta})...")
    stats = analyze_cue_effectiveness(data, delta=args.delta)
    
    print_report(stats)
    
    # 保存日志文件
    log_file = save_report_to_file(stats, log_dir=args.log_dir)
    print(f"\n统计报告已保存至: {log_file}")
    
    # 可选：保存结果到 JSON
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 转换 cue_ranking 为可序列化格式
        stats["cue_ranking"] = [
            {"cue_name": name, **values} 
            for name, values in stats["cue_ranking"]
        ]
        
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"\n统计结果已保存至: {args.output_json}")


if __name__ == "__main__":
    main()
