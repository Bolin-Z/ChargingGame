"""
分析 flow_scale_factor 实验结果

对比三种算法（MADDPG/IDDPG/MFDDPG）在流量缩放后的表现
与基线数据对比，验证 P0 修改效果
"""

import json
import numpy as np
from pathlib import Path


def load_step_records(filepath):
    """加载 step_records.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['records'], data['metadata']


def analyze_algorithm(records, metadata, name):
    """分析单个算法的结果"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Episodes: {metadata['total_episodes']}, Steps: {metadata['total_steps']}")

    # 提取 relative_change_rate
    change_rates = [r['relative_change_rate'] for r in records]

    # 基本统计
    print(f"\n--- relative_change_rate ---")
    print(f"  最小值: {min(change_rates):.6f}")
    print(f"  平均值: {np.mean(change_rates):.6f}")

    # 低于阈值的比例
    threshold = 0.01
    below_threshold = sum(1 for r in change_rates if r < threshold)
    print(f"  低于阈值(0.01)的比例: {below_threshold}/{len(change_rates)} ({100*below_threshold/len(change_rates):.1f}%)")

    # 连续低于阈值的最长序列
    max_consecutive = 0
    current_consecutive = 0
    for r in change_rates:
        if r < threshold:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    print(f"  连续低于阈值最长: {max_consecutive} steps")

    # 价格边界趋势分析
    agents = list(records[0]['actual_prices'].keys())

    boundary_ratios = []
    for r in records:
        total_prices = 0
        boundary_prices = 0
        for agent in agents:
            for p in r['actual_prices'][agent]:
                total_prices += 1
                if abs(p - 0.5) < 0.1 or abs(p - 2.0) < 0.1:
                    boundary_prices += 1
        boundary_ratios.append(boundary_prices / total_prices)

    # 分段统计
    print(f"\n--- 价格边界趋势 ---")
    if len(boundary_ratios) >= 100:
        first_50 = boundary_ratios[:50]
        last_50 = boundary_ratios[-50:]
        print(f"  前 50 步边界价格比例: {np.mean(first_50)*100:.1f}%")
        print(f"  后 50 步边界价格比例: {np.mean(last_50)*100:.1f}%")
        diff = np.mean(last_50) - np.mean(first_50)
        if diff > 0.05:
            trend = "明显向边界移动 ↑"
        elif diff > 0:
            trend = "轻微向边界移动 →"
        else:
            trend = "无变化或远离边界 ↔"
        print(f"  趋势: {trend} (Δ={diff*100:+.1f}%)")

    # 最后一步的价格分布
    last_prices = records[-1]['actual_prices']
    print(f"\n  最后 step 价格:")
    for agent in sorted(agents):
        prices = last_prices[agent]
        print(f"    {agent}: {[round(p, 2) for p in prices]}")

    # 收益范围
    all_rewards = [r['rewards'][a] for r in records for a in agents]
    print(f"\n--- 收益范围 ---")
    print(f"  [{min(all_rewards):.0f}, {max(all_rewards):.0f}]")

    return {
        'name': name,
        'total_steps': len(records),
        'below_threshold_ratio': below_threshold / len(change_rates),
        'max_consecutive': max_consecutive,
        'first_50_boundary': np.mean(first_50) if len(boundary_ratios) >= 100 else None,
        'last_50_boundary': np.mean(last_50) if len(boundary_ratios) >= 100 else None,
        'min_change_rate': min(change_rates),
        'avg_change_rate': np.mean(change_rates),
    }


def print_comparison_table(results, baseline=None):
    """打印对比表格"""
    print(f"\n{'='*80}")
    print(f"  实验结果汇总对比")
    print(f"{'='*80}")

    # 表头
    print(f"\n{'指标':<25} | ", end='')
    for r in results:
        print(f"{r['name']:>12} | ", end='')
    if baseline:
        print(f"{'基线MADDPG':>12}")
    else:
        print()

    print(f"{'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+", end='')
    if baseline:
        print(f"-{'-'*12}")
    else:
        print()

    # 数据行
    def print_row(label, key, fmt='.1f', multiply=100, suffix='%'):
        print(f"{label:<25} | ", end='')
        for r in results:
            val = r[key]
            if val is not None:
                if multiply:
                    val = val * multiply
                print(f"{val:{fmt}}{suffix:>5} | ", end='')
            else:
                print(f"{'N/A':>12} | ", end='')
        if baseline and key in baseline:
            val = baseline[key]
            if multiply:
                val = val * multiply
            print(f"{val:{fmt}}{suffix:>5}")
        else:
            print()

    print_row('低于阈值比例', 'below_threshold_ratio')
    print_row('前50步边界比例', 'first_50_boundary')
    print_row('后50步边界比例', 'last_50_boundary')
    print_row('最小change_rate', 'min_change_rate', fmt='.5f', multiply=None, suffix='')
    print_row('平均change_rate', 'avg_change_rate', fmt='.4f', multiply=None, suffix='')

    print(f"\n{'连续低于阈值最长':<25} | ", end='')
    for r in results:
        print(f"{r['max_consecutive']:>7} steps | ", end='')
    if baseline:
        print(f"{baseline['max_consecutive']:>7} steps")
    else:
        print()


def main():
    results_dir = Path(__file__).parent / 'results' / 'siouxfalls'

    # 查找最新的实验结果（01_06 开头的）
    algorithms = ['MADDPG', 'IDDPG', 'MFDDPG']
    results = []

    print("加载 flow_scale 实验数据...")
    print("=" * 60)

    for algo in algorithms:
        algo_dir = results_dir / algo / 'seed42'
        if not algo_dir.exists():
            print(f"  {algo}: 目录不存在")
            continue

        # 找到 01_06 开头的最新目录
        dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith('01_06')])
        if not dirs:
            print(f"  {algo}: 没有找到 01_06 的实验")
            continue

        latest_dir = dirs[-1]
        records_path = latest_dir / 'step_records.json'

        if records_path.exists():
            print(f"  {algo}: {latest_dir.name}")
            records, metadata = load_step_records(records_path)
            result = analyze_algorithm(records, metadata, algo)
            results.append(result)

    # 基线数据（来自 doc/task_tune.md 第四节）
    baseline = {
        'below_threshold_ratio': 0.006,  # 0.6%
        'max_consecutive': 2,
        'first_50_boundary': 0.579,  # 57.9%
        'last_50_boundary': 0.569,   # 56.9%
        'min_change_rate': 0.00266,
        'avg_change_rate': 0.072,
    }

    # 打印对比表格
    print_comparison_table(results, baseline)

    # P0 效果评估
    print(f"\n{'='*60}")
    print(f"  P0 流量缩放效果评估")
    print(f"{'='*60}")

    for r in results:
        if r['name'] == 'MADDPG':
            print(f"\n{r['name']}:")

            # 边界趋势变化
            if r['first_50_boundary'] and r['last_50_boundary']:
                diff = r['last_50_boundary'] - r['first_50_boundary']
                baseline_diff = baseline['last_50_boundary'] - baseline['first_50_boundary']

                if diff > 0 and baseline_diff <= 0:
                    print(f"  ✅ 边界趋势: 从无变化变为向边界移动 (Δ={diff*100:+.1f}% vs 基线{baseline_diff*100:+.1f}%)")
                elif diff > baseline_diff:
                    print(f"  ✅ 边界趋势改善: Δ={diff*100:+.1f}% vs 基线{baseline_diff*100:+.1f}%")
                else:
                    print(f"  ❌ 边界趋势未改善: Δ={diff*100:+.1f}% vs 基线{baseline_diff*100:+.1f}%")

            # 收敛改善
            if r['max_consecutive'] > baseline['max_consecutive']:
                print(f"  ✅ 连续收敛步数: {r['max_consecutive']} vs 基线 {baseline['max_consecutive']}")
            else:
                print(f"  ❌ 连续收敛步数未改善: {r['max_consecutive']} vs 基线 {baseline['max_consecutive']}")

            if r['below_threshold_ratio'] > baseline['below_threshold_ratio']:
                print(f"  ✅ 低于阈值比例: {r['below_threshold_ratio']*100:.1f}% vs 基线 {baseline['below_threshold_ratio']*100:.1f}%")

    print(f"\n{'='*60}")
    print("分析完成！")


if __name__ == '__main__':
    main()
