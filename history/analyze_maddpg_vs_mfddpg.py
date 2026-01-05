"""
分析 MADDPG vs MF-DDPG 实验数据对比

目的：诊断 MADDPG 不收敛的原因
- 分析 relative_change_rate 的变化趋势
- 分析价格是否有向边界 (0.5/2.0) 移动的趋势
- 对比两种算法在相同 step 数时的表现
"""

import json
import numpy as np
from pathlib import Path


def load_step_records(filepath):
    """加载 step_records.json"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # 结构是 {metadata: {...}, records: [...]}
    print(f"  文件: {filepath}")
    print(f"  元数据: episodes={data['metadata']['total_episodes']}, steps={data['metadata']['total_steps']}")
    return data['records']


def analyze_convergence(records, name):
    """分析收敛指标"""
    print(f"\n{'='*60}")
    print(f"  {name} 收敛分析")
    print(f"{'='*60}")

    # 提取 relative_change_rate
    change_rates = [r['relative_change_rate'] for r in records]

    # 基本统计
    print(f"\n总 steps 数: {len(records)}")
    print(f"\n--- relative_change_rate 统计 ---")
    print(f"  最小值: {min(change_rates):.6f}")
    print(f"  最大值: {max(change_rates):.6f}")
    print(f"  平均值: {np.mean(change_rates):.6f}")
    print(f"  标准差: {np.std(change_rates):.6f}")

    # 分段统计（前100步、中间、后100步）
    if len(change_rates) >= 300:
        first_100 = change_rates[:100]
        last_100 = change_rates[-100:]
        print(f"\n--- 分段平均 relative_change_rate ---")
        print(f"  前 100 步: {np.mean(first_100):.6f}")
        print(f"  后 100 步: {np.mean(last_100):.6f}")
        print(f"  变化趋势: {'下降 ↓' if np.mean(last_100) < np.mean(first_100) else '上升/震荡 ↔'}")

    # 低于阈值的比例
    threshold = 0.01
    below_threshold = sum(1 for r in change_rates if r < threshold)
    print(f"\n--- 收敛检测 (阈值={threshold}) ---")
    print(f"  低于阈值的 steps: {below_threshold}/{len(change_rates)} ({100*below_threshold/len(change_rates):.1f}%)")

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

    return change_rates


def analyze_price_trend(records, name):
    """分析价格向边界移动的趋势"""
    print(f"\n--- {name} 价格边界趋势分析 ---")

    # 提取所有智能体的价格 (字段名是 actual_prices)
    agents = list(records[0]['actual_prices'].keys())
    n_periods = len(records[0]['actual_prices'][agents[0]])

    # 计算每个 step 的边界价格比例
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
    if len(boundary_ratios) >= 100:
        first_50 = boundary_ratios[:50]
        last_50 = boundary_ratios[-50:]
        print(f"  前 50 步边界价格比例: {np.mean(first_50)*100:.1f}%")
        print(f"  后 50 步边界价格比例: {np.mean(last_50)*100:.1f}%")
        print(f"  趋势: {'向边界移动 →' if np.mean(last_50) > np.mean(first_50) else '无明显趋势'}")

    # 最后一步的价格分布
    last_prices = records[-1]['actual_prices']
    print(f"\n  最后 step 各智能体价格:")
    for agent in sorted(agents):
        prices = last_prices[agent]
        print(f"    {agent}: {[round(p, 2) for p in prices]}")

    return boundary_ratios


def analyze_rewards(records, name):
    """分析收益情况"""
    print(f"\n--- {name} 收益分析 ---")

    agents = list(records[0]['rewards'].keys())

    # 计算各智能体的平均收益
    agent_rewards = {agent: [] for agent in agents}
    for r in records:
        for agent in agents:
            agent_rewards[agent].append(r['rewards'][agent])

    print(f"  智能体平均收益:")
    for agent in sorted(agents):
        rewards = agent_rewards[agent]
        print(f"    {agent}: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")

    # 收益范围（用于确定静态归一化常数）
    all_rewards = [r['rewards'][a] for r in records for a in agents]
    print(f"\n  全局收益范围: [{min(all_rewards):.1f}, {max(all_rewards):.1f}]")
    print(f"  建议静态归一化常数: {max(all_rewards) * 1.1:.0f}")


def compare_algorithms(maddpg_records, mfddpg_records):
    """对比两种算法"""
    print(f"\n{'='*60}")
    print(f"  算法对比")
    print(f"{'='*60}")

    # 在相同 step 数下对比
    min_steps = min(len(maddpg_records), len(mfddpg_records))
    compare_points = [50, 100, 200, 300, min_steps-1]
    compare_points = [p for p in compare_points if p < min_steps]

    print(f"\n--- 相同 step 数时的 relative_change_rate ---")
    print(f"  {'Step':>6} | {'MADDPG':>10} | {'MF-DDPG':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")

    for step in compare_points:
        maddpg_rate = maddpg_records[step]['relative_change_rate']
        mfddpg_rate = mfddpg_records[step]['relative_change_rate']
        print(f"  {step:>6} | {maddpg_rate:>10.6f} | {mfddpg_rate:>10.6f}")


def main():
    results_dir = Path(__file__).parent.parent / 'results'

    # 加载数据
    print("加载实验数据...")
    maddpg_path = results_dir / 'step_records_maddpg.json'
    mfddpg_path = results_dir / 'step_records_mfddpg.json'

    if not maddpg_path.exists():
        print(f"错误: 找不到 {maddpg_path}")
        return
    if not mfddpg_path.exists():
        print(f"错误: 找不到 {mfddpg_path}")
        return

    maddpg_records = load_step_records(maddpg_path)
    mfddpg_records = load_step_records(mfddpg_path)

    print(f"MADDPG: {len(maddpg_records)} steps")
    print(f"MF-DDPG: {len(mfddpg_records)} steps")

    # 分析 MADDPG
    maddpg_rates = analyze_convergence(maddpg_records, "MADDPG")
    analyze_price_trend(maddpg_records, "MADDPG")
    analyze_rewards(maddpg_records, "MADDPG")

    # 分析 MF-DDPG
    mfddpg_rates = analyze_convergence(mfddpg_records, "MF-DDPG")
    analyze_price_trend(mfddpg_records, "MF-DDPG")
    analyze_rewards(mfddpg_records, "MF-DDPG")

    # 算法对比
    compare_algorithms(maddpg_records, mfddpg_records)

    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
