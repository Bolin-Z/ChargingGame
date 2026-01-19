"""
流量范围统计脚本

目的：在三个数据集上运行 UE-DTA（使用中点价格），统计流量分布情况，
      为 flow_scale_factor 配置提供数据依据。

使用方法：
    python scripts/analyze_flow_range.py

输出：
    - 每个数据集的流量统计（mean, max, min, std, percentiles）
    - 建议的 flow_scale_factor 值
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def analyze_dataset(network_dir: str, network_name: str) -> dict:
    """
    分析单个数据集的流量范围

    Args:
        network_dir: 数据集目录（相对于项目根目录）
        network_name: 数据集名称

    Returns:
        dict: 流量统计信息
    """
    print(f"\n{'='*60}")
    print(f"分析数据集: {network_name}")
    print(f"{'='*60}")

    # 创建环境
    print("正在初始化环境...")
    env = EVCSChargingGameEnv(network_dir=network_dir, network_name=network_name)

    # 获取价格范围（从 charging_nodes 字典中提取）
    # charging_nodes: {agent_id: [min_price, max_price], ...}
    first_agent = list(env.charging_nodes.keys())[0]
    price_min, price_max = env.charging_nodes[first_agent]
    mid_price = (price_min + price_max) / 2

    print(f"价格范围: [{price_min}, {price_max}]")
    print(f"中点价格: {mid_price}")
    print(f"充电站数量: {env.n_agents}")
    print(f"时段数量: {env.n_periods}")

    # 重置环境
    env.reset()

    # 构造中点价格动作（动作空间是 [0, 1]，需要转换）
    # action = 0 -> price_min, action = 1 -> price_max
    # mid_price 对应的 action = 0.5
    mid_action = 0.5
    actions = {
        agent: np.full(env.n_periods, mid_action, dtype=np.float32)
        for agent in env.agents
    }

    print(f"\n正在运行 UE-DTA（使用中点价格）...")

    # 执行一次 step
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 提取流量数据
    # 从 charging_flow_history 获取
    charging_flows = env.charging_flow_history[-1]  # shape: (n_agents, n_periods)

    print(f"\n流量矩阵形状: {charging_flows.shape}")

    # 统计信息
    all_flows = charging_flows.flatten()

    stats = {
        'network_name': network_name,
        'n_agents': env.n_agents,
        'n_periods': env.n_periods,
        'price_range': [price_min, price_max],
        'mid_price': mid_price,
        'flow_matrix': charging_flows,
        'flow_mean': np.mean(all_flows),
        'flow_std': np.std(all_flows),
        'flow_min': np.min(all_flows),
        'flow_max': np.max(all_flows),
        'flow_p50': np.percentile(all_flows, 50),
        'flow_p75': np.percentile(all_flows, 75),
        'flow_p90': np.percentile(all_flows, 90),
        'flow_p95': np.percentile(all_flows, 95),
        'flow_p99': np.percentile(all_flows, 99),
        'total_flow': np.sum(all_flows),
        # 按站点统计
        'flow_per_agent': np.sum(charging_flows, axis=1),  # 每个站点的总流量
        'flow_per_period': np.sum(charging_flows, axis=0),  # 每个时段的总流量
    }

    # 打印统计结果
    print(f"\n--- 流量统计 ---")
    print(f"总流量: {stats['total_flow']:.0f} 辆")
    print(f"Mean: {stats['flow_mean']:.1f}")
    print(f"Std:  {stats['flow_std']:.1f}")
    print(f"Min:  {stats['flow_min']:.0f}")
    print(f"Max:  {stats['flow_max']:.0f}")
    print(f"P50:  {stats['flow_p50']:.1f}")
    print(f"P75:  {stats['flow_p75']:.1f}")
    print(f"P90:  {stats['flow_p90']:.1f}")
    print(f"P95:  {stats['flow_p95']:.1f}")
    print(f"P99:  {stats['flow_p99']:.1f}")

    print(f"\n--- 各站点总流量 ---")
    for i, agent in enumerate(env.agents):
        print(f"  {agent}: {stats['flow_per_agent'][i]:.0f}")

    print(f"\n--- 各时段总流量 ---")
    for period in range(env.n_periods):
        print(f"  Period {period}: {stats['flow_per_period'][period]:.0f}")

    # 清理环境
    env.close()

    return stats


def suggest_flow_scale_factor(stats: dict) -> float:
    """
    根据流量统计建议 flow_scale_factor

    策略：使用 P95 或 Max 的较小者，确保归一化后大部分值在 [0, 1] 范围内
    """
    # 使用 P95 作为基准，留出一定余量
    p95 = stats['flow_p95']
    max_flow = stats['flow_max']

    # 取 P95 的 1.2 倍作为 scale factor，或者取 max 的 1.0 倍
    suggested = max(p95 * 1.2, max_flow * 1.0)

    # 向上取整到合适的数值（50 的倍数）
    suggested = np.ceil(suggested / 50) * 50

    return max(suggested, 50.0)  # 最小值 50


def main():
    """主函数"""
    print("="*60)
    print("流量范围统计工具")
    print("用于确定各数据集的 flow_scale_factor 配置")
    print("="*60)

    # 数据集配置
    datasets = [
        ('data/siouxfalls', 'siouxfalls'),
        ('data/berlin_friedrichshain', 'berlin_friedrichshain'),
        ('data/anaheim', 'anaheim'),
    ]

    all_stats = []

    for network_dir, network_name in datasets:
        try:
            stats = analyze_dataset(network_dir, network_name)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n[ERROR] 分析 {network_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果和建议
    print("\n")
    print("="*60)
    print("汇总结果与建议")
    print("="*60)

    print("\n| 数据集 | 站点数 | Mean | Max | P95 | 建议 flow_scale_factor |")
    print("|--------|--------|------|-----|-----|------------------------|")

    suggestions = {}
    for stats in all_stats:
        suggested = suggest_flow_scale_factor(stats)
        suggestions[stats['network_name']] = suggested
        print(f"| {stats['network_name']:14s} | {stats['n_agents']:6d} | {stats['flow_mean']:4.0f} | {stats['flow_max']:3.0f} | {stats['flow_p95']:3.0f} | {suggested:22.0f} |")

    print("\n建议在各数据集的 settings.json 中添加:")
    for name, factor in suggestions.items():
        print(f'  "{name}": "flow_scale_factor": {factor:.0f}')

    print("\n完成!")

    return all_stats, suggestions


if __name__ == "__main__":
    main()
