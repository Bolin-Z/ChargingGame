"""
流量详细分析脚本

输出每个站点、每个时段的流量值，以及归一化后的结果
"""

import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def analyze_dataset(network_dir: str, network_name: str, flow_scale_factor: float):
    """分析单个数据集"""
    print(f"\n{'='*70}")
    print(f"数据集: {network_name}  |  flow_scale_factor: {flow_scale_factor}")
    print(f"{'='*70}")

    # 创建环境并运行
    env = EVCSChargingGameEnv(network_dir=network_dir, network_name=network_name)
    env.reset()

    # 中点价格动作
    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env.step(actions)

    # 获取流量矩阵
    flows = env.charging_flow_history[-1]  # (n_agents, n_periods)
    flows_normalized = flows / flow_scale_factor

    agents = env.agents
    n_periods = env.n_periods

    # 打印原始流量矩阵
    print(f"\n原始流量矩阵 (站点 x 时段):")
    print(f"{'Station':<10}", end="")
    for p in range(n_periods):
        print(f"{'P'+str(p):>8}", end="")
    print(f"{'Total':>10}")
    print("-" * (10 + 8*n_periods + 10))

    for i, agent in enumerate(agents):
        print(f"{agent:<10}", end="")
        for p in range(n_periods):
            print(f"{flows[i,p]:>8.0f}", end="")
        print(f"{np.sum(flows[i]):>10.0f}")

    # 打印归一化后的流量矩阵
    print(f"\n归一化后流量矩阵 (÷ {flow_scale_factor}):")
    print(f"{'Station':<10}", end="")
    for p in range(n_periods):
        print(f"{'P'+str(p):>8}", end="")
    print(f"{'Max':>10}")
    print("-" * (10 + 8*n_periods + 10))

    for i, agent in enumerate(agents):
        print(f"{agent:<10}", end="")
        for p in range(n_periods):
            val = flows_normalized[i,p]
            print(f"{val:>8.2f}", end="")
        print(f"{np.max(flows_normalized[i]):>10.2f}")

    # 统计归一化后的分布
    all_normalized = flows_normalized.flatten()
    non_zero = all_normalized[all_normalized > 0]

    print(f"\n归一化后统计:")
    print(f"  非零值数量: {len(non_zero)} / {len(all_normalized)}")
    print(f"  非零值 Mean: {np.mean(non_zero):.2f}" if len(non_zero) > 0 else "  无非零值")
    print(f"  非零值 Min:  {np.min(non_zero):.2f}" if len(non_zero) > 0 else "")
    print(f"  非零值 Max:  {np.max(non_zero):.2f}" if len(non_zero) > 0 else "")
    print(f"  全部值 Max:  {np.max(all_normalized):.2f}")

    env.close()
    return flows, flows_normalized


def main():
    # flow_scale_factor: 让流量与价格 [0.5, 2.0] 保持相同数量级
    datasets = [
        ('data/siouxfalls', 'siouxfalls', 100),
        ('data/berlin_friedrichshain', 'berlin_friedrichshain', 25),
        ('data/anaheim', 'anaheim', 20),
    ]

    for network_dir, network_name, scale in datasets:
        try:
            analyze_dataset(network_dir, network_name, scale)
        except Exception as e:
            print(f"[ERROR] {network_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
