"""
UE-DTA 收敛测试脚本

测试目标：
1. time_value_coefficient: 找到时间成本与充电成本平衡的值
2. ue_convergence_threshold: 观察不同阈值下的收敛行为

输出：
- 每个参数组合的 UE 迭代次数、收敛状态
- 时间成本 vs 充电成本比例
- 充电站流量分布（标准差）
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def test_single_config(network_dir: str, network_name: str,
                       time_value_coeff: float, ue_threshold: float,
                       fixed_price_ratio: float = 0.5) -> dict:
    """
    测试单个参数配置

    Args:
        network_dir: 网络数据目录
        network_name: 网络名称
        time_value_coeff: 时间价值系数
        ue_threshold: UE收敛阈值
        fixed_price_ratio: 固定价格比例 (0=最低价, 1=最高价, 0.5=中间价)

    Returns:
        测试结果字典
    """
    # 创建环境
    env = EVCSChargingGameEnv(
        network_dir=network_dir,
        network_name=network_name,
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    # 直接修改环境对象的参数（不修改文件）
    env.time_value_coefficient = time_value_coeff
    env.ue_convergence_threshold = ue_threshold

    # 重置环境
    obs, info = env.reset()

    # 构造固定价格动作（所有充电站相同价格）
    actions = {}
    for agent in env.agents:
        actions[agent] = np.full(env.n_periods, fixed_price_ratio, dtype=np.float32)

    # 执行一步，记录时间
    start_time = time.time()
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    elapsed_time = time.time() - start_time

    # 提取结果
    ue_stats = infos.get('ue_stats', {})

    # 计算时间成本和充电成本的比例
    # 从 ue_stats 中获取信息
    charging_avg_cost = ue_stats.get('charging_avg_cost', 0)
    uncharging_avg_cost = ue_stats.get('uncharging_avg_cost', 0)

    # 充电车辆的总成本 = 时间成本 + 充电成本
    # 非充电车辆的成本 = 纯时间成本
    # 因此：充电成本 ≈ charging_avg_cost - uncharging_avg_cost (近似)
    # 时间成本 ≈ uncharging_avg_cost

    # 更准确的方式：从 charging_flow 计算
    charging_flows = env.charging_flow_history[-1] if env.charging_flow_history else np.zeros((env.n_agents, env.n_periods))

    # 计算充电站流量统计
    total_flow_per_station = charging_flows.sum(axis=1)  # 每个站的总流量
    flow_std = np.std(total_flow_per_station)
    flow_mean = np.mean(total_flow_per_station)
    flow_cv = flow_std / flow_mean if flow_mean > 0 else 0  # 变异系数

    # 计算实际价格
    prices = env.actions_to_prices_matrix(actions)
    avg_price = np.mean(prices)

    result = {
        "time_value_coeff": time_value_coeff,
        "ue_threshold": ue_threshold,
        "ue_converged": infos.get('ue_converged', False),
        "ue_iterations": infos.get('ue_iterations', 0),
        "final_cost_gap": ue_stats.get('all_avg_cost_gap', float('inf')),
        "charging_avg_cost": charging_avg_cost,
        "uncharging_avg_cost": uncharging_avg_cost,
        "cost_diff": charging_avg_cost - uncharging_avg_cost,  # 充电带来的额外成本
        "time_cost_estimate": uncharging_avg_cost,  # 时间成本估计
        "flow_total": total_flow_per_station.sum(),
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "flow_cv": flow_cv,
        "elapsed_time": elapsed_time,
        "total_rewards": sum(rewards.values()),
        "avg_price": avg_price,
        "completed_ratio": ue_stats.get('completed_total_vehicles', 0) / max(ue_stats.get('total_vehicles', 1), 1),
    }

    return result


def run_parameter_sweep(network_dir: str, network_name: str):
    """运行参数扫描"""

    # 参数范围
    time_value_coeffs = [0.001, 0.005, 0.01, 0.02, 0.05]
    ue_thresholds = [0.5, 1.0, 2.0]

    print("=" * 80)
    print("UE-DTA 收敛测试 - 参数扫描")
    print("=" * 80)
    print(f"网络: {network_name}")
    print(f"time_value_coefficient: {time_value_coeffs}")
    print(f"ue_convergence_threshold: {ue_thresholds}")
    print("=" * 80)

    results = []

    for ue_thresh in ue_thresholds:
        print(f"\n--- ue_convergence_threshold = {ue_thresh} ---")
        print(f"{'time_coeff':>12} | {'UE迭代':>6} | {'收敛':>4} | {'成本差':>8} | "
              f"{'充电成本':>10} | {'时间成本':>10} | {'比例':>8} | {'流量CV':>8} | {'耗时':>6}")
        print("-" * 100)

        for tv_coeff in time_value_coeffs:
            result = test_single_config(
                network_dir, network_name,
                time_value_coeff=tv_coeff,
                ue_threshold=ue_thresh
            )
            results.append(result)

            # 计算时间成本与充电成本的比例
            if result['cost_diff'] > 0:
                ratio = result['time_cost_estimate'] / result['cost_diff']
                ratio_str = f"{ratio:.2f}:1"
            else:
                ratio_str = "N/A"

            converged_str = "是" if result['ue_converged'] else "否"

            print(f"{tv_coeff:>12.4f} | {result['ue_iterations']:>6d} | {converged_str:>4} | "
                  f"{result['final_cost_gap']:>8.3f} | {result['cost_diff']:>10.2f} | "
                  f"{result['time_cost_estimate']:>10.2f} | {ratio_str:>8} | "
                  f"{result['flow_cv']:>8.3f} | {result['elapsed_time']:>5.1f}s")

    # 输出分析建议
    print("\n" + "=" * 80)
    print("分析建议")
    print("=" * 80)

    # 找到时间成本与充电成本比例接近 1:1 的配置
    best_balance = None
    best_ratio_diff = float('inf')

    for r in results:
        if r['cost_diff'] > 0 and r['time_cost_estimate'] > 0:
            ratio = r['time_cost_estimate'] / r['cost_diff']
            ratio_diff = abs(ratio - 1.0)  # 距离 1:1 的差距
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_balance = r

    if best_balance:
        print(f"\n[推荐] 时间成本与充电成本最平衡的配置:")
        print(f"  time_value_coefficient = {best_balance['time_value_coeff']}")
        ratio = best_balance['time_cost_estimate'] / best_balance['cost_diff']
        print(f"  时间成本:充电成本 ≈ {ratio:.2f}:1")
        print(f"  UE收敛迭代次数: {best_balance['ue_iterations']}")

    # 返回结果供进一步分析
    return results


def analyze_cost_components(network_dir: str, network_name: str, time_value_coeff: float):
    """
    详细分析成本组成

    在指定的 time_value_coefficient 下，分析：
    1. 典型的旅行时间范围
    2. 充电价格范围
    3. 时间成本 vs 充电成本的具体数值
    """
    print("\n" + "=" * 80)
    print(f"成本组成详细分析 (time_value_coefficient = {time_value_coeff})")
    print("=" * 80)

    # 加载配置获取价格范围
    settings_path = os.path.join(network_dir, f"{network_name}_settings.json")
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    charging_nodes = settings["charging_nodes"]
    charging_demand = settings["charging_demand_per_vehicle"]

    # 价格范围
    price_min = min(v[0] for v in charging_nodes.values())
    price_max = max(v[1] for v in charging_nodes.values())
    price_mid = (price_min + price_max) / 2

    print(f"\n充电价格范围: [{price_min}, {price_max}] 元/kWh")
    print(f"单次充电需求: {charging_demand} kWh")
    print(f"充电成本范围: [{price_min * charging_demand:.1f}, {price_max * charging_demand:.1f}] 元")

    # 估算时间成本（基于仿真时间和典型旅行时间）
    sim_time = settings["simulation_time"]
    print(f"\n仿真时长: {sim_time}s = {sim_time/60:.1f}min")

    # 假设典型旅行时间
    typical_travel_times = [300, 600, 900, 1200]  # 5, 10, 15, 20 分钟
    print(f"\n典型旅行时间下的成本对比:")
    print(f"{'旅行时间':>10} | {'时间成本':>10} | {'充电成本(中)':>12} | {'比例':>10}")
    print("-" * 50)

    charging_cost_mid = price_mid * charging_demand
    for tt in typical_travel_times:
        time_cost = time_value_coeff * tt
        ratio = time_cost / charging_cost_mid if charging_cost_mid > 0 else 0
        print(f"{tt/60:>8.1f}min | {time_cost:>10.2f} | {charging_cost_mid:>12.2f} | {ratio:>8.2f}:1")

    # 计算达到 1:1 比例所需的旅行时间
    target_travel_time = charging_cost_mid / time_value_coeff if time_value_coeff > 0 else float('inf')
    print(f"\n[参考] 达到 1:1 比例所需的旅行时间: {target_travel_time:.0f}s = {target_travel_time/60:.1f}min")


if __name__ == "__main__":
    # 配置
    network_dir = "data/berlin_friedrichshain"
    network_name = "berlin_friedrichshain"

    # 运行参数扫描
    results = run_parameter_sweep(network_dir, network_name)

    # 对当前配置进行详细分析
    analyze_cost_components(network_dir, network_name, time_value_coeff=0.005)
