"""
对比不同 demand_multiplier 下各充电站的充电流量

目的：在保证收敛（GM < 3%）的前提下，选择合适的 dm 使充电流量不会过低

执行命令: python history/compare_dm_charging_flow.py
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def run_ue_dta_with_flow(env, n_iterations: int = 50) -> dict:
    """运行 UE-DTA 并收集充电流量"""

    # 设置固定中点价格
    actions = {}
    for agent in env.agents:
        actions[agent] = np.full(env.n_periods, 0.5, dtype=np.float32)
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    # 获取 OD 映射
    dict_od_to_charging_vehid = defaultdict(list)
    dict_od_to_uncharging_vehid = defaultdict(list)
    W_template = env._EVCSChargingGameEnv__create_simulation_world()

    for key, veh in W_template.VEHICLES.items():
        o = veh.orig.name
        d = veh.dest.name
        if veh.attribute.get("charging_car", False):
            dict_od_to_charging_vehid[(o, d)].append(key)
        else:
            dict_od_to_uncharging_vehid[(o, d)].append(key)

    total_charging_vehicles = sum(len(v) for v in dict_od_to_charging_vehid.values()) * env.deltan
    total_uncharging_vehicles = sum(len(v) for v in dict_od_to_uncharging_vehid.values()) * env.deltan
    total_vehicles = total_charging_vehicles + total_uncharging_vehicles

    del W_template

    # 初始化路径
    current_routes = env._EVCSChargingGameEnv__initialize_routes(
        dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
    )

    # 运行迭代
    final_flows = None
    final_gm = None

    for iteration in range(n_iterations):
        W = env._EVCSChargingGameEnv__create_simulation_world()
        env._EVCSChargingGameEnv__apply_routes_to_vehicles(W, current_routes)
        W.exec_simulation()

        stats, new_routes, charging_flows = env._EVCSChargingGameEnv__route_choice_update(
            W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
            current_routes, iteration
        )
        current_routes = new_routes
        final_flows = charging_flows
        final_gm = stats['all_relative_gap_global_mean'] * 100

        del W

    return {
        'final_gm': final_gm,
        'charging_flows': final_flows,  # shape: (n_agents, n_periods)
        'total_vehicles': total_vehicles,
        'total_charging_vehicles': total_charging_vehicles,
        'total_uncharging_vehicles': total_uncharging_vehicles,
        'agents': env.agents,
        'n_periods': env.n_periods
    }


def modify_settings(settings_path: str, dm: float) -> dict:
    """临时修改配置文件"""
    with open(settings_path, 'r', encoding='utf-8') as f:
        original = json.load(f)

    modified = original.copy()
    modified['demand_multiplier'] = dm

    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(modified, f, indent=4, ensure_ascii=False)

    return original


def restore_settings(settings_path: str, original: dict):
    """恢复原始配置"""
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(original, f, indent=4, ensure_ascii=False)


def main():
    print("=" * 80)
    print("不同 demand_multiplier 下充电站流量对比")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    settings_path = "data/berlin_friedrichshain/berlin_friedrichshain_settings.json"

    # 测试的 dm 值
    dm_values = [1.8, 2.0, 2.2, 2.5]
    n_iterations = 50

    results = {}

    # 保存原始配置
    with open(settings_path, 'r', encoding='utf-8') as f:
        original_settings = json.load(f)

    try:
        for dm in dm_values:
            print(f"\n测试 dm={dm}...")
            start_time = time.time()

            # 修改配置
            modify_settings(settings_path, dm)

            # 创建环境
            env = EVCSChargingGameEnv(
                network_dir="data/berlin_friedrichshain",
                network_name="berlin_friedrichshain",
                random_seed=42,
                max_steps=10,
                convergence_threshold=0.01,
                stable_steps_required=3
            )
            env.reset()

            # 运行测试
            result = run_ue_dta_with_flow(env, n_iterations)
            elapsed = time.time() - start_time

            results[dm] = result
            print(f"  完成，GM={result['final_gm']:.2f}%, 耗时={elapsed:.1f}s")

            env.close()

    finally:
        restore_settings(settings_path, original_settings)
        print("\n配置文件已恢复")

    # 输出对比结果
    print("\n" + "=" * 80)
    print("1. 总体流量对比")
    print("=" * 80)

    print(f"\n{'dm':<6} {'GM%':<8} {'总车辆':<10} {'充电车辆':<12} {'非充电车辆':<12} {'总充电流量':<12}")
    print("-" * 70)

    for dm in dm_values:
        r = results[dm]
        total_flow = np.sum(r['charging_flows'])
        status = "✅" if r['final_gm'] < 3.0 else "❌"
        print(f"{dm:<6} {r['final_gm']:<8.2f} {r['total_vehicles']:<10} "
              f"{r['total_charging_vehicles']:<12} {r['total_uncharging_vehicles']:<12} "
              f"{total_flow:<12.0f} {status}")

    # 各充电站流量对比
    print("\n" + "=" * 80)
    print("2. 各充电站总流量对比")
    print("=" * 80)

    agents = results[dm_values[0]]['agents']

    # 表头
    header = f"{'充电站':<12}"
    for dm in dm_values:
        header += f"dm={dm:<8}"
    print(f"\n{header}")
    print("-" * (12 + 12 * len(dm_values)))

    # 每个充电站
    station_flows = {dm: {} for dm in dm_values}
    for agent_idx, agent in enumerate(agents):
        row = f"{agent:<12}"
        for dm in dm_values:
            flow = np.sum(results[dm]['charging_flows'][agent_idx])
            station_flows[dm][agent] = flow
            row += f"{flow:<12.0f}"
        print(row)

    # 汇总行
    print("-" * (12 + 12 * len(dm_values)))
    row = f"{'总计':<12}"
    for dm in dm_values:
        total = sum(station_flows[dm].values())
        row += f"{total:<12.0f}"
    print(row)

    # 各时段流量对比
    print("\n" + "=" * 80)
    print("3. 各时段总流量对比")
    print("=" * 80)

    n_periods = results[dm_values[0]]['n_periods']

    header = f"{'时段':<8}"
    for dm in dm_values:
        header += f"dm={dm:<10}"
    print(f"\n{header}")
    print("-" * (8 + 12 * len(dm_values)))

    for period in range(n_periods):
        row = f"{period:<8}"
        for dm in dm_values:
            period_flow = np.sum(results[dm]['charging_flows'][:, period])
            row += f"{period_flow:<12.0f}"
        print(row)

    # 流量分布分析
    print("\n" + "=" * 80)
    print("4. 流量分布分析")
    print("=" * 80)

    for dm in dm_values:
        r = results[dm]
        flows = r['charging_flows']
        total_flow = np.sum(flows)
        station_totals = np.sum(flows, axis=1)

        print(f"\ndm={dm} (GM={r['final_gm']:.2f}%):")
        print(f"  总充电流量: {total_flow:.0f} 辆")
        print(f"  平均每站流量: {np.mean(station_totals):.1f} 辆")
        print(f"  流量最大站: {agents[np.argmax(station_totals)]} ({np.max(station_totals):.0f} 辆)")
        print(f"  流量最小站: {agents[np.argmin(station_totals)]} ({np.min(station_totals):.0f} 辆)")
        print(f"  流量标准差: {np.std(station_totals):.1f}")

        # 流量为0的充电站
        zero_flow_stations = [agents[i] for i, f in enumerate(station_totals) if f == 0]
        if zero_flow_stations:
            print(f"  ⚠️ 流量为0的站点: {zero_flow_stations}")

    # 建议
    print("\n" + "=" * 80)
    print("5. 建议")
    print("=" * 80)

    # 找出满足 GM < 3% 且流量最大的配置
    valid_configs = [(dm, results[dm]) for dm in dm_values if results[dm]['final_gm'] < 3.0]

    if valid_configs:
        best_dm, best_result = max(valid_configs, key=lambda x: np.sum(x[1]['charging_flows']))
        print(f"\n推荐配置: dm={best_dm}")
        print(f"  - GM: {best_result['final_gm']:.2f}% (< 3% ✅)")
        print(f"  - 总充电流量: {np.sum(best_result['charging_flows']):.0f} 辆")
        print(f"  - 总车辆数: {best_result['total_vehicles']} 辆")
        print(f"  - 充电车辆占比: {best_result['total_charging_vehicles']/best_result['total_vehicles']*100:.1f}%")
    else:
        print("\n⚠️ 没有满足 GM < 3% 的配置")


if __name__ == "__main__":
    main()
