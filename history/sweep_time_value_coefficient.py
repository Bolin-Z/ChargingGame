"""
time_value_coefficient 参数扫描

测试不同的时间价值系数对 Gap 收敛的影响
"""

import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def test_time_value_coefficient(tvc: float, n_iterations: int = 50) -> dict:
    """测试指定的 time_value_coefficient"""

    env = EVCSChargingGameEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    # 覆盖 time_value_coefficient
    env.time_value_coefficient = tvc

    env.reset()

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
    del W_template

    # 初始化路径
    env.current_routes_specified = env._EVCSChargingGameEnv__initialize_routes(
        dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
    )

    # 运行迭代
    gm_history = []
    for iteration in range(n_iterations):
        W = env._EVCSChargingGameEnv__create_simulation_world()
        env._EVCSChargingGameEnv__apply_routes_to_vehicles(W, env.current_routes_specified)
        W.exec_simulation()

        stats, new_routes, _ = env._EVCSChargingGameEnv__route_choice_update(
            W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
            env.current_routes_specified, iteration
        )
        env.current_routes_specified = new_routes

        gm = stats['all_relative_gap_global_mean'] * 100
        gm_history.append(gm)

        del W

    # 最后一轮收集分类数据
    W = env._EVCSChargingGameEnv__create_simulation_world()
    env._EVCSChargingGameEnv__apply_routes_to_vehicles(W, env.current_routes_specified)
    W.exec_simulation()

    charging_gaps = []
    uncharging_gaps = []
    charging_costs = []
    uncharging_costs = []

    # 充电车辆
    for od_pair, veh_ids in dict_od_to_charging_vehid.items():
        available_routes = env.dict_od_to_routes["charging"][od_pair]
        for veh_id in veh_ids:
            if veh_id not in W.VEHICLES:
                continue
            veh = W.VEHICLES[veh_id]
            if veh.state != "end":
                continue

            current_cost = env._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow(
                veh, W, np.zeros((env.n_agents, env.n_periods))
            )
            charging_costs.append(current_cost)

            best_cost = current_cost
            for route_links in available_routes:
                route_obj = W.defRoute(route_links)
                alt_cost = env._EVCSChargingGameEnv__estimate_route_cost(
                    route_obj, veh.departure_time_in_second, True
                )
                if alt_cost < best_cost:
                    best_cost = alt_cost

            if current_cost > 0:
                gap = (current_cost - best_cost) / current_cost
            else:
                gap = 0.0
            charging_gaps.append(gap)

    # 非充电车辆
    for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
        available_routes = env.dict_od_to_routes["uncharging"][od_pair]
        for veh_id in veh_ids:
            if veh_id not in W.VEHICLES:
                continue
            veh = W.VEHICLES[veh_id]
            if veh.state != "end":
                continue

            route, timestamps = veh.traveled_route(include_departure_time=True)
            travel_time = timestamps[-1] - timestamps[0]
            current_cost = env.time_value_coefficient * travel_time
            uncharging_costs.append(current_cost)

            best_cost = current_cost
            for route_links in available_routes:
                route_obj = W.defRoute(route_links)
                alt_cost = env._EVCSChargingGameEnv__estimate_route_cost(
                    route_obj, veh.departure_time_in_second, False
                )
                if alt_cost < best_cost:
                    best_cost = alt_cost

            if current_cost > 0:
                gap = (current_cost - best_cost) / current_cost
            else:
                gap = 0.0
            uncharging_gaps.append(gap)

    del W
    env.close()

    return {
        'tvc': tvc,
        'final_gm': gm_history[-1],
        'min_gm': min(gm_history[-20:]),  # 最后20轮的最小值
        'avg_gm': np.mean(gm_history[-20:]),  # 最后20轮的平均值
        'charging_gm': np.mean(charging_gaps) * 100 if charging_gaps else 0,
        'uncharging_gm': np.mean(uncharging_gaps) * 100 if uncharging_gaps else 0,
        'charging_cost': np.mean(charging_costs) if charging_costs else 0,
        'uncharging_cost': np.mean(uncharging_costs) if uncharging_costs else 0,
    }


def main():
    print("=" * 80)
    print("time_value_coefficient 参数扫描")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 测试不同的 time_value_coefficient
    # 当前值是 0.005，测试范围从 0.005 到 0.1
    tvc_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]

    results = []

    for tvc in tvc_values:
        print(f"\n{'='*60}")
        print(f"测试 time_value_coefficient = {tvc}")
        print(f"{'='*60}")

        result = test_time_value_coefficient(tvc, n_iterations=50)
        results.append(result)

        print(f"\n结果: GM={result['final_gm']:.2f}%, "
              f"充电={result['charging_gm']:.2f}%, "
              f"非充电={result['uncharging_gm']:.2f}%")

    # 汇总表格
    print("\n" + "=" * 80)
    print("参数扫描结果汇总")
    print("=" * 80)

    print(f"\n{'TVC':<8} {'总GM%':<10} {'充电GM%':<12} {'非充电GM%':<12} "
          f"{'充电成本':<12} {'非充电成本':<12} {'成本比':<10}")
    print("-" * 80)

    for r in results:
        cost_ratio = r['charging_cost'] / r['uncharging_cost'] if r['uncharging_cost'] > 0 else 0
        print(f"{r['tvc']:<8.3f} {r['final_gm']:<10.2f} {r['charging_gm']:<12.2f} "
              f"{r['uncharging_gm']:<12.2f} {r['charging_cost']:<12.2f} "
              f"{r['uncharging_cost']:<12.2f} {cost_ratio:<10.1f}x")

    # 找最优值
    best = min(results, key=lambda x: x['final_gm'])
    print(f"\n最优 time_value_coefficient: {best['tvc']}")
    print(f"  总 GM: {best['final_gm']:.2f}%")
    print(f"  充电 GM: {best['charging_gm']:.2f}%")
    print(f"  非充电 GM: {best['uncharging_gm']:.2f}%")

    print("\n" + "=" * 80)
    print("扫描完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
