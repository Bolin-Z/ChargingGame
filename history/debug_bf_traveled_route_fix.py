"""
诊断脚本2：验证 traveled_route 时间差异是否是问题根源

检查 EVCSChargingGameEnv 中使用 traveled_route 的地方，
对比使用 include_departure_time=True 和 False 的差异

执行命令: python history/debug_bf_traveled_route_fix.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def analyze_traveled_route_behavior():
    """分析 traveled_route 的两种调用方式"""
    print("\n" + "=" * 60)
    print("分析 traveled_route 行为差异")
    print("=" * 60)

    env = EVCSChargingGameEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=5,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()
    W = env.W

    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    print(f"\n运行仿真...")
    W.exec_simulation()

    print(f"\n对比 traveled_route 两种调用方式（前10辆完成的车辆）:")
    print("-" * 80)

    checked = 0
    for veh in W.VEHICLES.values():
        if veh.state == "end" and checked < 10:
            # 默认调用方式（当前代码使用的）
            route1, ts1 = veh.traveled_route()
            travel_time_default = ts1[-1] - ts1[0]

            # 包含出发时间的调用方式
            route2, ts2 = veh.traveled_route(include_departure_time=True)
            travel_time_with_departure = ts2[-1] - ts2[0]

            print(f"\n车辆: {veh.name[:40]}")
            print(f"  departure_time: {veh.departure_time}")
            print(f"  veh.travel_time: {veh.travel_time}")
            print(f"  默认调用 (include_departure_time=False):")
            print(f"    timestamps[0]: {ts1[0]}, timestamps[-1]: {ts1[-1]}")
            print(f"    计算的 travel_time: {travel_time_default}")
            print(f"  包含出发时间 (include_departure_time=True):")
            print(f"    timestamps[0]: {ts2[0]}, timestamps[-1]: {ts2[-1]}")
            print(f"    计算的 travel_time: {travel_time_with_departure}")

            # 哪种更接近 veh.travel_time?
            diff_default = abs(travel_time_default - veh.travel_time)
            diff_with_departure = abs(travel_time_with_departure - veh.travel_time)

            if diff_with_departure < diff_default:
                print(f"  ✅ include_departure_time=True 更准确 (差异: {diff_with_departure} vs {diff_default})")
            else:
                print(f"  ⚠️ 两者差异相当或默认更好")

            checked += 1

    env.close()


def check_env_usage_of_traveled_route():
    """检查 EVCSChargingGameEnv 中 traveled_route 的使用情况"""
    print("\n" + "=" * 60)
    print("检查代码中 traveled_route 的使用")
    print("=" * 60)

    env_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "env", "EVCSChargingGameEnv.py"
    )

    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找出所有使用 traveled_route 的地方
    lines = content.split('\n')
    print(f"\nEVCSChargingGameEnv.py 中使用 traveled_route 的位置:")
    for i, line in enumerate(lines, 1):
        if 'traveled_route' in line:
            print(f"  行 {i}: {line.strip()[:80]}")


def simulate_ue_with_corrected_travel_time():
    """模拟使用修正后的 travel_time 计算 relative_gap"""
    print("\n" + "=" * 60)
    print("模拟修正后的 relative_gap 计算")
    print("=" * 60)

    env = EVCSChargingGameEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=5,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()
    W = env.W

    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    print(f"\n运行仿真...")
    W.exec_simulation()

    # 收集使用两种方式计算的 relative_gap
    gaps_default = []
    gaps_corrected = []

    print(f"\n对比两种计算方式的 relative_gap（充电车辆）:")

    for veh in W.VEHICLES.values():
        if veh.state != "end":
            continue
        if not veh.attribute.get("charging_car", False):
            continue

        # 获取实际路径
        route, ts_default = veh.traveled_route()
        _, ts_corrected = veh.traveled_route(include_departure_time=True)

        # 计算实际成本（两种方式）
        travel_time_default = ts_default[-1] - ts_default[0]
        travel_time_corrected = ts_corrected[-1] - ts_corrected[0]

        current_cost_default = env.time_value_coefficient * travel_time_default
        current_cost_corrected = env.time_value_coefficient * travel_time_corrected

        # 估计最优成本（使用 departure_time_in_second）
        route_obj = W.defRoute([l.name for l in route.links])
        estimated_travel_time = route_obj.actual_travel_time(veh.departure_time_in_second)
        best_cost = env.time_value_coefficient * estimated_travel_time

        # 添加充电成本
        for i, link in enumerate(route.links):
            if hasattr(link, 'attribute') and link.attribute.get("charging_link", False):
                if link.name.startswith("charging_"):
                    charging_node = link.name.split("charging_")[1]
                    entry_time = ts_default[i] if i < len(ts_default) else ts_default[-1]
                    period = int(entry_time / env.period_duration) % env.n_periods
                    agent_idx = env.agents.index(f"evcs_{charging_node}")
                    charging_price = env.current_prices[agent_idx, period]
                    charging_cost = charging_price * env.charging_demand_per_vehicle
                    current_cost_default += charging_cost
                    current_cost_corrected += charging_cost
                    best_cost += charging_cost

        # 计算 relative_gap
        if current_cost_default > 0:
            gap_default = (current_cost_default - best_cost) / current_cost_default
            gaps_default.append(gap_default)

        if current_cost_corrected > 0:
            gap_corrected = (current_cost_corrected - best_cost) / current_cost_corrected
            gaps_corrected.append(gap_corrected)

    # 统计
    if gaps_default and gaps_corrected:
        print(f"\n统计结果（{len(gaps_default)} 辆充电车辆）:")
        print(f"  默认方式 (include_departure_time=False):")
        print(f"    mean: {np.mean(gaps_default):.4%}")
        print(f"    p90:  {np.percentile(gaps_default, 90):.4%}")
        print(f"    p95:  {np.percentile(gaps_default, 95):.4%}")
        print(f"  修正方式 (include_departure_time=True):")
        print(f"    mean: {np.mean(gaps_corrected):.4%}")
        print(f"    p90:  {np.percentile(gaps_corrected, 90):.4%}")
        print(f"    p95:  {np.percentile(gaps_corrected, 95):.4%}")

        print(f"\n结论:")
        if np.mean(gaps_corrected) < np.mean(gaps_default):
            print(f"  ✅ 使用 include_departure_time=True 可以降低 gap")
            print(f"     差异: {np.mean(gaps_default) - np.mean(gaps_corrected):.4%}")
        else:
            print(f"  ⚠️ 修正方式并未改善 gap")

    env.close()


def main():
    print("\n" + "#" * 60)
    print("# traveled_route 时间差异分析")
    print("#" * 60)

    analyze_traveled_route_behavior()
    check_env_usage_of_traveled_route()
    simulate_ue_with_corrected_travel_time()

    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    print("""
如果 include_departure_time=True 能改善 gap，考虑修改：
1. EVCSChargingGameEnv.__calculate_actual_vehicle_cost_and_flow()
2. EVCSChargingGameEnv.__route_choice_update() 中非充电车辆的处理

或者检查 uxsimpp 和原 UXsim 的 traveled_route 行为是否一致。
""")


if __name__ == "__main__":
    main()
