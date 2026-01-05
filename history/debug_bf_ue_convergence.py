"""
诊断脚本：分析 BF 数据集 UE-DTA 不收敛问题

检查项目：
1. actual_travel_time 时间单位是否正确
2. traveled_route 返回的时间是否正确
3. 成本计算是否存在单位不一致问题
4. 收敛指标的具体数值

执行命令: python history/debug_bf_ue_convergence.py
"""

import os
import sys
import json
import numpy as np

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def test_time_units():
    """测试1：检查时间单位是否一致"""
    print("\n" + "=" * 60)
    print("测试1：时间单位一致性检查")
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

    # 获取 World 对象
    W = env.W

    print(f"\n基础参数:")
    print(f"  delta_t (秒/时步): {W.delta_t}")
    print(f"  deltan (platoon大小): {W.deltan}")
    print(f"  t_max (仿真总时长): {W.t_max}")

    # 设置固定价格并运行一次仿真
    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    # 仅运行一次仿真（不做路径更新）
    print(f"\n运行仿真...")
    env.W.exec_simulation()

    # 检查车辆时间
    print(f"\n车辆时间检查（前10辆完成的车辆）:")
    completed_count = 0
    for veh in W.VEHICLES.values():
        if veh.state == "end" and completed_count < 10:
            route, timestamps = veh.traveled_route()
            travel_time = timestamps[-1] - timestamps[0]

            print(f"\n  车辆 {veh.name[:30]}...")
            print(f"    departure_time (C++): {veh.departure_time}")
            print(f"    departure_time_in_second: {veh.departure_time_in_second}")
            print(f"    traveled_route timestamps: {timestamps[:3]}...{timestamps[-1:]}")
            print(f"    计算的 travel_time: {travel_time}")
            print(f"    veh.travel_time: {veh.travel_time}")

            # 检查是否匹配
            if abs(travel_time - veh.travel_time) > 1:
                print(f"    ⚠️ 不匹配！差异: {abs(travel_time - veh.travel_time)}")
            else:
                print(f"    ✅ 匹配")

            completed_count += 1

    env.close()
    return True


def test_actual_travel_time():
    """测试2：检查 link.actual_travel_time 方法"""
    print("\n" + "=" * 60)
    print("测试2：Link.actual_travel_time 方法检查")
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

    # 检查几条链路的 actual_travel_time
    print(f"\n链路 actual_travel_time 检查:")
    for i, link in enumerate(W.LINKS[:5]):
        print(f"\n  链路 {link.name}:")
        print(f"    length: {link.length} m")
        print(f"    vmax (free_flow_speed): {link.vmax} m/s")
        print(f"    free_flow_travel_time: {link.length / link.vmax:.2f} s")

        # 检查不同时刻的 actual_travel_time
        for t in [0, 1000, 2000, 5000]:
            att = link.actual_travel_time(t)
            print(f"    actual_travel_time(t={t}s): {att:.2f} s")

        # 检查缓存相关属性
        if hasattr(link, '_traveltime_cache'):
            print(f"    _traveltime_cache_delta_t: {link._traveltime_cache_delta_t}")
            print(f"    _traveltime_cache length: {len(link._traveltime_cache)}")
        else:
            print(f"    traveltime_real length: {len(link.traveltime_real)}")

    env.close()
    return True


def test_ue_convergence_one_iteration():
    """测试3：运行一轮 UE-DTA 并检查收敛指标"""
    print("\n" + "=" * 60)
    print("测试3：UE-DTA 一轮迭代详细分析")
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

    # 设置固定价格
    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}

    # 运行 step
    print(f"\n运行 env.step()...")
    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    # infos 是全局字典，直接访问
    print(f"\n收敛信息:")
    print(f"  ue_converged: {infos['ue_converged']}")
    print(f"  ue_iterations: {infos['ue_iterations']}")

    if 'ue_stats' in infos:
        stats = infos['ue_stats']
        print(f"\n收敛指标:")
        print(f"  all_relative_gap_global_mean: {stats.get('all_relative_gap_global_mean', 'N/A'):.4%}")
        print(f"  all_relative_gap_p90: {stats.get('all_relative_gap_p90', 'N/A'):.4%}")
        print(f"  all_relative_gap_p95: {stats.get('all_relative_gap_p95', 'N/A'):.4%}")

        print(f"\n完成率:")
        total_veh = stats.get('total_vehicles', 0)
        completed_veh = stats.get('completed_total_vehicles', 0)
        print(f"  总车辆: {total_veh}")
        print(f"  完成车辆: {completed_veh}")
        print(f"  完成率: {completed_veh / total_veh:.2%}" if total_veh > 0 else "  完成率: N/A")

        print(f"\n成本统计:")
        print(f"  all_avg_cost: {stats.get('all_avg_cost', 'N/A'):.4f}")
        print(f"  charging_avg_cost: {stats.get('charging_avg_cost', 'N/A'):.4f}")
        print(f"  uncharging_avg_cost: {stats.get('uncharging_avg_cost', 'N/A'):.4f}")

    env.close()
    return True


def test_cost_calculation_consistency():
    """测试4：检查成本计算一致性（当前成本 vs 估计成本）"""
    print("\n" + "=" * 60)
    print("测试4：成本计算一致性检查")
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

    # 运行仿真
    W.exec_simulation()

    print(f"\n成本计算对比（前5辆完成的充电车辆）:")
    checked = 0
    for veh in W.VEHICLES.values():
        if veh.state == "end" and veh.attribute.get("charging_car", False) and checked < 5:
            route, timestamps = veh.traveled_route()

            # 实际成本（从 traveled_route）
            actual_travel_time = timestamps[-1] - timestamps[0]
            actual_time_cost = env.time_value_coefficient * actual_travel_time

            # 估计成本（使用 actual_travel_time 方法）
            route_obj = W.defRoute([l.name for l in route.links])
            estimated_travel_time = route_obj.actual_travel_time(veh.departure_time_in_second)
            estimated_time_cost = env.time_value_coefficient * estimated_travel_time

            print(f"\n  车辆 {veh.name[:30]}...")
            print(f"    departure_time_in_second: {veh.departure_time_in_second}")
            print(f"    实际 travel_time: {actual_travel_time:.2f} s")
            print(f"    估计 travel_time: {estimated_travel_time:.2f} s")
            print(f"    实际 time_cost: {actual_time_cost:.4f}")
            print(f"    估计 time_cost: {estimated_time_cost:.4f}")

            diff = abs(actual_travel_time - estimated_travel_time)
            if diff > 10:  # 超过10秒差异
                print(f"    ⚠️ 差异较大: {diff:.2f} s")
            else:
                print(f"    ✅ 差异正常: {diff:.2f} s")

            checked += 1

    env.close()
    return True


def main():
    """运行所有诊断测试"""
    print("\n" + "#" * 60)
    print("# BF 数据集 UE-DTA 收敛问题诊断")
    print("#" * 60)

    results = []

    try:
        results.append(("时间单位一致性", test_time_units()))
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("时间单位一致性", False))

    try:
        results.append(("actual_travel_time", test_actual_travel_time()))
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("actual_travel_time", False))

    try:
        results.append(("UE-DTA迭代分析", test_ue_convergence_one_iteration()))
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("UE-DTA迭代分析", False))

    try:
        results.append(("成本计算一致性", test_cost_calculation_consistency()))
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("成本计算一致性", False))

    # 汇总
    print("\n" + "=" * 60)
    print("诊断结果汇总")
    print("=" * 60)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print("\n完成！")


if __name__ == "__main__":
    main()
