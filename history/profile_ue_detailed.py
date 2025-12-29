"""
UE-DTA 内部耗时详细分析脚本

分析单轮 UE-DTA 迭代中各部分的耗时：
1. create_simulation_world - 创建仿真世界
2. apply_routes_to_vehicles - 应用路径
3. exec_simulation - UXSim 仿真执行
4. route_choice_update - 路径选择更新
   4.1 calculate_actual_cost - 计算实际成本
   4.2 estimate_route_cost - 估算备选路径成本
   4.3 switch_decision - 切换决策
"""

import time
import sys
import os
import numpy as np
from collections import defaultdict
from functools import wraps

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


class DetailedProfiler:
    """细粒度计时器"""

    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)

    def record(self, name, elapsed):
        self.timings[name].append(elapsed)
        self.call_counts[name] += 1

    def summary(self):
        print("\n" + "="*80)
        print("UE-DTA 内部耗时详细分析")
        print("="*80)

        # 计算总时间
        total = sum(sum(v) for v in self.timings.values())

        print(f"\n{'组件':<45} {'总耗时(ms)':<12} {'占比':<10} {'次数':<8} {'平均(ms)':<10}")
        print("-"*80)

        # 按总耗时排序
        sorted_items = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        for name, times in sorted_items:
            total_ms = sum(times) * 1000
            pct = (sum(times) / total * 100) if total > 0 else 0
            count = len(times)
            avg_ms = np.mean(times) * 1000 if times else 0
            print(f"{name:<45} {total_ms:<12.2f} {pct:<10.1f}% {count:<8} {avg_ms:<10.3f}")

        print("-"*80)
        print(f"{'总计':<45} {total*1000:<12.2f}")
        print("="*80)


def profile_ue_iteration(env, profiler):
    """分析单次 UE-DTA 迭代的各部分耗时"""

    # 获取内部数据结构
    dict_od_to_charging_vehid = defaultdict(list)
    dict_od_to_uncharging_vehid = defaultdict(list)

    # 1. 创建仿真世界
    t0 = time.perf_counter()
    W = env._EVCSChargingGameEnv__create_simulation_world()
    profiler.record("1. create_simulation_world", time.perf_counter() - t0)

    # 构建 OD 映射
    for key, veh in W.VEHICLES.items():
        o = veh.orig.name
        d = veh.dest.name
        if veh.attribute["charging_car"]:
            dict_od_to_charging_vehid[(o, d)].append(key)
        else:
            dict_od_to_uncharging_vehid[(o, d)].append(key)

    # 初始化路径（如果需要）
    if not hasattr(env, 'current_routes_specified'):
        env.current_routes_specified = env._EVCSChargingGameEnv__initialize_routes(
            dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
        )

    # 2. 应用路径
    t0 = time.perf_counter()
    env._EVCSChargingGameEnv__apply_routes_to_vehicles(W, env.current_routes_specified)
    profiler.record("2. apply_routes_to_vehicles", time.perf_counter() - t0)

    # 3. 执行仿真
    t0 = time.perf_counter()
    W.exec_simulation()
    profiler.record("3. exec_simulation (UXSim)", time.perf_counter() - t0)

    # 4. 路径选择更新 - 分解分析
    t_route_update_start = time.perf_counter()

    # 统计各子部分
    t_actual_cost_total = 0
    t_estimate_cost_total = 0
    t_switch_decision_total = 0
    n_vehicles_processed = 0
    n_route_evaluations = 0

    charging_flows = np.zeros((env.n_agents, env.n_periods))
    new_routes_specified = {}

    # 处理充电车辆
    for od_pair, veh_ids in dict_od_to_charging_vehid.items():
        available_routes = env.dict_od_to_routes["charging"][od_pair]

        for veh_id in veh_ids:
            veh = W.VEHICLES[veh_id]
            n_vehicles_processed += 1

            if veh.state != "end":
                new_routes_specified[veh_id] = env.current_routes_specified[veh_id]
                continue

            # 4.1 计算实际成本
            t0 = time.perf_counter()
            current_cost = env._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow(
                veh, W, charging_flows
            )
            t_actual_cost_total += time.perf_counter() - t0

            current_route = env.current_routes_specified[veh_id]
            best_cost = current_cost
            best_route = current_route

            # 4.2 估算备选路径成本
            if available_routes:
                for route_links in available_routes:
                    t0 = time.perf_counter()
                    route_obj = W.defRoute(route_links)
                    alt_cost = env._EVCSChargingGameEnv__estimate_route_cost(
                        route_obj, veh.departure_time_in_second, True
                    )
                    t_estimate_cost_total += time.perf_counter() - t0
                    n_route_evaluations += 1

                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_links

            # 4.3 切换决策
            t0 = time.perf_counter()
            if current_cost > 0:
                relative_gap = (current_cost - best_cost) / current_cost
            else:
                relative_gap = 0.0

            if relative_gap > 0:
                gap_factor = min(1.0, env.ue_switch_gamma * relative_gap)
                decay_factor = 1.0 / (1.0 + env.ue_switch_alpha * 0)  # iteration=0
                switch_prob = gap_factor * decay_factor
                if np.random.random() < switch_prob:
                    new_routes_specified[veh_id] = best_route
                else:
                    new_routes_specified[veh_id] = current_route
            else:
                new_routes_specified[veh_id] = current_route
            t_switch_decision_total += time.perf_counter() - t0

    # 处理非充电车辆（简化，只统计数量）
    for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
        for veh_id in veh_ids:
            new_routes_specified[veh_id] = env.current_routes_specified[veh_id]

    profiler.record("4.1 calculate_actual_cost", t_actual_cost_total)
    profiler.record("4.2 estimate_route_cost", t_estimate_cost_total)
    profiler.record("4.3 switch_decision", t_switch_decision_total)
    profiler.record("4. route_choice_update (总计)", time.perf_counter() - t_route_update_start)

    # 更新路径
    env.current_routes_specified = new_routes_specified

    return {
        'n_vehicles': n_vehicles_processed,
        'n_route_evaluations': n_route_evaluations,
        'avg_routes_per_vehicle': n_route_evaluations / max(1, n_vehicles_processed)
    }


def main():
    print("="*80)
    print("UE-DTA 内部耗时详细分析")
    print("="*80)

    # 创建环境
    print("\n正在创建环境...")
    env = EVCSChargingGameEnv(
        network_dir="data/siouxfalls",
        network_name="siouxfalls",
        random_seed=42,
        max_steps=50,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    # 重置环境
    observations, _ = env.reset()

    # 设置一个固定价格（中间价格）
    actions = {agent: np.full(env.n_periods, 0.5) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    # 创建分析器
    profiler = DetailedProfiler()

    # 运行多轮 UE-DTA 迭代
    n_iterations = 10
    print(f"\n开始分析 {n_iterations} 轮 UE-DTA 迭代...")
    print("-"*80)

    stats_list = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        stats = profile_ue_iteration(env, profiler)
        elapsed = time.perf_counter() - t0
        stats_list.append(stats)

        print(f"迭代 {i+1:2d}: {elapsed*1000:7.1f}ms | "
              f"车辆数: {stats['n_vehicles']:4d} | "
              f"路径评估: {stats['n_route_evaluations']:5d} | "
              f"平均路径/车: {stats['avg_routes_per_vehicle']:.1f}")

    # 输出统计摘要
    profiler.summary()

    # 额外分析
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)

    # 计算各部分占比
    exec_sim_total = sum(profiler.timings.get("3. exec_simulation (UXSim)", []))
    route_update_total = sum(profiler.timings.get("4. route_choice_update (总计)", []))
    estimate_cost_total = sum(profiler.timings.get("4.2 estimate_route_cost", []))

    total_all = exec_sim_total + route_update_total

    print(f"\n瓶颈分析:")
    print(f"  - UXSim 仿真占比: {exec_sim_total/total_all*100:.1f}%")
    print(f"  - 路径更新占比: {route_update_total/total_all*100:.1f}%")
    if route_update_total > 0:
        print(f"    - 其中路径成本估算占路径更新的: {estimate_cost_total/route_update_total*100:.1f}%")

    avg_vehicles = np.mean([s['n_vehicles'] for s in stats_list])
    avg_evaluations = np.mean([s['n_route_evaluations'] for s in stats_list])

    print(f"\n计算规模:")
    print(f"  - 平均每轮处理车辆: {avg_vehicles:.0f}")
    print(f"  - 平均每轮路径评估: {avg_evaluations:.0f}")
    print(f"  - 每次路径评估耗时: {estimate_cost_total/sum(s['n_route_evaluations'] for s in stats_list)*1000:.3f}ms")

    print("\n优化建议:")
    if exec_sim_total > route_update_total:
        print("  → 主要瓶颈是 UXSim 仿真")
        print("  → 建议: 增大 deltan, 减少 simulation_time, 或使用更简单的网络")
    else:
        print("  → 主要瓶颈是路径成本计算")
        print("  → 建议: 减少 routes_per_od, 或缓存路径成本")


if __name__ == "__main__":
    main()
