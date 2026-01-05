"""
BF 数据集高 Gap 车辆的绝对/相对成本分析

验证假设：对于非充电车辆，相对时间Gap和相对成本Gap是否完全相同（tvc被约分）
分析高Gap车辆的成本结构特征
诊断：为什么某些车辆的 best_time 为 0
"""

import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


class CostAnalysisEnv(EVCSChargingGameEnv):
    """扩展环境，收集详细的成本数据"""

    def run_with_cost_details(self, n_iterations: int = 50, diagnose_zero_best: bool = True) -> dict:
        """运行 UE-DTA 并收集详细成本数据"""

        self.price_history = []

        # 设置固定中点价格
        actions = {}
        for agent in self.agents:
            actions[agent] = np.full(self.n_periods, 0.5, dtype=np.float32)
        self._EVCSChargingGameEnv__update_prices_from_actions(actions)

        # 获取 OD 映射
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        W_template = self._EVCSChargingGameEnv__create_simulation_world()

        for key, veh in W_template.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)
        del W_template

        # 初始化路径
        self.current_routes_specified = self._EVCSChargingGameEnv__initialize_routes(
            dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
        )

        # 运行迭代
        final_cost_details = []

        for iteration in range(n_iterations):
            W = self._EVCSChargingGameEnv__create_simulation_world()
            self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)
            W.exec_simulation()

            cost_details = []

            # 非充电车辆（重点分析）
            for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
                available_routes = self.dict_od_to_routes["uncharging"][od_pair]

                for veh_id in veh_ids:
                    if veh_id not in W.VEHICLES:
                        continue
                    veh = W.VEHICLES[veh_id]

                    if veh.state != "end":
                        continue

                    # 实际通行时间
                    route, timestamps = veh.traveled_route(include_departure_time=True)
                    tt_current = timestamps[-1] - timestamps[0]
                    cost_current = self.time_value_coefficient * tt_current

                    # 当前走的路径
                    current_route_links = [link.name for link in route if link is not None]

                    # 找最佳路径，同时记录所有候选路径成本
                    best_tt = tt_current
                    best_cost = cost_current
                    best_route_idx = -1  # 当前路径
                    all_route_costs = []  # 记录所有路径的估计成本

                    for idx, route_links in enumerate(available_routes):
                        route_obj = W.defRoute(route_links)
                        alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                            route_obj, veh.departure_time_in_second, False
                        )
                        # 反推时间
                        alt_tt = alt_cost / self.time_value_coefficient if self.time_value_coefficient > 0 else 0

                        all_route_costs.append({
                            'route_idx': idx,
                            'route_links': route_links,
                            'estimated_cost': alt_cost,
                            'estimated_tt': alt_tt
                        })

                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_tt = alt_tt
                            best_route_idx = idx

                    # 计算各种 Gap
                    abs_time_gap = tt_current - best_tt
                    abs_cost_gap = cost_current - best_cost

                    if tt_current > 0:
                        rel_time_gap = abs_time_gap / tt_current
                    else:
                        rel_time_gap = 0.0

                    if cost_current > 0:
                        rel_cost_gap = abs_cost_gap / cost_current
                    else:
                        rel_cost_gap = 0.0

                    detail = {
                        'veh_id': veh_id,
                        'od': od_pair,
                        'is_charging': False,
                        'departure_time': veh.departure_time_in_second,
                        # 时间
                        'tt_current': tt_current,
                        'tt_best': best_tt,
                        'abs_time_gap': abs_time_gap,
                        'rel_time_gap': rel_time_gap,
                        # 成本
                        'cost_current': cost_current,
                        'cost_best': best_cost,
                        'abs_cost_gap': abs_cost_gap,
                        'rel_cost_gap': rel_cost_gap,
                        # 路径数
                        'n_routes': len(available_routes),
                        # 诊断信息（最后一轮收集）
                        'current_route': current_route_links,
                        'best_route_idx': best_route_idx,
                        'all_route_costs': all_route_costs if (iteration == n_iterations - 1 and diagnose_zero_best) else None
                    }
                    cost_details.append(detail)

            # 充电车辆
            for od_pair, veh_ids in dict_od_to_charging_vehid.items():
                available_routes = self.dict_od_to_routes["charging"][od_pair]

                for veh_id in veh_ids:
                    if veh_id not in W.VEHICLES:
                        continue
                    veh = W.VEHICLES[veh_id]

                    if veh.state != "end":
                        continue

                    # 实际通行时间（包含充电时间）
                    route, timestamps = veh.traveled_route(include_departure_time=True)
                    tt_current = timestamps[-1] - timestamps[0]

                    # 实际总成本（通行时间成本 + 充电成本）
                    cost_current = self._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow(
                        veh, W, np.zeros((self.n_agents, self.n_periods))
                    )

                    # 找最佳路径
                    best_cost = cost_current
                    best_tt = tt_current

                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                            route_obj, veh.departure_time_in_second, True
                        )
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            # 对充电车辆，无法简单反推时间

                    # 计算 Gap
                    abs_time_gap = 0  # 充电车辆无法简单计算时间差（充电时间固定）
                    abs_cost_gap = cost_current - best_cost

                    if tt_current > 0:
                        rel_time_gap = 0  # 充电车辆不计算时间Gap
                    else:
                        rel_time_gap = 0.0

                    if cost_current > 0:
                        rel_cost_gap = abs_cost_gap / cost_current
                    else:
                        rel_cost_gap = 0.0

                    cost_details.append({
                        'veh_id': veh_id,
                        'od': od_pair,
                        'is_charging': True,
                        'departure_time': veh.departure_time_in_second,
                        # 时间
                        'tt_current': tt_current,
                        'tt_best': best_tt,
                        'abs_time_gap': abs_time_gap,
                        'rel_time_gap': rel_time_gap,
                        # 成本
                        'cost_current': cost_current,
                        'cost_best': best_cost,
                        'abs_cost_gap': abs_cost_gap,
                        'rel_cost_gap': rel_cost_gap,
                        # 路径数
                        'n_routes': len(available_routes)
                    })

            # 更新路径
            stats, new_routes, _ = self._EVCSChargingGameEnv__route_choice_update(
                W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                self.current_routes_specified, iteration
            )
            self.current_routes_specified = new_routes

            gm = stats['all_relative_gap_global_mean'] * 100
            print(f"迭代 {iteration+1:2d}: GM={gm:.2f}%")

            if iteration == n_iterations - 1:
                final_cost_details = cost_details

            del W

        return {
            'final_gm': gm,
            'cost_details': final_cost_details,
            'tvc': self.time_value_coefficient
        }


def analyze_cost_structure(cost_details: list, tvc: float):
    """分析成本结构"""

    # 分离充电/非充电
    uncharging = [d for d in cost_details if not d['is_charging']]
    charging = [d for d in cost_details if d['is_charging']]

    print("\n" + "=" * 70)
    print("1. 验证：非充电车辆的相对时间Gap vs 相对成本Gap")
    print("=" * 70)

    if uncharging:
        time_gaps = [d['rel_time_gap'] for d in uncharging]
        cost_gaps = [d['rel_cost_gap'] for d in uncharging]

        # 计算差异
        diffs = [abs(t - c) for t, c in zip(time_gaps, cost_gaps)]

        print(f"非充电车辆数: {len(uncharging)}")
        print(f"time_value_coefficient: {tvc}")
        print()
        print(f"相对时间Gap 均值: {np.mean(time_gaps)*100:.4f}%")
        print(f"相对成本Gap 均值: {np.mean(cost_gaps)*100:.4f}%")
        print(f"两者差异最大值:   {np.max(diffs)*100:.6f}%")
        print(f"两者差异均值:     {np.mean(diffs)*100:.6f}%")

        if np.max(diffs) < 1e-10:
            print("\n✅ 验证通过：相对时间Gap 与 相对成本Gap 完全相同（tvc被约分）")
        else:
            print("\n⚠️ 存在差异，需要进一步检查")

    print("\n" + "=" * 70)
    print("2. 非充电车辆的绝对成本差异分析")
    print("=" * 70)

    if uncharging:
        # 按相对Gap分组
        low_gap = [d for d in uncharging if d['rel_cost_gap'] < 0.03]
        mid_gap = [d for d in uncharging if 0.03 <= d['rel_cost_gap'] < 0.10]
        high_gap = [d for d in uncharging if d['rel_cost_gap'] >= 0.10]

        print(f"\n{'Gap区间':<15} {'车辆数':<10} {'平均绝对Gap(元)':<18} {'平均绝对Gap(秒)':<18} {'平均成本(元)':<15}")
        print("-" * 80)

        for name, group in [("低Gap(<3%)", low_gap), ("中Gap(3-10%)", mid_gap), ("高Gap(>=10%)", high_gap)]:
            if group:
                avg_abs_cost = np.mean([d['abs_cost_gap'] for d in group])
                avg_abs_time = np.mean([d['abs_time_gap'] for d in group])
                avg_cost = np.mean([d['cost_current'] for d in group])
                print(f"{name:<15} {len(group):<10} {avg_abs_cost:<18.4f} {avg_abs_time:<18.1f} {avg_cost:<15.4f}")

    print("\n" + "=" * 70)
    print("3. 高Gap车辆详细分析 (相对Gap >= 10%)")
    print("=" * 70)

    high_gap_vehicles = [d for d in uncharging if d['rel_cost_gap'] >= 0.10]
    high_gap_vehicles.sort(key=lambda x: x['rel_cost_gap'], reverse=True)

    if high_gap_vehicles:
        print(f"\n高Gap非充电车辆数: {len(high_gap_vehicles)}")
        print(f"\nTop 20 高Gap车辆:")
        print(f"{'OD对':<12} {'出发时间':<10} {'当前时间(s)':<12} {'最佳时间(s)':<12} {'绝对差(s)':<10} {'相对Gap%':<10}")
        print("-" * 70)

        for d in high_gap_vehicles[:20]:
            od_str = f"{d['od'][0]}->{d['od'][1]}"
            print(f"{od_str:<12} {d['departure_time']:<10.0f} {d['tt_current']:<12.1f} {d['tt_best']:<12.1f} {d['abs_time_gap']:<10.1f} {d['rel_cost_gap']*100:<10.2f}")

        print(f"\n高Gap车辆的成本/时间统计:")
        print(f"  平均当前时间: {np.mean([d['tt_current'] for d in high_gap_vehicles]):.1f}s")
        print(f"  平均最佳时间: {np.mean([d['tt_best'] for d in high_gap_vehicles]):.1f}s")
        print(f"  平均绝对时间差: {np.mean([d['abs_time_gap'] for d in high_gap_vehicles]):.1f}s")
        print(f"  平均相对Gap: {np.mean([d['rel_cost_gap'] for d in high_gap_vehicles])*100:.2f}%")

    print("\n" + "=" * 70)
    print("4. 按通行时间分组的Gap分析")
    print("=" * 70)

    if uncharging:
        # 按通行时间分组
        bins = [(0, 60), (60, 120), (120, 180), (180, 300), (300, 600), (600, float('inf'))]

        print(f"\n{'时间区间(s)':<15} {'车辆数':<10} {'平均相对Gap%':<15} {'平均绝对Gap(s)':<18}")
        print("-" * 60)

        for low, high in bins:
            group = [d for d in uncharging if low <= d['tt_current'] < high]
            if group:
                label = f"{low}-{high if high != float('inf') else '∞'}"
                avg_rel = np.mean([d['rel_cost_gap'] for d in group]) * 100
                avg_abs = np.mean([d['abs_time_gap'] for d in group])
                print(f"{label:<15} {len(group):<10} {avg_rel:<15.2f} {avg_abs:<18.1f}")

    print("\n" + "=" * 70)
    print("5. 充电 vs 非充电车辆对比")
    print("=" * 70)

    print(f"\n{'类型':<12} {'车辆数':<10} {'平均成本(元)':<15} {'平均绝对Gap(元)':<18} {'平均相对Gap%':<15}")
    print("-" * 75)

    if charging:
        avg_cost = np.mean([d['cost_current'] for d in charging])
        avg_abs = np.mean([d['abs_cost_gap'] for d in charging])
        avg_rel = np.mean([d['rel_cost_gap'] for d in charging]) * 100
        print(f"{'充电':<12} {len(charging):<10} {avg_cost:<15.2f} {avg_abs:<18.4f} {avg_rel:<15.2f}")

    if uncharging:
        avg_cost = np.mean([d['cost_current'] for d in uncharging])
        avg_abs = np.mean([d['abs_cost_gap'] for d in uncharging])
        avg_rel = np.mean([d['rel_cost_gap'] for d in uncharging]) * 100
        print(f"{'非充电':<12} {len(uncharging):<10} {avg_cost:<15.4f} {avg_abs:<18.4f} {avg_rel:<15.2f}")

    print("\n" + "=" * 70)
    print("6. 结论")
    print("=" * 70)

    if uncharging and charging:
        unc_abs = np.mean([d['abs_cost_gap'] for d in uncharging])
        unc_rel = np.mean([d['rel_cost_gap'] for d in uncharging]) * 100
        ch_abs = np.mean([d['abs_cost_gap'] for d in charging])
        ch_rel = np.mean([d['rel_cost_gap'] for d in charging]) * 100

        print(f"\n非充电车辆:")
        print(f"  - 平均绝对Gap: {unc_abs:.4f} 元 ({unc_abs/tvc:.1f} 秒)")
        print(f"  - 平均相对Gap: {unc_rel:.2f}%")

        print(f"\n充电车辆:")
        print(f"  - 平均绝对Gap: {ch_abs:.4f} 元")
        print(f"  - 平均相对Gap: {ch_rel:.2f}%")

        print(f"\n关键发现:")
        if unc_abs < ch_abs:
            print(f"  ✅ 非充电车辆的绝对误差 ({unc_abs:.4f}元) < 充电车辆 ({ch_abs:.4f}元)")
            print(f"  ⚠️ 但相对误差 ({unc_rel:.2f}%) > 充电车辆 ({ch_rel:.2f}%)")
            print(f"  → 高相对Gap是成本基数小导致的数学放大效应")


def diagnose_zero_best_time(cost_details: list, tvc: float):
    """诊断 best_time 为 0 的异常情况"""

    print("\n" + "=" * 80)
    print("7. 诊断：best_time 为 0 的异常车辆")
    print("=" * 80)

    # 找出 best_time 接近 0 的非充电车辆
    uncharging = [d for d in cost_details if not d['is_charging']]
    zero_best = [d for d in uncharging if d['tt_best'] < 1.0]  # best_time < 1秒

    print(f"\n总非充电车辆数: {len(uncharging)}")
    print(f"best_time < 1秒 的车辆数: {len(zero_best)}")

    if not zero_best:
        print("\n✅ 没有异常车辆")
        return

    print(f"\n占比: {len(zero_best)/len(uncharging)*100:.1f}%")

    # 详细分析前 5 个异常车辆
    print("\n" + "-" * 80)
    print("详细诊断（前 5 个异常车辆）")
    print("-" * 80)

    for i, d in enumerate(zero_best[:5]):
        print(f"\n【车辆 {i+1}】{d['veh_id']}")
        print(f"  OD对: {d['od'][0]} -> {d['od'][1]}")
        print(f"  出发时间: {d['departure_time']:.0f}s")
        print(f"  实际通行时间: {d['tt_current']:.1f}s")
        print(f"  最佳估计时间: {d['tt_best']:.1f}s")
        print(f"  相对Gap: {d['rel_cost_gap']*100:.2f}%")
        print(f"  当前走的路径: {d['current_route']}")
        print(f"  最佳路径索引: {d['best_route_idx']}")

        if d['all_route_costs']:
            print(f"\n  所有候选路径的估计成本:")
            for rc in d['all_route_costs']:
                is_best = " ← BEST" if rc['estimated_cost'] == d['cost_best'] else ""
                print(f"    路径{rc['route_idx']}: 估计成本={rc['estimated_cost']:.4f}元, "
                      f"估计时间={rc['estimated_tt']:.1f}s{is_best}")
                print(f"           链路: {rc['route_links'][:5]}{'...' if len(rc['route_links']) > 5 else ''}")

    # 统计分析
    print("\n" + "-" * 80)
    print("异常车辆统计")
    print("-" * 80)

    # 按 OD 对分组
    od_counts = defaultdict(int)
    for d in zero_best:
        od_counts[d['od']] += 1

    print(f"\n按 OD 对分布 (Top 10):")
    sorted_ods = sorted(od_counts.items(), key=lambda x: x[1], reverse=True)
    for od, count in sorted_ods[:10]:
        print(f"  {od[0]} -> {od[1]}: {count} 辆")

    # 检查是否有共同的最佳路径特征
    if zero_best[0]['all_route_costs']:
        print(f"\n最佳路径的估计成本分布:")
        best_costs = [d['cost_best'] for d in zero_best]
        print(f"  最小: {min(best_costs):.6f}")
        print(f"  最大: {max(best_costs):.6f}")
        print(f"  均值: {np.mean(best_costs):.6f}")

        # 检查是否都是某个特定路径
        print(f"\n最佳路径索引分布:")
        idx_counts = defaultdict(int)
        for d in zero_best:
            idx_counts[d['best_route_idx']] += 1
        for idx, count in sorted(idx_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  路径索引 {idx}: {count} 辆 ({count/len(zero_best)*100:.1f}%)")


def diagnose_route_cost_estimation(cost_details: list, env):
    """深入诊断路径成本估计问题"""

    print("\n" + "=" * 80)
    print("8. 深入诊断：actual_travel_time 返回值检查")
    print("=" * 80)

    # 找一个异常车辆
    uncharging = [d for d in cost_details if not d['is_charging']]
    zero_best = [d for d in uncharging if d['tt_best'] < 1.0 and d['all_route_costs']]

    if not zero_best:
        print("\n没有可诊断的异常车辆")
        return

    # 取第一个异常车辆进行深入分析
    sample = zero_best[0]
    print(f"\n样本车辆: {sample['veh_id']}")
    print(f"OD对: {sample['od']}")
    print(f"出发时间: {sample['departure_time']}s")

    # 重新创建仿真世界获取链路信息
    W = env._EVCSChargingGameEnv__create_simulation_world()

    # 检查 W.LINKS 类型
    print(f"\nW.LINKS 类型: {type(W.LINKS)}")
    if isinstance(W.LINKS, list):
        print(f"W.LINKS 长度: {len(W.LINKS)}")
        if W.LINKS:
            first_link = W.LINKS[0]
            print(f"第一个链路类型: {type(first_link)}, name={getattr(first_link, 'name', 'N/A')}")

    # 检查路径中的链路名称
    print(f"\n路径中的链路名称:")
    sample_route = sample['all_route_costs'][0]['route_links'][:3]
    print(f"  前3个链路: {sample_route}")

    # 使用 get_link 获取链路
    print(f"\n使用 W.get_link() 获取链路:")
    for link_name in sample_route:
        try:
            link = W.get_link(link_name)
            print(f"  {link_name}: 获取成功, length={link.length:.1f}m, vmax={link.vmax:.1f}m/s")
        except Exception as e:
            print(f"  {link_name}: ❌ 获取失败 - {e}")

    # 检查各候选路径的链路级 actual_travel_time
    print(f"\n检查各候选路径的链路级 actual_travel_time:")

    for rc in sample['all_route_costs'][:3]:  # 只检查前3条路径
        print(f"\n  路径 {rc['route_idx']}: {rc['route_links'][:5]}...")
        print(f"  估计总成本: {rc['estimated_cost']:.6f}元")
        print(f"  估计总时间: {rc['estimated_tt']:.1f}s")

        # 逐链路检查 - 使用 get_link
        total_time = 0.0
        print(f"  链路级详情:")
        for link_name in rc['route_links']:
            try:
                link = W.get_link(link_name)
                current_time = sample['departure_time'] + total_time
                att = link.actual_travel_time(current_time)
                free_flow_time = link.length / link.vmax if link.vmax > 0 else 0

                # 检查 traveltime_real 状态
                tr_len = len(link.traveltime_real) if hasattr(link, 'traveltime_real') else 0
                tr_sample = list(link.traveltime_real)[:3] if tr_len > 0 else []

                print(f"    {link_name}: att={att:.2f}s, free_flow={free_flow_time:.2f}s, "
                      f"traveltime_real长度={tr_len}, 前3值={tr_sample}")
                total_time += att
            except Exception as e:
                print(f"    {link_name}: ❌ 错误 - {e}")

        print(f"  链路时间累加: {total_time:.1f}s")

    # 额外：直接调用 __estimate_route_cost 检查
    print(f"\n直接调用 __estimate_route_cost 验证:")
    for rc in sample['all_route_costs'][:3]:
        route_obj = W.defRoute(rc['route_links'])
        cost = env._EVCSChargingGameEnv__estimate_route_cost(
            route_obj, sample['departure_time'], False
        )
        print(f"  路径{rc['route_idx']}: 直接调用成本={cost:.6f}元, 记录成本={rc['estimated_cost']:.6f}元")

    del W


def main():
    print("=" * 70)
    print("BF 数据集高 Gap 车辆成本结构分析")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    env = CostAnalysisEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()

    print(f"\ntime_value_coefficient: {env.time_value_coefficient}")
    print(f"routes_per_od: {env.routes_per_od}")
    print(f"运行 50 轮 UE-DTA 迭代...\n")

    result = env.run_with_cost_details(n_iterations=50, diagnose_zero_best=True)

    analyze_cost_structure(result['cost_details'], result['tvc'])

    # 新增诊断
    diagnose_zero_best_time(result['cost_details'], result['tvc'])
    diagnose_route_cost_estimation(result['cost_details'], env)

    env.close()

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
