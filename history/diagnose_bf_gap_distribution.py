"""
BF 数据集 Gap 分布诊断

分析：
1. 哪些 OD 对贡献了高 Gap？
2. Gap 是否与出发时段相关？
3. 增加 routes_per_od 是否有帮助？
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


class GapDiagnosticEnv(EVCSChargingGameEnv):
    """扩展环境，收集详细的 Gap 数据"""

    def run_with_gap_details(self, n_iterations: int = 50) -> dict:
        """运行 UE-DTA 并收集详细 Gap 数据"""

        # 重置状态
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

        # 运行迭代，收集最后一轮的详细数据
        final_gap_details = []

        for iteration in range(n_iterations):
            W = self._EVCSChargingGameEnv__create_simulation_world()
            self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)
            W.exec_simulation()

            # 收集 Gap 详情
            gap_details = []

            # 充电车辆
            for od_pair, veh_ids in dict_od_to_charging_vehid.items():
                available_routes = self.dict_od_to_routes["charging"][od_pair]

                for veh_id in veh_ids:
                    if veh_id not in W.VEHICLES:
                        continue
                    veh = W.VEHICLES[veh_id]

                    if veh.state != "end":
                        continue

                    current_cost = self._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow(
                        veh, W, np.zeros((self.n_agents, self.n_periods))
                    )

                    best_cost = current_cost
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                            route_obj, veh.departure_time_in_second, True
                        )
                        if alt_cost < best_cost:
                            best_cost = alt_cost

                    if current_cost > 0:
                        gap = (current_cost - best_cost) / current_cost
                    else:
                        gap = 0.0

                    # 计算出发时段
                    period = int(veh.departure_time_in_second / self.period_duration)
                    period = max(0, min(period, self.n_periods - 1))

                    gap_details.append({
                        'veh_id': veh_id,
                        'od': od_pair,
                        'departure_time': veh.departure_time_in_second,
                        'period': period,
                        'is_charging': True,
                        'current_cost': current_cost,
                        'best_cost': best_cost,
                        'gap': gap,
                        'n_available_routes': len(available_routes)
                    })

            # 非充电车辆
            for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
                available_routes = self.dict_od_to_routes["uncharging"][od_pair]

                for veh_id in veh_ids:
                    if veh_id not in W.VEHICLES:
                        continue
                    veh = W.VEHICLES[veh_id]

                    if veh.state != "end":
                        continue

                    route, timestamps = veh.traveled_route(include_departure_time=True)
                    travel_time = timestamps[-1] - timestamps[0]
                    current_cost = self.time_value_coefficient * travel_time

                    best_cost = current_cost
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                            route_obj, veh.departure_time_in_second, False
                        )
                        if alt_cost < best_cost:
                            best_cost = alt_cost

                    if current_cost > 0:
                        gap = (current_cost - best_cost) / current_cost
                    else:
                        gap = 0.0

                    period = int(veh.departure_time_in_second / self.period_duration)
                    period = max(0, min(period, self.n_periods - 1))

                    gap_details.append({
                        'veh_id': veh_id,
                        'od': od_pair,
                        'departure_time': veh.departure_time_in_second,
                        'period': period,
                        'is_charging': False,
                        'current_cost': current_cost,
                        'best_cost': best_cost,
                        'gap': gap,
                        'n_available_routes': len(available_routes)
                    })

            # 更新路径
            stats, new_routes, _ = self._EVCSChargingGameEnv__route_choice_update(
                W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                self.current_routes_specified, iteration
            )
            self.current_routes_specified = new_routes

            gm = stats['all_relative_gap_global_mean'] * 100
            print(f"迭代 {iteration+1:2d}: GM={gm:.2f}%")

            # 保存最后一轮数据
            if iteration == n_iterations - 1:
                final_gap_details = gap_details

            del W

        return {
            'final_gm': gm,
            'gap_details': final_gap_details
        }


def analyze_gap_distribution(gap_details: list):
    """分析 Gap 分布"""

    gaps = [d['gap'] for d in gap_details]

    print("\n" + "=" * 60)
    print("Gap 分布统计")
    print("=" * 60)
    print(f"总车辆数: {len(gaps)}")
    print(f"GM (平均): {np.mean(gaps)*100:.2f}%")
    print(f"中位数: {np.median(gaps)*100:.2f}%")
    print(f"P90: {np.percentile(gaps, 90)*100:.2f}%")
    print(f"P95: {np.percentile(gaps, 95)*100:.2f}%")
    print(f"最大值: {np.max(gaps)*100:.2f}%")

    # Gap 分布
    print("\nGap 分布:")
    bins = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for g in gaps if bins[i] <= g < bins[i+1])
        pct = count / len(gaps) * 100
        bar = "#" * int(pct / 2)
        print(f"  {bins[i]*100:5.1f}%-{bins[i+1]*100:5.1f}%: {count:5d} ({pct:5.1f}%) {bar}")

    # 高 Gap 车辆 (>10%) 分析
    high_gap = [d for d in gap_details if d['gap'] > 0.1]
    print(f"\n高 Gap 车辆 (>10%): {len(high_gap)} 辆 ({len(high_gap)/len(gaps)*100:.1f}%)")

    # 按时段分析
    print("\n" + "=" * 60)
    print("按时段分析")
    print("=" * 60)
    period_gaps = defaultdict(list)
    for d in gap_details:
        period_gaps[d['period']].append(d['gap'])

    print(f"{'时段':<6} {'车辆数':<8} {'GM%':<8} {'P90%':<8} {'P95%':<8}")
    print("-" * 40)
    for period in sorted(period_gaps.keys()):
        g = period_gaps[period]
        print(f"{period:<6} {len(g):<8} {np.mean(g)*100:<8.2f} {np.percentile(g,90)*100:<8.2f} {np.percentile(g,95)*100:<8.2f}")

    # 按 OD 对分析（找高 Gap OD）
    print("\n" + "=" * 60)
    print("高 Gap OD 对 (Top 10)")
    print("=" * 60)
    od_gaps = defaultdict(list)
    for d in gap_details:
        od_gaps[d['od']].append(d['gap'])

    od_stats = []
    for od, g in od_gaps.items():
        od_stats.append({
            'od': od,
            'count': len(g),
            'gm': np.mean(g),
            'max': np.max(g),
            'n_routes': gap_details[next(i for i, d in enumerate(gap_details) if d['od'] == od)]['n_available_routes']
        })

    od_stats.sort(key=lambda x: x['gm'], reverse=True)

    print(f"{'OD对':<15} {'车辆数':<8} {'GM%':<8} {'Max%':<8} {'路径数':<8}")
    print("-" * 50)
    for stat in od_stats[:10]:
        od_str = f"{stat['od'][0]}->{stat['od'][1]}"
        print(f"{od_str:<15} {stat['count']:<8} {stat['gm']*100:<8.2f} {stat['max']*100:<8.2f} {stat['n_routes']:<8}")

    # 充电 vs 非充电
    print("\n" + "=" * 60)
    print("充电 vs 非充电车辆")
    print("=" * 60)
    charging = [d for d in gap_details if d['is_charging']]
    uncharging = [d for d in gap_details if not d['is_charging']]

    print(f"{'类型':<12} {'车辆数':<10} {'GM%':<10} {'P95%':<10}")
    print("-" * 45)
    if charging:
        print(f"{'充电':<12} {len(charging):<10} {np.mean([d['gap'] for d in charging])*100:<10.2f} {np.percentile([d['gap'] for d in charging], 95)*100:<10.2f}")
    if uncharging:
        print(f"{'非充电':<12} {len(uncharging):<10} {np.mean([d['gap'] for d in uncharging])*100:<10.2f} {np.percentile([d['gap'] for d in uncharging], 95)*100:<10.2f}")

    # 路径数 vs Gap 的关系
    print("\n" + "=" * 60)
    print("路径数 vs Gap")
    print("=" * 60)
    routes_gaps = defaultdict(list)
    for d in gap_details:
        routes_gaps[d['n_available_routes']].append(d['gap'])

    print(f"{'路径数':<10} {'车辆数':<10} {'GM%':<10}")
    print("-" * 35)
    for n_routes in sorted(routes_gaps.keys()):
        g = routes_gaps[n_routes]
        print(f"{n_routes:<10} {len(g):<10} {np.mean(g)*100:<10.2f}")


def main():
    print("=" * 60)
    print("BF 数据集 Gap 分布诊断")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    env = GapDiagnosticEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()

    print(f"\nroutes_per_od: {env.routes_per_od}")
    print(f"运行 50 轮 UE-DTA 迭代...\n")

    result = env.run_with_gap_details(n_iterations=50)

    analyze_gap_distribution(result['gap_details'])

    env.close()

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
