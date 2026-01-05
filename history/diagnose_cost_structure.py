"""
成本结构对 Gap 的影响分析

验证假设：非充电车辆成本基数小，导致相对 Gap 被放大
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

    def run_cost_analysis(self, n_iterations: int = 50) -> dict:
        """运行 UE-DTA 并收集成本详情"""

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
        for iteration in range(n_iterations):
            W = self._EVCSChargingGameEnv__create_simulation_world()
            self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)
            W.exec_simulation()

            stats, new_routes, _ = self._EVCSChargingGameEnv__route_choice_update(
                W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                self.current_routes_specified, iteration
            )
            self.current_routes_specified = new_routes

            gm = stats['all_relative_gap_global_mean'] * 100
            print(f"迭代 {iteration+1:2d}: GM={gm:.2f}%")

            del W

        # 最后一轮收集详细数据
        W = self._EVCSChargingGameEnv__create_simulation_world()
        self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)
        W.exec_simulation()

        cost_details = []

        # 充电车辆
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"][od_pair]

            for veh_id in veh_ids:
                if veh_id not in W.VEHICLES:
                    continue
                veh = W.VEHICLES[veh_id]
                if veh.state != "end":
                    continue

                # 计算当前成本的各个组成部分
                route, timestamps = veh.traveled_route(include_departure_time=True)
                travel_time = timestamps[-1] - timestamps[0]
                travel_time_cost = self.time_value_coefficient * travel_time

                # 充电成本（简化：假设使用了充电站）
                charging_cost = self.charging_demand_per_vehicle * 1.25  # 中点价格

                current_cost = travel_time_cost + charging_cost

                # 计算最优替代成本
                best_cost = current_cost
                for route_links in available_routes:
                    route_obj = W.defRoute(route_links)
                    alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                        route_obj, veh.departure_time_in_second, True
                    )
                    if alt_cost < best_cost:
                        best_cost = alt_cost

                absolute_gap = current_cost - best_cost
                relative_gap = absolute_gap / current_cost if current_cost > 0 else 0

                cost_details.append({
                    'veh_id': veh_id,
                    'is_charging': True,
                    'travel_time': travel_time,
                    'travel_time_cost': travel_time_cost,
                    'charging_cost': charging_cost,
                    'total_cost': current_cost,
                    'best_cost': best_cost,
                    'absolute_gap': absolute_gap,
                    'relative_gap': relative_gap
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
                travel_time_cost = self.time_value_coefficient * travel_time

                current_cost = travel_time_cost  # 无充电成本

                best_cost = current_cost
                for route_links in available_routes:
                    route_obj = W.defRoute(route_links)
                    alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                        route_obj, veh.departure_time_in_second, False
                    )
                    if alt_cost < best_cost:
                        best_cost = alt_cost

                absolute_gap = current_cost - best_cost
                relative_gap = absolute_gap / current_cost if current_cost > 0 else 0

                cost_details.append({
                    'veh_id': veh_id,
                    'is_charging': False,
                    'travel_time': travel_time,
                    'travel_time_cost': travel_time_cost,
                    'charging_cost': 0,
                    'total_cost': current_cost,
                    'best_cost': best_cost,
                    'absolute_gap': absolute_gap,
                    'relative_gap': relative_gap
                })

        del W
        return cost_details


def analyze_cost_structure(cost_details: list):
    """分析成本结构"""

    charging = [d for d in cost_details if d['is_charging']]
    uncharging = [d for d in cost_details if not d['is_charging']]

    print("\n" + "=" * 70)
    print("成本结构对比分析")
    print("=" * 70)

    # 基础统计
    print("\n### 成本基数对比")
    print("-" * 70)
    print(f"{'指标':<25} {'充电车辆':<20} {'非充电车辆':<20}")
    print("-" * 70)

    if charging:
        c_travel_cost = np.mean([d['travel_time_cost'] for d in charging])
        c_charging_cost = np.mean([d['charging_cost'] for d in charging])
        c_total_cost = np.mean([d['total_cost'] for d in charging])
        c_travel_time = np.mean([d['travel_time'] for d in charging])
    else:
        c_travel_cost = c_charging_cost = c_total_cost = c_travel_time = 0

    if uncharging:
        u_travel_cost = np.mean([d['travel_time_cost'] for d in uncharging])
        u_total_cost = np.mean([d['total_cost'] for d in uncharging])
        u_travel_time = np.mean([d['travel_time'] for d in uncharging])
    else:
        u_travel_cost = u_total_cost = u_travel_time = 0

    print(f"{'平均通行时间 (秒)':<25} {c_travel_time:<20.1f} {u_travel_time:<20.1f}")
    print(f"{'平均通行时间成本 (元)':<25} {c_travel_cost:<20.2f} {u_travel_cost:<20.2f}")
    print(f"{'平均充电成本 (元)':<25} {c_charging_cost:<20.2f} {'0.00':<20}")
    print(f"{'平均总成本 (元)':<25} {c_total_cost:<20.2f} {u_total_cost:<20.2f}")

    # 成本占比
    if c_total_cost > 0:
        c_travel_pct = c_travel_cost / c_total_cost * 100
        c_charging_pct = c_charging_cost / c_total_cost * 100
        print(f"\n充电车辆成本构成: 通行时间 {c_travel_pct:.1f}% + 充电 {c_charging_pct:.1f}%")

    # Gap 对比
    print("\n" + "=" * 70)
    print("Gap 对比分析")
    print("=" * 70)

    if charging:
        c_abs_gap = np.mean([d['absolute_gap'] for d in charging])
        c_rel_gap = np.mean([d['relative_gap'] for d in charging]) * 100
    else:
        c_abs_gap = c_rel_gap = 0

    if uncharging:
        u_abs_gap = np.mean([d['absolute_gap'] for d in uncharging])
        u_rel_gap = np.mean([d['relative_gap'] for d in uncharging]) * 100
    else:
        u_abs_gap = u_rel_gap = 0

    print(f"\n{'指标':<25} {'充电车辆':<20} {'非充电车辆':<20}")
    print("-" * 70)
    print(f"{'平均绝对 Gap (元)':<25} {c_abs_gap:<20.4f} {u_abs_gap:<20.4f}")
    print(f"{'平均相对 Gap (%)':<25} {c_rel_gap:<20.2f} {u_rel_gap:<20.2f}")

    # 关键验证：如果假设正确，绝对 Gap 应该相近，但相对 Gap 差异大
    print("\n" + "=" * 70)
    print("假设验证")
    print("=" * 70)

    if c_abs_gap > 0 and u_abs_gap > 0:
        abs_gap_ratio = u_abs_gap / c_abs_gap
        rel_gap_ratio = u_rel_gap / c_rel_gap if c_rel_gap > 0 else float('inf')
        cost_base_ratio = c_total_cost / u_total_cost if u_total_cost > 0 else float('inf')

        print(f"\n绝对 Gap 比值 (非充电/充电): {abs_gap_ratio:.2f}x")
        print(f"相对 Gap 比值 (非充电/充电): {rel_gap_ratio:.2f}x")
        print(f"成本基数比值 (充电/非充电): {cost_base_ratio:.2f}x")

        print("\n验证结论:")
        if abs_gap_ratio < 5 and rel_gap_ratio > 10:
            print("  ✅ 假设成立：绝对 Gap 相近，但相对 Gap 差异大")
            print("  → 非充电车辆的高相对 Gap 是由成本基数小导致的")
        elif abs_gap_ratio > 5:
            print("  ❌ 假设不成立：非充电车辆的绝对 Gap 也更大")
            print("  → 可能是路径选择问题，不仅仅是成本基数问题")
        else:
            print("  ⚠️ 需要进一步分析")

    # 模拟：如果给非充电车辆加上"虚拟充电成本"会怎样
    print("\n" + "=" * 70)
    print("模拟实验：给非充电车辆加虚拟充电成本")
    print("=" * 70)

    if uncharging:
        virtual_charging_cost = 62.5  # 与充电车辆相同的充电成本

        adjusted_gaps = []
        for d in uncharging:
            new_total = d['total_cost'] + virtual_charging_cost
            new_best = d['best_cost'] + virtual_charging_cost  # 假设最优路径也加相同成本
            new_rel_gap = d['absolute_gap'] / new_total if new_total > 0 else 0
            adjusted_gaps.append(new_rel_gap)

        adjusted_gm = np.mean(adjusted_gaps) * 100

        print(f"\n原始非充电车辆 GM: {u_rel_gap:.2f}%")
        print(f"加虚拟成本后 GM:   {adjusted_gm:.2f}%")
        print(f"降幅: {u_rel_gap - adjusted_gm:.2f}%")

        if adjusted_gm < 1:
            print("\n  ✅ 加虚拟成本后 GM 接近充电车辆水平")
            print("  → 确认：高 Gap 是成本基数问题，不是路径选择问题")


def main():
    print("=" * 70)
    print("成本结构对 Gap 影响分析")
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
    print(f"charging_demand_per_vehicle: {env.charging_demand_per_vehicle}")
    print(f"运行 50 轮 UE-DTA 迭代...\n")

    cost_details = env.run_cost_analysis(n_iterations=50)

    analyze_cost_structure(cost_details)

    env.close()

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
