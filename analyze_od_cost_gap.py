"""
分析各OD对的cost_gap分布
目的：检查"全局平均"是否掩盖了局部不均衡

方法：继承EVCSChargingGameEnv，重写__run_simulation以收集OD级别数据
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from uxsim import World


class ODAnalysisEnv(EVCSChargingGameEnv):
    """扩展环境类，用于收集OD级别的cost_gap数据"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.od_analysis_results = []

    def _run_simulation_with_od_analysis(self, forced_iterations: int) -> tuple:
        """
        运行UE-DTA并收集每轮的OD级别数据

        Args:
            forced_iterations: 强制迭代次数（忽略收敛判断）
        """
        self.od_analysis_results = []

        # 获取充电和非充电车辆的OD映射
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        W_template = self._EVCSChargingGameEnv__create_simulation_world()

        for key, veh in W_template.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute["charging_car"]:
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)

        # 初始化路径分配
        if not hasattr(self, 'current_routes_specified') or not self.current_routes_specified:
            self.current_routes_specified = self._EVCSChargingGameEnv__initialize_routes(
                dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
            )

        final_charging_flows = None
        final_stats = None

        with tqdm(range(forced_iterations), desc="OD分析", leave=True,
                  bar_format='{desc} | {n}/{total} [{elapsed}<{remaining}]') as pbar:
            for iteration in pbar:
                # 创建新的仿真实例
                W = self._EVCSChargingGameEnv__create_simulation_world()

                # 应用当前路径分配
                self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)

                # 执行仿真
                W.exec_simulation()

                # 收集OD级别数据（在路径切换之前）
                od_data = self._collect_od_data(
                    W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid
                )

                # 执行路径切换
                stats, new_routes, charging_flows = self._EVCSChargingGameEnv__route_choice_update(
                    W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, self.current_routes_specified
                )

                # 保存分析结果
                iteration_result = {
                    "iteration": iteration + 1,
                    "od_data": od_data,
                    "global_stats": {
                        "all_avg_cost_gap": stats['all_avg_cost_gap'],
                        "charging_avg_cost_gap": stats['charging_avg_cost_gap'],
                        "uncharging_avg_cost_gap": stats['uncharging_avg_cost_gap'],
                        "total_route_switches": stats['total_route_switches']
                    }
                }
                self.od_analysis_results.append(iteration_result)

                final_charging_flows = charging_flows
                final_stats = stats

                # 更新路径分配
                self.current_routes_specified = new_routes

                # 计算OD级别统计
                od_avg_gaps = [od["avg_cost_gap"] for od in od_data if od.get("avg_cost_gap") is not None]
                od_max_avg = max(od_avg_gaps) if od_avg_gaps else 0

                pbar.set_description(
                    f"迭代{iteration+1} | 全局avg:{stats['all_avg_cost_gap']:.3f} | "
                    f"OD_max_avg:{od_max_avg:.3f} | 切换:{stats['total_route_switches']}"
                )

        return final_charging_flows, final_stats

    def _collect_od_data(self, W: World, dict_od_to_charging_vehid: dict,
                         dict_od_to_uncharging_vehid: dict) -> list:
        """收集每个OD对的cost_gap数据"""
        od_stats = []
        dummy_flows = np.zeros((self.n_agents, self.n_periods))

        # 处理充电车辆
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            cost_gaps = []
            available_routes = self.dict_od_to_routes["charging"][od_pair]

            for veh_id in veh_ids:
                veh = W.VEHICLES[veh_id]
                if veh.state != "end":
                    continue

                # 计算当前实际成本
                current_cost = self._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow(
                    veh, W, dummy_flows
                )

                # 找最优路径成本
                best_cost = current_cost
                for route_links in available_routes:
                    route_obj = W.defRoute(route_links)
                    alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                        route_obj, veh.departure_time_in_second, True
                    )
                    if alt_cost < best_cost:
                        best_cost = alt_cost

                cost_gap = current_cost - best_cost
                cost_gaps.append(cost_gap)

            if cost_gaps:
                od_stats.append({
                    "od": f"{od_pair[0]}->{od_pair[1]}",
                    "type": "charging",
                    "n_vehicles": len(cost_gaps),
                    "avg_cost_gap": round(np.mean(cost_gaps), 3),
                    "max_cost_gap": round(np.max(cost_gaps), 3),
                    "min_cost_gap": round(np.min(cost_gaps), 3),
                    "std_cost_gap": round(np.std(cost_gaps), 3) if len(cost_gaps) > 1 else 0
                })

        # 处理非充电车辆
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            cost_gaps = []
            available_routes = self.dict_od_to_routes["uncharging"][od_pair]

            for veh_id in veh_ids:
                veh = W.VEHICLES[veh_id]
                if veh.state != "end":
                    continue

                # 计算当前实际成本（非充电车辆只有时间成本）
                r, ts = veh.traveled_route()
                travel_time = ts[-1] - ts[0] if len(ts) >= 2 else 0
                current_cost = self.time_value_coefficient * travel_time

                # 找最优路径成本
                best_cost = current_cost
                for route_links in available_routes:
                    route_obj = W.defRoute(route_links)
                    alt_cost = self._EVCSChargingGameEnv__estimate_route_cost(
                        route_obj, veh.departure_time_in_second, False
                    )
                    if alt_cost < best_cost:
                        best_cost = alt_cost

                cost_gap = current_cost - best_cost
                cost_gaps.append(cost_gap)

            if cost_gaps:
                od_stats.append({
                    "od": f"{od_pair[0]}->{od_pair[1]}",
                    "type": "uncharging",
                    "n_vehicles": len(cost_gaps),
                    "avg_cost_gap": round(np.mean(cost_gaps), 3),
                    "max_cost_gap": round(np.max(cost_gaps), 3),
                    "min_cost_gap": round(np.min(cost_gaps), 3),
                    "std_cost_gap": round(np.std(cost_gaps), 3) if len(cost_gaps) > 1 else 0
                })

        return od_stats


def analyze_od_cost_gaps(network_dir: str, network_name: str, output_file: str,
                         fixed_price_ratio: float = 0.5, ue_iterations: int = 5):
    """分析各OD对的cost_gap分布"""

    env = ODAnalysisEnv(
        network_dir=network_dir,
        network_name=network_name,
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    obs, info = env.reset()

    # 设置固定价格
    actions = {}
    for agent in env.agents:
        actions[agent] = np.full(env.n_periods, fixed_price_ratio, dtype=np.float32)

    prices_matrix = env.actions_to_prices_matrix(actions)
    env.current_prices = prices_matrix
    env.price_history.append(prices_matrix.copy())

    # 运行分析
    env._run_simulation_with_od_analysis(forced_iterations=ue_iterations)

    # 生成输出
    results = {
        "network": network_name,
        "config": {
            "fixed_price_ratio": fixed_price_ratio,
            "ue_iterations": ue_iterations,
            "time_value_coefficient": env.time_value_coefficient,
            "ue_convergence_threshold": env.ue_convergence_threshold
        },
        "iterations": []
    }

    for iter_data in env.od_analysis_results:
        # 计算汇总统计
        od_data = iter_data["od_data"]
        all_avg_gaps = [od["avg_cost_gap"] for od in od_data]
        all_max_gaps = [od["max_cost_gap"] for od in od_data]

        summary = {
            "iteration": iter_data["iteration"],
            "global_avg_cost_gap": iter_data["global_stats"]["all_avg_cost_gap"],
            "od_level_max_avg_gap": max(all_avg_gaps) if all_avg_gaps else 0,
            "od_level_avg_of_avg_gap": round(np.mean(all_avg_gaps), 3) if all_avg_gaps else 0,
            "od_level_max_of_max_gap": max(all_max_gaps) if all_max_gaps else 0,
            "total_route_switches": iter_data["global_stats"]["total_route_switches"],
            "n_od_pairs": len(od_data)
        }

        # 找出高gap的OD对
        high_gap_ods = [od for od in od_data if od["avg_cost_gap"] > 1.0]
        summary["n_high_gap_ods"] = len(high_gap_ods)
        summary["high_gap_ods"] = sorted(high_gap_ods, key=lambda x: -x["avg_cost_gap"])[:10]

        # OD gap分布
        if all_avg_gaps:
            summary["od_gap_distribution"] = {
                "p50": round(np.percentile(all_avg_gaps, 50), 3),
                "p75": round(np.percentile(all_avg_gaps, 75), 3),
                "p90": round(np.percentile(all_avg_gaps, 90), 3),
                "p95": round(np.percentile(all_avg_gaps, 95), 3),
                "max": round(max(all_avg_gaps), 3)
            }

        results["iterations"].append(summary)

    # 生成分析结论
    results["analysis"] = generate_analysis(results)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 打印关键结论
    print("\n" + "=" * 60)
    print("分析结论")
    print("=" * 60)
    for key, value in results["analysis"].items():
        print(f"{key}: {value}")

    return results


def generate_analysis(results: dict) -> dict:
    """生成分析结论"""
    analysis = {}

    if not results["iterations"]:
        return {"conclusion": "无数据"}

    first = results["iterations"][0]
    last = results["iterations"][-1]

    # 第一轮的全局avg vs OD级别max_avg
    global_avg = first["global_avg_cost_gap"]
    od_max_avg = first["od_level_max_avg_gap"]
    ratio = od_max_avg / global_avg if global_avg > 0.001 else float('inf')

    analysis["first_iter_global_avg"] = global_avg
    analysis["first_iter_od_max_avg"] = od_max_avg
    analysis["first_iter_ratio"] = round(ratio, 2)

    # 判断是否存在伪均衡
    if global_avg < 1.0 and od_max_avg > 2.0:
        analysis["pseudo_equilibrium"] = True
        analysis["conclusion"] = (
            f"存在伪均衡: 全局avg={global_avg:.3f}看似收敛，"
            f"但OD级别max_avg={od_max_avg:.3f}，ratio={ratio:.1f}x"
        )
    elif global_avg < 1.0 and ratio > 3:
        analysis["pseudo_equilibrium"] = True
        analysis["conclusion"] = (
            f"存在局部不均衡: 全局avg={global_avg:.3f}，"
            f"但部分OD的avg_gap高达{od_max_avg:.3f}，被平均掩盖"
        )
    else:
        analysis["pseudo_equilibrium"] = False
        analysis["conclusion"] = (
            f"均衡状态合理: 全局avg={global_avg:.3f}，"
            f"OD级别max_avg={od_max_avg:.3f}，ratio={ratio:.1f}x"
        )

    # 收敛趋势
    analysis["convergence_trend"] = {
        "global_avg": [it["global_avg_cost_gap"] for it in results["iterations"]],
        "od_max_avg": [it["od_level_max_avg_gap"] for it in results["iterations"]]
    }

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析OD对的cost_gap分布")
    parser.add_argument("--network", type=str, default="berlin_friedrichshain",
                       help="网络名称")
    parser.add_argument("--iterations", type=int, default=5,
                       help="UE迭代次数")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")

    args = parser.parse_args()

    if args.network == "berlin_friedrichshain":
        network_dir = "data/berlin_friedrichshain"
    elif args.network == "siouxfalls":
        network_dir = "data/siouxfalls"
    else:
        network_dir = f"data/{args.network}"

    output_file = args.output or f"od_cost_gap_analysis_{args.network}.json"

    analyze_od_cost_gaps(
        network_dir=network_dir,
        network_name=args.network,
        output_file=output_file,
        ue_iterations=args.iterations
    )
