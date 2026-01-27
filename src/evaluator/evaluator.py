# src/evaluator/evaluator.py
"""
EVCSRewardEvaluator: 无状态收益评估器

设计原则：
1. 纯函数：prices + seed → EvalResult
2. 无状态：每次评估独立，不依赖历史
3. 可并行：多个 Evaluator 实例可并行运行
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import floor
from typing import Optional
from collections import defaultdict

import numpy as np

from uxsimpp_extended.uxsimpp import World, Vehicle

from .network_data import NetworkData, RouteInfo


@dataclass
class EvalResult:
    """评估结果"""
    rewards: dict[str, float]           # 各充电站收益
    flows: dict[str, np.ndarray]        # 各充电站各时段流量 {agent: (n_periods,)}
    ue_iterations: int                  # UE-DTA 迭代次数
    converged: bool                     # 是否收敛
    ue_stats: dict | None = None        # 可选的 UE 统计信息


@dataclass
class VehicleInfo:
    """车辆信息（用于模板复制）"""
    name: str
    origin: str
    destination: str
    departure_time: float
    is_charging: bool


class EVCSRewardEvaluator:
    """
    无状态收益评估器

    使用 NetworkData 进行收益评估，每次调用独立，可并行执行。
    """

    def __init__(self, network_data: NetworkData):
        """
        初始化评估器

        Args:
            network_data: 可序列化的网络数据
        """
        self.network_data = network_data
        self.settings = network_data.settings

        # 提取常用参数
        self.n_agents = network_data.n_agents
        self.n_periods = network_data.n_periods
        self.agent_names = network_data.agent_names
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agent_names)}

        # UE-DTA 参数
        self.ue_max_iterations = self.settings.get("ue_max_iterations", 50)
        self.ue_convergence_threshold = self.settings.get("ue_convergence_threshold", 0.02)
        self.ue_convergence_stable_rounds = self.settings.get("ue_convergence_stable_rounds", 3)
        self.ue_min_completed_ratio = self.settings.get("ue_min_completed_ratio", 0.95)
        self.ue_switch_gamma = self.settings.get("ue_switch_gamma", 10.0)
        self.ue_switch_alpha = self.settings.get("ue_switch_alpha", 0.1)
        self.ue_uncompleted_switch_prob = self.settings.get("ue_uncompleted_switch_prob", 0.3)

        # 仿真参数
        self.simulation_time = self.settings.get("simulation_time", 7200)
        self.deltan = self.settings.get("deltan", 5)
        self.period_duration = self.simulation_time / self.n_periods
        self.time_value_coefficient = self.settings.get("time_value_coefficient", 0.005)
        self.charging_demand_per_vehicle = self.settings.get("charging_demand_per_vehicle", 30)

        # 构建模板车辆列表（只执行一次，确保车辆集合确定性）
        self._template_vehicles: list[VehicleInfo] = self._build_template_vehicles()

        # 预构建 OD 映射（基于模板车辆）
        self._od_mappings = self._build_od_mappings_from_template()

    def evaluate(self, prices: dict[str, np.ndarray], seed: int | None = None) -> EvalResult:
        """
        单次评估：价格 -> 收益

        Args:
            prices: 各充电站各时段价格 {agent_name: (n_periods,)}
            seed: 随机种子（固定则结果确定性）

        Returns:
            EvalResult: 收益、流量、UE 统计信息
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. 使用预构建的 OD 映射
        od_mappings = self._od_mappings

        # 2. 贪心路径分配作为初始状态
        routes_specified = self._initialize_routes_greedy(od_mappings)

        # 3. UE-DTA 迭代直到收敛
        final_flows = None
        final_stats = None
        convergence_counter = 0

        for iteration in range(self.ue_max_iterations):
            # 3.1 创建新 World
            W = self._create_world()

            # 3.2 应用路径分配
            self._apply_routes(W, routes_specified)

            # 3.3 执行仿真
            W.exec_simulation()

            # 3.4 路径选择更新，获取流量和统计
            flows, stats, new_routes = self._route_choice_update(
                W, od_mappings, routes_specified, prices, iteration
            )

            # 3.5 释放 World 资源
            W.release()
            del W

            # 保存结果
            final_flows = flows
            final_stats = stats

            # 3.6 收敛检测
            if self._check_convergence(stats):
                convergence_counter += 1
                if convergence_counter >= self.ue_convergence_stable_rounds:
                    break
            else:
                convergence_counter = 0

            routes_specified = new_routes

        # 4. 计算收益
        rewards = self._calculate_rewards(prices, final_flows)

        # 5. 转换 flows 格式: np.ndarray -> dict
        flows_dict = {
            self.agent_names[i]: final_flows[i]
            for i in range(self.n_agents)
        }

        return EvalResult(
            rewards=rewards,
            flows=flows_dict,
            ue_iterations=iteration + 1,
            converged=(convergence_counter >= self.ue_convergence_stable_rounds),
            ue_stats=final_stats
        )

    # ========== 初始化方法 ==========

    def _build_template_vehicles(self) -> list[VehicleInfo]:
        """
        构建模板车辆列表

        创建临时 World，通过 adddemand 生成车辆，提取车辆信息后释放 World。
        确保车辆集合的确定性。

        Returns:
            list[VehicleInfo]: 车辆信息列表
        """
        # 创建临时 World
        W = self._create_world_from_demands()

        # 提取车辆信息
        template_vehicles = []
        for veh in W.VEHICLES.values():
            template_vehicles.append(VehicleInfo(
                name=veh.name,
                origin=veh.orig.name,
                destination=veh.dest.name,
                departure_time=veh.departure_time_in_second,
                is_charging=veh.attribute.get("charging_car", False)
            ))

        # 释放临时 World
        W.release()
        del W

        return template_vehicles

    def _create_world_from_demands(self) -> World:
        """
        从需求数据创建 World（仅用于构建模板）

        Returns:
            World: 包含所有车辆的 World 实例
        """
        network_name = self.settings.get("network_name", "network")

        W = World(
            name=f"{network_name}_template",
            deltan=self.deltan,
            tmax=self.simulation_time,
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=0,
            user_attribute={}
        )

        # 添加节点
        for node in self.network_data.nodes:
            W.addNode(node.name, node.x, node.y)

        # 添加链路
        for link in self.network_data.links:
            W.addLink(
                link.name,
                link.start_node,
                link.end_node,
                length=link.length,
                free_flow_speed=link.free_flow_speed,
                jam_density=link.jam_density,
                merge_priority=link.merge_priority,
                attribute={"charging_link": link.is_charging_link}
            )

        # 通过 adddemand 生成车辆
        for demand in self.network_data.demands:
            if demand.flow <= 0:
                continue
            W.adddemand(
                demand.origin,
                demand.destination,
                demand.start_t,
                demand.end_t,
                demand.flow,
                attribute={"charging_car": demand.is_charging}
            )

        return W

    def _build_od_mappings_from_template(self) -> dict:
        """
        从模板车辆构建 OD 映射

        Returns:
            dict: {
                "charging": {(o, d): [veh_name, ...]},
                "uncharging": {(o, d): [veh_name, ...]}
            }
        """
        charging_od = defaultdict(list)
        uncharging_od = defaultdict(list)

        for veh in self._template_vehicles:
            od = (veh.origin, veh.destination)
            if veh.is_charging:
                charging_od[od].append(veh.name)
            else:
                uncharging_od[od].append(veh.name)

        return {
            "charging": dict(charging_od),
            "uncharging": dict(uncharging_od)
        }

    # ========== 核心方法 ==========

    def _create_world(self) -> World:
        """
        从模板车辆创建 World 实例

        Returns:
            World: 新的仿真世界实例
        """
        network_name = self.settings.get("network_name", "network")

        W = World(
            name=network_name,
            deltan=self.deltan,
            tmax=self.simulation_time,
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=0,
            user_attribute={}
        )

        # 添加节点
        for node in self.network_data.nodes:
            W.addNode(node.name, node.x, node.y)

        # 添加链路
        for link in self.network_data.links:
            W.addLink(
                link.name,
                link.start_node,
                link.end_node,
                length=link.length,
                free_flow_speed=link.free_flow_speed,
                jam_density=link.jam_density,
                merge_priority=link.merge_priority,
                attribute={"charging_link": link.is_charging_link}
            )

        # 从模板复制车辆
        for veh_info in self._template_vehicles:
            Vehicle(
                W,
                veh_info.origin,
                veh_info.destination,
                veh_info.departure_time,
                predefined_route=None,
                departure_time_is_time_step=False,
                attribute={"charging_car": veh_info.is_charging},
                name=veh_info.name
            )

        return W

    def _initialize_routes_greedy(self, od_mappings: dict) -> dict[str, list[str]]:
        """
        贪心路径分配（选择最短路径）

        Args:
            od_mappings: 车辆 OD 映射

        Returns:
            dict: {veh_name: [link_name, ...]}
        """
        routes_specified = {}
        routes = self.network_data.routes

        # 充电车辆：选择充电路径中的最短路径
        for od_pair, veh_names in od_mappings["charging"].items():
            available_routes = routes["charging"].get(od_pair, [])
            if available_routes:
                best_route = available_routes[0].links  # 第一条是最短的
                for veh_name in veh_names:
                    routes_specified[veh_name] = best_route

        # 非充电车辆：选择非充电路径中的最短路径
        for od_pair, veh_names in od_mappings["uncharging"].items():
            available_routes = routes["uncharging"].get(od_pair, [])
            if available_routes:
                best_route = available_routes[0].links  # 第一条是最短的
                for veh_name in veh_names:
                    routes_specified[veh_name] = best_route

        return routes_specified

    def _apply_routes(self, W: World, routes_specified: dict[str, list[str]]):
        """
        应用路径分配到车辆

        Args:
            W: World 实例
            routes_specified: {veh_name: [link_name, ...]}
        """
        for veh_name, route_links in routes_specified.items():
            if veh_name in W.VEHICLES:
                veh = W.VEHICLES[veh_name]
                veh.assign_route(route_links)

    def _route_choice_update(self, W: World, od_mappings: dict,
                              routes_specified: dict[str, list[str]],
                              prices: dict[str, np.ndarray],
                              iteration: int) -> tuple[np.ndarray, dict, dict]:
        """
        路径选择更新（UE-DTA 核心逻辑）

        使用方案C切换机制：P_switch = min(1, gamma * gap_rel) / (1 + alpha * n)

        Args:
            W: World 实例
            od_mappings: 车辆 OD 映射
            routes_specified: 当前路径分配
            prices: 当前价格
            iteration: 当前迭代轮数

        Returns:
            tuple: (flows, stats, new_routes_specified)
        """
        new_routes_specified = {}
        routes = self.network_data.routes

        # 初始化充电流量矩阵
        charging_flows = np.zeros((self.n_agents, self.n_periods))

        # 统计信息
        charging_costs = []
        uncharging_costs = []
        charging_relative_gaps = []
        uncharging_relative_gaps = []
        charging_route_switches = 0
        uncharging_route_switches = 0
        completed_charging = 0
        completed_uncharging = 0
        total_charging = 0
        total_uncharging = 0

        # 处理充电车辆
        for od_pair, veh_names in od_mappings["charging"].items():
            available_routes = routes["charging"].get(od_pair, [])

            for veh_name in veh_names:
                total_charging += 1
                veh = W.VEHICLES.get(veh_name)
                if veh is None:
                    continue

                current_route = routes_specified.get(veh_name, [])

                # 未完成车辆：随机切换
                if veh.state != "end":
                    if np.random.random() < self.ue_uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r.links != current_route]
                        if other_routes:
                            new_routes_specified[veh_name] = other_routes[np.random.randint(len(other_routes))].links
                            charging_route_switches += 1
                        else:
                            new_routes_specified[veh_name] = current_route
                    else:
                        new_routes_specified[veh_name] = current_route
                    continue

                completed_charging += 1

                # 计算实际成本并统计流量
                current_cost = self._calculate_vehicle_cost_and_flow(veh, prices, charging_flows)
                charging_costs.append(current_cost)

                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route

                for route_info in available_routes:
                    alt_cost = self._estimate_route_cost(W, route_info, veh.departure_time_in_second, prices, True)
                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_info.links

                # 计算相对成本差
                relative_gap = (current_cost - best_cost) / current_cost if current_cost > 0 else 0.0
                charging_relative_gaps.append(relative_gap)

                # 路径切换决策
                if relative_gap > 0:
                    gap_factor = min(1.0, self.ue_switch_gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + self.ue_switch_alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes_specified[veh_name] = best_route
                        charging_route_switches += 1
                    else:
                        new_routes_specified[veh_name] = current_route
                else:
                    new_routes_specified[veh_name] = current_route

        # 处理非充电车辆
        for od_pair, veh_names in od_mappings["uncharging"].items():
            available_routes = routes["uncharging"].get(od_pair, [])

            for veh_name in veh_names:
                total_uncharging += 1
                veh = W.VEHICLES.get(veh_name)
                if veh is None:
                    continue

                current_route = routes_specified.get(veh_name, [])

                # 未完成车辆：随机切换
                if veh.state != "end":
                    if np.random.random() < self.ue_uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r.links != current_route]
                        if other_routes:
                            new_routes_specified[veh_name] = other_routes[np.random.randint(len(other_routes))].links
                            uncharging_route_switches += 1
                        else:
                            new_routes_specified[veh_name] = current_route
                    else:
                        new_routes_specified[veh_name] = current_route
                    continue

                completed_uncharging += 1

                # 计算实际成本
                traveled_route, timestamps = veh.traveled_route()
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
                current_cost = self.time_value_coefficient * travel_time
                uncharging_costs.append(current_cost)

                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route

                for route_info in available_routes:
                    alt_cost = self._estimate_route_cost(W, route_info, veh.departure_time_in_second, prices, False)
                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_info.links

                # 计算相对成本差
                relative_gap = (current_cost - best_cost) / current_cost if current_cost > 0 else 0.0
                uncharging_relative_gaps.append(relative_gap)

                # 路径切换决策
                if relative_gap > 0:
                    gap_factor = min(1.0, self.ue_switch_gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + self.ue_switch_alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes_specified[veh_name] = best_route
                        uncharging_route_switches += 1
                    else:
                        new_routes_specified[veh_name] = current_route
                else:
                    new_routes_specified[veh_name] = current_route

        # 计算统计信息
        all_relative_gaps = charging_relative_gaps + uncharging_relative_gaps
        total_vehicles = total_charging + total_uncharging
        completed_vehicles = completed_charging + completed_uncharging

        stats = {
            "all_relative_gap_global_mean": np.mean(all_relative_gaps) if all_relative_gaps else 0.0,
            "all_relative_gap_p90": np.percentile(all_relative_gaps, 90) if all_relative_gaps else 0.0,
            "all_relative_gap_p95": np.percentile(all_relative_gaps, 95) if all_relative_gaps else 0.0,
            "charging_route_switches": charging_route_switches,
            "uncharging_route_switches": uncharging_route_switches,
            "total_route_switches": charging_route_switches + uncharging_route_switches,
            "completed_charging_vehicles": completed_charging * self.deltan,
            "completed_uncharging_vehicles": completed_uncharging * self.deltan,
            "total_charging_vehicles": total_charging * self.deltan,
            "total_uncharging_vehicles": total_uncharging * self.deltan,
            "completed_total_vehicles": completed_vehicles * self.deltan,
            "total_vehicles": total_vehicles * self.deltan,
            "completed_ratio": completed_vehicles / total_vehicles if total_vehicles > 0 else 0.0
        }

        return charging_flows, stats, new_routes_specified

    def _calculate_vehicle_cost_and_flow(self, veh, prices: dict[str, np.ndarray],
                                          charging_flows: np.ndarray) -> float:
        """
        计算车辆实际成本并统计充电流量

        Args:
            veh: 车辆对象
            prices: 价格字典
            charging_flows: 充电流量矩阵（会被原地修改）

        Returns:
            float: 车辆总成本
        """
        traveled_route, timestamps = veh.traveled_route()

        # 时间成本
        travel_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        time_cost = self.time_value_coefficient * travel_time

        # 充电成本
        charging_cost = 0.0
        if veh.attribute.get("charging_car", False):
            for i, link in enumerate(traveled_route):
                if link.attribute.get("charging_link", False):
                    charging_entry_time = timestamps[i]
                    if link.name.startswith("charging_"):
                        charging_node = link.name.split("charging_")[1]
                        period = self._get_period(charging_entry_time)
                        agent_idx = self.agent_name_mapping.get(charging_node)
                        if agent_idx is not None:
                            charging_flows[agent_idx, period] += self.deltan
                            charging_price = self._get_price(prices, charging_entry_time, charging_node)
                            charging_cost = charging_price * self.charging_demand_per_vehicle
                    break

        return time_cost + charging_cost

    def _estimate_route_cost(self, W: World, route_info: RouteInfo,
                              departure_time: float, prices: dict[str, np.ndarray],
                              is_charging: bool) -> float:
        """
        估算路径成本

        Args:
            W: World 实例
            route_info: 路径信息
            departure_time: 出发时间
            prices: 价格字典
            is_charging: 是否为充电车辆

        Returns:
            float: 预期总成本
        """
        current_time = departure_time
        charging_cost = 0.0

        route_obj = W.defRoute(route_info.links)

        for link in route_obj.links:
            link_travel_time = link.actual_travel_time(current_time)

            if is_charging and link.attribute.get("charging_link", False):
                if link.name.startswith("charging_"):
                    charging_node = link.name.split("charging_")[1]
                    charging_price = self._get_price(prices, current_time, charging_node)
                    charging_cost = charging_price * self.charging_demand_per_vehicle

            current_time += link_travel_time

        total_travel_time = current_time - departure_time
        time_cost = self.time_value_coefficient * total_travel_time

        return time_cost + charging_cost

    def _check_convergence(self, stats: dict) -> bool:
        """
        检查 UE-DTA 是否收敛

        Args:
            stats: 统计信息字典

        Returns:
            bool: 是否收敛
        """
        gap_converged = stats["all_relative_gap_global_mean"] <= self.ue_convergence_threshold
        completion_ok = stats["completed_ratio"] >= self.ue_min_completed_ratio
        return gap_converged and completion_ok

    def _calculate_rewards(self, prices: dict[str, np.ndarray],
                           flows: np.ndarray) -> dict[str, float]:
        """
        计算各充电站收益

        Args:
            prices: 各充电站各时段价格
            flows: 充电流量矩阵 (n_agents, n_periods)

        Returns:
            dict: {agent_name: total_reward}
        """
        rewards = {}

        for agent_idx, agent_name in enumerate(self.agent_names):
            total_reward = 0.0
            for period in range(self.n_periods):
                price = prices[agent_name][period]
                flow = flows[agent_idx, period]
                total_reward += price * flow * self.charging_demand_per_vehicle
            rewards[agent_name] = total_reward

        return rewards

    # ========== 辅助方法 ==========

    def _get_period(self, t: float) -> int:
        """
        获取时刻 t 对应的电价时段

        Args:
            t: 时间戳（秒）

        Returns:
            int: 时段索引 [0, n_periods-1]
        """
        return max(0, min(floor(t / self.period_duration), self.n_periods - 1))

    def _get_price(self, prices: dict[str, np.ndarray], t: float, node: str) -> float:
        """
        获取指定时刻、指定节点的电价

        Args:
            prices: 价格字典
            t: 时间戳（秒）
            node: 充电节点名称

        Returns:
            float: 电价
        """
        period = self._get_period(t)
        return prices[node][period]
