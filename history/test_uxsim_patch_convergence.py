"""
测试脚本：验证 UXsim + Patch 方案的 UE-DTA 收敛性

目的：确认原版 UXsim + patch.py 方案能够在 BF 数据集上收敛，
作为 uxsimpp 修复后的对照基准。

实现方式：参考 EVCSChargingGameEnv.py 的完整实现

执行命令: python history/test_uxsim_patch_convergence.py
"""

import os
import sys
import json
import csv
import time
import numpy as np
from collections import defaultdict
from itertools import islice
from math import floor

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用原版 UXsim + patch
import uxsim
from src.env.patch import patch_uxsim
import networkx as nx

# 应用补丁
patch_uxsim()


class UEDTATester:
    """UE-DTA 收敛性测试器（基于 UXsim + Patch）"""

    def __init__(self, network_dir: str, network_name: str):
        self.network_dir = network_dir
        self.network_name = network_name

        # 加载参数
        self._load_parameters()

        # 加载网络并创建自环充电链路
        self._load_network()

        # 计算路径集合
        self._compute_routes()

        # 设置固定价格（中点价格）
        self._setup_fixed_prices()

    def _load_parameters(self):
        """加载参数"""
        settings_file = os.path.join(self.network_dir, f"{self.network_name}_settings.json")
        with open(settings_file, 'r') as f:
            settings = json.load(f)

        self.simulation_time = settings["simulation_time"]
        self.deltan = settings["deltan"]
        self.demand_multiplier = settings["demand_multiplier"]
        self.charging_car_rate = settings["charging_car_rate"]
        self.charging_link_length = settings["charging_link_length"]
        self.charging_link_free_flow_speed = settings["charging_link_free_flow_speed"]
        self.charging_periods = settings["charging_periods"]
        self.charging_nodes = settings["charging_nodes"]
        self.period_duration = self.simulation_time / self.charging_periods
        self.routes_per_od = settings["routes_per_od"]
        self.time_value_coefficient = settings["time_value_coefficient"]
        self.charging_demand_per_vehicle = settings["charging_demand_per_vehicle"]
        self.ue_convergence_threshold = settings["ue_convergence_threshold"]
        self.ue_convergence_stable_rounds = settings["ue_convergence_stable_rounds"]
        self.ue_max_iterations = settings["ue_max_iterations"]
        self.ue_min_completed_ratio = settings["ue_min_completed_ratio"]
        self.ue_switch_gamma = settings["ue_switch_gamma"]
        self.ue_switch_alpha = settings["ue_switch_alpha"]
        self.ue_uncompleted_switch_prob = settings["ue_uncompleted_switch_prob"]

        print(f"参数加载完成:")
        print(f"  simulation_time: {self.simulation_time}s")
        print(f"  deltan: {self.deltan}")
        print(f"  demand_multiplier: {self.demand_multiplier}")
        print(f"  charging_nodes: {len(self.charging_nodes)}个")

    def _load_network(self):
        """加载网络（包含自环充电链路）"""
        node_path = os.path.join(self.network_dir, f"{self.network_name}_nodes.csv")
        link_path = os.path.join(self.network_dir, f"{self.network_name}_links.csv")
        demand_path = os.path.join(self.network_dir, f"{self.network_name}_demand.csv")

        # 创建 World
        self.W = uxsim.World(
            name="test_ue",
            deltan=self.deltan,
            tmax=self.simulation_time,
            print_mode=0,
            save_mode=0,
            random_seed=42
        )

        # 加载节点
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":
                    name, x, y = r
                    self.W.addNode(name, float(x), float(y))

        # 加载链路
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":
                    link_name = r[0]
                    start_node, end_node = r[1], r[2]
                    length = float(r[3])
                    free_flow_speed = float(r[4])
                    jam_density = float(r[5])
                    merge_priority = float(r[6])

                    self.W.addLink(
                        link_name, start_node, end_node,
                        length=length,
                        free_flow_speed=free_flow_speed,
                        jam_density=jam_density,
                        merge_priority=merge_priority,
                        attribute={"charging_link": False}
                    )

        # 创建自环充电链路
        for node in self.charging_nodes.keys():
            charging_link_name = f"charging_{node}"
            self.W.addLink(
                charging_link_name,
                start_node=node,
                end_node=node,  # 自环
                length=self.charging_link_length,
                free_flow_speed=self.charging_link_free_flow_speed,
                attribute={"charging_link": True}
            )

        # 加载交通需求
        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    flow = float(r[4]) * self.demand_multiplier

                    # 充电车辆需求
                    self.W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * self.charging_car_rate,
                        attribute={"charging_car": True}
                    )

                    # 非充电车辆需求
                    self.W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * (1 - self.charging_car_rate),
                        attribute={"charging_car": False}
                    )

        print(f"网络加载完成:")
        print(f"  节点数: {len(self.W.NODES)}")
        print(f"  链路数: {len(self.W.LINKS)}")
        print(f"  Vehicle对象数: {len(self.W.VEHICLES)} (实际车辆: {len(self.W.VEHICLES) * self.deltan})")

    def _compute_routes(self):
        """计算路径集合（参考 EVCSChargingGameEnv.__compute_routes）"""
        k = self.routes_per_od

        # 获取所有OD对
        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            o = veh.orig.name
            d = veh.dest.name
            od_pairs.add((o, d))

        self.dict_od_to_routes = {
            "uncharging": {},
            "charging": {}
        }

        print(f"计算路径集合 ({len(od_pairs)}个OD对)...")

        charging_no_path = 0
        uncharging_no_path = 0
        for o, d in od_pairs:
            unc_routes = self._enumerate_k_shortest_routes(o, d, k)
            chr_routes = self._enumerate_k_shortest_charge_routes(o, d, k)
            self.dict_od_to_routes["uncharging"][(o, d)] = unc_routes
            self.dict_od_to_routes["charging"][(o, d)] = chr_routes
            if not unc_routes:
                uncharging_no_path += 1
            if not chr_routes:
                charging_no_path += 1

        print(f"路径计算完成: 无非充电路径OD对={uncharging_no_path}, 无充电路径OD对={charging_no_path}")

    def _enumerate_k_shortest_routes(self, source: str, target: str, k: int):
        """枚举非充电需求k最短路径"""
        G = nx.DiGraph()
        link_dict = {}

        for link in self.W.LINKS:  # UXsim 原版 LINKS 是列表
            if not link.attribute.get("charging_link", False):
                G.add_edge(link.start_node.name, link.end_node.name, weight=link.length/link.u)
                link_dict[(link.start_node.name, link.end_node.name)] = link.name

        try:
            k_shortest_paths = list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))
        except nx.NetworkXNoPath:
            return []

        routes = []
        for path in k_shortest_paths:
            route = [link_dict[(path[i], path[i + 1])] for i in range(len(path) - 1)]
            routes.append(route)

        return routes

    def _enumerate_k_shortest_charge_routes(self, source: str, target: str, k: int):
        """枚举充电需求k最短路径（基于多状态图）"""
        G = nx.DiGraph()
        link_dict = {}

        for link in self.W.LINKS:  # UXsim 原版 LINKS 是列表
            start_node = link.start_node.name
            end_node = link.end_node.name
            weight = link.length / link.u

            if link.attribute.get("charging_link", False):
                # 自环充电链路：未充电状态 -> 已充电状态
                node = start_node
                G.add_edge(f"uncharged_{node}", f"charged_{node}", weight=weight)
                link_dict[(f"uncharged_{node}", f"charged_{node}")] = link.name
            else:
                # 普通链路：状态保持
                G.add_edge(f"uncharged_{start_node}", f"uncharged_{end_node}", weight=weight)
                G.add_edge(f"charged_{start_node}", f"charged_{end_node}", weight=weight)
                link_dict[(f"uncharged_{start_node}", f"uncharged_{end_node}")] = link.name
                link_dict[(f"charged_{start_node}", f"charged_{end_node}")] = link.name

        source_state = f"uncharged_{source}"
        target_state = f"charged_{target}"

        try:
            k_shortest_paths = list(islice(nx.shortest_simple_paths(G, source_state, target_state, weight='weight'), k))
        except nx.NetworkXNoPath:
            return []

        routes = []
        for path in k_shortest_paths:
            route = [link_dict[(path[i], path[i + 1])] for i in range(len(path) - 1)]
            routes.append(route)

        return routes

    def _setup_fixed_prices(self):
        """设置固定价格（中点价格）"""
        self.agents = list(self.charging_nodes.keys())
        self.n_agents = len(self.agents)
        self.n_periods = self.charging_periods
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        # 价格矩阵：使用区间中点
        self.prices = np.zeros((self.n_agents, self.n_periods))
        for agent_idx, agent in enumerate(self.agents):
            min_price, max_price = self.charging_nodes[agent]
            mid_price = (min_price + max_price) / 2
            self.prices[agent_idx, :] = mid_price

        print(f"价格设置: 所有充电站所有时段使用中点价格 = {(0.5 + 2.0) / 2}")

    def _get_period(self, t: float) -> int:
        """获取时段索引"""
        return max(0, min(floor(t / self.period_duration), self.n_periods - 1))

    def _get_price(self, t: float, node: str) -> float:
        """获取价格"""
        period = self._get_period(t)
        agent_idx = self.agent_name_mapping[node]
        return self.prices[agent_idx, period]

    def _create_simulation_world(self) -> uxsim.World:
        """创建仿真世界实例"""
        W = uxsim.World(
            name="sim",
            deltan=self.deltan,
            tmax=self.simulation_time,
            print_mode=0,
            save_mode=0,
            random_seed=42
        )

        # 复制节点
        for node in self.W.NODES:  # UXsim 原版 NODES 是列表
            W.addNode(node.name, node.x, node.y)

        # 复制链路
        for link in self.W.LINKS:  # UXsim 原版 LINKS 是列表
            W.addLink(
                link.name,
                link.start_node.name,
                link.end_node.name,
                length=link.length,
                free_flow_speed=link.u,
                jam_density=link.kappa,
                merge_priority=link.merge_priority,
                attribute=link.attribute.copy()
            )

        # 复制交通需求（延迟分配路径）
        vehicle_count = 0
        for veh in self.W.VEHICLES.values():
            unique_name = f"{veh.orig.name}-{veh.dest.name}-{veh.departure_time}-{vehicle_count}"
            uxsim.Vehicle(
                W, veh.orig.name, veh.dest.name,
                veh.departure_time,
                predefined_route=None,
                departure_time_is_time_step=1,
                attribute=veh.attribute.copy(),
                name=unique_name
            )
            vehicle_count += 1

        return W

    def _initialize_routes(self, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid):
        """初始化路径分配（贪心策略）"""
        routes_specified = {}

        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"].get(od_pair, [])
            if available_routes:
                best_route = available_routes[0]
                for veh_id in veh_ids:
                    routes_specified[veh_id] = best_route

        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"].get(od_pair, [])
            if available_routes:
                best_route = available_routes[0]
                for veh_id in veh_ids:
                    routes_specified[veh_id] = best_route

        return routes_specified

    def _apply_routes_to_vehicles(self, W, routes_specified):
        """为车辆分配路径"""
        for veh_id, route_links in routes_specified.items():
            if veh_id in W.VEHICLES:
                veh = W.VEHICLES[veh_id]
                veh.assign_route(route_links)

    def _calculate_actual_vehicle_cost(self, veh, charging_flows: np.ndarray) -> float:
        """计算车辆实际总成本并统计充电流量"""
        route, timestamps = veh.traveled_route(include_departure_time=True)

        travel_time = timestamps[-1] - timestamps[0]
        time_cost = self.time_value_coefficient * travel_time
        charging_cost = 0.0

        if veh.attribute.get("charging_car", False):
            for i, link in enumerate(route):
                if link is not None and link.attribute.get("charging_link", False):
                    charging_entry_time = timestamps[i]
                    if link.name.startswith("charging_"):
                        charging_node = link.name.split("charging_")[1]

                        charging_period = self._get_period(charging_entry_time)
                        agent_idx = self.agent_name_mapping.get(charging_node)

                        if agent_idx is not None:
                            charging_flows[agent_idx, charging_period] += self.deltan
                            charging_price = self._get_price(charging_entry_time, charging_node)
                            charging_cost = charging_price * self.charging_demand_per_vehicle
                        break

        return time_cost + charging_cost

    def _estimate_route_cost(self, W, route_links, departure_time: float, is_charging_vehicle: bool) -> float:
        """估算路径成本"""
        current_time = departure_time
        charging_cost = 0.0

        for link_name in route_links:
            link = W.get_link(link_name)
            if link is None:
                continue

            link_travel_time = link.actual_travel_time(current_time)

            if is_charging_vehicle and link.attribute.get("charging_link", False):
                if link.name.startswith("charging_"):
                    charging_node = link.name.split("charging_")[1]
                    if charging_node in self.agent_name_mapping:
                        charging_price = self._get_price(current_time, charging_node)
                        charging_cost = charging_price * self.charging_demand_per_vehicle

            current_time += link_travel_time

        total_travel_time = current_time - departure_time
        time_cost = self.time_value_coefficient * total_travel_time

        return time_cost + charging_cost

    def _route_choice_update(self, W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                              current_routes_specified, iteration):
        """执行路径选择与切换逻辑"""
        new_routes_specified = {}
        charging_flows = np.zeros((self.n_agents, self.n_periods))

        all_relative_gaps = []
        total_route_switches = 0
        completed_vehicles = 0
        total_vehicles = 0

        # 处理充电车辆
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"].get(od_pair, [])

            for veh_id in veh_ids:
                if veh_id not in W.VEHICLES:
                    continue
                veh = W.VEHICLES[veh_id]
                total_vehicles += 1

                current_route = current_routes_specified.get(veh_id, available_routes[0] if available_routes else [])

                if veh.state != "end":
                    # 未完成车辆：随机切换
                    if np.random.random() < self.ue_uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r != current_route]
                        if other_routes:
                            new_routes_specified[veh_id] = other_routes[np.random.randint(len(other_routes))]
                            total_route_switches += 1
                        else:
                            new_routes_specified[veh_id] = current_route
                    else:
                        new_routes_specified[veh_id] = current_route
                    continue

                completed_vehicles += 1
                current_cost = self._calculate_actual_vehicle_cost(veh, charging_flows)

                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route

                for route_links in available_routes:
                    alt_cost = self._estimate_route_cost(W, route_links, veh.departure_time * self.W.DELTAT, True)
                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_links

                # 计算相对成本差
                relative_gap = (current_cost - best_cost) / current_cost if current_cost > 0 else 0.0
                all_relative_gaps.append(relative_gap)

                # 路径切换决策
                if relative_gap > 0:
                    gap_factor = min(1.0, self.ue_switch_gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + self.ue_switch_alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes_specified[veh_id] = best_route
                        total_route_switches += 1
                    else:
                        new_routes_specified[veh_id] = current_route
                else:
                    new_routes_specified[veh_id] = current_route

        # 处理非充电车辆
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"].get(od_pair, [])

            for veh_id in veh_ids:
                if veh_id not in W.VEHICLES:
                    continue
                veh = W.VEHICLES[veh_id]
                total_vehicles += 1

                current_route = current_routes_specified.get(veh_id, available_routes[0] if available_routes else [])

                if veh.state != "end":
                    if np.random.random() < self.ue_uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r != current_route]
                        if other_routes:
                            new_routes_specified[veh_id] = other_routes[np.random.randint(len(other_routes))]
                            total_route_switches += 1
                        else:
                            new_routes_specified[veh_id] = current_route
                    else:
                        new_routes_specified[veh_id] = current_route
                    continue

                completed_vehicles += 1

                route, timestamps = veh.traveled_route(include_departure_time=True)
                travel_time = timestamps[-1] - timestamps[0]
                current_cost = self.time_value_coefficient * travel_time

                best_cost = current_cost
                best_route = current_route

                for route_links in available_routes:
                    alt_cost = self._estimate_route_cost(W, route_links, veh.departure_time * self.W.DELTAT, False)
                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_links

                relative_gap = (current_cost - best_cost) / current_cost if current_cost > 0 else 0.0
                all_relative_gaps.append(relative_gap)

                if relative_gap > 0:
                    gap_factor = min(1.0, self.ue_switch_gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + self.ue_switch_alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes_specified[veh_id] = best_route
                        total_route_switches += 1
                    else:
                        new_routes_specified[veh_id] = current_route
                else:
                    new_routes_specified[veh_id] = current_route

        # 计算统计信息
        stats = {
            "global_mean": np.mean(all_relative_gaps) if all_relative_gaps else 0.0,
            "p90": np.percentile(all_relative_gaps, 90) if all_relative_gaps else 0.0,
            "p95": np.percentile(all_relative_gaps, 95) if all_relative_gaps else 0.0,
            "route_switches": total_route_switches,
            "completed_vehicles": completed_vehicles * self.deltan,
            "total_vehicles": total_vehicles * self.deltan,
            "completed_ratio": completed_vehicles / total_vehicles if total_vehicles > 0 else 0.0
        }

        return stats, new_routes_specified, charging_flows

    def run_ue_dta(self):
        """运行 UE-DTA 迭代"""
        print(f"\n{'='*60}")
        print("开始 UE-DTA 迭代")
        print(f"{'='*60}")

        # 获取 OD 映射
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)

        W_template = self._create_simulation_world()
        for key, veh in W_template.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)
        del W_template

        # 初始化路径
        current_routes_specified = self._initialize_routes(dict_od_to_charging_vehid, dict_od_to_uncharging_vehid)

        # 检查是否所有车辆都有路径
        total_vehicles = sum(len(v) for v in dict_od_to_charging_vehid.values()) + \
                         sum(len(v) for v in dict_od_to_uncharging_vehid.values())
        missing_routes = total_vehicles - len(current_routes_specified)
        print(f"初始化路径分配: {len(current_routes_specified)}条 (总车辆: {total_vehicles}, 缺失: {missing_routes})")

        if missing_routes > 0:
            print(f"警告: {missing_routes} 辆车没有分配到路径!")

        # 迭代求解
        ue_convergence_counter = 0

        for iteration in range(self.ue_max_iterations):
            start_time = time.time()

            # 创建仿真实例
            W = self._create_simulation_world()

            # 应用路径
            self._apply_routes_to_vehicles(W, current_routes_specified)

            # 执行仿真
            W.exec_simulation()

            elapsed = time.time() - start_time

            # 路径选择更新
            stats, new_routes_specified, charging_flows = self._route_choice_update(
                W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                current_routes_specified, iteration
            )

            del W

            # 打印进度
            print(f"迭代 {iteration+1:3d}: "
                  f"完成率={stats['completed_ratio']*100:.1f}% | "
                  f"Gap: GM={stats['global_mean']*100:.2f}% P90={stats['p90']*100:.2f}% P95={stats['p95']*100:.2f}% | "
                  f"切换={stats['route_switches']} | "
                  f"耗时={elapsed:.1f}s")

            current_routes_specified = new_routes_specified

            # 收敛判断
            gap_converged = stats['global_mean'] <= self.ue_convergence_threshold
            completion_ok = stats['completed_ratio'] >= self.ue_min_completed_ratio

            if gap_converged and completion_ok:
                ue_convergence_counter += 1
                if ue_convergence_counter >= self.ue_convergence_stable_rounds:
                    print(f"\n收敛! 连续{self.ue_convergence_stable_rounds}轮达到阈值")
                    return True, iteration + 1, stats
            else:
                ue_convergence_counter = 0

        print(f"\n未收敛，达到最大迭代次数 {self.ue_max_iterations}")
        return False, self.ue_max_iterations, stats

    def test_traveltime_actual(self):
        """测试 traveltime_actual 的更新情况"""
        print(f"\n{'='*60}")
        print("检查 traveltime_actual 更新情况")
        print(f"{'='*60}")

        # 创建并运行一次仿真
        W = self._create_simulation_world()

        # 初始化路径
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        for key, veh in W.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)

        routes_specified = self._initialize_routes(dict_od_to_charging_vehid, dict_od_to_uncharging_vehid)
        self._apply_routes_to_vehicles(W, routes_specified)

        W.exec_simulation()

        # 检查前 10 条链路
        for i, link in enumerate(W.LINKS[:10]):  # UXsim 原版 LINKS 是列表
            tt_actual = link.traveltime_actual
            unique_values = len(set(tt_actual))
            free_flow_tt = link.length / link.u

            non_freeflow = [t for t in tt_actual if abs(t - free_flow_tt) > 0.1]

            print(f"链路 {link.name}:")
            print(f"  length: {link.length:.1f}m, free_flow_tt: {free_flow_tt:.1f}s")
            print(f"  traveltime_actual unique values: {unique_values}")
            print(f"  非自由流值数量: {len(non_freeflow)}")
            if non_freeflow:
                print(f"  非自由流值样本: {non_freeflow[:5]}")


def main():
    print("\n" + "#" * 60)
    print("# 测试 UXsim + Patch 方案的 UE-DTA 收敛性")
    print("# (完整实现：参考 EVCSChargingGameEnv.py)")
    print("#" * 60)

    network_dir = "data/berlin_friedrichshain"
    network_name = "berlin_friedrichshain"

    try:
        tester = UEDTATester(network_dir, network_name)
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 先测试 traveltime_actual 更新
    tester.test_traveltime_actual()

    # 运行 UE-DTA
    converged, iterations, final_stats = tester.run_ue_dta()

    print(f"\n{'='*60}")
    print("测试结果")
    print(f"{'='*60}")
    print(f"收敛状态: {'收敛' if converged else '未收敛'}")
    print(f"迭代次数: {iterations}")
    print(f"最终 Gap: GM={final_stats['global_mean']*100:.2f}% P90={final_stats['p90']*100:.2f}% P95={final_stats['p95']*100:.2f}%")
    print(f"完成率: {final_stats['completed_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
