"""
UE-DTA 性能对比测试脚本

对比 UXsim+Patch 和 uxsimpp_extended 在 UE-DTA 迭代各环节的耗时。

使用方法:
    python debug_ue_dta_performance.py --backend uxsimpp   # 测试 uxsimpp_extended
    python debug_ue_dta_performance.py --backend uxsim     # 测试 UXsim+Patch
    python debug_ue_dta_performance.py --both              # 两者都测试并对比
"""

import time
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from itertools import islice
from math import floor
import networkx as nx

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# ============================================================
# 配置参数
# ============================================================
NETWORK_DIR = os.path.join(project_root, "data", "siouxfalls")
NETWORK_NAME = "siouxfalls"


# ============================================================
# 性能计时器
# ============================================================
class PerformanceTimer:
    """性能计时器，记录各环节耗时"""

    def __init__(self, name=""):
        self.name = name
        self.records = defaultdict(list)
        self._start_times = {}

    def start(self, label):
        """开始计时"""
        self._start_times[label] = time.perf_counter()

    def stop(self, label):
        """停止计时并记录"""
        if label in self._start_times:
            duration = time.perf_counter() - self._start_times[label]
            self.records[label].append(duration)
            del self._start_times[label]
            return duration
        return 0

    def record(self, label, duration):
        """直接记录耗时"""
        self.records[label].append(duration)

    def summary(self):
        """输出统计摘要"""
        print(f"\n{'=' * 70}")
        print(f"性能统计摘要: {self.name}")
        print(f"{'=' * 70}")
        print(f"{'环节':<40} {'次数':>6} {'总耗时(ms)':>12} {'平均(ms)':>10}")
        print(f"{'-' * 70}")

        total_time = 0
        for label in self.records:
            times = self.records[label]
            avg = np.mean(times) * 1000
            total = np.sum(times) * 1000
            count = len(times)
            total_time += total
            print(f"{label:<40} {count:>6} {total:>12.2f} {avg:>10.2f}")

        print(f"{'-' * 70}")
        print(f"{'总计':<40} {'':<6} {total_time:>12.2f}")
        print(f"{'=' * 70}")

        return dict(self.records)


# ============================================================
# UE-DTA 测试类（抽象基类）
# ============================================================
class UEDTAPerformanceTest:
    """UE-DTA 性能测试基类"""

    def __init__(self, backend_name):
        self.backend_name = backend_name
        self.timer = PerformanceTimer(backend_name)
        self.settings = None
        self.W = None  # 模板 World
        self.charging_nodes = None
        self.dict_od_to_routes = None

    def load_settings(self):
        """加载配置文件"""
        settings_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_settings.json")
        with open(settings_path, "r") as f:
            self.settings = json.load(f)
        return self.settings

    def create_world(self):
        """创建仿真世界 - 子类实现"""
        raise NotImplementedError

    def load_network(self, W):
        """加载路网 - 子类实现"""
        raise NotImplementedError

    def create_simulation_world(self):
        """创建仿真实例 - 子类实现"""
        raise NotImplementedError

    def apply_routes_to_vehicles(self, W, routes_specified):
        """应用路径到车辆 - 子类实现"""
        raise NotImplementedError

    def exec_simulation(self, W):
        """执行仿真 - 子类实现"""
        raise NotImplementedError

    def route_choice_update(self, W, routes_specified, iteration):
        """路径选择更新 - 子类实现"""
        raise NotImplementedError

    def run_test(self):
        """运行完整测试"""
        print(f"\n{'#' * 70}")
        print(f"# 测试后端: {self.backend_name}")
        print(f"{'#' * 70}")

        # 1. 加载配置
        self.timer.start("1. 加载配置")
        self.load_settings()
        self.timer.stop("1. 加载配置")

        # 2. 创建模板世界
        self.timer.start("2. 创建模板世界")
        self.W = self.create_world()
        self.timer.stop("2. 创建模板世界")

        # 3. 加载路网
        self.timer.start("3. 加载路网")
        self.load_network(self.W)
        self.timer.stop("3. 加载路网")

        # 4. 计算路径集合
        self.timer.start("4. 计算路径集合")
        self.compute_routes()
        self.timer.stop("4. 计算路径集合")

        # 5. 构建 OD-车辆映射
        self.timer.start("5. 构建OD-车辆映射")
        dict_od_to_charging_vehid, dict_od_to_uncharging_vehid = self.build_od_vehicle_mapping()
        self.timer.stop("5. 构建OD-车辆映射")

        # 6. 初始化路径分配
        self.timer.start("6. 初始化路径分配")
        current_routes = self.initialize_routes(dict_od_to_charging_vehid, dict_od_to_uncharging_vehid)
        self.timer.stop("6. 初始化路径分配")

        # 7. UE-DTA 迭代
        print(f"\n开始 UE-DTA 迭代...")
        ue_max_iterations = self.settings.get("ue_max_iterations", 100)
        ue_convergence_threshold = self.settings.get("ue_convergence_threshold", 0.01)
        ue_min_completed_ratio = self.settings.get("ue_min_completed_ratio", 0.95)
        ue_convergence_stable_rounds = self.settings.get("ue_convergence_stable_rounds", 3)

        convergence_counter = 0

        for iteration in range(ue_max_iterations):
            iter_start = time.perf_counter()

            # 7.1 创建仿真世界
            self.timer.start("7.1 创建仿真世界")
            W_sim = self.create_simulation_world()
            self.timer.stop("7.1 创建仿真世界")

            # 7.2 应用路径到车辆
            self.timer.start("7.2 应用路径到车辆")
            self.apply_routes_to_vehicles(W_sim, current_routes)
            self.timer.stop("7.2 应用路径到车辆")

            # 7.3 执行仿真
            self.timer.start("7.3 执行仿真")
            self.exec_simulation(W_sim)
            self.timer.stop("7.3 执行仿真")

            # 7.4 路径选择更新
            self.timer.start("7.4 路径选择更新")
            stats, new_routes = self.route_choice_update(
                W_sim,
                dict_od_to_charging_vehid,
                dict_od_to_uncharging_vehid,
                current_routes,
                iteration
            )
            self.timer.stop("7.4 路径选择更新")

            current_routes = new_routes

            # 7.5 收敛判断
            self.timer.start("7.5 收敛判断")
            gm = stats['all_relative_gap_global_mean']
            completed_ratio = stats['completed_ratio']
            gap_converged = gm <= ue_convergence_threshold
            completion_ok = completed_ratio >= ue_min_completed_ratio

            if gap_converged and completion_ok:
                convergence_counter += 1
            else:
                convergence_counter = 0

            converged = convergence_counter >= ue_convergence_stable_rounds
            self.timer.stop("7.5 收敛判断")

            iter_time = time.perf_counter() - iter_start

            print(f"  迭代 {iteration+1:3d} | 完成率: {completed_ratio*100:5.1f}% | "
                  f"GM: {gm*100:5.2f}% | 切换: {stats['total_switches']:4d} | "
                  f"耗时: {iter_time*1000:.0f}ms")

            if converged:
                print(f"  ✓ 收敛于第 {iteration+1} 轮")
                break

        # 输出统计摘要
        return self.timer.summary()


# ============================================================
# UXsim + Patch 实现
# ============================================================
class UXsimPatchTest(UEDTAPerformanceTest):
    """使用 UXsim + Patch 的测试"""

    def __init__(self):
        super().__init__("UXsim + Patch")
        # 导入并应用 patch
        from uxsim import World as UXsimWorld
        from src.env.patch import patch_uxsim
        patch_uxsim()
        self.UXsimWorld = UXsimWorld
        from uxsim import Vehicle as UXsimVehicle
        self.UXsimVehicle = UXsimVehicle

    def create_world(self):
        return self.UXsimWorld(
            name=self.settings["network_name"],
            deltan=self.settings["deltan"],
            tmax=self.settings["simulation_time"],
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=42
        )

    def load_network(self, W):
        # 加载节点
        node_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_nodes.csv")
        import csv
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":
                    name, x, y = r
                    W.addNode(name, float(x), float(y))

        # 加载链路
        link_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_links.csv")
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":
                    link_name, start_node, end_node = r[0], r[1], r[2]
                    length = float(r[3])
                    free_flow_speed = float(r[4])
                    jam_density = float(r[5])
                    merge_priority = float(r[6])
                    link = W.addLink(
                        link_name, start_node, end_node,
                        length=length,
                        free_flow_speed=free_flow_speed,
                        jam_density=jam_density,
                        merge_priority=merge_priority
                    )
                    link.attribute = {"charging_link": False}

        # 创建充电链路
        self.charging_nodes = self.settings["charging_nodes"]
        for node in self.charging_nodes.keys():
            charging_link_name = f"charging_{node}"
            link = W.addLink(
                charging_link_name,
                start_node=node,
                end_node=node,
                length=self.settings["charging_link_length"],
                free_flow_speed=self.settings["charging_link_free_flow_speed"]
            )
            link.attribute = {"charging_link": True}

        # 加载需求
        demand_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_demand.csv")
        demand_multiplier = self.settings.get("demand_multiplier", 1.0)
        charging_car_rate = self.settings["charging_car_rate"]

        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    flow = float(r[4]) * demand_multiplier

                    # 充电车辆
                    W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * charging_car_rate,
                        attribute={"charging_car": True}
                    )
                    # 非充电车辆
                    W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * (1 - charging_car_rate),
                        attribute={"charging_car": False}
                    )

    def compute_routes(self):
        """计算路径集合"""
        k = self.settings["routes_per_od"]

        # 获取所有 OD 对
        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            o = veh.orig.name
            d = veh.dest.name
            od_pairs.add((o, d))

        self.dict_od_to_routes = {
            "uncharging": {},
            "charging": {}
        }

        for o, d in od_pairs:
            self.dict_od_to_routes["uncharging"][(o, d)] = self._enumerate_k_shortest_routes(o, d, k)
            self.dict_od_to_routes["charging"][(o, d)] = self._enumerate_k_shortest_charge_routes(o, d, k)

    def _enumerate_k_shortest_routes(self, source, target, k):
        """枚举非充电路径"""
        G = nx.DiGraph()
        link_dict = {}
        for link in self.W.LINKS:
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

    def _enumerate_k_shortest_charge_routes(self, source, target, k):
        """枚举充电路径（多状态图）"""
        G = nx.DiGraph()
        link_dict = {}

        for link in self.W.LINKS:
            start_node = link.start_node.name
            end_node = link.end_node.name
            weight = link.length / link.u

            if link.attribute.get("charging_link", False):
                node = start_node
                G.add_edge(f"uncharged_{node}", f"charged_{node}", weight=weight)
                link_dict[(f"uncharged_{node}", f"charged_{node}")] = link.name
            else:
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

    def build_od_vehicle_mapping(self):
        """构建 OD-车辆映射"""
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)

        for key, veh in self.W.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)

        return dict_od_to_charging_vehid, dict_od_to_uncharging_vehid

    def initialize_routes(self, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid):
        """初始化路径分配（贪心）"""
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

    def create_simulation_world(self):
        """创建仿真实例"""
        W = self.UXsimWorld(
            name=self.settings["network_name"] + "_sim",
            deltan=self.settings["deltan"],
            tmax=self.settings["simulation_time"],
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=42
        )

        # 复制节点
        for node in self.W.NODES:
            W.addNode(node.name, node.x, node.y)

        # 复制链路
        for link in self.W.LINKS:
            new_link = W.addLink(
                link.name,
                link.start_node.name,
                link.end_node.name,
                length=link.length,
                free_flow_speed=link.u,
                jam_density=link.kappa,
                merge_priority=link.merge_priority
            )
            new_link.attribute = link.attribute.copy()

        # 复制车辆
        for veh in self.W.VEHICLES.values():
            new_veh = self.UXsimVehicle(
                W, veh.orig.name, veh.dest.name,
                veh.departure_time,
                departure_time_is_time_step=True,
                attribute=veh.attribute.copy()
            )

        return W

    def apply_routes_to_vehicles(self, W, routes_specified):
        """应用路径到车辆"""
        for veh_id, route_links in routes_specified.items():
            if veh_id in W.VEHICLES:
                veh = W.VEHICLES[veh_id]
                veh.assign_route(route_links)

    def exec_simulation(self, W):
        """执行仿真"""
        W.exec_simulation()

    def route_choice_update(self, W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                            current_routes, iteration):
        """路径选择更新"""
        new_routes = {}
        relative_gaps = []
        total_switches = 0
        completed_count = 0
        total_count = 0

        gamma = self.settings.get("ue_switch_gamma", 5)
        alpha = self.settings.get("ue_switch_alpha", 0.08)
        uncompleted_switch_prob = self.settings.get("ue_uncompleted_switch_prob", 0.3)
        time_value_coefficient = self.settings.get("time_value_coefficient", 0.005)
        charging_demand = self.settings.get("charging_demand_per_vehicle", 50)
        period_duration = self.settings["simulation_time"] / self.settings["charging_periods"]

        def get_period(t):
            return max(0, min(floor(t / period_duration), self.settings["charging_periods"] - 1))

        # 计算固定价格（与 EVCSChargingGameEnv 一致）
        # 使用归一化价格 0.5，对应实际价格 = price_min + 0.5 * (price_max - price_min)
        charging_nodes = self.settings.get("charging_nodes", {})
        if charging_nodes:
            first_node_range = list(charging_nodes.values())[0]
            price_min, price_max = first_node_range[0], first_node_range[1]
            price_ratio = 0.5  # 归一化价格
            fixed_price = price_min + price_ratio * (price_max - price_min)
        else:
            fixed_price = 0.5  # 默认值

        # 处理充电车辆
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"].get(od_pair, [])

            for veh_id in veh_ids:
                total_count += 1
                veh = W.VEHICLES.get(veh_id)
                if veh is None:
                    new_routes[veh_id] = current_routes.get(veh_id, [])
                    continue

                current_route = current_routes.get(veh_id, [])

                if veh.state != "end":
                    # 未完成车辆随机切换
                    if np.random.random() < uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r != current_route]
                        if other_routes:
                            new_routes[veh_id] = other_routes[np.random.randint(len(other_routes))]
                            total_switches += 1
                        else:
                            new_routes[veh_id] = current_route
                    else:
                        new_routes[veh_id] = current_route
                    continue

                completed_count += 1

                # 计算当前路径成本
                route, timestamps = veh.traveled_route()
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
                current_cost = time_value_coefficient * travel_time + fixed_price * charging_demand

                # 寻找最优路径
                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_travel_time = route_obj.actual_travel_time(veh.departure_time_in_second)
                        alt_cost = time_value_coefficient * alt_travel_time + fixed_price * charging_demand
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links

                # 计算相对 gap
                if current_cost > 0:
                    relative_gap = (current_cost - best_cost) / current_cost
                else:
                    relative_gap = 0
                relative_gaps.append(relative_gap)

                # 切换决策
                if relative_gap > 0:
                    gap_factor = min(1.0, gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes[veh_id] = best_route
                        total_switches += 1
                    else:
                        new_routes[veh_id] = current_route
                else:
                    new_routes[veh_id] = current_route

        # 处理非充电车辆（简化处理）
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"].get(od_pair, [])

            for veh_id in veh_ids:
                total_count += 1
                veh = W.VEHICLES.get(veh_id)
                if veh is None:
                    new_routes[veh_id] = current_routes.get(veh_id, [])
                    continue

                current_route = current_routes.get(veh_id, [])

                if veh.state != "end":
                    if np.random.random() < uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r != current_route]
                        if other_routes:
                            new_routes[veh_id] = other_routes[np.random.randint(len(other_routes))]
                            total_switches += 1
                        else:
                            new_routes[veh_id] = current_route
                    else:
                        new_routes[veh_id] = current_route
                    continue

                completed_count += 1

                # 计算成本
                route, timestamps = veh.traveled_route()
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
                current_cost = time_value_coefficient * travel_time

                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_travel_time = route_obj.actual_travel_time(veh.departure_time_in_second)
                        alt_cost = time_value_coefficient * alt_travel_time
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links

                if current_cost > 0:
                    relative_gap = (current_cost - best_cost) / current_cost
                else:
                    relative_gap = 0
                relative_gaps.append(relative_gap)

                if relative_gap > 0:
                    gap_factor = min(1.0, gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes[veh_id] = best_route
                        total_switches += 1
                    else:
                        new_routes[veh_id] = current_route
                else:
                    new_routes[veh_id] = current_route

        stats = {
            'all_relative_gap_global_mean': np.mean(relative_gaps) if relative_gaps else 0,
            'completed_ratio': completed_count / total_count if total_count > 0 else 0,
            'total_switches': total_switches
        }

        return stats, new_routes


# ============================================================
# uxsimpp_extended 实现
# ============================================================
class UXsimppTest(UEDTAPerformanceTest):
    """使用 uxsimpp_extended 的测试"""

    def __init__(self):
        super().__init__("uxsimpp_extended")
        from uxsimpp_extended.uxsimpp import (
            World, Vehicle, Link, Node, Route, newWorld
        )
        self.World = World
        self.Vehicle = Vehicle
        self.Route = Route

    def create_world(self):
        return self.World(
            name=self.settings["network_name"],
            deltan=self.settings["deltan"],
            tmax=self.settings["simulation_time"],
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=42
        )

    def load_network(self, W):
        # 加载节点
        node_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_nodes.csv")
        import csv
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":
                    name, x, y = r
                    W.addNode(name, float(x), float(y))

        # 加载链路
        link_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_links.csv")
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":
                    link_name, start_node, end_node = r[0], r[1], r[2]
                    length = float(r[3])
                    free_flow_speed = float(r[4])
                    jam_density = float(r[5])
                    merge_priority = float(r[6])
                    W.addLink(
                        link_name, start_node, end_node,
                        length=length,
                        free_flow_speed=free_flow_speed,
                        jam_density=jam_density,
                        merge_priority=merge_priority,
                        attribute={"charging_link": False}
                    )

        # 创建充电链路
        self.charging_nodes = self.settings["charging_nodes"]
        for node in self.charging_nodes.keys():
            charging_link_name = f"charging_{node}"
            W.addLink(
                charging_link_name,
                start_node=node,
                end_node=node,
                length=self.settings["charging_link_length"],
                free_flow_speed=self.settings["charging_link_free_flow_speed"],
                attribute={"charging_link": True}
            )

        # 加载需求
        demand_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_demand.csv")
        demand_multiplier = self.settings.get("demand_multiplier", 1.0)
        charging_car_rate = self.settings["charging_car_rate"]

        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    flow = float(r[4]) * demand_multiplier

                    # 充电车辆
                    W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * charging_car_rate,
                        attribute={"charging_car": True}
                    )
                    # 非充电车辆
                    W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * (1 - charging_car_rate),
                        attribute={"charging_car": False}
                    )

    def compute_routes(self):
        """计算路径集合"""
        k = self.settings["routes_per_od"]

        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            o = veh.orig.name
            d = veh.dest.name
            od_pairs.add((o, d))

        self.dict_od_to_routes = {
            "uncharging": {},
            "charging": {}
        }

        for o, d in od_pairs:
            self.dict_od_to_routes["uncharging"][(o, d)] = self._enumerate_k_shortest_routes(o, d, k)
            self.dict_od_to_routes["charging"][(o, d)] = self._enumerate_k_shortest_charge_routes(o, d, k)

    def _enumerate_k_shortest_routes(self, source, target, k):
        G = nx.DiGraph()
        link_dict = {}
        for link in self.W.LINKS:
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

    def _enumerate_k_shortest_charge_routes(self, source, target, k):
        G = nx.DiGraph()
        link_dict = {}

        for link in self.W.LINKS:
            start_node = link.start_node.name
            end_node = link.end_node.name
            weight = link.length / link.u

            if link.attribute.get("charging_link", False):
                node = start_node
                G.add_edge(f"uncharged_{node}", f"charged_{node}", weight=weight)
                link_dict[(f"uncharged_{node}", f"charged_{node}")] = link.name
            else:
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

    def build_od_vehicle_mapping(self):
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)

        for veh in self.W.VEHICLES.values():
            key = veh.name
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)

        return dict_od_to_charging_vehid, dict_od_to_uncharging_vehid

    def initialize_routes(self, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid):
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

    def create_simulation_world(self):
        W = self.World(
            name=self.settings["network_name"] + "_sim",
            deltan=self.settings["deltan"],
            tmax=self.settings["simulation_time"],
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=42
        )

        # 复制节点
        for node in self.W.NODES:
            W.addNode(node.name, node.x, node.y)

        # 复制链路
        for link in self.W.LINKS:
            W.addLink(
                link.name,
                link.start_node.name,
                link.end_node.name,
                length=link.length,
                free_flow_speed=link.u,
                jam_density=link.kappa,
                merge_priority=link.merge_priority,
                attribute=link.attribute.copy() if hasattr(link, 'attribute') else {}
            )

        # 复制车辆
        for veh in self.W.VEHICLES.values():
            self.Vehicle(
                W, veh.orig.name, veh.dest.name,
                veh.departure_time,
                attribute=veh.attribute.copy() if hasattr(veh, 'attribute') else {}
            )

        return W

    def apply_routes_to_vehicles(self, W, routes_specified):
        # 构建名称到车辆的映射（避免重复构建）
        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}

        for veh_id, route_links in routes_specified.items():
            veh = veh_dict.get(veh_id)
            if veh is not None:
                veh.assign_route(route_links)

    def exec_simulation(self, W):
        W.exec_simulation()

    def route_choice_update(self, W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                            current_routes, iteration):
        new_routes = {}
        relative_gaps = []
        total_switches = 0
        completed_count = 0
        total_count = 0

        gamma = self.settings.get("ue_switch_gamma", 5)
        alpha = self.settings.get("ue_switch_alpha", 0.08)
        uncompleted_switch_prob = self.settings.get("ue_uncompleted_switch_prob", 0.3)
        time_value_coefficient = self.settings.get("time_value_coefficient", 0.005)
        charging_demand = self.settings.get("charging_demand_per_vehicle", 50)
        period_duration = self.settings["simulation_time"] / self.settings["charging_periods"]

        # 计算固定价格（与 EVCSChargingGameEnv 一致）
        # 使用归一化价格 0.5，对应实际价格 = price_min + 0.5 * (price_max - price_min)
        charging_nodes = self.settings.get("charging_nodes", {})
        if charging_nodes:
            first_node_range = list(charging_nodes.values())[0]
            price_min, price_max = first_node_range[0], first_node_range[1]
            price_ratio = 0.5  # 归一化价格
            fixed_price = price_min + price_ratio * (price_max - price_min)
        else:
            fixed_price = 0.5  # 默认值

        # 构建名称到车辆的映射
        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}

        # 处理充电车辆
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"].get(od_pair, [])

            for veh_id in veh_ids:
                total_count += 1
                veh = veh_dict.get(veh_id)
                if veh is None:
                    new_routes[veh_id] = current_routes.get(veh_id, [])
                    continue

                current_route = current_routes.get(veh_id, [])

                if veh.state != "end":
                    if np.random.random() < uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r != current_route]
                        if other_routes:
                            new_routes[veh_id] = other_routes[np.random.randint(len(other_routes))]
                            total_switches += 1
                        else:
                            new_routes[veh_id] = current_route
                    else:
                        new_routes[veh_id] = current_route
                    continue

                completed_count += 1

                route, timestamps = veh.traveled_route()
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
                current_cost = time_value_coefficient * travel_time + fixed_price * charging_demand

                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    dep_time = veh.departure_time_in_second
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_travel_time = route_obj.actual_travel_time(dep_time)
                        alt_cost = time_value_coefficient * alt_travel_time + fixed_price * charging_demand
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links

                if current_cost > 0:
                    relative_gap = (current_cost - best_cost) / current_cost
                else:
                    relative_gap = 0
                relative_gaps.append(relative_gap)

                if relative_gap > 0:
                    gap_factor = min(1.0, gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes[veh_id] = best_route
                        total_switches += 1
                    else:
                        new_routes[veh_id] = current_route
                else:
                    new_routes[veh_id] = current_route

        # 处理非充电车辆
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"].get(od_pair, [])

            for veh_id in veh_ids:
                total_count += 1
                veh = veh_dict.get(veh_id)
                if veh is None:
                    new_routes[veh_id] = current_routes.get(veh_id, [])
                    continue

                current_route = current_routes.get(veh_id, [])

                if veh.state != "end":
                    if np.random.random() < uncompleted_switch_prob and available_routes:
                        other_routes = [r for r in available_routes if r != current_route]
                        if other_routes:
                            new_routes[veh_id] = other_routes[np.random.randint(len(other_routes))]
                            total_switches += 1
                        else:
                            new_routes[veh_id] = current_route
                    else:
                        new_routes[veh_id] = current_route
                    continue

                completed_count += 1

                route, timestamps = veh.traveled_route()
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
                current_cost = time_value_coefficient * travel_time

                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    dep_time = veh.departure_time_in_second
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_travel_time = route_obj.actual_travel_time(dep_time)
                        alt_cost = time_value_coefficient * alt_travel_time
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links

                if current_cost > 0:
                    relative_gap = (current_cost - best_cost) / current_cost
                else:
                    relative_gap = 0
                relative_gaps.append(relative_gap)

                if relative_gap > 0:
                    gap_factor = min(1.0, gamma * relative_gap)
                    decay_factor = 1.0 / (1.0 + alpha * iteration)
                    switch_prob = gap_factor * decay_factor
                    if np.random.random() < switch_prob:
                        new_routes[veh_id] = best_route
                        total_switches += 1
                    else:
                        new_routes[veh_id] = current_route
                else:
                    new_routes[veh_id] = current_route

        stats = {
            'all_relative_gap_global_mean': np.mean(relative_gaps) if relative_gaps else 0,
            'completed_ratio': completed_count / total_count if total_count > 0 else 0,
            'total_switches': total_switches
        }

        return stats, new_routes


# ============================================================
# 主函数
# ============================================================
def compare_results(uxsim_records, uxsimpp_records):
    """对比两个后端的结果"""
    print(f"\n{'=' * 80}")
    print("性能对比分析")
    print(f"{'=' * 80}")

    # 收集所有环节
    all_labels = set(uxsim_records.keys()) | set(uxsimpp_records.keys())

    print(f"\n{'环节':<40} {'UXsim(ms)':>12} {'uxsimpp(ms)':>12} {'加速比':>10}")
    print(f"{'-' * 80}")

    uxsim_total = 0
    uxsimpp_total = 0

    for label in sorted(all_labels):
        uxsim_times = uxsim_records.get(label, [])
        uxsimpp_times = uxsimpp_records.get(label, [])

        uxsim_sum = sum(uxsim_times) * 1000
        uxsimpp_sum = sum(uxsimpp_times) * 1000

        uxsim_total += uxsim_sum
        uxsimpp_total += uxsimpp_sum

        if uxsimpp_sum > 0:
            speedup = uxsim_sum / uxsimpp_sum
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{label:<40} {uxsim_sum:>12.2f} {uxsimpp_sum:>12.2f} {speedup_str:>10}")

    print(f"{'-' * 80}")
    overall_speedup = uxsim_total / uxsimpp_total if uxsimpp_total > 0 else 0
    print(f"{'总计':<40} {uxsim_total:>12.2f} {uxsimpp_total:>12.2f} {overall_speedup:.2f}x")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="UE-DTA 性能对比测试")
    parser.add_argument("--backend", choices=["uxsim", "uxsimpp"],
                        help="测试单个后端")
    parser.add_argument("--both", action="store_true",
                        help="测试两个后端并对比")
    args = parser.parse_args()

    if not args.backend and not args.both:
        args.both = True

    uxsim_records = None
    uxsimpp_records = None

    if args.both or args.backend == "uxsim":
        print("\n" + "=" * 80)
        print("测试 UXsim + Patch")
        print("=" * 80)
        try:
            test = UXsimPatchTest()
            uxsim_records = test.run_test()
        except Exception as e:
            print(f"UXsim + Patch 测试失败: {e}")
            import traceback
            traceback.print_exc()

    if args.both or args.backend == "uxsimpp":
        print("\n" + "=" * 80)
        print("测试 uxsimpp_extended")
        print("=" * 80)
        try:
            test = UXsimppTest()
            uxsimpp_records = test.run_test()
        except Exception as e:
            print(f"uxsimpp_extended 测试失败: {e}")
            import traceback
            traceback.print_exc()

    if args.both and uxsim_records and uxsimpp_records:
        compare_results(uxsim_records, uxsimpp_records)


if __name__ == "__main__":
    main()
