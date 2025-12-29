"""
路径选择更新子步骤性能分析脚本

专门分析 route_choice_update 函数中各个子步骤的耗时。
"""

import time
import os
import sys
import json
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
# 细分计时器
# ============================================================
class DetailedTimer:
    """细分计时器"""

    def __init__(self):
        self.records = defaultdict(list)
        self._start_times = {}

    def start(self, label):
        self._start_times[label] = time.perf_counter()

    def stop(self, label):
        if label in self._start_times:
            duration = time.perf_counter() - self._start_times[label]
            self.records[label].append(duration)
            del self._start_times[label]
            return duration
        return 0

    def summary(self, title=""):
        print(f"\n{'=' * 80}")
        print(f"细分耗时分析: {title}")
        print(f"{'=' * 80}")
        print(f"{'子步骤':<50} {'调用次数':>8} {'总耗时(ms)':>12} {'平均(ms)':>10} {'占比':>8}")
        print(f"{'-' * 80}")

        total_time = sum(sum(times) for times in self.records.values())

        sorted_items = sorted(self.records.items(), key=lambda x: sum(x[1]), reverse=True)

        for label, times in sorted_items:
            total_ms = sum(times) * 1000
            avg_ms = np.mean(times) * 1000
            count = len(times)
            pct = (sum(times) / total_time * 100) if total_time > 0 else 0
            print(f"{label:<50} {count:>8} {total_ms:>12.2f} {avg_ms:>10.4f} {pct:>7.1f}%")

        print(f"{'-' * 80}")
        print(f"{'总计':<50} {'':<8} {total_time * 1000:>12.2f}")
        print(f"{'=' * 80}")


# ============================================================
# 测试类
# ============================================================
class RouteChoiceUpdateAnalyzer:
    """路径选择更新分析器"""

    def __init__(self):
        from uxsimpp_extended.uxsimpp import World, Vehicle, Route
        self.World = World
        self.Vehicle = Vehicle
        self.Route = Route
        self.settings = None
        self.W = None
        self.charging_nodes = None
        self.dict_od_to_routes = None
        self.timer = DetailedTimer()

    def load_settings(self):
        settings_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_settings.json")
        with open(settings_path, "r") as f:
            self.settings = json.load(f)
        return self.settings

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
        import csv
        # 加载节点
        node_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_nodes.csv")
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

                    W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * charging_car_rate,
                        attribute={"charging_car": True}
                    )
                    W.adddemand(
                        origin, destination, start_t, end_t,
                        flow * (1 - charging_car_rate),
                        attribute={"charging_car": False}
                    )

    def compute_routes(self):
        k = self.settings["routes_per_od"]
        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            o = veh.orig.name
            d = veh.dest.name
            od_pairs.add((o, d))

        self.dict_od_to_routes = {"uncharging": {}, "charging": {}}

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

        for node in self.W.NODES:
            W.addNode(node.name, node.x, node.y)

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

        for veh in self.W.VEHICLES.values():
            self.Vehicle(
                W, veh.orig.name, veh.dest.name,
                veh.departure_time,
                attribute=veh.attribute.copy() if hasattr(veh, 'attribute') else {}
            )

        return W

    def apply_routes_to_vehicles(self, W, routes_specified):
        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}
        for veh_id, route_links in routes_specified.items():
            veh = veh_dict.get(veh_id)
            if veh is not None:
                veh.assign_route(route_links)

    def route_choice_update_detailed(self, W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                                     current_routes, iteration):
        """带详细计时的路径选择更新"""
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

        charging_nodes = self.settings.get("charging_nodes", {})
        if charging_nodes:
            first_node_range = list(charging_nodes.values())[0]
            price_min, price_max = first_node_range[0], first_node_range[1]
            fixed_price = price_min + 0.5 * (price_max - price_min)
        else:
            fixed_price = 0.5

        # 1. 构建名称到车辆的映射
        self.timer.start("1.构建veh_dict")
        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}
        self.timer.stop("1.构建veh_dict")

        # 处理充电车辆
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"].get(od_pair, [])

            for veh_id in veh_ids:
                total_count += 1

                # 2. 获取车辆对象
                self.timer.start("2.veh_dict查找")
                veh = veh_dict.get(veh_id)
                self.timer.stop("2.veh_dict查找")

                if veh is None:
                    new_routes[veh_id] = current_routes.get(veh_id, [])
                    continue

                current_route = current_routes.get(veh_id, [])

                # 3. 检查车辆状态
                self.timer.start("3.检查veh.state")
                veh_state = veh.state
                self.timer.stop("3.检查veh.state")

                if veh_state != "end":
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

                # 4. 调用 traveled_route()
                self.timer.start("4.traveled_route()")
                route, timestamps = veh.traveled_route()
                self.timer.stop("4.traveled_route()")

                # 5. 计算当前成本
                self.timer.start("5.计算当前成本")
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
                current_cost = time_value_coefficient * travel_time + fixed_price * charging_demand
                self.timer.stop("5.计算当前成本")

                best_cost = current_cost
                best_route = current_route

                # 6. 遍历候选路径
                if available_routes:
                    for route_links in available_routes:
                        # 6a. 创建 Route 对象
                        self.timer.start("6a.defRoute()")
                        route_obj = W.defRoute(route_links)
                        self.timer.stop("6a.defRoute()")

                        # 6b. 获取出发时间
                        self.timer.start("6b.departure_time_in_second")
                        dep_time = veh.departure_time_in_second
                        self.timer.stop("6b.departure_time_in_second")

                        # 6c. 计算 actual_travel_time
                        self.timer.start("6c.actual_travel_time()")
                        alt_travel_time = route_obj.actual_travel_time(dep_time)
                        self.timer.stop("6c.actual_travel_time()")

                        # 6d. 计算替代成本
                        self.timer.start("6d.计算替代成本")
                        alt_cost = time_value_coefficient * alt_travel_time + fixed_price * charging_demand
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links
                        self.timer.stop("6d.计算替代成本")

                # 7. 计算相对gap和切换决策
                self.timer.start("7.切换决策逻辑")
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
                self.timer.stop("7.切换决策逻辑")

        # 处理非充电车辆（同样逻辑）
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"].get(od_pair, [])

            for veh_id in veh_ids:
                total_count += 1

                self.timer.start("2.veh_dict查找")
                veh = veh_dict.get(veh_id)
                self.timer.stop("2.veh_dict查找")

                if veh is None:
                    new_routes[veh_id] = current_routes.get(veh_id, [])
                    continue

                current_route = current_routes.get(veh_id, [])

                self.timer.start("3.检查veh.state")
                veh_state = veh.state
                self.timer.stop("3.检查veh.state")

                if veh_state != "end":
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

                self.timer.start("4.traveled_route()")
                route, timestamps = veh.traveled_route()
                self.timer.stop("4.traveled_route()")

                self.timer.start("5.计算当前成本")
                travel_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
                current_cost = time_value_coefficient * travel_time
                self.timer.stop("5.计算当前成本")

                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    for route_links in available_routes:
                        self.timer.start("6a.defRoute()")
                        route_obj = W.defRoute(route_links)
                        self.timer.stop("6a.defRoute()")

                        self.timer.start("6b.departure_time_in_second")
                        dep_time = veh.departure_time_in_second
                        self.timer.stop("6b.departure_time_in_second")

                        self.timer.start("6c.actual_travel_time()")
                        alt_travel_time = route_obj.actual_travel_time(dep_time)
                        self.timer.stop("6c.actual_travel_time()")

                        self.timer.start("6d.计算替代成本")
                        alt_cost = time_value_coefficient * alt_travel_time
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links
                        self.timer.stop("6d.计算替代成本")

                self.timer.start("7.切换决策逻辑")
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
                self.timer.stop("7.切换决策逻辑")

        stats = {
            'all_relative_gap_global_mean': np.mean(relative_gaps) if relative_gaps else 0,
            'completed_ratio': completed_count / total_count if total_count > 0 else 0,
            'total_switches': total_switches
        }

        return stats, new_routes

    def run_analysis(self, num_iterations=3):
        """运行分析"""
        print("=" * 80)
        print("路径选择更新子步骤性能分析")
        print("=" * 80)

        # 1. 初始化
        print("\n[1/5] 加载配置...")
        self.load_settings()

        print("[2/5] 创建模板世界...")
        self.W = self.create_world()
        self.load_network(self.W)

        print("[3/5] 计算路径集合...")
        self.compute_routes()

        print("[4/5] 构建OD-车辆映射...")
        dict_od_to_charging_vehid, dict_od_to_uncharging_vehid = self.build_od_vehicle_mapping()
        current_routes = self.initialize_routes(dict_od_to_charging_vehid, dict_od_to_uncharging_vehid)

        print(f"[5/5] 运行 {num_iterations} 次迭代进行性能分析...\n")

        for iteration in range(num_iterations):
            print(f"  迭代 {iteration + 1}/{num_iterations}...")

            # 创建仿真世界
            W_sim = self.create_simulation_world()

            # 应用路径
            self.apply_routes_to_vehicles(W_sim, current_routes)

            # 执行仿真
            W_sim.exec_simulation()

            # 带详细计时的路径选择更新
            iter_start = time.perf_counter()
            stats, new_routes = self.route_choice_update_detailed(
                W_sim,
                dict_od_to_charging_vehid,
                dict_od_to_uncharging_vehid,
                current_routes,
                iteration
            )
            iter_time = time.perf_counter() - iter_start

            print(f"    完成率: {stats['completed_ratio']*100:.1f}% | "
                  f"GM: {stats['all_relative_gap_global_mean']*100:.2f}% | "
                  f"耗时: {iter_time*1000:.0f}ms")

            current_routes = new_routes

        # 输出统计
        self.timer.summary("route_choice_update 子步骤")

        # 额外统计
        print("\n" + "=" * 80)
        print("补充信息")
        print("=" * 80)
        print(f"车辆总数: {len(list(self.W.VEHICLES.values()))}")
        print(f"充电车辆OD对数: {len(dict_od_to_charging_vehid)}")
        print(f"非充电车辆OD对数: {len(dict_od_to_uncharging_vehid)}")
        print(f"每个OD对的路径数: {self.settings['routes_per_od']}")


if __name__ == "__main__":
    analyzer = RouteChoiceUpdateAnalyzer()
    analyzer.run_analysis(num_iterations=3)
