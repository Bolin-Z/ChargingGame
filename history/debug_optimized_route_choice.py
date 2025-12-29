"""
验证预缓存优化效果

在 route_choice_update 中应用预缓存策略
"""

import time
import os
import sys
import json
import numpy as np
from collections import defaultdict
from itertools import islice
import networkx as nx

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

NETWORK_DIR = os.path.join(project_root, "data", "siouxfalls")
NETWORK_NAME = "siouxfalls"


class OptimizedRouteChoiceTest:
    """优化后的路径选择更新测试"""

    def __init__(self):
        from uxsimpp_extended.uxsimpp import World, Vehicle, Route
        self.World = World
        self.Vehicle = Vehicle
        self.Route = Route
        self.settings = None
        self.W = None
        self.charging_nodes = None
        self.dict_od_to_routes = None

    def load_settings(self):
        settings_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_settings.json")
        with open(settings_path, "r") as f:
            self.settings = json.load(f)

    def create_world(self):
        return self.World(
            name=self.settings["network_name"],
            deltan=self.settings["deltan"],
            tmax=self.settings["simulation_time"],
            print_mode=0, save_mode=0, show_mode=0,
            random_seed=42
        )

    def load_network(self, W):
        import csv
        node_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_nodes.csv")
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":
                    W.addNode(r[0], float(r[1]), float(r[2]))

        link_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_links.csv")
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":
                    W.addLink(r[0], r[1], r[2], length=float(r[3]),
                             free_flow_speed=float(r[4]), jam_density=float(r[5]),
                             merge_priority=float(r[6]), attribute={"charging_link": False})

        self.charging_nodes = self.settings["charging_nodes"]
        for node in self.charging_nodes.keys():
            W.addLink(f"charging_{node}", node, node,
                     length=self.settings["charging_link_length"],
                     free_flow_speed=self.settings["charging_link_free_flow_speed"],
                     attribute={"charging_link": True})

        demand_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_demand.csv")
        demand_multiplier = self.settings.get("demand_multiplier", 1.0)
        charging_car_rate = self.settings["charging_car_rate"]

        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    flow = float(r[4]) * demand_multiplier
                    W.adddemand(origin, destination, start_t, end_t,
                               flow * charging_car_rate, attribute={"charging_car": True})
                    W.adddemand(origin, destination, start_t, end_t,
                               flow * (1 - charging_car_rate), attribute={"charging_car": False})

    def compute_routes(self):
        k = self.settings["routes_per_od"]
        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            od_pairs.add((veh.orig.name, veh.dest.name))

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
            paths = list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))
        except nx.NetworkXNoPath:
            return []
        return [[link_dict[(paths[j][i], paths[j][i+1])] for i in range(len(paths[j])-1)] for j in range(len(paths))]

    def _enumerate_k_shortest_charge_routes(self, source, target, k):
        G = nx.DiGraph()
        link_dict = {}
        for link in self.W.LINKS:
            s, e = link.start_node.name, link.end_node.name
            w = link.length / link.u
            if link.attribute.get("charging_link", False):
                G.add_edge(f"uncharged_{s}", f"charged_{s}", weight=w)
                link_dict[(f"uncharged_{s}", f"charged_{s}")] = link.name
            else:
                G.add_edge(f"uncharged_{s}", f"uncharged_{e}", weight=w)
                G.add_edge(f"charged_{s}", f"charged_{e}", weight=w)
                link_dict[(f"uncharged_{s}", f"uncharged_{e}")] = link.name
                link_dict[(f"charged_{s}", f"charged_{e}")] = link.name
        try:
            paths = list(islice(nx.shortest_simple_paths(G, f"uncharged_{source}", f"charged_{target}", weight='weight'), k))
        except nx.NetworkXNoPath:
            return []
        return [[link_dict[(paths[j][i], paths[j][i+1])] for i in range(len(paths[j])-1)] for j in range(len(paths))]

    def build_od_vehicle_mapping(self):
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        for veh in self.W.VEHICLES.values():
            key = veh.name
            o, d = veh.orig.name, veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)
        return dict_od_to_charging_vehid, dict_od_to_uncharging_vehid

    def initialize_routes(self, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid):
        routes_specified = {}
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            routes = self.dict_od_to_routes["charging"].get(od_pair, [])
            if routes:
                for veh_id in veh_ids:
                    routes_specified[veh_id] = routes[0]
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            routes = self.dict_od_to_routes["uncharging"].get(od_pair, [])
            if routes:
                for veh_id in veh_ids:
                    routes_specified[veh_id] = routes[0]
        return routes_specified

    def create_simulation_world(self):
        W = self.World(
            name=self.settings["network_name"] + "_sim",
            deltan=self.settings["deltan"],
            tmax=self.settings["simulation_time"],
            print_mode=0, save_mode=0, show_mode=0,
            random_seed=42
        )
        for node in self.W.NODES:
            W.addNode(node.name, node.x, node.y)
        for link in self.W.LINKS:
            W.addLink(link.name, link.start_node.name, link.end_node.name,
                     length=link.length, free_flow_speed=link.u,
                     jam_density=link.kappa, merge_priority=link.merge_priority,
                     attribute=link.attribute.copy() if hasattr(link, 'attribute') else {})
        for veh in self.W.VEHICLES.values():
            self.Vehicle(W, veh.orig.name, veh.dest.name, veh.departure_time,
                        attribute=veh.attribute.copy() if hasattr(veh, 'attribute') else {})
        return W

    def apply_routes_to_vehicles(self, W, routes_specified):
        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}
        for veh_id, route_links in routes_specified.items():
            veh = veh_dict.get(veh_id)
            if veh is not None:
                veh.assign_route(route_links)

    def route_choice_update_original(self, W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                                     current_routes, iteration):
        """原始版本（无优化）"""
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
            fixed_price = first_node_range[0] + 0.5 * (first_node_range[1] - first_node_range[0])
        else:
            fixed_price = 0.5

        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}

        for od_pair, veh_ids in list(dict_od_to_charging_vehid.items()) + list(dict_od_to_uncharging_vehid.items()):
            is_charging = od_pair in dict_od_to_charging_vehid
            available_routes = self.dict_od_to_routes["charging" if is_charging else "uncharging"].get(od_pair, [])

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
                if is_charging:
                    current_cost = time_value_coefficient * travel_time + fixed_price * charging_demand
                else:
                    current_cost = time_value_coefficient * travel_time

                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    for route_links in available_routes:
                        route_obj = W.defRoute(route_links)
                        alt_travel_time = route_obj.actual_travel_time(veh.departure_time_in_second)
                        if is_charging:
                            alt_cost = time_value_coefficient * alt_travel_time + fixed_price * charging_demand
                        else:
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

    def route_choice_update_optimized(self, W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                                      current_routes, iteration):
        """优化版本：预缓存 traveltime_real"""
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
            fixed_price = first_node_range[0] + 0.5 * (first_node_range[1] - first_node_range[0])
        else:
            fixed_price = 0.5

        veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}

        # ========== 关键优化：预缓存所有链路的 traveltime_real ==========
        link_cache = {}
        delta_t = None
        for link in W.LINKS:
            if delta_t is None:
                delta_t = link.W.delta_t
            tr = list(link.traveltime_real)  # 只复制一次！
            link_cache[link.name] = {
                'traveltime_real': tr,
                'max_idx': len(tr) - 1
            }

        def cached_actual_travel_time(link_name, t):
            """使用缓存计算旅行时间"""
            cache = link_cache[link_name]
            tt_idx = int(t // delta_t)
            if tt_idx > cache['max_idx']:
                tt_idx = cache['max_idx']
            elif tt_idx < 0:
                tt_idx = 0
            return cache['traveltime_real'][tt_idx]

        def cached_route_travel_time(route_links, start_time):
            """使用缓存计算路径旅行时间"""
            tt = 0
            current_t = start_time
            for link_name in route_links:
                link_tt = cached_actual_travel_time(link_name, current_t)
                tt += link_tt
                current_t += link_tt
            return tt
        # ========== 优化结束 ==========

        for od_pair, veh_ids in list(dict_od_to_charging_vehid.items()) + list(dict_od_to_uncharging_vehid.items()):
            is_charging = od_pair in dict_od_to_charging_vehid
            available_routes = self.dict_od_to_routes["charging" if is_charging else "uncharging"].get(od_pair, [])

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
                if is_charging:
                    current_cost = time_value_coefficient * travel_time + fixed_price * charging_demand
                else:
                    current_cost = time_value_coefficient * travel_time

                best_cost = current_cost
                best_route = current_route

                if available_routes:
                    dep_time = veh.departure_time_in_second
                    for route_links in available_routes:
                        # 使用缓存计算旅行时间
                        alt_travel_time = cached_route_travel_time(route_links, dep_time)
                        if is_charging:
                            alt_cost = time_value_coefficient * alt_travel_time + fixed_price * charging_demand
                        else:
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

    def run_comparison(self, num_iterations=3):
        """运行对比测试"""
        print("=" * 80)
        print("预缓存优化效果验证")
        print("=" * 80)

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

        print(f"[5/5] 运行 {num_iterations} 次迭代对比测试...\n")

        original_times = []
        optimized_times = []

        for iteration in range(num_iterations):
            print(f"迭代 {iteration + 1}/{num_iterations}:")

            # 创建仿真世界
            W_sim = self.create_simulation_world()
            self.apply_routes_to_vehicles(W_sim, current_routes)
            W_sim.exec_simulation()

            # 测试原始版本
            np.random.seed(42)  # 固定随机种子确保可比性
            start = time.perf_counter()
            stats_orig, new_routes_orig = self.route_choice_update_original(
                W_sim, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                current_routes, iteration
            )
            t_orig = (time.perf_counter() - start) * 1000
            original_times.append(t_orig)

            # 测试优化版本
            np.random.seed(42)  # 固定随机种子确保可比性
            start = time.perf_counter()
            stats_opt, new_routes_opt = self.route_choice_update_optimized(
                W_sim, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                current_routes, iteration
            )
            t_opt = (time.perf_counter() - start) * 1000
            optimized_times.append(t_opt)

            speedup = t_orig / t_opt if t_opt > 0 else 0

            print(f"  原始版本: {t_orig:8.0f} ms | GM: {stats_orig['all_relative_gap_global_mean']*100:.2f}%")
            print(f"  优化版本: {t_opt:8.0f} ms | GM: {stats_opt['all_relative_gap_global_mean']*100:.2f}%")
            print(f"  加速比:   {speedup:.1f}x\n")

            current_routes = new_routes_opt

        print("=" * 80)
        print("总结")
        print("=" * 80)
        avg_orig = np.mean(original_times)
        avg_opt = np.mean(optimized_times)
        print(f"原始版本平均耗时: {avg_orig:.0f} ms")
        print(f"优化版本平均耗时: {avg_opt:.0f} ms")
        print(f"平均加速比:       {avg_orig/avg_opt:.1f}x")


if __name__ == "__main__":
    test = OptimizedRouteChoiceTest()
    test.run_comparison(num_iterations=3)
