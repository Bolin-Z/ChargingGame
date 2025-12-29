"""
SF 网络性能对比测试

比较 UXsim 1.8.2 + Patch 与 uxsimpp_extended 在 Sioux Falls 数据集上的单轮仿真性能
使用 EVCSChargingGameEnv 的方式加载网络并分配预定路径
"""
import sys
import os
import subprocess
import json

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "siouxfalls")
NETWORK_NAME = "siouxfalls"


def run_uxsim_patch_test():
    """测试 UXsim 1.8.2 + Patch 方案（使用 EVCSChargingGameEnv 的原始版本）"""
    code = f'''
import sys
import os
import time
import json

# 添加项目路径
sys.path.insert(0, r"{PROJECT_ROOT}")

# 临时切换 EVCSChargingGameEnv 使用 UXsim 1.8.2 + Patch
# 需要修改导入，这里我们直接复制关键逻辑

sys.path.insert(0, os.path.join(r"{PROJECT_ROOT}", "UXsim-1.8.2"))

from src.env.patch import patch_uxsim
patch_uxsim()

import csv
import networkx as nx
from itertools import islice
from collections import defaultdict
from uxsim import World, Vehicle

# 加载配置
with open(os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_settings.json"), "r") as f:
    settings = json.load(f)

deltan = settings["deltan"]
simulation_time = settings["simulation_time"]
demand_multiplier = settings["demand_multiplier"]
charging_car_rate = settings["charging_car_rate"]
charging_link_length = settings["charging_link_length"]
charging_link_free_flow_speed = settings["charging_link_free_flow_speed"]
charging_nodes = settings["charging_nodes"]
routes_per_od = settings["routes_per_od"]

results = {{}}
total_start = time.perf_counter()

# ============ 1. 创建 World ============
t0 = time.perf_counter()
W = World(
    name="{NETWORK_NAME}",
    deltan=deltan,
    tmax=simulation_time,
    random_seed=42,
    print_mode=0,
    save_mode=0
)
results["create_world_ms"] = (time.perf_counter() - t0) * 1000

# ============ 2. 加载节点 ============
t0 = time.perf_counter()
node_path = os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_nodes.csv")
with open(node_path, "r") as f:
    for r in csv.reader(f):
        if r[1] != "x":
            name, x, y = r
            W.addNode(name, float(x), float(y))
results["add_nodes_ms"] = (time.perf_counter() - t0) * 1000
results["node_count"] = len(W.NODES)

# ============ 3. 加载链路 ============
t0 = time.perf_counter()
link_path = os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_links.csv")
link_dict = {{}}  # (start, end) -> link_name
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
                merge_priority=merge_priority
            )
            link_dict[(start_node, end_node)] = link_name

# 为链路添加 attribute
for link in W.LINKS:
    link.attribute = {{"charging_link": False}}

results["add_links_ms"] = (time.perf_counter() - t0) * 1000

# ============ 4. 创建充电链路 ============
t0 = time.perf_counter()
for node in charging_nodes.keys():
    charging_link_name = f"charging_{{node}}"
    link = W.addLink(
        charging_link_name,
        start_node=node,
        end_node=node,
        length=charging_link_length,
        free_flow_speed=charging_link_free_flow_speed
    )
    link.attribute = {{"charging_link": True}}
    link_dict[(node, node)] = charging_link_name
results["add_charging_links_ms"] = (time.perf_counter() - t0) * 1000
results["link_count"] = len(W.LINKS)

# ============ 5. 计算路径 ============
t0 = time.perf_counter()

def enumerate_k_shortest_routes(source, target, k, include_charging=False):
    G = nx.DiGraph()
    if include_charging:
        # 多状态图：uncharged/charged
        for link in W.LINKS:
            start = link.start_node.name
            end = link.end_node.name
            weight = link.length / link.u
            if link.attribute["charging_link"]:
                G.add_edge(f"uncharged_{{start}}", f"charged_{{end}}", weight=weight)
            else:
                G.add_edge(f"uncharged_{{start}}", f"uncharged_{{end}}", weight=weight)
                G.add_edge(f"charged_{{start}}", f"charged_{{end}}", weight=weight)
        try:
            paths = list(islice(nx.shortest_simple_paths(G, f"uncharged_{{source}}", f"charged_{{target}}", weight='weight'), k))
        except nx.NetworkXNoPath:
            return []
    else:
        for link in W.LINKS:
            if not link.attribute["charging_link"]:
                G.add_edge(link.start_node.name, link.end_node.name, weight=link.length/link.u)
        try:
            paths = list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))
        except nx.NetworkXNoPath:
            return []

    # 转换为链路名称
    routes = []
    for path in paths:
        route = []
        for i in range(len(path) - 1):
            n1 = path[i].replace("uncharged_", "").replace("charged_", "")
            n2 = path[i+1].replace("uncharged_", "").replace("charged_", "")
            if (n1, n2) in link_dict:
                route.append(link_dict[(n1, n2)])
        if route:
            routes.append(route)
    return routes

# 收集OD对
od_pairs = set()
demand_path = os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_demand.csv")
with open(demand_path, "r") as f:
    for r in csv.reader(f):
        if r[2] != "start_t":
            od_pairs.add((r[0], r[1]))

# 计算路径
dict_od_to_routes = {{"uncharging": {{}}, "charging": {{}}}}
for o, d in od_pairs:
    dict_od_to_routes["uncharging"][(o, d)] = enumerate_k_shortest_routes(o, d, routes_per_od, False)
    dict_od_to_routes["charging"][(o, d)] = enumerate_k_shortest_routes(o, d, routes_per_od, True)

results["compute_routes_ms"] = (time.perf_counter() - t0) * 1000
results["od_pairs"] = len(od_pairs)

# ============ 6. 加载交通需求并分配路径 ============
t0 = time.perf_counter()
with open(demand_path, "r") as f:
    for r in csv.reader(f):
        if r[2] != "start_t":
            origin, destination = r[0], r[1]
            start_t, end_t = float(r[2]), float(r[3])
            flow = float(r[4]) * demand_multiplier

            # 充电车辆
            charging_routes = dict_od_to_routes["charging"].get((origin, destination), [])
            if charging_routes:
                W.adddemand(
                    origin, destination, start_t, end_t,
                    flow * charging_car_rate,
                    attribute={{"charging_car": True}}
                )

            # 非充电车辆
            uncharging_routes = dict_od_to_routes["uncharging"].get((origin, destination), [])
            if uncharging_routes:
                W.adddemand(
                    origin, destination, start_t, end_t,
                    flow * (1 - charging_car_rate),
                    attribute={{"charging_car": False}}
                )

results["add_demand_ms"] = (time.perf_counter() - t0) * 1000
results["vehicle_objects"] = len(W.VEHICLES)
results["actual_vehicles"] = len(W.VEHICLES) * deltan

# ============ 7. 为车辆分配路径 ============
t0 = time.perf_counter()
for veh in W.VEHICLES.values():
    o = veh.orig.name
    d = veh.dest.name
    if veh.attribute and veh.attribute.get("charging_car"):
        routes = dict_od_to_routes["charging"].get((o, d), [])
    else:
        routes = dict_od_to_routes["uncharging"].get((o, d), [])
    if routes:
        veh.assign_route(routes[0])  # 使用最短路径
results["assign_routes_ms"] = (time.perf_counter() - t0) * 1000

# ============ 8. 执行仿真 ============
t0 = time.perf_counter()
W.exec_simulation()
results["exec_simulation_ms"] = (time.perf_counter() - t0) * 1000

# ============ 9. 遍历车辆获取状态 ============
t0 = time.perf_counter()
completed_count = 0
for veh in W.VEHICLES.values():
    if veh.state == "end":
        completed_count += 1
results["iterate_vehicles_ms"] = (time.perf_counter() - t0) * 1000
results["completed_vehicles"] = completed_count

# ============ 10. 获取 traveled_route（采样100辆） ============
t0 = time.perf_counter()
sample_count = min(100, len(W.VEHICLES))
sample_vehs = list(W.VEHICLES.values())[:sample_count]
for veh in sample_vehs:
    if veh.state == "end":
        route, timestamps = veh.traveled_route()
results["traveled_route_100_ms"] = (time.perf_counter() - t0) * 1000

results["total_ms"] = (time.perf_counter() - total_start) * 1000

# 输出结果
print(json.dumps(results))
'''

    print("=" * 70)
    print("测试 UXsim 1.8.2 + Patch 方案")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("错误:", result.stderr)
        return None

    try:
        results = json.loads(result.stdout.strip())
        return results
    except json.JSONDecodeError:
        print("输出解析失败:")
        print(result.stdout)
        print("STDERR:", result.stderr)
        return None


def run_uxsimpp_extended_test():
    """测试 uxsimpp_extended 方案"""
    code = f'''
import sys
import os
import time
import csv
import json
import networkx as nx
from itertools import islice
from collections import defaultdict

# uxsimpp_extended 是通过 pip install -e . 安装的
from uxsimpp_extended.uxsimpp import (
    World, Vehicle, Link, Node,
    newWorld, VEHICLE_STATE
)

# 加载配置
with open(os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_settings.json"), "r") as f:
    settings = json.load(f)

deltan = settings["deltan"]
simulation_time = settings["simulation_time"]
demand_multiplier = settings["demand_multiplier"]
charging_car_rate = settings["charging_car_rate"]
charging_link_length = settings["charging_link_length"]
charging_link_free_flow_speed = settings["charging_link_free_flow_speed"]
charging_nodes = settings["charging_nodes"]
routes_per_od = settings["routes_per_od"]

results = {{}}
total_start = time.perf_counter()

# ============ 1. 创建 World ============
t0 = time.perf_counter()
W = World(
    name="{NETWORK_NAME}",
    deltan=deltan,
    tmax=simulation_time,
    random_seed=42,
    print_mode=0,
    save_mode=0,
    show_mode=0,
    user_attribute={{}}
)
results["create_world_ms"] = (time.perf_counter() - t0) * 1000

# ============ 2. 加载节点 ============
t0 = time.perf_counter()
node_path = os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_nodes.csv")
with open(node_path, "r") as f:
    for r in csv.reader(f):
        if r[1] != "x":
            name, x, y = r
            W.addNode(name, float(x), float(y))
results["add_nodes_ms"] = (time.perf_counter() - t0) * 1000
results["node_count"] = len(W.NODES)

# ============ 3. 加载链路 ============
t0 = time.perf_counter()
link_path = os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_links.csv")
link_dict = {{}}  # (start, end) -> link_name
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
                attribute={{"charging_link": False}}
            )
            link_dict[(start_node, end_node)] = link_name
results["add_links_ms"] = (time.perf_counter() - t0) * 1000

# ============ 4. 创建充电链路 ============
t0 = time.perf_counter()
for node in charging_nodes.keys():
    charging_link_name = f"charging_{{node}}"
    W.addLink(
        charging_link_name,
        start_node=node,
        end_node=node,
        length=charging_link_length,
        free_flow_speed=charging_link_free_flow_speed,
        attribute={{"charging_link": True}}
    )
    link_dict[(node, node)] = charging_link_name
results["add_charging_links_ms"] = (time.perf_counter() - t0) * 1000
results["link_count"] = len(W.LINKS)

# ============ 5. 计算路径 ============
t0 = time.perf_counter()

def enumerate_k_shortest_routes(source, target, k, include_charging=False):
    G = nx.DiGraph()
    if include_charging:
        for link in W.LINKS:
            start = link.start_node.name
            end = link.end_node.name
            weight = link.length / link.u
            if link.attribute["charging_link"]:
                G.add_edge(f"uncharged_{{start}}", f"charged_{{end}}", weight=weight)
            else:
                G.add_edge(f"uncharged_{{start}}", f"uncharged_{{end}}", weight=weight)
                G.add_edge(f"charged_{{start}}", f"charged_{{end}}", weight=weight)
        try:
            paths = list(islice(nx.shortest_simple_paths(G, f"uncharged_{{source}}", f"charged_{{target}}", weight='weight'), k))
        except nx.NetworkXNoPath:
            return []
    else:
        for link in W.LINKS:
            if not link.attribute["charging_link"]:
                G.add_edge(link.start_node.name, link.end_node.name, weight=link.length/link.u)
        try:
            paths = list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))
        except nx.NetworkXNoPath:
            return []

    routes = []
    for path in paths:
        route = []
        for i in range(len(path) - 1):
            n1 = path[i].replace("uncharged_", "").replace("charged_", "")
            n2 = path[i+1].replace("uncharged_", "").replace("charged_", "")
            if (n1, n2) in link_dict:
                route.append(link_dict[(n1, n2)])
        if route:
            routes.append(route)
    return routes

od_pairs = set()
demand_path = os.path.join(r"{DATA_DIR}", "{NETWORK_NAME}_demand.csv")
with open(demand_path, "r") as f:
    for r in csv.reader(f):
        if r[2] != "start_t":
            od_pairs.add((r[0], r[1]))

dict_od_to_routes = {{"uncharging": {{}}, "charging": {{}}}}
for o, d in od_pairs:
    dict_od_to_routes["uncharging"][(o, d)] = enumerate_k_shortest_routes(o, d, routes_per_od, False)
    dict_od_to_routes["charging"][(o, d)] = enumerate_k_shortest_routes(o, d, routes_per_od, True)

results["compute_routes_ms"] = (time.perf_counter() - t0) * 1000
results["od_pairs"] = len(od_pairs)

# ============ 6. 加载交通需求并分配路径 ============
t0 = time.perf_counter()
with open(demand_path, "r") as f:
    for r in csv.reader(f):
        if r[2] != "start_t":
            origin, destination = r[0], r[1]
            start_t, end_t = float(r[2]), float(r[3])
            flow = float(r[4]) * demand_multiplier

            charging_routes = dict_od_to_routes["charging"].get((origin, destination), [])
            if charging_routes:
                W.adddemand(
                    origin, destination, start_t, end_t,
                    flow * charging_car_rate,
                    attribute={{"charging_car": True}}
                )

            uncharging_routes = dict_od_to_routes["uncharging"].get((origin, destination), [])
            if uncharging_routes:
                W.adddemand(
                    origin, destination, start_t, end_t,
                    flow * (1 - charging_car_rate),
                    attribute={{"charging_car": False}}
                )

results["add_demand_ms"] = (time.perf_counter() - t0) * 1000
results["vehicle_objects"] = len(W.VEHICLES)
results["actual_vehicles"] = len(W.VEHICLES) * deltan

# ============ 7. 为车辆分配路径 ============
t0 = time.perf_counter()
for veh in W.VEHICLES.values():
    o = veh.orig.name
    d = veh.dest.name
    if veh.attribute and veh.attribute.get("charging_car"):
        routes = dict_od_to_routes["charging"].get((o, d), [])
    else:
        routes = dict_od_to_routes["uncharging"].get((o, d), [])
    if routes:
        veh.assign_route(routes[0])
results["assign_routes_ms"] = (time.perf_counter() - t0) * 1000

# ============ 8. 执行仿真 ============
t0 = time.perf_counter()
W.exec_simulation()
results["exec_simulation_ms"] = (time.perf_counter() - t0) * 1000

# ============ 9. 遍历车辆获取状态 ============
t0 = time.perf_counter()
completed_count = 0
for veh in W.VEHICLES.values():
    if veh.state == "end":
        completed_count += 1
results["iterate_vehicles_ms"] = (time.perf_counter() - t0) * 1000
results["completed_vehicles"] = completed_count

# ============ 10. 获取 traveled_route（采样100辆） ============
t0 = time.perf_counter()
sample_count = min(100, len(W.VEHICLES))
sample_vehs = list(W.VEHICLES.values())[:sample_count]
for veh in sample_vehs:
    if veh.state == "end":
        route, timestamps = veh.traveled_route()
results["traveled_route_100_ms"] = (time.perf_counter() - t0) * 1000

results["total_ms"] = (time.perf_counter() - total_start) * 1000

print(json.dumps(results))
'''

    print("\n" + "=" * 70)
    print("测试 uxsimpp_extended 方案")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("错误:", result.stderr)
        return None

    try:
        results = json.loads(result.stdout.strip())
        return results
    except json.JSONDecodeError:
        print("输出解析失败:")
        print(result.stdout)
        print("STDERR:", result.stderr)
        return None


def format_comparison(uxsim_results, uxsimpp_results):
    """格式化对比结果"""
    print("\n" + "=" * 70)
    print("性能对比结果 (Sioux Falls 网络 - 单轮仿真)")
    print("=" * 70)

    print(f"\n网络规模:")
    print(f"  节点数: {uxsim_results['node_count']}")
    print(f"  链路数: {uxsim_results['link_count']}")
    print(f"  OD对数: {uxsim_results['od_pairs']}")
    print(f"  Vehicle对象数: {uxsim_results['vehicle_objects']}")
    print(f"  实际车辆数: {uxsim_results['actual_vehicles']}")

    print(f"\n{'环节':<25} {'UXsim+Patch':>15} {'uxsimpp_ext':>15} {'对比':>12}")
    print("-" * 70)

    metrics = [
        ("create_world_ms", "创建 World"),
        ("add_nodes_ms", "加载节点"),
        ("add_links_ms", "加载链路"),
        ("add_charging_links_ms", "创建充电链路"),
        ("compute_routes_ms", "计算路径"),
        ("add_demand_ms", "加载交通需求"),
        ("assign_routes_ms", "分配路径"),
        ("exec_simulation_ms", "执行仿真"),
        ("iterate_vehicles_ms", "遍历车辆状态"),
        ("traveled_route_100_ms", "traveled_route(100)"),
        ("total_ms", "总耗时"),
    ]

    for key, name in metrics:
        uxsim_val = uxsim_results.get(key, 0)
        uxsimpp_val = uxsimpp_results.get(key, 0)

        if uxsimpp_val > 0 and uxsim_val > 0:
            ratio = uxsim_val / uxsimpp_val
            if ratio >= 1:
                ratio_str = f"{ratio:.1f}x 快"
            else:
                ratio_str = f"{1/ratio:.1f}x 慢"
        else:
            ratio_str = "-"

        print(f"{name:<25} {uxsim_val:>12.2f}ms {uxsimpp_val:>12.2f}ms {ratio_str:>12}")

    print(f"\n完成率:")
    uxsim_rate = uxsim_results['completed_vehicles'] / uxsim_results['vehicle_objects'] * 100
    uxsimpp_rate = uxsimpp_results['completed_vehicles'] / uxsimpp_results['vehicle_objects'] * 100
    print(f"  UXsim+Patch: {uxsim_results['completed_vehicles']}/{uxsim_results['vehicle_objects']} ({uxsim_rate:.1f}%)")
    print(f"  uxsimpp_ext: {uxsimpp_results['completed_vehicles']}/{uxsimpp_results['vehicle_objects']} ({uxsimpp_rate:.1f}%)")

    print("\n" + "=" * 70)
    print("瓶颈分析")
    print("=" * 70)

    print(f"\nUXsim+Patch 各环节占比:")
    for key, name in metrics[:-1]:
        val = uxsim_results.get(key, 0)
        pct = val / uxsim_results['total_ms'] * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<20} {val:>8.1f}ms ({pct:>5.1f}%) {bar}")

    print(f"\nuxsimpp_extended 各环节占比:")
    for key, name in metrics[:-1]:
        val = uxsimpp_results.get(key, 0)
        pct = val / uxsimpp_results['total_ms'] * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<20} {val:>8.1f}ms ({pct:>5.1f}%) {bar}")


def main():
    print("\n" + "#" * 70)
    print("# Sioux Falls 网络单轮仿真性能对比")
    print("# UXsim 1.8.2 + Patch vs uxsimpp_extended")
    print("# 所有车辆使用预定路径")
    print("#" * 70)

    uxsim_results = run_uxsim_patch_test()
    uxsimpp_results = run_uxsimpp_extended_test()

    if uxsim_results and uxsimpp_results:
        format_comparison(uxsim_results, uxsimpp_results)
    else:
        print("\n测试失败，无法进行对比")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
