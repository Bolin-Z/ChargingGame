"""
诊断脚本：模拟完整的 UE-DTA 单轮循环，定位真正的性能瓶颈
"""
import os
import sys
import time
import json
import csv
import numpy as np
from collections import defaultdict

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from uxsimpp_extended.uxsimpp import World, Vehicle, Route


def load_network_and_demand():
    """加载网络和需求数据"""
    network_dir = os.path.join(project_root, "data", "siouxfalls")
    settings_path = os.path.join(network_dir, "siouxfalls_settings.json")

    with open(settings_path, "r") as f:
        settings = json.load(f)

    # 创建模板 World
    W = World(
        name="perf_test",
        deltan=settings["deltan"],
        tmax=settings["simulation_time"],
        print_mode=0,
        save_mode=0,
        show_mode=0,
        random_seed=42
    )

    # 加载节点
    node_path = os.path.join(network_dir, "siouxfalls_nodes.csv")
    with open(node_path, "r") as f:
        for r in csv.reader(f):
            if r[1] != "x":
                W.addNode(r[0], float(r[1]), float(r[2]))

    # 加载链路
    link_path = os.path.join(network_dir, "siouxfalls_links.csv")
    with open(link_path, "r") as f:
        for r in csv.reader(f):
            if r[3] != "length":
                W.addLink(
                    r[0], r[1], r[2],
                    length=float(r[3]),
                    free_flow_speed=float(r[4]),
                    jam_density=float(r[5]),
                    merge_priority=float(r[6]),
                    attribute={"charging_link": False}
                )

    # 创建充电链路
    for node in settings["charging_nodes"].keys():
        W.addLink(
            f"charging_{node}", node, node,
            length=settings["charging_link_length"],
            free_flow_speed=settings["charging_link_free_flow_speed"],
            attribute={"charging_link": True}
        )

    # 加载需求
    demand_path = os.path.join(network_dir, "siouxfalls_demand.csv")
    demand_multiplier = settings.get("demand_multiplier", 1.0)
    charging_car_rate = settings["charging_car_rate"]

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

    return W, settings


def create_simulation_world(template_W, settings):
    """从模板创建仿真世界"""
    W = World(
        name="sim",
        deltan=settings["deltan"],
        tmax=settings["simulation_time"],
        print_mode=0,
        save_mode=0,
        show_mode=0,
        random_seed=42
    )

    # 复制节点
    for node in template_W.NODES:
        W.addNode(node.name, node.x, node.y)

    # 复制链路
    for link in template_W.LINKS:
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
    for veh in template_W.VEHICLES.values():
        Vehicle(
            W, veh.orig.name, veh.dest.name,
            veh.departure_time,
            attribute=veh.attribute.copy() if hasattr(veh, 'attribute') else {}
        )

    return W


def simulate_route_choice_update(W, available_routes_map, current_routes):
    """模拟 __route_choice_update 的核心逻辑"""
    new_routes = {}

    # 构建车辆字典
    veh_dict = {veh.name: veh for veh in W.VEHICLES.values()}

    for veh_id, current_route in current_routes.items():
        veh = veh_dict.get(veh_id)
        if veh is None:
            new_routes[veh_id] = current_route
            continue

        if veh.state != "end":
            new_routes[veh_id] = current_route
            continue

        # 获取可用路径
        od_pair = (veh.orig.name, veh.dest.name)
        available_routes = available_routes_map.get(od_pair, [])

        best_cost = float('inf')
        best_route = current_route

        # 遍历所有可用路径
        for route_links in available_routes:
            # 创建 Route 对象
            route_obj = W.defRoute(route_links)
            # 计算成本
            alt_travel_time = route_obj.actual_travel_time(veh.departure_time)
            alt_cost = 0.005 * alt_travel_time + 0.5 * 50  # 简化成本

            if alt_cost < best_cost:
                best_cost = alt_cost
                best_route = route_links

        new_routes[veh_id] = best_route

    return new_routes


def main():
    print("=" * 70)
    print("诊断：完整 UE-DTA 单轮循环性能分析")
    print("=" * 70)

    # 1. 加载网络和需求
    print("\n1. 加载网络和需求...")
    t0 = time.perf_counter()
    template_W, settings = load_network_and_demand()
    t_load = time.perf_counter() - t0
    print(f"   耗时: {t_load*1000:.0f} ms")
    print(f"   车辆数: {len(list(template_W.VEHICLES.values()))}")

    # 2. 构建 OD-车辆映射
    print("\n2. 构建 OD-车辆映射...")
    t0 = time.perf_counter()
    od_to_vehids = defaultdict(list)
    for veh in template_W.VEHICLES.values():
        od_pair = (veh.orig.name, veh.dest.name)
        od_to_vehids[od_pair].append(veh.name)
    t_mapping = time.perf_counter() - t0
    print(f"   耗时: {t_mapping*1000:.0f} ms")
    print(f"   OD对数: {len(od_to_vehids)}")

    # 3. 简单路径分配（每个OD只用一条路径）
    print("\n3. 初始化路径分配...")
    t0 = time.perf_counter()
    current_routes = {}
    available_routes_map = {}

    # 简化：为每个OD创建一条简单路径
    for od_pair, veh_ids in od_to_vehids.items():
        # 这里简化处理，实际应该用 k-shortest paths
        route = [f"{od_pair[0]}-{od_pair[1]}"]  # 简化
        available_routes_map[od_pair] = [route]
        for veh_id in veh_ids:
            current_routes[veh_id] = route
    t_init_routes = time.perf_counter() - t0
    print(f"   耗时: {t_init_routes*1000:.0f} ms")

    # 4. 创建仿真世界
    print("\n4. 创建仿真世界...")
    t0 = time.perf_counter()
    W_sim = create_simulation_world(template_W, settings)
    t_create = time.perf_counter() - t0
    print(f"   耗时: {t_create*1000:.0f} ms")

    # 5. 执行仿真
    print("\n5. 执行仿真...")
    t0 = time.perf_counter()
    W_sim.exec_simulation()
    t_sim = time.perf_counter() - t0
    print(f"   耗时: {t_sim*1000:.0f} ms")

    # 6. 统计完成车辆
    print("\n6. 统计完成车辆...")
    t0 = time.perf_counter()
    completed = sum(1 for v in W_sim.VEHICLES.values() if v.state == "end")
    total = len(list(W_sim.VEHICLES.values()))
    t_stat = time.perf_counter() - t0
    print(f"   耗时: {t_stat*1000:.0f} ms")
    print(f"   完成: {completed}/{total} ({completed/total*100:.1f}%)")

    # 7. 测试 defRoute + actual_travel_time 的开销
    print("\n7. 测试 Route 创建和旅行时间计算开销...")

    # 获取一些真实的路径
    sample_routes = []
    for link in W_sim.LINKS[:10]:
        sample_routes.append([link.name])

    N = 10000
    t0 = time.perf_counter()
    for _ in range(N):
        for route_links in sample_routes:
            route_obj = W_sim.defRoute(route_links)
            _ = route_obj.actual_travel_time(100)
    t_route = time.perf_counter() - t0
    calls = N * len(sample_routes)
    print(f"   {calls} 次调用耗时: {t_route*1000:.0f} ms")
    print(f"   每次调用: {t_route/calls*1000:.4f} ms")

    # 8. 估算完整 route_choice_update 开销
    print("\n8. 估算完整 route_choice_update 开销...")
    n_completed = completed
    n_routes_per_od = 5  # 每个OD 5条路径
    total_route_calls = n_completed * n_routes_per_od
    estimated_time = total_route_calls * (t_route / calls)
    print(f"   完成车辆: {n_completed}")
    print(f"   每车辆检查路径数: {n_routes_per_od}")
    print(f"   总 Route 调用次数: {total_route_calls}")
    print(f"   预计 route_choice_update 耗时: {estimated_time*1000:.0f} ms")

    # 9. 总结
    print("\n" + "=" * 70)
    print("单轮 UE-DTA 各环节耗时估算:")
    print("=" * 70)
    print(f"  创建仿真世界:      {t_create*1000:>8.0f} ms")
    print(f"  执行仿真:          {t_sim*1000:>8.0f} ms")
    print(f"  路径选择更新:      {estimated_time*1000:>8.0f} ms (估算)")
    print(f"  -" * 35)
    total_estimated = t_create + t_sim + estimated_time
    print(f"  预计单轮总耗时:    {total_estimated*1000:>8.0f} ms")
    print(f"\n  实际观察到的耗时:  ~156,000 ms (每轮)")
    print(f"  差异倍数: {156000 / (total_estimated*1000):.0f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
