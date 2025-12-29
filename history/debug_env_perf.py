"""
诊断脚本：定位 EVCSChargingGameEnv 的性能瓶颈
"""
import os
import sys
import time
import numpy as np

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def test_env_performance():
    print("=" * 70)
    print("诊断：EVCSChargingGameEnv 性能瓶颈定位")
    print("=" * 70)

    # 导入环境
    from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv

    # 创建环境
    print("\n1. 创建环境...")
    start = time.perf_counter()
    env = EVCSChargingGameEnv(
        network_dir=os.path.join(project_root, "data", "siouxfalls"),
        network_name="siouxfalls",
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    print(f"   环境创建耗时: {(time.perf_counter() - start)*1000:.0f} ms")

    # 重置环境
    print("\n2. 重置环境...")
    start = time.perf_counter()
    env.reset(seed=42)
    print(f"   环境重置耗时: {(time.perf_counter() - start)*1000:.0f} ms")

    # 准备动作
    actions = {agent_id: np.array([0.5] * env.n_periods, dtype=np.float32)
               for agent_id in env.agents}

    # 执行一步（这会触发 UE-DTA 循环）
    print("\n3. 执行 step（包含 UE-DTA 循环）...")
    print("   [这里会显示 UE-DTA 进度条]\n")

    start = time.perf_counter()
    observations, rewards, terminations, truncations, infos = env.step(actions)
    step_time = time.perf_counter() - start

    print(f"\n   step 总耗时: {step_time:.1f} 秒")

    # 从 infos 中获取 UE-DTA 统计信息
    if '__all__' in infos and 'ue_stats' in infos['__all__']:
        stats = infos['__all__']['ue_stats']
        print(f"   完成车辆: {stats.get('completed_total_vehicles', 'N/A')}")
        print(f"   总车辆: {stats.get('total_vehicles', 'N/A')}")

    print("\n" + "=" * 70)


def test_route_cost_breakdown():
    """
    分解 __estimate_route_cost 的各部分耗时
    """
    print("\n" + "=" * 70)
    print("诊断：路径成本计算分解测试")
    print("=" * 70)

    from uxsimpp_extended.uxsimpp import World, Vehicle, Route
    import json

    # 加载配置
    network_dir = os.path.join(project_root, "data", "siouxfalls")
    settings_path = os.path.join(network_dir, "siouxfalls_settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    # 创建 World
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
    import csv
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

    # 添加一些车辆
    for i in range(100):
        Vehicle(W, "1", "20", i * 50, attribute={"charging_car": True})

    print(f"\n网络: {len(W.NODES)} 节点, {len(W.LINKS)} 链路")

    # 执行仿真
    print("执行仿真...")
    W.exec_simulation()
    print("仿真完成")

    # 创建一个测试路径
    route_links = ["1-3", "3-12", "charging_12"]
    route = W.defRoute(route_links)

    print(f"\n测试路径: {route_links}")
    print(f"路径链路数: {len(route.links)}")

    # 检查缓存
    for link in route.links:
        has_cache = hasattr(link, '_traveltime_cache')
        print(f"  {link.name}: 有缓存={has_cache}")

    # 性能测试：不同操作的耗时
    N = 10000
    departure_time = 100.0

    print(f"\n性能测试 ({N} 次迭代):")

    # 1. 纯 actual_travel_time 调用
    start = time.perf_counter()
    for _ in range(N):
        for link in route.links:
            _ = link.actual_travel_time(departure_time)
    t1 = time.perf_counter() - start
    print(f"  1. link.actual_travel_time(): {t1*1000:.2f} ms")

    # 2. 加上 link.attribute 访问
    start = time.perf_counter()
    for _ in range(N):
        for link in route.links:
            _ = link.actual_travel_time(departure_time)
            _ = link.attribute["charging_link"]
    t2 = time.perf_counter() - start
    print(f"  2. + link.attribute['charging_link']: {t2*1000:.2f} ms (差值: {(t2-t1)*1000:.2f} ms)")

    # 3. 加上 link.name 访问和字符串操作
    start = time.perf_counter()
    for _ in range(N):
        for link in route.links:
            _ = link.actual_travel_time(departure_time)
            if link.attribute["charging_link"]:
                if link.name.startswith("charging_"):
                    _ = link.name.split("charging_")[1]
    t3 = time.perf_counter() - start
    print(f"  3. + link.name 字符串操作: {t3*1000:.2f} ms (差值: {(t3-t2)*1000:.2f} ms)")

    # 4. 完整模拟 __estimate_route_cost 逻辑
    price_history = np.array([[[0.5] * 8] * 4])  # 模拟价格历史
    time_value_coefficient = 0.005
    charging_demand = 50

    start = time.perf_counter()
    for _ in range(N):
        current_time = departure_time
        charging_cost = 0.0
        for link in route.links:
            link_travel_time = link.actual_travel_time(current_time)
            if link.attribute["charging_link"]:
                if link.name.startswith("charging_"):
                    charging_node = link.name.split("charging_")[1]
                    # 模拟 __get_price
                    period = int(current_time // 1200) % 8
                    charging_price = 0.5  # 简化
                    charging_cost = charging_price * charging_demand
            current_time += link_travel_time
        total_travel_time = current_time - departure_time
        time_cost = time_value_coefficient * total_travel_time
        total_cost = time_cost + charging_cost
    t4 = time.perf_counter() - start
    print(f"  4. 完整成本计算逻辑: {t4*1000:.2f} ms (差值: {(t4-t3)*1000:.2f} ms)")

    # 5. 对比 Route.actual_travel_time
    start = time.perf_counter()
    for _ in range(N):
        _ = route.actual_travel_time(departure_time)
    t5 = time.perf_counter() - start
    print(f"  5. Route.actual_travel_time(): {t5*1000:.2f} ms")

    print(f"\n结论:")
    print(f"  逐链路遍历 vs Route方法: {t4/t5:.1f}x 慢")
    print(f"  每次成本计算耗时: {t4/N*1000:.3f} ms")

    # 估算大规模场景
    n_vehicles = 6601
    n_routes_per_vehicle = 5
    total_calls = n_vehicles * n_routes_per_vehicle
    estimated_time = total_calls * (t4/N)
    print(f"\n大规模场景估算:")
    print(f"  {n_vehicles} 车辆 × {n_routes_per_vehicle} 路径 = {total_calls} 次成本计算")
    print(f"  预计耗时: {estimated_time:.1f} 秒")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 先运行分解测试
    test_route_cost_breakdown()

    # 再运行完整环境测试（可选，耗时较长）
    # test_env_performance()
