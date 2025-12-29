"""
actual_travel_time 内部细分性能分析

分析 Link.actual_travel_time 各个操作的耗时
"""

import time
import os
import sys
import json

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

NETWORK_DIR = os.path.join(project_root, "data", "siouxfalls")
NETWORK_NAME = "siouxfalls"

def main():
    from uxsimpp_extended.uxsimpp import World, Vehicle, Route

    # 加载配置
    settings_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    # 创建仿真世界
    W = World(
        name=settings["network_name"],
        deltan=settings["deltan"],
        tmax=settings["simulation_time"],
        print_mode=0, save_mode=0, show_mode=0,
        random_seed=42
    )

    # 加载网络
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

    # 添加一些车辆
    demand_path = os.path.join(NETWORK_DIR, f"{NETWORK_NAME}_demand.csv")
    with open(demand_path, "r") as f:
        for r in csv.reader(f):
            if r[2] != "start_t":
                W.adddemand(r[0], r[1], float(r[2]), float(r[3]),
                           float(r[4]) * 0.1, attribute={})

    # 执行仿真
    print("执行仿真...")
    W.exec_simulation()
    print("仿真完成")

    # 获取一个链路用于测试
    link = list(W.LINKS)[0]
    print(f"\n测试链路: {link.name}")
    print(f"traveltime_real 长度: {len(link.traveltime_real)}")

    N = 100000  # 测试次数

    print(f"\n{'=' * 70}")
    print(f"各操作耗时测试 (N={N})")
    print(f"{'=' * 70}")

    # 测试 1: 访问 link.W
    start = time.perf_counter()
    for _ in range(N):
        w = link.W
    t1 = (time.perf_counter() - start) * 1000
    print(f"1. link.W 访问:                    {t1:8.2f} ms ({t1/N*1000:.4f} μs/次)")

    # 测试 2: 访问 link.W.delta_t
    start = time.perf_counter()
    for _ in range(N):
        dt = link.W.delta_t
    t2 = (time.perf_counter() - start) * 1000
    print(f"2. link.W.delta_t 访问:            {t2:8.2f} ms ({t2/N*1000:.4f} μs/次)")

    # 测试 3: 访问 link.traveltime_real (整个 vector)
    start = time.perf_counter()
    for _ in range(N):
        tr = link.traveltime_real
    t3 = (time.perf_counter() - start) * 1000
    print(f"3. link.traveltime_real 访问:      {t3:8.2f} ms ({t3/N*1000:.4f} μs/次)")

    # 测试 4: len(link.traveltime_real)
    start = time.perf_counter()
    for _ in range(N):
        l = len(link.traveltime_real)
    t4 = (time.perf_counter() - start) * 1000
    print(f"4. len(link.traveltime_real):      {t4:8.2f} ms ({t4/N*1000:.4f} μs/次)")

    # 测试 5: link.traveltime_real[100] (索引访问)
    start = time.perf_counter()
    for _ in range(N):
        v = link.traveltime_real[100]
    t5 = (time.perf_counter() - start) * 1000
    print(f"5. link.traveltime_real[100]:      {t5:8.2f} ms ({t5/N*1000:.4f} μs/次)")

    # 测试 6: link.traveltime_real[-1] (末尾访问)
    start = time.perf_counter()
    for _ in range(N):
        v = link.traveltime_real[-1]
    t6 = (time.perf_counter() - start) * 1000
    print(f"6. link.traveltime_real[-1]:       {t6:8.2f} ms ({t6/N*1000:.4f} μs/次)")

    # 测试 7: 完整的 actual_travel_time 函数
    t = 1000.0
    start = time.perf_counter()
    for _ in range(N):
        tt = link.actual_travel_time(t)
    t7 = (time.perf_counter() - start) * 1000
    print(f"7. link.actual_travel_time(t):     {t7:8.2f} ms ({t7/N*1000:.4f} μs/次)")

    # 测试 8: 缓存 delta_t 后的访问
    delta_t = link.W.delta_t
    start = time.perf_counter()
    for _ in range(N):
        tt_idx = int(t // delta_t)
        v = link.traveltime_real[tt_idx]
    t8 = (time.perf_counter() - start) * 1000
    print(f"8. 缓存 delta_t 后索引访问:        {t8:8.2f} ms ({t8/N*1000:.4f} μs/次)")

    # 测试 9: 先获取整个数组再索引
    start = time.perf_counter()
    for _ in range(N):
        tr = link.traveltime_real
        tt_idx = int(t // delta_t)
        v = tr[tt_idx]
    t9 = (time.perf_counter() - start) * 1000
    print(f"9. 先获取数组再索引:              {t9:8.2f} ms ({t9/N*1000:.4f} μs/次)")

    # 测试 10: 预先缓存数组后多次索引
    tr_cached = list(link.traveltime_real)
    start = time.perf_counter()
    for _ in range(N):
        tt_idx = int(t // delta_t)
        v = tr_cached[tt_idx]
    t10 = (time.perf_counter() - start) * 1000
    print(f"10. 预缓存数组后索引:              {t10:8.2f} ms ({t10/N*1000:.4f} μs/次)")

    print(f"\n{'=' * 70}")
    print("分析结论")
    print(f"{'=' * 70}")

    # 计算瓶颈
    if t3 > t1 * 5:
        print(f"⚠️  traveltime_real 访问是主要瓶颈 ({t3/t1:.1f}x 于 link.W)")
    if t5 > t10 * 5:
        print(f"⚠️  直接索引 vs 缓存后索引: {t5/t10:.1f}x 性能差距")

    # Route.actual_travel_time 模拟测试
    print(f"\n{'=' * 70}")
    print("Route.actual_travel_time 模拟测试")
    print(f"{'=' * 70}")

    # 构建一个多链路路径
    links = list(W.LINKS)[:5]  # 取5个链路
    print(f"测试路径包含 {len(links)} 个链路")

    N2 = 10000

    # 当前实现方式
    start = time.perf_counter()
    for _ in range(N2):
        tt = 0
        current_t = 1000.0
        for l in links:
            link_tt = l.actual_travel_time(current_t)
            tt += link_tt
            current_t += link_tt
    t_current = (time.perf_counter() - start) * 1000
    print(f"当前实现:                          {t_current:8.2f} ms ({t_current/N2:.4f} ms/次)")

    # 优化方式1: 预缓存 delta_t 和 traveltime_real
    link_caches = []
    for l in links:
        link_caches.append({
            'delta_t': l.W.delta_t,
            'traveltime_real': list(l.traveltime_real),
            'max_idx': len(l.traveltime_real) - 1
        })

    start = time.perf_counter()
    for _ in range(N2):
        tt = 0
        current_t = 1000.0
        for cache in link_caches:
            tt_idx = int(current_t // cache['delta_t'])
            if tt_idx > cache['max_idx']:
                tt_idx = cache['max_idx']
            elif tt_idx < 0:
                tt_idx = 0
            link_tt = cache['traveltime_real'][tt_idx]
            tt += link_tt
            current_t += link_tt
    t_optimized = (time.perf_counter() - start) * 1000
    print(f"预缓存优化:                        {t_optimized:8.2f} ms ({t_optimized/N2:.4f} ms/次)")

    print(f"\n性能提升: {t_current/t_optimized:.1f}x")


if __name__ == "__main__":
    main()
