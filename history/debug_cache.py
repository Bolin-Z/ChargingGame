"""
诊断脚本：验证 uxsimpp_extended 的 traveltime 缓存是否正确工作
"""
import os
import sys

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from uxsimpp_extended.uxsimpp import World, Vehicle, Route
import time


def test_cache():
    print("=" * 60)
    print("诊断：traveltime 缓存测试")
    print("=" * 60)

    # 创建简单的测试网络
    W = World(
        name="cache_test",
        deltan=5,
        tmax=1000,
        print_mode=0,
        save_mode=0,
        show_mode=0,
        random_seed=42
    )

    # 创建节点
    W.addNode("A", 0, 0)
    W.addNode("B", 1000, 0)
    W.addNode("C", 2000, 0)

    # 创建链路
    W.addLink("A-B", "A", "B", length=1000, free_flow_speed=20, attribute={"test": True})
    W.addLink("B-C", "B", "C", length=1000, free_flow_speed=20, attribute={"test": True})

    # 创建一些车辆
    for i in range(10):
        Vehicle(W, "A", "C", i * 50)

    print(f"\n1. 仿真前检查:")
    link_ab = W.get_link("A-B")
    print(f"   link_ab id: {id(link_ab)}")
    print(f"   link_ab 有 _traveltime_cache: {hasattr(link_ab, '_traveltime_cache')}")

    # 执行仿真
    print(f"\n2. 执行仿真...")
    W.exec_simulation()
    print(f"   仿真完成")

    # 仿真后检查
    print(f"\n3. 仿真后检查:")

    # 检查 W.LINKS 中的对象
    links_from_LINKS = W.LINKS
    link_ab_from_LINKS = [l for l in links_from_LINKS if l.name == "A-B"][0]
    print(f"   W.LINKS 中的 A-B link id: {id(link_ab_from_LINKS)}")
    print(f"   有 _traveltime_cache: {hasattr(link_ab_from_LINKS, '_traveltime_cache')}")
    if hasattr(link_ab_from_LINKS, '_traveltime_cache'):
        print(f"   缓存长度: {len(link_ab_from_LINKS._traveltime_cache)}")

    # 检查 W.get_link 返回的对象
    link_ab_from_get = W.get_link("A-B")
    print(f"\n   W.get_link('A-B') id: {id(link_ab_from_get)}")
    print(f"   有 _traveltime_cache: {hasattr(link_ab_from_get, '_traveltime_cache')}")

    # 检查对象是否相同
    print(f"\n   link_ab_from_LINKS is link_ab_from_get: {link_ab_from_LINKS is link_ab_from_get}")
    print(f"   link_ab is link_ab_from_get: {link_ab is link_ab_from_get}")

    # 创建 Route 并检查其 links
    print(f"\n4. Route 检查:")
    route = W.defRoute(["A-B", "B-C"])
    route_link_ab = route.links[0]
    print(f"   Route.links[0] id: {id(route_link_ab)}")
    print(f"   有 _traveltime_cache: {hasattr(route_link_ab, '_traveltime_cache')}")
    print(f"   route_link_ab is link_ab_from_get: {route_link_ab is link_ab_from_get}")

    # 性能测试
    print(f"\n5. 性能测试:")

    # 测试有缓存时的性能
    if hasattr(route_link_ab, '_traveltime_cache'):
        start = time.perf_counter()
        for _ in range(10000):
            _ = route_link_ab.actual_travel_time(100)
        elapsed_cached = time.perf_counter() - start
        print(f"   有缓存: 10000次调用耗时 {elapsed_cached*1000:.2f} ms")

    # 测试无缓存时的性能（删除缓存）
    if hasattr(route_link_ab, '_traveltime_cache'):
        cache_backup = route_link_ab._traveltime_cache
        delattr(route_link_ab, '_traveltime_cache')

        start = time.perf_counter()
        for _ in range(10000):
            _ = route_link_ab.actual_travel_time(100)
        elapsed_no_cache = time.perf_counter() - start
        print(f"   无缓存: 10000次调用耗时 {elapsed_no_cache*1000:.2f} ms")

        # 恢复缓存
        route_link_ab._traveltime_cache = cache_backup

        if elapsed_no_cache > 0:
            print(f"   加速比: {elapsed_no_cache / elapsed_cached:.1f}x")
    else:
        print(f"   [警告] 没有找到缓存，无法进行性能对比")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_cache()
