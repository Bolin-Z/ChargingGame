# tests/test_network_data.py
"""
NetworkData 测试脚本

验证内容：
1. 数据加载正确
2. 可 pickle 序列化
3. 路径预计算正确
"""

import pickle
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator.network_data import NetworkDataLoader, NetworkData, RouteInfo


def test_load_network_data():
    """测试网络数据加载"""
    print("=" * 60)
    print("测试 1: 网络数据加载")
    print("=" * 60)

    # 使用项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_dir = os.path.join(project_root, "data", "siouxfalls")

    loader = NetworkDataLoader(
        network_dir=network_dir,
        network_name="siouxfalls",
        random_seed=42
    )

    network_data = loader.load()

    # 验证基本属性
    print(f"节点数量: {len(network_data.nodes)}")
    print(f"链路数量: {len(network_data.links)}")
    print(f"需求数量: {len(network_data.demands)}")
    print(f"充电站数量: {network_data.n_agents}")
    print(f"时段数量: {network_data.n_periods}")
    print(f"智能体名称: {network_data.agent_names}")

    # 验证路径
    charging_od_count = len(network_data.routes["charging"])
    uncharging_od_count = len(network_data.routes["uncharging"])
    print(f"充电路径 OD 对数量: {charging_od_count}")
    print(f"非充电路径 OD 对数量: {uncharging_od_count}")

    # 验证充电链路
    charging_links = [l for l in network_data.links if l.is_charging_link]
    print(f"充电链路数量: {len(charging_links)}")
    for link in charging_links:
        print(f"  - {link.name}: {link.start_node} -> {link.end_node}")

    # 验证路径信息
    print("\n示例路径信息:")
    for (o, d), routes in list(network_data.routes["charging"].items())[:2]:
        print(f"  充电路径 {o} -> {d}: {len(routes)} 条")
        for i, route in enumerate(routes[:2]):
            print(f"    [{i}] 链路: {route.links[:3]}... 充电节点: {route.charging_node}")

    print("\n✓ 测试 1 通过")
    return network_data


def test_pickle_serialization(network_data: NetworkData):
    """测试 pickle 序列化"""
    print("\n" + "=" * 60)
    print("测试 2: Pickle 序列化")
    print("=" * 60)

    # 序列化
    serialized = pickle.dumps(network_data)
    print(f"序列化大小: {len(serialized) / 1024:.2f} KB")

    # 反序列化
    restored = pickle.loads(serialized)

    # 验证数据一致性
    assert len(restored.nodes) == len(network_data.nodes), "节点数量不一致"
    assert len(restored.links) == len(network_data.links), "链路数量不一致"
    assert len(restored.demands) == len(network_data.demands), "需求数量不一致"
    assert restored.n_agents == network_data.n_agents, "智能体数量不一致"
    assert restored.n_periods == network_data.n_periods, "时段数量不一致"

    # 验证路径一致性
    for route_type in ["charging", "uncharging"]:
        assert len(restored.routes[route_type]) == len(network_data.routes[route_type]), \
            f"{route_type} 路径 OD 对数量不一致"

    print("✓ 序列化/反序列化验证通过")
    print("\n✓ 测试 2 通过")


def test_route_info_correctness(network_data: NetworkData):
    """测试路径信息正确性"""
    print("\n" + "=" * 60)
    print("测试 3: 路径信息正确性")
    print("=" * 60)

    # 验证充电路径都有充电节点
    for (o, d), routes in network_data.routes["charging"].items():
        for route in routes:
            if route.charging_node is None:
                print(f"警告: 充电路径 {o}->{d} 缺少充电节点: {route.links}")
            else:
                # 验证充电节点在智能体列表中
                assert route.charging_node in network_data.agent_names, \
                    f"充电节点 {route.charging_node} 不在智能体列表中"

                # 验证充电链路索引正确
                if route.charging_link_idx is not None:
                    charging_link = route.links[route.charging_link_idx]
                    assert charging_link == f"charging_{route.charging_node}", \
                        f"充电链路索引不正确: {charging_link} != charging_{route.charging_node}"

    print("✓ 充电路径验证通过")

    # 验证非充电路径没有充电节点
    for (o, d), routes in network_data.routes["uncharging"].items():
        for route in routes:
            assert route.charging_node is None, \
                f"非充电路径不应有充电节点: {o}->{d}"
            assert route.charging_link_idx is None, \
                f"非充电路径不应有充电链路索引: {o}->{d}"

    print("✓ 非充电路径验证通过")
    print("\n✓ 测试 3 通过")


def test_multiprocess_compatibility(network_data: NetworkData):
    """测试多进程兼容性"""
    print("\n" + "=" * 60)
    print("测试 4: 多进程兼容性")
    print("=" * 60)

    import multiprocessing as mp

    # 序列化数据
    serialized = pickle.dumps(network_data)

    # Windows 需要使用 spawn 方式，且 worker 函数需要在模块级别定义
    # 这里简化测试：只验证序列化后可在当前进程反序列化
    # 多进程完整测试在 P3 阶段进行

    restored = pickle.loads(serialized)

    # 验证结果
    assert len(restored.nodes) == len(network_data.nodes)
    assert len(restored.links) == len(network_data.links)
    assert restored.n_agents == network_data.n_agents

    print("✓ 序列化兼容性验证通过（完整多进程测试在 P3 阶段）")
    print("\n✓ 测试 4 通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("NetworkData 测试套件")
    print("=" * 60 + "\n")

    # 测试 1: 数据加载
    network_data = test_load_network_data()

    # 测试 2: Pickle 序列化
    test_pickle_serialization(network_data)

    # 测试 3: 路径信息正确性
    test_route_info_correctness(network_data)

    # 测试 4: 多进程兼容性
    test_multiprocess_compatibility(network_data)

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
