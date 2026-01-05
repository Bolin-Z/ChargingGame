"""
诊断节点 8 的路径问题

分析：
1. 节点 8 的网络位置和连通性
2. 8->16 等高 Gap OD 对的最短路径
3. 是否存在路径选择瓶颈
"""

import os
import sys
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_network_graph(network_dir, network_name):
    """从 CSV 文件构建网络图"""
    links_file = os.path.join(network_dir, f"{network_name}_links.csv")
    df = pd.read_csv(links_file)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            int(row['start']),
            int(row['end']),
            name=str(row['name']),
            length=row['length'],
            u=row['u']
        )
    return G


def analyze_node8_connectivity():
    """分析节点 8 的连通性"""

    print("=" * 60)
    print("节点 8 网络拓扑分析")
    print("=" * 60)

    network_dir = "data/berlin_friedrichshain"
    network_name = "berlin_friedrichshain"

    G = build_network_graph(network_dir, network_name)

    # 节点 8 的出边
    print("\n节点 8 的出边链路:")
    print("-" * 40)
    out_edges = list(G.out_edges(8, data=True))
    for u, v, data in out_edges:
        length = data.get('length', 0)
        speed = data.get('u', 0)
        print(f"  8 -> {v}: 长度={length:.1f}m, 速度={speed:.1f}m/s")

    # 节点 8 的入边
    print("\n节点 8 的入边链路:")
    print("-" * 40)
    in_edges = list(G.in_edges(8, data=True))
    for u, v, data in in_edges:
        length = data.get('length', 0)
        speed = data.get('u', 0)
        print(f"  {u} -> 8: 长度={length:.1f}m, 速度={speed:.1f}m/s")

    # 分析瓶颈：节点 8 的邻居节点的出度
    print("\n" + "=" * 60)
    print("节点 8 邻居的连通性分析")
    print("=" * 60)

    neighbors = list(G.successors(8))
    for neighbor in neighbors:
        neighbor_out = list(G.successors(neighbor))
        neighbor_out_filtered = [n for n in neighbor_out if n != 8]
        print(f"\n节点 {neighbor} 的出边（排除返回8）:")
        for next_node in neighbor_out_filtered:
            print(f"  8 -> {neighbor} -> {next_node}")

    # 高 Gap OD 对
    high_gap_ods = [
        (8, 16),  # GM=53%
        (8, 6),   # GM=34%
        (8, 12),  # GM=30%
        (8, 23),  # GM=30%
        (8, 5),   # GM=28%
    ]

    print("\n" + "=" * 60)
    print("高 Gap OD 对的最短路径分析")
    print("=" * 60)

    for origin, dest in high_gap_ods:
        print(f"\n--- OD: {origin} -> {dest} ---")
        try:
            path = nx.shortest_path(G, origin, dest, weight='length')
            length = nx.shortest_path_length(G, origin, dest, weight='length')
            path_str = ' -> '.join(map(str, path))
            print(f"最短路径: {path_str}")
            print(f"路径长度: {length:.0f}m, 跳数: {len(path)-1}")
            print(f"第一跳: 8 -> {path[1]}")
        except nx.NetworkXNoPath:
            print("  无路径!")

    # 统计从节点 8 出发的所有最短路径的第一跳分布
    print("\n" + "=" * 60)
    print("节点 8 出发的最短路径第一跳分布")
    print("=" * 60)

    first_hop_count = {n: 0 for n in neighbors}
    total_dests = 0

    for dest in G.nodes():
        if dest == 8:
            continue
        try:
            path = nx.shortest_path(G, 8, dest, weight='length')
            if len(path) > 1:
                first_hop = path[1]
                if first_hop in first_hop_count:
                    first_hop_count[first_hop] += 1
                    total_dests += 1
        except nx.NetworkXNoPath:
            pass

    print(f"\n从节点 8 可达的目的地总数: {total_dests}")
    print("\n第一跳分布:")
    for hop, count in sorted(first_hop_count.items(), key=lambda x: -x[1]):
        pct = count / total_dests * 100 if total_dests > 0 else 0
        print(f"  8 -> {hop}: {count} 个目的地 ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    print("""
关键发现：
1. 节点 8 只有 4 条出边，形成网络瓶颈
2. 从节点 8 出发的所有路径都必须经过这 4 条出边之一
3. 当某条出边拥堵时，替代路径有限
4. 预计算的 10 条路径可能都集中在少数几条出边上

建议：
1. 增加 routes_per_od 可能无效（受拓扑限制）
2. 考虑对边缘节点特殊处理
3. 或接受这是结构性限制
""")


if __name__ == "__main__":
    analyze_node8_connectivity()
