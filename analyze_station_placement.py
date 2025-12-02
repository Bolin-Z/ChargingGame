"""
充电站选址分析脚本

功能：
1. 运行纯交通仿真，统计链路和节点流量
2. 基于链路流量选取Top 20链路端点
3. 基于节点流量选取Top 20节点
4. 可视化对比当前选址与新选址方案
"""

import csv
import json
import os
from collections import defaultdict
from uxsim import World
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def run_traffic_simulation(network_dir: str, network_name: str, demand_multiplier: float = 3.5):
    """
    运行纯交通仿真，返回链路流量和节点流量统计

    Args:
        network_dir: 网络数据目录
        network_name: 网络名称（文件名前缀）
        demand_multiplier: 需求放大系数

    Returns:
        W: UXSim World对象
        link_flows: {link_name: cumulative_flow}
        node_flows: {node_name: total_flow (in + out)}
    """
    # 加载配置
    settings_path = os.path.join(network_dir, f"{network_name}_settings.json")
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    print(f"=== 运行纯交通仿真 ===")
    print(f"网络: {settings['network_name']}")
    print(f"需求放大系数: {demand_multiplier}x")

    # 创建UXSim World
    W = World(
        name="station_placement_analysis",
        deltan=settings["deltan"],
        tmax=settings["simulation_time"],
        print_mode=1,
        save_mode=0,
        show_mode=0,
        random_seed=42
    )

    # 加载节点
    nodes = {}
    nodes_path = os.path.join(network_dir, f"{network_name}_nodes.csv")
    with open(nodes_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_name = row["name"]
            x, y = float(row["x"]), float(row["y"])
            W.addNode(node_name, x, y)
            nodes[node_name] = (x, y)
    print(f"加载 {len(nodes)} 个节点")

    # 加载链路
    links_info = {}  # {link_name: (start, end)}
    links_path = os.path.join(network_dir, f"{network_name}_links.csv")
    with open(links_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_name = row["name"]
            start, end = row["start"], row["end"]
            links_info[link_name] = (start, end)

            W.addLink(
                name=link_name,
                start_node=start,
                end_node=end,
                length=float(row["length"]),
                free_flow_speed=float(row["u"]),
                jam_density=float(row["kappa"]),
                number_of_lanes=1
            )
    print(f"加载 {len(links_info)} 条链路")

    # 加载需求
    demand_count = 0
    demand_path = os.path.join(network_dir, f"{network_name}_demand.csv")
    with open(demand_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = float(row["q"])
            if q > 0:
                W.adddemand(
                    row["orig"], row["dest"],
                    float(row["start_t"]), float(row["end_t"]),
                    q * demand_multiplier
                )
                demand_count += 1
    print(f"加载 {demand_count} 个OD需求")

    # 运行仿真
    print("\n运行仿真中...")
    W.exec_simulation()
    W.analyzer.print_simple_stats()

    # 统计链路流量
    print("\n=== 统计流量 ===")
    link_flows = {}
    for link in W.LINKS:
        # cum_departure是时间序列列表，取最后一个值表示总通过车辆数(platoon数)
        flow = link.cum_departure[-1] * W.DELTAN
        link_flows[link.name] = flow

    # 统计节点流量（流入 + 流出）
    node_inflow = defaultdict(float)
    node_outflow = defaultdict(float)

    for link_name, flow in link_flows.items():
        start, end = links_info[link_name]
        node_outflow[start] += flow  # 从该节点流出
        node_inflow[end] += flow      # 流入该节点

    node_flows = {}
    for node_name in nodes.keys():
        node_flows[node_name] = node_inflow[node_name] + node_outflow[node_name]

    return W, link_flows, node_flows, nodes, links_info


def select_top_stations(link_flows, node_flows, links_info, n_stations=20):
    """
    基于流量选取充电站位置

    Args:
        link_flows: 链路流量字典
        node_flows: 节点流量字典
        links_info: 链路信息 {link_name: (start, end)}
        n_stations: 选取数量

    Returns:
        link_based_nodes: 基于链路流量选取的节点列表
        node_based_nodes: 基于节点流量选取的节点列表
    """
    # 方案A: 基于链路流量
    sorted_links = sorted(link_flows.items(), key=lambda x: x[1], reverse=True)
    link_based_nodes = []
    link_based_details = []

    for link_name, flow in sorted_links:
        start, end = links_info[link_name]
        # 选择链路终点作为充电站位置
        if end not in link_based_nodes:
            link_based_nodes.append(end)
            link_based_details.append((link_name, end, flow))
        if len(link_based_nodes) >= n_stations:
            break

    # 方案B: 基于节点流量
    sorted_nodes = sorted(node_flows.items(), key=lambda x: x[1], reverse=True)
    node_based_nodes = [node for node, _ in sorted_nodes[:n_stations]]
    node_based_details = sorted_nodes[:n_stations]

    return link_based_nodes, node_based_nodes, link_based_details, node_based_details


def visualize_comparison(nodes, links_info, current_stations, link_based_nodes,
                         node_based_nodes, node_flows, output_path):
    """
    可视化对比不同选址方案
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # 准备节点坐标和流量
    x_coords = [nodes[n][0] for n in nodes]
    y_coords = [nodes[n][1] for n in nodes]
    flows = [node_flows.get(n, 0) for n in nodes]
    max_flow = max(flows) if flows else 1

    # 节点大小映射（基于流量）
    min_size, max_size = 10, 200
    sizes = [min_size + (f / max_flow) * (max_size - min_size) for f in flows]

    for ax_idx, (ax, title, selected_nodes) in enumerate([
        (axes[0], "当前充电站选址\n(基于度数)", current_stations),
        (axes[1], "方案A: 基于链路流量\n(Top 20链路终点)", link_based_nodes),
        (axes[2], "方案B: 基于节点流量\n(Top 20节点)", node_based_nodes)
    ]):
        # 绘制链路
        for link_name, (start, end) in links_info.items():
            if start in nodes and end in nodes:
                x1, y1 = nodes[start]
                x2, y2 = nodes[end]
                ax.plot([x1, x2], [y1, y2], 'lightgray', linewidth=0.3, zorder=1)

        # 绘制所有节点（大小反映流量）
        node_list = list(nodes.keys())
        for i, node in enumerate(node_list):
            x, y = nodes[node]
            size = sizes[i]

            if node in selected_nodes and node in current_stations:
                # 重叠节点 - 红色五角星
                ax.scatter(x, y, s=size*1.5, c='red', marker='*',
                          edgecolors='darkred', linewidths=0.5, zorder=4)
            elif node in selected_nodes:
                # 新选节点 - 绿色圆形
                ax.scatter(x, y, s=size, c='limegreen', marker='o',
                          edgecolors='darkgreen', linewidths=0.5, zorder=3)
            elif node in current_stations:
                # 仅当前选址 - 蓝色方形
                ax.scatter(x, y, s=size, c='dodgerblue', marker='s',
                          edgecolors='darkblue', linewidths=0.5, zorder=3)
            else:
                # 普通节点 - 灰色
                ax.scatter(x, y, s=size*0.5, c='silver', marker='o',
                          edgecolors='gray', linewidths=0.2, zorder=2, alpha=0.5)

        # 计算重叠数
        overlap = set(selected_nodes) & set(current_stations)
        new_nodes = set(selected_nodes) - set(current_stations)
        removed = set(current_stations) - set(selected_nodes)

        ax.set_title(f"{title}\n重叠: {len(overlap)}, 新增: {len(new_nodes)}, 移除: {len(removed)}",
                    fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='red', edgecolor='darkred', label='重叠节点 (保留)'),
        mpatches.Patch(facecolor='limegreen', edgecolor='darkgreen', label='新选节点'),
        mpatches.Patch(facecolor='dodgerblue', edgecolor='darkblue', label='原节点 (移除)'),
        mpatches.Patch(facecolor='silver', edgecolor='gray', label='普通节点', alpha=0.5),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Berlin Friedrichshain 充电站选址方案对比\n(节点大小反映流量)', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存: {output_path}")


def generate_report(current_stations, link_based_nodes, node_based_nodes,
                    link_based_details, node_based_details, node_flows):
    """生成分析报告"""

    print("\n" + "="*60)
    print("充电站选址分析报告")
    print("="*60)

    # 当前选址
    print("\n【当前选址】(基于度数，20个节点)")
    print(f"节点: {sorted(current_stations, key=int)}")
    current_total_flow = sum(node_flows.get(n, 0) for n in current_stations)
    print(f"总流量覆盖: {current_total_flow:,.0f} 车次")

    # 方案A
    print("\n【方案A: 基于链路流量】(Top 20链路终点)")
    print(f"节点: {sorted(link_based_nodes, key=int)}")
    link_total_flow = sum(node_flows.get(n, 0) for n in link_based_nodes)
    print(f"总流量覆盖: {link_total_flow:,.0f} 车次")
    print("\nTop 10 链路详情:")
    for i, (link, node, flow) in enumerate(link_based_details[:10], 1):
        print(f"  {i}. 链路 {link} -> 节点 {node}, 流量: {flow:,.0f}")

    # 方案B
    print("\n【方案B: 基于节点流量】(Top 20节点)")
    print(f"节点: {sorted(node_based_nodes, key=int)}")
    node_total_flow = sum(node_flows.get(n, 0) for n in node_based_nodes)
    print(f"总流量覆盖: {node_total_flow:,.0f} 车次")
    print("\nTop 10 节点详情:")
    for i, (node, flow) in enumerate(node_based_details[:10], 1):
        print(f"  {i}. 节点 {node}, 流量: {flow:,.0f}")

    # 对比分析
    print("\n【对比分析】")

    overlap_link = set(link_based_nodes) & set(current_stations)
    overlap_node = set(node_based_nodes) & set(current_stations)
    overlap_ab = set(link_based_nodes) & set(node_based_nodes)

    print(f"\n当前 vs 方案A:")
    print(f"  重叠节点: {len(overlap_link)} 个 - {sorted(overlap_link, key=int)}")
    print(f"  流量提升: {(link_total_flow/current_total_flow - 1)*100:+.1f}%")

    print(f"\n当前 vs 方案B:")
    print(f"  重叠节点: {len(overlap_node)} 个 - {sorted(overlap_node, key=int)}")
    print(f"  流量提升: {(node_total_flow/current_total_flow - 1)*100:+.1f}%")

    print(f"\n方案A vs 方案B:")
    print(f"  重叠节点: {len(overlap_ab)} 个 - {sorted(overlap_ab, key=int)}")

    # 推荐
    print("\n【推荐】")
    if node_total_flow >= link_total_flow:
        print("推荐采用 方案B (基于节点流量)")
        print(f"新的 charging_nodes 配置:")
        new_config = {n: [0.5, 2.0] for n in sorted(node_based_nodes, key=int)}
        print(json.dumps(new_config, indent=2))
    else:
        print("推荐采用 方案A (基于链路流量)")
        new_config = {n: [0.5, 2.0] for n in sorted(link_based_nodes, key=int)}
        print(json.dumps(new_config, indent=2))

    return new_config


def main():
    # 配置 - 从项目根目录运行
    network_dir = "data/berlin_friedrichshain"
    network_name = "berlin_friedrichshain"  # 文件名前缀
    demand_multiplier = 3.5

    # 当前充电站配置
    current_stations = ["17", "26", "40", "49", "53", "66", "77", "93", "95", "107",
                       "116", "145", "148", "154", "157", "163", "164", "166", "177", "185"]

    # 运行仿真
    W, link_flows, node_flows, nodes, links_info = run_traffic_simulation(
        network_dir, network_name, demand_multiplier
    )

    # 选取充电站
    link_based_nodes, node_based_nodes, link_details, node_details = select_top_stations(
        link_flows, node_flows, links_info, n_stations=20
    )

    # 生成报告
    new_config = generate_report(
        current_stations, link_based_nodes, node_based_nodes,
        link_details, node_details, node_flows
    )

    # 可视化对比
    output_path = os.path.join(network_dir, "station_placement_comparison.png")
    visualize_comparison(
        nodes, links_info, current_stations,
        link_based_nodes, node_based_nodes, node_flows, output_path
    )

    print("\n分析完成!")


if __name__ == "__main__":
    main()
