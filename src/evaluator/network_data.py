# src/evaluator/network_data.py
"""
NetworkData: 可序列化的网络数据结构

设计原则：
1. 纯 Python 数据结构，可 pickle 序列化
2. 存储链路名称字符串，不存储 C++ 对象引用
3. 路径预计算只执行一次，多进程复用
"""

from __future__ import annotations

import os
import json
import csv
import logging
from dataclasses import dataclass, field
from typing import Optional
from itertools import islice

import numpy as np
import networkx as nx

# 仅在加载时使用 uxsimpp_extended，NetworkData 本身不依赖它
from uxsimpp_extended.uxsimpp import World, Vehicle


@dataclass
class NodeData:
    """节点数据"""
    name: str
    x: float
    y: float


@dataclass
class LinkData:
    """链路数据"""
    name: str
    start_node: str
    end_node: str
    length: float              # 米
    free_flow_speed: float     # 米/秒
    jam_density: float         # 车辆/米/车道
    merge_priority: float
    is_charging_link: bool     # 是否为充电链路


@dataclass
class DemandData:
    """交通需求数据"""
    origin: str
    destination: str
    start_t: float
    end_t: float
    flow: float                # 车辆数/单位时间
    is_charging: bool          # 是否为充电需求


@dataclass
class RouteInfo:
    """预处理的路径信息"""
    links: list[str]              # 链路名称列表
    charging_link_idx: int | None  # 充电链路在路径中的索引（或 None）
    charging_node: str | None      # 充电节点名称（或 None）


@dataclass
class NetworkData:
    """
    可序列化的网络数据容器

    主进程加载一次，序列化传递给各 Worker。
    """
    # 参数配置
    settings: dict

    # 网络拓扑
    nodes: list[NodeData]
    links: list[LinkData]
    demands: list[DemandData]

    # 预计算路径集合（区分充电/非充电）
    # {"charging": {(o,d): [RouteInfo, ...]}, "uncharging": {(o,d): [RouteInfo, ...]}}
    routes: dict[str, dict[tuple[str, str], list[RouteInfo]]]

    # 派生属性（从 settings 提取，方便访问）
    charging_nodes: dict[str, list[float]] = field(default_factory=dict)
    n_agents: int = 0
    n_periods: int = 0
    agent_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化派生属性"""
        self.charging_nodes = self.settings.get("charging_nodes", {})
        self.n_agents = len(self.charging_nodes)
        self.n_periods = self.settings.get("charging_periods", 8)
        self.agent_names = list(self.charging_nodes.keys())


class NetworkDataLoader:
    """
    网络数据加载器

    从文件加载网络数据，预计算路径集合，生成可序列化的 NetworkData。
    """

    def __init__(self, network_dir: str, network_name: str, random_seed: Optional[int] = 42):
        """
        初始化加载器

        Args:
            network_dir: 网络文件夹路径
            network_name: 网络名称
            random_seed: 随机种子
        """
        self.network_dir = network_dir
        self.network_name = network_name
        self.random_seed = random_seed

    def load(self) -> NetworkData:
        """
        加载网络数据并预计算路径

        Returns:
            NetworkData: 可序列化的网络数据
        """
        logging.debug(f"加载网络数据: {self.network_name}")

        # 1. 加载参数配置
        settings = self._load_settings()

        # 2. 加载节点
        nodes = self._load_nodes()

        # 3. 加载链路
        links = self._load_links(settings)

        # 4. 加载需求
        demands = self._load_demands(settings)

        # 5. 预计算路径（需要临时创建 World）
        routes = self._compute_routes(settings, nodes, links, demands)

        logging.debug(f"网络数据加载完成: {len(nodes)} 节点, {len(links)} 链路, {len(demands)} 需求")

        return NetworkData(
            settings=settings,
            nodes=nodes,
            links=links,
            demands=demands,
            routes=routes
        )

    def _load_settings(self) -> dict:
        """加载参数配置"""
        settings_path = os.path.join(self.network_dir, f"{self.network_name}_settings.json")
        logging.debug(f"加载配置: {settings_path}")

        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        return settings

    def _load_nodes(self) -> list[NodeData]:
        """加载节点数据"""
        node_path = os.path.join(self.network_dir, f"{self.network_name}_nodes.csv")
        logging.debug(f"加载节点: {node_path}")

        nodes = []
        with open(node_path, "r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r[1] != "x":  # 跳过表头
                    name, x, y = r[0], float(r[1]), float(r[2])
                    nodes.append(NodeData(name=name, x=x, y=y))

        return nodes

    def _load_links(self, settings: dict) -> list[LinkData]:
        """加载链路数据（包括自环充电链路）"""
        link_path = os.path.join(self.network_dir, f"{self.network_name}_links.csv")
        logging.debug(f"加载链路: {link_path}")

        links = []

        # 加载普通链路
        with open(link_path, "r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r[3] != "length":  # 跳过表头
                    links.append(LinkData(
                        name=r[0],
                        start_node=r[1],
                        end_node=r[2],
                        length=float(r[3]),
                        free_flow_speed=float(r[4]),
                        jam_density=float(r[5]),
                        merge_priority=float(r[6]),
                        is_charging_link=False
                    ))

        # 创建自环充电链路
        charging_nodes = settings.get("charging_nodes", {})
        charging_link_length = settings.get("charging_link_length", 3000)
        charging_link_free_flow_speed = settings.get("charging_link_free_flow_speed", 10)

        for node in charging_nodes.keys():
            charging_link_name = f"charging_{node}"
            links.append(LinkData(
                name=charging_link_name,
                start_node=node,
                end_node=node,  # 自环
                length=charging_link_length,
                free_flow_speed=charging_link_free_flow_speed,
                jam_density=0.2,  # 默认值
                merge_priority=1.0,
                is_charging_link=True
            ))
            logging.debug(f"创建充电链路: {charging_link_name}")

        return links

    def _load_demands(self, settings: dict) -> list[DemandData]:
        """加载交通需求数据"""
        demand_path = os.path.join(self.network_dir, f"{self.network_name}_demand.csv")
        logging.debug(f"加载需求: {demand_path}")

        demand_multiplier = settings.get("demand_multiplier", 1.0)
        charging_car_rate = settings.get("charging_car_rate", 0.3)

        demands = []
        with open(demand_path, "r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":  # 跳过表头
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    base_flow = float(r[4]) * demand_multiplier

                    # 充电车辆需求
                    demands.append(DemandData(
                        origin=origin,
                        destination=destination,
                        start_t=start_t,
                        end_t=end_t,
                        flow=base_flow * charging_car_rate,
                        is_charging=True
                    ))

                    # 非充电车辆需求
                    demands.append(DemandData(
                        origin=origin,
                        destination=destination,
                        start_t=start_t,
                        end_t=end_t,
                        flow=base_flow * (1 - charging_car_rate),
                        is_charging=False
                    ))

        return demands

    def _compute_routes(self, settings: dict, nodes: list[NodeData],
                        links: list[LinkData], demands: list[DemandData]) -> dict:
        """
        预计算路径集合

        需要临时创建 World 来获取 OD 对，然后用 NetworkX 计算路径。
        """
        logging.debug("预计算路径集合...")

        k = settings.get("routes_per_od", 5)

        # 获取所有 OD 对
        od_pairs = set()
        for demand in demands:
            od_pairs.add((demand.origin, demand.destination))

        # 构建 NetworkX 图（用于路径计算）
        # 普通图（非充电路径）
        G_normal = nx.DiGraph()
        # 多状态图（充电路径）
        G_charging = nx.DiGraph()

        link_dict = {}  # (start, end) -> link_name

        for link in links:
            weight = link.length / link.free_flow_speed

            if link.is_charging_link:
                # 自环充电链路：uncharged -> charged 状态转换
                node = link.start_node
                G_charging.add_edge(f"uncharged_{node}", f"charged_{node}", weight=weight)
                link_dict[(f"uncharged_{node}", f"charged_{node}")] = link.name
            else:
                # 普通链路
                start, end = link.start_node, link.end_node

                # 非充电图：直接添加
                G_normal.add_edge(start, end, weight=weight)
                link_dict[(start, end)] = link.name

                # 充电图：两个状态都添加
                G_charging.add_edge(f"uncharged_{start}", f"uncharged_{end}", weight=weight)
                G_charging.add_edge(f"charged_{start}", f"charged_{end}", weight=weight)
                link_dict[(f"uncharged_{start}", f"uncharged_{end}")] = link.name
                link_dict[(f"charged_{start}", f"charged_{end}")] = link.name

        # 计算路径
        routes = {
            "charging": {},
            "uncharging": {}
        }

        for o, d in od_pairs:
            # 非充电路径
            routes["uncharging"][(o, d)] = self._enumerate_k_shortest_routes(
                G_normal, o, d, k, link_dict, is_charging=False
            )

            # 充电路径
            routes["charging"][(o, d)] = self._enumerate_k_shortest_routes(
                G_charging, f"uncharged_{o}", f"charged_{d}", k, link_dict, is_charging=True
            )

        logging.debug(f"路径计算完成: {len(od_pairs)} 个 OD 对")

        return routes

    def _enumerate_k_shortest_routes(self, G: nx.DiGraph, source: str, target: str,
                                      k: int, link_dict: dict, is_charging: bool) -> list[RouteInfo]:
        """
        枚举 k 最短路径

        Args:
            G: NetworkX 有向图
            source: 起点
            target: 终点
            k: 路径数量
            link_dict: 边到链路名称的映射
            is_charging: 是否为充电路径（用于解析充电信息）

        Returns:
            list[RouteInfo]: 路径信息列表
        """
        try:
            k_shortest_paths = list(islice(
                nx.shortest_simple_paths(G, source, target, weight='weight'), k
            ))
        except nx.NetworkXNoPath:
            logging.warning(f"无法找到路径: {source} -> {target}")
            return []

        routes = []
        for path in k_shortest_paths:
            # 转换为链路名称列表
            links = []
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in link_dict:
                    links.append(link_dict[edge])
                else:
                    logging.warning(f"未找到链路: {edge}")

            # 提取充电信息
            charging_node = None
            charging_link_idx = None

            if is_charging:
                for idx, link_name in enumerate(links):
                    if link_name.startswith("charging_"):
                        charging_node = link_name.split("charging_")[1]
                        charging_link_idx = idx
                        break

            routes.append(RouteInfo(
                links=links,
                charging_link_idx=charging_link_idx,
                charging_node=charging_node
            ))

        return routes
