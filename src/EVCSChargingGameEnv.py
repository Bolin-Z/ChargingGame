# EVCSChargingGameEnv v3.0 - 基于自环充电链路 + 预定路径的实现
# 
# 设计思路：
# 1. 使用自环充电链路模拟充电行为（简化网络拓扑）
# 2. 集成PredefinedRouteVehicle确保严格按预定路径行驶  
# 3. 保持完整的PettingZoo ParallelEnv接口
# 4. 实现Day-to-Day动态均衡的UE-DTA仿真

import os
import json
import csv
import logging

from uxsim import World
import networkx as nx
from itertools import islice
from math import floor
from texttable import Texttable
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from typing import Dict, Any, Optional
from collections import defaultdict


class EVCSChargingGameEnv(ParallelEnv):
    """ 
    电动汽车充电站博弈环境 v3.0
    基于自环充电链路 + 预定路径的实现
    """
    metadata = {"name": "evcs_charging_game_v3"}
    
    def __init__(self,
                 network_dir: str,
                 network_name: str,
                 random_seed: Optional[int] = 42,
                 max_steps: int = 1000,
                 convergence_threshold: float = 0.01):
        """
        初始化充电站博弈环境
        
        Args:
            network_dir: 网络文件夹路径
            network_name: 网络名称
            random_seed: 随机种子
            max_steps: 单episode最大步数
            convergence_threshold: 博弈价格收敛阈值
        """
        super().__init__()
        
        # 环境参数
        self.env_random_seed = random_seed
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold

        # 初始化网络和路径
        self.__load_case(network_dir, network_name)
        
        # 设置智能体相关属性
        self.__possible_agents = [str(node) for node in self.charging_nodes.keys()]
        self.agents = self.__possible_agents.copy()
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        
        # MADRL 参数
        self.n_agents = len(self.charging_nodes)
        self.n_periods = self.charging_periods
        
        # 环境状态变量
        self.current_step = 0
        self.price_history = []  # List[np.array(n_agents, n_periods)]
        self.charging_flow_history = []  # List[np.array(n_agents, n_periods)]
    
    @property 
    def possible_agents(self):
        """ PettingZoo要求的智能体列表属性 """
        return self.__possible_agents
    
    def observation_space(self, agent: str) -> spaces.Dict:
        """ 返回指定智能体的观测空间 
        
        Args:
            agent: 智能体ID（在本环境中所有智能体观测空间相同，此参数满足PettingZoo接口要求）
        """
        return spaces.Dict({
            "last_round_all_prices": spaces.Box(
                low=0.0, high=1.0, 
                shape=(self.n_agents, self.n_periods), 
                dtype=np.float32
            ),
            "own_charging_flow": spaces.Box(
                low=0, high=np.inf, 
                shape=(self.n_periods,), 
                dtype=np.float32
            )
        })
    
    def action_space(self, agent: str) -> spaces.Box:
        """ 返回指定智能体的动作空间 
        
        Args:
            agent: 智能体ID（在本环境中所有智能体动作空间相同，此参数满足PettingZoo接口要求）
        """
        return spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.n_periods,), 
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """ 重置环境 """
        pass
    
    def step(self, actions: Dict[str, np.ndarray]) -> tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        执行一步博弈：智能体出价 → UE-DTA仿真 → 计算奖励和观测
        
        Args:
            actions: {agent_id: np.array([period0_price, period1_price, ...])}
            
        Returns:
            observations: {agent_id: {"last_round_all_prices": ..., "own_charging_flow": ...}}
            rewards: {agent_id: float}
            terminations: {agent_id: bool}
            truncations: {agent_id: bool}
            infos: {agent_id: dict}
        """
        pass

    def __load_case(self, network_dir: str, network_name: str):
        """ 加载环境用例 """
        logging.info(f"加载环境用例: {network_name}")
        
        # 加载参数
        self.__load_parameters(network_dir, network_name)
        
        # 加载路网
        self.__load_network(network_dir, network_name)
        
        # 计算路径集合
        self.__compute_routes()
        
        logging.info(f"环境用例加载完成")

    def __load_parameters(self, network_dir: str, network_name: str):
        """ 加载参数 """
        logging.info(f"加载参数...")
        with open(os.path.join(network_dir, f"{network_name}_settings.json"), "r") as f:
            settings = json.load(f)
            # 网络名称
            self.network_name = settings["network_name"]
            # 模拟时间(s)
            self.simulation_time = settings["simulation_time"]
            # Platoon 大小（veh代表车辆数）
            self.deltan = settings["deltan"]
            # 充电车辆比例
            self.charging_car_rate = settings["charging_car_rate"]
            # 充电链路长度(m)
            self.charging_link_length = settings["charging_link_length"]
            # 充电链路自由流速度(m/s)
            self.charging_link_free_flow_speed = settings["charging_link_free_flow_speed"]
            # 电价时段数
            self.charging_periods = settings["charging_periods"]
            # 充电节点与价格区间
            self.charging_nodes = settings["charging_nodes"]
            # 每个定价时段的时长(秒)
            self.period_duration = self.simulation_time / self.charging_periods
            # 每OD路径数
            self.routes_per_od = settings["routes_per_od"]
            # 时间价值系数
            self.time_value_coefficient = settings["time_value_coefficient"]
            # 单位充电需求量(kWh)
            self.charging_demand_per_vehicle = settings["charging_demand_per_vehicle"]
            # UE-DTA 收敛阈值
            self.ue_convergence_threshold = settings["ue_convergence_threshold"]
            # UE-DTA 最大迭代次数
            self.ue_max_iterations = settings["ue_max_iterations"]
            # UE-DTA 交换概率
            self.ue_swap_probability = settings["ue_swap_probability"]

    def __load_network(self, network_dir: str, network_name: str):
        """ 加载路网 """
        logging.info(f"加载路网 {self.network_name}...")
        
        node_path = os.path.join(network_dir, f"{network_name}_nodes.csv")
        link_path = os.path.join(network_dir, f"{network_name}_links.csv")
        demand_path = os.path.join(network_dir, f"{network_name}_demand.csv")

        # 使用增强的PredefinedRouteWorld
        self.W = PredefinedRouteWorld(
            name=self.network_name,
            deltan=self.deltan,
            tmax=self.simulation_time,
            random_seed=self.env_random_seed,
            user_attribute={}
        )

        # 加载节点
        logging.info(f"加载节点 {node_path}...")
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":  # 跳过表头
                    name, x, y = r
                    self.W.addNode(name, float(x), float(y))

        # 加载链路
        logging.info(f"加载链路 {link_path}...")
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":  # 跳过表头
                    link_name, start_node, end_node = r[0], r[1], r[2]
                    length = float(r[3])
                    free_flow_speed = float(r[4])
                    jam_density = float(r[5])
                    merge_priority = float(r[6])
                    
                    self.W.addLink(
                        link_name, start_node, end_node,
                        length=length,
                        free_flow_speed=free_flow_speed,
                        jam_density=jam_density,
                        merge_priority=merge_priority,
                        attribute={"charging_link": False}
                    )

        # 创建自环充电链路（仅使用settings.json中的参数）
        logging.info("创建自环充电链路...")
        for node in self.charging_nodes.keys():
            charging_link_name = f"charging_{node}"
            self.W.addLink(
                charging_link_name,
                start_node=node,
                end_node=node,  # 自环：起点=终点
                length=self.charging_link_length,
                free_flow_speed=self.charging_link_free_flow_speed,
                attribute={"charging_link": True}  # 标记为充电链路
            )
            
            logging.debug(f"创建充电链路: {charging_link_name} ({node}->{node})")

        # 加载交通需求
        logging.info(f"加载交通需求 {demand_path}...")
        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":  # 跳过表头
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    volume = float(r[4])
                    
                    try:
                        # 充电车辆需求
                        self.W.adddemand(
                            origin, destination, start_t, end_t,
                            volume * self.charging_car_rate,
                            float(r[5]) * self.charging_car_rate,
                            attribute={"charging_car": True}
                        )
                        
                        # 非充电车辆需求
                        self.W.adddemand(
                            origin, destination, start_t, end_t,
                            volume * (1 - self.charging_car_rate),
                            float(r[5]) * (1 - self.charging_car_rate),
                            attribute={"charging_car": False}
                        )
                        
                    except:
                        # 处理数据格式不一致的情况
                        # 充电车辆需求
                        self.W.adddemand(
                            origin, destination, start_t, end_t,
                            volume * self.charging_car_rate,
                            attribute={"charging_car": True}
                        )
                        
                        # 非充电车辆需求
                        self.W.adddemand(
                            origin, destination, start_t, end_t,
                            volume * (1 - self.charging_car_rate),
                            attribute={"charging_car": False}
                        )

    def __compute_routes(self):
        """ 计算路径集合 """
        logging.info(f"计算路径集合...")
        k = self.routes_per_od
        
        # 获取所有OD对
        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            o = veh.orig.name
            d = veh.dest.name
            od_pairs.add((o, d))
        
        # 初始化路径字典
        self.dict_od_to_routes = {
            "uncharging": {},
            "charging": {}
        }
        
        # 计算非充电路径和充电路径
        for o, d in od_pairs:
            self.dict_od_to_routes["uncharging"][(o, d)] = self.__enumerate_k_shortest_routes(o, d, k)
            self.dict_od_to_routes["charging"][(o, d)] = self.__enumerate_k_shortest_charge_routes(o, d, k)
        
        logging.info(f"路径计算完成：{len(od_pairs)}个OD对")

    def __enumerate_k_shortest_routes(self, source: str, target: str, k: int):
        """ 枚举非充电需求k最短路径 """
        G = nx.DiGraph()
        link_dict = {}
        # 构建只包含非充电链路的图
        for link in self.W.LINKS:
            if not link.attribute["charging_link"]:  # 排除充电链路
                G.add_edge(link.start_node.name, link.end_node.name, weight=link.length/link.u)
                link_dict[(link.start_node.name, link.end_node.name)] = link.name
        
        # 计算k最短路径
        k_shortest_paths = list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))
        
        # 转换为链路名称列表
        routes = []
        for path in k_shortest_paths:
            route = [link_dict[(path[i], path[i + 1])] for i in range(len(path) - 1)]
            routes.append(route)
        
        logging.debug(f"无充电需求路径：{source} -> {target} 数量：{len(routes)}")
        for route in routes:
            logging.debug(f"\t{route}")
        return routes

    def __enumerate_k_shortest_charge_routes(self, source: str, target: str, k: int):
        """ 枚举充电需求k最短路径（基于多状态图）"""
        G = nx.DiGraph()
        link_dict = {}
        
        # 构建多状态图：uncharged_节点 和 charged_节点
        for link in self.W.LINKS:
            start_node = link.start_node.name
            end_node = link.end_node.name
            weight = link.length / link.u
            
            if link.attribute["charging_link"]:
                # 自环充电链路：未充电状态 -> 已充电状态（状态转换）
                # 注意：自环充电链路的起点=终点
                node = start_node  # 自环节点
                G.add_edge(f"uncharged_{node}", f"charged_{node}", weight=weight)
                link_dict[(f"uncharged_{node}", f"charged_{node}")] = link.name
            else:
                # 普通链路：状态保持
                G.add_edge(f"uncharged_{start_node}", f"uncharged_{end_node}", weight=weight)
                G.add_edge(f"charged_{start_node}", f"charged_{end_node}", weight=weight)
                link_dict[(f"uncharged_{start_node}", f"uncharged_{end_node}")] = link.name
                link_dict[(f"charged_{start_node}", f"charged_{end_node}")] = link.name
        
        # 搜索路径：从 uncharged_source 到 charged_target
        source_state = f"uncharged_{source}"
        target_state = f"charged_{target}"
        
        k_shortest_paths = list(islice(nx.shortest_simple_paths(G, source_state, target_state, weight='weight'), k))
        
        # 转换为链路名称列表
        routes = []
        for path in k_shortest_paths:
            route = [link_dict[(path[i], path[i + 1])] for i in range(len(path) - 1)]
            routes.append(route)
        
        logging.debug(f"充电需求路径：{source} -> {target} 数量：{len(routes)}")
        for route in routes:
            logging.debug(f"\t{route}")
        return routes

    def __init_charging_prices(self, mode="random"):
        """ 初始化充电价格 """
        pass

    def __get_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """ 获取观测 """
        pass

    def __normalize_prices(self, price_matrix: np.ndarray) -> np.ndarray:
        """ 归一化价格 """
        pass

    def __update_prices_from_actions(self, actions: Dict[str, np.ndarray]):
        """ 从动作更新价格 """
        pass

    def __check_convergence(self) -> bool:
        """ 检查收敛 """
        pass

    def __create_simulation_world(self) -> World:
        """ 创建仿真世界 """
        pass

    def actions_to_prices(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """ 动作转价格 """
        pass

    def __initialize_routes(self, dict_od_to_charging_vehid: defaultdict, dict_od_to_uncharging_vehid: defaultdict, use_greedy: bool = True) -> Dict[str, list]:
        """ 初始化路径 """
        pass

    def __apply_routes_to_vehicles(self, W: World, routes_specified: Dict[str, list]):
        """ 为车辆分配路径 """
        pass

    def __get_period(self, t: float) -> int:
        """ 获取时段 """
        pass

    def __get_price(self, t: float, node: str) -> float:
        """ 获取价格 """
        pass

    def __calculate_actual_vehicle_cost_and_flow(self, veh, W: World, charging_flows: np.ndarray) -> float:
        """ 计算车辆成本和流量 """
        pass

    def __estimate_route_cost(self, route_obj, departure_time: float, is_charging_vehicle: bool) -> float:
        """ 估计路径成本 """
        pass

    def __route_choice_update(self, W: World, dict_od_to_charging_vehid: defaultdict, dict_od_to_uncharging_vehid: defaultdict, current_routes_specified: Dict[str, list]) -> tuple[float, Dict[str, list], np.ndarray]:
        """ 路径选择更新 """
        pass

    def __run_simulation(self) -> np.ndarray:
        """ 运行仿真 """
        pass

    def __calculate_rewards(self, charging_flows: np.ndarray) -> Dict[str, float]:
        """ 计算奖励 """
        pass


# =============================================================================
# 增强的UXsim类 - PredefinedRouteVehicle & PredefinedRouteWorld
# =============================================================================

import uxsim

class PredefinedRouteVehicle(uxsim.Vehicle):
    def __init__(self, W, orig, dest, departure_time, predefined_route=None, name=None, **kwargs):
        """
        支持预定路径的车辆类
        
        Parameters
        ----------
        W : World
            World对象
        orig : str | Node
            起点节点
        dest : str | Node
            终点节点  
        departure_time : int
            出发时间
        predefined_route : list, optional
            预定路径序列（链路名称列表），可为None（延迟分配）
        name : str, optional
            车辆名称
        **kwargs
            其他参数
        """
        # 调用父类初始化
        super().__init__(W, orig, dest, departure_time, name=name, **kwargs)
        
        # 预定路径相关属性
        self.predefined_route_links = []  # Link对象列表，供仿真使用
        self.route_assigned = False
        self.route_index = 0
        
        # 如果传入了预定路径，立即分配
        if predefined_route is not None:
            self.assign_route(predefined_route)
    
    def assign_route(self, route_names):
        """
        分配预定路径（将路径名称列表转换为Link对象列表）
        
        Parameters
        ----------
        route_names : list
            路径名称列表
        """
        if not route_names:
            return
            
        self.predefined_route_links = []
        for link_name in route_names:
            link = self.W.get_link(link_name)
            if link is None:
                raise ValueError(f"Link {link_name} not found in network")
            self.predefined_route_links.append(link)
        
        self.route_assigned = True
        self.route_index = 0
        
        # 初始化第一个链路
        if self.predefined_route_links:
            self.route_next_link = self.predefined_route_links[0]
            self.route_index = 1
    
    def route_next_link_choice(self):
        """
        选择下一个链路 - 如果有预定路径则严格按照预定路径执行
        """
        if not self.route_assigned or not self.predefined_route_links:
            # 回退到标准UXsim路径选择
            return super().route_next_link_choice()
        
        if self.route_index >= len(self.predefined_route_links):
            self.route_next_link = None
            return None  # 路径完成
        
        self.route_next_link = self.predefined_route_links[self.route_index]
        self.route_index += 1
        return self.route_next_link


class PredefinedRouteWorld(uxsim.World):
    def addVehicle(self, predefined_route, departure_time, name=None, **kwargs):
        """
        添加支持预定路径的车辆
        
        Parameters
        ----------
        predefined_route : list
            预定路径序列（链路名称列表）
        departure_time : int
            出发时间
        name : str, optional
            车辆名称
        **kwargs
            其他参数
        """
        veh = PredefinedRouteVehicle(self, None, None, departure_time, 
                                   predefined_route=predefined_route, name=name, **kwargs)
        return veh
    
    def adddemand(self, orig, dest, t_start, t_end, flow=-1, volume=-1, attribute=None, direct_call=True):
        """
        重写adddemand，使用PredefinedRouteVehicle代替标准Vehicle
        
        Parameters
        ----------
        orig : str | Node
            起点节点
        dest : str | Node  
            终点节点
        t_start : float
            开始时间
        t_end : float
            结束时间
        flow : float, optional
            流量（车辆/秒）
        volume : float, optional
            交通量
        attribute : any, optional
            车辆属性
        direct_call : bool, optional
            直接调用标志
        """
        if volume > 0:
            flow = volume/(t_end-t_start)
        
        f = 0
        for t in range(int(t_start/self.DELTAT), int(t_end/self.DELTAT)):
            f += flow*self.DELTAT
            while f >= self.DELTAN:
                # 使用PredefinedRouteVehicle，但不传入预定路径（延迟分配）
                PredefinedRouteVehicle(self, orig, dest, t, 
                                     predefined_route=None,  # 延迟分配
                                     departure_time_is_time_step=1, 
                                     attribute=attribute)
                f -= self.DELTAN

