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
    电动汽车充电站博弈环境
    """
    metadata = {"name": "evcs_charging_game"}
    
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
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        elif self.env_random_seed is not None:
            np.random.seed(self.env_random_seed)
            
        # 重置环境状态
        self.current_step = 0
        self.price_history = []  # 环境启动时无报价历史
        self.charging_flow_history = []  # 环境启动时无流量历史
        
        # 生成初始观测
        observations = self.__get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
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
        # 1. 将归一化动作映射到实际价格并更新价格历史
        self.__update_prices_from_actions(actions)
        
        # 2. 运行UE-DTA仿真获取充电流量
        charging_flows = self.__run_simulation()
        
        # 3. 计算奖励
        rewards = self.__calculate_rewards(charging_flows)
        
        # 4. 更新充电流量历史
        self.charging_flow_history.append(charging_flows)
        
        # 5. 更新状态
        self.current_step += 1
        
        # 6. 判断终止条件
        terminated = self.__check_convergence()
        truncated = self.current_step >= self.max_steps
        
        # 7. 生成新观测
        observations = self.__get_observations()
        
        # 8. 构建返回值
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {}
        
        return observations, rewards, terminations, truncations, infos

    def __load_case(self, network_dir: str, network_name: str):
        logging.info(f"加载环境用例...")
        # 设置随机种子
        np.random.seed(self.env_random_seed)
        # 加载参数
        self.__load_parameters(network_dir, network_name)
        # 加载网络
        self.__load_network(network_dir, network_name)
        # 计算路径
        self.__compute_routes()


    def __load_parameters(self, network_dir: str, network_name: str):
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
        logging.info(f"加载路网 {self.network_name}...")
        node_path = os.path.join(network_dir, f"{network_name}_nodes.csv")
        link_path = os.path.join(network_dir, f"{network_name}_links.csv")
        demand_path = os.path.join(network_dir, f"{network_name}_demand.csv")

        self.W:World = World(
            name=self.network_name,
            deltan=self.deltan,
            tmax=self.simulation_time,
            random_seed=self.env_random_seed,
            user_attribute={}
        )

        # 直接加载分离节点
        logging.info(f"加载节点并创建分离节点 {node_path}...")
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":  # 跳过表头
                    name, x, y = r
                    
                    if name in self.charging_nodes:
                        # 充电节点：创建分离节点 name_i 和 name_o
                        # 为可视化设置略微不同的坐标
                        offset = 0.1  # 坐标偏移量
                        self.W.addNode(f"{name}_i", float(x) - offset, float(y))
                        self.W.addNode(f"{name}_o", float(x) + offset, float(y))
                        logging.debug(f"创建充电节点分离：{name} -> {name}_i, {name}_o")
                    else:
                        # 普通节点：直接创建
                        self.W.addNode(name, float(x), float(y))

        
        # 直接重定向链路加载
        logging.info(f"加载链路并重定向到分离节点 {link_path}...")
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":  # 跳过表头
                    link_name, start_node, end_node = r[0], r[1], r[2]
                    length = float(r[3])
                    free_flow_speed = float(r[4])
                    jam_density = float(r[5])
                    merge_priority = float(r[6])
                    # 确定重定向后的起点和终点
                    actual_start = f"{start_node}_o" if start_node in self.charging_nodes else start_node
                    actual_end = f"{end_node}_i" if end_node in self.charging_nodes else end_node
                    
                    # 创建重定向后的链路
                    self.W.addLink(
                        link_name, actual_start, actual_end,
                        length=length,
                        free_flow_speed = free_flow_speed,
                        jam_density=jam_density,
                        merge_priority=merge_priority,
                        attribute={"charging_link": False}
                    )
                    
                    logging.debug(f"链路重定向：{start_node}->{end_node} 变为 {actual_start}->{actual_end}")
        
        # 直接创建内部链路（bypass 和 charging）
        logging.info("为充电节点创建内部链路...")
        for charging_node in self.charging_nodes.keys():
            start_node = f"{charging_node}_i"
            end_node = f"{charging_node}_o"
            
            # 创建旁路链路（非充电）
            bypass_name = f"{charging_node}_bypass"
            self.W.addLink(
                bypass_name, start_node, end_node,
                length=1, # 设置旁路链路长度为1 实际通行时间几乎可忽略
                free_flow_speed=100,
                attribute={"charging_link": False}
            )
            
            # 创建充电链路
            charging_name = f"{charging_node}_charging"
            self.W.addLink(
                charging_name, start_node, end_node,
                length=self.charging_link_length,
                free_flow_speed=self.charging_link_free_flow_speed,
                attribute={"charging_link": True}
            )
            logging.debug(f"创建内部链路：{bypass_name}, {charging_name}")

        # 直接重定向交通需求加载
        logging.info(f"加载交通需求并重定向到分离节点 {demand_path}...")
        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":  # 跳过表头
                    origin, destination = r[0], r[1]
                    start_t, end_t = float(r[2]), float(r[3])
                    volume = float(r[4])
                    
                    # 确定重定向后的起点和终点 可以选择在起点或者终点充电
                    actual_origin = f"{origin}_i" if origin in self.charging_nodes else origin
                    actual_destination = f"{destination}_o" if destination in self.charging_nodes else destination
                    
                    try:
                        # 充电车辆需求
                        self.W.adddemand(
                            actual_origin, actual_destination, start_t, end_t,
                            volume * self.charging_car_rate,
                            float(r[5]) * self.charging_car_rate,
                            attribute={"charging_car": True}
                        )
                        
                        # 非充电车辆需求
                        self.W.adddemand(
                            actual_origin, actual_destination, start_t, end_t,
                            volume * (1 - self.charging_car_rate),
                            float(r[5]) * (1 - self.charging_car_rate),
                            attribute={"charging_car": False}
                        )
                        
                    except:
                        # 处理数据格式不一致的情况
                        # 充电车辆需求
                        self.W.adddemand(
                            actual_origin, actual_destination, start_t, end_t,
                            volume * self.charging_car_rate,
                            attribute={"charging_car": True}
                        )
                        
                        # 非充电车辆需求
                        self.W.adddemand(
                            actual_origin, actual_destination, start_t, end_t,
                            volume * (1 - self.charging_car_rate),
                            attribute={"charging_car": False}
                        )
                    
                    logging.debug(f"需求重定向：{origin}->{destination} 变为 {actual_origin}->{actual_destination}")
        
        logging.info("交通需求重定向加载完成")
        logging.info("节点分离网络加载完成")

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
            if not link.attribute["charging_link"]:
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
                # 充电链路：未充电状态 -> 已充电状态（状态转换）
                G.add_edge(f"uncharged_{start_node}", f"charged_{end_node}", weight=weight)
                link_dict[(f"uncharged_{start_node}", f"charged_{end_node}")] = link.name
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
        """
        初始化充电价格
        
        Args:
            mode: 初始化模式，"random"为随机初始化，"midpoint"为中点初始化
        
        Returns:
            np.array(n_agents, n_periods): 初始化的价格矩阵
        """
        logging.info(f"初始化充电价格 (模式: {mode})...")
        
        prices = np.zeros((self.n_agents, self.n_periods))
        
        for agent_idx, agent in enumerate(self.agents):
            bounds = self.charging_nodes[agent]
            min_price, max_price = bounds[0], bounds[1]
            
            if mode == "random":
                # 在价格区间内随机采样
                for period in range(self.n_periods):
                    prices[agent_idx, period] = np.random.uniform(min_price, max_price)
            elif mode == "midpoint":
                # 初始化为价格区间中点
                midpoint = (min_price + max_price) / 2.0
                for period in range(self.n_periods):
                    prices[agent_idx, period] = midpoint
            else:
                raise ValueError(f"不支持的初始化模式: {mode}")
        
        return prices
    
    def __get_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        生成所有智能体的观测
        
        Returns:
            Dict[agent_id, Dict[obs_key, np.ndarray]]: 各智能体的观测字典
        """
        observations = {}
        
        # 获取上轮所有价格的归一化版本
        if len(self.price_history) >= 1:
            # 使用上一轮的价格
            last_prices = self.__normalize_prices(self.price_history[-1])
        else:
            # 环境刚启动时没有报价历史，使用零矩阵
            last_prices = np.zeros((self.n_agents, self.n_periods))
        
        # 获取当前/上轮的充电流量
        if len(self.charging_flow_history) > 0:
            last_flows = self.charging_flow_history[-1]
        else:
            # 没有流量历史时使用零矩阵
            last_flows = np.zeros((self.n_agents, self.n_periods))
            
        for agent_idx, agent in enumerate(self.agents):
            observations[agent] = {
                "last_round_all_prices": last_prices.astype(np.float32),
                "own_charging_flow": last_flows[agent_idx].astype(np.float32)
            }
            
        return observations
    
    def __normalize_prices(self, price_matrix: np.ndarray) -> np.ndarray:
        """
        将实际价格矩阵归一化到[0,1]区间
        
        Args:
            price_matrix: 实际价格矩阵 (n_agents, n_periods)
            
        Returns:
            np.ndarray: 归一化价格矩阵 (n_agents, n_periods)
        """
        normalized = np.zeros((self.n_agents, self.n_periods))
        
        for agent_idx, agent in enumerate(self.agents):
            bounds = self.charging_nodes[agent]
            min_price, max_price = bounds[0], bounds[1]
            
            for period in range(self.n_periods):
                actual_price = price_matrix[agent_idx, period]
                # 归一化: (价格 - 最小值) / (最大值 - 最小值)
                normalized[agent_idx, period] = (actual_price - min_price) / (max_price - min_price)
        
        return normalized
    
    def __update_prices_from_actions(self, actions: Dict[str, np.ndarray]):
        """
        将归一化动作映射到实际价格并更新价格历史
        
        Args:
            actions: 智能体动作字典 {agent_id: np.array(n_periods)} 值域[0,1]
        """
        # 使用公共接口获取价格矩阵
        new_prices = self.actions_to_prices(actions)
        
        # 添加到价格历史
        self.price_history.append(new_prices)
        
        logging.debug(f"步骤{self.current_step}: 更新价格到历史记录，当前历史长度={len(self.price_history)}")
        
        return new_prices
    
    def __check_convergence(self) -> bool:
        """
        检查价格是否收敛（基于L2范数）
        
        Returns:
            bool: 是否已收敛
        """
        if len(self.price_history) < 2:
            return False
            
        # 计算所有智能体价格向量的变化
        current_prices = self.price_history[-1]
        previous_prices = self.price_history[-2]
        
        # 使用L2范数判断收敛
        price_change = np.linalg.norm(current_prices - previous_prices)
        
        converged = price_change < self.convergence_threshold
        logging.debug(f"步骤{self.current_step}: 价格变化L2范数={price_change:.6f}, 收敛阈值={self.convergence_threshold}, 是否收敛={converged}")
        
        return converged
    
    def __create_simulation_world(self) -> World:
        """
        创建配置了当前充电价格的仿真世界实例，包含完整的节点分离网络结构
        
        Returns:
            World: 新的仿真世界实例
        """
        W = World(
            name=self.network_name + f"_step{self.current_step}",
            deltan=self.deltan,
            tmax=self.simulation_time,
            print_mode=0,  # 静默模式
            save_mode=0,   # 不保存文件
            show_mode=0,
            random_seed=self.env_random_seed if self.env_random_seed is not None else 0,
            user_attribute={}
        )
        
        # 复制所有节点（包括分离节点）
        for node in self.W.NODES:
            W.addNode(node.name, node.x, node.y)
        
        # 复制所有链路（包括重定向链路、旁路链路、充电链路）
        charging_links_created = 0
        bypass_links_created = 0
        normal_links_created = 0
        
        for link in self.W.LINKS:
            W.addLink(
                link.name, 
                link.start_node.name, 
                link.end_node.name,
                length=link.length,
                free_flow_speed=link.u,
                jam_density=link.kappa,
                merge_priority=link.merge_priority,
                attribute=link.attribute.copy()
            )
            
            # 统计链路类型
            if link.attribute.get("charging_link", False):
                charging_links_created += 1
            elif "bypass" in link.name:
                bypass_links_created += 1
            else:
                normal_links_created += 1
        
        logging.debug(f"复制网络结构完成: {normal_links_created}条普通链路, {bypass_links_created}条旁路链路, {charging_links_created}条充电链路")
        
        # 复制交通需求（已重定向到分离节点）
        vehicles_created = 0
        charging_vehicles = 0
        
        for veh in self.W.VEHICLES.values():
            W.addVehicle(
                veh.orig.name, veh.dest.name, 
                veh.departure_time,
                departure_time_is_time_step=True,
                attribute=veh.attribute.copy()
            )
            vehicles_created += 1
            if veh.attribute.get("charging_car", False):
                charging_vehicles += 1
        
        logging.debug(f"复制交通需求完成: {vehicles_created}辆车辆，其中{charging_vehicles}辆充电车辆")
        
        return W
    
    def actions_to_prices(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将归一化动作映射到实际价格（纯函数，不修改环境状态）
        
        Args:
            actions: 智能体动作字典 {agent_id: np.array(n_periods)} 值域[0,1]
        
        Returns:
            np.ndarray: 价格矩阵 (n_agents, n_periods)
        """
        # 创建价格矩阵
        prices = np.zeros((self.n_agents, self.n_periods))
        
        for agent_idx, agent in enumerate(self.agents):
            bounds = self.charging_nodes[agent]
            min_price, max_price = bounds[0], bounds[1]
            
            for period in range(self.n_periods):
                # 将[0,1]动作映射到[min_price, max_price]
                normalized_action = actions[agent][period]
                actual_price = min_price + normalized_action * (max_price - min_price)
                prices[agent_idx, period] = actual_price
        
        return prices
    
    def __initialize_routes(self, dict_od_to_charging_vehid: defaultdict, dict_od_to_uncharging_vehid: defaultdict, use_greedy: bool = True) -> Dict[str, list]:
        """
        为所有车辆分配初始路径
        
        Args:
            dict_od_to_charging_vehid: 充电车辆的OD映射
            dict_od_to_uncharging_vehid: 非充电车辆的OD映射  
            use_greedy: True使用贪心策略(最短路径)，False使用随机策略
            
        Returns:
            Dict[str, list]: 车辆ID到路径的映射 {veh_id: [link_name1, link_name2, ...]}
        """
        routes_specified = {}
        
        if use_greedy:
            # 贪心策略：选择最短路径
            for od_pair, veh_ids in dict_od_to_charging_vehid.items():
                available_routes = self.dict_od_to_routes["charging"][od_pair]
                if available_routes:
                    best_route = available_routes[0]  # 第一条就是最短的
                    for veh_id in veh_ids:
                        routes_specified[veh_id] = best_route
            
            for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
                available_routes = self.dict_od_to_routes["uncharging"][od_pair]
                if available_routes:
                    best_route = available_routes[0]  # 第一条就是最短的
                    for veh_id in veh_ids:
                        routes_specified[veh_id] = best_route
                        
            logging.debug(f"初始化路径分配（贪心策略）：充电{len(dict_od_to_charging_vehid)}个OD对，非充电{len(dict_od_to_uncharging_vehid)}个OD对")
            
        else:
            # 随机策略：随机选择路径
            for od_pair, veh_ids in dict_od_to_charging_vehid.items():
                available_routes = self.dict_od_to_routes["charging"][od_pair]
                if available_routes:
                    for veh_id in veh_ids:
                        random_route_idx = np.random.choice(len(available_routes))
                        routes_specified[veh_id] = available_routes[random_route_idx]
            
            for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
                available_routes = self.dict_od_to_routes["uncharging"][od_pair]
                if available_routes:
                    for veh_id in veh_ids:
                        random_route_idx = np.random.choice(len(available_routes))
                        routes_specified[veh_id] = available_routes[random_route_idx]
                        
            logging.info(f"初始化路径分配（随机策略）：充电{len(dict_od_to_charging_vehid)}个OD对，非充电{len(dict_od_to_uncharging_vehid)}个OD对")
        
        return routes_specified
    
    def __apply_routes_to_vehicles(self, W: World, routes_specified: Dict[str, list]):
        """
        为车辆分配指定的路径
        
        Args:
            W: World实例
            routes_specified: 车辆ID到路径的映射 {veh_id: [link_name1, link_name2, ...]}
        """
        charging_route_count = 0
        total_route_count = 0
        
        for veh_id, route_links in routes_specified.items():
            if veh_id in W.VEHICLES:
                total_route_count += 1
                veh = W.VEHICLES[veh_id]
                
                # 检查是否是充电车辆且路径包含充电链路（节点分离方案中为 *_charging 链路）
                if veh.attribute.get("charging_car", False):
                    # 确保route_links是字符串列表
                    route_names = [link if isinstance(link, str) else link.name for link in route_links]
                    has_charging_link = any("_charging" in link_name for link_name in route_names)
                    if has_charging_link:
                        charging_route_count += 1
                        
                # 应用路径（严格模式：只能使用指定路径中的链路）
                veh.enforce_route(route_links, set_avoid=True)
        
        logging.debug(f"路径分配完成：总计{total_route_count}辆，充电路径{charging_route_count}条")
    
    def __get_period(self, t: float) -> int:
        """
        获取 t 时刻对应的电价时间段
        
        Args:
            t: 时间戳（秒）
            
        Returns:
            int: 时段索引 [0, n_periods-1]
        """
        return max(0, min(floor(t / self.period_duration), self.n_periods - 1))
    
    def __get_price(self, t: float, node: str) -> float:
        """
        获取 t 时刻 node 的电价（当前step内价格固定）
        
        Args:
            t: 时间戳（秒）
            node: 充电节点名称
            
        Returns:
            float: 该时刻该节点的电价
        """
        period = self.__get_period(t)
        agent_idx = self.agent_name_mapping[node]
        
        # 使用当前step的价格（price_history[-1]）
        current_prices = self.price_history[-1]
        return current_prices[agent_idx, period]
    
    def __calculate_actual_vehicle_cost_and_flow(self, veh, W: World, charging_flows: np.ndarray) -> float:
        """
        计算车辆实际总成本并同时统计充电流量
        
        Args:
            veh: 车辆对象
            W: World实例
            charging_flows: 充电流量矩阵 (n_agents, n_periods)，会被原地修改
            
        Returns:
            float: 车辆总成本（元）
        """
        route, timestamps = veh.traveled_route()
        
        # 计算时间成本
        travel_time = timestamps[-1] - timestamps[0]
        time_cost = self.time_value_coefficient * travel_time
        
        # 计算充电成本并统计流量（仅充电车辆）
        charging_cost = 0.0

        if not veh.attribute["charging_car"]:
            logging.error(f"车辆{veh.name}不是充电车辆")

        if veh.attribute.get("charging_car", False):
            flag = False
            for i, link in enumerate(route):
                if link.attribute.get("charging_link", False):
                    flag = True
                    # 获取进入充电链路的时刻
                    charging_entry_time = timestamps[i]
                    # 从节点分离方案的链路名称提取充电节点（格式：{node}_charging）
                    if "_charging" in link.name:
                        charging_node = link.name.split("_charging")[0]
                    else:
                        continue
                        
                    # 获取时段和智能体索引
                    charging_period = self.__get_period(charging_entry_time)
                    agent_idx = self.agent_name_mapping[charging_node]
                    
                    # 统计充电流量（每个Vehicle对象代表DELTAN辆实际车辆）
                    charging_flows[agent_idx, charging_period] += self.deltan
                    
                    # 获取该时刻的充电价格并计算成本
                    charging_price = self.__get_price(charging_entry_time, charging_node)
                    charging_cost = charging_price * self.charging_demand_per_vehicle
                    
                    logging.debug(f"车辆{veh.name}在时刻{charging_entry_time:.1f}({charging_period}时段)使用充电站{charging_node}")
                    break  # 每辆车只充电一次
            if not flag:
                # logging.error(f"车辆{veh.name}没有使用充电链路")
                # logging.error(f"分配路径: {veh.links_prefer}")
                # logging.error(f"实际路径: {route}")
                pass
        
        total_cost = time_cost + charging_cost
        return total_cost
    
    def __estimate_route_cost(self, route_obj, departure_time: float, is_charging_vehicle: bool) -> float:
        """
        估算路径的总成本（预期成本）
        
        Args:
            route_obj: UXSim路径对象
            departure_time: 出发时间
            is_charging_vehicle: 是否为充电车辆
            
        Returns:
            float: 预期总成本（元）
        """
        current_time = departure_time
        charging_cost = 0.0
        
        for link in route_obj.links:
            link_travel_time = link.actual_travel_time(current_time)
            
            # 仅为充电车辆计算充电成本
            if is_charging_vehicle and link.attribute.get("charging_link", False):
                if "_charging" in link.name:
                    # 从节点分离方案的链路名称提取充电节点（格式：{node}_charging）
                    charging_node = link.name.split("_charging")[0]
                    charging_price = self.__get_price(current_time, charging_node)
                    charging_cost = charging_price * self.charging_demand_per_vehicle
            
            current_time += link_travel_time
        
        total_travel_time = current_time - departure_time
        time_cost = self.time_value_coefficient * total_travel_time
        
        return time_cost + charging_cost
    
    def __route_choice_update(self, W: World, dict_od_to_charging_vehid: defaultdict, dict_od_to_uncharging_vehid: defaultdict, current_routes_specified: Dict[str, list]) -> tuple[float, Dict[str, list], np.ndarray]:
        """
        执行路径选择与切换逻辑，返回总成本差、新路径分配和充电流量统计
        
        Args:
            W: World实例
            dict_od_to_charging_vehid: 充电车辆的OD映射
            dict_od_to_uncharging_vehid: 非充电车辆的OD映射
            current_routes_specified: 当前路径分配
            
        Returns:
            tuple: (总成本差, 新路径分配字典, 充电流量矩阵)
        """
        new_routes_specified = {}
        total_cost_gap = 0.0
        
        # 初始化充电流量统计矩阵
        charging_flows = np.zeros((self.n_agents, self.n_periods))
        
        # 记录路径选择的详细信息
        charging_vehicles_processed = 0
        route_switches = 0
        
        logging.error(f"---充电veh数目: {len(dict_od_to_charging_vehid) * self.deltan}")

        # 为充电车辆执行路径选择
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"][od_pair]
            
            for veh_id in veh_ids:
                veh = W.VEHICLES[veh_id]
                charging_vehicles_processed += 1
                
                # 跳过未完成行程的车辆
                if veh.state != "end":
                    new_routes_specified[veh_id] = current_routes_specified[veh_id]  # 保持原分配路径
                    continue
                
                # 计算当前路径的实际成本并统计充电流量
                current_cost = self.__calculate_actual_vehicle_cost_and_flow(veh, W, charging_flows)
                current_route = current_routes_specified[veh_id]  # 使用原分配的路径
                
                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route
                
                if available_routes:
                    for route_links in available_routes:
                        # 计算替代路径的预期成本
                        route_obj = W.defRoute(route_links)
                        alt_cost = self.__estimate_route_cost(route_obj, veh.departure_time_in_second, True)
                        
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links
                
                # 计算成本差
                cost_gap = current_cost - best_cost
                total_cost_gap += cost_gap
                
                # 路径切换决策（随机切换机制）
                switched = False
                if cost_gap > 0 and np.random.random() < self.ue_swap_probability:
                    new_routes_specified[veh_id] = best_route
                    switched = True
                    route_switches += 1
                else:
                    new_routes_specified[veh_id] = current_route
        
        # 为非充电车辆执行路径选择
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"][od_pair]
            
            for veh_id in veh_ids:
                veh = W.VEHICLES[veh_id]
                
                # 跳过未完成行程的车辆
                if veh.state != "end":
                    new_routes_specified[veh_id] = current_routes_specified[veh_id]  # 保持原分配路径
                    continue
                
                # 计算当前路径的实际成本（非充电车辆，不统计流量）
                route, timestamps = veh.traveled_route()
                for link in veh.links_prefer:
                    if link not in route:
                        logging.error(f"车辆{veh.name}的路径不包含指定路径")
                        logging.error(f"分配路径: {veh.links_prefer}")
                        logging.error(f"实际路径: {route}")
    
                travel_time = timestamps[-1] - timestamps[0]
                current_cost = self.time_value_coefficient * travel_time
                
                current_route = current_routes_specified[veh_id]  # 使用原分配的路径
                
                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route
                
                if available_routes:
                    for route_links in available_routes:
                        # 计算替代路径的预期成本
                        route_obj = W.defRoute(route_links)
                        alt_cost = self.__estimate_route_cost(route_obj, veh.departure_time_in_second, False)
                        
                        if alt_cost < best_cost:
                            best_cost = alt_cost
                            best_route = route_links
                
                # 计算成本差
                cost_gap = current_cost - best_cost
                total_cost_gap += cost_gap
                
                # 路径切换决策
                if cost_gap > 0 and np.random.random() < self.ue_swap_probability:
                    new_routes_specified[veh_id] = best_route
                else:
                    new_routes_specified[veh_id] = current_route
        
        logging.debug(f"路径选择完成：处理{charging_vehicles_processed}辆充电车辆，{route_switches}次路径切换")
        logging.debug(f"各站充电流量：{dict(zip(self.agents, charging_flows.sum(axis=1)))}")
        
        return total_cost_gap, new_routes_specified, charging_flows
    
    def __run_simulation(self) -> np.ndarray:
        """
        运行基于day-to-day动态均衡的UXSim仿真并返回充电流量统计
        
        Returns:
            np.ndarray: 充电流量矩阵 (n_agents, n_periods)
        """
        logging.info("开始UE-DTA求解...")
        
        # 获取充电和非充电车辆的OD映射
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        W_template = self.__create_simulation_world()
        
        for key, veh in W_template.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute.get("charging_car", False):
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)
        
        logging.error(f"车辆数目：{len(W_template.VEHICLES) * self.deltan}")
        logging.error(f"充电车辆数目：{len(dict_od_to_charging_vehid) * self.deltan}")
        logging.error(f"非充电车辆数目：{len(dict_od_to_uncharging_vehid) * self.deltan}")

        # 初始化路径分配（如果是第一次调用）
        if not hasattr(self, 'current_routes_specified'):
            self.current_routes_specified = self.__initialize_routes(dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True)
        
        # Day-to-day迭代求解
        final_charging_flows = None
        
        for iteration in range(self.ue_max_iterations):
            # 创建新的仿真实例
            W = self.__create_simulation_world()
            
            # 应用当前路径分配
            self.__apply_routes_to_vehicles(W, self.current_routes_specified)
            
            # 执行仿真
            W.exec_simulation()
            
            # 检查未完成车辆
            unfinished_trips = W.analyzer.trip_all - W.analyzer.trip_completed
            if unfinished_trips > 0:
                logging.warning(f"第{iteration+1}轮：{unfinished_trips}/{W.analyzer.trip_all} 车辆未完成行程")
            
            # 计算成本差并执行路径切换，同时获取充电流量
            total_cost_gap, new_routes_specified, charging_flows = self.__route_choice_update(W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, self.current_routes_specified)
            average_cost_gap = total_cost_gap / len(W.VEHICLES)
            
            # 保存最终的充电流量
            final_charging_flows = charging_flows
            
            # 动态显示进度
            print(f"\\r第{iteration+1}轮：平均成本差={average_cost_gap:.2f}元，总行程时间={W.analyzer.total_travel_time:.1f}秒", end="", flush=True)
            
            # 更新路径分配
            self.current_routes_specified = new_routes_specified
            
            # 收敛判断
            if average_cost_gap < self.ue_convergence_threshold:
                print(f"\\nUE-DTA在第{iteration+1}轮收敛，平均成本差={average_cost_gap:.2f}元")
                break
        
        # 如果未收敛
        if iteration == self.ue_max_iterations - 1:
            print(f"\\nUE-DTA达到最大迭代次数{self.ue_max_iterations}，平均成本差={average_cost_gap:.2f}元")
        
        # 统计完成行程的充电车辆数量
        completed_charging_vehicles = 0
        total_charging_vehicles = 0
        for veh in W.VEHICLES.values():
            if veh.attribute.get("charging_car", False):
                total_charging_vehicles += 1
                if veh.state == "end":
                    completed_charging_vehicles += 1
        
        logging.info(f"充电车辆统计：{completed_charging_vehicles}/{total_charging_vehicles} 完成行程")
        logging.info(f"充电流量汇总：{dict(zip(self.agents, final_charging_flows.sum(axis=1)))}")
        
        return final_charging_flows
    
    def __calculate_rewards(self, charging_flows: np.ndarray) -> Dict[str, float]:
        """
        计算各智能体的奖励（收益）
        
        Args:
            charging_flows: 充电流量矩阵 (n_agents, n_periods)
            
        Returns:
            Dict[str, float]: 各智能体的收益 {agent_id: reward}
        """
        rewards = {}
        current_prices = self.price_history[-1]  # 使用当前价格
        
        for agent_idx, agent in enumerate(self.agents):
            total_reward = 0.0
            for period in range(self.n_periods):
                price = current_prices[agent_idx, period]
                flow = charging_flows[agent_idx, period]
                total_reward += price * flow * self.charging_demand_per_vehicle
            rewards[agent] = total_reward
            
        logging.debug(f"步骤{self.current_step}: 各智能体收益={rewards}")
        return rewards