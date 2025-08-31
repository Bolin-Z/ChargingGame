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

class EVCSGameEnv(ParallelEnv):
    """ 电动汽车充电站博弈环境 """
    metadata = {"name": "evcs_game"}
    def __init__(self,
                 network_dir: str,
                 network_name: str,
                 random_seed: Optional[int] = None,
                 max_steps: int = 1000,
                 convergence_threshold: float = 0.01):
        """
        Args:
            network_dir: 网络文件夹路径
            network_name: 网络名称
            random_seed: 随机种子
            max_steps: 单episode最大步数
            convergence_threshold: 价格收敛阈值
        """
        super().__init__()
        
        # 环境参数
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.env_random_seed = random_seed
        
        # 初始化网络和路径
        self._init(network_dir, network_name)

        # MADRL 参数
        self.n_agents = len(self._charging_nodes)
        self.n_periods = self._charging_periods

        # 固定智能体列表
        self.agents = [str(node) for node in self._charging_nodes.keys()]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        
        # 环境状态
        self.current_step = 0
        self.price_history = []  # 历史价格记录
        self.last_charging_flows = {agent: np.zeros(self.n_periods) for agent in self.agents}
    
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
        if seed is not None:
            np.random.seed(seed)
        elif self.env_random_seed is not None:
            np.random.seed(self.env_random_seed)
            
        # 重置环境状态
        self.current_step = 0
        self.price_history = []
        self.last_charging_flows = {agent: np.zeros(self.n_periods) for agent in self.agents}
        
        # 随机初始化价格
        self._init_charging_prices()
        
        # 生成初始观测
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]) -> tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """ 执行一步博弈 """
        # 将归一化动作映射到实际价格
        self._update_prices_from_actions(actions)
        
        # 运行仿真
        charging_flows = self._run_simulation()
        
        # 计算奖励
        rewards = self._calculate_rewards(charging_flows)
        
        # 更新状态
        self.current_step += 1
        self.last_charging_flows = charging_flows
        
        # 判断终止条件
        terminated = self._check_convergence()
        truncated = self.current_step >= self.max_steps
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        
        # 生成新观测
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """ 生成所有智能体的观测 """
        observations = {}
        
        # 获取上轮所有价格的归一化版本
        if len(self.price_history) > 0:
            last_prices = self._normalize_prices(self.price_history[-1])
        else:
            last_prices = np.zeros((self.n_agents, self.n_periods))
            
        for agent in self.agents:
            observations[agent] = {
                "last_round_all_prices": last_prices.astype(np.float32),
                "own_charging_flow": self.last_charging_flows[agent].astype(np.float32)
            }
            
        return observations
    
    def _get_period(self, t:float):
        """ 获取 t 时刻对应的电价时间段 """
        return max(0,
        min(floor(t / self._period_duration)),
        self._charging_periods - 1)
    
    def _get_price(self, t:float, node:str):
        """ 获取 t 时刻 node 的电价 """
        return self.charging_prices[self._get_period(t)][node]
    
    def _set_price(self, p:int, node:str, price:float):
        """ 设置 p 时段 node 的电价 """
        self.charging_prices[p][node] = price
    
    def _init(self, network_dir:str, network_name:str):
        logging.info(f"初始化环境 {network_name}...")
        
        # 设置随机种子（用于所有随机操作）
        if self.env_random_seed is not None:
            np.random.seed(self.env_random_seed)
            
        # 加载网络
        self._load_network(network_dir, network_name)
        # 计算路径
        self._compute_routes()
        # 初始化充电价格
        self._init_charging_prices()
        
        logging.info(f"初始化环境 {network_name} 完成")
        logging.info("路网信息：")
        if logging.getLogger().level == logging.INFO:
            print(self.draw_world_info())
        logging.info("初始电价：")
        if logging.getLogger().level == logging.INFO:
            print(self.draw_charging_prices())
    
    def _load_network(self, network_dir:str, network_name:str):
        """ 加载网络 """

        logging.info(f"加载路网 {network_name}...")
        settings_path = os.path.join(network_dir, f"{network_name}_settings.json")
        node_path = os.path.join(network_dir, f"{network_name}_nodes.csv")
        link_path = os.path.join(network_dir, f"{network_name}_links.csv")
        demand_path = os.path.join(network_dir, f"{network_name}_demand.csv")

        logging.info(f"加载配置文件 {settings_path}...")
        with open(settings_path, "r") as f:
            settings = json.load(f)

            self.network_name:str = settings["network_name"]
            
            self._simulation_time:float = float(settings["simulation_time"])
            self._deltan:int = int(settings["deltan"])

            self._charging_car_rate:float = float(settings["charging_car_rate"])
            self._charging_link_length:float = float(settings["charging_link_length"])
            self._charging_link_free_flow_speed:float = float(settings["charging_link_free_flow_speed"])
            
            self._charging_periods:int = int(settings["charging_periods"])
            self._charging_nodes:dict[str,list[float]] = settings["charging_nodes"]
            self._period_duration:float = float(self._simulation_time / self._charging_periods)

            self._routes_per_od:int = int(settings["routes_per_od"])

        logging.info(f"初始化路网 {self.network_name}...")
        self.W:World = World(
            name=self.network_name,
            deltan=self._deltan,
            tmax=self._simulation_time,
            print_mode=1,
            save_mode=1,
            show_mode=0,
            random_seed=self.env_random_seed if self.env_random_seed is not None else 0,
            user_attribute={}
        )

        # 加载节点
        logging.info(f"加载节点 {node_path}...")
        with open(node_path, "r") as f:
            for r in csv.reader(f):
                if r[1] != "x":
                    name, x, y = r
                    self.W.addNode(name, float(x), float(y))
        
        # 加载边
        logging.info(f"加载边 {link_path}...")
        with open(link_path, "r") as f:
            for r in csv.reader(f):
                if r[3] != "length":
                    self.W.addLink(
                        r[0], r[1], r[2], # name start_node end_node
                        length=float(r[3]), 
                        free_flow_speed=float(r[4]), 
                        jam_density=float(r[5]), 
                        merge_priority=float(r[6]),
                        attribute={"charging_link" : False})
        
        # 加载需求
        logging.info(f"加载需求 {demand_path}...")
        with open(demand_path, "r") as f:
            for r in csv.reader(f):
                if r[2] != "start_t":
                    try:
                        self.W.adddemand(r[0], r[1], float(r[2]), float(r[3]), 
                                        float(r[4]) * self._charging_car_rate, 
                                        float(r[5]) * self._charging_car_rate,
                                        attribute={"charging_car":True})
                        self.W.adddemand(r[0], r[1], float(r[2]), float(r[3]),
                                        float(r[4]) * (1 - self._charging_car_rate),
                                        float(r[5]) * (1 - self._charging_car_rate),
                                        attribute={"charging_car":False})
                    except:
                        self.W.adddemand(r[0], r[1], float(r[2]), float(r[3]), 
                                        float(r[4]) * self._charging_car_rate, 
                                        attribute={"charging_car":True})
                        self.W.adddemand(r[0], r[1], float(r[2]), float(r[3]),
                                        float(r[4]) * (1 - self._charging_car_rate),
                                        attribute={"charging_car":False})
        
        # 添加虚拟充电链路
        logging.info(f"添加虚拟充电链路...")
        for node in self._charging_nodes.keys():
            self.W.addLink(
                "charging_" + node, node, node,
                length=self._charging_link_length,
                free_flow_speed=self._charging_link_free_flow_speed,
                attribute={"charging_link":True})

    def _compute_routes(self):
        """ 计算路径 """
        logging.info(f"计算路径集合...")
        k = self._routes_per_od
        od_pairs = set()
        for veh in self.W.VEHICLES.values():
            o = veh.orig.name
            d = veh.dest.name
            od_pairs.add((o,d))
        
        self.dict_od_to_routes = {
            "uncharging": {},
            "charging": {}
        }

        for o, d in od_pairs:
            self.dict_od_to_routes["uncharging"][o,d] = self._enumerate_k_shortest_routes(o, d, k)
            self.dict_od_to_routes["charging"][o,d] = self._enumerate_k_shortest_charge_routes(o, d, k)
        
    def _enumerate_k_shortest_routes(self, source:str, target:str, k:int):
        """ 枚举无充电需求k最短路径 """
        G = nx.DiGraph()
        link_dict = {}
        for l in self.W.LINKS:
            if not l.attribute["charging_link"]:
                G.add_edge(l.start_node.name, l.end_node.name, weight=l.length/l.u)
                link_dict[l.start_node.name, l.end_node.name] = l.name
        k_shortest_paths = list(islice(nx.shortest_simple_paths(G, self.W.get_node(source).name, self.W.get_node(target).name, weight='weight'), k))
        routes = []
        for path in k_shortest_paths:
            route = [link_dict[path[i], path[i + 1]] for i in range(len(path) - 1)]
            routes.append(route)
        logging.debug(f"无充电需求路径：{source} -> {target} 数量：{len(routes)}")
        for route in routes:
            logging.debug(f"\t{route}")
        return routes
    
    def _enumerate_k_shortest_charge_routes(self, source:str, target:str, k:int):
        """ 枚举有充电需求k最短路径 """
        G = nx.DiGraph()
        link_dict = {}
        for l in self.W.LINKS:
            start_node, end_node, weight = l.start_node.name, l.end_node.name, l.length/l.u
            if not l.attribute["charging_link"]: # 非充电链路
                # 第一层图
                G.add_edge("0_" + start_node, "0_" + end_node, weight=weight)
                link_dict["0_" + start_node, "0_" + end_node] = l.name
                # 第二层图
                G.add_edge("1_" + start_node, "1_" + end_node, weight=weight)
                link_dict["1_" + start_node, "1_" + end_node] = l.name
            else: # 充电链路
                G.add_edge("0_" + start_node, "1_" + end_node, weight=weight)
                link_dict["0_" + start_node, "1_" + end_node] = l.name
        source_node = "0_" + source
        target_node = "1_" + target
        k_shortest_paths = list(islice(nx.shortest_simple_paths(G, source_node, target_node, weight='weight'), k))
        routes = []
        for path in k_shortest_paths:
            route = [link_dict[path[i], path[i + 1]] for i in range(len(path) - 1)]
            routes.append(route)
        logging.debug(f"充电需求路径：{source} -> {target} 数量：{len(routes)}")
        for route in routes:
            logging.debug(f"\t{route}")
        return routes
    
    def _init_charging_prices(self):
        logging.info(f"随机初始化充电价格...")
        self.charging_prices:list[dict[str, float]] = []
        
        for _ in range(self._charging_periods):
            period_prices = {}
            for node, bounds in self._charging_nodes.items():
                # 在价格区间内随机采样
                random_price = np.random.uniform(bounds[0], bounds[1])
                period_prices[node] = random_price
            self.charging_prices.append(period_prices)

    def draw_world_info(self) -> str:
        table = Texttable()
        table.add_rows([
            ["网络名称", self.network_name],
            ["节点数量", len(self.W.NODES)],
            ["链路数量", len(self.W.LINKS) - len(self._charging_nodes)],
            ["充电节点数量", len(self._charging_nodes)],
            ["OD对数量", len(self.dict_od_to_routes["uncharging"])],
            ["汽车数量", len(self.W.VEHICLES)],
        ],header=False)
        return table.draw()

    def draw_charging_prices(self) -> str:
        """ 绘制充电价格表格 """
        table = Texttable()
        
        title = ["时段\节点"]
        title.extend([node for node in self.charging_prices[0].keys()])
        table.header(title)
        
        for p, prices in enumerate(self.charging_prices):
            row = [f"第{p + 1}时段"]
            row.extend([price for price in prices.values()])
            table.add_row(row)
        
        return table.draw()
    
    def _update_prices_from_actions(self, actions: Dict[str, np.ndarray]):
        """ 将归一化动作映射到实际价格并更新充电价格 """
        new_prices = []
        
        for period in range(self._charging_periods):
            period_prices = {}
            for agent in self.agents:
                bounds = self._charging_nodes[agent]
                # 将[0,1]动作映射到[min_price, max_price]
                normalized_action = actions[agent][period]
                actual_price = bounds[0] + normalized_action * (bounds[1] - bounds[0])
                period_prices[agent] = actual_price
            new_prices.append(period_prices)
        
        self.charging_prices = new_prices
        # 记录到历史
        price_matrix = [[period_prices[agent] for agent in self.agents] for period_prices in new_prices]
        self.price_history.append(price_matrix)
    
    def _normalize_prices(self, price_matrix: list[list[float]]) -> np.ndarray:
        """ 将实际价格矩阵归一化到[0,1]区间 """
        normalized = np.zeros((self.n_agents, self.n_periods))
        
        for period in range(self.n_periods):
            for agent_idx, agent in enumerate(self.agents):
                bounds = self._charging_nodes[agent]
                actual_price = price_matrix[period][agent_idx]
                # 归一化: (价格 - 最小值) / (最大值 - 最小值)
                normalized[agent_idx, period] = (actual_price - bounds[0]) / (bounds[1] - bounds[0])
        
        return normalized
    
    def _check_convergence(self) -> bool:
        """ 检查价格是否收敛 """
        if len(self.price_history) < 2:
            return False
            
        # 计算所有智能体价格向量的变化
        current_prices = np.array(self.price_history[-1])
        previous_prices = np.array(self.price_history[-2])
        
        # 使用L2范数判断收敛
        price_change = np.linalg.norm(current_prices - previous_prices)
        
        return price_change < self.convergence_threshold
    
    def _run_simulation(self) -> Dict[str, np.ndarray]:
        """ 运行UXSim仿真并返回充电流量统计 
        
        TODO: 需要与用户讨论仿真实现细节
        """
        # 临时返回随机流量用于测试
        charging_flows = {}
        for agent in self.agents:
            charging_flows[agent] = np.random.rand(self.n_periods) * 100  # 随机流量
        return charging_flows
    
    def _calculate_rewards(self, charging_flows: Dict[str, np.ndarray]) -> Dict[str, float]:
        """ 计算各智能体的奖励（收益）"""
        rewards = {}
        
        for agent in self.agents:
            total_reward = 0.0
            for period in range(self.n_periods):
                price = self.charging_prices[period][agent]
                flow = charging_flows[agent][period]
                total_reward += price * flow
            rewards[agent] = total_reward
            
        return rewards


