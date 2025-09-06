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
        return max(0, min(floor(t / self._period_duration), self._charging_periods - 1))
    
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

            self.network_name:str = settings["network_name"]  # 网络名称
            
            self._simulation_time:float = float(settings["simulation_time"])  # 仿真总时长(秒)
            self._deltan:int = int(settings["deltan"])  # 车队大小(车辆数/platoon)

            self._charging_car_rate:float = float(settings["charging_car_rate"])  # 需要充电的车辆比例
            self._charging_link_length:float = float(settings["charging_link_length"])  # 充电链路长度(米)
            self._charging_link_free_flow_speed:float = float(settings["charging_link_free_flow_speed"])  # 充电链路自由流速度(米/秒)
            
            self._charging_periods:int = int(settings["charging_periods"])  # 动态定价时段数
            self._charging_nodes:dict[str,list[float]] = settings["charging_nodes"]  # 充电节点及价格边界
            self._period_duration:float = float(self._simulation_time / self._charging_periods)  # 每个定价时段的时长(秒)

            self._routes_per_od:int = int(settings["routes_per_od"])  # 每个OD对的候选路径数

            # UE-DTA 求解参数
            self._time_value_coefficient:float = float(settings["time_value_coefficient"])  # 时间价值系数(元/秒)
            self._charging_demand_per_vehicle:float = float(settings["charging_demand_per_vehicle"])  # 每辆充电车固定充电量(kWh)
            self._ue_convergence_threshold:float = float(settings["ue_convergence_threshold"])  # UE收敛阈值(元)
            self._ue_max_iterations:int = int(settings["ue_max_iterations"])  # UE最大迭代次数
            self._ue_swap_probability:float = float(settings["ue_swap_probability"])  # 路径切换概率

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
        
        # 为充电节点的出边添加充电功能
        logging.info(f"为充电节点添加充电功能...")
        charging_links_created = 0
        for node in self._charging_nodes.keys():
            # 找到从充电节点出发的第一条链路，标记为充电链路
            for link in self.W.LINKS:
                if link.start_node.name == node and not link.attribute.get("charging_link", False):
                    # 复制原链路作为充电版本
                    charging_link_name = f"charging_{link.name}"
                    self.W.addLink(
                        charging_link_name, 
                        link.start_node.name, 
                        link.end_node.name,
                        length=link.length,
                        free_flow_speed=link.u,
                        jam_density=link.kappa,
                        merge_priority=link.merge_priority,
                        attribute={"charging_link": True, "original_link": link.name}
                    )
                    charging_links_created += 1
                    break  # 每个充电节点只添加一条充电链路
        
        logging.info(f"创建了{charging_links_created}条充电链路")

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
        charging_links_added = 0
        
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
                # 充电链路连接第0层到第1层（强制充电行为建模）
                G.add_edge("0_" + start_node, "1_" + end_node, weight=weight)
                link_dict["0_" + start_node, "1_" + end_node] = l.name
                charging_links_added += 1
        
        # 验证充电链路已添加
        if charging_links_added == 0:
            logging.warning(f"双层图构建时未找到充电链路！")
        else:
            logging.debug(f"双层图构建完成，添加了{charging_links_added}条充电链路")
            
        source_node = "0_" + source
        target_node = "1_" + target
        
        # 确保目标节点在第1层图中存在
        if target_node not in G.nodes():
            logging.warning(f"目标节点{target_node}不在双层图中！")
        
        try:
            k_shortest_paths = list(islice(nx.shortest_simple_paths(G, source_node, target_node, weight='weight'), k))
        except nx.NetworkXNoPath:
            logging.warning(f"无法找到从{source}到{target}的充电路径！")
            # 返回普通路径作为后备方案
            return self._enumerate_k_shortest_routes(source, target, k)
            
        routes = []
        for path in k_shortest_paths:
            route = [link_dict[path[i], path[i + 1]] for i in range(len(path) - 1)]
            # 验证路径是否包含充电链路
            has_charging = any("charging_" in link_name for link_name in route)
            if has_charging:
                routes.append(route)
            else:
                logging.warning(f"充电路径{route}不包含充电链路！")
                
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
        """ 运行基于day-to-day动态均衡的UXSim仿真并返回充电流量统计 """
        logging.info("开始UE-DTA求解...")
        
        # 获取充电和非充电车辆的OD映射
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        W_template = self._create_simulation_world()
        
        for key, veh in W_template.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute["charging_car"]:
                dict_od_to_charging_vehid[o,d].append(key)
            else:
                dict_od_to_uncharging_vehid[o,d].append(key)
        
        # 初始化路径分配（如果是第一次调用）
        if not hasattr(self, 'current_routes_specified'):
            self.current_routes_specified = self._initialize_routes(dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True)
        
        # Day-to-day迭代求解
        for iteration in range(self._ue_max_iterations):
            # 创建新的仿真实例
            W = self._create_simulation_world()
            
            # 应用当前路径分配
            self._apply_routes_to_vehicles(W, self.current_routes_specified)
            
            # 执行仿真
            W.exec_simulation()
            
            # 检查未完成车辆
            unfinished_trips = W.analyzer.trip_all - W.analyzer.trip_completed
            if unfinished_trips > 0:
                logging.warning(f"第{iteration+1}轮：{unfinished_trips}/{W.analyzer.trip_all} 车辆未完成行程")
            
            # 计算成本差并执行路径切换
            total_cost_gap, new_routes_specified = self._route_choice_update(W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid)
            average_cost_gap = total_cost_gap / len(W.VEHICLES)
            
            # 动态显示进度
            print(f"\r第{iteration+1}轮：平均成本差={average_cost_gap:.2f}元，总行程时间={W.analyzer.total_travel_time:.1f}秒", end="", flush=True)
            
            # 更新路径分配
            self.current_routes_specified = new_routes_specified
            
            # 收敛判断
            if average_cost_gap < self._ue_convergence_threshold:
                print(f"\nUE-DTA在第{iteration+1}轮收敛，平均成本差={average_cost_gap:.2f}元")
                break
        
        # 如果未收敛
        if iteration == self._ue_max_iterations - 1:
            print(f"\nUE-DTA达到最大迭代次数{self._ue_max_iterations}，平均成本差={average_cost_gap:.2f}元")
        
        # 统计充电流量
        charging_flows = self._extract_charging_flows(W)
        
        # 统计完成行程的充电车辆数量
        completed_charging_vehicles = 0
        total_charging_vehicles = 0
        for veh in W.VEHICLES.values():
            if veh.attribute["charging_car"]:
                total_charging_vehicles += 1
                if veh.state == "end":
                    completed_charging_vehicles += 1
        
        logging.info(f"充电车辆统计：{completed_charging_vehicles}/{total_charging_vehicles} 完成行程")
        
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
    
    def _create_simulation_world(self) -> World:
        """ 创建配置了当前充电价格的仿真世界实例 """
        W = World(
            name=self.network_name + f"_step{self.current_step}",
            deltan=self._deltan,
            tmax=self._simulation_time,
            print_mode=0,  # 静默模式
            save_mode=0,   # 不保存文件
            show_mode=0,
            random_seed=self.env_random_seed if self.env_random_seed is not None else 0,
            user_attribute={}
        )
        
        # 复制网络结构（节点、链路、需求）
        self._copy_network_structure(W)
        
        return W
    
    def _initialize_routes(self, dict_od_to_charging_vehid: defaultdict, dict_od_to_uncharging_vehid: defaultdict, use_greedy: bool = True) -> Dict[str, list]:
        """ 为所有车辆分配初始路径
        
        Args:
            dict_od_to_charging_vehid: 充电车辆的OD映射
            dict_od_to_uncharging_vehid: 非充电车辆的OD映射  
            use_greedy: True使用贪心策略(最短路径)，False使用随机策略
        """
        routes_specified = {}
        
        if use_greedy:
            # 贪心策略：选择最短路径
            for od_pair, veh_ids in dict_od_to_charging_vehid.items():
                best_route = self.dict_od_to_routes["charging"][od_pair][0]  # 第一条就是最短的
                for veh_id in veh_ids:
                    routes_specified[veh_id] = best_route
            
            for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
                best_route = self.dict_od_to_routes["uncharging"][od_pair][0]  # 第一条就是最短的
                for veh_id in veh_ids:
                    routes_specified[veh_id] = best_route
                    
            logging.info(f"初始化路径分配（贪心策略）：充电{len(dict_od_to_charging_vehid)}个OD对，非充电{len(dict_od_to_uncharging_vehid)}个OD对")
            
        else:
            # 随机策略：随机选择路径
            for od_pair, veh_ids in dict_od_to_charging_vehid.items():
                available_routes = self.dict_od_to_routes["charging"][od_pair]
                for veh_id in veh_ids:
                    random_route_idx = np.random.choice(len(available_routes))
                    routes_specified[veh_id] = available_routes[random_route_idx]
            
            for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
                available_routes = self.dict_od_to_routes["uncharging"][od_pair]
                for veh_id in veh_ids:
                    random_route_idx = np.random.choice(len(available_routes))
                    routes_specified[veh_id] = available_routes[random_route_idx]
                    
            logging.info(f"初始化路径分配（随机策略）：充电{len(dict_od_to_charging_vehid)}个OD对，非充电{len(dict_od_to_uncharging_vehid)}个OD对")
        
        return routes_specified
    
    def _calculate_actual_vehicle_cost(self, veh, W: World) -> float:
        """ 计算车辆实际总成本（基于已行驶路径）"""
        route, timestamps = veh.traveled_route()
        
        # 计算时间成本
        travel_time = timestamps[-1] - timestamps[0]
        time_cost = self._time_value_coefficient * travel_time
        
        # 计算充电成本（仅充电车辆）
        charging_cost = 0.0
        if veh.attribute["charging_car"]:
            for i, link in enumerate(route):
                if link.attribute.get("charging_link", False):
                    # 获取进入充电链路的时刻
                    charging_entry_time = timestamps[i]
                    # 从链路名称提取充电节点
                    if "charging_" in link.name:
                        # 从 charging_X-Y 格式提取充电节点X
                        charging_node = link.name.split("_")[1].split("-")[0]
                    else:
                        continue
                    # 获取该时刻的充电价格
                    charging_price = self._get_price(charging_entry_time, charging_node)
                    # 计算充电成本
                    charging_cost = charging_price * self._charging_demand_per_vehicle
                    break  # 每辆车只充电一次
        
        total_cost = time_cost + charging_cost
        return total_cost
    
    def _estimate_route_cost(self, route_obj, departure_time: float, is_charging_vehicle: bool) -> float:
        """ 估算路径的总成本（预期成本）"""
        current_time = departure_time
        charging_cost = 0.0
        
        for link in route_obj.links:
            link_travel_time = link.actual_travel_time(current_time)
            
            # 仅为充电车辆计算充电成本
            if is_charging_vehicle and link.attribute.get("charging_link", False):
                if "charging_" in link.name:
                    # 从 charging_X-Y 格式提取充电节点X
                    charging_node = link.name.split("_")[1].split("-")[0]
                    charging_price = self._get_price(current_time, charging_node)
                    charging_cost = charging_price * self._charging_demand_per_vehicle
            
            current_time += link_travel_time
        
        total_travel_time = current_time - departure_time
        time_cost = self._time_value_coefficient * total_travel_time
        
        return time_cost + charging_cost
    
    def _route_choice_update(self, W: World, dict_od_to_charging_vehid: defaultdict, dict_od_to_uncharging_vehid: defaultdict) -> tuple[float, Dict[str, list]]:
        """ 执行路径选择与切换逻辑，返回总成本差和新路径分配 """
        new_routes_specified = {}
        total_cost_gap = 0.0
        
        # 记录路径选择的详细信息
        charging_vehicles_processed = 0
        route_switches = 0
        
        # 为充电车辆执行路径选择
        for od_pair, veh_ids in dict_od_to_charging_vehid.items():
            available_routes = self.dict_od_to_routes["charging"][od_pair]
            
            for veh_id in veh_ids:
                veh = W.VEHICLES[veh_id]
                charging_vehicles_processed += 1
                
                # 跳过未完成行程的车辆
                if veh.state != "end":
                    new_routes_specified[veh_id] = self.current_routes_specified[veh_id]  # 保持原分配路径
                    continue
                
                # 计算当前路径的实际成本
                current_cost = self._calculate_actual_vehicle_cost(veh, W)
                current_route = self.current_routes_specified[veh_id]  # 使用原分配的路径
                
                # 详细日志（仅前3辆车）
                if charging_vehicles_processed <= 3:
                    logging.info(f"充电车辆{veh.name}({od_pair}): 当前路径{current_route}, 实际成本{current_cost:.2f}元")
                
                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route
                
                for route_links in available_routes:
                    # 计算替代路径的预期成本
                    route_obj = W.defRoute(route_links)
                    alt_cost = self._estimate_route_cost(route_obj, veh.departure_time_in_second, True)
                    
                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_links
                
                # 计算成本差
                cost_gap = current_cost - best_cost
                total_cost_gap += cost_gap
                
                # 路径切换决策（随机切换机制）
                switched = False
                if cost_gap > 0 and np.random.random() < self._ue_swap_probability:
                    new_routes_specified[veh_id] = best_route
                    switched = True
                    route_switches += 1
                else:
                    new_routes_specified[veh_id] = current_route
                
                # 详细日志（仅前3辆车）
                if charging_vehicles_processed <= 3:
                    logging.info(f"  最优路径{best_route}, 最优成本{best_cost:.2f}元, 成本差{cost_gap:.2f}元, 切换={switched}")
        
        # 为非充电车辆执行路径选择
        for od_pair, veh_ids in dict_od_to_uncharging_vehid.items():
            available_routes = self.dict_od_to_routes["uncharging"][od_pair]
            
            for veh_id in veh_ids:
                veh = W.VEHICLES[veh_id]
                
                # 跳过未完成行程的车辆
                if veh.state != "end":
                    new_routes_specified[veh_id] = self.current_routes_specified[veh_id]  # 保持原分配路径
                    continue
                
                # 计算当前路径的实际成本
                current_cost = self._calculate_actual_vehicle_cost(veh, W)
                current_route = self.current_routes_specified[veh_id]  # 使用原分配的路径
                
                # 寻找最优替代路径
                best_cost = current_cost
                best_route = current_route
                
                for route_links in available_routes:
                    # 计算替代路径的预期成本
                    route_obj = W.defRoute(route_links)
                    alt_cost = self._estimate_route_cost(route_obj, veh.departure_time_in_second, False)
                    
                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_route = route_links
                
                # 计算成本差
                cost_gap = current_cost - best_cost
                total_cost_gap += cost_gap
                
                # 路径切换决策
                if cost_gap > 0 and np.random.random() < self._ue_swap_probability:
                    new_routes_specified[veh_id] = best_route
                else:
                    new_routes_specified[veh_id] = current_route
        
        logging.info(f"路径选择完成：处理{charging_vehicles_processed}辆充电车辆，{route_switches}次路径切换")
        return total_cost_gap, new_routes_specified
    
    def _extract_charging_flows(self, W: World) -> Dict[str, np.ndarray]:
        """ 从仿真结果中提取各充电站按时段的流量统计 """
        charging_flows = {agent: np.zeros(self.n_periods) for agent in self.agents}
        
        charging_vehicle_count = 0
        found_charging_links = 0
        
        # 首先验证World中的充电链路
        charging_links_in_world = [link for link in W.LINKS if link.attribute.get("charging_link", False)]
        logging.debug(f"World中的充电链路数量：{len(charging_links_in_world)}")
        for link in charging_links_in_world[:3]:  # 只显示前3个
            logging.debug(f"  充电链路: {link.name}, 属性: {link.attribute}")
        
        # 遍历所有车辆，统计充电流量
        for veh in W.VEHICLES.values():
            # 只统计充电车辆且已完成行程的车辆
            if not veh.attribute["charging_car"] or veh.state != "end":
                continue
                
            charging_vehicle_count += 1
            route, timestamps = veh.traveled_route()
            
            # 添加详细的路径调试
            if charging_vehicle_count <= 3:  # 只调试前3辆车
                actual_route_names = [link.name for link in route]
                assigned_route = self.current_routes_specified.get(veh.name, [])
                logging.info(f"充电车辆{veh.name}:")
                logging.info(f"  分配路径: {assigned_route}")
                logging.info(f"  实际路径: {actual_route_names}")
                
            # 找到充电链路和进入时刻
            for i, link in enumerate(route):
                if link.attribute.get("charging_link", False):
                    found_charging_links += 1
                    charging_entry_time = timestamps[i]
                    
                    # 从链路名称提取充电节点
                    if "charging_" in link.name:
                        # 从 charging_X-Y 格式提取充电节点X
                        charging_node = link.name.split("_")[1].split("-")[0]
                    else:
                        continue  # 跳过不是充电链路的
                        
                    charging_period = self._get_period(charging_entry_time)
                    
                    # 累加该时段的充电流量（每个Vehicle对象代表DELTAN辆实际车辆）
                    charging_flows[charging_node][charging_period] += W.DELTAN
                    logging.info(f"车辆{veh.name}在时刻{charging_entry_time:.1f}({charging_period}时段)使用充电站{charging_node}")
                    break  # 每辆车只计算一次充电
        
        logging.info(f"充电流量统计：{charging_vehicle_count}辆完成充电车辆中，{found_charging_links}辆找到充电链路")
        logging.info(f"各站充电流量：{dict(charging_flows)}")
        
        # 转换为浮点数数组
        for agent in self.agents:
            charging_flows[agent] = charging_flows[agent].astype(np.float32)
            
        return charging_flows
    
    def _copy_network_structure(self, W: World):
        """ 将原始网络结构复制到新的World实例中 """
        
        # 复制节点
        for node in self.W.NODES:
            W.addNode(node.name, node.x, node.y)
        
        # 复制普通链路
        for link in self.W.LINKS:
            if not link.attribute.get("charging_link", False):
                W.addLink(
                    link.name, link.start_node.name, link.end_node.name,
                    length=link.length,
                    free_flow_speed=link.u,
                    jam_density=link.kappa,
                    merge_priority=link.merge_priority,
                    attribute=link.attribute.copy()
                )
        
        # 复制充电链路
        charging_links_created = 0
        for link in self.W.LINKS:
            if link.attribute.get("charging_link", False):
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
                charging_links_created += 1
        logging.debug(f"为新World实例复制了{charging_links_created}条充电链路")
        
        # 复制交通需求
        for veh in self.W.VEHICLES.values():
            W.addVehicle(
                veh.orig.name, veh.dest.name, 
                veh.departure_time,
                departure_time_is_time_step=True,
                attribute=veh.attribute.copy()
            )
    
    def _apply_routes_to_vehicles(self, W: World, routes_specified: Dict[str, list]):
        """ 为车辆分配指定的路径 """
        charging_route_count = 0
        total_route_count = 0
        
        for veh_id, route_links in routes_specified.items():
            if veh_id in W.VEHICLES:
                total_route_count += 1
                veh = W.VEHICLES[veh_id]
                
                # 检查是否是充电车辆且路径包含充电链路
                if veh.attribute["charging_car"]:
                    # 确保route_links是字符串列表
                    route_names = [link if isinstance(link, str) else link.name for link in route_links]
                    has_charging_link = any("charging_" in link_name for link_name in route_names)
                    if has_charging_link:
                        charging_route_count += 1
                        
                # 应用路径
                veh.enforce_route(route_links)
        
        logging.info(f"路径分配完成：总计{total_route_count}辆，充电路径{charging_route_count}条")


