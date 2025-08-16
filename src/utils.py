import csv
import json
from math import floor
import os

import uxsim
import networkx as nx
from itertools import islice


def load_network(network_dir:str, network_name:str):
    """ 加载网络 
    network_dir: 路网文件夹路径
    network_name: 路网名称
    """
    
    settings_path = os.path.join(network_dir, f"{network_name}_settings.json")
    node_path = os.path.join(network_dir, f"{network_name}_nodes.csv")
    link_path = os.path.join(network_dir, f"{network_name}_links.csv")
    demand_path = os.path.join(network_dir, f"{network_name}_demand.csv")

    with open(settings_path, "r") as f:
        settings = json.load(f)
    
    # 创建模拟环境
    W = uxsim.World(
        name=settings["network_name"],
        deltan=settings["deltan"],
        tmax=settings["simulation_time"],
        print_mode=1,
        save_mode=1,
        show_mode=0,
        random_seed=settings["random_seed"],
        user_attribute={}
    )
    # 设置 EVCS 相关参数
    W.EVCS = {
        "charging_periods": settings["charging_periods"],
        "period_duration": settings["simulation_time"] / settings["charging_periods"],
        "charging_nodes": settings["charging_nodes"],
    }
    # 初始化电价列表
    W.EVCS["charging_prices"] = [
        {
            node : 1.0 for node in W.EVCS["charging_nodes"]
        } for _ in range(settings["charging_periods"])
    ]
    # 加载节点
    with open(node_path, "r") as f:
        for r in csv.reader(f):
            if r[1] != "x":
                name, x, y = r
                W.addNode(name, float(x), float(y))
    # 加载边
    with open(link_path, "r") as f:
        for r in csv.reader(f):
            if r[3] != "length":
                W.addLink(
                    r[0], r[1], r[2], # name start_node end_node
                    length=float(r[3]), 
                    free_flow_speed=float(r[4]), 
                    jam_density=float(r[5]), 
                    merge_priority=float(r[6]),
                    attribute={"charging_link" : False})
    
    # 加载需求
    charging_car_rate = settings["charging_car_rate"]
    with open(demand_path, "r") as f:
        for r in csv.reader(f):
            if r[2] != "start_t":
                try:
                    W.adddemand(r[0], r[1], float(r[2]), float(r[3]), 
                                float(r[4]) * charging_car_rate, 
                                float(r[5]) * charging_car_rate, 
                                attribute={"charging_car":True})
                    W.adddemand(r[0], r[1], float(r[2]), float(r[3]),
                                float(r[4]) * (1 - charging_car_rate),
                                float(r[5]) * (1 - charging_car_rate),
                                attribute={"charging_car":False})
                except:
                    W.adddemand(r[0], r[1], float(r[2]), float(r[3]), 
                                float(r[4]) * charging_car_rate, 
                                attribute={"charging_car":True})
                    W.adddemand(r[0], r[1], float(r[2]), float(r[3]),
                                float(r[4]) * (1 - charging_car_rate),
                                attribute={"charging_car":False})
    # 添加虚拟充电链路
    for node in W.EVCS["charging_nodes"]:
        W.addLink(
            "charging_" + node, node, node,
            length=settings["charging_link_length"],
            free_flow_speed=settings["charging_link_free_flow_speed"],
            attribute={"charging_link":True})
    
    return W

def compute_routes(W:uxsim.World, k:int=1):
    """计算路径集合
    W: 模拟环境
    k: 路径数量
    """
    
    # 无充电需求路径集合
    def enumerate_k_shortest_routes(W:uxsim.World, source:str, target:str, k:int=1):
        G = nx.DiGraph()
        link_dict = {}
        for l in W.LINKS:
            if not l.attribute["charging_link"]:
                G.add_edge(l.start_node.name, l.end_node.name, weight=l.length/l.u)
                link_dict[l.start_node.name, l.end_node.name] = l.name
        k_shortest_paths = list(islice(nx.shortest_simple_paths(G, W.get_node(source).name, W.get_node(target).name, weight='weight'), k))
        routes = []
        for path in k_shortest_paths:
            route = [link_dict[path[i], path[i + 1]] for i in range(len(path) - 1)]
            routes.append(route)
        return routes
    
    # 有充电需求路径集合
    def enumerate_k_shortest_charge_routes(W:uxsim.World, source:str, target:str, k:int=1):
        G = nx.DiGraph()
        link_dict = {}
        for l in W.LINKS:
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
        return routes
    
    n_routes_per_od = k
    od_pairs = set()
    for veh in W.VEHICLES.values():
        o = veh.orig.name
        d = veh.dest.name
        od_pairs.add((o,d))
    
    dict_od_to_routes = {
        "uncharging": {},
        "charging": {}
    }
    for o, d in od_pairs:
        dict_od_to_routes["uncharging"][o,d] = enumerate_k_shortest_routes(W, o, d, n_routes_per_od)
        dict_od_to_routes["charging"][o,d] = enumerate_k_shortest_charge_routes(W, o, d, n_routes_per_od)
    
    return dict_od_to_routes

# 获取 t 时刻对应的电价时间段
def get_period(W:uxsim.World, t:float):
    return max(0,
        min(floor(t / W.EVCS["period_duration"]),
            W.EVCS["charging_periods"] - 1))

# 获取 t 时刻 node 的电价
def get_price(W:uxsim.World, t:float, node:str):
    return W.EVCS["charging_prices"][get_period(W, t)][node]

# 设置 p 时段节点 node 的电价
def set_price(W:uxsim.World, p:int, node:str, price:float):
    W.EVCS["charging_prices"][p][node] = price
    




    

