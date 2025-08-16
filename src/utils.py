import csv
import json
from math import floor
import os

import uxsim


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

# 获取 t 时刻对应的电价时间段
def get_period(W:uxsim.World, t:float):
    return min(floor(t / W.TMAX * W.EVCS["charging_periods"]), W.EVCS["charging_periods"] - 1)

# 获取 t 时刻 node 的电价
def get_price(W:uxsim.World, t:float, node:str):
    return W.EVCS["charging_prices"][get_period(W, t)][node]

# 设置 p 时段节点 node 的电价
def set_price(W:uxsim.World, p:int, node:str, price:float):
    W.EVCS["charging_prices"][p][node] = price
    




    

