"""
UXsim++ Extended - 支持预定路径的交通仿真库

基于 UXsim++ 扩展，添加预定路径功能以支持充电博弈环境。
"""

import random
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import sys

################################
# 从 C++ 扩展模块导入类和函数
from . import trafficppy

# 保存原始 C++ 类的引用
_CppWorld = trafficppy.World
_CppLink = trafficppy.Link
_CppNode = trafficppy.Node
_CppVehicle = trafficppy.Vehicle
create_world = trafficppy.create_world
add_node = trafficppy.add_node
add_link = trafficppy.add_link
add_demand = trafficppy.add_demand

# 暂时保留原始类引用，稍后会被包装类覆盖
Link = _CppLink
Node = _CppNode

################################
# 兼容性包装：VEHICLES 字典化访问
################################

class VehiclesDict:
    """
    将 C++ vector<Vehicle*> 包装为类字典访问
    支持 W.VEHICLES[key], W.VEHICLES.values(), W.VEHICLES.items(), len(), in 等操作

    性能优化：使用字典缓存避免 O(n²) 复杂度
    - 缓存在车辆数量变化时自动失效并重建
    - 避免每次 [] 或 in 操作都重建整个字典
    """
    def __init__(self, world):
        self._world = world
        self._cached_dict = None  # 缓存的 {name: vehicle} 字典
        self._cache_len = -1      # 缓存时的车辆数量，用于检测失效
        self._cpp_vehicles_ref = None  # 缓存 C++ vehicles 引用

    def _get_cpp_vehicles(self):
        """获取 C++ vehicles vector（缓存引用避免重复 pybind11 调用）"""
        if self._cpp_vehicles_ref is None:
            self._cpp_vehicles_ref = self._world._cpp_world.VEHICLES
        return self._cpp_vehicles_ref

    def _get_dict(self):
        """获取字典（带缓存，车辆数量变化时自动重建）"""
        cpp_vehicles = self._get_cpp_vehicles()
        current_len = len(cpp_vehicles)
        if self._cached_dict is None or self._cache_len != current_len:
            # 缓存失效，重建字典
            self._cached_dict = {v.name: v for v in cpp_vehicles}
            self._cache_len = current_len
        return self._cached_dict

    def invalidate_cache(self):
        """手动使缓存失效（添加/删除车辆后调用）"""
        self._cached_dict = None
        self._cache_len = -1
        self._cpp_vehicles_ref = None

    def __getitem__(self, key):
        # 支持整数索引和字符串名称
        if isinstance(key, int):
            # 整数索引：直接访问 C++ vector
            vehicles = self._get_cpp_vehicles()
            if key < 0:
                key = len(vehicles) + key
            if 0 <= key < len(vehicles):
                return vehicles[key]
            raise IndexError(f"Vehicle index {key} out of range")
        else:
            # 字符串名称：使用缓存字典
            return self._get_dict()[key]

    def __contains__(self, key):
        return key in self._get_dict()

    def __len__(self):
        return len(self._get_cpp_vehicles())

    def __iter__(self):
        return iter(self._get_dict())

    def keys(self):
        return self._get_dict().keys()

    def values(self):
        # 直接返回 C++ vector 的所有元素，避免同名车辆被覆盖
        # 注意：返回 list 而非 dict_values，因为可能存在同名车辆
        return list(self._get_cpp_vehicles())

    def items(self):
        return self._get_dict().items()

    def get(self, key, default=None):
        return self._get_dict().get(key, default)


################################
# 兼容性包装：World 类
################################

class World:
    """
    World 包装类 - 兼容 EVCSChargingGameEnv 的调用方式

    支持的参数（兼容 UXsim 1.8.2 风格）:
        name, deltan, tmax, random_seed, print_mode, save_mode, show_mode, user_attribute
    """
    def __init__(self,
                 name="",
                 tmax=3600,
                 deltan=5,
                 tau=1,
                 duo_update_time=600,
                 duo_update_weight=0.5,
                 print_mode=1,
                 random_seed=None,
                 vehicle_detailed_log=1,
                 # 以下参数为兼容性保留，不实际使用
                 save_mode=0,
                 show_mode=0,
                 user_attribute=None):
        """
        创建仿真世界

        Parameters
        ----------
        name : str
            世界名称
        tmax : float
            仿真总时长（秒）
        deltan : int
            车队大小（每个Vehicle对象代表的实际车辆数）
        tau : float
            反应时间
        duo_update_time : float
            DUO路径选择更新间隔
        duo_update_weight : float
            DUO更新权重
        print_mode : int
            是否打印仿真进度
        random_seed : int or None
            随机种子
        vehicle_detailed_log : int
            是否记录车辆详细日志
        save_mode : int
            （兼容参数，忽略）
        show_mode : int
            （兼容参数，忽略）
        user_attribute : dict
            （兼容参数，忽略）
        """
        if random_seed is None:
            random_seed = random.randint(0, 2**8)

        # 调用 C++ create_world 创建底层 World
        self._cpp_world = create_world(
            name,              # world_name
            tmax,              # t_max
            deltan,            # delta_n
            tau,               # tau
            duo_update_time,   # duo_update_time
            duo_update_weight, # duo_update_weight
            0,                 # route_choice_uncertainty
            int(print_mode),   # print_mode
            random_seed,       # random_seed
            vehicle_detailed_log,  # vehicle_log_mode
        )

        # 保存兼容性属性
        self.user_attribute = user_attribute if user_attribute is not None else {}

        # ========== 关键修复：保持对所有 Vehicle 对象的 Python 引用 ==========
        # pybind11 的 Vehicle 对象如果没有 Python 端引用，会被垃圾回收，
        # 导致 C++ 端的 World::vehicles 中出现悬垂指针。
        # 这个列表确保所有创建的 Vehicle 在 World 生命周期内保持有效。
        self._vehicle_refs = []

        # ========== 关键修复：缓存 Link 对象的 Python 引用 ==========
        # pybind11 每次访问 C++ 对象时可能返回不同的 Python 包装对象，
        # 导致动态属性（如 attribute）丢失。
        # 这个字典缓存 {link_name: link_obj}，确保每次获取同一个链路时返回同一个 Python 对象。
        self._link_refs = {}

        # 创建字典化的 VEHICLES 访问器
        # 注意：不再缓存 _cpp_vehicles，VehiclesDict 每次动态获取最新数据
        self._vehicles_dict = VehiclesDict(self)

    # ========== 代理 C++ World 的属性 ==========
    @property
    def VEHICLES(self):
        """返回字典化的 Vehicles 访问器"""
        return self._vehicles_dict

    @property
    def LINKS(self):
        """返回缓存的 Links 列表，确保动态属性（如 attribute）保持有效"""
        # 遍历 C++ LINKS，确保每个 Link 都在缓存中
        for cpp_link in self._cpp_world.LINKS:
            if cpp_link.name not in self._link_refs:
                # 新链路，添加到缓存（这种情况不应该发生，因为 addLink 会添加）
                self._link_refs[cpp_link.name] = cpp_link
        # 返回缓存中的链路列表（按 C++ 顺序）
        return [self._link_refs.get(l.name, l) for l in self._cpp_world.LINKS]

    @property
    def NODES(self):
        return self._cpp_world.NODES

    @property
    def name(self):
        return self._cpp_world.name

    @property
    def delta_t(self):
        return self._cpp_world.delta_t

    @property
    def DELTAT(self):
        return self._cpp_world.delta_t

    @property
    def deltan(self):
        return self._cpp_world.deltan

    @property
    def t_max(self):
        return self._cpp_world.t_max

    @property
    def TMAX(self):
        return self._cpp_world.t_max

    @property
    def time(self):
        return self._cpp_world.time

    @property
    def timestep(self):
        return self._cpp_world.timestep

    # ========== 代理 C++ World 的方法 ==========
    def get_node(self, name):
        return self._cpp_world.get_node(name)

    def get_link(self, name):
        """获取链路（从缓存中返回，确保动态属性保持有效）"""
        if name in self._link_refs:
            return self._link_refs[name]
        # 缓存中没有，从 C++ 获取并缓存
        link = self._cpp_world.get_link(name)
        if link is not None:
            self._link_refs[name] = link
        return link

    def get_vehicle(self, name):
        return self._cpp_world.get_vehicle(name)

    def initialize_adj_matrix(self):
        return self._cpp_world.initialize_adj_matrix()

    def main_loop(self, duration_t=-1, until_t=-1):
        return self._cpp_world.main_loop(duration_t, until_t)

    def check_simulation_ongoing(self):
        return self._cpp_world.check_simulation_ongoing()

    def print_scenario_stats(self):
        return self._cpp_world.print_scenario_stats()

    def print_simple_results(self):
        return self._cpp_world.print_simple_results()

    def update_adj_time_matrix(self):
        return self._cpp_world.update_adj_time_matrix()


################################
# 兼容性包装：Vehicle 类
################################

class Vehicle:
    """
    Vehicle 包装类 - 兼容 EVCSChargingGameEnv 的调用方式

    支持两种调用方式:
    1. Vehicle(W, orig, dest, departure_time, ...) - EVCSChargingGameEnv 风格
    2. Vehicle(W, name, departure_time, orig, dest) - C++ 风格（通过位置参数）
    """
    def __new__(cls, W, arg1, arg2, arg3=None, arg4=None,
                predefined_route=None,
                departure_time_is_time_step=False,
                attribute=None,
                name=None):
        """
        智能构造函数，自动识别调用方式并创建 C++ Vehicle

        调用方式1 (EVCSChargingGameEnv 风格):
            Vehicle(W, orig, dest, departure_time, ...)

        调用方式2 (uxsimpp.py adddemand 风格):
            Vehicle(W, name, departure_time, orig, dest)
        """
        # 获取底层 C++ World
        cpp_world = W._cpp_world if isinstance(W, World) else W

        # 判断调用方式：如果 arg3 是数字，则是 EVCSChargingGameEnv 风格
        # 如果 arg2 是数字，则是 C++ 风格
        if arg4 is not None:
            # 4个位置参数：C++ 风格 (W, name, departure_time, orig, dest)
            veh_name = arg1
            departure_time = arg2
            orig = arg3
            dest = arg4
        elif arg3 is not None:
            # 3个位置参数：EVCSChargingGameEnv 风格 (W, orig, dest, departure_time)
            orig = arg1
            dest = arg2
            departure_time = arg3
            # 自动生成名称
            veh_name = name if name else f"{orig}-{dest}-{departure_time}"
        else:
            # 只有2个位置参数：可能是错误调用
            raise ValueError("Vehicle() requires at least 3 positional arguments")

        # 处理节点类型
        if hasattr(orig, 'name'):
            orig = orig.name
        if hasattr(dest, 'name'):
            dest = dest.name

        # 创建 C++ Vehicle
        cpp_veh = _CppVehicle(cpp_world, veh_name, departure_time, orig, dest)

        # ========== 关键修复：保持 Python 引用防止垃圾回收 ==========
        # 需要从 World 包装类中获取 _vehicle_refs
        if isinstance(W, World):
            W._vehicle_refs.append(cpp_veh)

        # Python 层附加 attribute
        cpp_veh.attribute = attribute if attribute is not None else {}

        # 如果提供了预定路径，分配路径
        if predefined_route is not None:
            route_names = []
            for link in predefined_route:
                if hasattr(link, 'name'):
                    route_names.append(link.name)
                else:
                    route_names.append(link)
            cpp_veh.assign_route_by_name(route_names)

        # 直接返回 C++ Vehicle 对象（不创建包装实例）
        return cpp_veh


################################

#####################################################
# 场景定义函数

def newWorld(name="",
             tmax=3600, deltan=5, tau=1,
             duo_update_time=600, duo_update_weight=0.5,
             print_mode=True,
             random_seed=None,
             vehicle_detailed_log=1):
    """
    创建仿真世界（使用 World 包装类）

    Parameters
    ----------
    name : str, optional
        世界名称，默认为空字符串
    tmax : float, optional
        仿真总时长，默认3600秒
    deltan : int, optional
        车队大小（每个Vehicle对象代表的实际车辆数），默认5辆
    tau : float, optional
        反应时间，默认1秒
    duo_update_time : float, optional
        DUO路径选择更新间隔，默认600秒
    duo_update_weight : float, optional
        DUO更新权重，默认0.5
    print_mode : bool, optional
        是否打印仿真进度，默认True
    random_seed : int or None, optional
        随机种子，默认None（自动生成）
    vehicle_detailed_log : int, optional
        是否记录车辆详细日志，默认1

    Returns
    -------
    World
        仿真世界对象（包装类）
    """
    return World(
        name=name,
        tmax=tmax,
        deltan=deltan,
        tau=tau,
        duo_update_time=duo_update_time,
        duo_update_weight=duo_update_weight,
        print_mode=int(print_mode),
        random_seed=random_seed,
        vehicle_detailed_log=vehicle_detailed_log
    )


def _get_cpp_world(W):
    """获取底层 C++ World 对象"""
    return W._cpp_world if isinstance(W, World) else W

def addNode(W, name, x, y, signal_intervals=[0], signal_offset=0):
    """
    添加节点

    Parameters
    ----------
    W : World
        仿真世界
    name : str
        节点名称
    x : float
        x坐标
    y : float
        y坐标
    signal_intervals : list of float, optional
        信号灯各相位时长，默认[0]（无信号灯）
    signal_offset : float, optional
        信号偏移，默认0

    Returns
    -------
    Node
        创建的节点
    """
    cpp_world = _get_cpp_world(W)
    add_node(cpp_world, name, x, y, signal_intervals, signal_offset)
    return W.get_node(name)
World.addNode = addNode

def addLink(W, name, start_node, end_node, length, free_flow_speed=20, jam_density=0.2, merge_priority=1, capacity_out=-1, signal_group=[0], attribute=None):
    """
    添加链路

    Parameters
    ----------
    W : World
        仿真世界
    name : str
        链路名称
    start_node : str or Node
        起始节点
    end_node : str or Node
        终止节点
    length : float
        链路长度
    free_flow_speed : float, optional
        自由流速度，默认20 m/s
    jam_density : float, optional
        拥挤密度，默认0.2 veh/m
    merge_priority : float, optional
        合流优先级，默认1
    capacity_out : float, optional
        出口通行能力，默认-1（无限制）
    signal_group : int or list, optional
        信号组，默认[0]
    attribute : any, optional
        用户自定义属性（元数据），默认None

    Returns
    -------
    Link
        创建的链路
    """
    if hasattr(start_node, 'name'):
        start_node = start_node.name
    if hasattr(end_node, 'name'):
        end_node = end_node.name

    if isinstance(signal_group, Iterable) and not isinstance(signal_group, (str, bytes)):
        signal_group = list(signal_group)
    else:
        signal_group = [signal_group]

    cpp_world = _get_cpp_world(W)
    add_link(cpp_world, name, start_node, end_node, free_flow_speed, jam_density, length, merge_priority, capacity_out, signal_group)
    link = W.get_link(name)
    # Python 层附加 attribute（C++ 层不处理）
    link.attribute = attribute if attribute is not None else {}
    return link
World.addLink = addLink

def adddemand(W, origin, destination, start_time, end_time, flow=-1, volume=-1, attribute=None, links_preferred_list=[]):
    """
    添加需求（车辆生成）

    在 Python 层循环创建车辆，以支持 attribute 参数。
    兼容 UXsim 1.8.2 的接口。

    Parameters
    ----------
    W : World
        仿真世界
    origin : str or Node
        起点
    destination : str or Node
        终点
    start_time : float
        需求开始时间
    end_time : float
        需求结束时间
    flow : float, optional
        车流率 (veh/s)，默认-1
    volume : float, optional
        总需求量，如果指定则忽略flow，默认-1
    attribute : any, optional
        用户自定义属性（元数据），默认None
    links_preferred_list : list of str or Link, optional
        偏好链路列表（DUO模式），默认为空
    """
    # 处理节点类型
    if hasattr(origin, 'name'):
        origin = origin.name
    if hasattr(destination, 'name'):
        destination = destination.name

    # 处理链路类型
    links_preferred_names = []
    for link in links_preferred_list:
        if hasattr(link, 'name'):
            links_preferred_names.append(link.name)
        else:
            links_preferred_names.append(link)

    # 如果指定了 volume，计算 flow
    if volume > 0:
        flow = volume / (end_time - start_time)

    # 获取底层 C++ World
    cpp_world = _get_cpp_world(W)

    # 在 Python 层循环创建车辆（与 C++ add_demand 逻辑一致）
    demand = 0.0
    t = start_time
    while t < end_time:
        demand += flow * W.delta_t
        while demand >= W.deltan:
            # 创建车辆名称
            veh_name = f"{origin}-{destination}-{t}"
            # 调用 C++ Vehicle 构造器
            veh = _CppVehicle(cpp_world, veh_name, t, origin, destination)
            # ========== 关键修复：保持 Python 引用防止垃圾回收 ==========
            if isinstance(W, World):
                W._vehicle_refs.append(veh)
            # Python 层附加 attribute
            veh.attribute = attribute if attribute is not None else {}
            # 设置偏好链路（DUO模式）
            # 注意：pybind11 对 vector 返回副本，不能用 append，需要直接赋值
            if links_preferred_names:
                preferred_links = []
                for ln_name in links_preferred_names:
                    link = W.get_link(ln_name)
                    if link is not None:
                        preferred_links.append(link)
                veh.links_preferred = preferred_links
            demand -= W.deltan
        t += W.delta_t
World.adddemand = adddemand

# ========== 新增：预定路径需求函数 ==========

def adddemand_predefined_route(W, origin, destination, start_time, end_time, flow, predefined_route, attribute=None):
    """
    添加需求（预定路径模式）

    车辆将严格按照指定的链路顺序行驶，支持自环链路。
    在 Python 层循环创建车辆，以支持 attribute 参数。

    Parameters
    ----------
    W : World
        仿真世界
    origin : str or Node
        起点
    destination : str or Node
        终点（可以从路径推断，但为了兼容性保留）
    start_time : float
        需求开始时间
    end_time : float
        需求结束时间
    flow : float
        车流率 (veh/s)
    predefined_route : list of str or Link
        预定路径，按顺序的链路列表
    attribute : any, optional
        用户自定义属性（元数据），默认None
    """
    if hasattr(origin, 'name'):
        origin = origin.name
    if hasattr(destination, 'name'):
        destination = destination.name

    route_names = []
    for link in predefined_route:
        if hasattr(link, 'name'):
            route_names.append(link.name)
        else:
            route_names.append(link)

    # 获取底层 C++ World
    cpp_world = _get_cpp_world(W)

    # 在 Python 层循环创建车辆（与 C++ add_demand 逻辑一致）
    demand = 0.0
    t = start_time
    while t < end_time:
        demand += flow * W.delta_t
        while demand >= W.deltan:
            # 创建车辆名称
            veh_name = f"{origin}-{destination}-{t}"
            # 调用 C++ Vehicle 构造器
            veh = _CppVehicle(cpp_world, veh_name, t, origin, destination)
            # ========== 关键修复：保持 Python 引用防止垃圾回收 ==========
            if isinstance(W, World):
                W._vehicle_refs.append(veh)
            # Python 层附加 attribute
            veh.attribute = attribute if attribute is not None else {}
            # 分配预定路径
            veh.assign_route_by_name(route_names)
            demand -= W.deltan
        t += W.delta_t
World.adddemand_predefined_route = adddemand_predefined_route

def create_vehicle_with_route(W, departure_time, predefined_route, name=None, attribute=None):
    """
    创建单个预定路径车辆

    Parameters
    ----------
    W : World
        仿真世界
    departure_time : float
        出发时间
    predefined_route : list of str or Link
        预定路径
    name : str, optional
        车辆名称，默认自动生成
    attribute : any, optional
        用户自定义属性（元数据），默认None

    Returns
    -------
    Vehicle
        创建的车辆
    """
    if not predefined_route:
        raise ValueError("predefined_route cannot be empty")

    # 从路径推断起点和终点
    first_link = W.get_link(predefined_route[0]) if isinstance(predefined_route[0], str) else predefined_route[0]
    last_link = W.get_link(predefined_route[-1]) if isinstance(predefined_route[-1], str) else predefined_route[-1]

    orig_name = first_link.start_node.name
    dest_name = last_link.end_node.name

    if name is None:
        name = f"veh_{orig_name}_{dest_name}_{departure_time}"

    # 获取底层 C++ World
    cpp_world = _get_cpp_world(W)

    # 创建车辆
    veh = _CppVehicle(cpp_world, name, departure_time, orig_name, dest_name)
    # ========== 关键修复：保持 Python 引用防止垃圾回收 ==========
    if isinstance(W, World):
        W._vehicle_refs.append(veh)
    # Python 层附加 attribute
    veh.attribute = attribute if attribute is not None else {}

    # 分配路径
    route_names = []
    for link in predefined_route:
        if hasattr(link, 'name'):
            route_names.append(link.name)
        else:
            route_names.append(link)

    veh.assign_route_by_name(route_names)

    return veh
World.create_vehicle_with_route = create_vehicle_with_route

# ========== __repr__ 方法 ==========

def link__repr__(s):
    return f"<Link `{s.name}`>"
_CppLink.__repr__ = link__repr__

def node__repr__(s):
    return f"<Node `{s.name}`>"
_CppNode.__repr__ = node__repr__

def veh__repr__(s):
    return f"<Vehicle `{s.name}`>"
_CppVehicle.__repr__ = veh__repr__

#####################################################
# 状态枚举辅助
# 对应 UXsim 1.8.2 的字符串状态

VEHICLE_STATE = {
    0: "home",   # vsHOME - 未出发
    1: "wait",   # vsWAIT - 等待进入路网
    2: "run",    # vsRUN  - 行驶中
    3: "end"     # vsEND  - 已到达
}

VEHICLE_STATE_REVERSE = {v: k for k, v in VEHICLE_STATE.items()}

# 保存 C++ 原始 state 描述符
_cpp_vehicle_state = _CppVehicle.state

def _vehicle_state_int(veh):
    """返回车辆状态的整数表示（C++ 原始值）"""
    return _cpp_vehicle_state.__get__(veh)

def _vehicle_state_str(veh):
    """返回车辆状态的字符串表示，兼容 UXsim 1.8.2"""
    state_int = _cpp_vehicle_state.__get__(veh)
    return VEHICLE_STATE.get(state_int, "unknown")

# 用字符串状态覆盖 C++ 的整数状态，保持与 UXsim 1.8.2 兼容
_CppVehicle.state = property(_vehicle_state_str)
# 保留整数状态访问（如需要）
_CppVehicle.state_int = property(_vehicle_state_int)

# 兼容 UXsim 1.8.2 的 departure_time_in_second 属性
# 在 uxsimpp 中，departure_time 已经是秒为单位
def _vehicle_departure_time_in_second(veh):
    """返回出发时间（秒），兼容 UXsim 1.8.2"""
    return veh.departure_time

_CppVehicle.departure_time_in_second = property(_vehicle_departure_time_in_second)

#####################################################
# 实例-名称-ID 转换函数

def Link_resolve(W, link_like, ret_type="instance"):
    """解析链路（支持实例、名称、ID）"""
    instance = None
    if isinstance(link_like, Link):
        instance = link_like
    elif isinstance(link_like, str):
        instance = W.get_link(link_like)
    elif isinstance(link_like, int):
        instance = W.LINKS[link_like]
    else:
        raise ValueError(f"Unknown Link {link_like}")

    if ret_type == "instance":
        return instance
    elif ret_type == "name":
        return instance.name
    elif ret_type == "id":
        return instance.id
    else:
        raise ValueError(f"Unknown ret_type {ret_type}")
World.Link_resolve = Link_resolve

def eq_Link(W, link_like1, link_like2):
    """比较两个链路是否相同"""
    return W.Link_resolve(link_like1, ret_type="name") == W.Link_resolve(link_like2, ret_type="name")
World.eq_Link = eq_Link

#####################################################
# 仿真执行函数

def exec_simulation(W, duration_t=-1, until_t=-1):
    """
    执行仿真

    Parameters
    ----------
    W : World
        仿真世界
    duration_t : float, optional
        仿真时长，默认-1（运行到结束）
    until_t : float, optional
        运行到指定时间，默认-1（运行到结束）
    """
    # 性能优化：执行延迟的批量路径分配
    _flush_pending_route_assignments(W)

    W.initialize_adj_matrix()
    W.main_loop(duration_t, until_t)
    # 仿真结束后自动构建 traveltime 缓存（优化 Route.actual_travel_time 性能）
    _build_traveltime_cache(W)
World.exec_simulation = exec_simulation


def _flush_pending_route_assignments(W):
    """
    执行延迟的批量路径分配

    将收集的路径分配请求一次性发送到 C++ 层，减少 Python-C++ 边界跨越
    """
    cpp_world = _get_cpp_world(W)
    pending = getattr(cpp_world, '_pending_route_assignments', None)
    if pending:
        # 调用 C++ 批量分配
        cpp_world.batch_assign_routes(pending)
        # 清空队列
        cpp_world._pending_route_assignments = []


def _build_traveltime_cache(W):
    """
    构建链路旅行时间缓存（绑定在每个 Link 上）

    pybind11 每次访问 vector 属性都会复制整个数组，预缓存可提升 ~66x 性能
    """
    delta_t = W.delta_t
    for link in W.LINKS:
        tr = list(link.traveltime_real)  # 只复制一次
        link._traveltime_cache = tr
        link._traveltime_cache_max_idx = len(tr) - 1
        link._traveltime_cache_delta_t = delta_t


#####################################################
# 简易状态分析函数

def inflow(l, t1, t2):
    """计算链路在时间段[t1, t2]的流入率"""
    return (l.arrival_curve[int(t2/l.W.delta_t)]-l.arrival_curve[int(t1/l.W.delta_t)])/(t2-t1)
Link.inflow = inflow

def outflow(l, t1, t2):
    """计算链路在时间段[t1, t2]的流出率"""
    return (l.departure_curve[int(t2/l.W.delta_t)]-l.departure_curve[int(t1/l.W.delta_t)])/(t2-t1)
Link.outflow = outflow

def actual_travel_time(link, t):
    """
    获取在时刻 t 进入该链路的车辆的实际旅行时间

    对应 UXsim 1.8.2 uxsim.py:741-760

    Parameters
    ----------
    t : float
        时间（秒）

    Returns
    -------
    float
        实际旅行时间（秒）
    """
    # 使用缓存（如果存在）
    cache = getattr(link, '_traveltime_cache', None)
    if cache is not None:
        tt = int(t // link._traveltime_cache_delta_t)
        if tt > link._traveltime_cache_max_idx:
            return cache[-1]
        if tt < 0:
            return cache[0]
        return cache[tt]
    else:
        # 无缓存时使用原始方式（仿真未结束时）
        tt = int(t // link.W.delta_t)
        if tt >= len(link.traveltime_real):
            return link.traveltime_real[-1]
        if tt < 0:
            return link.traveltime_real[0]
        return link.traveltime_real[tt]

Link.actual_travel_time = actual_travel_time

def link_inflow(W, l, t1, t2):
    if type(l) is str:
        l = W.get_link(l)
    return l.inflow(t1, t2)
World.link_inflow = link_inflow

def link_outflow(W, l, t1, t2):
    if type(l) is str:
        l = W.get_link(l)
    return l.outflow(t1, t2)
World.link_outflow = link_outflow


#####################################################
# Route 类
# 对应 UXsim 1.8.2 uxsim.py:2632-2736

class Route:
    """
    路径类，存储连续的链路序列

    对应 UXsim 1.8.2 的 Route 类
    """

    def __init__(self, W, links, name="", trust_input=False):
        """
        定义一条路径

        Parameters
        ----------
        W : World
            所属的仿真世界
        links : list
            链路列表（Link 对象或链路名称字符串）
        name : str, optional
            路径名称
        trust_input : bool, optional
            如果为 True，跳过连续性验证以提高性能
        """
        self.W = W
        self.name = name
        self.links = []

        if not trust_input:
            # 验证链路连续性
            for i in range(len(links) - 1):
                l1 = W.get_link(links[i]) if isinstance(links[i], str) else links[i]
                l2 = W.get_link(links[i + 1]) if isinstance(links[i + 1], str) else links[i + 1]
                # 检查 l1 的终点是否有出链路 l2
                if l2 in l1.end_node.out_links:
                    self.links.append(l1)
                else:
                    raise Exception(f"Route is not defined by consecutive links: {links}, {l1} -> {l2}")
            if len(links) >= 2:
                self.links.append(l2)
            elif len(links) >= 1:
                l = W.get_link(links[0]) if isinstance(links[0], str) else links[0]
                self.links.append(l)
        else:
            # 信任输入，直接使用
            self.links = links

        self.links_name = [l.name for l in self.links]

    def __repr__(self):
        return f"<Route {self.name}: {self.links}>"

    def __iter__(self):
        """支持 for link in route 迭代"""
        return iter(self.links)

    def __len__(self):
        return len(self.links)

    def __eq__(self, other):
        """如果两条路径的链路相同，则路径相同"""
        if isinstance(other, Route):
            return [l.name for l in self.links] == [l.name for l in other.links]
        return NotImplemented

    def actual_travel_time(self, t, return_details=False):
        """
        计算从时刻 t 开始行驶该路径的实际旅行时间

        对应 UXsim 1.8.2 uxsim.py:2706-2736

        Parameters
        ----------
        t : float
            出发时间（秒）
        return_details : bool, optional
            如果为 True，同时返回每段链路的旅行时间

        Returns
        -------
        float
            总旅行时间
        list (如果 return_details=True)
            每段链路的旅行时间列表
        """
        tt = 0
        tts = []

        for l in self.links:
            link_tt = l.actual_travel_time(t)
            tt += link_tt
            t += link_tt
            tts.append(link_tt)

        if return_details:
            return tt, tts
        else:
            return tt


def defRoute(W, links, name="", trust_input=False):
    """
    创建路径对象

    Parameters
    ----------
    W : World
        仿真世界
    links : list
        链路列表
    name : str, optional
        路径名称
    trust_input : bool, optional
        是否跳过验证

    Returns
    -------
    Route
        路径对象
    """
    return Route(W, links, name, trust_input)

World.defRoute = defRoute


#####################################################
# Vehicle.traveled_route 方法
# 对应 UXsim 1.8.2 uxsim.py:1260-1311

def traveled_route(veh, include_arrival_time=True, include_departure_time=False):
    """
    返回该车辆实际行驶的路径

    对应 UXsim 1.8.2 uxsim.py:1260-1311

    Parameters
    ----------
    include_arrival_time : bool, optional
        如果为 True，返回到达终点的时间。-1 表示未到达终点。
    include_departure_time : bool, optional
        如果为 True，返回从起点出发的时间。

    Returns
    -------
    Route
        实际行驶的路径
    list
        进入各链路的时间列表（秒）。如果 include_arrival_time=True，最后一个元素是到达时间。
        如果 include_departure_time=True，第一个元素是出发时间。
    """
    route = []
    ts = []

    log_t_link = veh.log_t_link

    # C++ 层 log_t_link 存储的是时步数，需要转换为秒数以兼容 UXsim 1.8.2
    delta_t = veh.W.delta_t

    for i, (timestep, link) in enumerate(log_t_link):
        # 时步数转换为秒数
        t = timestep * delta_t
        # 第一条记录的 None 是 "home"
        if link is None and i == 0:
            if include_departure_time:
                ts.append(t)
        elif link is not None:
            # 链路记录
            ts.append(t)
            route.append(link)
        # 最后一条记录的 None 是 "end"（在下面单独处理）

    # 处理最后一条记录
    if log_t_link:
        timestep, link = log_t_link[-1]
        t = timestep * delta_t
        if include_arrival_time:
            # 判断是否是 "end"：最后一条记录的 link 为 None 且不是第一条记录
            if link is None and len(log_t_link) > 1:
                ts.append(t)
            else:
                ts.append(-1)

    return Route(veh.W, route, trust_input=True), ts

_CppVehicle.traveled_route = traveled_route


#####################################################
# Vehicle.assign_route 包装
# 支持字符串列表参数（兼容 EVCSChargingGameEnv 调用）
# 性能优化：延迟批量执行，减少 Python-C++ 边界跨越

def _vehicle_assign_route_wrapper(veh, route):
    """
    分配预定路径（包装方法，支持 Link 对象列表或字符串列表）

    性能优化：
    1. 路径未变时跳过
    2. 收集到队列，在 exec_simulation 时批量执行（减少 Python-C++ 边界跨越）

    Parameters
    ----------
    route : list of Link or list of str
        预定路径，可以是 Link 对象列表或链路名称字符串列表
    """
    # 快速获取路径名列表
    first = route[0] if route else None
    if first is None:
        return
    if isinstance(first, str):
        route_names = route  # 直接使用，不复制（外部通常不会修改）
    else:
        route_names = [link.name for link in route]

    # 路径变更检测：使用 try/except 避免 getattr 开销
    try:
        cached = veh._cached_route
        # 快速比较：先比较长度
        if cached is not None and len(cached) == len(route_names) and cached == route_names:
            return  # 路径未变，跳过
    except AttributeError:
        pass

    # 更新缓存（只有字符串列表需要复制，避免外部修改影响）
    veh._cached_route = list(route_names) if isinstance(first, str) else route_names

    # 收集到 World 的延迟队列（使用 vehicle id 作为索引）
    cpp_world = veh.W
    try:
        pending = cpp_world._pending_route_assignments
    except AttributeError:
        pending = []
        cpp_world._pending_route_assignments = pending

    pending.append((veh.id, route_names if isinstance(first, str) else veh._cached_route))

# 保存原始 C++ assign_route 方法
_cpp_vehicle_assign_route = _CppVehicle.assign_route

# 用包装方法替换
_CppVehicle.assign_route = _vehicle_assign_route_wrapper


#####################################################
# 简易可视化

def plot_cumcurves(l, col):
    """绘制链路的累积曲线"""
    plt.plot([t*l.W.delta_t for t in range(len(l.arrival_curve))], l.arrival_curve, color=col)
    plt.plot([t*l.W.delta_t for t in range(len(l.arrival_curve))], l.departure_curve, color=col)


#####################################################
# 工具函数

def eq_tol(val, check, rel_tol=0.1, abs_tol=0.0, print_mode=True):
    """测试用：带容差的相等比较"""
    if check == 0 and abs_tol == 0:
        abs_tol = 0.1
    if print_mode:
        print(val, check)
    return abs(val - check) <= abs(check*rel_tol) + abs_tol

def show_variables():
    """显示模块中所有可用的变量和类"""
    dir_list = [e for e in dir() if not e.startswith("_")]
    dir_list_World = [e for e in dir(World) if not e.startswith("_")]
    dir_list_Link = [e for e in dir(Link) if not e.startswith("_")]
    dir_list_Node = [e for e in dir(Node) if not e.startswith("_")]
    dir_list_Vehicle = [e for e in dir(Vehicle) if not e.startswith("_")]

    print("module:")
    for e in dir_list:
        print("\t", e)
    print("World:")
    for e in dir_list_World:
        print("\t", e)
    print("Node:")
    for e in dir_list_Node:
        print("\t", e)
    print("Link:")
    for e in dir_list_Link:
        print("\t", e)
    print("Vehicle:")
    for e in dir_list_Vehicle:
        print("\t", e)
