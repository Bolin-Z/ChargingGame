# UXSim Monkey Patch 模块

import uxsim
import io
from matplotlib import pyplot as plt
from uxsim.analyzer import get_font_for_matplotlib, load_font_data


def patch_uxsim():
    """
    统一的 UXSim 补丁入口点
    """
    # 保存原始方法
    global _original_vehicle_init, _original_vehicle_update
    _original_vehicle_init = uxsim.Vehicle.__init__
    _original_vehicle_update = uxsim.Vehicle.update
    
    # 应用补丁
    uxsim.Analyzer.__init__ = _patched_analyzer_init # 修复 Analyzer 初始化时的文件夹创建问题
    uxsim.Vehicle.__init__ = _patched_vehicle_init # 添加预定路径支持
    uxsim.Vehicle.assign_route = _patched_vehicle_assign_route # 新增路径分配方法
    uxsim.Vehicle.route_next_link_choice = _patched_vehicle_route_next_link_choice # 转移确认机制
    uxsim.Vehicle.update = _patched_vehicle_update # 预定路径状态管理
    uxsim.World.addVehicle = _patched_world_add_vehicle # 新增预定路径车辆创建方法
    uxsim.World.adddemand = _patched_world_adddemand # 修改为使用增强Vehicle

    
def _patched_analyzer_init(s, W, font_pillow=None, font_matplotlib=None):
    """
    Parameters
    ----------
    W : object
        The world to which this belongs.
    font_pillow : str, optional
        The path to the font file for Pillow. If not provided, the default font for English and Japanese is used.
    font_matplotlib : str, optional
        The font name for Matplotlib. If not provided, the default font for English and Japanese is used.
    """
    s.W = W

    # os.makedirs(f"out{s.W.name}", exist_ok=True)

    #基礎統計量
    s.average_speed = 0
    s.average_speed_count = 0
    s.trip_completed = 0
    s.trip_all = 0
    s.total_travel_time = 0
    s.average_travel_time = 0

    #フラグ
    s.flag_edie_state_computed = 0
    s.flag_trajectory_computed = 0
    s.flag_pandas_convert = 0
    s.flag_od_analysis = 0

    # visualization data
    s.font_data = load_font_data(font_pillow)
    s.font_file_like = io.BytesIO(s.font_data)
    
    plt.rcParams["font.family"] = get_font_for_matplotlib(font_matplotlib)


def _patched_vehicle_init(s, W, orig, dest, departure_time, predefined_route=None, name=None, **kwargs):
    """
    支持预定路径参数的 Vehicle 初始化
    
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
    # 调用原始初始化
    _original_vehicle_init(s, W, orig, dest, departure_time, name=name, **kwargs)
    
    # 添加预定路径相关属性
    s.predefined_route_links = []  # Link对象列表，供仿真使用
    s.route_index = 0
    
    # 如果传入了预定路径，立即分配
    if predefined_route is not None:
        s.assign_route(predefined_route)


def _patched_vehicle_assign_route(s, route_names):
    """
    分配预定路径（将路径名称列表转换为Link对象列表）
    
    Parameters
    ----------
    route_names : list
        路径名称列表
    """
    if not route_names:
        return

    s.predefined_route_links = []
    for link_name in route_names:
        link = s.W.get_link(link_name)
        if link is None:
            raise ValueError(f"Link {link_name} not found in network")
        
        s.predefined_route_links.append(link)
    
    # 初始化 route_index
    s.route_index = 0
    
    # 初始化第一个链路并设置 links_prefer 确保 Node.generate() 选择正确链路
    if s.predefined_route_links:
        s.route_next_link = s.predefined_route_links[0]            
        s.links_prefer = [s.predefined_route_links[0]]


def _patched_vehicle_route_next_link_choice(s):
    """
    选择下一个链路 - 使用转移确认机制确保严格按照预定路径执行
    
    核心思想：通过比较route_next_link与当前link来确认转移是否成功，
    只有确认转移成功后才递增route_index
    """
    # 如果没有预定路径，使用默认行为
    if not hasattr(s, 'predefined_route_links') or not s.predefined_route_links:
        # 调用原始方法或保持空实现
        if hasattr(s, '_original_route_next_link_choice'):
            s._original_route_next_link_choice()
        return

    # 步骤1：检查转移确认
    if (s.route_next_link is not None and 
        s.link is not None and 
        s.route_next_link == s.link):
        # 当前路径已完成, 递增 route_index
        s.route_index += 1
    
    # 步骤2：选择下一个目标链路
    if s.route_index >= len(s.predefined_route_links):
        # 路径完成, 设置route_next_link为None
        s.route_next_link = None
        return
    
    # 设置下一个预定链路作为目标
    s.route_next_link = s.predefined_route_links[s.route_index]


def _patched_vehicle_update(s):
    """
    增强的 update 方法，支持预定路径管理
    """
    # 如果没有预定路径，使用原始行为
    if not hasattr(s, 'predefined_route_links') or not s.predefined_route_links:
        _original_vehicle_update(s)
        return
    
    # 记录日志
    s.record_log()
    
    if s.state == "home":
        # 出发逻辑
        if s.W.T >= s.departure_time:
            s.state = "wait"
            s.orig.generation_queue.append(s)
    if s.state == "wait":
        # 等待在起点的垂直队列
        if s.W.route_choice_update_gradual:
            s.route_pref_update()
        pass    
    if s.state == "run":
        # 在链路内行驶
        s.v = (s.x_next - s.x) / s.W.DELTAT
        s.x_old = s.x
        s.x = s.x_next
        
        # 到达链路末端
        if s.x == s.link.length:
            if s.link.end_node in s.node_event.keys():
                s.node_event[s.link.end_node]()
            
            if s.W.route_choice_update_gradual:
                s.route_pref_update()
            
            # 检查预定路径是否完成
            if s.route_index >= len(s.predefined_route_links):
                # 预定路径已完成，准备结束行程
                s.flag_waiting_for_trip_end = 1
                if s.link.vehicles[0] == s:
                    s.end_trip()
                # 没有 taxi 模式
            elif len(s.link.end_node.outlinks.values()) == 0 and s.trip_abort == 1:
                # 行程中止：死胡同
                s.flag_trip_aborted = 1
                s.route_next_link = None
                s.flag_waiting_for_trip_end = 1
                if s.link.vehicles[0] == s:
                    s.end_trip()
                    
            else:
                # 请求链路转移
                s.route_next_link_choice()
                s.link.end_node.incoming_vehicles.append(s)
                
    if s.state in ["end", "abort"]:
        # 行程结束
        pass
    
    # 用户自定义函数
    if s.user_function is not None:
        s.user_function(s)


def _patched_world_add_vehicle(self, predefined_route, departure_time, name=None, **kwargs):
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
    veh = uxsim.Vehicle(self, None, None, departure_time, 
                       predefined_route=predefined_route, name=name, **kwargs)
    return veh


def _patched_world_adddemand(self, orig, dest, t_start, t_end, flow=-1, volume=-1, attribute=None, direct_call=True):
    """
    重写adddemand，使用增强的Vehicle代替标准Vehicle
    
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
            # 使用增强的Vehicle（已支持预定路径），但不传入预定路径（延迟分配）
            uxsim.Vehicle(self, orig, dest, t, 
                         predefined_route=None,  # 延迟分配
                         departure_time_is_time_step=1, 
                         attribute=attribute)
            f -= self.DELTAN