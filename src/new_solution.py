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
        
        self.route_index = 0
        
        # 初始化第一个链路并设置links_prefer确保Node.generate()选择正确链路
        if self.predefined_route_links:
            self.route_next_link = self.predefined_route_links[0]            
            # 设置links_prefer确保Node.generate()选择正确的第一个链路
            self.links_prefer = [self.predefined_route_links[0]]
    
    def update(self):
        """
        重写update方法，优先检查预定路径完成，而不是目标节点到达
        这是technical_validation.py成功的关键实现
        
        注意：使用连续的if语句（不是elif），与UXSim保持一致，
        因为vehicle状态可能在同一timestep内发生变化
        """        
        # 记录日志
        self.record_log()
        
        if self.state == "home":
            # 出发逻辑
            if self.W.T >= self.departure_time:
                self.state = "wait"
                self.orig.generation_queue.append(self)
                
        if self.state == "wait":
            # 等待在起点的垂直队列
            if self.W.route_choice_update_gradual:
                self.route_pref_update()
                
        if self.state == "run":
            # 在链路内行驶
            self.v = (self.x_next - self.x) / self.W.DELTAT
            self.x_old = self.x
            self.x = self.x_next
            
            # 到达链路末端
            if self.x == self.link.length:
                if self.link.end_node in self.node_event.keys():
                    self.node_event[self.link.end_node]()
                
                if self.W.route_choice_update_gradual:
                    self.route_pref_update()
                
                # ✅ 关键修改：优先检查预定路径完成，而不是目标节点
                if self.route_index >= len(self.predefined_route_links):
                    # 预定路径已完成，准备结束行程
                    self.flag_waiting_for_trip_end = 1
                    if self.link.vehicles[0] == self:
                        self.end_trip()
                        
                elif len(self.link.end_node.outlinks.values()) == 0 and self.trip_abort == 1:
                    # 行程中止：死胡同
                    self.flag_trip_aborted = 1
                    self.route_next_link = None
                    self.flag_waiting_for_trip_end = 1
                    if self.link.vehicles[0] == self:
                        self.end_trip()
                        
                else:
                    # 请求链路转移
                    self.route_next_link_choice()
                    self.link.end_node.incoming_vehicles.append(self)
                    
        if self.state in ["end", "abort"]:
            # 行程结束
            pass
        
        # 用户自定义函数
        if self.user_function is not None:
            self.user_function(self)

    def route_next_link_choice(self):
        """
        选择下一个链路 - 使用转移确认机制确保严格按照预定路径执行
        
        核心思想：通过比较route_next_link与当前link来确认转移是否成功，
        只有确认转移成功后才递增route_index
        """

        # 步骤1：检查转移确认
        if (self.route_next_link is not None and 
            self.link is not None and 
            self.route_next_link == self.link):
            # ✅ 转移成功，现在可以递增索引选择下一条预定链路
            self.route_index += 1
        
        # 步骤2：选择下一个目标链路
        if self.route_index >= len(self.predefined_route_links):
            # 路径完成
            self.route_next_link = None
            return
        
        # 设置下一个预定链路作为目标
        self.route_next_link = self.predefined_route_links[self.route_index]


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
