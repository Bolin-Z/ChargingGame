import uxsim

# Custom Vehicle class supporting predefined routes
class PredefinedRouteVehicle(uxsim.Vehicle):
    def __init__(self, W, predefined_route, departure_time, name=None, **kwargs):
        """
        Initialize a vehicle with predefined route support
        
        Parameters
        ----------
        W : World
            World object
        predefined_route : list
            Predefined route sequence (list of link names)
        departure_time : int
            Departure time
        name : str, optional
            Vehicle name
        **kwargs
            Other parameters
        """
        # Validate that all links in the route exist
        for link_name in predefined_route:
            if link_name not in W.LINKS_NAME_DICT:
                raise ValueError(f"Link {link_name} not found in network")
        
        # Validate route connectivity
        for i in range(len(predefined_route) - 1):
            current_link = W.get_link(predefined_route[i])
            next_link = W.get_link(predefined_route[i + 1])
            if current_link.end_node != next_link.start_node:
                raise ValueError(f"Route not connected at {current_link.name} -> {next_link.name}")
        
        # Derive origin and destination from predefined route
        first_link = W.get_link(predefined_route[0])
        last_link = W.get_link(predefined_route[-1])
        orig = first_link.start_node
        dest = last_link.end_node
        
        # Call parent class initialization
        super().__init__(W, orig, dest, departure_time, name=name, **kwargs)
        
        # Set predefined route related attributes
        self.predefined_route_names = predefined_route.copy()
        self.route_index = 0
        
        # Initialize first link immediately - this is the key fix!
        self.route_next_link = W.get_link(predefined_route[0])
        self.route_index = 1  # Already selected first link
        
        # Set route preferences to strongly prefer the predefined route
        # Override the default route preference to ensure predefined route is followed
        self.route_pref = {l.id: 0 for l in W.LINKS}
        for link_name in predefined_route:
            link = W.get_link(link_name)
            self.route_pref[link.id] = 1  # Give high preference to predefined links
        
        self.links_prefer = []
        self.links_avoid = []
    
    def route_next_link_choice(self):
        """
        Select next link - directly return the next link in predefined route
        """
        if self.route_index >= len(self.predefined_route_names):
            self.route_next_link = None
            return None  # Route completed
        
        try:
            next_link = self.W.get_link(self.predefined_route_names[self.route_index])
            self.route_next_link = next_link
            self.route_index += 1  # Increment route index
            return next_link
        except KeyError:
            print(f"Warning: Link {self.predefined_route_names[self.route_index]} not found")
            self.route_next_link = None
            return None
    
    def update(self):
        """
        Update vehicle state - modified destination judgment logic
        """
        # Record logs
        self.record_log()

        if self.state == "home":
            # Depart
            if self.W.T >= self.departure_time:
                self.state = "wait"
                self.orig.generation_queue.append(self)
                
        if self.state == "wait":
            # Wait in vertical queue at departure node
            if self.W.route_choice_update_gradual:
                self.route_pref_update()
            pass
            
        if self.state == "run":
            # Drive within the link
            self.v = (self.x_next-self.x)/self.W.DELTAT
            self.x_old = self.x
            self.x = self.x_next

            # At the end of the link
            if self.x == self.link.length:
                if self.link.end_node in self.node_event.keys():
                    self.node_event[self.link.end_node]()
                
                if self.W.route_choice_update_gradual:
                    self.route_pref_update()
                
                # Modified destination judgment: check if predefined route is completed
                if self.route_index >= len(self.predefined_route_names):
                    # Predefined route completed, end trip
                    self.flag_waiting_for_trip_end = 1
                    if self.link.vehicles[0] == self:
                        self.end_trip()
                elif self.link.end_node == self.dest and not hasattr(self, 'allow_repeat_dest'):
                    # Reached destination and not allowed to repeat, end trip
                    self.flag_waiting_for_trip_end = 1
                    if self.link.vehicles[0] == self:
                        self.end_trip()
                elif self.mode == "taxi":
                    # Taxi mode: continue to next destination
                    if len(self.dest_list) > 0:
                        self.dest = self.dest_list.pop(0)
                    else:
                        self.dest = None
                        self.dest_list = []
                    self.route_pref_update(weight=1)
                    self.route_next_link_choice()
                    self.link.end_node.incoming_vehicles.append(self)
                elif len(self.link.end_node.outlinks.values()) == 0 and self.trip_abort == 1:
                    # Prepare to abort trip due to dead end
                    self.flag_trip_aborted = 1
                    self.route_next_link = None
                    self.flag_waiting_for_trip_end = 1
                    if self.link.vehicles[0] == self:
                        self.end_trip()
                else:
                    # Request link transfer
                    self.route_next_link_choice()
                    self.link.end_node.incoming_vehicles.append(self)
        
        if self.state in ["end", "abort"]:
            # Trip ended
            pass

        if self.user_function is not None:
            self.user_function(self)
    
    def get_route_progress(self):
        """
        Get route execution progress
        """
        return {
            'current_index': self.route_index,
            'total_links': len(self.predefined_route_names),
            'progress_percentage': (self.route_index / len(self.predefined_route_names)) * 100 if len(self.predefined_route_names) > 0 else 0,
            'current_link': self.predefined_route_names[self.route_index] if self.route_index < len(self.predefined_route_names) else None,
            'remaining_links': self.predefined_route_names[self.route_index:] if self.route_index < len(self.predefined_route_names) else []
        }

# Custom World class supporting predefined route vehicles
class PredefinedRouteWorld(uxsim.World):
    def addVehicle(self, predefined_route, departure_time, name=None, **kwargs):
        """
        Add a vehicle with predefined route support
        
        Parameters
        ----------
        predefined_route : list
            Predefined route sequence (list of link names)
        departure_time : int
            Departure time
        name : str, optional
            Vehicle name
        **kwargs
            Other parameters
        """
        veh = PredefinedRouteVehicle(self, predefined_route, departure_time, name=name, **kwargs)
        return veh

# Test code
for i in range(1):
    W = PredefinedRouteWorld(name="test", deltan=1, tmax=2000, print_mode=0)
    W.addNode("A", 0, 0)
    W.addNode("B", 1, 1)
    W.addNode("C", 2, 2)
    W.addLink("A-B","A", "B", 1000)
    W.addLink("A-A","A", "A", 1000)
    W.addLink("B-A", "B", "A", 1000)
    W.addLink("A-C","A", "C", 10000) # a long link
    W.addLink("B-C", "B", "C", 1000)
    
    # Create vehicle with predefined route
    predefined_route = ["A-B", "B-A", "A-A","A-B", "B-A", "A-C"]  # Predefined route: A -> B -> A -> B -> A -> C
    veh = W.addVehicle(predefined_route, 0, name=f"veh_{i}")
    W.exec_simulation()

    route, ts = veh.traveled_route()
    route_names = [link.name for link in route]
    progress = veh.get_route_progress()
    print(f"[{i}]:\t Actual route: {route_names}")
    print(f"[{i}]:\t Predefined route: {predefined_route}")
    print(f"[{i}]:\t Completion progress: {progress['progress_percentage']:.1f}%")
    print(f"[{i}]:\t Unfinished vehicles: {W.analyzer.trip_all - W.analyzer.trip_completed}")
    print(f"[{i}]:\t Travel time: {ts[-1] - ts[0]}")
    for t in ts:
        print(f"[{i}]:\t - {t}")

# Test with original uxsim for comparison
print("\n=== Original UXSim comparison ===")
for i in range(1):
    W = uxsim.World(name="test", deltan=1, tmax=2000, print_mode=0)
    W.addNode("A", 0, 0)
    W.addNode("B", 1, 1)
    W.addNode("C", 2, 2)
    W.addLink("A-B","A", "B", 1000)
    W.addLink("A-A","A", "A", 1000)
    W.addLink("B-A", "B", "A", 1000)
    W.addLink("A-C","A", "C", 10000) # a long link
    W.addLink("B-C", "B", "C", 1000)
    
    veh = W.addVehicle("A", "C", 0)
    W.exec_simulation()

    route, ts = veh.traveled_route()
    route_names = [link.name for link in route]
    print(f"[{i}]:\t Original route: {route_names}")
    print(f"[{i}]:\t Travel time: {ts[-1] - ts[0]}")
    for t in ts:
        print(f"[{i}]:\t - {t}")