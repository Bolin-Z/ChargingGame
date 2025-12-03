"""
Submodule for Dynamic Traffic Assignment solvers.
"""
import random
import time
import warnings
from pprint import pprint
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ..Utilities import enumerate_k_shortest_routes, enumerate_k_random_routes, estimate_congestion_externality_route
from ..utils import *
from . import ALNS

import warnings


class SolverDUE:

    def __init__(s, func_World):
        """
        Solve quasi Dynamic User Equilibrium (DUE) problem using day-to-day dynamics.

        Parameters
        ----------
        func_World : function
            function that returns a World object with nodes, links, and demand specifications
            
        Notes
        -----
            This function computes a near dynamic user equilibrium state as a steady state of day-to-day dynamical routing game.

            Specifically, on day `i`, vehicles choose their route based on actual travel time on day `i-1` with the same departure time.
            If there are shorter travel time route, they will change with probability `swap_prob`.
            This process is repeated until `max_iter` day.
            It is expected that this process eventually reach a steady state.
            Due to the problem complexity, it does not necessarily reach or converge to Nash equilibrium or any other stationary points.
            However, in the literature, it is argued that the steady state can be considered as a reasonable proxy for Nash equilibrium or dynamic equilibrium state.
            There are some theoretical background for it; but intuitively speaking, the steady state can be considered as a realistic state that people's rational behavior will reach.

            This method is based on the following literature:
            Ishihara, M., & Iryo, T. (2015). Dynamic Traffic Assignment by Markov Chain. Journal of Japan Society of Civil Engineers, Ser. D3 (Infrastructure Planning and Management), 71(5), I_503-I_509. (in Japanese). https://doi.org/10.2208/jscejipm.71.I_503
            Iryo, T., Urata, J., & Kawase, R. (2024). Traffic Flow Simulator and Travel Demand Simulators for Assessing Congestion on Roads After a Major Earthquake. In APPLICATION OF HIGH-PERFORMANCE COMPUTING TO EARTHQUAKE-RELATED PROBLEMS (pp. 413-447). https://doi.org/10.1142/9781800614635_0007
            Iryo, T., Watling, D., & Hazelton, M. (2024). Estimating Markov Chain Mixing Times: Convergence Rate Towards Equilibrium of a Stochastic Process Traffic Assignment Model. Transportation Science. https://doi.org/10.1287/trsc.2024.0523
        """
        s.func_World = func_World
        s.W_sol = None  #final solution
        s.W_intermid_solution = None    #latest solution in the iterative process. Can be used when a user terminates the solution algorithm
        s.dfs_link = []



        #warnings.warn("DTA solver is experimental and may not work as expected. It is functional but unstable.")
    

    def solve(s, max_iter, n_routes_per_od=10, swap_prob=0.05, print_progress=True):
        """
        Solve quasi Dynamic User Equilibrium (DUE) problem using day-to-day dynamics. WIP.

        Parameters
        ----------
        max_iter : int
            maximum number of iterations
        n_routes_per_od : int
            number of routes to enumerate for each OD pair
        swap_prob : float
            probability of route swap
        print_progress : bool
            whether to print the information

        Returns
        -------
        W : World
            World object with quasi DUE solution (if properly converged)
        
        Notes
        -----
        `self.W_sol` is the final solution. 
        `self.W_intermid_solution` is the latest solution in the iterative process. Can be used when a user terminates the solution algorithm.
        """
        s.start_time = time.time()

        W_orig = s.func_World()
        if print_progress:
            W_orig.print_scenario_stats()

        # enumerate routes for each OD pair
        n_routes_per_od = n_routes_per_od

        dict_od_to_vehid = defaultdict(lambda: [])
        for key, veh in W_orig.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            dict_od_to_vehid[o,d].append(key)

        # dict_od_to_routes = {}
        # for o,d in dict_od_to_vehid.keys():
        #     routes = enumerate_k_shortest_routes(W_orig, o, d, k=n_routes_per_od)
        #     dict_od_to_routes[o,d] = routes

        if W_orig.finalized == False:
            W_orig.finalize_scenario()
        dict_od_to_routes = enumerate_k_random_routes(W_orig, k=n_routes_per_od)

        if print_progress:
            print(f"number of OD pairs: {len(dict_od_to_routes.keys())}, number of routes: {sum([len(val) for val in dict_od_to_routes.values()])}")

        # day-to-day dynamics
        s.ttts = []
        s.n_swaps = []
        s.potential_swaps = []
        s.t_gaps = []
        s.route_log = []
        s.cost_log = []
        swap_prob = swap_prob
        max_iter = max_iter

        print("solving DUE...")
        for i in range(max_iter):
            W = s.func_World()
            if i != max_iter-1:
                W.vehicle_logging_timestep_interval = -1

            if i != 0:
                for key in W.VEHICLES:
                    if key in routes_specified:
                        W.VEHICLES[key].enforce_route(routes_specified[key])
            
            route_set = defaultdict(lambda: []) #routes[o,d] = [Route, Route, ...]
            for o,d in dict_od_to_vehid.keys():
                for r in dict_od_to_routes[o,d]:
                    route_set[o,d].append(W.defRoute(r))

            # simulation
            W.exec_simulation()

            # results
            W.analyzer.print_simple_stats()
            #W.analyzer.network_average()

            # trip completion check
            unfinished_trips = W.analyzer.trip_all - W.analyzer.trip_completed
            if unfinished_trips > 0:
                warnings.warn(f"Warning: {unfinished_trips} / {W.analyzer.trip_all} vehicles have not finished their trips. The DUE solver assumes that all vehicles finish their trips during the simulation duration. Consider increasing the simulation time limit or checking the network configuration.", UserWarning)

            # attach route choice set to W object for later re-use at different solvers like DSO-GA
            W.dict_od_to_routes = dict_od_to_routes
            
            s.W_intermid_solution = W

            s.dfs_link.append(W.analyzer.link_to_pandas())

            # route swap
            routes_specified = {}
            route_actual = {}
            cost_actual = {}
            n_swap = 0
            total_t_gap = 0
            potential_n_swap = 0
            for key,veh in W.VEHICLES.items():
                flag_swap = random.random() < swap_prob
                o = veh.orig.name
                d = veh.dest.name
                r, ts = veh.traveled_route()
                travel_time = ts[-1]-ts[0]
                
                route_actual[key] = [rr.name for rr in r]
                cost_actual[key] = travel_time

                if veh.state != "end":
                    continue

                flag_route_changed = False
                route_changed = None
                t_gap = 0

                cost_current = r.actual_travel_time(ts[0])
                
                potential_n_swap_updated = potential_n_swap
                for alt_route in route_set[o,d]:
                    cost_alt = alt_route.actual_travel_time(ts[0])
                    if cost_alt < cost_current:
                        if flag_route_changed == False or (cost_alt < cost_current):
                            t_gap = cost_current - cost_alt
                            potential_n_swap_updated = potential_n_swap + W.DELTAN
                            if flag_swap:
                                flag_route_changed = True
                                route_changed = alt_route
                                cost_current = cost_alt
                                
                potential_n_swap = potential_n_swap_updated
                
                total_t_gap += t_gap
                routes_specified[key] = r
                if flag_route_changed:
                    n_swap += W.DELTAN
                    routes_specified[key] = route_changed

            t_gap_per_vehicle = total_t_gap/len(W.VEHICLES)
            if print_progress:
                print(f' iter {i}: time gap: {t_gap_per_vehicle:.1f}, potential route change: {potential_n_swap}, route change: {n_swap}, total travel time: {W.analyzer.total_travel_time: .1f}, delay ratio: {W.analyzer.average_delay/W.analyzer.average_travel_time: .3f}')

            s.route_log.append(route_actual)
            s.cost_log.append(cost_actual)

            s.ttts.append(int(W.analyzer.total_travel_time))
            s.n_swaps.append(n_swap)
            s.potential_swaps.append(potential_n_swap)
            s.t_gaps.append(t_gap_per_vehicle)
        
        s.end_time = time.time()

        print("DUE summary:")
        last_iters = int(max_iter/4)
        print(f" total travel time: initial {s.ttts[0]:.1f} -> average of last {last_iters} iters {np.average(s.ttts[-last_iters:]):.1f}")
        print(f" number of potential route changes: initial {s.potential_swaps[0]:.1f} -> average of last {last_iters} iters {np.average(s.potential_swaps[-last_iters:]):.1f}")
        print(f" route travel time gap: initial {s.t_gaps[0]:.1f} -> average of last {last_iters} iters {np.average(s.t_gaps[-last_iters:]):.1f}")
        print(f" computation time: {s.end_time - s.start_time:.1f} seconds")

        s.W_sol = W
        return s.W_sol




    def plot_convergence(s):
        """
        Plots convergence metrics for a Dynamic Traffic Assignment (DTA) solution.
        This function creates three separate plots:
        1. Total travel time across iterations
        2. Number of route changes (swaps) across iterations
        3. Travel time gap between chosen routes and minimum cost routes across iterations 
        """

        # iteration plot
        plt.figure(figsize=(6,2))
        plt.title("total travel time")
        plt.plot(s.ttts)
        plt.xlabel("iter")

        plt.figure(figsize=(6,2))
        plt.title("number of route change")
        plt.plot(s.n_swaps)
        plt.ylim(0,None)
        plt.xlabel("iter")

        plt.figure(figsize=(6,2))
        plt.title("travel time difference between chosen route and minimum cost route")
        plt.plot(s.t_gaps)
        plt.ylim(0,None)
        plt.xlabel("iter")

        plt.show()



        # plt.figure()
        # plot_multiple_y(ys=[s.ttts, s.n_swaps, s.potential_swaps, s.total_t_gaps, np.array(s.total_t_gaps)/np.array(s.potential_swaps)], labels=["total travel time", "number of route change", "number of potential route change", "time gap for potential route change", "time gap per potential route change"])
        # plt.xlabel("iter")
        # plt.show()


    def plot_link_stats(s):
        """
        Generate two plots to visualize the evolution of link-level traffic statistics across iterations.
        The first plot shows traffic volume changes for each link over iterations.
        The second plot shows average travel time changes for each link over iterations.
        """

        plt.figure()
        plt.title("traffic volume")
        for i in range(len(s.dfs_link[0])):
            vols = [df["traffic_volume"][i] for df in s.dfs_link]
            plt.plot(vols, label=s.dfs_link[0]["link"][i])
        plt.xlabel("iteration")
        plt.ylabel("volume (veh)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.figure()
        plt.title("average travel time")
        for i in range(len(s.dfs_link[0])):
            vols = [df["average_travel_time"][i] for df in s.dfs_link]
            plt.plot(vols, label=s.dfs_link[0]["link"][i])
        plt.xlabel("iteration")
        plt.ylabel("time (s)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()


    

    def plot_vehicle_stats(s, orig=None, dest=None):
        """
        Plot travel time statistics for vehicles based on their origin and destination.
        This function visualizes the average travel time and standard deviation for vehicles
        matching the specified origin and destination criteria. The data is plotted against 
        the departure time of each vehicle.

        Parameters
        ----------
        orig : str, optional
            Filter vehicles by origin. If None, vehicles from all origins are included.
        dest : str, optional
            Filter vehicles by destination. If None, vehicles to all destinations are included.
            
        Notes
        -----
        - The function uses the second half of the available data (from length/2 to length)
          from the cost_log to compute statistics.
        - The plot shows departure time on the x-axis and average travel time on the y-axis,
          with error bars representing the standard deviation.
        """

        ave_TT = []
        std_TT = []
        depature_time = []
        for vehid in s.route_log[0].keys():
            if (s.W_sol.VEHICLES[vehid].orig.name == orig or orig == None) and (s.W_sol.VEHICLES[vehid].dest.name == dest or dest == None):
                length = len(s.route_log)
                ts = [s.cost_log[day][vehid] for day in range(int(length/2), length)]
                ave_TT.append(np.average(ts))
                std_TT.append(np.std(ts))
                depature_time.append(s.W_sol.VEHICLES[vehid].departure_time_in_second)

        plt.figure()
        orig_ = orig
        if orig == None:
            orig_ = "any"
        dest_ = dest
        if dest == None:
            dest_ = "any"
        plt.title(f"orig: {orig_}, dest: {dest_}")
        plt.errorbar(x=depature_time, y=ave_TT, yerr=std_TT, 
                fmt='bx', ecolor="#aaaaff", capsize=0, label=r"travel time (mean $\pm$ std)")
        plt.xlabel("departure time of vehicle")
        plt.ylabel("travel time")
        plt.legend()
        plt.show()
