"""
诊断脚本：对 EVCSChargingGameEnv 进行插桩分析，定位真正的性能瓶颈
"""
import os
import sys
import time
import numpy as np
from functools import wraps

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 全局计时器
timing_records = {}


def timeit(name):
    """装饰器：记录函数执行时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if name not in timing_records:
                timing_records[name] = []
            timing_records[name].append(elapsed)
            return result
        return wrapper
    return decorator


def patch_env_for_profiling():
    """给 EVCSChargingGameEnv 的关键方法添加计时"""
    from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv

    # 保存原始方法
    original_create_simulation_world = EVCSChargingGameEnv._EVCSChargingGameEnv__create_simulation_world
    original_apply_routes = EVCSChargingGameEnv._EVCSChargingGameEnv__apply_routes_to_vehicles
    original_route_choice_update = EVCSChargingGameEnv._EVCSChargingGameEnv__route_choice_update
    original_estimate_route_cost = EVCSChargingGameEnv._EVCSChargingGameEnv__estimate_route_cost
    original_calculate_actual_cost = EVCSChargingGameEnv._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow

    # 包装方法
    def timed_create_simulation_world(self):
        start = time.perf_counter()
        result = original_create_simulation_world(self)
        elapsed = time.perf_counter() - start
        if "create_simulation_world" not in timing_records:
            timing_records["create_simulation_world"] = []
        timing_records["create_simulation_world"].append(elapsed)
        return result

    def timed_apply_routes(self, W, routes):
        start = time.perf_counter()
        result = original_apply_routes(self, W, routes)
        elapsed = time.perf_counter() - start
        if "apply_routes_to_vehicles" not in timing_records:
            timing_records["apply_routes_to_vehicles"] = []
        timing_records["apply_routes_to_vehicles"].append(elapsed)
        return result

    def timed_route_choice_update(self, W, dict_charging, dict_uncharging, current_routes, iteration):
        start = time.perf_counter()
        result = original_route_choice_update(self, W, dict_charging, dict_uncharging, current_routes, iteration)
        elapsed = time.perf_counter() - start
        if "route_choice_update" not in timing_records:
            timing_records["route_choice_update"] = []
        timing_records["route_choice_update"].append(elapsed)
        return result

    # 应用包装
    EVCSChargingGameEnv._EVCSChargingGameEnv__create_simulation_world = timed_create_simulation_world
    EVCSChargingGameEnv._EVCSChargingGameEnv__apply_routes_to_vehicles = timed_apply_routes
    EVCSChargingGameEnv._EVCSChargingGameEnv__route_choice_update = timed_route_choice_update

    # 计数器：估算成本和实际成本计算次数
    timing_records["estimate_route_cost_count"] = [0]
    timing_records["calculate_actual_cost_count"] = [0]

    def counted_estimate_route_cost(self, route_obj, departure_time, is_charging):
        timing_records["estimate_route_cost_count"][0] += 1
        return original_estimate_route_cost(self, route_obj, departure_time, is_charging)

    def counted_calculate_actual_cost(self, veh, W, charging_flows):
        timing_records["calculate_actual_cost_count"][0] += 1
        return original_calculate_actual_cost(self, veh, W, charging_flows)

    EVCSChargingGameEnv._EVCSChargingGameEnv__estimate_route_cost = counted_estimate_route_cost
    EVCSChargingGameEnv._EVCSChargingGameEnv__calculate_actual_vehicle_cost_and_flow = counted_calculate_actual_cost

    return EVCSChargingGameEnv


def print_timing_summary():
    """打印计时摘要"""
    print("\n" + "=" * 70)
    print("EVCSChargingGameEnv 性能分析摘要")
    print("=" * 70)

    total_time = 0
    for name, times in timing_records.items():
        if name.endswith("_count"):
            print(f"  {name}: {times[0]} 次调用")
        else:
            total = sum(times) * 1000
            avg = np.mean(times) * 1000 if times else 0
            count = len(times)
            total_time += total
            print(f"  {name}:")
            print(f"    调用次数: {count}")
            print(f"    总耗时: {total:.0f} ms")
            print(f"    平均耗时: {avg:.0f} ms")

    print("-" * 70)
    print(f"  记录的总耗时: {total_time:.0f} ms")
    print("=" * 70)


def main():
    print("=" * 70)
    print("诊断：EVCSChargingGameEnv 插桩性能分析")
    print("=" * 70)

    # 应用插桩
    print("\n1. 应用性能插桩...")
    EVCSChargingGameEnv = patch_env_for_profiling()
    print("   插桩完成")

    # 创建环境
    print("\n2. 创建环境...")
    t0 = time.perf_counter()
    env = EVCSChargingGameEnv(
        network_dir=os.path.join(project_root, "data", "siouxfalls"),
        network_name="siouxfalls",
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    t_init = time.perf_counter() - t0
    print(f"   环境创建耗时: {t_init*1000:.0f} ms")

    # 重置环境
    print("\n3. 重置环境...")
    t0 = time.perf_counter()
    env.reset(seed=42)
    t_reset = time.perf_counter() - t0
    print(f"   环境重置耗时: {t_reset*1000:.0f} ms")

    # 准备动作
    actions = {agent_id: np.array([0.5] * env.n_periods, dtype=np.float32)
               for agent_id in env.agents}

    # 执行一步
    print("\n4. 执行 step（包含 UE-DTA 循环）...")
    print("   [观察下方的 UE-DTA 进度条]\n")

    t0 = time.perf_counter()
    observations, rewards, terminations, truncations, infos = env.step(actions)
    t_step = time.perf_counter() - t0

    print(f"\n   step 总耗时: {t_step:.1f} 秒")

    # 打印计时摘要
    print_timing_summary()

    # 额外分析
    print("\n额外分析:")
    if "route_choice_update" in timing_records and timing_records["route_choice_update"]:
        rcu_times = timing_records["route_choice_update"]
        total_rcu = sum(rcu_times)
        print(f"  route_choice_update 占 step 的比例: {total_rcu/t_step*100:.1f}%")

    if "estimate_route_cost_count" in timing_records:
        count = timing_records["estimate_route_cost_count"][0]
        if "route_choice_update" in timing_records:
            total_rcu_time = sum(timing_records["route_choice_update"])
            if count > 0:
                avg_per_call = total_rcu_time / count * 1000
                print(f"  estimate_route_cost 总调用次数: {count}")
                print(f"  平均每次调用耗时 (估算): {avg_per_call:.3f} ms")


if __name__ == "__main__":
    main()
