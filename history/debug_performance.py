"""
诊断脚本2：性能分析

分析 uxsimpp_extended 各环节的耗时
使用子进程避免模块冲突
"""
import sys
import os
import subprocess


def profile_uxsim_original():
    """分析原版 UXsim 各环节耗时"""
    code = '''
import sys
import time
sys.path.insert(0, "UXsim-1.8.2")

from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.2f} ms")

from uxsim import World

num_nodes = 25
num_links = 50
num_vehicles = 1000
tmax = 3600

print("=" * 60)
print("UXsim 1.8.2 性能分析")
print("=" * 60)
print(f"测试规模: {num_nodes}节点, {num_links}链路, ~{num_vehicles}车辆, tmax={tmax}s")
print("-" * 60)

total_start = time.perf_counter()

with timer("1. 创建 World"):
    W = World(name="perf_test", deltan=5, tmax=tmax, print_mode=0, save_mode=0)

with timer(f"2. 创建 {num_nodes} 个节点"):
    for i in range(num_nodes):
        W.addNode(f"node_{i}", i * 100, 0)

with timer(f"3. 创建 {num_links} 条链路"):
    link_count = 0
    for i in range(num_nodes - 1):
        W.addLink(f"link_{i}_{i+1}", f"node_{i}", f"node_{i+1}",
                 length=1000, free_flow_speed=20)
        link_count += 1
        if link_count >= num_links:
            break
    for i in range(num_nodes - 2):
        if link_count >= num_links:
            break
        W.addLink(f"link_{i}_{i+2}", f"node_{i}", f"node_{i+2}",
                 length=2000, free_flow_speed=20)
        link_count += 1

with timer(f"4. 创建车辆 (adddemand)"):
    for i in range(0, num_nodes - 5, 5):
        W.adddemand(f"node_{i}", f"node_{i+5}", 0, 1800, flow=0.1)

actual_vehicles = len(W.VEHICLES)
print(f"  实际创建车辆数: {actual_vehicles}")

with timer("5. 执行仿真 (exec_simulation)"):
    W.exec_simulation()

with timer("6. 遍历 VEHICLES.values()"):
    count = 0
    for veh in W.VEHICLES.values():
        count += 1

with timer("7. 重复访问 VEHICLES 10次"):
    for _ in range(10):
        for veh in W.VEHICLES.values():
            _ = veh.state

with timer("8. 获取所有车辆的 traveled_route"):
    for veh in W.VEHICLES.values():
        route, ts = veh.traveled_route()

with timer("9. 遍历 LINKS"):
    for link in W.LINKS:
        _ = link.name

total_elapsed = time.perf_counter() - total_start
print("-" * 60)
print(f"总耗时: {total_elapsed*1000:.2f} ms")
'''
    print("运行 UXsim 1.8.2 性能测试...")
    result = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


def profile_uxsimpp_extended():
    """分析 uxsimpp_extended 各环节耗时"""
    code = '''
import sys
import time

from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.2f} ms")

from uxsimpp_extended import World, addNode, addLink, adddemand

num_nodes = 25
num_links = 50
num_vehicles = 1000
tmax = 3600

print("=" * 60)
print("uxsimpp_extended 性能分析")
print("=" * 60)
print(f"测试规模: {num_nodes}节点, {num_links}链路, ~{num_vehicles}车辆, tmax={tmax}s")
print("-" * 60)

total_start = time.perf_counter()

with timer("1. 创建 World"):
    W = World(name="perf_test", deltan=5, tmax=tmax, print_mode=0)

with timer(f"2. 创建 {num_nodes} 个节点"):
    for i in range(num_nodes):
        addNode(W, f"node_{i}", i * 100, 0)

with timer(f"3. 创建 {num_links} 条链路"):
    link_count = 0
    for i in range(num_nodes - 1):
        addLink(W, f"link_{i}_{i+1}", f"node_{i}", f"node_{i+1}",
               length=1000, free_flow_speed=20)
        link_count += 1
        if link_count >= num_links:
            break
    for i in range(num_nodes - 2):
        if link_count >= num_links:
            break
        addLink(W, f"link_{i}_{i+2}", f"node_{i}", f"node_{i+2}",
               length=2000, free_flow_speed=20)
        link_count += 1

with timer(f"4. 创建车辆 (adddemand)"):
    for i in range(0, num_nodes - 5, 5):
        adddemand(W, f"node_{i}", f"node_{i+5}", 0, 1800, flow=0.1)

actual_vehicles = len(W.VEHICLES)
print(f"  实际创建车辆数: {actual_vehicles}")

with timer("5. 初始化邻接矩阵"):
    W.initialize_adj_matrix()

with timer("6. 执行仿真 (main_loop)"):
    W.main_loop()

with timer("7. 遍历 VEHICLES.values()"):
    count = 0
    for veh in W.VEHICLES.values():
        count += 1

with timer("8. 重复访问 VEHICLES 10次"):
    for _ in range(10):
        for veh in W.VEHICLES.values():
            _ = veh.state

with timer("9. 获取所有车辆的 traveled_route"):
    for veh in W.VEHICLES.values():
        route, ts = veh.traveled_route()

with timer("10. 遍历 LINKS"):
    for link in W.LINKS:
        _ = link.name

total_elapsed = time.perf_counter() - total_start
print("-" * 60)
print(f"总耗时: {total_elapsed*1000:.2f} ms")
'''
    print("\n运行 uxsimpp_extended 性能测试...")
    result = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


def profile_vehicle_creation_detail():
    """详细分析车辆创建的开销"""
    code = '''
import sys
import time

from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.2f} ms")

from uxsimpp_extended import World, addNode, addLink
from uxsimpp_extended.uxsimpp import _CppVehicle, _get_cpp_world

W = World(name="veh_test", deltan=5, tmax=3600, print_mode=0)
addNode(W, "A", 0, 0)
addNode(W, "B", 1000, 0)
addLink(W, "link_AB", "A", "B", length=1000, free_flow_speed=20)

num_vehicles = 500
cpp_world = _get_cpp_world(W)

print("=" * 60)
print("车辆创建详细分析")
print("=" * 60)
print(f"创建 {num_vehicles} 辆车的详细耗时:")
print("-" * 60)

# 方式1：完整流程（保存引用 + 设置 attribute）
start = time.perf_counter()
for i in range(num_vehicles):
    veh = _CppVehicle(cpp_world, f"veh1_{i}", i * 10, "A", "B")
    W._vehicle_refs.append(veh)
    veh.attribute = {"test": True}
elapsed1 = time.perf_counter() - start
print(f"  方式1 (保存引用+attribute): {elapsed1*1000:.2f} ms")

# 清理
W._vehicle_refs.clear()

# 方式2：仅 C++ 创建
start = time.perf_counter()
for i in range(num_vehicles):
    veh = _CppVehicle(cpp_world, f"veh2_{i}", i * 10, "A", "B")
elapsed2 = time.perf_counter() - start
print(f"  方式2 (仅C++创建): {elapsed2*1000:.2f} ms")

# 方式3：保存引用但不设置 attribute
start = time.perf_counter()
for i in range(num_vehicles):
    veh = _CppVehicle(cpp_world, f"veh3_{i}", i * 10, "A", "B")
    W._vehicle_refs.append(veh)
elapsed3 = time.perf_counter() - start
print(f"  方式3 (仅保存引用): {elapsed3*1000:.2f} ms")

print("-" * 60)
print(f"Python 开销占比: {(elapsed1-elapsed2)/elapsed1*100:.1f}%")
print(f"  其中 attribute 设置: {(elapsed1-elapsed3)/elapsed1*100:.1f}%")
print(f"  其中 list.append: {(elapsed3-elapsed2)/elapsed1*100:.1f}%")
'''
    print("\n运行车辆创建详细分析...")
    result = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


if __name__ == "__main__":
    print("性能诊断脚本\n")
    profile_uxsim_original()
    profile_uxsimpp_extended()
    profile_vehicle_creation_detail()
