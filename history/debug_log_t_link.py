"""
诊断脚本1：验证 log_t_link 时间单位差异

比较 UXsim 1.8.2 和 uxsimpp_extended 的 log_t_link 语义
分别在独立进程中运行以避免模块冲突
"""
import sys
import os
import subprocess

def test_uxsim_original():
    """测试原版 UXsim 1.8.2 的 log_t_link"""
    code = '''
import sys
sys.path.insert(0, "UXsim-1.8.2")
from uxsim import World

# 创建简单网络
W = World(name="test", deltan=5, tmax=600, print_mode=0, save_mode=0)

# 两个节点，一条链路
W.addNode("A", 0, 0)
W.addNode("B", 1000, 0)
W.addLink("link_AB", "A", "B", length=1000, free_flow_speed=20)

# 添加车辆，使用 flow 而非 volume
W.adddemand("A", "B", 100, 110, flow=0.5)

W.exec_simulation()

# 获取车辆
if len(W.VEHICLES) == 0:
    print("ERROR: 没有创建车辆")
    sys.exit(1)

veh = list(W.VEHICLES.values())[0]

print("=" * 60)
print("UXsim 1.8.2 的 log_t_link 分析")
print("=" * 60)
print(f"DELTAT (秒/时步) = {W.DELTAT}")
print(f"车辆 departure_time (内部) = {veh.departure_time}")
print(f"车辆 departure_time_in_second = {veh.departure_time_in_second}")
print(f"")
print(f"log_t_link 内容:")
for i, (t, link) in enumerate(veh.log_t_link):
    link_name = link.name if hasattr(link, 'name') else link
    print(f"  [{i}] t={t}, link={link_name}")

# 分析
print(f"")
print(f"--- 单位分析 ---")
first_t = veh.log_t_link[0][0]
print(f"log_t_link[0][0] = {first_t}")
print(f"departure_time_in_second = {veh.departure_time_in_second}")
print(f"")
if abs(first_t - veh.departure_time_in_second) < 1:
    print(f"结论: log_t_link 存储的是【秒数】")
else:
    print(f"结论: log_t_link 存储的不是秒数")
    print(f"  first_t / DELTAT = {first_t / W.DELTAT}")

# traveled_route
route, timestamps = veh.traveled_route()
print(f"")
print(f"traveled_route 返回的 timestamps: {timestamps}")
if len(timestamps) >= 2:
    travel_time = timestamps[-1] - timestamps[0]
    print(f"计算的旅行时间 = {travel_time}")
    print(f"车辆记录的 travel_time = {veh.travel_time}")
'''
    print("=" * 60)
    print("运行 UXsim 1.8.2 测试...")
    print("=" * 60)
    result = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


def test_uxsimpp_extended():
    """测试 uxsimpp_extended 的 log_t_link"""
    # 注意：uxsimpp_extended 是通过 pip install -e . 安装的
    # 直接从包名导入，不需要 src.env 前缀
    code = '''
import sys

# uxsimpp_extended 是独立安装的包
from uxsimpp_extended import World, addNode, addLink, adddemand

# 创建简单网络
W = World(name="test", deltan=5, tmax=600, print_mode=0)

# 两个节点，一条链路
addNode(W, "A", 0, 0)
addNode(W, "B", 1000, 0)
addLink(W, "link_AB", "A", "B", length=1000, free_flow_speed=20)

# 添加车辆
adddemand(W, "A", "B", 100, 110, flow=0.5)

W.exec_simulation()

# 获取车辆
if len(W.VEHICLES) == 0:
    print("ERROR: 没有创建车辆")
    sys.exit(1)

veh = list(W.VEHICLES.values())[0]

print("=" * 60)
print("uxsimpp_extended 的 log_t_link 分析")
print("=" * 60)
print(f"delta_t (秒/时步) = {W.delta_t}")
print(f"车辆 departure_time = {veh.departure_time}")
print(f"")
print(f"log_t_link 内容:")
for i, (t, link) in enumerate(veh.log_t_link):
    link_name = link.name if link is not None else "None(home/end)"
    print(f"  [{i}] t={t}, link={link_name}")

# 分析
print(f"")
print(f"--- 单位分析 ---")
first_t = veh.log_t_link[0][0]
print(f"log_t_link[0][0] = {first_t}")
print(f"departure_time (秒) = {veh.departure_time}")
print(f"")
if abs(first_t - veh.departure_time) < 1:
    print(f"结论: log_t_link 存储的是【秒数】")
elif abs(first_t * W.delta_t - veh.departure_time) < 1:
    print(f"结论: log_t_link 存储的是【时步数】")
    print(f"  转换为秒: {first_t} * {W.delta_t} = {first_t * W.delta_t}")
else:
    print(f"结论: 无法判断单位")

# traveled_route
route, timestamps = veh.traveled_route()
print(f"")
print(f"traveled_route 返回的 timestamps: {timestamps}")
if len(timestamps) >= 2:
    travel_time = timestamps[-1] - timestamps[0]
    print(f"计算的旅行时间 = {travel_time}")
    print(f"车辆记录的 travel_time = {veh.travel_time}")
    print(f"")
    if abs(travel_time - veh.travel_time) < 1:
        print(f"timestamps 单位正确（秒）")
    else:
        print(f"timestamps 单位可能有问题！")
        print(f"  如果 timestamps 是时步数: {travel_time * W.delta_t} 秒")
'''
    print("\n" + "=" * 60)
    print("运行 uxsimpp_extended 测试...")
    print("=" * 60)
    result = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


if __name__ == "__main__":
    print("诊断 log_t_link 时间单位差异\n")
    test_uxsim_original()
    test_uxsimpp_extended()
