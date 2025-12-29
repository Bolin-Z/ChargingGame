"""诊断测试脚本 - 测试 Vehicle 生命周期修复"""
import sys

print("导入模块...")
from uxsimpp_extended.uxsimpp import World, Vehicle, adddemand

print("创建 World...")
W = World(
    name="test",
    tmax=3000.0,
    deltan=5.0,
    tau=1.0,
    duo_update_time=300.0,
    duo_update_weight=0.25,
    print_mode=0,
    random_seed=42
)

print("添加节点和链路...")
W.addNode("orig", 0, 0)
W.addNode("dest", 1, 0)
link = W.addLink("link", "orig", "dest", 10000, 20, 0.2, 1)

print("\n=== 测试 adddemand（使用修复后的引用保持机制）===")
origin = "orig"
destination = "dest"
start_time = 0
end_time = 100
flow = 0.5

# 使用 adddemand 函数（内部会保持引用）
adddemand(W, origin, destination, start_time, end_time, flow)

print(f"车辆总数: {len(W.VEHICLES)}")
print(f"Python 引用数: {len(W._vehicle_refs)}")

print("\n从 VEHICLES 读取的 departure_time:")
for i, v in enumerate(W.VEHICLES.values()):
    print(f"  [{i}] name='{v.name}', departure_time={v.departure_time}")

# 验证数据正确性
print("\n=== 验证数据正确性 ===")
all_correct = True
for v in W.VEHICLES.values():
    expected_t = float(v.name.split('-')[-1])
    if abs(v.departure_time - expected_t) > 0.01:
        print(f"[FAIL] {v.name}: expected departure_time={expected_t}, got {v.departure_time}")
        all_correct = False

if all_correct:
    print("[PASS] 所有车辆的 departure_time 正确!")
else:
    print("[FAIL] 存在 departure_time 错误")

print("\n完成!")
