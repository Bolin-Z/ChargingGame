"""
EVCSChargingGameEnv v3.0 单步测试脚本

测试功能：
- 环境初始化和重置
- 单步step仿真
- 车流量统计
- 奖励计算验证
"""
from src.EVCSChargingGameEnv import EVCSChargingGameEnv
import logging
import numpy as np

# 设置日志级别为INFO以查看详细过程
logging.basicConfig(level=logging.INFO)

print("="*80)
print("EVCSChargingGameEnv v3.0 单步测试")
print("="*80)

# 初始化多智能体充电站博弈环境
print("\n🚀 初始化环境...")
env = EVCSChargingGameEnv(
    network_dir="./siouxfalls", 
    network_name="siouxfalls",
    random_seed=42,
    max_steps=50,
    convergence_threshold=0.05
)

print(f"✅ 环境初始化成功")
print(f"   充电站数量: {env.n_agents}")
print(f"   时段数量: {env.n_periods}")
print(f"   智能体列表: {env.agents}")

# 测试环境重置
print("\n🔄 重置环境...")
observations, infos = env.reset()
print(f"✅ 环境重置成功")
print(f"   观测空间维度: {[f'{agent}: {type(obs).__name__}' for agent, obs in observations.items()]}")
for agent, obs in observations.items():
    print(f"   {agent}: {[f'{key}: {val.shape}' for key, val in obs.items()]}")

# 单步仿真测试
print("\n🎮 执行单步仿真测试...")

# 生成随机动作（模拟MADRL算法的价格决策）
actions = {}
for agent in env.agents:
    # 生成在[0.3, 0.7]范围内的随机价格动作
    actions[agent] = np.random.uniform(0.3, 0.7, env.n_periods)

print(f"动作(归一化价格): {[f'{agent}: {actions[agent].round(2).tolist()}' for agent in env.agents]}")

# 执行步骤
print("\n⚡ 开始UE-DTA仿真...")
observations, rewards, terminations, truncations, infos = env.step(actions)

print(f"\n✅ 单步仿真完成")
print(f"奖励: {[f'{agent}: {reward:.2f}' for agent, reward in rewards.items()]}")
print(f"终止状态: {terminations}")
print(f"截断状态: {truncations}")

# 显示车流量统计
if len(env.charging_flow_history) > 0:
    print(f"\n🚗 车流量统计:")
    latest_flows = env.charging_flow_history[-1]
    for agent_idx, agent in enumerate(env.agents):
        total_flow = int(latest_flows[agent_idx].sum())
        print(f"{agent}: 总计 {total_flow} 辆车")
        print(f"        各时段: {[int(flow) for flow in latest_flows[agent_idx]]}")

# 显示价格变化
print(f"\n💰 价格变化:")
for i, prices in enumerate(env.price_history):
    print(f"Step {i}: {prices.round(2).tolist()}")

print("\n" + "="*80)
print("🎉 v3.0单步测试完成！")
print("="*80)