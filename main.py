from src.env import EVCSGameEnv
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# 初始化多智能体充电站博弈环境
env = EVCSGameEnv(
    network_dir="./siouxfalls", 
    network_name="siouxfalls",
    random_seed=42,
    max_steps=50,
    convergence_threshold=0.05
)

# 测试环境重置
observations, infos = env.reset()
print("环境重置成功")
print(f"智能体: {env.agents}")
print(f"观测空间样本: {list(observations.keys())}")

# 测试一步仿真
random_actions = {}
for agent in env.agents:
    random_actions[agent] = np.random.random(env.n_periods)

print("\n执行随机动作测试...")
observations, rewards, terminations, truncations, infos = env.step(random_actions)
print(f"奖励: {rewards}")
print(f"仿真完成")