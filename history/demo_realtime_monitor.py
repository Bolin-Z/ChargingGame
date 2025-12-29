"""模拟训练过程的实时监控图表演示

布局：上2下1
- 左上：Step层价格相对变化率（累积）
- 右上：当前Step的UE-DTA内层迭代（GM/P90/P95，每Step清空）
- 下方：各智能体收益（累积）
"""
import matplotlib.pyplot as plt
import numpy as np
import time

# 启用交互模式
plt.ion()

# 参数设置
n_agents = 10
total_steps = 100
ue_iterations_per_step = 30  # 每个Step的UE-DTA迭代次数
step_interval = 0.05  # Step内UE-DTA每轮更新间隔（秒）
convergence_threshold = 0.01

# 外层数据存储（累积）
steps = []
convergence_data = []
rewards_data = {i: [] for i in range(n_agents)}
episode_bounds = []

# 内层数据存储（每Step清空）
ue_iterations = []
gm_data = []
p90_data = []
p95_data = []

# 创建图表（方案D: 上2下1，上方宽度比3:2）
fig = plt.figure(figsize=(14, 8))
fig.suptitle('Training Monitor - MADDPG_SiouxFalls', fontsize=14, fontweight='bold')

# 使用GridSpec实现不等宽布局
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1])

# 上方两个图（3:2宽度比）
ax_conv = fig.add_subplot(gs[0, 0:3])  # 左上，占3列
ax_ue = fig.add_subplot(gs[0, 3:5])    # 右上，占2列
# 下方宽图（收益）
ax_reward = fig.add_subplot(gs[1, :])

# === 左上：Step收敛指标 ===
line_conv, = ax_conv.plot([], [], 'b-', linewidth=1.5, marker='o', markersize=4)
ax_conv.axhline(y=convergence_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
ax_conv.set_xlabel('Step')
ax_conv.set_ylabel('Price Change Rate')
ax_conv.set_title('Step Convergence')
ax_conv.set_xlim(0, total_steps)
ax_conv.set_ylim(0, 0.5)
ax_conv.grid(True, alpha=0.3)
ax_conv.legend(loc='upper right')

# === 右上：UE-DTA内层迭代 ===
line_gm, = ax_ue.plot([], [], 'b-', linewidth=1.5, label='GM')
line_p90, = ax_ue.plot([], [], 'orange', linewidth=1.5, label='P90')
line_p95, = ax_ue.plot([], [], 'r-', linewidth=1.5, label='P95')
ax_ue.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='1% Threshold')
ax_ue.set_xlabel('UE-DTA Iteration')
ax_ue.set_ylabel('Convergence Metric')
ax_ue.set_title('UE-DTA Convergence (Current Step)')
ax_ue.set_xlim(0, ue_iterations_per_step)
ax_ue.set_ylim(0, 0.15)
ax_ue.grid(True, alpha=0.3)
ax_ue.legend(loc='upper right')

# === 下方：收益曲线 ===
reward_lines = []
colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
for i in range(n_agents):
    line, = ax_reward.plot([], [], alpha=0.7, linewidth=1.2, color=colors[i])
    reward_lines.append(line)
ax_reward.set_xlabel('Step')
ax_reward.set_ylabel('Reward')
ax_reward.set_title(f'Agent Rewards ({n_agents} agents)')
ax_reward.set_xlim(0, total_steps)
ax_reward.set_ylim(0, 800)
ax_reward.grid(True, alpha=0.3)

plt.tight_layout()


def generate_ue_iteration_data(iteration, total_iterations):
    """生成单轮UE-DTA迭代数据"""
    # 指数衰减 + 噪声
    decay = np.exp(-iteration * 0.2)
    gm = 0.12 * decay + np.random.normal(0, 0.005)
    p90 = 0.10 * decay + np.random.normal(0, 0.008)
    p95 = 0.08 * decay + np.random.normal(0, 0.006)
    return max(0.001, abs(gm)), max(0.001, abs(p90)), max(0.001, abs(p95))


def generate_step_result(step, episode):
    """生成Step完成后的结果数据"""
    # 收敛指标：随训练进行逐渐下降
    base_conv = 0.3 * np.exp(-step * 0.08)
    conv = base_conv + np.random.normal(0, 0.015)
    conv = max(0.001, abs(conv))

    # 收益：逐渐稳定
    base_reward = 300 + step * 5
    agent_rewards = []
    for i in range(n_agents):
        r = base_reward * np.random.uniform(0.6, 1.4) + np.random.normal(0, 25)
        agent_rewards.append(max(0, r))

    return conv, agent_rewards


# 模拟训练过程
print("=" * 50)
print("Training Monitor Demo")
print("=" * 50)
print(f"Total Steps: {total_steps}")
print(f"UE-DTA iterations per Step: {ue_iterations_per_step}")
print("Close the plot window to stop.\n")

current_episode = 0
steps_in_episode = 0
episode_length = 10

try:
    for step in range(total_steps):
        # 检测episode边界
        if step > 0 and steps_in_episode >= episode_length:
            episode_bounds.append(step)
            current_episode += 1
            steps_in_episode = 0
            # 添加episode边界线
            ax_conv.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
            ax_reward.axvline(x=step, color='gray', linestyle=':', alpha=0.5)

        # === 清空UE-DTA内层数据 ===
        ue_iterations.clear()
        gm_data.clear()
        p90_data.clear()
        p95_data.clear()
        line_gm.set_data([], [])
        line_p90.set_data([], [])
        line_p95.set_data([], [])
        ax_ue.set_title(f'UE-DTA Convergence (Step {step + 1})')

        # === 模拟UE-DTA内层迭代 ===
        for ue_iter in range(ue_iterations_per_step):
            gm, p90, p95 = generate_ue_iteration_data(ue_iter, ue_iterations_per_step)

            ue_iterations.append(ue_iter + 1)
            gm_data.append(gm)
            p90_data.append(p90)
            p95_data.append(p95)

            # 更新UE-DTA图
            line_gm.set_data(ue_iterations, gm_data)
            line_p90.set_data(ue_iterations, p90_data)
            line_p95.set_data(ue_iterations, p95_data)

            # 动态调整Y轴
            max_val = max(max(gm_data), max(p90_data), max(p95_data))
            ax_ue.set_ylim(0, max(0.05, max_val * 1.2))

            # 更新标题
            fig.suptitle(f'Training Monitor - Episode {current_episode + 1}, '
                        f'Step {step + 1}/{total_steps}, UE-DTA iter {ue_iter + 1}',
                        fontsize=14, fontweight='bold')

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(step_interval)

        # === Step完成，更新外层数据 ===
        conv, agent_rewards = generate_step_result(step, current_episode)

        steps.append(step + 1)
        convergence_data.append(conv)
        for i in range(n_agents):
            rewards_data[i].append(agent_rewards[i])

        # 更新外层图表
        line_conv.set_data(steps, convergence_data)
        for i in range(n_agents):
            reward_lines[i].set_data(steps, rewards_data[i])

        # 动态调整Y轴
        if convergence_data:
            ax_conv.set_ylim(0, max(0.1, max(convergence_data) * 1.2))
        all_rewards = [r for i in range(n_agents) for r in rewards_data[i]]
        if all_rewards:
            ax_reward.set_ylim(0, max(all_rewards) * 1.1)

        fig.canvas.draw()
        fig.canvas.flush_events()

        steps_in_episode += 1
        print(f"Step {step + 1}/{total_steps} complete | "
              f"Episode {current_episode + 1} | "
              f"Conv={conv:.4f} | "
              f"Final GM={gm_data[-1]:.4f}")

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")

print("\nSimulation complete!")
plt.ioff()
plt.show()
