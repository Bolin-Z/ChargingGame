# tests/test_evaluator_vs_env.py
"""
EVCSRewardEvaluator 与旧环境 EVCSChargingGameEnv 对比测试

验证内容：
1. 相同价格输入，比较收益和流量输出
2. 分析差异来源（如有）

注意：由于设计差异，结果可能不完全一致：
- 旧环境：路径分配从上一个 step 继承
- 新评估器：每次从贪心分配开始（确保确定性）
"""

import sys
import os
import time
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator.network_data import NetworkDataLoader
from src.evaluator.evaluator import EVCSRewardEvaluator
from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def compare_single_step():
    """对比单次评估/step 的结果"""
    print("=" * 70)
    print("对比测试: EVCSRewardEvaluator vs EVCSChargingGameEnv")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_dir = os.path.join(project_root, "data", "siouxfalls")
    network_name = "siouxfalls"

    # ========== 初始化新评估器 ==========
    print("\n[1] 初始化新评估器...")
    loader = NetworkDataLoader(network_dir=network_dir, network_name=network_name, random_seed=42)
    network_data = loader.load()
    evaluator = EVCSRewardEvaluator(network_data)
    print(f"    充电站: {evaluator.agent_names}")
    print(f"    时段数: {evaluator.n_periods}")

    # ========== 初始化旧环境 ==========
    print("\n[2] 初始化旧环境...")
    env = EVCSChargingGameEnv(network_dir=network_dir, network_name=network_name, random_seed=42)
    env.reset(seed=42)
    print(f"    充电站: {env.agents}")
    print(f"    时段数: {env.n_periods}")

    # ========== 构造测试价格 ==========
    # 使用中间价格
    prices_dict = {}
    actions_dict = {}
    for agent_name in evaluator.agent_names:
        bounds = network_data.charging_nodes[agent_name]
        mid_price = (bounds[0] + bounds[1]) / 2
        prices_dict[agent_name] = np.full(evaluator.n_periods, mid_price)
        # 归一化动作 [0, 1]
        actions_dict[agent_name] = np.full(evaluator.n_periods, 0.5, dtype=np.float32)

    print(f"\n[3] 测试价格: {list(prices_dict.values())[0][0]:.2f} (中间价格)")

    # ========== 运行新评估器 ==========
    print("\n[4] 运行新评估器...")
    start_time = time.time()
    eval_result = evaluator.evaluate(prices_dict, seed=42)
    eval_time = time.time() - start_time
    print(f"    耗时: {eval_time:.2f} 秒")
    print(f"    UE 迭代: {eval_result.ue_iterations}, 收敛: {eval_result.converged}")

    # ========== 运行旧环境 step ==========
    print("\n[5] 运行旧环境 step...")
    start_time = time.time()
    obs, rewards_env, terminated, truncated, info = env.step(actions_dict)
    env_time = time.time() - start_time
    print(f"    耗时: {env_time:.2f} 秒")
    print(f"    UE 迭代: {info.get('ue_iterations', 'N/A')}, 收敛: {info.get('ue_converged', 'N/A')}")

    # ========== 对比结果 ==========
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)

    print("\n收益对比:")
    print(f"{'充电站':<10} {'新评估器':<15} {'旧环境':<15} {'差异':<15} {'差异%':<10}")
    print("-" * 65)

    total_reward_new = 0
    total_reward_old = 0
    for agent_name in evaluator.agent_names:
        reward_new = eval_result.rewards[agent_name]
        reward_old = rewards_env[agent_name]
        diff = reward_new - reward_old
        diff_pct = (diff / reward_old * 100) if reward_old != 0 else 0
        total_reward_new += reward_new
        total_reward_old += reward_old
        print(f"{agent_name:<10} {reward_new:<15.2f} {reward_old:<15.2f} {diff:<15.2f} {diff_pct:<10.2f}%")

    print("-" * 65)
    total_diff = total_reward_new - total_reward_old
    total_diff_pct = (total_diff / total_reward_old * 100) if total_reward_old != 0 else 0
    print(f"{'总计':<10} {total_reward_new:<15.2f} {total_reward_old:<15.2f} {total_diff:<15.2f} {total_diff_pct:<10.2f}%")

    print("\n流量对比 (各时段总和):")
    print(f"{'充电站':<10} {'新评估器':<15} {'旧环境':<15} {'差异':<15}")
    print("-" * 55)

    # 获取旧环境的流量
    flows_env = env.charging_flow_history[-1] if env.charging_flow_history else None

    total_flow_new = 0
    total_flow_old = 0
    for i, agent_name in enumerate(evaluator.agent_names):
        flow_new = np.sum(eval_result.flows[agent_name])
        flow_old = np.sum(flows_env[i]) if flows_env is not None else 0
        diff = flow_new - flow_old
        total_flow_new += flow_new
        total_flow_old += flow_old
        print(f"{agent_name:<10} {flow_new:<15.0f} {flow_old:<15.0f} {diff:<15.0f}")

    print("-" * 55)
    print(f"{'总计':<10} {total_flow_new:<15.0f} {total_flow_old:<15.0f} {total_flow_new - total_flow_old:<15.0f}")

    # ========== 详细流量对比（按时段）==========
    print("\n详细流量对比 (按时段):")
    for agent_name in evaluator.agent_names:
        agent_idx = evaluator.agent_name_mapping[agent_name]
        flow_new = eval_result.flows[agent_name]
        flow_old = flows_env[agent_idx] if flows_env is not None else np.zeros(evaluator.n_periods)

        print(f"\n  {agent_name}:")
        print(f"    新评估器: {flow_new}")
        print(f"    旧环境:   {flow_old}")
        print(f"    差异:     {flow_new - flow_old}")

    # ========== 分析 ==========
    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)

    # 检查差异是否在可接受范围内
    reward_diff_pct = abs(total_diff_pct)
    flow_diff = abs(total_flow_new - total_flow_old)

    if reward_diff_pct < 5 and flow_diff < 50:
        print("\n✓ 结果差异在可接受范围内 (收益差异 < 5%, 流量差异 < 50)")
        print("  差异来源可能是:")
        print("  1. 路径初始化策略不同（新：贪心，旧：可能继承）")
        print("  2. 随机数序列的微小差异")
    elif reward_diff_pct < 20:
        print("\n⚠ 结果存在一定差异，但在合理范围内")
        print("  这是预期的，因为设计上有以下差异:")
        print("  1. 新评估器每次从贪心分配开始")
        print("  2. 旧环境可能继承上一步的路径分配")
    else:
        print("\n✗ 结果差异较大，需要进一步调查")

    return eval_result, rewards_env


def compare_multiple_steps():
    """对比多次连续 step 的结果（测试状态依赖）"""
    print("\n" + "=" * 70)
    print("多步对比测试: 连续评估")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_dir = os.path.join(project_root, "data", "siouxfalls")
    network_name = "siouxfalls"

    # 初始化
    loader = NetworkDataLoader(network_dir=network_dir, network_name=network_name, random_seed=42)
    network_data = loader.load()
    evaluator = EVCSRewardEvaluator(network_data)

    env = EVCSChargingGameEnv(network_dir=network_dir, network_name=network_name, random_seed=42)
    env.reset(seed=42)

    # 构造不同的价格序列
    price_levels = [0.3, 0.5, 0.7]  # 低、中、高

    print("\n连续 3 步评估，价格分别为 30%, 50%, 70% 位置:")

    for step, level in enumerate(price_levels):
        prices_dict = {}
        actions_dict = {}
        for agent_name in evaluator.agent_names:
            bounds = network_data.charging_nodes[agent_name]
            price = bounds[0] + level * (bounds[1] - bounds[0])
            prices_dict[agent_name] = np.full(evaluator.n_periods, price)
            actions_dict[agent_name] = np.full(evaluator.n_periods, level, dtype=np.float32)

        # 新评估器（每次独立）
        eval_result = evaluator.evaluate(prices_dict, seed=42)

        # 旧环境（可能有状态继承）
        obs, rewards_env, _, _, info = env.step(actions_dict)

        print(f"\n  Step {step + 1} (价格水平 {level*100:.0f}%):")
        print(f"    新评估器总收益: {sum(eval_result.rewards.values()):.2f}")
        print(f"    旧环境总收益:   {sum(rewards_env.values()):.2f}")
        print(f"    新评估器 UE 迭代: {eval_result.ue_iterations}")
        print(f"    旧环境 UE 迭代:   {info.get('ue_iterations', 'N/A')}")


def main():
    """运行所有对比测试"""
    print("\n" + "=" * 70)
    print("EVCSRewardEvaluator vs EVCSChargingGameEnv 对比测试套件")
    print("=" * 70)

    # 单步对比
    compare_single_step()

    # 多步对比（可选，耗时较长）
    print("\n" + "-" * 70)
    run_multi = input("是否运行多步对比测试? (y/n, 默认 n): ").strip().lower()
    if run_multi == 'y':
        compare_multiple_steps()

    print("\n" + "=" * 70)
    print("对比测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
