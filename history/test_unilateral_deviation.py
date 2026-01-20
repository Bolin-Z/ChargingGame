"""
单边偏离测试：验证解是否为纳什均衡

纳什均衡定义：任何智能体单方面改变策略都无法获得更高收益

测试方法：
1. 固定其他智能体的策略
2. 让一个智能体尝试不同价格
3. 如果偏离后收益增加，说明不是均衡
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List
from pathlib import Path

# history/ 目录下的脚本需要回退一级到项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from src.utils.config import PROFILE_SIOUXFALLS


def load_final_prices_from_experiment(result_dir: str) -> Dict[str, List[float]]:
    """从实验结果目录中读取最终价格"""
    json_path = Path(result_dir) / "step_records.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取最后一条记录的actual_prices
    last_record = data['records'][-1]
    final_prices = last_record['actual_prices']

    print(f"从实验结果加载最终价格 (step {last_record['step']}):")
    for agent, prices in final_prices.items():
        print(f"  Agent {agent}: {[round(p, 2) for p in prices]}")

    return final_prices


def prices_to_actions(prices: Dict[str, List[float]],
                      price_min: float = 0.5,
                      price_max: float = 2.0) -> Dict[str, np.ndarray]:
    """将实际价格转换为归一化动作 [0, 1]"""
    actions = {}
    for agent, price_list in prices.items():
        normalized = [(p - price_min) / (price_max - price_min) for p in price_list]
        actions[agent] = np.array(normalized, dtype=np.float32)
    return actions


def run_fixed_strategy(env: EVCSChargingGameEnv,
                       actions: Dict[str, np.ndarray],
                       n_runs: int = 5) -> Dict[str, float]:
    """
    运行固定策略多次，返回平均收益

    由于UE-DTA内部存在随机路径切换，多次运行取平均以平滑随机波动

    Args:
        env: 环境实例
        actions: 各智能体的动作
        n_runs: 运行次数

    Returns:
        avg_rewards: 各智能体的平均收益
    """
    all_rewards = {agent: [] for agent in env.agents}

    for _ in range(n_runs):
        env.reset()
        obs, rewards, terms, truncs, infos = env.step(actions)
        for agent, r in rewards.items():
            all_rewards[agent].append(r)

    avg_rewards = {agent: np.mean(r_list) for agent, r_list in all_rewards.items()}
    return avg_rewards


def test_unilateral_deviation(env: EVCSChargingGameEnv,
                               base_prices: Dict[str, List[float]],
                               test_agent: str,
                               deviation_prices: List[List[float]],
                               n_runs: int = 5) -> List[Dict]:
    """
    测试单个智能体的单边偏离

    Args:
        env: 环境
        base_prices: 基准价格策略
        test_agent: 测试偏离的智能体
        deviation_prices: 要测试的偏离价格列表
        n_runs: 每个策略运行次数（取平均）

    Returns:
        results: 偏离测试结果列表
    """
    results = []

    # 1. 先测试基准策略
    base_actions = prices_to_actions(base_prices)
    base_rewards = run_fixed_strategy(env, base_actions, n_runs)
    base_reward = base_rewards[test_agent]

    results.append({
        'type': 'baseline',
        'prices': base_prices[test_agent],
        'reward': base_reward,
        'all_rewards': base_rewards,
    })

    # 2. 测试各种偏离
    for dev_prices in deviation_prices:
        # 复制基准策略，只修改测试智能体
        dev_price_dict = {k: v.copy() for k, v in base_prices.items()}
        dev_price_dict[test_agent] = dev_prices

        dev_actions = prices_to_actions(dev_price_dict)
        dev_rewards = run_fixed_strategy(env, dev_actions, n_runs)
        dev_reward = dev_rewards[test_agent]

        results.append({
            'type': 'deviation',
            'prices': dev_prices,
            'reward': dev_reward,
            'all_rewards': dev_rewards,
            'gain': dev_reward - base_reward,
            'gain_pct': (dev_reward - base_reward) / base_reward * 100 if base_reward != 0 else 0,
        })

    return results


def generate_deviation_prices(base_prices: List[float],
                               price_min: float = 0.5,
                               price_max: float = 2.0) -> List[List[float]]:
    """
    生成偏离价格列表

    策略：
    1. 全部降到最低价
    2. 全部升到最高价
    3. 均匀偏移
    """
    deviations = []

    # 全部最低价
    deviations.append([price_min] * len(base_prices))

    # 全部最高价
    deviations.append([price_max] * len(base_prices))

    # 中间价格
    mid_price = (price_min + price_max) / 2
    deviations.append([mid_price] * len(base_prices))

    # 比基准降低 0.3
    lower = [max(price_min, p - 0.3) for p in base_prices]
    deviations.append(lower)

    # 比基准升高 0.3
    higher = [min(price_max, p + 0.3) for p in base_prices]
    deviations.append(higher)

    return deviations


def print_deviation_results(agent: str, results: List[Dict], solution_name: str):
    """打印偏离测试结果"""
    print(f"\n{'='*60}")
    print(f"  智能体 {agent} 单边偏离测试 ({solution_name})")
    print(f"{'='*60}")

    baseline = results[0]
    print(f"\n基准策略:")
    print(f"  价格: {[round(p, 2) for p in baseline['prices']]}")
    print(f"  收益: {baseline['reward']:.1f}")

    print(f"\n偏离测试:")
    print(f"  {'价格策略':<35} | {'收益':>10} | {'变化':>12} | {'结论':<10}")
    print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    has_profitable_deviation = False

    for r in results[1:]:
        prices_str = str([round(p, 1) for p in r['prices']])[:35]
        gain_str = f"{r['gain']:+.1f} ({r['gain_pct']:+.1f}%)"

        if r['gain'] > baseline['reward'] * 0.01:  # 超过1%的收益增加
            conclusion = "可盈利偏离!"
            has_profitable_deviation = True
        elif r['gain'] < -baseline['reward'] * 0.01:
            conclusion = "亏损"
        else:
            conclusion = "无显著变化"

        print(f"  {prices_str:<35} | {r['reward']:>10.1f} | {gain_str:>12} | {conclusion:<10}")

    return has_profitable_deviation


def main(result_dir: str = None):
    """
    单边偏离测试主函数

    Args:
        result_dir: 实验结果目录路径，如果不指定则使用默认路径
    """
    # 默认实验结果目录
    if result_dir is None:
        result_dir = "results/siouxfalls/MADDPG/seed42/01_19_23_05"

    print("="*60)
    print("  单边偏离测试：验证纳什均衡")
    print("="*60)
    print(f"\n实验目录: {result_dir}")

    # 从实验结果加载最终价格
    experiment_prices = load_final_prices_from_experiment(result_dir)

    scenario = PROFILE_SIOUXFALLS
    n_runs = 5  # 每个策略运行次数（取平均以平滑随机波动）

    # 创建环境
    print("\n初始化环境...")
    env = EVCSChargingGameEnv(
        network_dir=scenario.network_dir,
        network_name=scenario.network_name,
        random_seed=42,
        max_steps=1000,
        convergence_threshold=scenario.convergence_threshold,
        stable_steps_required=scenario.stable_steps_required
    )

    agents = env.agents
    print(f"智能体: {agents}")
    print(f"每个策略运行 {n_runs} 次取平均")

    # ============== 测试实验结果 ==============
    print("\n" + "#"*60)
    print("#  测试实验最终价格")
    print("#"*60)

    profitable_agents = []

    for agent in agents:
        dev_prices = generate_deviation_prices(experiment_prices[agent])
        results = test_unilateral_deviation(env, experiment_prices, agent, dev_prices, n_runs)
        has_profit = print_deviation_results(agent, results, "实验最终价格")
        if has_profit:
            profitable_agents.append(agent)

    # ============== 总结 ==============
    print("\n" + "#"*60)
    print("#  总结")
    print("#"*60)

    if not profitable_agents:
        print("  ✅ 没有智能体可以通过单边偏离获利 → 可能是纳什均衡")
    else:
        print(f"  ❌ 智能体 {profitable_agents} 可以通过偏离获利 → 不是纳什均衡")

    env.close()
    print("\n测试完成!")


if __name__ == '__main__':
    main()
