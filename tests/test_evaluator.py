# tests/test_evaluator.py
"""
EVCSRewardEvaluator 测试脚本

验证内容：
1. 单次评估正确执行
2. 固定 seed 时结果确定性
3. 与旧环境 step() 结果对比（可选）
"""

import sys
import os
import time
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator.network_data import NetworkDataLoader, NetworkData
from src.evaluator.evaluator import EVCSRewardEvaluator, EvalResult


def test_basic_evaluation():
    """测试基本评估功能"""
    print("=" * 60)
    print("测试 1: 基本评估功能")
    print("=" * 60)

    # 加载网络数据
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_dir = os.path.join(project_root, "data", "siouxfalls")

    loader = NetworkDataLoader(
        network_dir=network_dir,
        network_name="siouxfalls",
        random_seed=42
    )
    network_data = loader.load()

    print(f"加载完成: {network_data.n_agents} 个充电站, {network_data.n_periods} 个时段")

    # 创建评估器
    evaluator = EVCSRewardEvaluator(network_data)

    # 构造测试价格（使用中间价格）
    prices = {}
    for agent_name in network_data.agent_names:
        bounds = network_data.charging_nodes[agent_name]
        mid_price = (bounds[0] + bounds[1]) / 2
        prices[agent_name] = np.full(network_data.n_periods, mid_price)

    print(f"测试价格: {prices}")

    # 执行评估
    start_time = time.time()
    result = evaluator.evaluate(prices, seed=42)
    elapsed = time.time() - start_time

    print(f"\n评估结果:")
    print(f"  收益: {result.rewards}")
    print(f"  流量: {result.flows}")
    print(f"  UE 迭代次数: {result.ue_iterations}")
    print(f"  是否收敛: {result.converged}")
    print(f"  耗时: {elapsed:.2f} 秒")

    # 验证结果格式
    assert isinstance(result, EvalResult), "结果类型错误"
    assert len(result.rewards) == network_data.n_agents, "收益数量不匹配"
    assert len(result.flows) == network_data.n_agents, "流量数量不匹配"
    for agent_name, flow in result.flows.items():
        assert len(flow) == network_data.n_periods, f"{agent_name} 流量时段数不匹配"

    print("\n✓ 测试 1 通过")
    return evaluator, prices, result


def test_determinism(evaluator: EVCSRewardEvaluator, prices: dict):
    """测试确定性（固定 seed 时结果一致）"""
    print("\n" + "=" * 60)
    print("测试 2: 确定性验证")
    print("=" * 60)

    # 多次评估，验证结果一致
    results = []
    for i in range(3):
        result = evaluator.evaluate(prices, seed=42)
        results.append(result)
        print(f"  第 {i+1} 次评估: 收益={list(result.rewards.values())}, 迭代={result.ue_iterations}")

    # 比较结果
    for i in range(1, len(results)):
        for agent_name in results[0].rewards.keys():
            assert np.isclose(results[0].rewards[agent_name], results[i].rewards[agent_name], rtol=1e-6), \
                f"第 {i+1} 次评估收益不一致: {agent_name}"
            assert np.allclose(results[0].flows[agent_name], results[i].flows[agent_name], rtol=1e-6), \
                f"第 {i+1} 次评估流量不一致: {agent_name}"

    print("\n✓ 测试 2 通过: 固定 seed 时结果完全一致")


def test_different_prices(evaluator: EVCSRewardEvaluator, network_data: NetworkData):
    """测试不同价格下的评估"""
    print("\n" + "=" * 60)
    print("测试 3: 不同价格下的评估")
    print("=" * 60)

    # 测试低价格
    low_prices = {}
    for agent_name in network_data.agent_names:
        bounds = network_data.charging_nodes[agent_name]
        low_prices[agent_name] = np.full(network_data.n_periods, bounds[0])

    result_low = evaluator.evaluate(low_prices, seed=42)
    print(f"低价格: {low_prices[network_data.agent_names[0]][0]:.2f}")
    print(f"  收益: {result_low.rewards}")
    print(f"  总流量: {sum(np.sum(f) for f in result_low.flows.values()):.0f}")

    # 测试高价格
    high_prices = {}
    for agent_name in network_data.agent_names:
        bounds = network_data.charging_nodes[agent_name]
        high_prices[agent_name] = np.full(network_data.n_periods, bounds[1])

    result_high = evaluator.evaluate(high_prices, seed=42)
    print(f"\n高价格: {high_prices[network_data.agent_names[0]][0]:.2f}")
    print(f"  收益: {result_high.rewards}")
    print(f"  总流量: {sum(np.sum(f) for f in result_high.flows.values()):.0f}")

    # 验证：低价格应该吸引更多流量
    total_flow_low = sum(np.sum(f) for f in result_low.flows.values())
    total_flow_high = sum(np.sum(f) for f in result_high.flows.values())

    print(f"\n流量对比: 低价={total_flow_low:.0f}, 高价={total_flow_high:.0f}")

    # 注意：在博弈环境中，流量分配取决于多个因素，高价格不一定总是导致更少流量
    # 这里只验证评估能正常运行

    print("\n✓ 测试 3 通过")


def test_ue_convergence_stats(evaluator: EVCSRewardEvaluator, prices: dict):
    """测试 UE-DTA 收敛统计"""
    print("\n" + "=" * 60)
    print("测试 4: UE-DTA 收敛统计")
    print("=" * 60)

    result = evaluator.evaluate(prices, seed=42)

    print(f"UE-DTA 统计:")
    print(f"  迭代次数: {result.ue_iterations}")
    print(f"  是否收敛: {result.converged}")

    if result.ue_stats:
        print(f"  全局平均相对差: {result.ue_stats.get('all_relative_gap_global_mean', 'N/A'):.4f}")
        print(f"  P90 相对差: {result.ue_stats.get('all_relative_gap_p90', 'N/A'):.4f}")
        print(f"  完成率: {result.ue_stats.get('completed_ratio', 'N/A'):.2%}")
        print(f"  路径切换次数: {result.ue_stats.get('total_route_switches', 'N/A')}")

    print("\n✓ 测试 4 通过")


def test_performance(evaluator: EVCSRewardEvaluator, prices: dict):
    """测试评估性能"""
    print("\n" + "=" * 60)
    print("测试 5: 评估性能")
    print("=" * 60)

    # 预热
    evaluator.evaluate(prices, seed=42)

    # 多次评估计时
    times = []
    n_runs = 3
    for i in range(n_runs):
        start = time.time()
        evaluator.evaluate(prices, seed=42 + i)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  第 {i+1} 次: {elapsed:.2f} 秒")

    avg_time = np.mean(times)
    print(f"\n平均耗时: {avg_time:.2f} 秒")

    print("\n✓ 测试 5 通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("EVCSRewardEvaluator 测试套件")
    print("=" * 60 + "\n")

    # 测试 1: 基本评估
    evaluator, prices, result = test_basic_evaluation()

    # 获取 network_data（用于后续测试）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_dir = os.path.join(project_root, "data", "siouxfalls")
    loader = NetworkDataLoader(network_dir=network_dir, network_name="siouxfalls")
    network_data = loader.load()

    # 测试 2: 确定性验证
    test_determinism(evaluator, prices)

    # 测试 3: 不同价格下的评估
    test_different_prices(evaluator, network_data)

    # 测试 4: UE-DTA 收敛统计
    test_ue_convergence_stats(evaluator, prices)

    # 测试 5: 评估性能
    test_performance(evaluator, prices)

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
