# tests/test_pool.py
"""
ParallelEvaluatorPool 测试脚本

验证内容：
1. 单次评估正确性
2. 批量并行评估正确性
3. 异步提交功能
4. 多 Worker 稳定性
5. 与串行评估结果一致性
6. 资源清理
"""

import sys
import os
import time
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_single_evaluate():
    """测试单次评估"""
    from src.evaluator import NetworkDataLoader, ParallelEvaluatorPool

    print("=" * 60)
    print("测试 1: 单次评估")
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

    print(f"加载完成: {network_data.n_agents} 个充电站")

    # 构造测试价格
    prices = {}
    for agent_name in network_data.agent_names:
        bounds = network_data.charging_nodes[agent_name]
        mid_price = (bounds[0] + bounds[1]) / 2
        prices[agent_name] = np.full(network_data.n_periods, mid_price)

    # 使用 ParallelEvaluatorPool 评估
    with ParallelEvaluatorPool(network_data, n_workers=2) as pool:
        print(f"启动进程池: {pool.n_workers} 个 Worker")

        start_time = time.time()
        result = pool.evaluate(prices, seed=42)
        elapsed = time.time() - start_time

        print(f"\n评估结果:")
        print(f"  收益: {result.rewards}")
        print(f"  UE 迭代次数: {result.ue_iterations}")
        print(f"  是否收敛: {result.converged}")
        print(f"  耗时: {elapsed:.2f} 秒")

    # 验证结果格式
    assert len(result.rewards) == network_data.n_agents, "收益数量不匹配"
    assert len(result.flows) == network_data.n_agents, "流量数量不匹配"

    print("\n✓ 测试 1 通过")
    return network_data, prices


def test_batch_evaluate(network_data, prices):
    """测试批量并行评估"""
    from src.evaluator import ParallelEvaluatorPool

    print("\n" + "=" * 60)
    print("测试 2: 批量并行评估")
    print("=" * 60)

    # 构造多组价格
    n_batch = 4
    prices_list = []
    seeds = []
    for i in range(n_batch):
        # 在基础价格上添加随机扰动
        perturbed_prices = {}
        for agent_name, base_price in prices.items():
            perturbed_prices[agent_name] = base_price * (1 + 0.1 * (i - n_batch // 2))
        prices_list.append(perturbed_prices)
        seeds.append(42 + i)

    # 并行评估
    with ParallelEvaluatorPool(network_data, n_workers=2) as pool:
        print(f"批量评估 {n_batch} 组价格...")

        start_time = time.time()
        results = pool.evaluate_batch(prices_list, seeds)
        elapsed = time.time() - start_time

        print(f"\n批量评估完成，耗时: {elapsed:.2f} 秒")
        for i, result in enumerate(results):
            total_reward = sum(result.rewards.values())
            print(f"  第 {i+1} 组: 总收益={total_reward:.2f}, 迭代={result.ue_iterations}")

    # 验证结果数量
    assert len(results) == n_batch, "结果数量不匹配"

    print("\n✓ 测试 2 通过")
    return results


def test_async_submit(network_data, prices):
    """测试异步提交"""
    from src.evaluator import ParallelEvaluatorPool

    print("\n" + "=" * 60)
    print("测试 3: 异步提交")
    print("=" * 60)

    with ParallelEvaluatorPool(network_data, n_workers=2) as pool:
        # 异步提交多个任务
        futures = []
        n_tasks = 3
        for i in range(n_tasks):
            future = pool.submit(prices, seed=42 + i)
            futures.append(future)
            print(f"  提交任务 {i+1}")

        # 等待结果
        print("\n等待结果...")
        for i, future in enumerate(futures):
            result = future.result()
            total_reward = sum(result.rewards.values())
            print(f"  任务 {i+1} 完成: 总收益={total_reward:.2f}")

    print("\n✓ 测试 3 通过")


def test_consistency_with_serial(network_data, prices):
    """测试与串行评估结果一致性"""
    from src.evaluator import EVCSRewardEvaluator, ParallelEvaluatorPool

    print("\n" + "=" * 60)
    print("测试 4: 与串行评估结果一致性")
    print("=" * 60)

    seed = 42

    # 串行评估
    evaluator = EVCSRewardEvaluator(network_data)
    serial_result = evaluator.evaluate(prices, seed=seed)
    print(f"串行评估收益: {serial_result.rewards}")

    # 并行评估
    with ParallelEvaluatorPool(network_data, n_workers=2) as pool:
        parallel_result = pool.evaluate(prices, seed=seed)
        print(f"并行评估收益: {parallel_result.rewards}")

    # 比较结果
    for agent_name in serial_result.rewards.keys():
        serial_reward = serial_result.rewards[agent_name]
        parallel_reward = parallel_result.rewards[agent_name]
        assert np.isclose(serial_reward, parallel_reward, rtol=1e-6), \
            f"{agent_name} 收益不一致: 串行={serial_reward}, 并行={parallel_reward}"

        serial_flow = serial_result.flows[agent_name]
        parallel_flow = parallel_result.flows[agent_name]
        assert np.allclose(serial_flow, parallel_flow, rtol=1e-6), \
            f"{agent_name} 流量不一致"

    print("\n✓ 测试 4 通过: 串行与并行结果完全一致")


def test_worker_stability(network_data, prices):
    """测试多 Worker 稳定性"""
    from src.evaluator import ParallelEvaluatorPool

    print("\n" + "=" * 60)
    print("测试 5: 多 Worker 稳定性")
    print("=" * 60)

    n_iterations = 5

    with ParallelEvaluatorPool(network_data, n_workers=2) as pool:
        print(f"连续执行 {n_iterations} 次批量评估...")

        for iteration in range(n_iterations):
            # 每次评估 2 组价格
            prices_list = [prices, prices]
            seeds = [42, 43]

            results = pool.evaluate_batch(prices_list, seeds)

            total_rewards = [sum(r.rewards.values()) for r in results]
            print(f"  第 {iteration+1} 轮: 收益={total_rewards}")

    print("\n✓ 测试 5 通过: Worker 稳定运行")


def test_n_workers_resolution():
    """测试 n_workers 参数解析"""
    from src.evaluator import ParallelEvaluatorPool
    from src.evaluator.pool import ParallelEvaluatorPool as Pool

    print("\n" + "=" * 60)
    print("测试 6: n_workers 参数解析")
    print("=" * 60)

    # 测试 -1（自动）
    auto_workers = Pool._resolve_n_workers(-1)
    cpu_count = os.cpu_count() or 1
    expected = min(cpu_count - 1, 4) if cpu_count > 1 else 1
    print(f"  n_workers=-1: 解析为 {auto_workers} (CPU 数={cpu_count}, 期望={expected})")
    assert auto_workers == expected, f"自动解析错误: {auto_workers} != {expected}"

    # 测试正整数
    explicit_workers = Pool._resolve_n_workers(3)
    print(f"  n_workers=3: 解析为 {explicit_workers}")
    assert explicit_workers == 3, "显式指定解析错误"

    # 测试非法值
    try:
        Pool._resolve_n_workers(0)
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"  n_workers=0: 正确抛出异常 - {e}")

    print("\n✓ 测试 6 通过")


def test_context_manager():
    """测试上下文管理器"""
    from src.evaluator import NetworkDataLoader, ParallelEvaluatorPool

    print("\n" + "=" * 60)
    print("测试 7: 上下文管理器")
    print("=" * 60)

    # 加载网络数据
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_dir = os.path.join(project_root, "data", "siouxfalls")
    loader = NetworkDataLoader(network_dir=network_dir, network_name="siouxfalls")
    network_data = loader.load()

    # 使用 with 语句
    pool = ParallelEvaluatorPool(network_data, n_workers=2)
    assert not pool.is_running, "Pool 创建后不应自动启动"

    with pool:
        assert pool.is_running, "进入 with 后应该启动"
        print("  Pool 在 with 内运行中")

    assert not pool.is_running, "退出 with 后应该关闭"
    print("  Pool 在 with 外已关闭")

    print("\n✓ 测试 7 通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("ParallelEvaluatorPool 测试套件")
    print("=" * 60 + "\n")

    # 测试 1: 单次评估
    network_data, prices = test_single_evaluate()

    # 测试 2: 批量并行评估
    test_batch_evaluate(network_data, prices)

    # 测试 3: 异步提交
    test_async_submit(network_data, prices)

    # 测试 4: 与串行评估结果一致性
    test_consistency_with_serial(network_data, prices)

    # 测试 5: 多 Worker 稳定性
    test_worker_stability(network_data, prices)

    # 测试 6: n_workers 参数解析
    test_n_workers_resolution()

    # 测试 7: 上下文管理器
    test_context_manager()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
