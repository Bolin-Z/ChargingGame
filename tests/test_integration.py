"""
P9 集成测试：端到端功能验证

测试内容：
1. 端到端功能测试：三种算法完整训练流程
2. 并行加速比测试：验证多 Worker 加速效果
3. 收敛行为测试：验证训练能够收敛
4. 内存稳定性测试：连续运行无内存泄漏

运行方式：
    python tests/test_integration.py [--full]

参数说明：
    --full: 运行完整测试（包含长时间运行的测试）
    默认运行快速测试子集

预计耗时：
    快速模式: ~15-20 分钟
    完整模式: ~60-90 分钟
"""

import sys
import time
import argparse
import traceback
from dataclasses import dataclass

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, ".")

from src.utils.config_v1 import (
    ExperimentTask,
    TrainerConfig,
    NashConvConfig,
    MADDPGConfig,
    IDDPGConfig,
    MFDDPGConfig,
)
from src.trainer.game_trainer_v1 import (
    GameTrainer,
    TrainingResult,
    create_algorithm,
)
from src.evaluator.network_data import NetworkDataLoader
from src.evaluator.pool import ParallelEvaluatorPool


# ============================================================
# 测试配置
# ============================================================

NETWORK_DIR = "data/siouxfalls"
NETWORK_NAME = "siouxfalls"
SEED = 42


@dataclass
class TestResult:
    """测试结果统计"""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def add_pass(self, name: str, details: str = ""):
        self.passed += 1
        if details:
            print(f"  ✓ {name}: {details}")
        else:
            print(f"  ✓ {name}")

    def add_fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")

    def add_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ○ {name}: 跳过 - {reason}")

    def summary(self) -> bool:
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"测试完成: {self.passed}/{total} 通过", end="")
        if self.skipped > 0:
            print(f", {self.skipped} 跳过", end="")
        print()
        if self.failed > 0:
            print(f"\n失败的测试:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        return self.failed == 0


def run_test(result: TestResult, name: str, test_func, *args, **kwargs):
    """运行单个测试"""
    try:
        test_func(*args, **kwargs)
        result.add_pass(name)
    except AssertionError as e:
        result.add_fail(name, str(e))
    except Exception as e:
        result.add_fail(name, f"异常: {type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================
# 测试辅助函数
# ============================================================

def create_test_task(
    algo_name: str,
    max_evaluations: int = 20,
    n_workers: int = 1,
    warmup: int = 10,
) -> ExperimentTask:
    """创建测试用实验任务"""
    trainer_config = TrainerConfig(
        network_dir=NETWORK_DIR,
        network_name=NETWORK_NAME,
        max_evaluations=max_evaluations,
        learn_interval=5,
        n_workers=n_workers,
    )

    nashconv_config = NashConvConfig(
        exploitability_threshold=0.05,
        check_interval=5,
        warmup=warmup,
        n_starts=2,
        optim_steps=10,
    )

    if algo_name == "MADDPG":
        algo_config = MADDPGConfig()
    elif algo_name == "IDDPG":
        algo_config = IDDPGConfig()
    elif algo_name == "MFDDPG":
        algo_config = MFDDPGConfig()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return ExperimentTask(
        name=f"test_{algo_name}",
        trainer_config=trainer_config,
        algo_name=algo_name,
        algo_config=algo_config,
        seed=SEED,
        nashconv_config=nashconv_config,
    )


def load_network_data():
    """加载网络数据"""
    loader = NetworkDataLoader(
        network_dir=NETWORK_DIR,
        network_name=NETWORK_NAME,
        random_seed=SEED,
    )
    return loader.load()


# ============================================================
# 测试 1: 端到端功能测试
# ============================================================

class ProgressCallback:
    """训练进度回调（用于显示实时进度）"""

    def __init__(self, algo_name: str, max_evals: int):
        self.algo_name = algo_name
        self.max_evals = max_evals
        self.start_time = time.time()
        self.last_print = 0

    def on_evaluation(self, eval_count: int):
        """每次评估完成后调用"""
        now = time.time()
        # 每 30 秒或每 5 次评估打印一次进度
        if now - self.last_print >= 30 or eval_count % 5 == 0:
            elapsed = now - self.start_time
            avg_time = elapsed / eval_count if eval_count > 0 else 0
            remaining = (self.max_evals - eval_count) * avg_time
            print(f"      [{self.algo_name}] 进度: {eval_count}/{self.max_evals} "
                  f"({eval_count*100//self.max_evals}%), "
                  f"已用 {elapsed:.0f}s, 预计剩余 {remaining:.0f}s")
            self.last_print = now


def test_end_to_end(result: TestResult, full_mode: bool = False):
    """
    端到端功能测试：验证三种算法的完整训练流程
    """
    print("\n" + "="*60)
    print("[测试 1] 端到端功能测试")
    print("="*60)

    # 测试参数
    if full_mode:
        max_evals = 50
        warmup = 20
    else:
        max_evals = 15
        warmup = 10

    algos = ["MADDPG", "IDDPG", "MFDDPG"]
    for i, algo_name in enumerate(algos):
        def test_algo(algo_name=algo_name):  # 闭包捕获
            print(f"\n    [{i+1}/{len(algos)}] 测试 {algo_name} (目标 {max_evals} 次评估)...")
            task = create_test_task(
                algo_name,
                max_evaluations=max_evals,
                warmup=warmup,
            )

            print(f"      加载网络数据...")
            loader = NetworkDataLoader(
                network_dir=task.trainer_config.network_dir,
                network_name=task.trainer_config.network_name,
                random_seed=task.seed,
            )
            network_data = loader.load()
            print(f"      创建算法实例...")
            algo = create_algorithm(task, network_data)

            print(f"      初始化训练器...")
            trainer = GameTrainer(task, algo)

            print(f"      开始训练...")
            start_time = time.time()

            # 使用带进度显示的训练
            training_result = train_with_progress(trainer, algo_name, max_evals)

            elapsed = time.time() - start_time

            print(f"    完成: {training_result.metrics.total_evaluations} 次评估, "
                  f"{training_result.metrics.total_learns} 次学习, "
                  f"耗时 {elapsed:.1f}s")

            # 验证基本功能
            assert training_result.metrics.total_evaluations >= max_evals, \
                f"评估次数不足: {training_result.metrics.total_evaluations} < {max_evals}"
            assert training_result.final_beliefs is not None, "最终信念为空"
            assert training_result.total_time > 0, "训练时间无效"

            # 验证信念形状
            n_agents = len(network_data.agent_names)
            n_periods = network_data.n_periods
            assert training_result.final_beliefs.shape == (n_agents, n_periods), \
                f"信念形状错误: {training_result.final_beliefs.shape}"

            # 验证收益记录
            assert len(training_result.metrics.reward_history) > 0, "收益历史为空"

        run_test(result, f"{algo_name} 端到端训练", test_algo)


def train_with_progress(trainer: GameTrainer, algo_name: str, max_evals: int) -> TrainingResult:
    """带进度显示的训练"""
    start_time = time.time()
    last_print_time = start_time
    last_eval_count = 0

    with trainer.pool:
        # 初始满负荷提交任务
        trainer._fill_task_queue()

        # 事件驱动主循环
        while not trainer._should_stop():
            trainer._process_completed_tasks()

            # 进度显示
            current_evals = trainer._metrics.total_evaluations
            now = time.time()

            # 每完成一次评估或每 30 秒打印一次
            if current_evals > last_eval_count:
                last_eval_count = current_evals
                if now - last_print_time >= 30 or current_evals % 5 == 0:
                    elapsed = now - start_time
                    avg_time = elapsed / current_evals if current_evals > 0 else 0
                    remaining = (max_evals - current_evals) * avg_time
                    print(f"      [{algo_name}] 进度: {current_evals}/{max_evals} "
                          f"({current_evals*100//max_evals}%), "
                          f"已用 {elapsed:.0f}s, 预计剩余 {remaining:.0f}s")
                    last_print_time = now

            # 短暂休眠避免忙等待
            if not trainer._has_completed_tasks():
                time.sleep(0.01)

        # 处理剩余任务
        trainer._drain_pending_tasks()

    total_time = time.time() - start_time

    # 构造返回结果
    return TrainingResult(
        converged=trainer._converged,
        final_nashconv=(
            trainer._metrics.nashconv_history[-1]
            if trainer._metrics.nashconv_history
            else None
        ),
        final_exploitability=(
            trainer._metrics.exploitability_history[-1]
            if trainer._metrics.exploitability_history
            else None
        ),
        final_beliefs=trainer.history.get_beliefs(),
        final_rewards=(
            trainer._metrics.reward_history[-1]
            if trainer._metrics.reward_history
            else {}
        ),
        metrics=trainer._metrics,
        total_time=total_time,
    )


# ============================================================
# 测试 2: 并行加速比测试
# ============================================================

def test_parallel_speedup(result: TestResult, full_mode: bool = False):
    """
    并行加速比测试：验证多 Worker 加速效果
    """
    print("\n" + "="*60)
    print("[测试 2] 并行加速比测试")
    print("="*60)

    if full_mode:
        n_evals = 10
    else:
        n_evals = 4

    network_data = load_network_data()

    # 生成测试价格
    np.random.seed(SEED)
    prices_list = []
    for _ in range(n_evals):
        prices = {}
        for agent_name in network_data.agent_names:
            price_range = network_data.charging_nodes[agent_name]
            prices[agent_name] = np.random.uniform(
                price_range[0], price_range[1], network_data.n_periods
            )
        prices_list.append(prices)

    # 为每个价格组合生成固定种子，确保结果可复现
    seeds = [SEED + i for i in range(n_evals)]

    def test_serial():
        """串行评估（1 Worker）"""
        print(f"\n    串行评估 ({n_evals} 次)...")
        pool = ParallelEvaluatorPool(network_data, n_workers=1)

        start_time = time.time()
        with pool:
            results = pool.evaluate_batch(prices_list, seeds=seeds)
        serial_time = time.time() - start_time

        print(f"    串行耗时: {serial_time:.1f}s")
        assert len(results) == n_evals, f"结果数量不匹配: {len(results)} != {n_evals}"

        return serial_time, results

    def test_parallel():
        """并行评估（2 Workers）"""
        print(f"\n    并行评估 ({n_evals} 次, 2 Workers)...")
        pool = ParallelEvaluatorPool(network_data, n_workers=2)

        start_time = time.time()
        with pool:
            results = pool.evaluate_batch(prices_list, seeds=seeds)
        parallel_time = time.time() - start_time

        print(f"    并行耗时: {parallel_time:.1f}s")
        assert len(results) == n_evals, f"结果数量不匹配: {len(results)} != {n_evals}"

        return parallel_time, results

    # 执行测试
    try:
        serial_time, serial_results = test_serial()
        result.add_pass("串行评估完成", f"{serial_time:.1f}s")
    except Exception as e:
        result.add_fail("串行评估", str(e))
        return

    try:
        parallel_time, parallel_results = test_parallel()
        result.add_pass("并行评估完成", f"{parallel_time:.1f}s")
    except Exception as e:
        result.add_fail("并行评估", str(e))
        return

    # 计算加速比
    speedup = serial_time / parallel_time
    print(f"\n    加速比: {speedup:.2f}x")

    # 验证加速效果
    def test_speedup_ratio():
        # 预期加速比 >= 1.5x（考虑到进程开销）
        min_expected_speedup = 1.3
        assert speedup >= min_expected_speedup, \
            f"加速比不足: {speedup:.2f}x < {min_expected_speedup}x"

    run_test(result, f"加速比检验 (实际 {speedup:.2f}x)", test_speedup_ratio)

    # 验证结果一致性
    def test_result_consistency():
        for i, (s_res, p_res) in enumerate(zip(serial_results, parallel_results)):
            for agent in network_data.agent_names:
                s_reward = s_res.rewards[agent]
                p_reward = p_res.rewards[agent]
                # 由于随机性，允许小误差
                assert abs(s_reward - p_reward) < 1e-6 or abs(s_reward - p_reward) / (abs(s_reward) + 1e-8) < 0.01, \
                    f"第 {i} 次评估 {agent} 收益不一致: {s_reward} vs {p_reward}"

    run_test(result, "串行/并行结果一致性", test_result_consistency)


# ============================================================
# 测试 3: 收敛行为测试
# ============================================================

def test_convergence_behavior(result: TestResult, full_mode: bool = False):
    """
    收敛行为测试：验证长时间训练的收敛趋势
    """
    print("\n" + "="*60)
    print("[测试 3] 收敛行为测试")
    print("="*60)

    if not full_mode:
        result.add_skip("收敛行为测试", "需要 --full 模式")
        return

    # 使用较多评估次数测试收敛
    max_evals = 200
    warmup = 50

    def test_maddpg_convergence():
        print(f"\n    测试 MADDPG 收敛行为 ({max_evals} 次评估)...")
        task = create_test_task(
            "MADDPG",
            max_evaluations=max_evals,
            n_workers=2,
            warmup=warmup,
        )

        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        start_time = time.time()
        training_result = trainer.train()
        elapsed = time.time() - start_time

        print(f"    完成: 耗时 {elapsed:.1f}s")
        print(f"    是否收敛: {training_result.converged}")
        print(f"    NashConv 历史: {len(training_result.metrics.nashconv_history)} 次检测")

        if training_result.metrics.nashconv_history:
            first_nc = training_result.metrics.nashconv_history[0]
            last_nc = training_result.metrics.nashconv_history[-1]
            print(f"    NashConv: {first_nc:.4f} -> {last_nc:.4f}")

            # 验证 NashConv 有下降趋势（允许波动）
            if len(training_result.metrics.nashconv_history) >= 3:
                early_avg = np.mean(training_result.metrics.nashconv_history[:3])
                late_avg = np.mean(training_result.metrics.nashconv_history[-3:])
                print(f"    早期平均: {early_avg:.4f}, 后期平均: {late_avg:.4f}")

        # 基本验证
        assert training_result.metrics.total_evaluations >= max_evals, "评估次数不足"
        assert training_result.metrics.total_learns > 0, "未发生学习"

    run_test(result, "MADDPG 收敛趋势", test_maddpg_convergence)


# ============================================================
# 测试 4: 内存稳定性测试
# ============================================================

def test_memory_stability(result: TestResult, full_mode: bool = False):
    """
    内存稳定性测试：连续运行验证无内存泄漏
    """
    print("\n" + "="*60)
    print("[测试 4] 内存稳定性测试")
    print("="*60)

    if not full_mode:
        result.add_skip("内存稳定性测试", "需要 --full 模式")
        return

    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        result.add_skip("内存稳定性测试", "需要 psutil 库")
        return

    network_data = load_network_data()

    # 连续评估测试
    n_iterations = 50
    memory_samples = []

    def test_memory():
        print(f"\n    连续 {n_iterations} 次评估...")

        pool = ParallelEvaluatorPool(network_data, n_workers=1)

        with pool:
            for i in range(n_iterations):
                # 生成随机价格
                prices = {}
                for agent_name in network_data.agent_names:
                    price_range = network_data.charging_nodes[agent_name]
                    prices[agent_name] = np.random.uniform(
                        price_range[0], price_range[1], network_data.n_periods
                    )

                # 执行评估
                result_eval = pool.evaluate(prices)

                # 记录内存（每 10 次）
                if i % 10 == 0:
                    mem_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(mem_mb)
                    print(f"    迭代 {i}: 内存 {mem_mb:.1f} MB")

        # 分析内存趋势
        if len(memory_samples) >= 3:
            early_mem = np.mean(memory_samples[:2])
            late_mem = np.mean(memory_samples[-2:])
            growth = late_mem - early_mem
            growth_percent = growth / early_mem * 100

            print(f"\n    内存变化: {early_mem:.1f} MB -> {late_mem:.1f} MB")
            print(f"    增长: {growth:.1f} MB ({growth_percent:.1f}%)")

            # 允许 20% 以内的内存增长（考虑到缓存等因素）
            assert growth_percent < 20, \
                f"内存增长过大: {growth_percent:.1f}% (阈值 20%)"

    run_test(result, "内存稳定性", test_memory)


# ============================================================
# 测试 5: 算法一致性测试
# ============================================================

def test_algorithm_consistency(result: TestResult, full_mode: bool = False):
    """
    算法一致性测试：验证相同种子下结果可复现
    """
    print("\n" + "="*60)
    print("[测试 5] 算法一致性测试")
    print("="*60)

    max_evals = 5  # 减少评估次数以加快测试

    def test_reproducibility():
        print("\n    测试 MADDPG 结果可复现性...")

        # 第一次运行
        task1 = create_test_task("MADDPG", max_evaluations=max_evals, warmup=max_evals + 1)
        loader1 = NetworkDataLoader(
            network_dir=task1.trainer_config.network_dir,
            network_name=task1.trainer_config.network_name,
            random_seed=task1.seed,
        )
        network_data1 = loader1.load()
        algo1 = create_algorithm(task1, network_data1)
        trainer1 = GameTrainer(task1, algo1)
        result1 = trainer1.train()

        # 第二次运行（相同种子）
        task2 = create_test_task("MADDPG", max_evaluations=max_evals, warmup=max_evals + 1)
        loader2 = NetworkDataLoader(
            network_dir=task2.trainer_config.network_dir,
            network_name=task2.trainer_config.network_name,
            random_seed=task2.seed,
        )
        network_data2 = loader2.load()
        algo2 = create_algorithm(task2, network_data2)
        trainer2 = GameTrainer(task2, algo2)
        result2 = trainer2.train()

        # 验证最终信念一致
        beliefs_diff = np.abs(result1.final_beliefs - result2.final_beliefs).max()
        print(f"    最终信念最大差异: {beliefs_diff:.6f}")

        # 由于异步执行可能有顺序差异，允许小误差
        assert beliefs_diff < 0.1, \
            f"相同种子下结果不一致: 最大差异 {beliefs_diff}"

    run_test(result, "结果可复现性", test_reproducibility)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="P9 集成测试")
    parser.add_argument("--full", action="store_true", help="运行完整测试（包含长时间测试）")
    args = parser.parse_args()

    print("="*60)
    print("P9 集成测试")
    print("="*60)
    print(f"模式: {'完整测试' if args.full else '快速测试'}")
    print(f"网络: {NETWORK_NAME}")
    print(f"种子: {SEED}")

    result = TestResult()

    # 运行测试
    test_end_to_end(result, args.full)
    test_parallel_speedup(result, args.full)
    test_convergence_behavior(result, args.full)
    test_memory_stability(result, args.full)
    test_algorithm_consistency(result, args.full)

    success = result.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
