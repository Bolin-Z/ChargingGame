"""
GameTrainer v1 版本测试

测试异步事件驱动的博弈训练器功能。

注意：完整测试需要真实的网络数据和较长运行时间。
这里主要测试组件初始化和基本逻辑。

运行方式：python tests/test_game_trainer_v1.py
"""

import sys
import traceback
import time

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
    TrainingMetrics,
    PendingTask,
    create_algorithm,
)
from src.evaluator.network_data import NetworkDataLoader


# 测试配置
NETWORK_DIR = "data/siouxfalls"
NETWORK_NAME = "siouxfalls"
SEED = 42


class TestResult:
    """测试结果统计"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def add_fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"测试完成: {self.passed}/{total} 通过")
        if self.failed > 0:
            print(f"\n失败的测试:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        return self.failed == 0


def run_test(result: TestResult, name: str, test_func):
    """运行单个测试"""
    try:
        test_func()
        result.add_pass(name)
    except AssertionError as e:
        result.add_fail(name, str(e))
    except Exception as e:
        result.add_fail(name, f"异常: {type(e).__name__}: {e}")
        traceback.print_exc()


def create_test_task(algo_name: str = "MADDPG", max_evaluations: int = 10):
    """创建测试用实验任务"""
    trainer_config = TrainerConfig(
        network_dir=NETWORK_DIR,
        network_name=NETWORK_NAME,
        max_evaluations=max_evaluations,
        learn_interval=5,
        n_workers=1,  # 测试时使用单 Worker
    )

    nashconv_config = NashConvConfig(
        exploitability_threshold=0.05,
        check_interval=5,
        warmup=5,  # 测试时减小 warmup
        n_starts=2,  # 测试时减少起点数
        optim_steps=10,  # 测试时减少步数
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


def test_config_v1(result: TestResult):
    """测试配置类"""
    print("\n[1] 测试配置类")

    def test_trainer_config():
        config = TrainerConfig(
            network_dir="data/siouxfalls",
            network_name="siouxfalls",
        )
        assert config.max_evaluations == 1000
        assert config.learn_interval == 5
        assert config.device == "auto"

    def test_nashconv_config():
        config = NashConvConfig()
        assert config.exploitability_threshold == 0.05
        assert config.check_interval == 10
        assert config.warmup == 100

    def test_experiment_task():
        task = create_test_task("MADDPG")
        assert task.algo_name == "MADDPG"
        assert task.seed == SEED

        # 测试输出路径生成
        output_path = task.get_output_path()
        assert "siouxfalls" in output_path
        assert "MADDPG" in output_path
        assert f"seed{SEED}" in output_path

    run_test(result, "TrainerConfig 默认值", test_trainer_config)
    run_test(result, "NashConvConfig 默认值", test_nashconv_config)
    run_test(result, "ExperimentTask 创建", test_experiment_task)


def test_create_algorithm(result: TestResult):
    """测试算法工厂函数"""
    print("\n[2] 测试算法工厂函数")

    def test_create_maddpg():
        task = create_test_task("MADDPG")
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        assert algo.name == "MADDPG"
        assert algo.agent_names == network_data.agent_names
        assert algo.n_periods == network_data.n_periods

    def test_create_iddpg():
        task = create_test_task("IDDPG")
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)
        assert algo.name == "IDDPG"

    def test_create_mfddpg():
        task = create_test_task("MFDDPG")
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)
        assert algo.name == "MFDDPG"

    def test_unknown_algorithm():
        task = create_test_task("MADDPG")
        task.algo_name = "UNKNOWN"
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()

        try:
            create_algorithm(task, network_data)
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "Unknown algorithm" in str(e)

    run_test(result, "创建 MADDPG", test_create_maddpg)
    run_test(result, "创建 IDDPG", test_create_iddpg)
    run_test(result, "创建 MFDDPG", test_create_mfddpg)
    run_test(result, "未知算法报错", test_unknown_algorithm)


def test_trainer_init(result: TestResult):
    """测试 GameTrainer 初始化"""
    print("\n[3] 测试 GameTrainer 初始化")

    def test_components_created():
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        # 检查组件初始化
        assert trainer.network_data is not None
        assert trainer.pool is not None
        assert trainer.history is not None
        assert trainer.nashconv_checker is not None

        # 检查初始状态
        assert trainer._metrics.total_evaluations == 0
        assert trainer._converged is False
        assert len(trainer._pending_tasks) == 0

    def test_history_initialized():
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        # 检查 history 配置
        assert trainer.history.agent_names == network_data.agent_names
        assert trainer.history.n_periods == network_data.n_periods

        # 检查初始信念
        beliefs = trainer.history.get_beliefs()
        assert beliefs.shape == (len(network_data.agent_names), network_data.n_periods)
        assert np.allclose(beliefs, 0.5)  # 默认初始值

    run_test(result, "组件正确创建", test_components_created)
    run_test(result, "History 正确初始化", test_history_initialized)


def test_actions_to_prices(result: TestResult):
    """测试动作到价格的转换"""
    print("\n[4] 测试动作到价格转换")

    def test_boundary_values():
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        # 测试边界值
        actions_min = {name: np.zeros(network_data.n_periods) for name in network_data.agent_names}
        actions_max = {name: np.ones(network_data.n_periods) for name in network_data.agent_names}

        prices_min = trainer._actions_to_prices(actions_min)
        prices_max = trainer._actions_to_prices(actions_max)

        for agent_name in network_data.agent_names:
            price_range = network_data.charging_nodes[agent_name]
            expected_min, expected_max = price_range[0], price_range[1]

            assert np.allclose(prices_min[agent_name], expected_min), f"{agent_name}: 最小价格不匹配"
            assert np.allclose(prices_max[agent_name], expected_max), f"{agent_name}: 最大价格不匹配"

    def test_mid_value():
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        # 测试中间值
        actions_mid = {name: np.full(network_data.n_periods, 0.5) for name in network_data.agent_names}
        prices_mid = trainer._actions_to_prices(actions_mid)

        for agent_name in network_data.agent_names:
            price_range = network_data.charging_nodes[agent_name]
            expected_mid = (price_range[0] + price_range[1]) / 2

            assert np.allclose(prices_mid[agent_name], expected_mid), f"{agent_name}: 中间价格不匹配"

    run_test(result, "边界值转换", test_boundary_values)
    run_test(result, "中间值转换", test_mid_value)


def test_price_change_rate(result: TestResult):
    """测试价格变化率计算"""
    print("\n[5] 测试价格变化率计算")

    def test_zero_change():
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        actions = {name: np.full(network_data.n_periods, 0.5) for name in network_data.agent_names}
        change_rate = trainer._compute_price_change_rate(actions, actions)

        assert change_rate == 0.0, "相同动作变化率应为 0"

    def test_positive_change():
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        prev_actions = {name: np.full(network_data.n_periods, 0.5) for name in network_data.agent_names}
        curr_actions = {name: np.full(network_data.n_periods, 0.6) for name in network_data.agent_names}

        change_rate = trainer._compute_price_change_rate(prev_actions, curr_actions)

        # 相对变化率 = |0.6 - 0.5| / (0.5 + 1e-8) ≈ 0.2
        assert 0.1 < change_rate < 0.3, f"变化率应约为 0.2，实际 {change_rate}"

    run_test(result, "零变化率", test_zero_change)
    run_test(result, "正向变化率", test_positive_change)


def test_training_metrics(result: TestResult):
    """测试训练指标数据结构"""
    print("\n[6] 测试训练指标")

    def test_metrics_init():
        metrics = TrainingMetrics()
        assert metrics.total_evaluations == 0
        assert metrics.total_learns == 0
        assert len(metrics.nashconv_history) == 0
        assert len(metrics.reward_history) == 0

    def test_metrics_update():
        metrics = TrainingMetrics()
        metrics.total_evaluations = 10
        metrics.total_learns = 2
        metrics.nashconv_history.append(0.1)
        metrics.reward_history.append({"agent1": 100.0})

        assert metrics.total_evaluations == 10
        assert metrics.total_learns == 2
        assert len(metrics.nashconv_history) == 1
        assert len(metrics.reward_history) == 1

    run_test(result, "指标初始化", test_metrics_init)
    run_test(result, "指标更新", test_metrics_update)


def test_training_result(result: TestResult):
    """测试训练结果数据结构"""
    print("\n[7] 测试训练结果")

    def test_result_structure():
        metrics = TrainingMetrics()
        metrics.total_evaluations = 100
        metrics.nashconv_history = [0.2, 0.15, 0.1, 0.05]

        training_result = TrainingResult(
            converged=True,
            final_nashconv=0.05,
            final_exploitability=0.0125,
            final_beliefs=np.zeros((4, 8)),
            final_rewards={"agent1": 100.0},
            metrics=metrics,
            total_time=60.0,
        )

        assert training_result.converged is True
        assert training_result.final_nashconv == 0.05
        assert training_result.total_time == 60.0
        assert training_result.metrics.total_evaluations == 100

    run_test(result, "结果结构正确", test_result_structure)


def test_short_training_run(result: TestResult):
    """测试短训练运行（集成测试）"""
    print("\n[8] 测试短训练运行（集成测试）")

    def test_maddpg_short_run():
        """运行少量评估的 MADDPG 训练"""
        task = create_test_task("MADDPG", max_evaluations=5)
        loader = NetworkDataLoader(
            network_dir=task.trainer_config.network_dir,
            network_name=task.trainer_config.network_name,
            random_seed=task.seed,
        )
        network_data = loader.load()
        algo = create_algorithm(task, network_data)

        trainer = GameTrainer(task, algo)

        print("\n    开始短训练（5 次评估）...")
        start_time = time.time()
        training_result = trainer.train()
        elapsed = time.time() - start_time
        print(f"    训练完成，耗时 {elapsed:.1f}s")

        # 验证结果
        assert training_result.metrics.total_evaluations >= 5, "应至少完成 5 次评估"
        assert training_result.final_beliefs is not None
        assert training_result.total_time > 0

        print(f"    总评估次数: {training_result.metrics.total_evaluations}")
        print(f"    总学习次数: {training_result.metrics.total_learns}")
        print(f"    是否收敛: {training_result.converged}")

    run_test(result, "MADDPG 短训练", test_maddpg_short_run)


def main():
    print("=" * 60)
    print("GameTrainer v1 版本测试")
    print("=" * 60)

    result = TestResult()

    test_config_v1(result)
    test_create_algorithm(result)
    test_trainer_init(result)
    test_actions_to_prices(result)
    test_price_change_rate(result)
    test_training_metrics(result)
    test_training_result(result)
    test_short_training_run(result)

    success = result.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
