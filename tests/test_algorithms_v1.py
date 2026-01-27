"""
算法 v1 版本测试

测试 MADDPGv1 / IDDPGv1 / MFDDPGv1 的接口正确性和功能完整性。
这些算法基于 AlgorithmBase 接口，接收 beliefs 矩阵作为观测输入。

运行方式：python tests/test_algorithms_v1.py
"""

import sys
import traceback
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, ".")

from src.algorithms.base import AlgorithmBase
from src.algorithms.maddpg.maddpg_v1 import MADDPGv1
from src.algorithms.iddpg.iddpg_v1 import IDDPGv1
from src.algorithms.mfddpg.mfddpg_v1 import MFDDPGv1
from src.game.history import GameHistory


# 测试配置
AGENT_NAMES = ["station_5", "station_12", "station_15", "station_20"]
N_AGENTS = len(AGENT_NAMES)
N_PERIODS = 8
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


def create_beliefs():
    """创建测试用信念矩阵"""
    np.random.seed(SEED)
    return np.random.rand(N_AGENTS, N_PERIODS).astype(np.float32)


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


def test_algorithm_base(result: TestResult):
    """测试算法基类接口"""
    print("\n[1] 测试算法基类接口")

    def test_inherits_from_base():
        for cls, expected_name in [(MADDPGv1, "MADDPG"), (IDDPGv1, "IDDPG"), (MFDDPGv1, "MFDDPG")]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            assert isinstance(algo, AlgorithmBase), f"{cls.__name__} 应继承自 AlgorithmBase"
            assert algo.name == expected_name, f"{cls.__name__}.name 应为 {expected_name}"

    def test_agent_names_property():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            assert algo.agent_names == AGENT_NAMES

    def test_n_periods_property():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            assert algo.n_periods == N_PERIODS

    run_test(result, "继承自 AlgorithmBase", test_inherits_from_base)
    run_test(result, "agent_names 属性", test_agent_names_property)
    run_test(result, "n_periods 属性", test_n_periods_property)


def test_take_action(result: TestResult):
    """测试 take_action 方法"""
    print("\n[2] 测试 take_action 方法")

    def test_returns_correct_format():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            pure_actions, noisy_actions = algo.take_action(beliefs, add_noise=True)

            assert isinstance(pure_actions, dict), f"{cls.__name__}: pure_actions 应为 dict"
            assert isinstance(noisy_actions, dict), f"{cls.__name__}: noisy_actions 应为 dict"
            assert set(pure_actions.keys()) == set(AGENT_NAMES)
            assert set(noisy_actions.keys()) == set(AGENT_NAMES)

            for name in AGENT_NAMES:
                assert pure_actions[name].shape == (N_PERIODS,)
                assert noisy_actions[name].shape == (N_PERIODS,)

    def test_values_in_range():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            pure_actions, noisy_actions = algo.take_action(beliefs, add_noise=True)

            for name in AGENT_NAMES:
                assert np.all(pure_actions[name] >= 0.0) and np.all(pure_actions[name] <= 1.0)
                assert np.all(noisy_actions[name] >= 0.0) and np.all(noisy_actions[name] <= 1.0)

    def test_no_noise_mode():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            pure_actions, noisy_actions = algo.take_action(beliefs, add_noise=False)

            for name in AGENT_NAMES:
                np.testing.assert_array_almost_equal(pure_actions[name], noisy_actions[name])

    def test_deterministic_no_noise():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            pure1, _ = algo.take_action(beliefs, add_noise=False)
            pure2, _ = algo.take_action(beliefs, add_noise=False)

            for name in AGENT_NAMES:
                np.testing.assert_array_almost_equal(pure1[name], pure2[name])

    run_test(result, "返回正确格式", test_returns_correct_format)
    run_test(result, "动作值在 [0,1] 范围", test_values_in_range)
    run_test(result, "无噪音模式", test_no_noise_mode)
    run_test(result, "无噪音确定性", test_deterministic_no_noise)


def test_store_experience(result: TestResult):
    """测试 store_experience 方法"""
    print("\n[3] 测试 store_experience 方法")

    def test_increases_buffer():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            beliefs = create_beliefs()

            # 获取初始 buffer 大小
            if hasattr(algo, '_replay_buffer'):
                initial_size = len(algo._replay_buffer)
            else:
                initial_size = len(list(algo._replay_buffers.values())[0])

            # 存储经验
            pure_actions, noisy_actions = algo.take_action(beliefs)
            rewards = {name: np.random.rand() for name in AGENT_NAMES}
            next_beliefs = np.random.rand(N_AGENTS, N_PERIODS).astype(np.float32)
            algo.store_experience(beliefs, noisy_actions, rewards, next_beliefs)

            # 检查 buffer 增长
            if hasattr(algo, '_replay_buffer'):
                assert len(algo._replay_buffer) == initial_size + 1
            else:
                for buffer in algo._replay_buffers.values():
                    assert len(buffer) == initial_size + 1

    run_test(result, "存储经验增加 buffer", test_increases_buffer)


def test_learn(result: TestResult):
    """测试 learn 方法"""
    print("\n[4] 测试 learn 方法")

    def test_returns_none_when_empty():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            result_learn = algo.learn()
            assert result_learn is None, f"{cls.__name__}: buffer 为空时应返回 None"

    def test_returns_metrics_after_experience():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)

            # 存储足够经验
            for _ in range(20):
                beliefs = np.random.rand(N_AGENTS, N_PERIODS).astype(np.float32)
                pure_actions, noisy_actions = algo.take_action(beliefs)
                rewards = {name: np.random.rand() * 100 for name in AGENT_NAMES}
                next_beliefs = np.random.rand(N_AGENTS, N_PERIODS).astype(np.float32)
                algo.store_experience(beliefs, noisy_actions, rewards, next_beliefs)

            # 学习
            metrics = algo.learn()

            assert metrics is not None, f"{cls.__name__}: 有经验后应返回指标"
            assert "agents" in metrics
            assert len(metrics["agents"]) > 0

            for agent_name, agent_metrics in metrics["agents"].items():
                assert "critic_loss" in agent_metrics
                assert "actor_loss" in agent_metrics
                assert "noise_sigma" in agent_metrics

    run_test(result, "buffer 为空时返回 None", test_returns_none_when_empty)
    run_test(result, "有经验后返回学习指标", test_returns_metrics_after_experience)


def test_nashconv_support(result: TestResult):
    """测试 NashConv 支持方法"""
    print("\n[5] 测试 NashConv 支持方法")

    def test_get_critics():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            critics = algo.get_critics()

            assert isinstance(critics, dict)
            assert set(critics.keys()) == set(AGENT_NAMES)

            for critic in critics.values():
                assert isinstance(critic, torch.nn.Module)

    def test_build_critic_input():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            all_actions = {name: np.random.rand(N_PERIODS).astype(np.float32) for name in AGENT_NAMES}

            for agent_name in AGENT_NAMES:
                critic_input = algo.build_critic_input(beliefs, agent_name, all_actions)

                assert isinstance(critic_input, torch.Tensor)
                assert critic_input.dim() == 2
                assert critic_input.shape[0] == 1

    def test_critic_forward():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            critics = algo.get_critics()
            all_actions = {name: np.random.rand(N_PERIODS).astype(np.float32) for name in AGENT_NAMES}

            for agent_name in AGENT_NAMES:
                critic = critics[agent_name]
                critic_input = algo.build_critic_input(beliefs, agent_name, all_actions)

                with torch.no_grad():
                    q_value = critic(critic_input)

                assert q_value.shape == (1, 1)

    run_test(result, "get_critics 返回 nn.Module", test_get_critics)
    run_test(result, "build_critic_input 返回 tensor", test_build_critic_input)
    run_test(result, "critic 前向传播", test_critic_forward)


def test_reset_noise(result: TestResult):
    """测试 reset_noise 方法"""
    print("\n[6] 测试 reset_noise 方法")

    def test_restores_sigma():
        beliefs = create_beliefs()
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)

            # 获取初始 sigma
            initial_sigmas = {
                name: agent.noise.initial_sigma
                for name, agent in algo._agents.items()
            }

            # 多次调用动作让噪音衰减
            for _ in range(100):
                algo.take_action(beliefs, add_noise=True)

            # 检查噪音已衰减
            for name, agent in algo._agents.items():
                assert agent.noise.sigma < initial_sigmas[name], "噪音应已衰减"

            # 重置噪音
            algo.reset_noise()

            # 检查噪音恢复
            for name, agent in algo._agents.items():
                assert agent.noise.sigma == initial_sigmas[name], "噪音应恢复初始值"

    run_test(result, "reset_noise 恢复初始 sigma", test_restores_sigma)


def test_gamma_zero(result: TestResult):
    """测试 γ=0 静态博弈设置"""
    print("\n[7] 测试 γ=0 静态博弈设置")

    def test_default_gamma():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
            assert algo._gamma == 0.0, f"{cls.__name__}: 默认 gamma 应为 0"

    def test_custom_gamma():
        for cls in [MADDPGv1, IDDPGv1, MFDDPGv1]:
            algo = cls(agent_names=AGENT_NAMES, n_periods=N_PERIODS, gamma=0.95, seed=SEED)
            assert algo._gamma == 0.95

    run_test(result, "默认 gamma=0", test_default_gamma)
    run_test(result, "可自定义 gamma", test_custom_gamma)


def test_critic_input_dimensions(result: TestResult):
    """测试 Critic 输入维度"""
    print("\n[8] 测试 Critic 输入维度")

    def test_maddpg_dim():
        algo = MADDPGv1(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
        # MADDPG: beliefs + 所有动作
        expected = N_AGENTS * N_PERIODS + N_AGENTS * N_PERIODS
        assert algo._critic_input_dim == expected, f"MADDPG: 期望 {expected}, 实际 {algo._critic_input_dim}"

    def test_iddpg_dim():
        algo = IDDPGv1(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
        # IDDPG: beliefs + 自身动作
        expected = N_AGENTS * N_PERIODS + N_PERIODS
        assert algo._critic_input_dim == expected, f"IDDPG: 期望 {expected}, 实际 {algo._critic_input_dim}"

    def test_mfddpg_dim():
        algo = MFDDPGv1(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)
        # MFDDPG: MF 状态 (2 * n_periods) + 自身动作
        expected = 2 * N_PERIODS + N_PERIODS
        assert algo._critic_input_dim == expected, f"MFDDPG: 期望 {expected}, 实际 {algo._critic_input_dim}"

    run_test(result, "MADDPG critic 维度", test_maddpg_dim)
    run_test(result, "IDDPG critic 维度", test_iddpg_dim)
    run_test(result, "MFDDPG critic 维度", test_mfddpg_dim)


def test_integration_with_game_history(result: TestResult):
    """测试与 GameHistory 的集成"""
    print("\n[9] 测试与 GameHistory 集成")

    def test_full_loop():
        algo = MADDPGv1(agent_names=AGENT_NAMES, n_periods=N_PERIODS, seed=SEED)

        # 创建 GameHistory
        history = GameHistory(
            agent_names=AGENT_NAMES,
            n_periods=N_PERIODS,
            ema_lambda=0.05,
        )

        # 模拟博弈循环
        for step in range(10):
            # 获取当前信念
            beliefs = history.get_beliefs()

            # 算法选择动作
            pure_actions, noisy_actions = algo.take_action(beliefs)

            # 模拟环境返回
            rewards = {name: np.random.rand() * 100 for name in AGENT_NAMES}
            flows = {name: np.random.rand(N_PERIODS) * 50 for name in AGENT_NAMES}

            # 更新信念
            history.update_belief(pure_actions)

            # 获取新信念
            next_beliefs = history.get_beliefs()

            # 记录到历史
            history.record(
                pure_actions=pure_actions,
                noisy_actions=noisy_actions,
                rewards=rewards,
                flows=flows,
            )

            # 存储经验
            algo.store_experience(beliefs, noisy_actions, rewards, next_beliefs)

        # 学习
        metrics = algo.learn()
        assert metrics is not None, "学习后应返回指标"

        # 检查历史记录
        assert history.total_evaluations == 10, "应记录 10 次评估"

    run_test(result, "完整博弈循环", test_full_loop)


def main():
    print("=" * 60)
    print("算法 v1 版本测试")
    print("=" * 60)

    result = TestResult()

    test_algorithm_base(result)
    test_take_action(result)
    test_store_experience(result)
    test_learn(result)
    test_nashconv_support(result)
    test_reset_noise(result)
    test_gamma_zero(result)
    test_critic_input_dimensions(result)
    test_integration_with_game_history(result)

    success = result.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
