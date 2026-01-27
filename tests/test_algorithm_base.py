# tests/test_algorithm_base.py
"""
AlgorithmBase 抽象基类测试脚本

验证内容：
1. AlgorithmBase 初始化
2. Agent 索引映射
3. 奖励归一化
4. 抽象方法签名
"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_algorithm_base_init():
    """测试 AlgorithmBase 初始化"""
    from src.algorithms.base import AlgorithmBase

    print("=" * 60)
    print("测试 1: AlgorithmBase 初始化")
    print("=" * 60)

    # 创建一个具体子类用于测试
    class DummyAlgorithm(AlgorithmBase):
        @property
        def name(self) -> str:
            return "Dummy"

        def take_action(self, beliefs, add_noise=True):
            pure = {n: np.zeros(self._n_periods) for n in self._agent_names}
            noisy = {n: np.zeros(self._n_periods) for n in self._agent_names}
            return pure, noisy

        def store_experience(self, beliefs, noisy_actions, rewards, next_beliefs):
            pass

        def learn(self):
            return None

        def get_critics(self):
            return {}

        def build_critic_input(self, beliefs, agent_name, all_actions):
            return torch.zeros(1, 10)

        def reset_noise(self):
            pass

    agent_names = ["station_5", "station_12", "station_17"]
    n_periods = 8

    algo = DummyAlgorithm(agent_names, n_periods)

    assert algo.agent_names == agent_names, "agent_names 不匹配"
    assert algo.n_periods == n_periods, "n_periods 不匹配"
    assert algo._n_agents == 3, "_n_agents 不正确"
    assert algo._device == "cpu", "_device 默认应为 cpu"

    print(f"  agent_names: {algo.agent_names}")
    print(f"  n_periods: {algo.n_periods}")
    print(f"  name: {algo.name}")
    print(f"  _n_agents: {algo._n_agents}")
    print(f"  _device: {algo._device}")

    print("\n✓ 测试 1 通过")


def test_agent_index_mapping():
    """测试 Agent 索引映射"""
    from src.algorithms.base import AlgorithmBase

    print("\n" + "=" * 60)
    print("测试 2: Agent 索引映射")
    print("=" * 60)

    class DummyAlgorithm(AlgorithmBase):
        @property
        def name(self):
            return "Dummy"

        def take_action(self, beliefs, add_noise=True):
            return {}, {}

        def store_experience(self, beliefs, noisy_actions, rewards, next_beliefs):
            pass

        def learn(self):
            return None

        def get_critics(self):
            return {}

        def build_critic_input(self, beliefs, agent_name, all_actions):
            return torch.zeros(1, 10)

        def reset_noise(self):
            pass

    agent_names = ["A", "B", "C"]
    algo = DummyAlgorithm(agent_names, n_periods=4)

    assert algo._get_agent_index("A") == 0
    assert algo._get_agent_index("B") == 1
    assert algo._get_agent_index("C") == 2

    print(f"  A -> {algo._get_agent_index('A')}")
    print(f"  B -> {algo._get_agent_index('B')}")
    print(f"  C -> {algo._get_agent_index('C')}")

    print("\n✓ 测试 2 通过")


def test_normalize_rewards():
    """测试奖励归一化"""
    from src.algorithms.base import AlgorithmBase

    print("\n" + "=" * 60)
    print("测试 3: 奖励归一化")
    print("=" * 60)

    class DummyAlgorithm(AlgorithmBase):
        @property
        def name(self):
            return "Dummy"

        def take_action(self, beliefs, add_noise=True):
            return {}, {}

        def store_experience(self, beliefs, noisy_actions, rewards, next_beliefs):
            pass

        def learn(self):
            return None

        def get_critics(self):
            return {}

        def build_critic_input(self, beliefs, agent_name, all_actions):
            return torch.zeros(1, 10)

        def reset_noise(self):
            pass

    algo = DummyAlgorithm(["A", "B"], n_periods=4)

    # 测试动态缩放
    rewards = {"A": 100.0, "B": 200.0}
    normalized = algo._normalize_rewards(rewards)
    assert normalized["A"] == 0.5, f"动态缩放 A 应为 0.5, 实际 {normalized['A']}"
    assert normalized["B"] == 1.0, f"动态缩放 B 应为 1.0, 实际 {normalized['B']}"
    print(f"  动态缩放: {rewards} -> {normalized}")

    # 测试固定缩放
    normalized_fixed = algo._normalize_rewards(rewards, reward_scale=400.0)
    assert normalized_fixed["A"] == 0.25
    assert normalized_fixed["B"] == 0.5
    print(f"  固定缩放 (scale=400): {rewards} -> {normalized_fixed}")

    # 测试零奖励
    zero_rewards = {"A": 0.0, "B": 0.0}
    normalized_zero = algo._normalize_rewards(zero_rewards)
    assert normalized_zero["A"] == 0.0
    assert normalized_zero["B"] == 0.0
    print(f"  零奖励: {zero_rewards} -> {normalized_zero}")

    print("\n✓ 测试 3 通过")


def test_take_action_signature():
    """测试 take_action 方法签名"""
    from src.algorithms.base import AlgorithmBase

    print("\n" + "=" * 60)
    print("测试 4: take_action 方法签名")
    print("=" * 60)

    class TestAlgorithm(AlgorithmBase):
        @property
        def name(self):
            return "Test"

        def take_action(self, beliefs, add_noise=True):
            # 验证输入
            assert beliefs.shape == (
                self._n_agents,
                self._n_periods,
            ), f"beliefs 形状应为 ({self._n_agents}, {self._n_periods})"

            # 返回 (pure_actions, noisy_actions)
            pure = {n: beliefs[i].copy() for i, n in enumerate(self._agent_names)}
            noisy = {n: beliefs[i].copy() + 0.1 for i, n in enumerate(self._agent_names)}
            return pure, noisy

        def store_experience(self, beliefs, noisy_actions, rewards, next_beliefs):
            pass

        def learn(self):
            return None

        def get_critics(self):
            return {}

        def build_critic_input(self, beliefs, agent_name, all_actions):
            return torch.zeros(1, 10)

        def reset_noise(self):
            pass

    agent_names = ["A", "B"]
    n_periods = 4
    algo = TestAlgorithm(agent_names, n_periods)

    beliefs = np.array([[0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7]], dtype=np.float32)

    pure, noisy = algo.take_action(beliefs, add_noise=True)

    # 验证返回格式
    assert isinstance(pure, dict), "pure_actions 应为 dict"
    assert isinstance(noisy, dict), "noisy_actions 应为 dict"
    assert set(pure.keys()) == set(agent_names), "pure_actions 应包含所有 agent"
    assert set(noisy.keys()) == set(agent_names), "noisy_actions 应包含所有 agent"

    for name in agent_names:
        assert pure[name].shape == (n_periods,), f"{name} pure 形状应为 ({n_periods},)"
        assert noisy[name].shape == (n_periods,), f"{name} noisy 形状应为 ({n_periods},)"

    print(f"  beliefs shape: {beliefs.shape}")
    print(f"  pure_actions keys: {list(pure.keys())}")
    print(f"  noisy_actions keys: {list(noisy.keys())}")
    print(f"  pure['A'] shape: {pure['A'].shape}")

    print("\n✓ 测试 4 通过")


def test_build_critic_input_signature():
    """测试 build_critic_input 方法签名"""
    from src.algorithms.base import AlgorithmBase

    print("\n" + "=" * 60)
    print("测试 5: build_critic_input 方法签名")
    print("=" * 60)

    class TestAlgorithm(AlgorithmBase):
        @property
        def name(self):
            return "Test"

        def take_action(self, beliefs, add_noise=True):
            return {}, {}

        def store_experience(self, beliefs, noisy_actions, rewards, next_beliefs):
            pass

        def learn(self):
            return None

        def get_critics(self):
            return {}

        def build_critic_input(self, beliefs, agent_name, all_actions):
            # 验证输入
            assert beliefs.shape == (self._n_agents, self._n_periods)
            assert agent_name in self._agent_names
            assert set(all_actions.keys()) == set(self._agent_names)

            # 返回 (1, critic_input_dim) 形状的 tensor
            critic_input_dim = self._n_agents * self._n_periods + self._n_agents * self._n_periods
            return torch.zeros(1, critic_input_dim)

        def reset_noise(self):
            pass

    agent_names = ["A", "B"]
    n_periods = 4
    algo = TestAlgorithm(agent_names, n_periods)

    beliefs = np.array([[0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
    all_actions = {"A": np.array([0.5, 0.5, 0.5, 0.5]), "B": np.array([0.4, 0.4, 0.4, 0.4])}

    critic_input = algo.build_critic_input(beliefs, "A", all_actions)

    assert isinstance(critic_input, torch.Tensor), "应返回 torch.Tensor"
    assert critic_input.dim() == 2, "应为 2D tensor"
    assert critic_input.shape[0] == 1, "batch 维度应为 1"

    print(f"  beliefs shape: {beliefs.shape}")
    print(f"  agent_name: 'A'")
    print(f"  all_actions keys: {list(all_actions.keys())}")
    print(f"  critic_input shape: {critic_input.shape}")

    print("\n✓ 测试 5 通过")


def test_noise_params_stored():
    """测试噪音参数存储（用于 reset_noise）"""
    from src.algorithms.base import AlgorithmBase

    print("\n" + "=" * 60)
    print("测试 6: 噪音参数存储")
    print("=" * 60)

    class DummyAlgorithm(AlgorithmBase):
        @property
        def name(self):
            return "Dummy"

        def take_action(self, beliefs, add_noise=True):
            return {}, {}

        def store_experience(self, beliefs, noisy_actions, rewards, next_beliefs):
            pass

        def learn(self):
            return None

        def get_critics(self):
            return {}

        def build_critic_input(self, beliefs, agent_name, all_actions):
            return torch.zeros(1, 10)

        def reset_noise(self):
            pass

    algo = DummyAlgorithm(
        ["A"], n_periods=4,
        noise_sigma=0.3,
        noise_decay=0.999,
        min_noise=0.02
    )

    assert algo._initial_noise_sigma == 0.3
    assert algo._noise_decay == 0.999
    assert algo._min_noise == 0.02

    print(f"  _initial_noise_sigma: {algo._initial_noise_sigma}")
    print(f"  _noise_decay: {algo._noise_decay}")
    print(f"  _min_noise: {algo._min_noise}")

    print("\n✓ 测试 6 通过")


def test_abstract_methods_required():
    """测试抽象方法必须实现"""
    from src.algorithms.base import AlgorithmBase

    print("\n" + "=" * 60)
    print("测试 7: 抽象方法必须实现")
    print("=" * 60)

    # 不完整的子类应该无法实例化
    class IncompleteAlgorithm(AlgorithmBase):
        @property
        def name(self):
            return "Incomplete"
        # 缺少其他抽象方法

    try:
        algo = IncompleteAlgorithm(["A"], n_periods=4)
        assert False, "不完整的子类不应该能实例化"
    except TypeError as e:
        print(f"  预期的 TypeError: {e}")
        assert "abstract" in str(e).lower()

    print("\n✓ 测试 7 通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("AlgorithmBase 测试套件")
    print("=" * 60 + "\n")

    test_algorithm_base_init()
    test_agent_index_mapping()
    test_normalize_rewards()
    test_take_action_signature()
    test_build_critic_input_signature()
    test_noise_params_stored()
    test_abstract_methods_required()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
