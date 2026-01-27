"""
NashConvChecker 单元测试
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.algorithms.maddpg.maddpg_v1 import MADDPGv1
from src.algorithms.iddpg.iddpg_v1 import IDDPGv1
from src.algorithms.mfddpg.mfddpg_v1 import MFDDPGv1
from src.game.nashconv import NashConvChecker


def test_compute_returns_correct_structure():
    """测试 compute 返回正确的数据结构"""
    print("测试: compute 返回正确的数据结构...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = MADDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)
    checker = NashConvChecker(n_starts=3, optim_steps=20, lr=0.01)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)
    current_actions = {
        name: np.random.rand(n_periods).astype(np.float32)
        for name in agent_names
    }

    nashconv, exploitability, regrets = checker.compute(
        algorithm, beliefs, current_actions
    )

    # 验证返回类型
    assert isinstance(nashconv, float), "nashconv 应为 float"
    assert isinstance(exploitability, float), "exploitability 应为 float"
    assert isinstance(regrets, dict), "regrets 应为 dict"

    # 验证 regrets 包含所有 agent
    assert set(regrets.keys()) == set(agent_names), "regrets 应包含所有 agent"

    # 验证非负性
    assert nashconv >= 0, "nashconv 应非负"
    assert exploitability >= 0, "exploitability 应非负"
    for name, regret in regrets.items():
        assert regret >= 0, f"{name} 的 regret 应非负"

    # 验证 exploitability = nashconv / n_agents
    assert abs(exploitability - nashconv / n_agents) < 1e-6, \
        "exploitability 应等于 nashconv / n_agents"

    print(f"  nashconv = {nashconv:.4f}")
    print(f"  exploitability = {exploitability:.4f}")
    print(f"  regrets = {regrets}")
    print("  PASSED\n")


def test_compute_with_iddpg():
    """测试 IDDPG 算法的 NashConv 计算"""
    print("测试: IDDPG 算法的 NashConv 计算...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = IDDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)
    checker = NashConvChecker(n_starts=3, optim_steps=20, lr=0.01)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)
    current_actions = {
        name: np.random.rand(n_periods).astype(np.float32)
        for name in agent_names
    }

    nashconv, exploitability, regrets = checker.compute(
        algorithm, beliefs, current_actions
    )

    assert nashconv >= 0, "nashconv 应非负"
    assert exploitability >= 0, "exploitability 应非负"
    assert len(regrets) == n_agents, "regrets 长度应等于 n_agents"

    print(f"  nashconv = {nashconv:.4f}")
    print(f"  exploitability = {exploitability:.4f}")
    print("  PASSED\n")


def test_compute_with_mfddpg():
    """测试 MFDDPG 算法的 NashConv 计算"""
    print("测试: MFDDPG 算法的 NashConv 计算...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = MFDDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)
    checker = NashConvChecker(n_starts=3, optim_steps=20, lr=0.01)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)
    current_actions = {
        name: np.random.rand(n_periods).astype(np.float32)
        for name in agent_names
    }

    nashconv, exploitability, regrets = checker.compute(
        algorithm, beliefs, current_actions
    )

    assert nashconv >= 0, "nashconv 应非负"
    assert exploitability >= 0, "exploitability 应非负"
    assert len(regrets) == n_agents, "regrets 长度应等于 n_agents"

    print(f"  nashconv = {nashconv:.4f}")
    print(f"  exploitability = {exploitability:.4f}")
    print("  PASSED\n")


def test_best_response_improves_q():
    """测试最佳响应确实能提高 Q 值"""
    print("测试: 最佳响应能提高 Q 值...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = MADDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)
    checker = NashConvChecker(n_starts=3, optim_steps=20, lr=0.01)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)
    current_actions = {
        name: np.random.rand(n_periods).astype(np.float32)
        for name in agent_names
    }

    critics = algorithm.get_critics()

    for agent_name in agent_names:
        critic = critics[agent_name]

        # 计算当前 Q 值
        current_q = checker._compute_q_value(
            algorithm, critic, beliefs, agent_name, current_actions
        )

        # 找最佳响应
        best_q = checker._find_best_response(
            algorithm, critic, beliefs, agent_name, current_actions
        )

        # 最佳响应的 Q 值应该 >= 当前 Q 值
        assert best_q >= current_q - 1e-6, \
            f"{agent_name}: best_q ({best_q:.4f}) 应 >= current_q ({current_q:.4f})"

        print(f"  {agent_name}: current_q = {current_q:.4f}, best_q = {best_q:.4f}")

    print("  PASSED\n")


def test_gradient_flow_in_build_critic_input():
    """测试 build_critic_input 支持梯度流"""
    print("测试: build_critic_input 支持梯度流...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = MADDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)

    # 创建需要梯度的动作
    action_with_grad = torch.rand(n_periods, requires_grad=True)

    # 混合动作字典
    all_actions = {
        name: (action_with_grad if name == agent_names[0] else np.random.rand(n_periods).astype(np.float32))
        for name in agent_names
    }

    # 构建 Critic 输入
    critic_input = algorithm.build_critic_input(beliefs, agent_names[0], all_actions)

    # 验证输入是 Tensor
    assert isinstance(critic_input, torch.Tensor), "critic_input 应为 Tensor"

    # 通过 Critic 计算 Q 值并反向传播
    critic = algorithm.get_critics()[agent_names[0]]
    q_value = critic(critic_input)
    q_value.backward()

    # 验证梯度存在
    assert action_with_grad.grad is not None, "动作应有梯度"
    assert action_with_grad.grad.shape == (n_periods,), "梯度形状应正确"

    print(f"  action_with_grad.grad = {action_with_grad.grad}")
    print("  PASSED\n")


def test_gradient_flow_iddpg():
    """测试 IDDPG 的 build_critic_input 支持梯度流"""
    print("测试: IDDPG build_critic_input 支持梯度流...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = IDDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)

    # 创建需要梯度的动作
    action_with_grad = torch.rand(n_periods, requires_grad=True)

    # 混合动作字典
    all_actions = {
        name: (action_with_grad if name == agent_names[0] else np.random.rand(n_periods).astype(np.float32))
        for name in agent_names
    }

    # 构建 Critic 输入
    critic_input = algorithm.build_critic_input(beliefs, agent_names[0], all_actions)

    # 通过 Critic 计算 Q 值并反向传播
    critic = algorithm.get_critics()[agent_names[0]]
    q_value = critic(critic_input)
    q_value.backward()

    # 验证梯度存在
    assert action_with_grad.grad is not None, "动作应有梯度"

    print(f"  action_with_grad.grad = {action_with_grad.grad}")
    print("  PASSED\n")


def test_gradient_flow_mfddpg():
    """测试 MFDDPG 的 build_critic_input 支持梯度流"""
    print("测试: MFDDPG build_critic_input 支持梯度流...")

    agent_names = ["agent_0", "agent_1", "agent_2"]
    n_periods = 4
    n_agents = len(agent_names)

    algorithm = MFDDPGv1(agent_names=agent_names, n_periods=n_periods, seed=42)

    beliefs = np.random.rand(n_agents, n_periods).astype(np.float32)

    # 创建需要梯度的动作
    action_with_grad = torch.rand(n_periods, requires_grad=True)

    # 混合动作字典
    all_actions = {
        name: (action_with_grad if name == agent_names[0] else np.random.rand(n_periods).astype(np.float32))
        for name in agent_names
    }

    # 构建 Critic 输入
    critic_input = algorithm.build_critic_input(beliefs, agent_names[0], all_actions)

    # 通过 Critic 计算 Q 值并反向传播
    critic = algorithm.get_critics()[agent_names[0]]
    q_value = critic(critic_input)
    q_value.backward()

    # 验证梯度存在
    assert action_with_grad.grad is not None, "动作应有梯度"

    print(f"  action_with_grad.grad = {action_with_grad.grad}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("NashConvChecker 单元测试")
    print("=" * 60 + "\n")

    test_compute_returns_correct_structure()
    test_compute_with_iddpg()
    test_compute_with_mfddpg()
    test_best_response_improves_q()
    test_gradient_flow_in_build_critic_input()
    test_gradient_flow_iddpg()
    test_gradient_flow_mfddpg()

    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)
