"""
NashConv / Exploitability 收敛检测器

通过多起点梯度上升计算每个 Agent 的最佳响应，
衡量当前策略组合距离纳什均衡的距离。

核心指标：
- NashConv = Σ Regret_i = Σ [Q_i(a_i*, a_{-i}) - Q_i(a_i, a_{-i})]
- Exploitability = NashConv / N（平均可剥削度）

设计要点：
- 独立于具体算法实现，通过 Algorithm 接口获取 Critic
- 多起点梯度上升避免局部最优
- 动作约束在 [0, 1] 范围内
"""

import numpy as np
import torch
import torch.nn as nn

from ..algorithms.base import AlgorithmBase


class NashConvChecker:
    """
    NashConv / Exploitability 收敛检测器

    通过多起点梯度上升寻找每个 Agent 的最佳响应，
    计算当前策略的 Exploitability。
    """

    def __init__(
        self,
        n_starts: int = 5,
        optim_steps: int = 50,
        lr: float = 0.01,
        action_low: float = 0.0,
        action_high: float = 1.0,
    ):
        """
        初始化 NashConvChecker

        Args:
            n_starts: 多起点数量（避免局部最优）
            optim_steps: 每个起点的梯度上升步数
            lr: 梯度上升学习率
            action_low: 动作下界
            action_high: 动作上界
        """
        self._n_starts = n_starts
        self._optim_steps = optim_steps
        self._lr = lr
        self._action_low = action_low
        self._action_high = action_high

    def compute(
        self,
        algorithm: AlgorithmBase,
        beliefs: np.ndarray,
        current_actions: dict[str, np.ndarray],
    ) -> tuple[float, float, dict[str, float]]:
        """
        计算当前策略的 NashConv 和 Exploitability

        Args:
            algorithm: 算法实例（提供 get_critics 和 build_critic_input）
            beliefs: 当前信念矩阵 (n_agents, n_periods)
            current_actions: 当前纯策略动作 {agent_name: (n_periods,)}

        Returns:
            tuple: (nashconv, exploitability, regrets)
                - nashconv: 所有 Agent 遗憾值总和
                - exploitability: 平均遗憾值（nashconv / n_agents）
                - regrets: 各 Agent 的遗憾值 {agent_name: regret}
        """
        critics = algorithm.get_critics()
        agent_names = algorithm.agent_names
        n_agents = len(agent_names)

        regrets = {}

        for agent_name in agent_names:
            critic = critics[agent_name]

            # 计算当前 Q 值
            current_q = self._compute_q_value(
                algorithm, critic, beliefs, agent_name, current_actions
            )

            # 多起点梯度上升寻找最佳响应
            best_q = self._find_best_response(
                algorithm, critic, beliefs, agent_name, current_actions
            )

            # 遗憾值 = 最佳 Q - 当前 Q（非负）
            regret = max(0.0, best_q - current_q)
            regrets[agent_name] = regret

        nashconv = sum(regrets.values())
        exploitability = nashconv / n_agents

        return nashconv, exploitability, regrets

    def _compute_q_value(
        self,
        algorithm: AlgorithmBase,
        critic: nn.Module,
        beliefs: np.ndarray,
        agent_name: str,
        all_actions: dict[str, np.ndarray],
    ) -> float:
        """计算给定状态-动作的 Q 值"""
        critic.eval()
        with torch.no_grad():
            critic_input = algorithm.build_critic_input(beliefs, agent_name, all_actions)
            q_value = critic(critic_input).item()
        return q_value

    def _find_best_response(
        self,
        algorithm: AlgorithmBase,
        critic: nn.Module,
        beliefs: np.ndarray,
        agent_name: str,
        current_actions: dict[str, np.ndarray],
    ) -> float:
        """
        多起点梯度上升寻找最佳响应

        固定其他 Agent 的动作，优化当前 Agent 的动作以最大化 Q 值。

        Returns:
            最佳响应对应的 Q 值
        """
        critic.eval()  # 不更新 Critic 参数
        action_dim = algorithm.n_periods
        device = next(critic.parameters()).device

        best_q = float("-inf")

        for start_idx in range(self._n_starts):
            # 初始化起点
            if start_idx == 0:
                # 第一个起点：使用当前动作
                init_action = current_actions[agent_name].copy()
            else:
                # 其他起点：随机初始化
                init_action = np.random.uniform(
                    self._action_low, self._action_high, size=(action_dim,)
                )

            # 创建可优化的动作张量
            action_tensor = torch.FloatTensor(init_action).to(device)
            action_tensor.requires_grad_(True)

            # 梯度上升优化
            optimizer = torch.optim.Adam([action_tensor], lr=self._lr)

            for _ in range(self._optim_steps):
                optimizer.zero_grad()

                # 构建动作字典：当前 Agent 用 Tensor，其他用 numpy
                all_actions_mixed = {
                    name: (action_tensor if name == agent_name else current_actions[name])
                    for name in current_actions
                }

                # 通过算法接口构建 Critic 输入（保持梯度链路）
                critic_input = algorithm.build_critic_input(
                    beliefs, agent_name, all_actions_mixed
                )

                # 计算 Q 值（最大化，所以取负作为 loss）
                q_value = critic(critic_input)
                loss = -q_value

                loss.backward()
                optimizer.step()

                # 投影到动作范围内
                with torch.no_grad():
                    action_tensor.clamp_(self._action_low, self._action_high)

            # 记录最优 Q 值
            with torch.no_grad():
                final_actions = {
                    name: (
                        action_tensor.cpu().numpy()
                        if name == agent_name
                        else current_actions[name]
                    )
                    for name in current_actions
                }
                final_q = self._compute_q_value(
                    algorithm, critic, beliefs, agent_name, final_actions
                )
                best_q = max(best_q, final_q)

        return best_q
