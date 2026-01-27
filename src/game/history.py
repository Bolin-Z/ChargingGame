"""
GameHistory: 博弈历史管理器

职责：
1. 维护 EMA 信念矩阵（所有 Agent 的价格均值估计）
2. 记录评估历史
3. 提供信念查询接口

设计说明：
- 信念矩阵是全局共享的，所有 Agent 观察相同的历史
- 各算法自己决定如何使用信念矩阵组织观测
- γ=0 静态博弈下不需要 policy_version
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationRecord:
    """单次评估记录"""

    eval_id: int  # 全局递增计数

    # 动作
    pure_actions: dict[str, np.ndarray]  # 纯策略动作 {agent_name: (n_periods,)}
    noisy_actions: dict[str, np.ndarray]  # 噪声动作（实际执行）

    # 环境返回
    rewards: dict[str, float]  # 各充电站收益
    flows: dict[str, np.ndarray]  # 各充电站各时段流量 (n_periods,)
    ue_info: dict = field(default_factory=dict)  # UE 统计信息


class GameHistory:
    """
    博弈历史管理器

    维护 Fictitious Play 风格的 EMA 信念更新机制。
    信念矩阵表示对所有 Agent 历史价格的指数移动平均估计。
    """

    def __init__(
        self,
        agent_names: list[str],
        n_periods: int,
        ema_lambda: float = 0.05,
        initial_belief: float = 0.5,
    ):
        """
        初始化 GameHistory

        Args:
            agent_names: Agent 名称列表（决定顺序）
            n_periods: 时段数量
            ema_lambda: EMA 更新系数，默认 0.05（关注最近 ~20 轮）
            initial_belief: 信念初始值，默认 0.5（价格空间中点）
        """
        self.agent_names = list(agent_names)
        self.n_agents = len(agent_names)
        self.n_periods = n_periods
        self.ema_lambda = ema_lambda

        # 建立 agent_name -> index 的映射
        self._agent_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.agent_names)
        }

        # EMA 信念矩阵 (n_agents, n_periods)，初始值为价格空间中点
        self.beliefs: np.ndarray = np.full(
            (self.n_agents, n_periods), initial_belief, dtype=np.float32
        )

        # 评估记录
        self.records: list[EvaluationRecord] = []

    def update_belief(self, pure_actions: dict[str, np.ndarray]) -> None:
        """
        用纯策略更新 EMA 信念

        EMA 公式：belief_new = (1 - λ) * belief_old + λ * action_pure

        Args:
            pure_actions: 各 Agent 的纯策略动作 {agent_name: (n_periods,)}
        """
        for agent_name, action in pure_actions.items():
            idx = self._agent_to_idx[agent_name]
            self.beliefs[idx] = (
                (1 - self.ema_lambda) * self.beliefs[idx]
                + self.ema_lambda * np.asarray(action, dtype=np.float32)
            )

    def get_beliefs(self) -> np.ndarray:
        """
        返回当前信念矩阵

        Returns:
            beliefs: (n_agents, n_periods) 的信念矩阵副本
        """
        return self.beliefs.copy()

    def record(
        self,
        pure_actions: dict[str, np.ndarray],
        noisy_actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        flows: dict[str, np.ndarray],
        ue_info: dict[str, Any] | None = None,
    ) -> EvaluationRecord:
        """
        记录单次评估结果

        Args:
            pure_actions: 纯策略动作
            noisy_actions: 噪声动作（实际执行）
            rewards: 各充电站收益
            flows: 各充电站各时段流量
            ue_info: UE 统计信息（可选）

        Returns:
            创建的 EvaluationRecord
        """
        record = EvaluationRecord(
            eval_id=len(self.records),
            pure_actions=pure_actions,
            noisy_actions=noisy_actions,
            rewards=rewards,
            flows=flows,
            ue_info=ue_info or {},
        )
        self.records.append(record)
        return record

    def get_agent_index(self, agent_name: str) -> int:
        """获取 Agent 在信念矩阵中的索引"""
        return self._agent_to_idx[agent_name]

    def get_agent_belief(self, agent_name: str) -> np.ndarray:
        """
        获取单个 Agent 的信念向量

        Args:
            agent_name: Agent 名称

        Returns:
            (n_periods,) 的信念向量
        """
        idx = self._agent_to_idx[agent_name]
        return self.beliefs[idx].copy()

    def reset(self, initial_belief: float = 0.5) -> None:
        """
        重置历史（新的博弈求解尝试）

        Args:
            initial_belief: 重置后的信念初始值
        """
        self.beliefs.fill(initial_belief)
        self.records.clear()

    @property
    def total_evaluations(self) -> int:
        """总评估次数"""
        return len(self.records)

    def get_last_record(self) -> EvaluationRecord | None:
        """获取最后一条评估记录"""
        return self.records[-1] if self.records else None

    def get_recent_records(self, n: int) -> list[EvaluationRecord]:
        """获取最近 n 条评估记录"""
        return self.records[-n:] if n > 0 else []

    def __repr__(self) -> str:
        return (
            f"GameHistory(agents={self.agent_names}, "
            f"n_periods={self.n_periods}, "
            f"total_evaluations={self.total_evaluations})"
        )
