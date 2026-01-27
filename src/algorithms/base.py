"""
Algorithm 统一接口定义

为 MADDPG / IDDPG / MFDDPG 提供统一的抽象基类。
支持 Fictitious Play 风格的博弈求解和 NashConv 收敛检测。

设计要点：
1. 观测格式：接收 beliefs 矩阵 (n_agents, n_periods)，算法内部负责格式转换
2. 动作输出：同时返回纯策略和噪声策略
3. NashConv 支持：提供 get_critics() 和 build_critic_input() 用于收敛检测
4. γ=0 静态博弈：Critic 直接拟合即时收益

参数来源：
- agent_names, n_periods 等参数由 GameTrainer 从 NetworkData 获取后传入
- Algorithm 不直接依赖数据集配置
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class AlgorithmBase(ABC):
    """
    算法抽象基类

    所有算法实现必须继承此基类，确保与 GameTrainer 和 NashConvChecker 兼容。
    参数由 GameTrainer 从 NetworkData 获取后传入。
    """

    def __init__(
        self,
        agent_names: list[str],
        n_periods: int,
        device: str = "cpu",
        noise_sigma: float = 0.2,
        noise_decay: float = 0.9995,
        min_noise: float = 0.01,
    ):
        """
        初始化算法基类

        Args:
            agent_names: Agent 名称列表（由 GameTrainer 从 NetworkData 传入）
            n_periods: 时段数量（由 GameTrainer 从 NetworkData 传入）
            device: 计算设备
            noise_sigma: 初始噪音标准差
            noise_decay: 噪音衰减率
            min_noise: 最小噪音标准差
        """
        self._agent_names = list(agent_names)
        self._n_agents = len(agent_names)
        self._n_periods = n_periods
        self._device = device

        # 噪音参数（用于 reset_noise）
        self._initial_noise_sigma = noise_sigma
        self._noise_decay = noise_decay
        self._min_noise = min_noise

        # 建立 agent_name -> index 的映射
        self._agent_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._agent_names)
        }

    @property
    def agent_names(self) -> list[str]:
        """Agent 名称列表"""
        return self._agent_names

    @property
    def n_periods(self) -> int:
        """时段数量（动作维度）"""
        return self._n_periods

    @property
    @abstractmethod
    def name(self) -> str:
        """
        算法名称标识

        Returns:
            str: "MADDPG" / "IDDPG" / "MFDDPG"
        """
        ...

    @abstractmethod
    def take_action(
        self,
        beliefs: np.ndarray,
        add_noise: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        所有 Agent 选择动作

        Args:
            beliefs: EMA 信念矩阵 (n_agents, n_periods)，包含自身
            add_noise: 是否添加探索噪音

        Returns:
            tuple: (pure_actions, noisy_actions)
                - pure_actions: 纯策略输出 {agent_name: (n_periods,)}
                - noisy_actions: 噪声策略 {agent_name: (n_periods,)}
        """
        ...

    @abstractmethod
    def store_experience(
        self,
        beliefs: np.ndarray,
        noisy_actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        next_beliefs: np.ndarray,
    ) -> None:
        """
        存入 ReplayBuffer

        Args:
            beliefs: 当前信念矩阵 (n_agents, n_periods)
            noisy_actions: 噪声动作（实际执行的）
            rewards: 收益 {agent_name: float}
            next_beliefs: 下一时刻信念矩阵 (n_agents, n_periods)

        Note:
            算法内部负责：
            1. 观测格式转换（MADDPG/IDDPG/MFDDPG 各自不同）
            2. 奖励归一化
            3. 存入各自的 ReplayBuffer
        """
        ...

    @abstractmethod
    def learn(self) -> dict | None:
        """
        从 ReplayBuffer 采样并更新网络

        Returns:
            dict | None: 学习指标字典（包含各 agent 的 loss、梯度范数等），
                         如果经验不足则返回 None
        """
        ...

    # === NashConv 计算所需 ===

    @abstractmethod
    def get_critics(self) -> dict[str, nn.Module]:
        """
        返回各 Agent 的 Critic 网络

        用于 NashConv 计算（多起点梯度上升寻找最佳响应）。

        Returns:
            dict[str, nn.Module]: {agent_name: critic_network}
        """
        ...

    @abstractmethod
    def build_critic_input(
        self,
        beliefs: np.ndarray,
        agent_name: str,
        all_actions: dict[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """
        为指定 Agent 构造 Critic 输入

        各算法自己负责格式转换：
        - MADDPG: 全局状态 + 所有动作
        - IDDPG: 局部状态 + 自身动作
        - MFDDPG: MF 状态 + 自身动作

        Args:
            beliefs: 信念矩阵 (n_agents, n_periods)
            agent_name: 目标 Agent 名称
            all_actions: 所有 Agent 的动作 {agent_name: (n_periods,)}
                         支持 np.ndarray 或 torch.Tensor（用于 NashConv 梯度计算）

        Returns:
            torch.Tensor: Critic 网络输入，shape=(1, critic_input_dim)

        Note:
            如果 all_actions 中包含 Tensor，会保持其梯度链路，
            用于 NashConv 计算时对动作进行梯度上升优化。
        """
        ...

    @abstractmethod
    def reset_noise(self) -> None:
        """
        重置探索噪音到初始值

        用途：假收敛时强制重新探索
        """
        ...

    # === 通用工具方法 ===

    def _get_agent_index(self, agent_name: str) -> int:
        """获取 Agent 在信念矩阵中的索引"""
        return self._agent_to_idx[agent_name]

    def _normalize_rewards(
        self, rewards: dict[str, float], reward_scale: float | None = None
    ) -> dict[str, float]:
        """
        奖励归一化

        Args:
            rewards: 原始奖励
            reward_scale: 固定缩放因子，None 则使用动态缩放

        Returns:
            归一化后的奖励
        """
        if reward_scale is not None:
            scale = reward_scale
        else:
            scale = max(rewards.values()) if rewards.values() else 1.0

        if scale > 0:
            return {agent: reward / scale for agent, reward in rewards.items()}
        else:
            return {agent: 0.0 for agent in rewards.keys()}
