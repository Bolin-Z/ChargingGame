"""
MF-DDPG核心算法实现

Mean Field Deep Deterministic Policy Gradient (MF-DDPG)
基于Mean Field近似的多智能体深度强化学习算法。

核心思想：
- 通过Mean Field近似将其他agent的集体行为压缩为低维度信息
- 完全独立训练：每个agent独立学习，无中心化协调
- 信息压缩：用平均场信息近似其他agent行为

关键特性：
- 最佳扩展性：不受agent数量影响，O(1)复杂度
- 理论简洁：符合去中心化学习的纯粹性
- 计算高效：最小的网络规模和参数量
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Any


# ============================================================================
# 工具函数：Mean Field状态计算
# ============================================================================

def compute_mean_field_state(agent_id: str, observation: Dict, all_agents: List[str]) -> np.ndarray:
    """
    计算单个agent的Mean Field状态

    将其他agent的价格信息压缩为均值，实现Mean Field近似。
    这是MF-DDPG的核心设计：用统计平均信息替代完整的全局信息。

    状态组成：
        - own_last_prices: 自身上轮价格（决策历史）
        - own_last_flow: 自身上轮充电流量（市场反馈）
        - mean_field_prices: 其他agent价格的平均值（竞争环境）

    Args:
        agent_id (str): 当前agent的ID（如"agent_0"）
        observation (dict): 环境返回的观测字典，包含:
            - "last_round_all_prices": np.ndarray, shape=(n_agents, n_periods)
            - "own_charging_flow": np.ndarray, shape=(n_periods,)
        all_agents (list): agent ID列表，用于确定索引顺序

    Returns:
        np.ndarray: Mean Field状态向量
            shape = (3 * n_periods,) 其中n_periods由observation决定

    示例:
        >>> observation = {
        ...     "last_round_all_prices": np.array([[0.5, 0.6], [0.4, 0.5], [0.6, 0.7], [0.5, 0.6]]),
        ...     "own_charging_flow": np.array([10.0, 12.0])
        ... }
        >>> all_agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        >>> state = compute_mean_field_state("agent_1", observation, all_agents)
        >>> # shape = (6,) 因为 n_periods=2, 所以 2+2+2=6
    """
    # 1. 提取全局价格矩阵
    all_prices = observation["last_round_all_prices"]  # shape: (n_agents, n_periods)

    # 2. 找到agent_id对应的索引
    agent_idx = all_agents.index(agent_id)

    # 3. 提取自身上轮价格
    own_last_prices = all_prices[agent_idx].flatten()  # shape: (n_periods,)

    # 4. 计算Mean Field状态：其他agent价格的平均值
    # 排除自己，计算其他所有agent的价格均值
    other_indices = [i for i in range(len(all_agents)) if i != agent_idx]
    mean_field_prices = np.mean(all_prices[other_indices], axis=0).flatten()  # shape: (n_periods,)

    # 5. 提取自身上轮充电流量
    own_last_flow = observation["own_charging_flow"].flatten()  # shape: (n_periods,)

    # 6. 拼接为完整的Mean Field状态
    mf_state = np.concatenate([own_last_prices, own_last_flow, mean_field_prices])

    return mf_state


def organize_mf_critic_state(actor_state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    组织MF-DDPG的Critic网络输入

    将Actor状态和当前动作拼接，形成Critic网络的输入。
    这是标准的Actor-Critic架构：Critic评估"状态-动作"对的价值。

    Args:
        actor_state (np.ndarray): Actor状态
            shape = (3 * n_periods,) 包含 [own_prices, own_flow, mean_field]
        action (np.ndarray): 当前动作（价格决策）
            shape = (n_periods,)

    Returns:
        np.ndarray: Critic输入向量
            shape = (4 * n_periods,) = actor_state + action

    示例:
        >>> actor_state = np.random.rand(24)  # 假设n_periods=8
        >>> action = np.random.rand(8)
        >>> critic_state = organize_mf_critic_state(actor_state, action)
        >>> critic_state.shape
        (32,)
    """
    # 简单拼接：[actor_state, action]
    critic_state = np.concatenate([actor_state.flatten(), action.flatten()])

    return critic_state


# ============================================================================
# 工具类：经验回放和探索策略
# ============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区

    用于存储智能体的经验轨迹，支持随机采样以打破数据相关性。
    使用循环缓冲区，当容量满时自动覆盖最旧的经验。
    与MADDPG/IDDPG完全相同，确保公平对比。
    """

    def __init__(self, capacity):
        """
        初始化经验回放缓冲区

        Args:
            capacity (int): 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """
        添加经验元组到缓冲区

        Args:
            experience (tuple): 经验元组 (mf_state, action, reward, next_mf_state, done)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        随机采样批次经验

        Args:
            batch_size (int): 采样批次大小

        Returns:
            list: 采样得到的经验列表

        Raises:
            ValueError: 当缓冲区大小小于批次大小时
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小{len(self.buffer)} < 批次大小{batch_size}")
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)


class GaussianNoise:
    """
    高斯噪音探索策略

    为动作添加高斯噪音以促进探索，噪音强度随时间指数衰减。
    适用于连续动作空间的探索。
    与MADDPG/IDDPG完全相同，确保公平对比。
    """

    def __init__(self, action_dim, sigma, sigma_decay, min_sigma):
        """
        初始化高斯噪音

        Args:
            action_dim (int): 动作维度
            sigma (float): 初始噪音标准差
            sigma_decay (float): 噪音衰减率，每次调用后sigma *= sigma_decay
            min_sigma (float): 噪音的最小标准差，防止探索完全停止
        """
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma

    def __call__(self, action):
        """
        为动作添加高斯噪音

        Args:
            action (np.ndarray): 原始动作，形状为 (action_dim,)

        Returns:
            np.ndarray: 添加噪音后的动作，裁剪到[0,1]范围
        """
        noise = np.random.normal(0, self.sigma, self.action_dim)
        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)
        return np.clip(action + noise, 0.0, 1.0)


# ============================================================================
# MFDDPGAgent：单智能体MF-DDPG实现
# ============================================================================
