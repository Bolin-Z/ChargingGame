"""
算法公共组件

提供 MADDPG / IDDPG / MFDDPG 共用的基础组件：
- ReplayBuffer: 经验回放缓冲区
- GaussianNoise: 高斯噪音探索策略
"""

from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """
    经验回放缓冲区

    用于存储智能体的经验轨迹，支持随机采样以打破数据相关性。
    使用循环缓冲区，当容量满时自动覆盖最旧的经验。
    """

    def __init__(self, capacity: int):
        """
        初始化经验回放缓冲区

        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, experience: tuple) -> None:
        """
        添加经验元组到缓冲区

        Args:
            experience: 经验元组，格式由使用方决定
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list:
        """
        随机采样批次经验

        Args:
            batch_size: 采样批次大小

        Returns:
            采样得到的经验列表

        Raises:
            ValueError: 当缓冲区大小小于批次大小时
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小 {len(self.buffer)} < 批次大小 {batch_size}")
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()


class GaussianNoise:
    """
    高斯噪音探索策略

    为动作添加高斯噪音以促进探索，噪音强度随时间指数衰减。
    适用于连续动作空间的探索。
    """

    def __init__(
        self,
        action_dim: int,
        sigma: float,
        sigma_decay: float,
        min_sigma: float,
    ):
        """
        初始化高斯噪音

        Args:
            action_dim: 动作维度
            sigma: 初始噪音标准差
            sigma_decay: 噪音衰减率，每次调用后 sigma *= sigma_decay
            min_sigma: 噪音的最小标准差，防止探索完全停止
        """
        self.action_dim = action_dim
        self.sigma = sigma
        self.initial_sigma = sigma  # 保存初始值用于 reset
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """
        为动作添加高斯噪音

        Args:
            action: 原始动作，形状为 (action_dim,)

        Returns:
            添加噪音后的动作，裁剪到 [0, 1] 范围
        """
        noise = np.random.normal(0, self.sigma, self.action_dim)
        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)
        return np.clip(action + noise, 0.0, 1.0)

    def reset(self) -> None:
        """重置噪音强度到初始值"""
        self.sigma = self.initial_sigma
