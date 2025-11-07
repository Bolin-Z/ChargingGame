# IDDPG算法模块
"""
独立深度确定性策略梯度算法实现

主要组件：
- networks.py: Actor和Critic网络架构（局部观测输入）
- iddpg.py: 核心算法实现（独立训练、局部状态、独立经验回放）

与MADDPG的核心区别：
- 完全去中心化训练：每个agent独立学习
- 局部状态输入：Critic仅使用自身历史和全局价格信息
- 独立经验回放：每个agent维护独立的ReplayBuffer
"""

from .networks import ActorNetwork, CriticNetwork
from .iddpg import DDPG, IndependentDDPG, ReplayBuffer, GaussianNoise

__all__ = [
    'ActorNetwork', 'CriticNetwork',
    'DDPG', 'IndependentDDPG', 'ReplayBuffer', 'GaussianNoise'
]
