# 算法模块
"""
算法模块包含各种多智能体强化学习算法：
- MADDPG: 多智能体深度确定性策略梯度
- 其他对比算法（未来扩展）
"""

from .maddpg import MADDPG

__all__ = ['MADDPG']