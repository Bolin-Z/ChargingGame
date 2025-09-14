# 训练器模块
"""
训练器模块包含：
- BaseTrainer: 抽象基类
- MADDPGTrainer: MADDPG算法训练器
- 其他对比算法训练器（未来扩展）
"""

from .base_trainer import BaseTrainer
from .maddpg_trainer import MADDPGTrainer

__all__ = ['BaseTrainer', 'MADDPGTrainer']