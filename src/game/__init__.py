"""
Game 模块

提供博弈训练相关的核心组件：
- GameHistory: 博弈历史管理器（EMA 信念 + 评估记录）
- NashConvChecker: NashConv / Exploitability 收敛检测器
- GameTrainer: 统一训练器（待实现）
"""

from src.game.history import EvaluationRecord, GameHistory
from src.game.nashconv import NashConvChecker

__all__ = ["GameHistory", "EvaluationRecord", "NashConvChecker"]
