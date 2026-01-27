# src/evaluator/__init__.py
"""
评估器模块

提供无状态的收益评估功能，支持并行评估。
"""

from .network_data import (
    NodeData,
    LinkData,
    DemandData,
    RouteInfo,
    NetworkData,
    NetworkDataLoader,
)

from .evaluator import (
    EvalResult,
    VehicleInfo,
    EVCSRewardEvaluator,
)

from .pool import (
    ParallelEvaluatorPool,
)

__all__ = [
    # network_data
    "NodeData",
    "LinkData",
    "DemandData",
    "RouteInfo",
    "NetworkData",
    "NetworkDataLoader",
    # evaluator
    "EvalResult",
    "VehicleInfo",
    "EVCSRewardEvaluator",
    # pool
    "ParallelEvaluatorPool",
]
