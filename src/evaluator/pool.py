# src/evaluator/pool.py
"""
ParallelEvaluatorPool: 并行评估池

管理多进程 Worker，提供批量评估接口。
每个 Worker 进程维护独立的 EVCSRewardEvaluator 实例。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, Future
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .network_data import NetworkData
    from .evaluator import EvalResult


# ============================================================
# Worker 进程全局变量和函数（必须在模块顶层定义，Windows spawn 兼容）
# ============================================================

_worker_evaluator = None


def _worker_initializer(network_data: "NetworkData"):
    """
    Worker 进程初始化函数

    在每个 Worker 进程启动时调用一次，创建 Evaluator 实例。
    NetworkData 通过 pickle 序列化传递。

    Args:
        network_data: 可序列化的网络数据
    """
    global _worker_evaluator
    # 延迟导入，避免循环依赖
    from .evaluator import EVCSRewardEvaluator
    _worker_evaluator = EVCSRewardEvaluator(network_data)


def _worker_evaluate(prices: dict[str, np.ndarray], seed: int | None) -> "EvalResult":
    """
    Worker 进程评估函数

    使用进程全局的 Evaluator 实例执行评估。

    Args:
        prices: 各充电站各时段价格 {agent_name: (n_periods,)}
        seed: 随机种子

    Returns:
        EvalResult: 评估结果
    """
    global _worker_evaluator
    if _worker_evaluator is None:
        raise RuntimeError("Worker evaluator not initialized")
    return _worker_evaluator.evaluate(prices, seed)


# ============================================================
# ParallelEvaluatorPool 主类
# ============================================================

class ParallelEvaluatorPool:
    """
    并行评估池

    管理多个 Worker 进程，每个进程维护独立的 EVCSRewardEvaluator。
    支持单次评估、批量评估、异步提交。

    使用示例:
        with ParallelEvaluatorPool(network_data, n_workers=4) as pool:
            result = pool.evaluate(prices, seed=42)
            results = pool.evaluate_batch(prices_list)
    """

    def __init__(self, network_data: "NetworkData", n_workers: int = -1):
        """
        初始化并行评估池

        Args:
            network_data: 可序列化的网络数据
            n_workers: Worker 进程数量
                       -1: 使用 min(cpu_count - 1, 4)
                       正整数: 指定数量
        """
        self.network_data = network_data
        self.n_workers = self._resolve_n_workers(n_workers)
        self._executor: ProcessPoolExecutor | None = None
        self._started = False

    @staticmethod
    def _resolve_n_workers(n_workers: int) -> int:
        """
        解析 Worker 数量配置

        Args:
            n_workers: 配置值，-1 表示自动

        Returns:
            int: 实际 Worker 数量
        """
        if n_workers == -1:
            cpu_count = os.cpu_count() or 1
            return min(cpu_count - 1, 4) if cpu_count > 1 else 1
        elif n_workers > 0:
            return n_workers
        else:
            raise ValueError(f"n_workers must be -1 or positive, got {n_workers}")

    def start(self):
        """
        启动进程池

        创建 ProcessPoolExecutor，每个 Worker 进程通过 initializer
        接收 NetworkData 并创建 Evaluator 实例。
        """
        if self._started:
            return

        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_initializer,
            initargs=(self.network_data,)
        )
        self._started = True

    def shutdown(self, wait: bool = True):
        """
        关闭进程池

        Args:
            wait: 是否等待所有任务完成
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            self._started = False

    def evaluate(self, prices: dict[str, np.ndarray], seed: int | None = None) -> "EvalResult":
        """
        单次评估（阻塞）

        Args:
            prices: 各充电站各时段价格
            seed: 随机种子

        Returns:
            EvalResult: 评估结果
        """
        if not self._started:
            self.start()

        future = self._executor.submit(_worker_evaluate, prices, seed)
        return future.result()

    def evaluate_batch(
        self,
        prices_list: list[dict[str, np.ndarray]],
        seeds: list[int] | None = None
    ) -> list["EvalResult"]:
        """
        批量并行评估（阻塞）

        Args:
            prices_list: 价格列表，每个元素是一组价格
            seeds: 随机种子列表，长度需与 prices_list 一致，None 表示不设置种子

        Returns:
            list[EvalResult]: 评估结果列表，顺序与输入一致
        """
        if not self._started:
            self.start()

        if seeds is None:
            seeds = [None] * len(prices_list)
        elif len(seeds) != len(prices_list):
            raise ValueError(
                f"seeds length ({len(seeds)}) must match prices_list length ({len(prices_list)})"
            )

        # 提交所有任务
        futures = [
            self._executor.submit(_worker_evaluate, prices, seed)
            for prices, seed in zip(prices_list, seeds)
        ]

        # 按顺序收集结果
        results = [f.result() for f in futures]
        return results

    def submit(
        self,
        prices: dict[str, np.ndarray],
        seed: int | None = None
    ) -> Future["EvalResult"]:
        """
        异步提交评估任务

        Args:
            prices: 各充电站各时段价格
            seed: 随机种子

        Returns:
            Future[EvalResult]: Future 对象，可通过 .result() 获取结果
        """
        if not self._started:
            self.start()

        return self._executor.submit(_worker_evaluate, prices, seed)

    def __enter__(self) -> "ParallelEvaluatorPool":
        """支持 with 语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时关闭进程池"""
        self.shutdown(wait=True)
        return False

    @property
    def is_running(self) -> bool:
        """进程池是否正在运行"""
        return self._started and self._executor is not None
