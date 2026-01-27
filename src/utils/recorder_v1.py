"""
实验记录与输出模块 (v1)

职责：
1. 收集训练过程中的所有数据
2. 训练结束后统一保存到 records.json
3. 生成实验摘要

数据结构与旧系统 MADDPGTrainer 的 step_records.json 保持相似风格。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..game.history import EvaluationRecord


@dataclass
class NashConvRecord:
    """NashConv 计算记录"""

    eval_id: int  # 对应的评估编号
    nashconv: float
    exploitability: float
    regrets: dict[str, float]  # 各 Agent 的后悔值


@dataclass
class LearnRecord:
    """学习步骤记录"""

    eval_id: int  # 对应的评估编号
    metrics: dict[str, Any]  # 学习指标（actor_loss, critic_loss, grad_norm 等）


class ExperimentRecorder:
    """
    实验记录器

    收集训练过程中的所有数据，训练结束后统一保存。
    """

    def __init__(
        self,
        experiment_name: str,
        agent_names: list[str],
        n_periods: int,
    ):
        """
        初始化记录器

        Args:
            experiment_name: 实验名称
            agent_names: Agent 名称列表
            n_periods: 时段数量
        """
        self.experiment_name = experiment_name
        self.agent_names = agent_names
        self.n_periods = n_periods
        self.start_time = datetime.now()

        # 评估记录（从 GameHistory 同步）
        self.evaluation_records: list[dict] = []

        # NashConv 记录
        self.nashconv_records: list[NashConvRecord] = []

        # 学习记录
        self.learn_records: list[LearnRecord] = []

        # 信念快照（每次评估时的信念矩阵）
        self.belief_snapshots: list[np.ndarray] = []

        # 价格变化率历史
        self.price_change_history: list[float] = []

    def record_evaluation(
        self,
        record: "EvaluationRecord",
        beliefs: np.ndarray,
        price_change_rate: float | None = None,
    ) -> None:
        """
        记录一次评估结果

        Args:
            record: GameHistory 的评估记录
            beliefs: 评估时的信念矩阵
            price_change_rate: 价格变化率（可选）
        """
        # 转换为可序列化的字典
        eval_dict = {
            "eval_id": record.eval_id,
            "pure_actions": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in record.pure_actions.items()
            },
            "noisy_actions": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in record.noisy_actions.items()
            },
            "rewards": {k: float(v) for k, v in record.rewards.items()},
            "flows": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in record.flows.items()
            },
            "ue_info": record.ue_info,
        }
        self.evaluation_records.append(eval_dict)

        # 保存信念快照
        self.belief_snapshots.append(beliefs.copy())

        # 保存价格变化率
        if price_change_rate is not None:
            self.price_change_history.append(price_change_rate)

    def record_nashconv(
        self,
        eval_id: int,
        nashconv: float,
        exploitability: float,
        regrets: dict[str, float],
    ) -> None:
        """
        记录 NashConv 计算结果

        Args:
            eval_id: 对应的评估编号
            nashconv: NashConv 值
            exploitability: Exploitability 值
            regrets: 各 Agent 的后悔值
        """
        self.nashconv_records.append(
            NashConvRecord(
                eval_id=eval_id,
                nashconv=nashconv,
                exploitability=exploitability,
                regrets=regrets,
            )
        )

    def record_learn(self, eval_id: int, metrics: dict[str, Any]) -> None:
        """
        记录学习步骤

        Args:
            eval_id: 对应的评估编号
            metrics: 学习指标
        """
        self.learn_records.append(LearnRecord(eval_id=eval_id, metrics=metrics))

    def save(
        self,
        output_dir: str,
        converged: bool,
        final_beliefs: np.ndarray,
        total_time: float,
    ) -> str:
        """
        保存所有记录到 JSON 文件

        Args:
            output_dir: 输出目录
            converged: 是否收敛
            final_beliefs: 最终信念矩阵
            total_time: 总训练时间（秒）

        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # 构建完整数据
        data = {
            "metadata": {
                "experiment_name": self.experiment_name,
                "agent_names": self.agent_names,
                "n_periods": self.n_periods,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "total_evaluations": len(self.evaluation_records),
                "total_nashconv_checks": len(self.nashconv_records),
                "total_learns": len(self.learn_records),
                "converged": converged,
            },
            "final_state": {
                "beliefs": final_beliefs.tolist(),
                "final_nashconv": (
                    self.nashconv_records[-1].nashconv
                    if self.nashconv_records
                    else None
                ),
                "final_exploitability": (
                    self.nashconv_records[-1].exploitability
                    if self.nashconv_records
                    else None
                ),
                "final_rewards": (
                    self.evaluation_records[-1]["rewards"]
                    if self.evaluation_records
                    else None
                ),
            },
            "records": self.evaluation_records,
            "nashconv_records": [
                {
                    "eval_id": r.eval_id,
                    "nashconv": r.nashconv,
                    "exploitability": r.exploitability,
                    "regrets": r.regrets,
                }
                for r in self.nashconv_records
            ],
            "learn_records": [
                {"eval_id": r.eval_id, "metrics": r.metrics} for r in self.learn_records
            ],
            "price_change_history": self.price_change_history,
        }

        # 保存到文件
        filepath = os.path.join(output_dir, "records.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 实验数据已保存到: {output_dir}")
        print(f"   📄 数据文件: records.json")
        print(f"   📊 评估次数: {len(self.evaluation_records)}")
        print(f"   📈 NashConv 检测次数: {len(self.nashconv_records)}")
        print(f"   🎓 学习次数: {len(self.learn_records)}")

        return filepath

    def get_summary(self) -> dict:
        """
        获取实验摘要

        Returns:
            摘要字典
        """
        return {
            "total_evaluations": len(self.evaluation_records),
            "total_nashconv_checks": len(self.nashconv_records),
            "total_learns": len(self.learn_records),
            "latest_nashconv": (
                self.nashconv_records[-1].nashconv if self.nashconv_records else None
            ),
            "latest_exploitability": (
                self.nashconv_records[-1].exploitability
                if self.nashconv_records
                else None
            ),
            "latest_rewards": (
                self.evaluation_records[-1]["rewards"]
                if self.evaluation_records
                else None
            ),
        }
