"""
GameTrainer: 异步事件驱动的博弈训练器

实现 task_play.md 描述的异步 Fictitious Play + MADDPG 静态博弈求解器。

核心设计：
1. 异步任务管理：保持 Worker 满载，解耦仿真与学习
2. 双轨记录：纯策略 → EMA 信念 / 噪声策略 → ReplayBuffer
3. K 步触发更新：每 K 次评估触发一次 learn()
4. 收敛检测：NashConv + 假收敛处理
5. 实时监控：PyQtGraph 可视化 + 完整记录保存

主循环（无 Episode 层）：
    while not_converged and total_evals < max_evaluations:
        1. 检查完成的 Future
        2. 双轨记录
        3. K 步触发 learn()
        4. 补充新任务保持队列满载
        5. 周期性 NashConv 检测
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from concurrent.futures import Future

import numpy as np
from tqdm import tqdm

from ..evaluator.network_data import NetworkDataLoader
from ..evaluator.pool import ParallelEvaluatorPool
from ..game.history import GameHistory
from ..game.nashconv import NashConvChecker
from ..utils.monitor_v1 import TrainingMonitorV1
from ..utils.recorder_v1 import ExperimentRecorder

if TYPE_CHECKING:
    from ..utils.config_v1 import ExperimentTask, NashConvConfig, MonitorConfig
    from ..algorithms.base import AlgorithmBase
    from ..evaluator.evaluator import EvalResult


@dataclass
class PendingTask:
    """待处理的评估任务"""

    future: Future["EvalResult"]
    pure_actions: dict[str, np.ndarray]
    noisy_actions: dict[str, np.ndarray]
    beliefs_snapshot: np.ndarray  # 提交时的信念快照（更新前）
    next_beliefs_snapshot: np.ndarray  # 信念更新后的快照（用于 ReplayBuffer）
    submit_time: float


@dataclass
class TrainingMetrics:
    """训练过程指标"""

    total_evaluations: int = 0
    total_learns: int = 0
    nashconv_history: list[float] = field(default_factory=list)
    exploitability_history: list[float] = field(default_factory=list)
    reward_history: list[dict[str, float]] = field(default_factory=list)
    price_change_history: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    """训练结果"""

    converged: bool
    final_nashconv: float | None
    final_exploitability: float | None
    final_beliefs: np.ndarray
    final_rewards: dict[str, float]
    metrics: TrainingMetrics
    total_time: float


class GameTrainer:
    """
    异步事件驱动的博弈训练器

    实现 Fictitious Play 风格的静态博弈求解，
    支持 MADDPG / IDDPG / MFDDPG 算法插件。
    """

    def __init__(
        self,
        task: "ExperimentTask",
        algorithm: "AlgorithmBase",
        monitor_config: "MonitorConfig | None" = None,
    ):
        """
        初始化 GameTrainer

        Args:
            task: 实验任务配置（包含 trainer_config 和 nashconv_config）
            algorithm: 已初始化的算法实例
            monitor_config: 监控配置（可选，None 则不启用监控）
        """
        self.task = task
        self.config = task.trainer_config
        self.nashconv_config = task.nashconv_config
        self.algorithm = algorithm

        # 加载网络数据
        loader = NetworkDataLoader(
            network_dir=self.config.network_dir,
            network_name=self.config.network_name,
            random_seed=task.seed,
        )
        self.network_data = loader.load()

        # 初始化组件
        self.pool = ParallelEvaluatorPool(
            network_data=self.network_data,
            n_workers=self.config.n_workers,
        )

        self.history = GameHistory(
            agent_names=self.network_data.agent_names,
            n_periods=self.network_data.n_periods,
        )

        self.nashconv_checker = NashConvChecker(
            n_starts=self.nashconv_config.n_starts,
            optim_steps=self.nashconv_config.optim_steps,
            lr=self.nashconv_config.lr,
        )

        # 初始化监控器
        if monitor_config is None:
            from ..utils.config_v1 import MonitorConfig
            monitor_config = MonitorConfig(enabled=False)

        self.monitor = TrainingMonitorV1(
            config=monitor_config,
            experiment_name=task.name,
            agent_names=self.network_data.agent_names,
            exploitability_threshold=self.nashconv_config.exploitability_threshold,
        )

        # 初始化记录器
        self.recorder = ExperimentRecorder(
            experiment_name=task.name,
            agent_names=self.network_data.agent_names,
            n_periods=self.network_data.n_periods,
        )

        # 运行时状态
        self._pending_tasks: list[PendingTask] = []
        self._metrics = TrainingMetrics()
        self._last_pure_actions: dict[str, np.ndarray] | None = None
        self._converged = False

    def train(self) -> TrainingResult:
        """
        执行异步事件驱动的训练主循环

        Returns:
            TrainingResult: 训练结果
        """
        start_time = time.time()

        with self.pool:
            # 1. 初始满负荷提交任务
            self._fill_task_queue()

            # 2. 事件驱动主循环（带进度条）
            with tqdm(
                total=self.config.max_evaluations,
                desc="训练进度",
                unit="eval",
                dynamic_ncols=True,
            ) as pbar:
                while not self._should_stop():
                    prev_evals = self._metrics.total_evaluations
                    self._process_completed_tasks()

                    # 更新进度条
                    new_evals = self._metrics.total_evaluations - prev_evals
                    if new_evals > 0:
                        pbar.update(new_evals)
                        # 更新后缀信息
                        postfix = {"收敛": "是" if self._converged else "否"}
                        if self._metrics.nashconv_history:
                            postfix["Expl"] = f"{self._metrics.exploitability_history[-1]:.4f}"
                        if self._metrics.reward_history:
                            total_reward = sum(self._metrics.reward_history[-1].values())
                            postfix["总收益"] = f"{total_reward:.1f}"
                        pbar.set_postfix(postfix)

                    # 短暂休眠避免忙等待，同时处理 GUI 事件
                    if not self._has_completed_tasks():
                        self.monitor._process_events()  # 保持 GUI 响应
                        time.sleep(0.01)

            # 3. 等待剩余任务完成
            self._drain_pending_tasks()

        total_time = time.time() - start_time

        # 4. 保存记录
        output_dir = self.task.get_output_path()
        self.recorder.save(
            output_dir=output_dir,
            converged=self._converged,
            final_beliefs=self.history.get_beliefs(),
            total_time=total_time,
        )

        # 5. 等待用户关闭监控窗口
        self.monitor.wait_for_close()

        # 构造返回结果
        return TrainingResult(
            converged=self._converged,
            final_nashconv=(
                self._metrics.nashconv_history[-1]
                if self._metrics.nashconv_history
                else None
            ),
            final_exploitability=(
                self._metrics.exploitability_history[-1]
                if self._metrics.exploitability_history
                else None
            ),
            final_beliefs=self.history.get_beliefs(),
            final_rewards=(
                self._metrics.reward_history[-1]
                if self._metrics.reward_history
                else {}
            ),
            metrics=self._metrics,
            total_time=total_time,
        )

    def _fill_task_queue(self) -> None:
        """
        填充任务队列至满载（队列大小 = Worker 数量）
        """
        while len(self._pending_tasks) < self.pool.n_workers:
            self._submit_new_task()

    def _submit_new_task(self) -> None:
        """
        提交一个新的评估任务

        信念更新时机：提交时立即更新（确保更新顺序与提交顺序一致）
        """
        # 获取当前信念（更新前）
        beliefs = self.history.get_beliefs()

        # 生成动作
        pure_actions, noisy_actions = self.algorithm.take_action(
            beliefs, add_noise=True
        )

        # === 提交时立即更新信念 ===
        self.history.update_belief(pure_actions)
        next_beliefs = self.history.get_beliefs()

        # 将噪声策略转换为实际价格
        prices = self._actions_to_prices(noisy_actions)

        # 异步提交
        future = self.pool.submit(prices)

        # 记录待处理任务（包含更新前后的信念快照）
        self._pending_tasks.append(
            PendingTask(
                future=future,
                pure_actions=pure_actions,
                noisy_actions=noisy_actions,
                beliefs_snapshot=beliefs.copy(),
                next_beliefs_snapshot=next_beliefs.copy(),
                submit_time=time.time(),
            )
        )

    def _actions_to_prices(
        self, actions: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        将归一化动作 [0, 1] 转换为实际价格

        Args:
            actions: 归一化动作 {agent_name: (n_periods,)}

        Returns:
            实际价格 {agent_name: (n_periods,)}
        """
        prices = {}
        for agent_name, action in actions.items():
            price_range = self.network_data.charging_nodes[agent_name]
            price_min, price_max = price_range[0], price_range[1]
            prices[agent_name] = price_min + action * (price_max - price_min)
        return prices

    def _has_completed_tasks(self) -> bool:
        """检查是否有已完成的任务"""
        return any(task.future.done() for task in self._pending_tasks)

    def _process_completed_tasks(self) -> None:
        """
        处理所有已完成的任务
        """
        completed = [task for task in self._pending_tasks if task.future.done()]

        for task in completed:
            self._pending_tasks.remove(task)
            self._handle_completed_task(task)

            # 立即补充新任务
            if not self._should_stop():
                self._submit_new_task()

    def _handle_completed_task(self, task: PendingTask) -> None:
        """
        处理单个已完成的任务

        注意：信念已在提交时更新，此处只处理结果记录和学习触发
        """
        result = task.future.result()

        # === 记录历史 ===
        record = self.history.record(
            pure_actions=task.pure_actions,
            noisy_actions=task.noisy_actions,
            rewards=result.rewards,
            flows=result.flows,
            ue_info={
                "ue_iterations": result.ue_iterations,
                "converged": result.converged,
            },
        )

        # === 存入 ReplayBuffer ===
        # 使用提交时预存的 next_beliefs_snapshot
        self.algorithm.store_experience(
            beliefs=task.beliefs_snapshot,
            noisy_actions=task.noisy_actions,
            rewards=result.rewards,
            next_beliefs=task.next_beliefs_snapshot,
        )

        # 更新指标
        self._metrics.total_evaluations += 1
        self._metrics.reward_history.append(result.rewards)

        # 计算价格变化率
        price_change_rate = None
        if self._last_pure_actions is not None:
            price_change_rate = self._compute_price_change_rate(
                self._last_pure_actions, task.pure_actions
            )
            self._metrics.price_change_history.append(price_change_rate)
        self._last_pure_actions = task.pure_actions

        # === 记录到 Recorder ===
        self.recorder.record_evaluation(
            record=record,
            beliefs=self.history.get_beliefs(),
            price_change_rate=price_change_rate,
        )

        # === 通知监控器 ===
        self.monitor.on_evaluation(
            eval_id=record.eval_id,
            rewards=result.rewards,
            beliefs=self.history.get_beliefs(),
        )

        # === K 步触发更新 ===
        if self._metrics.total_evaluations % self.config.learn_interval == 0:
            learn_metrics = self.algorithm.learn()
            if learn_metrics is not None:
                self._metrics.total_learns += 1
                # 记录学习指标
                self.recorder.record_learn(
                    eval_id=record.eval_id,
                    metrics=learn_metrics,
                )

        # === 周期性 NashConv 检测 ===
        self._maybe_check_convergence()

    def _compute_price_change_rate(
        self,
        prev_actions: dict[str, np.ndarray],
        curr_actions: dict[str, np.ndarray],
    ) -> float:
        """
        计算纯策略价格的相对变化率

        Returns:
            所有 Agent 平均相对变化率
        """
        changes = []
        for agent_name in prev_actions:
            prev = prev_actions[agent_name]
            curr = curr_actions[agent_name]
            # 避免除以零
            relative_change = np.abs(curr - prev) / (np.abs(prev) + 1e-8)
            changes.append(np.mean(relative_change))
        return float(np.mean(changes))

    def _maybe_check_convergence(self) -> None:
        """
        周期性检查收敛条件

        条件：
        1. 已过 warmup 阶段
        2. 达到检测间隔
        """
        total_evals = self._metrics.total_evaluations

        # warmup 阶段不检测
        if total_evals < self.nashconv_config.warmup:
            return

        # 检测间隔
        if total_evals % self.nashconv_config.check_interval != 0:
            return

        # 获取当前纯策略
        last_record = self.history.get_last_record()
        if last_record is None:
            return

        beliefs = self.history.get_beliefs()
        current_actions = last_record.pure_actions

        # 计算 NashConv
        nashconv, exploitability, regrets = self.nashconv_checker.compute(
            algorithm=self.algorithm,
            beliefs=beliefs,
            current_actions=current_actions,
        )

        self._metrics.nashconv_history.append(nashconv)
        self._metrics.exploitability_history.append(exploitability)

        # === 记录到 Recorder ===
        self.recorder.record_nashconv(
            eval_id=last_record.eval_id,
            nashconv=nashconv,
            exploitability=exploitability,
            regrets=regrets,
        )

        # === 通知监控器 ===
        self.monitor.on_nashconv(
            eval_id=last_record.eval_id,
            nashconv=nashconv,
            exploitability=exploitability,
        )

        # 检查收敛
        if exploitability < self.nashconv_config.exploitability_threshold:
            self._converged = True
            return

        # 假收敛检测：价格变化率低但 NashConv 高
        if self._metrics.price_change_history:
            recent_change = np.mean(self._metrics.price_change_history[-10:])
            if (
                recent_change < self.nashconv_config.price_change_threshold
                and exploitability >= self.nashconv_config.exploitability_threshold
            ):
                # 假收敛：重置探索噪音
                self.algorithm.reset_noise()

    def _should_stop(self) -> bool:
        """
        检查是否应该停止训练

        停止条件：
        1. 已收敛
        2. 达到最大评估次数
        """
        if self._converged:
            return True
        if self._metrics.total_evaluations >= self.config.max_evaluations:
            return True
        return False

    def _drain_pending_tasks(self) -> None:
        """
        处理剩余的待处理任务（等待完成）

        注意：信念已在提交时更新，此处只记录结果
        """
        for task in self._pending_tasks:
            try:
                result = task.future.result(timeout=300)  # 最多等待 5 分钟
                # 简单记录，不再触发学习
                self.history.record(
                    pure_actions=task.pure_actions,
                    noisy_actions=task.noisy_actions,
                    rewards=result.rewards,
                    flows=result.flows,
                )
                self._metrics.total_evaluations += 1
            except Exception:
                pass  # 忽略超时或异常
        self._pending_tasks.clear()


# ============================================================
# 算法工厂函数
# ============================================================

def create_algorithm(
    task: "ExperimentTask",
    network_data: Any,
) -> "AlgorithmBase":
    """
    根据实验任务创建算法实例

    Args:
        task: 实验任务配置
        network_data: 网络数据（提供 agent_names, n_periods）

    Returns:
        初始化的算法实例
    """
    from ..algorithms.maddpg.maddpg_v1 import MADDPGv1
    from ..algorithms.iddpg.iddpg_v1 import IDDPGv1
    from ..algorithms.mfddpg.mfddpg_v1 import MFDDPGv1

    algo_name = task.algo_name.upper()
    config = task.algo_config

    # 解析设备
    device = task.trainer_config.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    common_kwargs = {
        "agent_names": network_data.agent_names,
        "n_periods": network_data.n_periods,
        "device": device,
        "noise_sigma": config.noise_sigma,
        "noise_decay": config.noise_decay,
        "min_noise": config.min_noise,
    }

    if algo_name == "MADDPG":
        return MADDPGv1(
            **common_kwargs,
            actor_hidden_sizes=config.actor_hidden_sizes,
            critic_hidden_sizes=config.critic_hidden_sizes,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            buffer_capacity=config.buffer_capacity,
            max_batch_size=config.max_batch_size,
        )
    elif algo_name == "IDDPG":
        return IDDPGv1(
            **common_kwargs,
            actor_hidden_sizes=config.actor_hidden_sizes,
            critic_hidden_sizes=config.critic_hidden_sizes,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            buffer_capacity=config.buffer_capacity,
            max_batch_size=config.max_batch_size,
        )
    elif algo_name == "MFDDPG":
        return MFDDPGv1(
            **common_kwargs,
            actor_hidden_sizes=config.actor_hidden_sizes,
            critic_hidden_sizes=config.critic_hidden_sizes,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            buffer_capacity=config.buffer_capacity,
            max_batch_size=config.max_batch_size,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_experiment(task: "ExperimentTask") -> TrainingResult:
    """
    便捷函数：运行单个实验任务

    Args:
        task: 实验任务配置

    Returns:
        TrainingResult: 训练结果
    """
    # 加载网络数据
    loader = NetworkDataLoader(
        network_dir=task.trainer_config.network_dir,
        network_name=task.trainer_config.network_name,
        random_seed=task.seed,
    )
    network_data = loader.load()

    # 创建算法
    algorithm = create_algorithm(task, network_data)

    # 创建训练器并运行
    trainer = GameTrainer(task, algorithm)
    return trainer.train()
