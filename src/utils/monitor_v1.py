"""
PyQtGraph 实时训练监控模块 (v1)

适用于新架构 GameTrainer（异步 Fictitious Play + MADDPG 静态博弈求解器）。

布局：2行2列
- 左上：NashConv / Exploitability 收敛曲线
- 右上：各 Agent 信念均值曲线
- 下方（跨2列）：各智能体收益 + 总收益

特性：
- 基于 PyQtGraph，专为实时数据设计
- 滑动窗口限制数据点数量
- 支持长时间训练监控
"""

import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils.config_v1 import MonitorConfig


class TrainingMonitorV1:
    """
    PyQtGraph 训练过程实时监控器 (v1)

    使用方式：
    1. Trainer 初始化时创建监控器
    2. 每次评估完成后调用 on_evaluation()
    3. 每次 NashConv 计算后调用 on_nashconv()
    4. 训练结束时调用 close()
    """

    def __init__(
        self,
        config: "MonitorConfig",
        experiment_name: str,
        agent_names: list[str],
        exploitability_threshold: float,
    ):
        """
        初始化监控器

        Args:
            config: 监控配置
            experiment_name: 实验名称（显示在标题）
            agent_names: 智能体名称列表
            exploitability_threshold: Exploitability 收敛阈值
        """
        self.config = config
        self.experiment_name = experiment_name
        self.agent_names = agent_names
        self.n_agents = len(agent_names)
        self.exploitability_threshold = exploitability_threshold

        # 如果未启用，直接返回
        if not config.enabled:
            self.win = None
            return

        # 延迟导入 pyqtgraph
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore

            self.pg = pg
            self.QtCore = QtCore
        except ImportError:
            print("[Monitor] pyqtgraph not available, disabling monitor")
            print("[Monitor] Install with: pip install pyqtgraph PyQt5")
            self.config.enabled = False
            self.win = None
            return

        # 数据存储（使用 deque 实现滑动窗口）
        max_pts = config.max_points

        # 评估计数
        self.eval_steps: deque = deque(maxlen=max_pts)

        # NashConv 数据（稀疏，仅在检测点有值）
        self.nashconv_steps: deque = deque(maxlen=max_pts)
        self.nashconv_data: deque = deque(maxlen=max_pts)
        self.exploitability_data: deque = deque(maxlen=max_pts)

        # 信念均值数据（每个 Agent 一条曲线）
        self.belief_mean_data: dict[str, deque] = {
            name: deque(maxlen=max_pts) for name in agent_names
        }

        # 收益数据
        self.rewards_data: dict[str, deque] = {
            name: deque(maxlen=max_pts) for name in agent_names
        }
        self.total_rewards_data: deque = deque(maxlen=max_pts)

        # 当前状态
        self.total_evaluations = 0

        # 上次更新时间（用于节流）
        self.last_update_time = 0
        self.update_interval = config.update_interval / 1000.0  # 转换为秒

        # 初始化图表
        self._init_figure()

    def _init_figure(self):
        """初始化 PyQtGraph 图表"""
        pg = self.pg

        # 配置 PyQtGraph
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        # 创建主窗口
        self.win = pg.GraphicsLayoutWidget(
            title=f"Training Monitor - {self.experiment_name}"
        )
        self.win.resize(1200, 700)
        self.win.show()

        # 定义颜色
        self.colors = {
            "blue": (31, 119, 180),
            "orange": (255, 127, 14),
            "red": (214, 39, 40),
            "green": (44, 160, 44),
            "gray": (127, 127, 127),
            "threshold": (200, 0, 0, 150),
        }

        # === 左上：NashConv / Exploitability 收敛曲线 ===
        self.plot_nashconv = self.win.addPlot(
            title="NashConv / Exploitability", row=0, col=0
        )
        self.plot_nashconv.setLabel("left", "Value")
        self.plot_nashconv.setLabel("bottom", "Evaluation")
        self.plot_nashconv.addLegend(offset=(60, 10))
        self.plot_nashconv.showGrid(x=True, y=True, alpha=0.3)

        self.curve_nashconv = self.plot_nashconv.plot(
            pen=pg.mkPen(color=self.colors["blue"], width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=self.colors["blue"],
            name="NashConv",
        )
        self.curve_exploitability = self.plot_nashconv.plot(
            pen=pg.mkPen(color=self.colors["orange"], width=2),
            symbol="s",
            symbolSize=5,
            symbolBrush=self.colors["orange"],
            name="Exploitability",
        )
        self.line_threshold = self.plot_nashconv.addLine(
            y=self.exploitability_threshold,
            pen=pg.mkPen(
                color=self.colors["threshold"],
                width=2,
                style=self.pg.QtCore.Qt.PenStyle.DashLine,
            ),
            label="Threshold",
        )

        # === 右上：各 Agent 信念均值曲线 ===
        self.plot_belief = self.win.addPlot(
            title="Agent Belief Mean (Price Strategy)", row=0, col=1
        )
        self.plot_belief.setLabel("left", "Belief Mean")
        self.plot_belief.setLabel("bottom", "Evaluation")
        self.plot_belief.showGrid(x=True, y=True, alpha=0.3)

        # 为每个智能体创建信念曲线
        self.belief_curves = {}
        colors = self._generate_colors(self.n_agents)
        for i, name in enumerate(self.agent_names):
            self.belief_curves[name] = self.plot_belief.plot(
                pen=pg.mkPen(color=colors[i], width=2),
                symbol="o",
                symbolSize=3,
                symbolBrush=colors[i],
                name=name if self.n_agents <= 10 else None,
            )

        if self.n_agents <= 10:
            self.plot_belief.addLegend(offset=(60, 10))

        # === 下方：收益曲线（跨2列）===
        self.win.nextRow()
        self.plot_reward = self.win.addPlot(
            title=f"Agent Rewards ({self.n_agents} agents)", row=1, col=0, colspan=2
        )
        self.plot_reward.setLabel("left", "Reward")
        self.plot_reward.setLabel("bottom", "Evaluation")
        self.plot_reward.showGrid(x=True, y=True, alpha=0.3)

        # 为每个智能体创建收益曲线
        self.reward_curves = {}
        for i, name in enumerate(self.agent_names):
            self.reward_curves[name] = self.plot_reward.plot(
                pen=pg.mkPen(color=colors[i], width=1.5),
                symbol="o",
                symbolSize=3,
                symbolBrush=colors[i],
                name=name if self.n_agents <= 10 else None,
            )

        # 添加总收益曲线（深灰色虚线）
        self.total_reward_curve = self.plot_reward.plot(
            pen=pg.mkPen(
                color=(80, 80, 80), width=2, style=self.pg.QtCore.Qt.PenStyle.DashLine
            ),
            symbol="d",
            symbolSize=4,
            symbolBrush=(80, 80, 80),
            name="Total",
        )

        if self.n_agents <= 10:
            self.plot_reward.addLegend(offset=(60, 10))

        # 处理事件以显示窗口
        self._process_events()

    def _generate_colors(self, n: int) -> list[tuple]:
        """生成 n 个区分度高的颜色"""
        colors = []
        for i in range(n):
            hue = (i * 137.508) % 360  # 黄金角
            h = hue / 60
            x = int(255 * (1 - abs(h % 2 - 1)))
            c = 255
            if h < 1:
                colors.append((c, x, 0))
            elif h < 2:
                colors.append((x, c, 0))
            elif h < 3:
                colors.append((0, c, x))
            elif h < 4:
                colors.append((0, x, c))
            elif h < 5:
                colors.append((x, 0, c))
            else:
                colors.append((c, 0, x))
        return colors

    def _process_events(self):
        """处理 Qt 事件（非阻塞刷新）"""
        if not self.config.enabled or self.win is None:
            return
        # 处理所有待处理事件，防止窗口"未响应"
        app = self.pg.QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def _should_update(self) -> bool:
        """检查是否应该更新（节流）"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def on_evaluation(
        self,
        eval_id: int,
        rewards: dict[str, float],
        beliefs: np.ndarray,
    ) -> None:
        """
        每次评估完成后调用

        Args:
            eval_id: 评估编号
            rewards: 各智能体收益 {agent_name: reward}
            beliefs: 当前信念矩阵 (n_agents, n_periods)
        """
        if not self.config.enabled or self.win is None:
            return

        self.total_evaluations = eval_id + 1
        self.eval_steps.append(eval_id + 1)

        # 更新信念均值数据
        if self.config.show_beliefs:
            for i, name in enumerate(self.agent_names):
                belief_mean = float(np.mean(beliefs[i]))
                self.belief_mean_data[name].append(belief_mean)

        # 更新收益数据
        if self.config.show_rewards:
            total_reward = 0.0
            for name in self.agent_names:
                reward = rewards.get(name, 0)
                total_reward += reward
                self.rewards_data[name].append(reward)
            self.total_rewards_data.append(total_reward)

        # 节流更新
        if not self._should_update():
            return

        self._update_plots()

    def on_nashconv(
        self,
        eval_id: int,
        nashconv: float,
        exploitability: float,
    ) -> None:
        """
        每次 NashConv 计算后调用

        Args:
            eval_id: 评估编号
            nashconv: NashConv 值
            exploitability: Exploitability 值
        """
        if not self.config.enabled or self.win is None:
            return

        if not self.config.show_nashconv:
            return

        self.nashconv_steps.append(eval_id + 1)
        self.nashconv_data.append(nashconv)
        self.exploitability_data.append(exploitability)

        # 强制更新（NashConv 计算较少，每次都更新）
        self._update_plots()

    def _update_plots(self):
        """更新所有图表"""
        steps_list = list(self.eval_steps)

        # 更新 NashConv 曲线
        if self.config.show_nashconv and self.nashconv_steps:
            nashconv_steps = list(self.nashconv_steps)
            self.curve_nashconv.setData(nashconv_steps, list(self.nashconv_data))
            self.curve_exploitability.setData(
                nashconv_steps, list(self.exploitability_data)
            )

        # 更新信念曲线
        if self.config.show_beliefs and steps_list:
            for name in self.agent_names:
                if self.belief_mean_data[name]:
                    self.belief_curves[name].setData(
                        steps_list, list(self.belief_mean_data[name])
                    )

        # 更新收益曲线
        if self.config.show_rewards and steps_list:
            for name in self.agent_names:
                if self.rewards_data[name]:
                    self.reward_curves[name].setData(
                        steps_list, list(self.rewards_data[name])
                    )
            if self.total_rewards_data:
                self.total_reward_curve.setData(
                    steps_list, list(self.total_rewards_data)
                )

        # 更新窗口标题
        self.win.setWindowTitle(
            f"Training Monitor - {self.experiment_name} | "
            f"Evaluations: {self.total_evaluations}"
        )

        self._process_events()

    def wait_for_close(self):
        """
        保持窗口打开，等待用户手动关闭

        训练完成后调用此方法，窗口会保持响应，
        用户可以查看结果后手动关闭窗口。
        """
        if not self.config.enabled or self.win is None:
            return

        # 更新窗口标题，提示用户训练已完成
        self.win.setWindowTitle(
            f"Training Monitor - {self.experiment_name} | "
            f"Evaluations: {self.total_evaluations} | "
            f"[训练完成 - 关闭窗口继续]"
        )

        # 进入事件循环，直到窗口被关闭
        app = self.pg.QtWidgets.QApplication.instance()
        if app is not None:
            # 设置窗口关闭信号
            self.win.destroyed.connect(app.quit)
            # 进入事件循环
            app.exec()

        self.win = None

    def close(self):
        """关闭监控器"""
        if self.win is not None:
            self.win.close()
            self.win = None
