"""
PyQtGraph 实时训练监控模块

提供训练过程的实时可视化，包括：
- Step层收敛指标（价格相对变化率）
- 各智能体收益曲线
- UE-DTA内层迭代收敛过程（GM/P90/P95）

布局：上2下1
- 左上(3:2)：Step收敛指标（累积）
- 右上(2:3)：当前Step的UE-DTA迭代（每Step清空）
- 下方：各智能体收益（累积）

特性：
- 基于 PyQtGraph，专为实时数据设计，性能优于 matplotlib
- 滑动窗口限制数据点数量，避免内存问题
- 支持长时间训练监控
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
import time
from .config import MonitorConfig


class TrainingMonitor:
    """
    PyQtGraph 训练过程实时监控器

    使用方式：
    1. Trainer 初始化时创建监控器
    2. Episode 开始时调用 on_episode_start()
    3. UE-DTA 每轮迭代时调用 on_ue_iteration()
    4. Step 完成时调用 on_step_end()
    5. 训练结束时调用 close()
    """

    def __init__(self,
                 config: MonitorConfig,
                 experiment_name: str,
                 n_agents: int,
                 agent_names: List[str],
                 convergence_threshold: float,
                 ue_threshold: float):
        """
        初始化监控器

        Args:
            config: 监控配置
            experiment_name: 实验名称（显示在标题）
            n_agents: 智能体数量
            agent_names: 智能体名称列表
            convergence_threshold: Step层收敛阈值
            ue_threshold: UE-DTA收敛阈值
        """
        self.config = config
        self.experiment_name = experiment_name
        self.n_agents = n_agents
        self.agent_names = agent_names
        self.convergence_threshold = convergence_threshold
        self.ue_threshold = ue_threshold

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

        # 外层数据存储（使用 deque 实现滑动窗口）
        max_pts = config.max_points
        self.steps: deque = deque(maxlen=max_pts)
        self.convergence_data: deque = deque(maxlen=max_pts)
        self.rewards_data: Dict[str, deque] = {
            name: deque(maxlen=max_pts) for name in agent_names
        }

        # 内层数据存储（每Step清空，不需要限制）
        self.ue_iterations: List[int] = []
        self.gm_data: List[float] = []
        self.p90_data: List[float] = []
        self.p95_data: List[float] = []

        # 当前状态
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0

        # 上次更新时间（用于节流）
        self.last_update_time = 0
        self.update_interval = config.update_interval / 1000.0  # 转换为秒

        # 初始化图表
        self._init_figure()

    def _init_figure(self):
        """初始化 PyQtGraph 图表"""
        pg = self.pg

        # 配置 PyQtGraph
        pg.setConfigOptions(antialias=True, background='w', foreground='k')

        # 创建主窗口
        self.win = pg.GraphicsLayoutWidget(title=f'Training Monitor - {self.experiment_name}')
        self.win.resize(1200, 700)
        self.win.show()

        # 定义颜色
        self.colors = {
            'blue': (31, 119, 180),
            'orange': (255, 127, 14),
            'red': (214, 39, 40),
            'gray': (127, 127, 127),
            'threshold': (200, 0, 0, 150),
        }

        # === 左上：Step收敛指标 ===
        self.plot_conv = self.win.addPlot(title="Step Convergence", row=0, col=0)
        self.plot_conv.setLabel('left', 'Price Change Rate')
        self.plot_conv.setLabel('bottom', 'Step')
        self.plot_conv.addLegend(offset=(60, 10))
        self.plot_conv.showGrid(x=True, y=True, alpha=0.3)
        self.plot_conv.getAxis('bottom').setTickSpacing(major=10, minor=1)

        self.curve_conv = self.plot_conv.plot(
            pen=pg.mkPen(color=self.colors['blue'], width=2),
            symbol='o', symbolSize=4, symbolBrush=self.colors['blue'],
            name='Price Change'
        )
        self.line_conv_threshold = self.plot_conv.addLine(
            y=self.convergence_threshold,
            pen=pg.mkPen(color=self.colors['threshold'], width=2, style=self.pg.QtCore.Qt.PenStyle.DashLine),
            label='Threshold'
        )

        # === 右上：UE-DTA内层迭代 ===
        self.plot_ue = self.win.addPlot(title="UE-DTA Convergence (Current Step)", row=0, col=1)
        self.plot_ue.setLabel('left', 'Relative Gap')
        self.plot_ue.setLabel('bottom', 'UE-DTA Iteration')
        self.plot_ue.addLegend(offset=(60, 10))
        self.plot_ue.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ue.getAxis('bottom').setTickSpacing(major=10, minor=1)

        self.curve_gm = self.plot_ue.plot(
            pen=pg.mkPen(color=self.colors['blue'], width=2), name='GM'
        )
        self.curve_p90 = self.plot_ue.plot(
            pen=pg.mkPen(color=self.colors['orange'], width=2), name='P90'
        )
        self.curve_p95 = self.plot_ue.plot(
            pen=pg.mkPen(color=self.colors['red'], width=2), name='P95'
        )
        self.line_ue_threshold = self.plot_ue.addLine(
            y=self.ue_threshold,
            pen=pg.mkPen(color=self.colors['gray'], width=1, style=self.pg.QtCore.Qt.PenStyle.DashLine),
            label='Threshold'
        )

        # === 下方：收益曲线（跨两列）===
        self.win.nextRow()
        self.plot_reward = self.win.addPlot(
            title=f"Agent Rewards ({self.n_agents} agents)",
            row=1, col=0, colspan=2
        )
        self.plot_reward.setLabel('left', 'Reward')
        self.plot_reward.setLabel('bottom', 'Step')
        self.plot_reward.showGrid(x=True, y=True, alpha=0.3)
        self.plot_reward.getAxis('bottom').setTickSpacing(major=10, minor=1)

        # 为每个智能体创建曲线（带数据点标记）
        self.reward_curves = {}
        colors = self._generate_colors(self.n_agents)
        for i, name in enumerate(self.agent_names):
            self.reward_curves[name] = self.plot_reward.plot(
                pen=self.pg.mkPen(color=colors[i], width=1.5),
                symbol='o', symbolSize=4, symbolBrush=colors[i],
                name=name if self.n_agents <= 10 else None
            )

        if self.n_agents <= 10:
            self.plot_reward.addLegend(offset=(60, 10))

        # 处理事件以显示窗口
        self._process_events()

    def _generate_colors(self, n: int) -> List[tuple]:
        """生成 n 个区分度高的颜色"""
        colors = []
        for i in range(n):
            hue = (i * 137.508) % 360  # 黄金角
            # HSV to RGB (简化版本)
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
        self.pg.QtWidgets.QApplication.processEvents()

    def _should_update(self) -> bool:
        """检查是否应该更新（节流）"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def on_episode_start(self, episode: int):
        """
        Episode开始时调用

        Args:
            episode: Episode编号（从0开始）
        """
        if not self.config.enabled or self.win is None:
            return

        self.current_episode = episode

    def on_ue_iteration(self, iteration: int, gm: float, p90: float, p95: float):
        """
        UE-DTA每轮迭代时调用

        Args:
            iteration: 迭代轮次（从0开始）
            gm: Global Mean相对成本差
            p90: P90相对成本差
            p95: P95相对成本差
        """
        if not self.config.enabled or self.win is None:
            return

        if not self.config.show_ue_dta:
            return

        # 如果是第一轮，清空数据
        if iteration == 0:
            self.ue_iterations.clear()
            self.gm_data.clear()
            self.p90_data.clear()
            self.p95_data.clear()
            self.plot_ue.setTitle(f'UE-DTA Convergence (Step {self.current_step + 1})')

        # 添加数据
        self.ue_iterations.append(iteration + 1)
        self.gm_data.append(gm)
        self.p90_data.append(p90)
        self.p95_data.append(p95)

        # 节流更新
        if not self._should_update():
            return

        # 更新曲线
        self.curve_gm.setData(self.ue_iterations, self.gm_data)
        self.curve_p90.setData(self.ue_iterations, self.p90_data)
        self.curve_p95.setData(self.ue_iterations, self.p95_data)

        # 更新窗口标题
        self.win.setWindowTitle(
            f'Training Monitor - {self.experiment_name} | '
            f'Episode {self.current_episode + 1}, Step {self.current_step + 1}, '
            f'UE iter {iteration + 1}'
        )

        self._process_events()

    def on_step_end(self, step: int, convergence_rate: float, rewards: Dict[str, float]):
        """
        Step完成时调用

        Args:
            step: 当前Episode内的Step编号（从0开始）
            convergence_rate: 价格相对变化率
            rewards: 各智能体收益 {agent_name: reward}
        """
        if not self.config.enabled or self.win is None:
            return

        self.current_step = step
        self.total_steps += 1

        # 更新外层数据
        self.steps.append(self.total_steps)

        if self.config.show_convergence:
            self.convergence_data.append(convergence_rate)
            self.curve_conv.setData(list(self.steps), list(self.convergence_data))

        if self.config.show_rewards:
            for name in self.agent_names:
                self.rewards_data[name].append(rewards.get(name, 0))
                self.reward_curves[name].setData(
                    list(self.steps), list(self.rewards_data[name])
                )

        # 更新窗口标题
        self.win.setWindowTitle(
            f'Training Monitor - {self.experiment_name} | '
            f'Episode {self.current_episode + 1}, Step {step + 1}'
        )

        self._process_events()

    def close(self):
        """关闭监控器"""
        if self.win is not None:
            self.win.close()
            self.win = None
