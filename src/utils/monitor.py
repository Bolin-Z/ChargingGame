"""
实时训练监控模块

提供训练过程的实时可视化，包括：
- Step层收敛指标（价格相对变化率）
- 各智能体收益曲线
- UE-DTA内层迭代收敛过程（GM/P90/P95）

布局：上2下1
- 左上(3:2)：Step收敛指标（累积）
- 右上(2:3)：当前Step的UE-DTA迭代（每Step清空）
- 下方：各智能体收益（累积）
"""

import numpy as np
from typing import Dict, List, Optional
from .config import MonitorConfig


class TrainingMonitor:
    """
    训练过程实时监控器

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
            convergence_threshold: Step层收敛阈值（从ScenarioProfile获取）
            ue_threshold: UE-DTA收敛阈值（从Env配置获取）
        """
        self.config = config
        self.experiment_name = experiment_name
        self.n_agents = n_agents
        self.agent_names = agent_names
        self.convergence_threshold = convergence_threshold
        self.ue_threshold = ue_threshold

        # 如果未启用，直接返回
        if not config.enabled:
            self.fig = None
            return

        # 延迟导入 matplotlib（避免无GUI环境报错）
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            self.plt = plt
        except ImportError:
            print("[Monitor] matplotlib not available, disabling monitor")
            self.config.enabled = False
            self.fig = None
            return

        # 外层数据存储（累积）
        self.steps: List[int] = []
        self.convergence_data: List[float] = []
        self.rewards_data: Dict[str, List[float]] = {name: [] for name in agent_names}
        self.episode_bounds: List[int] = []

        # 内层数据存储（每Step清空）
        self.ue_iterations: List[int] = []
        self.gm_data: List[float] = []
        self.p90_data: List[float] = []
        self.p95_data: List[float] = []

        # 当前状态
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0  # 跨Episode累积

        # 初始化图表
        self._init_figure()

    def _init_figure(self):
        """初始化图表"""
        plt = self.plt

        # 启用交互模式
        plt.ion()

        # 创建图表
        self.fig = plt.figure(figsize=self.config.figure_size)
        self.fig.suptitle(f'Training Monitor - {self.experiment_name}',
                         fontsize=14, fontweight='bold')

        # 使用GridSpec实现不等宽布局（3:2）
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 5, figure=self.fig, height_ratios=[1, 1])

        # 上方两个图（3:2宽度比）
        self.ax_conv = self.fig.add_subplot(gs[0, 0:3])   # 左上，占3列
        self.ax_ue = self.fig.add_subplot(gs[0, 3:5])     # 右上，占2列
        # 下方宽图（收益）
        self.ax_reward = self.fig.add_subplot(gs[1, :])

        # === 左上：Step收敛指标 ===
        self.line_conv, = self.ax_conv.plot([], [], 'b-', linewidth=1.5,
                                             marker='o', markersize=3)
        self.ax_conv.axhline(y=self.convergence_threshold, color='r',
                             linestyle='--', alpha=0.7, label='Threshold')
        self.ax_conv.set_xlabel('Step')
        self.ax_conv.set_ylabel('Price Change Rate')
        self.ax_conv.set_title('Step Convergence')
        self.ax_conv.set_xlim(0, 100)
        self.ax_conv.set_ylim(0, 0.5)
        self.ax_conv.grid(True, alpha=0.3)
        self.ax_conv.legend(loc='upper right')

        # === 右上：UE-DTA内层迭代 ===
        self.line_gm, = self.ax_ue.plot([], [], 'b-', linewidth=1.5, label='GM')
        self.line_p90, = self.ax_ue.plot([], [], 'orange', linewidth=1.5, label='P90')
        self.line_p95, = self.ax_ue.plot([], [], 'r-', linewidth=1.5, label='P95')
        self.ax_ue.axhline(y=self.ue_threshold, color='gray',
                           linestyle='--', alpha=0.5, label='Threshold')
        self.ax_ue.set_xlabel('UE-DTA Iteration')
        self.ax_ue.set_ylabel('Relative Gap')
        self.ax_ue.set_title('UE-DTA Convergence (Current Step)')
        self.ax_ue.set_xlim(0, 50)
        self.ax_ue.set_ylim(0, 0.15)
        self.ax_ue.grid(True, alpha=0.3)
        self.ax_ue.legend(loc='upper right')

        # === 下方：收益曲线 ===
        self.reward_lines = {}
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_agents))
        for i, name in enumerate(self.agent_names):
            line, = self.ax_reward.plot([], [], alpha=0.7, linewidth=1.2,
                                         color=colors[i], label=name)
            self.reward_lines[name] = line
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_title(f'Agent Rewards ({self.n_agents} agents)')
        self.ax_reward.set_xlim(0, 100)
        self.ax_reward.set_ylim(0, 1000)
        self.ax_reward.grid(True, alpha=0.3)
        if self.n_agents <= 10:
            self.ax_reward.legend(loc='upper left', ncol=min(5, self.n_agents))

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_episode_start(self, episode: int):
        """
        Episode开始时调用

        Args:
            episode: Episode编号（从0开始）
        """
        if not self.config.enabled or self.fig is None:
            return

        self.current_episode = episode

        # 如果不是第一个Episode，标记边界
        if self.total_steps > 0:
            self.episode_bounds.append(self.total_steps)
            # 添加垂直虚线
            self.ax_conv.axvline(x=self.total_steps, color='gray',
                                linestyle=':', alpha=0.5)
            self.ax_reward.axvline(x=self.total_steps, color='gray',
                                   linestyle=':', alpha=0.5)

    def on_ue_iteration(self, iteration: int, gm: float, p90: float, p95: float):
        """
        UE-DTA每轮迭代时调用

        Args:
            iteration: 迭代轮次（从0开始）
            gm: Global Mean相对成本差
            p90: P90相对成本差
            p95: P95相对成本差
        """
        if not self.config.enabled or self.fig is None:
            return

        if not self.config.show_ue_dta:
            return

        # 如果是第一轮，清空数据
        if iteration == 0:
            self.ue_iterations.clear()
            self.gm_data.clear()
            self.p90_data.clear()
            self.p95_data.clear()
            self.line_gm.set_data([], [])
            self.line_p90.set_data([], [])
            self.line_p95.set_data([], [])
            self.ax_ue.set_title(f'UE-DTA Convergence (Step {self.current_step + 1})')

        # 添加数据
        self.ue_iterations.append(iteration + 1)
        self.gm_data.append(gm)
        self.p90_data.append(p90)
        self.p95_data.append(p95)

        # 更新曲线
        self.line_gm.set_data(self.ue_iterations, self.gm_data)
        self.line_p90.set_data(self.ue_iterations, self.p90_data)
        self.line_p95.set_data(self.ue_iterations, self.p95_data)

        # 动态调整坐标轴
        max_iter = max(self.ue_iterations) if self.ue_iterations else 50
        self.ax_ue.set_xlim(0, max(50, max_iter * 1.1))

        max_val = max(max(self.gm_data), max(self.p90_data), max(self.p95_data))
        self.ax_ue.set_ylim(0, max(0.05, max_val * 1.2))

        # 更新标题
        self.fig.suptitle(
            f'Training Monitor - {self.experiment_name} | '
            f'Episode {self.current_episode + 1}, Step {self.current_step + 1}, '
            f'UE-DTA iter {iteration + 1}',
            fontsize=14, fontweight='bold'
        )

        # 刷新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_step_end(self, step: int, convergence_rate: float, rewards: Dict[str, float]):
        """
        Step完成时调用

        Args:
            step: 当前Episode内的Step编号（从0开始）
            convergence_rate: 价格相对变化率
            rewards: 各智能体收益 {agent_name: reward}
        """
        if not self.config.enabled or self.fig is None:
            return

        self.current_step = step
        self.total_steps += 1

        # 更新外层数据
        self.steps.append(self.total_steps)

        if self.config.show_convergence:
            self.convergence_data.append(convergence_rate)
            self.line_conv.set_data(self.steps, self.convergence_data)

        if self.config.show_rewards:
            for name in self.agent_names:
                self.rewards_data[name].append(rewards.get(name, 0))
                self.reward_lines[name].set_data(self.steps, self.rewards_data[name])

        # 动态调整坐标轴
        max_step = max(self.steps) if self.steps else 100
        self.ax_conv.set_xlim(0, max(100, max_step * 1.1))
        self.ax_reward.set_xlim(0, max(100, max_step * 1.1))

        if self.convergence_data:
            max_conv = max(self.convergence_data)
            self.ax_conv.set_ylim(0, max(0.1, max_conv * 1.2))

        all_rewards = [r for rewards_list in self.rewards_data.values()
                       for r in rewards_list]
        if all_rewards:
            self.ax_reward.set_ylim(0, max(all_rewards) * 1.1)

        # 更新标题
        self.fig.suptitle(
            f'Training Monitor - {self.experiment_name} | '
            f'Episode {self.current_episode + 1}, Step {step + 1}',
            fontsize=14, fontweight='bold'
        )

        # 刷新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """关闭监控器"""
        if self.fig is not None:
            self.plt.ioff()
            self.plt.close(self.fig)
            self.fig = None
