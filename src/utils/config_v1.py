"""
新系统架构配置 (v1)

适用于 task_play.md / task_env.md 描述的异步 Fictitious Play + MADDPG 静态博弈求解器。

配置层级：
1. 算法配置 (MADDPGConfig/IDDPGConfig/MFDDPGConfig): 算法超参数
2. NashConvConfig: NashConv 收敛检测配置
3. TrainerConfig: 训练器配置（网络数据 + 训练控制）
4. ExperimentTask: 实验任务单元（配置组合 + 种子）

与旧系统 config.py 完全独立，即使相同的配置类也独立维护。
"""

from dataclasses import dataclass, field
from typing import Union
from datetime import datetime
import os


# ============================================================
# 算法配置 (Algorithm Configs)
# ============================================================

@dataclass
class MADDPGConfig:
    """MADDPG 算法配置"""

    # === 网络结构 ===
    actor_hidden_sizes: tuple = (64, 64)
    critic_hidden_sizes: tuple = (128, 64)

    # === 学习参数 ===
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.0          # 静态博弈，Critic 直接拟合即时收益
    tau: float = 0.01           # 软更新系数

    # === 经验回放 ===
    buffer_capacity: int = 10000
    max_batch_size: int = 64

    # === 探索策略 ===
    noise_sigma: float = 0.2
    noise_decay: float = 0.995
    min_noise: float = 0.02


@dataclass
class IDDPGConfig:
    """IDDPG 算法配置（与 MADDPG 对齐）"""

    # === 网络结构 ===
    actor_hidden_sizes: tuple = (64, 64)
    critic_hidden_sizes: tuple = (128, 64)

    # === 学习参数 ===
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.0
    tau: float = 0.01

    # === 经验回放 ===
    buffer_capacity: int = 10000
    max_batch_size: int = 64

    # === 探索策略 ===
    noise_sigma: float = 0.2
    noise_decay: float = 0.995
    min_noise: float = 0.02


@dataclass
class MFDDPGConfig:
    """MF-DDPG 算法配置（与 MADDPG 对齐）"""

    # === 网络结构 ===
    actor_hidden_sizes: tuple = (64, 64)
    critic_hidden_sizes: tuple = (128, 64)

    # === 学习参数 ===
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.0
    tau: float = 0.01

    # === 经验回放 ===
    buffer_capacity: int = 10000
    max_batch_size: int = 64

    # === 探索策略 ===
    noise_sigma: float = 0.2
    noise_decay: float = 0.995
    min_noise: float = 0.02


# 算法配置类型联合
AlgoConfig = Union[MADDPGConfig, IDDPGConfig, MFDDPGConfig]


# ============================================================
# NashConv 配置
# ============================================================

@dataclass
class NashConvConfig:
    """
    NashConv / Exploitability 收敛检测配置

    控制 NashConvChecker 的行为参数。
    """

    # === 收敛阈值 ===
    exploitability_threshold: float = 0.05  # Exploitability 收敛阈值

    # === 计算频率 ===
    check_interval: int = 10                # 每隔多少次评估计算一次 NashConv
    warmup: int = 100                       # 开始计算 NashConv 的最小评估次数

    # === 多起点梯度上升参数 ===
    n_starts: int = 5                       # 多起点数量（避免局部最优）
    optim_steps: int = 50                   # 每个起点的梯度上升步数
    lr: float = 0.01                        # 梯度上升学习率

    # === 假收敛检测 ===
    price_change_threshold: float = 0.01   # 价格变化率阈值（低于此值视为停滞）


# ============================================================
# 训练器配置
# ============================================================

@dataclass
class TrainerConfig:
    """
    GameTrainer 配置

    控制异步事件驱动主循环的行为参数。
    无 Episode 层，单次实验 = 单次求解尝试。
    """

    # === 网络数据 ===
    network_dir: str                        # 网络数据文件夹路径（相对于项目根目录）
    network_name: str                       # 网络名称

    # === 训练控制 ===
    max_evaluations: int = 1000             # 最大评估次数
    learn_interval: int = 5                 # K 步触发更新（每 K 次评估调用一次 learn）

    # === 系统配置 ===
    device: str = 'auto'                    # 计算设备：'auto' / 'cpu' / 'cuda'
    output_dir: str = 'results'             # 实验结果输出根目录
    n_workers: int = -1                     # 并行 Worker 数，-1 表示自动


# ============================================================
# 预定义训练器配置
# ============================================================

# SiouxFalls: 小网络 (4 个充电站)
TRAINER_CONFIG_SIOUXFALLS = TrainerConfig(
    network_dir='data/siouxfalls',
    network_name='siouxfalls',
    max_evaluations=1000,
    learn_interval=5,
)

# Berlin Friedrichshain: 中等网络 (20 个充电站)
TRAINER_CONFIG_BERLIN = TrainerConfig(
    network_dir='data/berlin_friedrichshain',
    network_name='berlin_friedrichshain',
    max_evaluations=2000,
    learn_interval=5,
)

# Anaheim: 大网络 (40 个充电站)
TRAINER_CONFIG_ANAHEIM = TrainerConfig(
    network_dir='data/anaheim',
    network_name='anaheim',
    max_evaluations=3000,
    learn_interval=5,
)


# ============================================================
# 实验任务单元
# ============================================================

@dataclass
class ExperimentTask:
    """
    实验任务单元

    代表一次具体的实验运行：[训练配置] + [算法] + [NashConv配置] + [随机种子]
    """

    name: str                               # 任务唯一标识名
    trainer_config: TrainerConfig           # 训练器配置
    algo_name: str                          # 算法名称: "MADDPG" / "IDDPG" / "MFDDPG"
    algo_config: AlgoConfig                 # 算法配置
    seed: int                               # 随机种子
    nashconv_config: NashConvConfig = field(default_factory=NashConvConfig)

    def get_output_path(self) -> str:
        """
        生成规范的输出路径

        格式: {output_dir}/{network_name}/{algo_name}/seed{seed}/{MM_DD_HH_mm}/
        示例: results/siouxfalls/MADDPG/seed42/01_26_15_30/
        """
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        return os.path.join(
            self.trainer_config.output_dir,
            self.trainer_config.network_name,
            self.algo_name,
            f"seed{self.seed}",
            timestamp
        )


# ============================================================
# 监控配置（可选）
# ============================================================

@dataclass
class MonitorConfig:
    """
    实时训练监控配置（可选）
    """

    # === 开关控制 ===
    enabled: bool = True

    # === 显示内容 ===
    show_nashconv: bool = True              # 显示 NashConv 曲线
    show_rewards: bool = True               # 显示智能体收益曲线
    show_beliefs: bool = True               # 显示信念演化

    # === 性能配置 ===
    max_points: int = 500                   # 每条曲线最大数据点数
    update_interval: int = 100              # 最小刷新间隔（毫秒）
