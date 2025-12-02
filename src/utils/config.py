"""
配置管理系统

三层配置架构：
1. 算法配置 (MADDPGConfig/IDDPGConfig/MFDDPGConfig): 算法超参数
2. ScenarioProfile (场景档案): 数据路径 + 训练规则，确保同场景下不同算法对比公平
3. ExperimentTask (实验任务单元): 场景档案 + 算法 + 种子，代表一次具体运行

路径说明：
- network_dir: 相对于项目根目录的路径（如 'data/siouxfalls'）
- output_dir: 相对于项目根目录的路径（如 'results'）
"""

from dataclasses import dataclass
from typing import Union
from datetime import datetime
import os


@dataclass
class MADDPGConfig:
    """MADDPG算法配置"""
    
    # === 网络结构配置 ===
    actor_hidden_sizes: tuple = (64, 64)     # Actor网络隐藏层结构（支持任意层数）
    critic_hidden_sizes: tuple = (128, 64)   # Critic网络隐藏层结构（支持任意层数）
    
    # === 学习参数配置 ===
    actor_lr: float = 0.001      # Actor网络学习率
    critic_lr: float = 0.001     # Critic网络学习率
    gamma: float = 0.99          # 折扣因子，用于计算未来奖励的权重
    tau: float = 0.01            # 软更新系数，控制目标网络更新速度
    
    # === 经验回放配置 ===
    buffer_capacity: int = 10000 # 经验回放缓冲区容量
    max_batch_size: int = 64     # 最大批次大小，实际会根据缓冲区大小动态调整
    
    # === 探索策略配置 ===
    noise_sigma: float = 0.1     # 高斯噪音初始标准差，控制探索强度
    noise_decay: float = 0.99  # 噪音衰减率，每次调用后噪音强度衰减
    min_noise: float = 0.0005      # 最小噪音标准差，防止探索完全停止


@dataclass
class IDDPGConfig:
    """
    IDDPG算法配置

    与MADDPG对齐的超参数，确保公平对比：
    - 除了网络输入维度，所有参数与MADDPG完全一致
    - 使用局部状态而非全局状态
    - 每个agent维护独立的经验回放Buffer
    """

    # === 网络结构配置（与MADDPG对齐）===
    actor_hidden_sizes: tuple = (64, 64)     # Actor网络隐藏层结构
    critic_hidden_sizes: tuple = (128, 64)   # Critic网络隐藏层结构

    # === 学习参数配置（与MADDPG对齐）===
    actor_lr: float = 0.001      # Actor网络学习率
    critic_lr: float = 0.001     # Critic网络学习率
    gamma: float = 0.99          # 折扣因子
    tau: float = 0.01            # 软更新系数

    # === 经验回放配置（与MADDPG对齐）===
    buffer_capacity: int = 10000 # 经验回放缓冲区容量（每个agent独立）
    max_batch_size: int = 64     # 最大批次大小

    # === 探索策略配置（与MADDPG对齐）===
    noise_sigma: float = 0.1     # 高斯噪音初始标准差
    noise_decay: float = 0.99    # 噪音衰减率
    min_noise: float = 0.0005    # 最小噪音标准差


@dataclass
class MFDDPGConfig:
    """
    MF-DDPG算法配置

    与MADDPG对齐的超参数，确保公平对比：
    - 除了网络输入维度，所有参数与MADDPG完全一致
    - 使用Mean Field状态压缩
    - 每个agent维护独立的经验回放Buffer
    - 独立训练 + Mean Field近似
    """

    # === 网络结构配置（与MADDPG对齐）===
    actor_hidden_sizes: tuple = (64, 64)     # Actor网络隐藏层结构
    critic_hidden_sizes: tuple = (128, 64)   # Critic网络隐藏层结构

    # === 学习参数配置（与MADDPG对齐）===
    actor_lr: float = 0.001      # Actor网络学习率
    critic_lr: float = 0.001     # Critic网络学习率
    gamma: float = 0.99          # 折扣因子
    tau: float = 0.01            # 软更新系数

    # === 经验回放配置（与MADDPG对齐）===
    buffer_capacity: int = 10000 # 经验回放缓冲区容量（每个agent独立）
    max_batch_size: int = 64     # 最大批次大小

    # === 探索策略配置（与MADDPG对齐）===
    noise_sigma: float = 0.1     # 高斯噪音初始标准差
    noise_decay: float = 0.99    # 噪音衰减率
    min_noise: float = 0.0005    # 最小噪音标准差


@dataclass
class ScenarioProfile:
    """
    场景档案 (Scenario Profile)

    绑定数据来源和训练规则，确保同一场景下不同算法使用统一标准进行对比。
    不同规模的网络需要不同的收敛阈值和训练步数。

    推荐使用预定义的场景档案：PROFILE_SIOUXFALLS, PROFILE_BERLIN
    """

    # === 网络数据配置 ===
    network_dir: str                     # 网络数据文件夹路径（相对于项目根目录）
    network_name: str                    # 网络名称（用于加载对应的文件）

    # === 训练控制配置 ===
    max_episodes: int                    # 最大episode数，每个episode尝试求解一次纳什均衡
    max_steps_per_episode: int           # 每个episode最大步数，防止无限循环

    # === 收敛控制配置 ===
    convergence_threshold: float         # 纳什均衡价格收敛阈值，价格变化小于此值认为收敛
    stable_steps_required: int           # 稳定收敛所需的连续步数
    stable_episodes_required: int        # 训练提前终止所需的连续收敛episodes数

    # === 系统配置（保留默认值）===
    device: str = 'auto'                 # 计算设备：'auto'自动选择, 'cpu'强制CPU, 'cuda'强制GPU
    output_dir: str = 'results'          # 实验结果输出根目录（相对于项目根目录）
    save_interval: int = 10              # 每隔多少episode保存一次模型


# 算法配置类型联合
AlgoConfig = Union[MADDPGConfig, IDDPGConfig, MFDDPGConfig]


# ============================================================
# 预定义场景档案 (Predefined Scenario Profiles)
# ============================================================

# SiouxFalls: 小网络 (4个充电站)
PROFILE_SIOUXFALLS = ScenarioProfile(
    network_dir='data/siouxfalls',
    network_name='siouxfalls',
    max_episodes=10,
    max_steps_per_episode=1000,
    convergence_threshold=0.01,
    stable_steps_required=5,
    stable_episodes_required=3,
)

# Berlin Friedrichshain: 大网络 (20个充电站)
PROFILE_BERLIN = ScenarioProfile(
    network_dir='data/berlin_friedrichshain',
    network_name='berlin_friedrichshain',
    max_episodes=100,
    max_steps_per_episode=1000,
    convergence_threshold=0.02,
    stable_steps_required=5,
    stable_episodes_required=3,
)


@dataclass
class ExperimentTask:
    """
    实验任务单元

    代表一次具体的实验运行：[场景档案] + [算法] + [随机种子]
    用于组织"多场景 x 多算法"的对比实验。
    """
    name: str                          # 任务唯一标识名 (如 "Sioux_MADDPG_Seed42")
    scenario: ScenarioProfile          # 场景档案 (包含数据路径和训练规则)
    algo_name: str                     # 算法名称 ("MADDPG", "IDDPG", "MFDDPG")
    algo_config: AlgoConfig            # 对应的算法配置对象
    seed: int                          # 本次运行的随机种子

    def get_output_path(self) -> str:
        """
        生成规范的输出路径

        格式: {output_dir}/{network_name}/{algo_name}/seed{seed}/{MM_DD_HH_mm}/
        示例: results/siouxfalls/MADDPG/seed42/12_02_15_10/
        """
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        return os.path.join(
            self.scenario.output_dir,
            self.scenario.network_name,
            self.algo_name,
            f"seed{self.seed}",
            timestamp
        )

