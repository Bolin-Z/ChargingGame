"""
简化配置管理系统

两层配置架构：
1. MADDPGConfig: MADDPG算法超参数
2. TrainingConfig: 训练流程控制参数（包含网络路径信息）

路径说明：
- network_dir: 相对于项目根目录的路径（如 'siouxfalls'）
- output_dir: 相对于项目根目录的路径（如 'results'）
"""

from dataclasses import dataclass


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
class TrainingConfig:
    """训练流程配置"""
    
    # === 网络数据配置 ===
    network_dir: str = 'siouxfalls'      # 网络数据文件夹路径（相对于项目根目录）
    network_name: str = 'siouxfalls'     # 网络名称（用于加载对应的文件）
    
    # === 训练控制配置 ===
    max_episodes: int = 10              # 最大episode数，每个episode尝试求解一次纳什均衡
    max_steps_per_episode: int = 1000     # 每个episode最大步数，防止无限循环
    
    # === 收敛控制配置 ===
    convergence_threshold: float = 0.01  # 纳什均衡价格收敛阈值，价格变化小于此值认为收敛
    stable_steps_required: int = 5       # 稳定收敛所需的连续步数，连续这么多步都小于阈值才认为收敛
    
    # === 系统配置 ===
    seed: int = 42                       # 随机种子，保证实验可重复
    device: str = 'auto'                 # 计算设备：'auto'自动选择, 'cpu'强制CPU, 'cuda'强制GPU
    
    # === 输出配置 ===
    output_dir: str = 'results'          # 实验结果输出根目录（相对于项目根目录）
    save_interval: int = 10              # 每隔多少episode保存一次模型（如果需要）


def get_default_configs():
    """
    获取默认配置
    
    Returns:
        tuple: (MADDPGConfig, TrainingConfig)
    """
    maddpg_config = MADDPGConfig()
    training_config = TrainingConfig()
    
    return maddpg_config, training_config