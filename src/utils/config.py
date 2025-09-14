"""
配置管理工具

统一管理训练配置和超参数
"""

import json
import os
import argparse
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    # 环境配置
    network_dir: str = 'siouxfalls'
    network_name: str = 'siouxfalls'
    max_steps_per_episode: int = 500  # 增加到500步，给算法足够时间寻找纳什均衡
    convergence_threshold: float = 0.01
    
    # MADDPG配置
    buffer_capacity: int = 10000
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.99
    tau: float = 0.01
    actor_hidden_sizes: list = None
    critic_hidden_sizes: list = None
    
    # 训练配置
    max_episodes: int = 200  # 减少episode数，每个episode有500步，总共10万步训练
    seed: int = 42
    device: str = 'auto'
    
    # 输出配置
    output_dir: str = 'results'
    log_level: str = 'INFO'
    save_interval: int = 20  # 每20轮保存一次模型（适配200轮总数）
    
    def __post_init__(self):
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [64, 64]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [128, 64]


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EVCS充电站价格博弈MADDPG求解')
    
    # 环境配置
    parser.add_argument('--network_dir', type=str, default='siouxfalls',
                        help='网络数据文件夹路径')
    parser.add_argument('--network_name', type=str, default='siouxfalls', 
                        help='网络名称')
    
    # 训练配置
    parser.add_argument('--max_episodes', type=int, default=200,
                        help='最大训练轮数（每轮寻找一次纳什均衡）')
    parser.add_argument('--max_steps_per_episode', type=int, default=500,
                        help='每轮最大步数（用于寻找纳什均衡）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # MADDPG配置
    parser.add_argument('--buffer_capacity', type=int, default=10000,
                        help='经验回放缓冲区容量')
    parser.add_argument('--actor_lr', type=float, default=0.001,
                        help='Actor网络学习率')
    parser.add_argument('--critic_lr', type=float, default=0.001,
                        help='Critic网络学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='软更新系数')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (cpu/cuda/auto)')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='日志级别 (DEBUG/INFO/WARNING/ERROR)')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='模型保存间隔')
    
    return parser.parse_args()


def load_config(config_path: str = None) -> TrainingConfig:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（可选）
    
    Returns:
        TrainingConfig: 训练配置对象
    """
    # 从命令行参数开始
    args = parse_arguments()
    
    # 如果提供了配置文件，加载并覆盖
    config_dict = vars(args)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        config_dict.update(file_config)
    
    # 创建配置对象
    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, save_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置对象
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)