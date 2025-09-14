"""
训练器抽象基类

定义所有算法训练器的统一接口，便于算法对比和实验管理
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm

from src.utils.config import TrainingConfig


class BaseTrainer(ABC):
    """
    训练器抽象基类
    
    所有算法训练器都应继承此类并实现抽象方法
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """
        初始化训练器基类
        
        Args:
            config: 训练配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # 训练状态
        self.current_episode = 0
        self.global_step = 0
        
        # 训练指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = {'actor': [], 'critic': []}
        self.convergence_episodes = []  # 记录收敛到纳什均衡的episode
        
        # 设备设置
        self.device = self._setup_device()
        
        # 创建输出目录
        self.output_dir = self._create_output_directories()
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.config.device
        
        self.logger.info(f"使用计算设备: {device}")
        if device == 'cuda':
            self.logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        
        return device
    
    def _create_output_directories(self) -> str:
        """创建输出目录结构"""
        output_dir = self.config.output_dir
        
        # 创建子目录
        dirs_to_create = [
            'logs',          # 日志文件
            'models',        # 保存的模型
            'results',       # 训练结果数据
            'plots'          # 图表和可视化
        ]
        
        for subdir in dirs_to_create:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        return output_dir
    
    @abstractmethod
    def setup(self):
        """
        训练前的设置（抽象方法）
        
        子类应实现：
        - 环境初始化
        - 算法/智能体创建
        - 其他必要的初始化工作
        """
        pass
    
    @abstractmethod
    def train_episode(self) -> Dict[str, Any]:
        """
        训练单个episode（抽象方法）
        
        Returns:
            Dict[str, Any]: episode训练结果统计
        """
        pass
    
    @abstractmethod
    def _get_agent_ids(self) -> list:
        """获取智能体ID列表（抽象方法）"""
        pass
    
    @abstractmethod
    def _reset_environment(self) -> tuple:
        """重置环境（抽象方法）"""
        pass
    
    @abstractmethod
    def _train_single_step(self, observations) -> Dict[str, Any]:
        """训练单个step（抽象方法）"""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        评估算法性能（抽象方法）
        
        Args:
            num_episodes: 评估使用的episode数量
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        pass
    
    def train(self) -> Dict[str, Any]:
        """
        主训练循环（基于step级别训练）
        
        Returns:
            Dict[str, Any]: 完整训练结果
        """
        self.logger.info("开始MADDPG训练...")
        self.setup()
        
        # 计算总的训练步数
        total_steps = self.config.max_episodes * self.config.max_steps_per_episode
        current_step = 0
        episode = 0
        
        # Step级别的训练循环（重构版）
        with tqdm(total=total_steps, desc="MADDPG训练") as pbar:
            # 初始化训练状态
            observations = None
            episode_rewards = {agent: 0.0 for agent in self._get_agent_ids()}
            episode_length = 0
            converged = False
            
            while current_step < total_steps:
                # 开始新的episode
                if observations is None:
                    episode += 1
                    self.current_episode = episode - 1
                    observations, _ = self._reset_environment()
                    episode_rewards = {agent: 0.0 for agent in self._get_agent_ids()}
                    episode_length = 0
                    converged = False
                
                # 执行单个step
                step_result = self._train_single_step(observations)
                
                # 更新状态
                observations = step_result['next_observations']
                rewards = step_result['rewards']
                terminated = step_result['terminated']
                truncated = step_result['truncated']
                learning_success = step_result['learning_success']
                
                # 累积奖励
                for agent in episode_rewards:
                    episode_rewards[agent] += rewards[agent]
                episode_length += 1
                current_step += 1
                
                # 立即更新进度条
                pbar.update(1)
                self._update_step_progress_bar(pbar, {
                    'total_reward': sum(episode_rewards.values()),
                    'episode_length': episode_length,
                    'converged': terminated,
                    'learning_success': learning_success
                }, episode, current_step, total_steps)
                
                # 检查episode是否结束
                if terminated or truncated:
                    converged = terminated
                    
                    # 记录episode指标
                    episode_result = {
                        'total_reward': sum(episode_rewards.values()),
                        'agent_rewards': episode_rewards,
                        'episode_length': episode_length,
                        'converged': converged,
                        'learning_success': learning_success
                    }
                    self._record_episode_metrics(episode_result)
                    
                    # 定期保存（基于episode）
                    if episode % self.config.save_interval == 0:
                        self._save_checkpoint(episode)
                    
                    # 重置observations以开始新episode
                    observations = None
                
                # 早停检查（可选）
                if self._should_early_stop():
                    self.logger.info(f"在第{episode}个episode达到早停条件")
                    break
        
        # 训练完成后的处理
        final_results = self._finalize_training()
        
        self.logger.info("MADDPG训练完成!")
        return final_results
    
    def _record_episode_metrics(self, episode_result: Dict[str, Any]):
        """记录episode指标"""
        if 'total_reward' in episode_result:
            self.episode_rewards.append(episode_result['total_reward'])
        if 'episode_length' in episode_result:
            self.episode_lengths.append(episode_result['episode_length'])
        if 'converged' in episode_result and episode_result['converged']:
            self.convergence_episodes.append(self.current_episode)
    
    def _update_progress_bar(self, pbar: tqdm, episode_result: Dict[str, Any]):
        """更新进度条显示（旧版本，episode级别）"""
        desc = f"MADDPG训练 Episode {self.current_episode+1}"
        if 'total_reward' in episode_result:
            desc += f" | 总奖励: {episode_result['total_reward']:.1f}"
        if 'episode_length' in episode_result:
            desc += f" | 步长: {episode_result['episode_length']}"
        if 'converged' in episode_result:
            desc += f" | 均衡: {'✓' if episode_result['converged'] else '✗'}"
        if 'learning_success' in episode_result:
            desc += f" | 学习: {'✓' if episode_result['learning_success'] else '✗'}"
        
        # 显示累计收敛率
        if len(self.convergence_episodes) > 0:
            convergence_rate = len(self.convergence_episodes) / (self.current_episode + 1)
            desc += f" | 收敛率: {convergence_rate:.1%}"
        
        pbar.set_description(desc)
    
    def _update_step_progress_bar(self, pbar: tqdm, episode_result: Dict[str, Any], 
                                  episode: int, current_step: int, total_steps: int):
        """更新step级别进度条显示"""
        desc = f"MADDPG训练 Step {current_step}/{total_steps}"
        desc += f" | Episode {episode}"
        
        if 'total_reward' in episode_result:
            desc += f" | 奖励: {episode_result['total_reward']:.1f}"
        if 'episode_length' in episode_result:
            desc += f" | 此轮步数: {episode_result['episode_length']}"
        if 'converged' in episode_result:
            desc += f" | 均衡: {'✓' if episode_result['converged'] else '✗'}"
        if 'learning_success' in episode_result:
            desc += f" | 学习: {'✓' if episode_result['learning_success'] else '✗'}"
        
        # 显示累计收敛率
        if len(self.convergence_episodes) > 0:
            convergence_rate = len(self.convergence_episodes) / episode
            desc += f" | 收敛率: {convergence_rate:.1%}"
        
        pbar.set_description(desc)
    
    def _should_early_stop(self) -> bool:
        """检查是否应该早停（子类可重写）"""
        # 基础实现：连续多次收敛可考虑早停
        if len(self.convergence_episodes) >= 10:
            recent_convergences = self.convergence_episodes[-10:]
            if all(recent_convergences[i+1] - recent_convergences[i] == 1 
                   for i in range(len(recent_convergences)-1)):
                return True
        return False
    
    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, 'models', f'checkpoint_episode_{episode}.pt')
        self.save_model(checkpoint_path)
        
        # 保存训练指标
        metrics_path = os.path.join(self.output_dir, 'results', f'metrics_episode_{episode}.json')
        metrics = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'convergence_episodes': self.convergence_episodes
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _finalize_training(self) -> Dict[str, Any]:
        """完成训练后的最终处理"""
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, 'models', 'final_model.pt')
        self.save_model(final_model_path)
        
        # 计算训练统计
        results = {
            'total_episodes': self.current_episode + 1,
            'total_convergences': len(self.convergence_episodes),
            'convergence_rate': len(self.convergence_episodes) / (self.current_episode + 1),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'final_model_path': final_model_path
        }
        
        # 保存最终结果
        results_path = os.path.join(self.output_dir, 'results', 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    @abstractmethod
    def save_model(self, path: str):
        """保存模型（抽象方法）"""
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """加载模型（抽象方法）"""
        pass
    
    def get_algorithm_name(self) -> str:
        """获取算法名称"""
        return self.__class__.__name__.replace('Trainer', '')