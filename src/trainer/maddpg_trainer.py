"""
MADDPG算法训练器

实现基于MADDPG的充电站价格博弈训练逻辑
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_trainer import BaseTrainer
from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from src.algorithms.maddpg import MADDPG
from src.utils.config import TrainingConfig


class MADDPGTrainer(BaseTrainer):
    """
    MADDPG算法训练器
    
    实现基于MADDPG的多智能体强化学习训练流程
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """
        初始化MADDPG训练器
        
        Args:
            config: 训练配置
            logger: 日志记录器
        """
        super().__init__(config, logger)
        
        # 算法特定属性
        self.env = None
        self.maddpg = None
        
        # 维度信息（将在setup中计算）
        self.obs_dim = None
        self.action_dim = None
        self.global_obs_dim = None
        self.agent_ids = None
    
    def setup(self):
        """训练前的设置"""
        self.logger.info("初始化MADDPG训练器...")
        
        # 1. 创建环境
        self._create_environment()
        
        # 2. 计算维度信息
        self._calculate_dimensions()
        
        # 3. 创建MADDPG智能体
        self._create_maddpg_agents()
        
        self.logger.info("MADDPG训练器初始化完成")
    
    def _create_environment(self):
        """创建充电站博弈环境"""
        self.logger.info("创建充电站博弈环境...")
        
        # 构建网络数据路径
        network_path = os.path.join(os.getcwd(), self.config.network_dir)
        
        self.env = EVCSChargingGameEnv(
            network_dir=network_path,
            network_name=self.config.network_name,
            random_seed=self.config.seed,
            max_steps=self.config.max_steps_per_episode,
            convergence_threshold=self.config.convergence_threshold
        )
        
        # 获取智能体ID列表
        self.agent_ids = self.env.agents
        self.logger.info(f"环境创建完成，智能体: {self.agent_ids}")
    
    def _calculate_dimensions(self):
        """计算观测、动作和全局状态维度"""
        # 获取样本观测和动作空间
        sample_agent = self.agent_ids[0]
        obs_space = self.env.observation_space(sample_agent)
        action_space = self.env.action_space(sample_agent)
        
        # 计算观测维度
        # 观测包含: last_round_all_prices (n_agents, n_periods) + own_charging_flow (n_periods,)
        n_agents = len(self.agent_ids)
        n_periods = self.env.n_periods
        
        # last_round_all_prices: (n_agents * n_periods) + own_charging_flow: (n_periods)
        self.obs_dim = n_agents * n_periods + n_periods
        
        # 动作维度：每个智能体输出n_periods个价格
        self.action_dim = n_periods
        
        # 全局观测维度（用于Critic网络）
        # 根据organize_global_state函数的逻辑：
        # 所有智能体的观测信息 + 所有智能体的动作信息
        # 每个智能体观测: last_round_all_prices (n_agents*n_periods) + own_charging_flow (n_periods)
        # 所有智能体观测总计: n_agents * (n_agents*n_periods + n_periods)
        # 所有智能体动作总计: n_agents * n_periods
        self.global_obs_dim = n_agents * (n_agents * n_periods + n_periods) + n_agents * n_periods
        
        self.logger.info(f"维度信息 - 观测: {self.obs_dim}, 动作: {self.action_dim}, 全局: {self.global_obs_dim}")
    
    def _create_maddpg_agents(self):
        """创建MADDPG智能体"""
        self.logger.info("创建MADDPG智能体...")
        
        self.maddpg = MADDPG(
            agent_ids=self.agent_ids,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            global_obs_dim=self.global_obs_dim,
            buffer_capacity=self.config.buffer_capacity,
            batch_size=64,  # 将使用动态调整
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            gamma=self.config.gamma,
            tau=self.config.tau,
            seed=self.config.seed,
            device=self.device
        )
        
        self.logger.info(f"MADDPG智能体创建完成，{len(self.agent_ids)}个智能体")
    
    def train_episode(self) -> Dict[str, Any]:
        """
        训练单个episode
        
        Returns:
            Dict[str, Any]: episode训练结果
        """
        # 重置环境
        observations, _ = self.env.reset(seed=self.config.seed + self.current_episode)
        
        episode_rewards = {agent: 0.0 for agent in self.agent_ids}
        episode_length = 0
        converged = False
        
        for step in range(self.config.max_steps_per_episode):
            # 智能体选择动作（训练时添加噪音）
            actions = self.maddpg.take_action(observations, add_noise=True)
            
            # 执行动作
            next_observations, rewards, terminations, truncations, _ = self.env.step(actions)
            
            # 检查是否结束（收敛或截断）
            done = any(terminations.values()) or any(truncations.values())
            dones = {agent: done for agent in self.agent_ids}
            
            # 存储经验
            self.maddpg.store_experience(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones
            )
            
            # 学习（如果有足够经验）
            learning_success = self.maddpg.learn()
            
            # 累积奖励
            for agent in self.agent_ids:
                episode_rewards[agent] += rewards[agent]
            
            episode_length += 1
            
            # 检查是否收敛到纳什均衡
            if any(terminations.values()):
                converged = True
                self.logger.debug(f"Episode {self.current_episode} 在第{step+1}步收敛到纳什均衡")
                break
            
            # 更新观测
            observations = next_observations
            
            # 如果被截断也要结束
            if any(truncations.values()):
                break
        
        # 计算episode总奖励
        total_reward = sum(episode_rewards.values())
        
        episode_result = {
            'total_reward': total_reward,
            'agent_rewards': episode_rewards,
            'episode_length': episode_length,
            'converged': converged,
            'learning_success': learning_success
        }
        
        return episode_result
    
    def _get_agent_ids(self) -> list:
        """获取智能体ID列表"""
        return self.agent_ids
    
    def _reset_environment(self) -> tuple:
        """重置环境"""
        return self.env.reset(seed=self.config.seed + self.current_episode)
    
    def _train_single_step(self, observations) -> Dict[str, Any]:
        """训练单个step"""
        # 智能体选择动作（训练时添加噪音）
        actions = self.maddpg.take_action(observations, add_noise=True)
        
        # 执行动作
        next_observations, rewards, terminations, truncations, _ = self.env.step(actions)
        
        # 检查是否结束（收敛或截断）
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        dones = {agent: terminated or truncated for agent in self.agent_ids}
        
        # 存储经验
        self.maddpg.store_experience(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones
        )
        
        # 学习（如果有足够经验）
        learning_success = self.maddpg.learn()
        
        return {
            'next_observations': next_observations,
            'rewards': rewards,
            'terminated': terminated,
            'truncated': truncated,
            'learning_success': learning_success
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        评估算法性能
        
        Args:
            num_episodes: 评估episode数量
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始评估，运行{num_episodes}个episodes...")
        
        eval_rewards = []
        eval_lengths = []
        eval_convergences = []
        
        for episode in range(num_episodes):
            # 重置环境
            observations, _ = self.env.reset(seed=self.config.seed + 10000 + episode)
            
            episode_reward = 0
            episode_length = 0
            converged = False
            
            for step in range(self.config.max_steps_per_episode):
                # 智能体选择动作（评估时不添加噪音）
                actions = self.maddpg.take_action(observations, add_noise=False)
                
                # 执行动作
                next_observations, rewards, terminations, truncations, _ = self.env.step(actions)
                
                # 累积奖励
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # 检查是否收敛
                if any(terminations.values()):
                    converged = True
                    break
                
                # 更新观测
                observations = next_observations
                
                # 如果被截断也要结束
                if any(truncations.values()):
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_convergences.append(converged)
        
        # 计算统计指标
        evaluation_results = {
            'average_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'average_length': np.mean(eval_lengths),
            'convergence_rate': np.mean(eval_convergences),
            'total_episodes': num_episodes
        }
        
        self.logger.info(f"评估完成 - 平均奖励: {evaluation_results['average_reward']:.2f}, "
                        f"收敛率: {evaluation_results['convergence_rate']:.2%}")
        
        return evaluation_results
    
    def save_model(self, path: str):
        """保存MADDPG模型"""
        if self.maddpg is None:
            self.logger.warning("尝试保存模型，但MADDPG未初始化")
            return
        
        # 创建保存目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存所有智能体的网络参数
        save_dict = {
            'config': self.config.__dict__,
            'agent_ids': self.agent_ids,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'global_obs_dim': self.global_obs_dim,
            'current_episode': self.current_episode,
            'agents': {}
        }
        
        for agent_id in self.agent_ids:
            agent = self.maddpg.agents[agent_id]
            save_dict['agents'][agent_id] = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_target_state_dict': agent.actor_target.state_dict(),
                'critic_target_state_dict': agent.critic_target.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()
            }
        
        torch.save(save_dict, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载MADDPG模型"""
        if not os.path.exists(path):
            self.logger.error(f"模型文件不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 恢复智能体网络参数
        for agent_id in self.agent_ids:
            if agent_id in checkpoint['agents']:
                agent = self.maddpg.agents[agent_id]
                agent_data = checkpoint['agents'][agent_id]
                
                agent.actor.load_state_dict(agent_data['actor_state_dict'])
                agent.critic.load_state_dict(agent_data['critic_state_dict'])
                agent.actor_target.load_state_dict(agent_data['actor_target_state_dict'])
                agent.critic_target.load_state_dict(agent_data['critic_target_state_dict'])
                agent.actor_optimizer.load_state_dict(agent_data['actor_optimizer_state_dict'])
                agent.critic_optimizer.load_state_dict(agent_data['critic_optimizer_state_dict'])
        
        # 恢复训练状态
        if 'current_episode' in checkpoint:
            self.current_episode = checkpoint['current_episode']
        
        self.logger.info(f"模型已从{path}加载")
    
    def get_nash_equilibrium(self) -> Dict[str, Any]:
        """
        获取当前策略对应的纳什均衡解
        
        Returns:
            Dict[str, Any]: 纳什均衡策略和收益
        """
        if self.maddpg is None:
            self.logger.error("MADDPG未初始化，无法获取纳什均衡")
            return {}
        
        # 运行一次评估获取当前策略下的均衡
        observations, _ = self.env.reset(seed=self.config.seed)
        
        # 使用确定性策略（无噪音）
        actions = self.maddpg.take_action(observations, add_noise=False)
        
        # 将归一化动作转换为实际价格
        actual_prices = self.env.actions_to_prices(actions)
        
        # 运行一步获取流量和收益
        _, rewards, _, _, _ = self.env.step(actions)
        
        nash_equilibrium = {
            'optimal_strategies': {
                agent_id: actual_prices[agent_idx].tolist() 
                for agent_idx, agent_id in enumerate(self.agent_ids)
            },
            'equilibrium_rewards': rewards,
            'normalized_actions': actions
        }
        
        return nash_equilibrium