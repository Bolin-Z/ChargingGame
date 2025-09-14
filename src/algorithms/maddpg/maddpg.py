"""
MADDPG核心算法实现

多智能体深度确定性策略梯度算法（Multi-Agent Deep Deterministic Policy Gradient）
基于"中心化训练，去中心化执行"的范式。

架构设计：
- DDPG: 单智能体DDPG实现，包含Actor/Critic网络和目标网络
- MADDPG: 多智能体协调器，管理多个DDPG智能体并实现中心化训练
- ReplayBuffer: 经验回放缓冲区，存储和采样训练经验
- GaussianNoise: 高斯噪音探索策略，用于动作探索

核心特性：
- 集中式训练：Critic网络可以访问全局信息
- 分布式执行：每个智能体独立使用自己的Actor网络决策
- 连续动作空间：适用于连续控制问题
- 多智能体学习：智能体间通过环境交互影响彼此
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Any

from .networks import ActorNetwork, CriticNetwork


class ReplayBuffer:
    """
    经验回放缓冲区
    
    用于存储智能体的经验轨迹，支持随机采样以打破数据相关性。
    使用循环缓冲区，当容量满时自动覆盖最旧的经验。
    """
    
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity (int): 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        """
        添加经验元组到缓冲区
        
        Args:
            experience (tuple): 经验元组 (observations, actions, rewards, next_observations, dones)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        随机采样批次经验
        
        Args:
            batch_size (int): 采样批次大小
            
        Returns:
            list: 采样得到的经验列表
            
        Raises:
            ValueError: 当缓冲区大小小于批次大小时
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小{len(self.buffer)} < 批次大小{batch_size}")
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)


class GaussianNoise:
    """
    高斯噪音探索策略
    
    为动作添加高斯噪音以促进探索，噪音强度随时间指数衰减。
    适用于连续动作空间的探索。
    """
    
    def __init__(self, action_dim, sigma=0.2, sigma_decay=0.9995, min_sigma=0.01):
        """
        初始化高斯噪音
        
        Args:
            action_dim (int): 动作维度
            sigma (float): 初始噪音标准差
            sigma_decay (float): 噪音衰减率，每次调用后sigma *= sigma_decay
            min_sigma (float): 噪音的最小标准差，防止探索完全停止
        """
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
    
    def __call__(self, action):
        """
        为动作添加高斯噪音
        
        Args:
            action (np.ndarray): 原始动作，形状为 (action_dim,)
            
        Returns:
            np.ndarray: 添加噪音后的动作，裁剪到[0,1]范围
        """
        noise = np.random.normal(0, self.sigma, self.action_dim)
        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)
        return np.clip(action + noise, 0.0, 1.0)


class DDPG:
    """
    单智能体DDPG实现
    
    深度确定性策略梯度算法的单智能体版本，包含Actor-Critic架构。
    每个DDPG智能体独立维护自己的网络和优化器。
    """
    
    def __init__(self, agent_id, obs_dim, action_dim, global_obs_dim,
                 actor_lr=0.001, critic_lr=0.001, device='cpu'):
        """
        初始化DDPG智能体
        
        Args:
            agent_id (str): 智能体唯一标识符
            obs_dim (int): 观测空间维度
            action_dim (int): 动作空间维度
            global_obs_dim (int): 全局观测维度（用于Critic网络）
            actor_lr (float): Actor网络学习率
            critic_lr (float): Critic网络学习率
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        self.agent_id = agent_id
        self.device = device
        
        # 创建主网络
        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(global_obs_dim).to(device)
        
        # 创建目标网络（用于稳定训练）
        self.actor_target = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(global_obs_dim).to(device)
        
        # 初始化目标网络参数（复制主网络参数）
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 创建探索噪音
        self.noise = GaussianNoise(action_dim)
    
    def take_action(self, obs, add_noise=True):
        """
        选择动作
        
        Args:
            obs (np.ndarray): 观测向量，形状为 (obs_dim,)
            add_noise (bool): 是否添加探索噪音
            
        Returns:
            np.ndarray: 选择的动作，形状为 (action_dim,)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).detach().cpu().numpy().flatten()
        
        if add_noise:
            action = self.noise(action)
        
        return action
    
    def soft_update(self, tau=0.01):
        """
        软更新目标网络
        
        使用指数移动平均更新目标网络参数：
        θ_target = τ * θ_main + (1 - τ) * θ_target
        
        Args:
            tau (float): 软更新系数，通常为小正数（如0.01）
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class MADDPG:
    """
    多智能体MADDPG协调器
    
    管理多个DDPG智能体，实现"中心化训练，去中心化执行"的MADDPG算法。
    在训练阶段，每个智能体的Critic网络可以访问全局信息；
    在执行阶段，每个智能体只使用自己的Actor网络和局部观测。
    """
    
    def __init__(self, agent_ids, obs_dim, action_dim, global_obs_dim,
                 buffer_capacity=10000, batch_size=64, actor_lr=0.001, critic_lr=0.001, 
                 gamma=0.99, tau=0.01, seed=None, device='cpu'):
        """
        初始化MADDPG协调器
        
        Args:
            agent_ids (list): 智能体ID列表
            obs_dim (int): 单个智能体的观测维度
            action_dim (int): 单个智能体的动作维度
            global_obs_dim (int): 全局观测维度（用于Critic网络）
            buffer_capacity (int): 经验回放缓冲区容量
            batch_size (int): 训练批次大小
            actor_lr (float): Actor网络学习率
            critic_lr (float): Critic网络学习率
            gamma (float): 折扣因子
            tau (float): 软更新系数
            seed (int, optional): 随机种子，用于reproducibility
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # 创建多个DDPG智能体
        self.agents = {}
        for agent_id in agent_ids:
            self.agents[agent_id] = DDPG(
                agent_id=agent_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                global_obs_dim=global_obs_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                device=device
            )
        
        # 创建共享的经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def take_action(self, observations, add_noise=True):
        """
        所有智能体选择动作
        
        Args:
            observations (dict): 智能体观测字典 {agent_id: observation}
            add_noise (bool): 是否添加探索噪音
            
        Returns:
            dict: 智能体动作字典 {agent_id: action}
        """
        actions = {}
        for agent_id in self.agent_ids:
            # 处理观测数据为单一向量
            obs = process_observations(observations[agent_id])
            actions[agent_id] = self.agents[agent_id].take_action(obs, add_noise)
        return actions
    
    def store_experience(self, observations, actions, rewards, next_observations, dones):
        """
        存储经验到回放缓冲区
        
        Args:
            observations (dict): 当前观测
            actions (dict): 执行的动作
            rewards (dict): 获得的奖励
            next_observations (dict): 下一步观测
            dones (dict): 终止标志
        """
        # 归一化奖励（保持博弈等价性）
        normalized_rewards = normalize_rewards(rewards)
        
        experience = (observations, actions, normalized_rewards, next_observations, dones)
        self.replay_buffer.add(experience)
    
    def _get_dynamic_batch_size(self):
        """
        动态调整批次大小，适配环境特点
        
        Returns:
            int: 当前应使用的批次大小
        """
        buffer_size = len(self.replay_buffer)
        if buffer_size < 32:
            return min(8, buffer_size)
        elif buffer_size < 128:
            return min(16, buffer_size)
        else:
            return min(32, buffer_size)
    
    def _parse_batch_experiences(self, batch_experiences):
        """
        解析批次经验数据
        
        Args:
            batch_experiences: 经验批次列表
            
        Returns:
            tuple: (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []
        
        for experience in batch_experiences:
            observations, actions, rewards, next_observations, dones = experience
            batch_obs.append(observations)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_obs.append(next_observations)
            batch_dones.append(dones)
        
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    
    def _update_critic(self, agent, agent_id, batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones):
        """
        更新单个智能体的Critic网络
        
        Args:
            agent: DDPG智能体实例
            agent_id: 智能体ID
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones: 批次数据
            
        Returns:
            float: Critic损失值
        """
        batch_size = len(batch_obs)
        
        # 构建当前状态的全局状态向量
        current_global_states = []
        for i in range(batch_size):
            global_state = organize_global_state(batch_obs[i], batch_actions[i])
            current_global_states.append(global_state)
        current_global_states = torch.FloatTensor(np.array(current_global_states)).to(self.device)
        
        # 构建下一状态的全局状态向量（需要使用目标Actor网络生成下一动作）
        next_global_states = []
        for i in range(batch_size):
            # 为所有智能体生成下一状态的动作（使用目标Actor网络）
            next_actions = {}
            for aid in self.agent_ids:
                next_obs_agent = process_observations(batch_next_obs[i][aid])
                next_obs_tensor = torch.FloatTensor(next_obs_agent).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    next_action = self.agents[aid].actor_target(next_obs_tensor).detach().cpu().numpy().flatten()
                next_actions[aid] = next_action
            
            # 组织下一状态的全局信息
            global_next_state = organize_global_state(batch_next_obs[i], next_actions)
            next_global_states.append(global_next_state)
        next_global_states = torch.FloatTensor(np.array(next_global_states)).to(self.device)
        
        # 获取当前智能体的奖励和终止标志
        current_rewards = torch.FloatTensor([batch_rewards[i][agent_id] for i in range(batch_size)]).unsqueeze(1).to(self.device)
        current_dones = torch.FloatTensor([batch_dones[i][agent_id] for i in range(batch_size)]).unsqueeze(1).to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            target_q_values = agent.critic_target(next_global_states)
            target_q_values = current_rewards + (1 - current_dones) * self.gamma * target_q_values
        
        # 计算当前Q值
        current_q_values = agent.critic(current_global_states)
        
        # 计算Critic损失（MSE）
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 反向传播更新Critic
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, agent, agent_id, batch_obs, batch_actions):
        """
        更新单个智能体的Actor网络
        
        Args:
            agent: DDPG智能体实例
            agent_id: 智能体ID
            batch_obs, batch_actions: 批次观测和动作数据
            
        Returns:
            float: Actor损失值
        """
        batch_size = len(batch_obs)
        
        # 准备观测数据
        obs_list = []
        for i in range(batch_size):
            current_obs = process_observations(batch_obs[i][agent_id])
            obs_list.append(current_obs)
        
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        
        # 生成当前智能体的动作（保持梯度）
        new_actions = agent.actor(obs_tensor)
        
        # 构建全局观测和动作张量（用于Critic网络输入）
        # 这里需要创建一个包含所有智能体信息的张量
        global_states_list = []
        
        for i in range(batch_size):
            # 当前智能体观测（需要展平）
            current_agent_obs = process_observations(batch_obs[i][agent_id])
            
            # 其他智能体观测
            other_obs = []
            for other_id in self.agent_ids:
                if other_id != agent_id:
                    other_agent_obs = process_observations(batch_obs[i][other_id])
                    other_obs.append(other_agent_obs)
            
            # 其他智能体动作
            other_actions = []
            for other_id in self.agent_ids:
                if other_id != agent_id:
                    other_actions.append(batch_actions[i][other_id])
            
            # 组合所有特征
            all_obs = np.concatenate([current_agent_obs] + other_obs)
            all_actions = np.concatenate([new_actions[i].detach().cpu().numpy()] + other_actions)
            
            global_state = np.concatenate([all_obs, all_actions])
            global_states_list.append(global_state)
        
        global_states_tensor = torch.FloatTensor(np.array(global_states_list)).to(self.device)
        
        # 计算Actor损失（策略梯度）
        q_values = agent.critic(global_states_tensor)
        actor_loss = -q_values.mean()  # 最大化Q值
        
        # 反向传播更新Actor
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        
        return actor_loss.item()
    
    def learn(self):
        """
        中心化训练：更新所有智能体的网络
        
        Returns:
            bool: 是否成功执行学习更新
        """
        # 检查是否有足够的经验开始学习
        min_buffer_size = 8  # 最小8个经验即开始学习
        if len(self.replay_buffer) < min_buffer_size:
            return False
        
        # 动态调整批次大小
        current_batch_size = self._get_dynamic_batch_size()
        
        # 从经验回放缓冲区采样
        batch_experiences = self.replay_buffer.sample(current_batch_size)
        
        # 解析批次经验
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self._parse_batch_experiences(batch_experiences)
        
        # 为每个智能体更新Critic网络
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            critic_loss = self._update_critic(agent, agent_id, batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
        
        # 为每个智能体更新Actor网络
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            actor_loss = self._update_actor(agent, agent_id, batch_obs, batch_actions)
        
        # 软更新所有智能体的目标网络
        self.update_all_targets(self.tau)
        
        return True
    
    def update_all_targets(self, tau=0.01):
        """
        更新所有智能体的目标网络
        
        Args:
            tau (float): 软更新系数
        """
        for agent in self.agents.values():
            agent.soft_update(tau)


# 工具函数

def process_observations(observation):
    """
    处理单个智能体观测数据，转换为网络输入向量
    
    将字典格式的观测数据展平为一维向量，用于神经网络输入。
    
    Args:
        observation (dict): 智能体观测，包含:
            - "last_round_all_prices": np.ndarray, 形状 (n_agents, n_periods)
            - "own_charging_flow": np.ndarray, 形状 (n_periods,)
    
    Returns:
        np.ndarray: 展平的观测向量，形状 (obs_dim,)
    """
    last_prices = observation["last_round_all_prices"].flatten()
    own_flow = observation["own_charging_flow"].flatten()
    return np.concatenate([last_prices, own_flow])


def organize_global_state(observations, actions):
    """
    组织全局状态信息，用于Critic网络输入
    
    将所有智能体的观测和动作信息组织成集中式输入向量。
    假设每个observation是字典格式，包含可展平的数组数据。
    
    Args:
        observations (dict): 所有智能体的观测 {agent_id: observation}
        actions (dict): 所有智能体的动作 {agent_id: action}
    
    Returns:
        np.ndarray: 全局状态向量
    """
    # 提取并展平所有观测信息
    obs_features = []
    sorted_agents = sorted(observations.keys())  # 确保顺序一致
    
    for agent_id in sorted_agents:
        agent_obs = observations[agent_id]
        if isinstance(agent_obs, dict):
            # 如果观测是字典，展平所有数值
            for key, value in agent_obs.items():
                obs_features.append(np.array(value).flatten())
        else:
            # 如果观测是数组，直接展平
            obs_features.append(np.array(agent_obs).flatten())
    
    # 提取并展平所有动作信息
    action_features = []
    for agent_id in sorted_agents:
        action = actions[agent_id]
        action_features.append(np.array(action).flatten())
    
    # 组织全局状态向量
    all_features = obs_features + action_features
    global_state = np.concatenate(all_features)
    
    return global_state


def normalize_rewards(rewards):
    """
    奖励归一化：当轮最大值正仿射变换
    
    使用当轮最大奖励进行归一化，保持博弈的纳什均衡不变。
    正仿射变换确保博弈论意义下的等价性。
    
    Args:
        rewards (dict): 原始奖励 {agent_id: reward}
    
    Returns:
        dict: 归一化奖励 {agent_id: normalized_reward}
    """
    max_reward = max(rewards.values()) if rewards.values() else 0
    if max_reward > 0:
        return {agent: reward / max_reward for agent, reward in rewards.items()}
    else:
        return {agent: 0.0 for agent in rewards.keys()}