"""
IDDPG核心算法实现

独立深度确定性策略梯度算法（Independent Deep Deterministic Policy Gradient）
基于"完全去中心化训练"的范式。

架构设计：
- DDPG: 单智能体DDPG实现，包含Actor/Critic网络和目标网络
- IndependentDDPG: 多智能体管理器，每个智能体完全独立训练
- ReplayBuffer: 经验回放缓冲区，存储和采样训练经验
- GaussianNoise: 高斯噪音探索策略，用于动作探索

核心特性：
- 完全独立训练：每个agent维护独立的经验回放和网络更新
- 局部状态使用：Critic仅使用自身历史和全局价格信息
- 与MADDPG公平对比：除网络输入维度外，所有参数保持一致
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
    与MADDPG完全相同，确保公平对比。
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
            experience (tuple): 经验元组 (observation, action, reward, next_observation, done)
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
    与MADDPG完全相同，确保公平对比。
    """

    def __init__(self, action_dim, sigma, sigma_decay, min_sigma):
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
    每个DDPG智能体独立维护自己的网络、优化器和经验回放。
    关键区别：Critic使用局部状态而非全局状态，维度由local_state_dim参数指定。
    """

    def __init__(self, agent_id, obs_dim, action_dim, local_state_dim,
                 actor_lr=0.001, critic_lr=0.001, device='cpu',
                 actor_hidden_sizes=(64, 64), critic_hidden_sizes=(128, 64),
                 noise_sigma=0.2, noise_decay=0.9995, min_noise=0.01):
        """
        初始化DDPG智能体

        Args:
            agent_id (str): 智能体唯一标识符
            obs_dim (int): 观测空间维度
            action_dim (int): 动作空间维度
            local_state_dim (int): 局部状态维度（用于Critic网络）
            actor_lr (float): Actor网络学习率
            critic_lr (float): Critic网络学习率
            device (str): 计算设备 ('cpu' 或 'cuda')
            actor_hidden_sizes (tuple): Actor网络隐藏层配置
            critic_hidden_sizes (tuple): Critic网络隐藏层配置
            noise_sigma (float): 探索噪音初始标准差
            noise_decay (float): 噪音衰减率
            min_noise (float): 最小噪音标准差
        """
        self.agent_id = agent_id
        self.device = device

        # 创建主网络（使用配置化的隐藏层）
        self.actor = ActorNetwork(obs_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic = CriticNetwork(local_state_dim, critic_hidden_sizes).to(device)

        # 创建目标网络（用于稳定训练）
        self.actor_target = ActorNetwork(obs_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic_target = CriticNetwork(local_state_dim, critic_hidden_sizes).to(device)

        # 初始化目标网络参数（复制主网络参数）
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 创建探索噪音（使用配置化参数）
        self.noise = GaussianNoise(action_dim, noise_sigma, noise_decay, min_noise)

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


class IndependentDDPG:
    """
    独立DDPG算法管理器

    管理多个DDPG智能体，每个智能体完全独立训练。
    关键特征：
    - 每个agent维护独立的ReplayBuffer
    - 完全去中心化的学习过程
    - Critic仅使用局部状态信息
    """

    def __init__(self, agent_ids, obs_dim, action_dim, local_state_dim,
                 buffer_capacity=10000, max_batch_size=64, actor_lr=0.001, critic_lr=0.001,
                 gamma=0.95, tau=0.01, seed=None, device='cpu',
                 actor_hidden_sizes=(64, 64), critic_hidden_sizes=(128, 64),
                 noise_sigma=0.2, noise_decay=0.9995, min_noise=0.01):
        """
        初始化独立DDPG算法

        Args:
            agent_ids (list): 智能体ID列表
            obs_dim (int): 单个智能体的观测维度
            action_dim (int): 单个智能体的动作维度
            local_state_dim (int): 局部状态维度（用于Critic网络）
            buffer_capacity (int): 经验回放缓冲区容量
            max_batch_size (int): 最大批次大小，实际会根据缓冲区大小动态调整
            actor_lr (float): Actor网络学习率
            critic_lr (float): Critic网络学习率
            gamma (float): 折扣因子
            tau (float): 软更新系数
            seed (int, optional): 随机种子，用于reproducibility
            device (str): 计算设备 ('cpu' 或 'cuda')
            actor_hidden_sizes (tuple): Actor网络隐藏层配置
            critic_hidden_sizes (tuple): Critic网络隐藏层配置
            noise_sigma (float): 探索噪音初始标准差
            noise_decay (float): 噪音衰减率
            min_noise (float): 最小噪音标准差
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
        self.batch_size = max_batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # 创建多个独立的DDPG智能体
        self.agents = {}
        for agent_id in agent_ids:
            self.agents[agent_id] = DDPG(
                agent_id=agent_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                local_state_dim=local_state_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                device=device,
                actor_hidden_sizes=actor_hidden_sizes,
                critic_hidden_sizes=critic_hidden_sizes,
                noise_sigma=noise_sigma,
                noise_decay=noise_decay,
                min_noise=min_noise
            )

        # 为每个智能体创建独立的经验回放缓冲区
        self.replay_buffers = {}
        for agent_id in agent_ids:
            self.replay_buffers[agent_id] = ReplayBuffer(buffer_capacity)

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
        存储经验到各智能体的独立回放缓冲区

        Args:
            observations (dict): 当前观测
            actions (dict): 执行的动作
            rewards (dict): 获得的奖励
            next_observations (dict): 下一步观测
            dones (dict): 终止标志
        """
        # 归一化奖励（保持博弈等价性）
        normalized_rewards = normalize_rewards(rewards)

        # 为每个agent独立存储经验
        for agent_id in self.agent_ids:
            experience = (
                observations[agent_id],
                actions[agent_id],
                normalized_rewards[agent_id],
                next_observations[agent_id],
                dones[agent_id]
            )
            self.replay_buffers[agent_id].add(experience)

    def _get_dynamic_batch_size(self, agent_id):
        """
        动态调整批次大小，各agent独立调整

        使用配置的max_batch_size作为上限，根据缓冲区大小动态调整

        Args:
            agent_id (str): 智能体ID

        Returns:
            int: 当前应使用的批次大小
        """
        buffer_size = len(self.replay_buffers[agent_id])
        max_batch_size = self.batch_size  # 使用配置的最大batch_size

        if buffer_size < max_batch_size // 2:
            # 缓冲区较小时，使用较小的batch_size
            return min(max_batch_size // 4, buffer_size)
        elif buffer_size < max_batch_size:
            # 缓冲区中等时，使用中等的batch_size
            return min(max_batch_size // 2, buffer_size)
        else:
            # 缓冲区足够时，使用配置的最大batch_size
            return min(max_batch_size, buffer_size)

    def _update_critic(self, agent, agent_id, batch_experiences):
        """
        更新单个智能体的Critic网络

        Args:
            agent: DDPG智能体实例
            agent_id: 智能体ID
            batch_experiences: 批次经验列表

        Returns:
            float: Critic损失值
        """
        batch_size = len(batch_experiences)

        # 解析批次经验
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []

        for exp in batch_experiences:
            obs, action, reward, next_obs, done = exp
            batch_obs.append(obs)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_obs.append(next_obs)
            batch_dones.append(done)

        # 构建当前状态的局部状态向量
        current_local_states = []
        for i in range(batch_size):
            local_state = organize_local_state(batch_obs[i], batch_actions[i])
            current_local_states.append(local_state)
        current_local_states = torch.FloatTensor(np.array(current_local_states)).to(self.device)

        # 构建下一状态的局部状态向量（使用目标Actor网络）
        next_local_states = []
        for i in range(batch_size):
            # 为当前智能体生成下一状态的动作（使用目标Actor网络）
            next_obs_processed = process_observations(batch_next_obs[i])
            next_obs_tensor = torch.FloatTensor(next_obs_processed).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_action = agent.actor_target(next_obs_tensor).detach().cpu().numpy().flatten()

            # 组织下一状态的局部信息
            local_next_state = organize_local_state(batch_next_obs[i], next_action)
            next_local_states.append(local_next_state)
        next_local_states = torch.FloatTensor(np.array(next_local_states)).to(self.device)

        # 获取当前智能体的奖励和终止标志
        current_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(self.device)
        current_dones = torch.FloatTensor(batch_dones).unsqueeze(1).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            target_q_next = agent.critic_target(next_local_states)
            target_q = current_rewards + self.gamma * target_q_next * (1 - current_dones)

        # 计算当前Q值
        current_q = agent.critic(current_local_states)

        # 计算Critic损失
        critic_loss = nn.MSELoss()(current_q, target_q)

        # 更新Critic网络
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, agent, agent_id, batch_experiences):
        """
        更新单个智能体的Actor网络

        Args:
            agent: DDPG智能体实例
            agent_id: 智能体ID
            batch_experiences: 批次经验列表

        Returns:
            float: Actor损失值
        """
        batch_size = len(batch_experiences)

        # 解析批次经验，获取观测
        batch_obs = [exp[0] for exp in batch_experiences]

        # 处理观测数据
        processed_obs = []
        for obs in batch_obs:
            obs_vec = process_observations(obs)
            processed_obs.append(obs_vec)
        obs_tensor = torch.FloatTensor(np.array(processed_obs)).to(self.device)

        # 生成当前策略的动作
        current_actions = agent.actor(obs_tensor)

        # 构建局部状态用于Critic评估
        local_states = []
        for i in range(batch_size):
            action = current_actions[i].detach().cpu().numpy()
            local_state = organize_local_state(batch_obs[i], action)
            local_states.append(local_state)
        local_states_tensor = torch.FloatTensor(np.array(local_states)).to(self.device)

        # Actor损失：最大化Q值（梯度上升）
        actor_loss = -agent.critic(local_states_tensor).mean()

        # 更新Actor网络
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        return actor_loss.item()

    def learn(self):
        """
        所有智能体独立学习更新

        每个agent从自己的独立Buffer采样和更新网络。
        完全去中心化的学习过程。

        Returns:
            bool: 是否成功进行了学习更新
        """
        # 为每个智能体独立学习
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]

            # 检查是否有足够经验
            buffer_size = len(self.replay_buffers[agent_id])
            if buffer_size < 8:  # 最小要求8个经验
                continue

            # 动态调整批次大小
            batch_size = self._get_dynamic_batch_size(agent_id)

            try:
                # 从该agent的独立Buffer采样
                batch_experiences = self.replay_buffers[agent_id].sample(batch_size)

                # 更新Critic网络
                critic_loss = self._update_critic(agent, agent_id, batch_experiences)

                # 更新Actor网络
                actor_loss = self._update_actor(agent, agent_id, batch_experiences)

                # 软更新目标网络
                agent.soft_update(self.tau)

            except ValueError:
                # 缓冲区不足，跳过该agent
                continue

        return True


# 工具函数

def process_observations(observation):
    """
    处理单个智能体观测数据，转换为网络输入向量

    将字典格式的观测数据展平为一维向量，用于Actor网络输入。
    与MADDPG完全相同，确保公平对比。

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


def organize_local_state(observation, action):
    """
    组织单个智能体的局部状态信息

    将观测和动作组合成Critic网络的输入状态。
    局部状态包含：可观测的全局价格历史 + 自身流量历史 + 当前动作。

    Args:
        observation (dict): 单个智能体的观测字典
            - "last_round_all_prices": np.ndarray, 形状 (n_agents, n_periods)
            - "own_charging_flow": np.ndarray, 形状 (n_periods,)
        action (np.ndarray): 单个智能体的动作，形状 (action_dim,)

    Returns:
        np.ndarray: 局部状态向量，形状 (local_state_dim,)
    """
    # 1. 全局价格历史（可观测信息）
    global_prices = observation["last_round_all_prices"].flatten()

    # 2. 自身充电流量（局部信息）
    own_flow = observation["own_charging_flow"].flatten()

    # 3. 自身当前动作（评估对象）
    own_action = action.flatten()

    return np.concatenate([global_prices, own_flow, own_action])


def normalize_rewards(rewards):
    """
    奖励归一化：当轮最大值正仿射变换

    使用当轮最大奖励进行归一化，保持博弈的纳什均衡不变。
    正仿射变换确保博弈论意义下的等价性。
    与MADDPG完全相同，确保公平对比。

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
