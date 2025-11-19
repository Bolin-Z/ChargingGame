"""
MF-DDPG核心算法实现

Mean Field Deep Deterministic Policy Gradient (MF-DDPG)
基于Mean Field近似的多智能体深度强化学习算法。

核心思想：
- 通过Mean Field近似将其他agent的集体行为压缩为低维度信息
- 完全独立训练：每个agent独立学习，无中心化协调
- 信息压缩：用平均场信息近似其他agent行为

关键特性：
- 最佳扩展性：不受agent数量影响，O(1)复杂度
- 理论简洁：符合去中心化学习的纯粹性
- 计算高效：最小的网络规模和参数量
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
    与MADDPG/IDDPG完全相同，确保公平对比。
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
            experience (tuple): 经验元组 (mf_state, action, reward, next_mf_state, done)
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
    与MADDPG/IDDPG完全相同，确保公平对比。
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

    基于Mean Field近似的DDPG智能体，包含Actor-Critic架构。
    每个智能体基于压缩的Mean Field状态独立学习。
    关键区别：使用Mean Field状态而非局部完整状态，实现最优的状态压缩。
    """

    def __init__(self, agent_id, mf_state_dim, action_dim, critic_state_dim,
                 actor_lr=0.001, critic_lr=0.001, device='cpu',
                 actor_hidden_sizes=(64, 64), critic_hidden_sizes=(128, 64),
                 noise_sigma=0.2, noise_decay=0.9995, min_noise=0.01):
        """
        初始化DDPG智能体

        Args:
            agent_id (str): 智能体唯一标识符
            mf_state_dim (int): Mean Field状态维度（用于Actor网络）
            action_dim (int): 动作空间维度
            critic_state_dim (int): Critic状态维度（mf_state + action）
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

        self.actor = ActorNetwork(mf_state_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic = CriticNetwork(critic_state_dim, critic_hidden_sizes).to(device)

        self.actor_target = ActorNetwork(mf_state_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic_target = CriticNetwork(critic_state_dim, critic_hidden_sizes).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.noise = GaussianNoise(action_dim, noise_sigma, noise_decay, min_noise)

    def take_action(self, mf_state, add_noise=True):
        """
        选择动作

        Args:
            mf_state (np.ndarray): Mean Field状态向量，形状为 (mf_state_dim,)
            add_noise (bool): 是否添加探索噪音

        Returns:
            np.ndarray: 选择的动作，形状为 (action_dim,)
        """
        mf_state_tensor = torch.FloatTensor(mf_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(mf_state_tensor).detach().cpu().numpy().flatten()

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


class MFDDPG:
    """
    MF-DDPG算法管理器

    管理多个DDPG智能体，每个智能体基于Mean Field状态独立训练。
    关键特征：
    - 每个agent维护独立的ReplayBuffer
    - 完全去中心化的学习过程
    - 使用Mean Field近似实现状态压缩
    """

    def __init__(self, agent_ids, mf_state_dim, action_dim, critic_state_dim,
                 buffer_capacity=10000, max_batch_size=64, actor_lr=0.001, critic_lr=0.001,
                 gamma=0.95, tau=0.01, seed=None, device='cpu',
                 actor_hidden_sizes=(64, 64), critic_hidden_sizes=(128, 64),
                 noise_sigma=0.2, noise_decay=0.9995, min_noise=0.01):
        """
        初始化MF-DDPG算法

        Args:
            agent_ids (list): 智能体ID列表
            mf_state_dim (int): Mean Field状态维度
            action_dim (int): 动作维度
            critic_state_dim (int): Critic状态维度
            buffer_capacity (int): 经验回放缓冲区容量
            max_batch_size (int): 最大批次大小
            actor_lr (float): Actor网络学习率
            critic_lr (float): Critic网络学习率
            gamma (float): 折扣因子
            tau (float): 软更新系数
            seed (int, optional): 随机种子
            device (str): 计算设备
            actor_hidden_sizes (tuple): Actor网络隐藏层配置
            critic_hidden_sizes (tuple): Critic网络隐藏层配置
            noise_sigma (float): 探索噪音初始标准差
            noise_decay (float): 噪音衰减率
            min_noise (float): 最小噪音标准差
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.mf_state_dim = mf_state_dim
        self.action_dim = action_dim
        self.batch_size = max_batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.agents = {}
        for agent_id in agent_ids:
            self.agents[agent_id] = DDPG(
                agent_id=agent_id,
                mf_state_dim=mf_state_dim,
                action_dim=action_dim,
                critic_state_dim=critic_state_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                device=device,
                actor_hidden_sizes=actor_hidden_sizes,
                critic_hidden_sizes=critic_hidden_sizes,
                noise_sigma=noise_sigma,
                noise_decay=noise_decay,
                min_noise=min_noise
            )

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
            mf_state = compute_mean_field_state(agent_id, observations[agent_id], self.agent_ids)
            actions[agent_id] = self.agents[agent_id].take_action(mf_state, add_noise)
        return actions

    def store_experience(self, observations, actions, rewards, next_observations, dones):
        """
        存储经验到各智能体的回放缓冲区

        Args:
            observations (dict): 当前观测
            actions (dict): 执行的动作
            rewards (dict): 获得的奖励
            next_observations (dict): 下一状态观测
            dones (dict): 终止标志
        """
        normalized_rewards = normalize_rewards(rewards, self.agent_ids)

        for agent_id in self.agent_ids:
            mf_state = compute_mean_field_state(agent_id, observations[agent_id], self.agent_ids)
            next_mf_state = compute_mean_field_state(agent_id, next_observations[agent_id], self.agent_ids)

            experience = (
                mf_state,
                actions[agent_id],
                normalized_rewards[agent_id],
                next_mf_state,
                dones[agent_id]
            )
            self.replay_buffers[agent_id].add(experience)

    def learn(self):
        """
        所有智能体独立学习更新网络
        """
        for agent_id in self.agent_ids:
            if len(self.replay_buffers[agent_id]) < 8:
                continue

            batch_size = self._get_dynamic_batch_size(agent_id)
            batch = self.replay_buffers[agent_id].sample(batch_size)

            self._update_agent(agent_id, batch)
            self.agents[agent_id].soft_update(self.tau)

    def _update_agent(self, agent_id, batch):
        """
        更新单个智能体的网络

        Args:
            agent_id (str): 智能体ID
            batch (list): 经验批次
        """
        agent = self.agents[agent_id]

        mf_states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp[1] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).unsqueeze(1).to(self.device)
        next_mf_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp[4] for exp in batch])).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = agent.actor_target(next_mf_states)
            next_critic_states = torch.cat([next_mf_states, next_actions], dim=1)
            next_q_values = agent.critic_target(next_critic_states)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        current_critic_states = torch.cat([mf_states, actions], dim=1)
        current_q_values = agent.critic(current_critic_states)

        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        predicted_actions = agent.actor(mf_states)
        predicted_critic_states = torch.cat([mf_states, predicted_actions], dim=1)
        actor_loss = -agent.critic(predicted_critic_states).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def _get_dynamic_batch_size(self, agent_id):
        """
        动态调整批次大小

        Args:
            agent_id (str): 智能体ID

        Returns:
            int: 调整后的批次大小
        """
        buffer_size = len(self.replay_buffers[agent_id])

        if buffer_size < self.batch_size // 2:
            return min(self.batch_size // 4, buffer_size)
        elif buffer_size < self.batch_size:
            return min(self.batch_size // 2, buffer_size)
        else:
            return min(self.batch_size, buffer_size)


def compute_mean_field_state(agent_id: str, observation: Dict, all_agents: List[str]) -> np.ndarray:
    """
    计算单个agent的Mean Field状态

    将其他agent的价格信息压缩为均值，实现Mean Field近似。
    这是MF-DDPG的核心设计：用统计平均信息替代完整的全局信息。

    状态组成：
        - own_last_prices: 自身上轮价格（决策历史）
        - own_last_flow: 自身上轮充电流量（市场反馈）
        - mean_field_prices: 其他agent价格的平均值（竞争环境）

    Args:
        agent_id (str): 当前agent的ID（如"agent_0"）
        observation (dict): 环境返回的观测字典，包含:
            - "last_round_all_prices": np.ndarray, shape=(n_agents, n_periods)
            - "own_charging_flow": np.ndarray, shape=(n_periods,)
        all_agents (list): agent ID列表，用于确定索引顺序

    Returns:
        np.ndarray: Mean Field状态向量，shape = (3 * n_periods,)
    """
    all_prices = observation["last_round_all_prices"]
    agent_idx = all_agents.index(agent_id)
    own_last_prices = all_prices[agent_idx].flatten()
    other_indices = [i for i in range(len(all_agents)) if i != agent_idx]
    mean_field_prices = np.mean(all_prices[other_indices], axis=0).flatten()
    own_last_flow = observation["own_charging_flow"].flatten()
    mf_state = np.concatenate([own_last_prices, own_last_flow, mean_field_prices])
    return mf_state


def normalize_rewards(rewards: Dict, agent_ids: List[str]) -> Dict:
    """
    归一化奖励（博弈特定归一化）

    使用当轮最大值进行正仿射变换，保持纳什均衡等价性。

    Args:
        rewards (dict): 原始奖励字典 {agent_id: reward}
        agent_ids (list): agent ID列表

    Returns:
        dict: 归一化后的奖励字典
    """
    reward_values = list(rewards.values())
    max_reward = max(reward_values) if reward_values else 1.0

    if max_reward > 0:
        normalized_rewards = {aid: rewards[aid] / max_reward for aid in agent_ids}
    else:
        normalized_rewards = {aid: 0.0 for aid in agent_ids}

    return normalized_rewards
