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
    每个DDPG智能体独立维护自己的网络和优化器。
    """
    
    def __init__(self, agent_id, obs_dim, action_dim, global_obs_dim,
                 actor_lr=0.001, critic_lr=0.001, device='cpu',
                 actor_hidden_sizes=(64, 64), critic_hidden_sizes=(128, 64),
                 noise_sigma=0.2, noise_decay=0.9995, min_noise=0.01):
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
        self.critic = CriticNetwork(global_obs_dim, critic_hidden_sizes).to(device)
        
        # 创建目标网络（用于稳定训练）
        self.actor_target = ActorNetwork(obs_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic_target = CriticNetwork(global_obs_dim, critic_hidden_sizes).to(device)
        
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


class MADDPG:
    """
    多智能体MADDPG协调器
    
    管理多个DDPG智能体，实现"中心化训练，去中心化执行"的MADDPG算法。
    在训练阶段，每个智能体的Critic网络可以访问全局信息；
    在执行阶段，每个智能体只使用自己的Actor网络和局部观测。
    """
    
    def __init__(self, agent_ids, obs_dim, action_dim, global_obs_dim,
                 buffer_capacity=10000, max_batch_size=64, actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99, tau=0.01, seed=None, device='cpu',
                 actor_hidden_sizes=(64, 64), critic_hidden_sizes=(128, 64),
                 noise_sigma=0.2, noise_decay=0.9995, min_noise=0.01,
                 flow_scale_factor=1.0, reward_scale=None):
        """
        初始化MADDPG协调器

        Args:
            agent_ids (list): 智能体ID列表
            obs_dim (int): 单个智能体的观测维度
            action_dim (int): 单个智能体的动作维度
            global_obs_dim (int): 全局观测维度（用于Critic网络）
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
            flow_scale_factor (float): 流量缩放因子，用于归一化流量观测
            reward_scale (float, optional): 固定奖励缩放因子。如果为 None，使用动态缩放（当步最大值）
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
        self.flow_scale_factor = flow_scale_factor
        self.reward_scale = reward_scale  # None 表示使用动态缩放
        
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
                device=device,
                actor_hidden_sizes=actor_hidden_sizes,
                critic_hidden_sizes=critic_hidden_sizes,
                noise_sigma=noise_sigma,
                noise_decay=noise_decay,
                min_noise=min_noise
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
            obs = process_observations(observations[agent_id], self.flow_scale_factor)
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
        normalized_rewards = normalize_rewards(rewards, self.reward_scale)

        experience = (observations, actions, normalized_rewards, next_observations, dones)
        self.replay_buffer.add(experience)
    
    def _get_dynamic_batch_size(self):
        """
        动态调整批次大小，适配环境特点
        
        使用配置的max_batch_size作为上限，根据缓冲区大小动态调整
        
        Returns:
            int: 当前应使用的批次大小
        """
        buffer_size = len(self.replay_buffer)
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
            dict: 包含 critic_loss, critic_grad_norm, q_value 统计的诊断指标
        """
        batch_size = len(batch_obs)

        # 构建当前状态的全局状态向量
        current_global_states = []
        for i in range(batch_size):
            global_state = organize_global_state(batch_obs[i], batch_actions[i], self.flow_scale_factor)
            current_global_states.append(global_state)
        current_global_states = torch.FloatTensor(np.array(current_global_states)).to(self.device)

        # 构建下一状态的全局状态向量（需要使用目标Actor网络生成下一动作）
        next_global_states = []
        for i in range(batch_size):
            # 为所有智能体生成下一状态的动作（使用目标Actor网络）
            next_actions = {}
            for aid in self.agent_ids:
                next_obs_agent = process_observations(batch_next_obs[i][aid], self.flow_scale_factor)
                next_obs_tensor = torch.FloatTensor(next_obs_agent).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    next_action = self.agents[aid].actor_target(next_obs_tensor).detach().cpu().numpy().flatten()
                next_actions[aid] = next_action

            # 组织下一状态的全局信息
            global_next_state = organize_global_state(batch_next_obs[i], next_actions, self.flow_scale_factor)
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

        # 收集梯度范数（在 step 之前）
        critic_grad_norm = self._compute_grad_norm(agent.critic)

        agent.critic_optimizer.step()

        # 返回诊断指标
        return {
            'critic_loss': critic_loss.item(),
            'critic_grad_norm': critic_grad_norm,
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item(),
            'q_value_max': current_q_values.max().item(),
            'q_value_min': current_q_values.min().item(),
            'target_q_mean': target_q_values.mean().item(),
        }
    
    def _update_actor(self, agent, agent_id, batch_obs, batch_actions):
        """
        更新单个智能体的Actor网络

        【修复】保持当前 agent 动作的梯度链路，不使用 detach()

        Args:
            agent: DDPG智能体实例
            agent_id: 智能体ID
            batch_obs, batch_actions: 批次观测和动作数据

        Returns:
            dict: 包含 actor_loss 和 actor_grad_norm 的诊断指标
        """
        batch_size = len(batch_obs)

        # 准备观测数据
        obs_list = []
        for i in range(batch_size):
            current_obs = process_observations(batch_obs[i][agent_id], self.flow_scale_factor)
            obs_list.append(current_obs)

        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)

        # 生成当前智能体的动作（保持梯度）
        new_actions = agent.actor(obs_tensor)  # shape: (batch_size, action_dim)

        # 【修复】构建全局状态时保持梯度链路
        # 使用张量版本的全局状态构建，当前 agent 的动作保持为 tensor
        global_states_tensor = self._build_global_state_for_actor_update(
            batch_obs, batch_actions, agent_id, new_actions
        )

        # 计算Actor损失（策略梯度）
        q_values = agent.critic(global_states_tensor)
        actor_loss = -q_values.mean()  # 最大化Q值

        # 反向传播更新Actor
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()

        # 收集梯度范数（在 step 之前）—— 这是关键诊断指标！
        actor_grad_norm = self._compute_grad_norm(agent.actor)

        agent.actor_optimizer.step()

        # 返回诊断指标
        return {
            'actor_loss': actor_loss.item(),
            'actor_grad_norm': actor_grad_norm,
        }

    def _build_global_state_for_actor_update(self, batch_obs, batch_actions, current_agent_id, current_agent_new_actions):
        """
        为 Actor 更新构建全局状态张量（保持梯度链路）

        与 organize_global_state 功能相同，但保持当前 agent 动作的梯度。
        其他 agent 的动作仍使用 numpy（它们不需要梯度）。

        Args:
            batch_obs: 批次观测数据
            batch_actions: 批次动作数据（来自经验回放）
            current_agent_id: 当前正在更新的 agent ID
            current_agent_new_actions: 当前 agent 的新动作（torch.Tensor，需要梯度）

        Returns:
            torch.Tensor: 全局状态张量，shape=(batch_size, global_state_dim)
        """
        batch_size = len(batch_obs)
        sorted_agents = sorted(batch_obs[0].keys())

        global_states = []

        for i in range(batch_size):
            # 1. 全局价格历史（去重）
            global_prices = batch_obs[i][sorted_agents[0]]["last_round_all_prices"].flatten()
            global_prices_tensor = torch.FloatTensor(global_prices).to(self.device)

            # 2. 所有智能体充电流量
            all_flows = []
            for agent_id in sorted_agents:
                flow = batch_obs[i][agent_id]["own_charging_flow"].flatten() / self.flow_scale_factor
                all_flows.append(flow)
            all_flows_tensor = torch.FloatTensor(np.concatenate(all_flows)).to(self.device)

            # 3. 所有智能体动作（关键：当前 agent 用 tensor，其他用 numpy）
            action_tensors = []
            for agent_id in sorted_agents:
                if agent_id == current_agent_id:
                    # 当前 agent：使用新生成的动作（保持梯度）
                    action_tensors.append(current_agent_new_actions[i])
                else:
                    # 其他 agent：使用经验回放中的动作（无需梯度）
                    other_action = torch.FloatTensor(batch_actions[i][agent_id].flatten()).to(self.device)
                    action_tensors.append(other_action)

            all_actions_tensor = torch.cat(action_tensors)

            # 拼接成完整的全局状态
            global_state = torch.cat([global_prices_tensor, all_flows_tensor, all_actions_tensor])
            global_states.append(global_state)

        return torch.stack(global_states)

    def _compute_grad_norm(self, network):
        """
        计算网络参数的梯度范数

        Args:
            network: PyTorch 网络模块

        Returns:
            float: 梯度的 L2 范数，如果没有梯度则返回 0.0
        """
        total_norm = 0.0
        for param in network.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def learn(self):
        """
        中心化训练：更新所有智能体的网络

        Returns:
            dict | None: 诊断指标字典，如果经验不足则返回 None
                - 每个 agent 的 actor_loss, actor_grad_norm
                - 每个 agent 的 critic_loss, critic_grad_norm, q_value 统计
                - 当前探索噪音强度 noise_sigma
        """
        # 检查是否有足够的经验开始学习
        min_buffer_size = 8  # 最小8个经验即开始学习
        if len(self.replay_buffer) < min_buffer_size:
            return None

        # 动态调整批次大小
        current_batch_size = self._get_dynamic_batch_size()

        # 从经验回放缓冲区采样
        batch_experiences = self.replay_buffer.sample(current_batch_size)

        # 解析批次经验
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self._parse_batch_experiences(batch_experiences)

        # 收集诊断指标
        metrics = {
            'batch_size': current_batch_size,
            'buffer_size': len(self.replay_buffer),
            'agents': {}
        }

        # 为每个智能体更新Critic网络
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            critic_metrics = self._update_critic(agent, agent_id, batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
            metrics['agents'][agent_id] = critic_metrics

        # 为每个智能体更新Actor网络
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            actor_metrics = self._update_actor(agent, agent_id, batch_obs, batch_actions)
            metrics['agents'][agent_id].update(actor_metrics)
            # 记录当前 agent 的探索噪音强度
            metrics['agents'][agent_id]['noise_sigma'] = agent.noise.sigma

        # 软更新所有智能体的目标网络
        self.update_all_targets(self.tau)

        return metrics
    
    def update_all_targets(self, tau=0.01):
        """
        更新所有智能体的目标网络
        
        Args:
            tau (float): 软更新系数
        """
        for agent in self.agents.values():
            agent.soft_update(tau)


# 工具函数

def process_observations(observation, flow_scale_factor=1.0):
    """
    处理单个智能体观测数据，转换为网络输入向量

    将字典格式的观测数据展平为一维向量，用于神经网络输入。
    流量会除以 flow_scale_factor 进行缩放，使其与价格处于相近的数量级。

    Args:
        observation (dict): 智能体观测，包含:
            - "last_round_all_prices": np.ndarray, 形状 (n_agents, n_periods)
            - "own_charging_flow": np.ndarray, 形状 (n_periods,)
        flow_scale_factor (float): 流量缩放因子，默认为1.0（不缩放）

    Returns:
        np.ndarray: 展平的观测向量，形状 (obs_dim,)
    """
    last_prices = observation["last_round_all_prices"].flatten()
    own_flow = observation["own_charging_flow"].flatten() / flow_scale_factor
    return np.concatenate([last_prices, own_flow])


def organize_global_state(observations, actions, flow_scale_factor=1.0):
    """
    组织全局状态信息，去除重复数据（优化版本）

    将所有智能体的观测和动作信息组织成集中式输入向量。
    优化策略：去除重复的全局价格信息，减少参数量。
    流量会除以 flow_scale_factor 进行缩放。

    Args:
        observations (dict): 所有智能体的观测 {agent_id: observation_dict}
        actions (dict): 所有智能体的动作 {agent_id: action_array}
        flow_scale_factor (float): 流量缩放因子，默认为1.0（不缩放）

    Returns:
        np.ndarray: 优化后的全局状态向量
    """
    sorted_agents = sorted(observations.keys())  # 确保顺序一致

    # 1. 全局价格历史（去重：所有agent观测中的价格信息相同，只取一份）
    global_prices = observations[sorted_agents[0]]["last_round_all_prices"].flatten()

    # 2. 所有智能体充电流量（无重复：每个agent的流量不同，需要全部保留）
    all_flows = []
    for agent_id in sorted_agents:
        flow = observations[agent_id]["own_charging_flow"].flatten() / flow_scale_factor
        all_flows.append(flow)
    all_charging_flows = np.concatenate(all_flows)

    # 3. 所有智能体当前动作（无重复：每个agent动作不同）
    all_actions = []
    for agent_id in sorted_agents:
        all_actions.append(actions[agent_id].flatten())
    all_current_actions = np.concatenate(all_actions)

    return np.concatenate([global_prices, all_charging_flows, all_current_actions])


def normalize_rewards(rewards, reward_scale=None):
    """
    奖励归一化：使用固定或动态缩放因子

    Args:
        rewards (dict): 原始奖励 {agent_id: reward}
        reward_scale (float, optional): 固定缩放因子。如果为 None，使用当步最大值（动态缩放）

    Returns:
        dict: 归一化奖励 {agent_id: normalized_reward}
    """
    if reward_scale is not None:
        # 固定缩放：使用预设的 reward_scale
        scale = reward_scale
    else:
        # 动态缩放：使用当步最大值（保持向后兼容）
        scale = max(rewards.values()) if rewards.values() else 1.0

    if scale > 0:
        return {agent: reward / scale for agent, reward in rewards.items()}
    else:
        return {agent: 0.0 for agent in rewards.keys()}