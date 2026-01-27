"""
MADDPG v1 - 适配 Fictitious Play 风格的 MADDPG 实现

基于 AlgorithmBase 接口，接收 beliefs 矩阵作为观测输入。
实现 CTDE（中心化训练，去中心化执行）范式。

核心特点：
- Actor: 接收完整 beliefs 矩阵 (n_agents × n_periods)
- Critic: 接收 beliefs + 所有 Agent 动作（中心化训练）
- γ=0: 静态博弈，Critic 直接拟合即时收益
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base import AlgorithmBase
from ..common import ReplayBuffer, GaussianNoise
from .networks import ActorNetwork, CriticNetwork


class DDPG:
    """
    单智能体 DDPG 实现

    包含 Actor-Critic 架构和目标网络。
    每个 DDPG 智能体独立维护自己的网络和优化器。
    """

    def __init__(
        self,
        agent_name: str,
        obs_dim: int,
        action_dim: int,
        critic_input_dim: int,
        actor_lr: float,
        critic_lr: float,
        actor_hidden_sizes: tuple,
        critic_hidden_sizes: tuple,
        noise_sigma: float,
        noise_decay: float,
        min_noise: float,
        device: str,
    ):
        """
        初始化 DDPG 智能体

        Args:
            agent_name: 智能体名称
            obs_dim: Actor 观测维度
            action_dim: 动作维度
            critic_input_dim: Critic 输入维度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            actor_hidden_sizes: Actor 隐藏层配置
            critic_hidden_sizes: Critic 隐藏层配置
            noise_sigma: 探索噪音初始标准差
            noise_decay: 噪音衰减率
            min_noise: 最小噪音标准差
            device: 计算设备
        """
        self.agent_name = agent_name
        self.device = device

        # 创建主网络
        self.actor = ActorNetwork(obs_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic = CriticNetwork(critic_input_dim, critic_hidden_sizes).to(device)

        # 创建目标网络
        self.actor_target = ActorNetwork(obs_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic_target = CriticNetwork(critic_input_dim, critic_hidden_sizes).to(device)

        # 初始化目标网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 创建探索噪音
        self.noise = GaussianNoise(action_dim, noise_sigma, noise_decay, min_noise)

    def take_action(self, obs: np.ndarray, add_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        选择动作

        Args:
            obs: 观测向量 (obs_dim,)
            add_noise: 是否添加探索噪音

        Returns:
            (noisy_action, pure_action)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pure_action = self.actor(obs_tensor).cpu().numpy().flatten()

        if add_noise:
            noisy_action = self.noise(pure_action.copy())
        else:
            noisy_action = pure_action.copy()

        return noisy_action, pure_action

    def soft_update(self, tau: float) -> None:
        """软更新目标网络"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class MADDPGv1(AlgorithmBase):
    """
    MADDPG v1 - 适配 Fictitious Play 的 MADDPG 实现

    中心化训练、去中心化执行（CTDE）范式：
    - 训练时 Critic 可访问全局信息（所有 Agent 的 beliefs 和动作）
    - 执行时 Actor 只使用 beliefs 矩阵
    """

    def __init__(
        self,
        agent_names: list[str],
        n_periods: int,
        # 网络参数
        actor_hidden_sizes: tuple[int, ...] = (64, 64),
        critic_hidden_sizes: tuple[int, ...] = (128, 64),
        # 学习参数
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        gamma: float = 0.0,
        tau: float = 0.01,
        # 经验回放
        buffer_capacity: int = 10000,
        max_batch_size: int = 64,
        min_buffer_size: int = 8,
        # 噪音参数
        noise_sigma: float = 0.2,
        noise_decay: float = 0.9995,
        min_noise: float = 0.01,
        # 设备
        device: str = "cpu",
        # 随机种子
        seed: int | None = None,
    ):
        """
        初始化 MADDPG v1

        Args:
            agent_names: Agent 名称列表
            n_periods: 时段数量（动作维度）
            actor_hidden_sizes: Actor 网络隐藏层配置
            critic_hidden_sizes: Critic 网络隐藏层配置
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子（静态博弈设为 0）
            tau: 软更新系数
            buffer_capacity: 经验回放容量
            max_batch_size: 最大批次大小
            min_buffer_size: 开始学习的最小经验数
            noise_sigma: 探索噪音初始标准差
            noise_decay: 噪音衰减率
            min_noise: 最小噪音标准差
            device: 计算设备
            seed: 随机种子
        """
        super().__init__(
            agent_names=agent_names,
            n_periods=n_periods,
            device=device,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            min_noise=min_noise,
        )

        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # 保存参数
        self._gamma = gamma
        self._tau = tau
        self._max_batch_size = max_batch_size
        self._min_buffer_size = min_buffer_size

        # 计算网络维度
        # Actor: beliefs 展平 -> (n_agents × n_periods)
        self._obs_dim = self._n_agents * self._n_periods
        # Critic: beliefs 展平 + 所有动作 -> (n_agents × n_periods) + (n_agents × n_periods)
        self._critic_input_dim = self._obs_dim + self._n_agents * self._n_periods

        # 创建智能体
        self._agents: dict[str, DDPG] = {}
        for agent_name in self._agent_names:
            self._agents[agent_name] = DDPG(
                agent_name=agent_name,
                obs_dim=self._obs_dim,
                action_dim=self._n_periods,
                critic_input_dim=self._critic_input_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                actor_hidden_sizes=actor_hidden_sizes,
                critic_hidden_sizes=critic_hidden_sizes,
                noise_sigma=noise_sigma,
                noise_decay=noise_decay,
                min_noise=min_noise,
                device=device,
            )

        # 共享经验回放（MADDPG 使用共享 Buffer）
        self._replay_buffer = ReplayBuffer(buffer_capacity)

    @property
    def name(self) -> str:
        return "MADDPG"

    def take_action(
        self,
        beliefs: np.ndarray,
        add_noise: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        所有 Agent 选择动作

        Args:
            beliefs: EMA 信念矩阵 (n_agents, n_periods)
            add_noise: 是否添加探索噪音

        Returns:
            (pure_actions, noisy_actions)
        """
        # beliefs 展平作为观测
        obs = beliefs.flatten().astype(np.float32)

        pure_actions = {}
        noisy_actions = {}

        for agent_name in self._agent_names:
            noisy, pure = self._agents[agent_name].take_action(obs, add_noise)
            pure_actions[agent_name] = pure
            noisy_actions[agent_name] = noisy

        return pure_actions, noisy_actions

    def store_experience(
        self,
        beliefs: np.ndarray,
        noisy_actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        next_beliefs: np.ndarray,
    ) -> None:
        """
        存入 ReplayBuffer

        Args:
            beliefs: 当前信念矩阵 (n_agents, n_periods)
            noisy_actions: 噪声动作
            rewards: 收益
            next_beliefs: 下一时刻信念矩阵
        """
        # 归一化奖励
        normalized_rewards = self._normalize_rewards(rewards)

        # 存储经验
        experience = (
            beliefs.copy(),
            {k: v.copy() for k, v in noisy_actions.items()},
            normalized_rewards,
            next_beliefs.copy(),
        )
        self._replay_buffer.add(experience)

    def learn(self) -> dict | None:
        """
        中心化训练：更新所有智能体的网络

        Returns:
            学习指标字典，经验不足返回 None
        """
        if len(self._replay_buffer) < self._min_buffer_size:
            return None

        # 动态调整批次大小
        batch_size = self._get_dynamic_batch_size()

        # 采样经验
        batch = self._replay_buffer.sample(batch_size)

        # 解析批次数据
        batch_beliefs = np.array([exp[0] for exp in batch])
        batch_actions = [exp[1] for exp in batch]
        batch_rewards = [exp[2] for exp in batch]
        batch_next_beliefs = np.array([exp[3] for exp in batch])

        # 收集指标
        metrics = {
            "batch_size": batch_size,
            "buffer_size": len(self._replay_buffer),
            "agents": {},
        }

        # 更新每个智能体
        for agent_name in self._agent_names:
            agent = self._agents[agent_name]

            # 更新 Critic
            critic_metrics = self._update_critic(
                agent, agent_name, batch_beliefs, batch_actions,
                batch_rewards, batch_next_beliefs, batch_size
            )

            # 更新 Actor
            actor_metrics = self._update_actor(
                agent, agent_name, batch_beliefs, batch_actions, batch_size
            )

            # 软更新目标网络
            agent.soft_update(self._tau)

            # 收集指标
            metrics["agents"][agent_name] = {
                **critic_metrics,
                **actor_metrics,
                "noise_sigma": agent.noise.sigma,
            }

        return metrics

    def _update_critic(
        self,
        agent: DDPG,
        agent_name: str,
        batch_beliefs: np.ndarray,
        batch_actions: list[dict],
        batch_rewards: list[dict],
        batch_next_beliefs: np.ndarray,
        batch_size: int,
    ) -> dict:
        """更新 Critic 网络"""
        # 构建当前状态的 Critic 输入
        current_critic_inputs = []
        for i in range(batch_size):
            critic_input = self._build_critic_input_internal(
                batch_beliefs[i], batch_actions[i]
            )
            current_critic_inputs.append(critic_input)
        current_critic_inputs = torch.FloatTensor(np.array(current_critic_inputs)).to(self._device)

        # 构建下一状态的 Critic 输入（使用目标网络生成动作）
        next_critic_inputs = []
        for i in range(batch_size):
            next_obs = batch_next_beliefs[i].flatten().astype(np.float32)
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self._device)

            # 所有 Agent 使用目标 Actor 生成下一动作
            next_actions = {}
            for name in self._agent_names:
                with torch.no_grad():
                    next_action = self._agents[name].actor_target(next_obs_tensor).cpu().numpy().flatten()
                next_actions[name] = next_action

            critic_input = self._build_critic_input_internal(
                batch_next_beliefs[i], next_actions
            )
            next_critic_inputs.append(critic_input)
        next_critic_inputs = torch.FloatTensor(np.array(next_critic_inputs)).to(self._device)

        # 获取奖励
        rewards = torch.FloatTensor([batch_rewards[i][agent_name] for i in range(batch_size)]).unsqueeze(1).to(self._device)

        # 计算目标 Q 值（γ=0 时直接等于即时奖励）
        with torch.no_grad():
            if self._gamma > 0:
                next_q = agent.critic_target(next_critic_inputs)
                target_q = rewards + self._gamma * next_q
            else:
                target_q = rewards

        # 计算当前 Q 值
        current_q = agent.critic(current_critic_inputs)

        # Critic 损失
        critic_loss = nn.MSELoss()(current_q, target_q)

        # 更新
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._compute_grad_norm(agent.critic)
        agent.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "critic_grad_norm": critic_grad_norm,
            "q_value_mean": current_q.mean().item(),
            "q_value_std": current_q.std().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def _update_actor(
        self,
        agent: DDPG,
        agent_name: str,
        batch_beliefs: np.ndarray,
        batch_actions: list[dict],
        batch_size: int,
    ) -> dict:
        """更新 Actor 网络"""
        # 准备观测
        obs_list = [batch_beliefs[i].flatten().astype(np.float32) for i in range(batch_size)]
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self._device)

        # 生成当前 Agent 的新动作（保持梯度）
        new_actions = agent.actor(obs_tensor)

        # 构建 Critic 输入（保持梯度链路）
        critic_inputs = self._build_critic_input_for_actor_update(
            batch_beliefs, batch_actions, agent_name, new_actions, batch_size
        )

        # Actor 损失：最大化 Q 值
        q_values = agent.critic(critic_inputs)
        actor_loss = -q_values.mean()

        # 更新
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._compute_grad_norm(agent.actor)
        agent.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "actor_grad_norm": actor_grad_norm,
        }

    def _build_critic_input_internal(
        self,
        beliefs: np.ndarray,
        actions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """构建 Critic 输入（numpy 版本）"""
        # beliefs 展平
        beliefs_flat = beliefs.flatten()

        # 所有动作按顺序拼接
        all_actions = np.concatenate([actions[name].flatten() for name in self._agent_names])

        return np.concatenate([beliefs_flat, all_actions]).astype(np.float32)

    def _build_critic_input_for_actor_update(
        self,
        batch_beliefs: np.ndarray,
        batch_actions: list[dict],
        current_agent: str,
        current_new_actions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """构建 Actor 更新时的 Critic 输入（保持梯度）"""
        critic_inputs = []

        for i in range(batch_size):
            # beliefs 展平
            beliefs_flat = torch.FloatTensor(batch_beliefs[i].flatten()).to(self._device)

            # 动作：当前 Agent 用新动作（tensor），其他用经验中的动作
            action_tensors = []
            for name in self._agent_names:
                if name == current_agent:
                    action_tensors.append(current_new_actions[i])
                else:
                    action_tensors.append(
                        torch.FloatTensor(batch_actions[i][name].flatten()).to(self._device)
                    )

            all_actions = torch.cat(action_tensors)
            critic_input = torch.cat([beliefs_flat, all_actions])
            critic_inputs.append(critic_input)

        return torch.stack(critic_inputs)

    def _get_dynamic_batch_size(self) -> int:
        """动态调整批次大小"""
        buffer_size = len(self._replay_buffer)

        if buffer_size < self._max_batch_size // 2:
            return min(self._max_batch_size // 4, buffer_size)
        elif buffer_size < self._max_batch_size:
            return min(self._max_batch_size // 2, buffer_size)
        else:
            return min(self._max_batch_size, buffer_size)

    def _compute_grad_norm(self, network: nn.Module) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for param in network.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    # === NashConv 支持方法 ===

    def get_critics(self) -> dict[str, nn.Module]:
        """返回各 Agent 的 Critic 网络"""
        return {name: agent.critic for name, agent in self._agents.items()}

    def build_critic_input(
        self,
        beliefs: np.ndarray,
        agent_name: str,
        all_actions: dict[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """
        为指定 Agent 构造 Critic 输入

        Args:
            beliefs: 信念矩阵 (n_agents, n_periods)
            agent_name: 目标 Agent 名称
            all_actions: 所有 Agent 的动作，支持 numpy 或 Tensor

        Returns:
            Critic 输入 tensor, shape=(1, critic_input_dim)
        """
        # beliefs 展平
        beliefs_flat = torch.FloatTensor(beliefs.flatten()).to(self._device)

        # 动作：支持混合类型输入
        action_tensors = []
        for name in self._agent_names:
            action = all_actions[name]
            if isinstance(action, torch.Tensor):
                # 保持 Tensor 的梯度链路
                action_tensors.append(action.flatten())
            else:
                # numpy 转 Tensor
                action_tensors.append(
                    torch.FloatTensor(action.flatten()).to(self._device)
                )

        all_actions_tensor = torch.cat(action_tensors)
        critic_input = torch.cat([beliefs_flat, all_actions_tensor]).unsqueeze(0)
        return critic_input

    def reset_noise(self) -> None:
        """重置所有 Agent 的探索噪音"""
        for agent in self._agents.values():
            agent.noise.reset()
