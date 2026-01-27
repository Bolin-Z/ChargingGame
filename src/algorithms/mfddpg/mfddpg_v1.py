"""
MFDDPG v1 - 适配 Fictitious Play 风格的 Mean Field DDPG 实现

基于 AlgorithmBase 接口，接收 beliefs 矩阵作为观测输入。
实现平均场近似范式，将其他 Agent 的行为压缩为统计均值。

核心特点：
- Actor: 接收自身 belief + 其他 Agent 均值 (2 × n_periods)
- Critic: 接收 MF 状态 + 自身动作
- γ=0: 静态博弈，Critic 直接拟合即时收益
- 每个 Agent 维护独立的 ReplayBuffer
- 扩展性最佳：状态维度不随 Agent 数量增长
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
    单智能体 DDPG 实现（Mean Field 版本）

    使用 Mean Field 状态：自身 belief + 其他 Agent 均值。
    """

    def __init__(
        self,
        agent_name: str,
        mf_state_dim: int,
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
            mf_state_dim: Mean Field 状态维度（Actor 输入）
            action_dim: 动作维度
            critic_input_dim: Critic 输入维度（MF 状态 + 动作）
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
        self.actor = ActorNetwork(mf_state_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic = CriticNetwork(critic_input_dim, critic_hidden_sizes).to(device)

        # 创建目标网络
        self.actor_target = ActorNetwork(mf_state_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic_target = CriticNetwork(critic_input_dim, critic_hidden_sizes).to(device)

        # 初始化目标网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 创建探索噪音
        self.noise = GaussianNoise(action_dim, noise_sigma, noise_decay, min_noise)

    def take_action(self, mf_state: np.ndarray, add_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        选择动作

        Args:
            mf_state: Mean Field 状态向量 (mf_state_dim,)
            add_noise: 是否添加探索噪音

        Returns:
            (noisy_action, pure_action)
        """
        mf_state_tensor = torch.FloatTensor(mf_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pure_action = self.actor(mf_state_tensor).cpu().numpy().flatten()

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


class MFDDPGv1(AlgorithmBase):
    """
    MFDDPG v1 - 适配 Fictitious Play 的 Mean Field DDPG 实现

    平均场近似范式：
    - 将其他 Agent 的行为压缩为统计均值
    - 状态维度不随 Agent 数量增长，扩展性最佳
    - 每个 Agent 独立训练
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
        初始化 MFDDPG v1

        Args:
            agent_names: Agent 名称列表
            n_periods: 时段数量（动作维度）
            actor_hidden_sizes: Actor 网络隐藏层配置
            critic_hidden_sizes: Critic 网络隐藏层配置
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子（静态博弈设为 0）
            tau: 软更新系数
            buffer_capacity: 经验回放容量（每个 Agent 独立）
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
        # MF 状态: 自身 belief + 其他均值 -> 2 × n_periods
        self._mf_state_dim = 2 * self._n_periods
        # Critic: MF 状态 + 自身动作 -> 2 × n_periods + n_periods = 3 × n_periods
        self._critic_input_dim = self._mf_state_dim + self._n_periods

        # 创建智能体
        self._agents: dict[str, DDPG] = {}
        for agent_name in self._agent_names:
            self._agents[agent_name] = DDPG(
                agent_name=agent_name,
                mf_state_dim=self._mf_state_dim,
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

        # 每个 Agent 独立的经验回放
        self._replay_buffers: dict[str, ReplayBuffer] = {
            name: ReplayBuffer(buffer_capacity) for name in self._agent_names
        }

    @property
    def name(self) -> str:
        return "MFDDPG"

    def _compute_mf_state(self, beliefs: np.ndarray, agent_name: str) -> np.ndarray:
        """
        计算 Mean Field 状态

        Args:
            beliefs: 信念矩阵 (n_agents, n_periods)
            agent_name: 目标 Agent 名称

        Returns:
            MF 状态向量 (2 × n_periods,)
        """
        agent_idx = self._get_agent_index(agent_name)

        # 自身 belief
        own_belief = beliefs[agent_idx].flatten()

        # 其他 Agent 的均值
        other_indices = [i for i in range(self._n_agents) if i != agent_idx]
        if len(other_indices) > 0:
            others_mean = np.mean(beliefs[other_indices], axis=0).flatten()
        else:
            # 单 Agent 情况，用自身作为均值
            others_mean = own_belief.copy()

        return np.concatenate([own_belief, others_mean]).astype(np.float32)

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
        pure_actions = {}
        noisy_actions = {}

        for agent_name in self._agent_names:
            # 计算 MF 状态
            mf_state = self._compute_mf_state(beliefs, agent_name)
            noisy, pure = self._agents[agent_name].take_action(mf_state, add_noise)
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
        存入各 Agent 的独立 ReplayBuffer

        Args:
            beliefs: 当前信念矩阵 (n_agents, n_periods)
            noisy_actions: 噪声动作
            rewards: 收益
            next_beliefs: 下一时刻信念矩阵
        """
        # 归一化奖励
        normalized_rewards = self._normalize_rewards(rewards)

        # 为每个 Agent 独立存储经验（存储 MF 状态）
        for agent_name in self._agent_names:
            mf_state = self._compute_mf_state(beliefs, agent_name)
            next_mf_state = self._compute_mf_state(next_beliefs, agent_name)

            experience = (
                mf_state,
                noisy_actions[agent_name].copy(),
                normalized_rewards[agent_name],
                next_mf_state,
            )
            self._replay_buffers[agent_name].add(experience)

    def learn(self) -> dict | None:
        """
        独立训练：每个 Agent 独立更新网络

        Returns:
            学习指标字典，如果没有任何 Agent 学习则返回 None
        """
        metrics = {"agents": {}}
        any_learned = False

        for agent_name in self._agent_names:
            agent = self._agents[agent_name]
            buffer = self._replay_buffers[agent_name]

            # 检查经验是否足够
            if len(buffer) < self._min_buffer_size:
                continue

            # 动态调整批次大小
            batch_size = self._get_dynamic_batch_size(agent_name)

            # 采样经验
            batch = buffer.sample(batch_size)

            # 解析批次数据
            batch_mf_states = np.array([exp[0] for exp in batch])
            batch_actions = np.array([exp[1] for exp in batch])
            batch_rewards = np.array([exp[2] for exp in batch])
            batch_next_mf_states = np.array([exp[3] for exp in batch])

            # 更新网络
            agent_metrics = self._update_agent(
                agent, batch_mf_states, batch_actions, batch_rewards, batch_next_mf_states, batch_size
            )

            # 软更新目标网络
            agent.soft_update(self._tau)

            # 收集指标
            metrics["agents"][agent_name] = {
                **agent_metrics,
                "noise_sigma": agent.noise.sigma,
                "buffer_size": len(buffer),
                "batch_size": batch_size,
            }
            any_learned = True

        return metrics if any_learned else None

    def _update_agent(
        self,
        agent: DDPG,
        batch_mf_states: np.ndarray,
        batch_actions: np.ndarray,
        batch_rewards: np.ndarray,
        batch_next_mf_states: np.ndarray,
        batch_size: int,
    ) -> dict:
        """更新单个 Agent 的网络"""
        # 转换为 tensor
        mf_states = torch.FloatTensor(batch_mf_states).to(self._device)
        actions = torch.FloatTensor(batch_actions).to(self._device)
        rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(self._device)
        next_mf_states = torch.FloatTensor(batch_next_mf_states).to(self._device)

        # === Critic 更新 ===
        with torch.no_grad():
            # 使用目标 Actor 生成下一动作
            next_actions = agent.actor_target(next_mf_states)
            next_critic_inputs = torch.cat([next_mf_states, next_actions], dim=1)
            next_q = agent.critic_target(next_critic_inputs)

            # 目标 Q 值（γ=0 时直接等于即时奖励）
            if self._gamma > 0:
                target_q = rewards + self._gamma * next_q
            else:
                target_q = rewards

        # 当前 Critic 输入
        current_critic_inputs = torch.cat([mf_states, actions], dim=1)
        current_q = agent.critic(current_critic_inputs)

        # Critic 损失
        critic_loss = nn.MSELoss()(current_q, target_q)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._compute_grad_norm(agent.critic)
        agent.critic_optimizer.step()

        # === Actor 更新 ===
        new_actions = agent.actor(mf_states)
        new_critic_inputs = torch.cat([mf_states, new_actions], dim=1)
        actor_loss = -agent.critic(new_critic_inputs).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._compute_grad_norm(agent.actor)
        agent.actor_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "critic_grad_norm": critic_grad_norm,
            "q_value_mean": current_q.mean().item(),
            "q_value_std": current_q.std().item(),
            "target_q_mean": target_q.mean().item(),
            "actor_loss": actor_loss.item(),
            "actor_grad_norm": actor_grad_norm,
        }

    def _get_dynamic_batch_size(self, agent_name: str) -> int:
        """动态调整批次大小"""
        buffer_size = len(self._replay_buffers[agent_name])

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

        MFDDPG 的 Critic 使用 MF 状态 + 自身动作。

        Args:
            beliefs: 信念矩阵 (n_agents, n_periods)
            agent_name: 目标 Agent 名称
            all_actions: 所有 Agent 的动作（只使用自身的），支持 numpy 或 Tensor

        Returns:
            Critic 输入 tensor, shape=(1, critic_input_dim)
        """
        # 计算 MF 状态
        mf_state = self._compute_mf_state(beliefs, agent_name)
        mf_state_tensor = torch.FloatTensor(mf_state).to(self._device)

        # 只使用自身动作
        own_action = all_actions[agent_name]
        if isinstance(own_action, torch.Tensor):
            # 保持 Tensor 的梯度链路
            action_tensor = own_action.flatten()
        else:
            # numpy 转 Tensor
            action_tensor = torch.FloatTensor(own_action.flatten()).to(self._device)

        critic_input = torch.cat([mf_state_tensor, action_tensor]).unsqueeze(0)
        return critic_input

    def reset_noise(self) -> None:
        """重置所有 Agent 的探索噪音"""
        for agent in self._agents.values():
            agent.noise.reset()
