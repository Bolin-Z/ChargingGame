"""
MADDPG神经网络架构定义

包含：
- ActorNetwork: 策略网络，可配置输入/输出/隐藏层维度
- CriticNetwork: 价值网络，可配置输入/隐藏层维度
"""

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Actor网络：将观测映射到动作
    
    可配置的策略网络架构，支持任意输入/输出/隐藏层维度。
    输出使用Sigmoid激活函数确保动作值在[0,1]范围内。
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[64, 64]):
        """
        初始化Actor网络
        
        Args:
            obs_dim: 观测空间维度
            action_dim: 动作空间维度  
            hidden_sizes: 隐藏层维度列表
        """
        super(ActorNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 构建网络层
        self.fc1 = nn.Linear(obs_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_dim)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier均匀分布权重初始化"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs):
        """
        前向传播
        
        Args:
            obs: 观测张量 (batch_size, obs_dim)
            
        Returns:
            torch.Tensor: 动作张量 (batch_size, action_dim), 值域[0,1]
        """
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid确保输出范围[0,1]
        return x


class CriticNetwork(nn.Module):
    """
    Critic网络：估算Q值
    
    可配置的价值网络架构，支持任意输入维度和隐藏层配置。
    用于中心化训练，接收全局状态信息并输出Q值估计。
    """
    
    def __init__(self, input_dim, hidden_sizes=[128, 64]):
        """
        初始化Critic网络
        
        Args:
            input_dim: 输入维度（集中式信息）
            hidden_sizes: 隐藏层维度列表
        """
        super(CriticNetwork, self).__init__()
        
        self.input_dim = input_dim
        
        # 构建网络层
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)  # 输出Q值
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier均匀分布权重初始化"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, global_state):
        """
        前向传播
        
        Args:
            global_state: 全局状态张量 (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Q值张量 (batch_size, 1)
        """
        x = torch.relu(self.fc1(global_state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 线性输出Q值
        return x