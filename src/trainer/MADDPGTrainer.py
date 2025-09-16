"""
MADDPG训练器

专注于求解充电站价格博弈的纳什均衡，实现三层结构：
- Episode层：博弈求解尝试
- Step层：智能体策略调整
- UE-DTA层：交通仿真响应（由环境提供）
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import logging
import json
from datetime import datetime

# 添加项目路径以便导入模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.algorithms.maddpg.maddpg import MADDPG
from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from src.utils.config import MADDPGConfig, TrainingConfig


class MADDPGTrainer:
    """
    MADDPG充电站价格博弈训练器
    
    专注于纳什均衡求解，采用清晰的三层结构：
    - Episode层：博弈求解尝试  
    - Step层：智能体策略调整
    - UE-DTA层：交通仿真响应
    """
    
    def __init__(self, maddpg_config: MADDPGConfig, training_config: TrainingConfig):
        """
        初始化MADDPGTrainer
        
        Args:
            maddpg_config: MADDPG算法配置
            training_config: 训练流程配置
        """
        self.config = training_config
        self.maddpg_config = maddpg_config
        
        # 处理设备配置
        if training_config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = training_config.device
            
        print(f"使用设备: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 1. 创建环境
        self.env = EVCSChargingGameEnv(
            network_dir=training_config.network_dir,
            network_name=training_config.network_name,
            random_seed=training_config.seed,
            max_steps=training_config.max_steps_per_episode,
            convergence_threshold=training_config.convergence_threshold,
            stable_steps_required=training_config.stable_steps_required
        )
        
        # 2. 从环境获取维度信息
        obs_space = self.env.observation_space(self.env.agents[0])
        obs_dim = sum(np.prod(space.shape) for space in obs_space.spaces.values())
        action_dim = self.env.action_space(self.env.agents[0]).shape[0]
        global_obs_dim = self.env.global_state_space().shape[0]
        
        # 3. 创建MADDPG算法
        self.maddpg = MADDPG(
            agent_ids=self.env.agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_obs_dim=global_obs_dim,
            buffer_capacity=maddpg_config.buffer_capacity,
            max_batch_size=maddpg_config.max_batch_size,
            actor_lr=maddpg_config.actor_lr,
            critic_lr=maddpg_config.critic_lr,
            gamma=maddpg_config.gamma,
            tau=maddpg_config.tau,
            seed=training_config.seed,
            device=self.device,
            actor_hidden_sizes=maddpg_config.actor_hidden_sizes,
            critic_hidden_sizes=maddpg_config.critic_hidden_sizes,
            noise_sigma=maddpg_config.noise_sigma,
            noise_decay=maddpg_config.noise_decay,
            min_noise=maddpg_config.min_noise
        )
        
        # 4. 训练状态跟踪
        self.convergence_episodes = []      # 收敛的episode列表
        self.episode_lengths = []           # 每个episode的长度
        self.step_records = []              # 每步详细记录（包含所有训练数据）
    
    def train(self) -> Dict:
        """
        主训练循环：寻找纳什均衡
        
        Episode层逻辑：每个episode都是对同一价格博弈的求解尝试
        成功标准：在单个episode中达到纳什均衡（价格收敛）
        
        Returns:
            Dict: 训练结果统计
        """
        convergence_count = 0
        
        with tqdm(total=self.config.max_episodes, desc="寻找纳什均衡", unit="episode", dynamic_ncols=True) as episode_pbar:
            
            for episode in range(self.config.max_episodes):
                
                # 重新初始化同一博弈（不是新博弈）
                observations, _ = self.env.reset()
                
                # Step层：策略调整循环
                converged_in_episode, episode_length = self._run_episode(episode, observations)
                
                # 记录episode统计
                self.episode_lengths.append(episode_length)
                
                if converged_in_episode:
                    convergence_count += 1
                    self.convergence_episodes.append(episode)
                    episode_pbar.set_postfix({
                        "收敛次数": convergence_count,
                        "收敛率": f"{convergence_count/(episode+1):.1%}"
                    })
                    
                    # 可选：如果连续多次收敛，可提前结束
                    if self._check_stable_convergence():
                        print(f"连续收敛，训练提前结束于episode {episode}")
                        break
                
                episode_pbar.update(1)
        
        # 生成训练结果
        results = self._generate_training_results()
        
        # 自动保存训练数据（使用配置的输出目录）
        experiment_dir = self.save_training_data(self.config.output_dir)
        results['experiment_dir'] = experiment_dir
        
        return results
    
    def _run_episode(self, episode: int, observations: Dict) -> Tuple[bool, int]:
        """
        Step层逻辑：在单个episode内调整智能体策略直到收敛或超时
        
        Args:
            episode: 当前episode编号
            observations: 初始观测
        
        Returns:
            Tuple[bool, int]: (是否收敛, episode长度)
        """
        with tqdm(total=self.config.max_steps_per_episode, 
                  desc=f"Episode {episode}", unit="step", leave=False, dynamic_ncols=True) as step_pbar:
            
            for step in range(self.config.max_steps_per_episode):
                
                # 智能体决策
                actions = self.maddpg.take_action(observations, add_noise=True)
                
                # 获取实际价格（在step之前）
                actual_prices = self.env.actions_to_prices_dict(actions)
                
                # 环境响应
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # 存储经验并学习
                self.maddpg.store_experience(observations, actions, rewards, next_observations, terminations)
                self.maddpg.learn()
                
                # 记录详细信息
                self.step_records.append({
                    'episode': episode,
                    'step': step,
                    'actions': actions.copy(),
                    'actual_prices': actual_prices.copy(),
                    'rewards': rewards.copy(),
                    'ue_info': infos,
                    'relative_change_rate': infos.get('relative_change_rate', float('inf'))
                })
                
                
                # 检查是否收敛（纳什均衡）
                if all(terminations.values()):
                    step_pbar.set_postfix({"状态": "收敛"})
                    step_pbar.update(self.config.max_steps_per_episode - step)
                    return True, step + 1
                
                # 更新观测
                observations = next_observations
                
                # 更新进度条
                step_pbar.set_postfix({
                    "UE迭代": infos.get('ue_iterations', 0),
                    "相对变化": f"{infos.get('relative_change_rate', float('inf')):.4f}"
                })
                step_pbar.update(1)
        
        # Episode超时未收敛
        step_pbar.set_postfix({"状态": "超时"})
        return False, self.config.max_steps_per_episode
    
    def get_nash_equilibrium(self) -> Dict:
        """
        获取所有纳什均衡解（从所有收敛的episode的稳定步骤中计算平均值）
        
        Returns:
            Dict: 包含所有纳什均衡解的统计，格式为：
                - status: 'converged' | 'no_convergence'
                - total_equilibria: 找到的均衡解数量
                - equilibria: List[Dict], 每个均衡解包含：
                    - episode: episode编号
                    - equilibrium_actions: 平均动作
                    - equilibrium_prices: 平均价格  
                    - equilibrium_rewards: 平均奖励
                    - stable_steps_count: 稳定步数
        """
        if not self.convergence_episodes:
            return {
                'status': 'no_convergence',
                'message': '未找到收敛的episode',
                'total_equilibria': 0,
                'equilibria': []
            }
        
        equilibria = []
        
        # 为每个收敛的episode计算纳什均衡解
        for episode_idx in self.convergence_episodes:
            # 从step_records中找到该episode的收敛步骤
            convergence_steps = [
                record for record in self.step_records 
                if record['episode'] == episode_idx
            ]
            
            if not convergence_steps:
                continue
            
            # 获取收敛时的稳定步骤数据（最后 stable_steps_required 步）
            stable_steps = convergence_steps[-self.config.stable_steps_required:]
            
            # 计算稳定步骤的平均动作
            equilibrium_actions = {}
            for agent in self.env.agents:
                agent_actions = []
                for step in stable_steps:
                    agent_actions.append(step['actions'][agent])
                # 计算平均动作
                equilibrium_actions[agent] = np.mean(agent_actions, axis=0)
            
            # 将平均动作转换为实际价格
            actual_prices = self.env.actions_to_prices_dict(equilibrium_actions)
            
            # 计算稳定步骤的平均奖励
            equilibrium_rewards = {}
            for agent in self.env.agents:
                agent_rewards = []
                for step in stable_steps:
                    agent_rewards.append(step['rewards'][agent])
                equilibrium_rewards[agent] = float(np.mean(agent_rewards))
            
            # 添加到均衡解列表
            equilibria.append({
                'episode': episode_idx,
                'final_step': stable_steps[-1]['step'],
                'stable_steps_count': len(stable_steps),
                'equilibrium_actions': equilibrium_actions,
                'equilibrium_prices': actual_prices,
                'equilibrium_rewards': equilibrium_rewards,
                'environment_info': stable_steps[-1]['ue_info']
            })
        
        return {
            'status': 'converged',
            'total_equilibria': len(equilibria),
            'equilibria': equilibria,
            'latest_equilibrium': equilibria[-1] if equilibria else None
        }
    
    def evaluate(self, num_episodes: int = 20) -> Dict:
        """
        评估训练效果：测试均衡稳定性
        
        Args:
            num_episodes: 评估轮数
        
        Returns:
            Dict: 评估结果
            
        TODO: 实现单方面偏离测试，验证纳什均衡性质
        """
        # 🔄 暂时留空，后续实现单方面偏离测试
        return {
            'status': 'not_implemented',
            'message': '将实现单方面偏离测试验证纳什均衡性质'
        }
    
    def _check_stable_convergence(self) -> bool:
        """
        检查是否达到稳定收敛（连续多次episode都收敛）
        
        Returns:
            bool: 是否连续收敛，可用于提前终止训练
        """
        if len(self.convergence_episodes) < self.config.stable_episodes_required:
            return False
        
        # 检查最近stable_episodes_required次episode是否都收敛
        recent_episodes = list(range(len(self.episode_lengths)))[-self.config.stable_episodes_required:]
        return all(ep in self.convergence_episodes for ep in recent_episodes)
    
    def _generate_training_results(self) -> Dict:
        """
        生成完整的训练结果统计
        
        Returns:
            Dict: 训练统计结果
        """
        total_episodes = len(self.episode_lengths)
        total_convergences = len(self.convergence_episodes)
        
        # 从step_records计算总UE迭代次数
        total_ue_iterations = sum(
            record['ue_info'].get('ue_iterations', 0) 
            for record in self.step_records
        )
        
        return {
            'total_episodes': total_episodes,
            'total_convergences': total_convergences,
            'convergence_rate': total_convergences / total_episodes if total_episodes > 0 else 0.0,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'total_ue_iterations': total_ue_iterations,
            'convergence_episodes': self.convergence_episodes,
            'final_nash_equilibrium': self.get_nash_equilibrium()
        }
    
    def save_training_data(self, output_dir: str = "results") -> str:
        """
        保存训练数据到JSON文件
        
        Args:
            output_dir: 输出根目录
            
        Returns:
            str: 保存的实验目录路径
        """
        # 创建带时间戳的实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 保存step_records到JSON文件
        step_records_path = os.path.join(experiment_dir, "step_records.json")
        
        # 准备保存数据，处理numpy数组
        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_episodes": len(self.episode_lengths),
                "total_steps": len(self.step_records),
                "convergence_episodes": self.convergence_episodes,
                "episode_lengths": self.episode_lengths
            },
            "records": []
        }
        
        # 转换step_records，处理numpy数组
        for record in self.step_records:
            converted_record = {
                "episode": int(record["episode"]),
                "step": int(record["step"]),
                "actions": {k: v.tolist() if hasattr(v, 'tolist') else v 
                           for k, v in record["actions"].items()},
                "actual_prices": {k: v.tolist() if hasattr(v, 'tolist') else v 
                                 for k, v in record["actual_prices"].items()},
                "rewards": {k: float(v) for k, v in record["rewards"].items()},
                "ue_info": record["ue_info"],
                "relative_change_rate": float(record["relative_change_rate"])
            }
            save_data["records"].append(converted_record)
        
        # 保存到JSON文件
        with open(step_records_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 训练数据已保存到: {experiment_dir}")
        print(f"   📁 实验目录: {experiment_dir}")
        print(f"   📄 数据文件: step_records.json")
        print(f"   📊 记录数量: {len(self.step_records)} 步")
        
        return experiment_dir