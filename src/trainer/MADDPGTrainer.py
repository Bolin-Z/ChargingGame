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
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import logging

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
        
        # 1. 创建环境
        self.env = EVCSChargingGameEnv(
            network_dir=training_config.network_dir,
            network_name=training_config.network_name,
            random_seed=training_config.seed,
            max_steps=training_config.max_steps_per_episode,
            convergence_threshold=training_config.convergence_threshold
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
            **maddpg_config.__dict__
        )
        
        # 4. 训练状态跟踪
        self.convergence_episodes = []      # 收敛的episode列表
        self.episode_lengths = []           # 每个episode的长度
        self.step_records = []              # 每步详细记录
        self.total_ue_iterations = 0        # 总UE仿真迭代次数
    
    def train(self) -> Dict:
        """
        主训练循环：寻找纳什均衡
        
        Episode层逻辑：每个episode都是对同一价格博弈的求解尝试
        成功标准：在单个episode中达到纳什均衡（价格收敛）
        
        Returns:
            Dict: 训练结果统计
        """
        convergence_count = 0
        
        with tqdm(total=self.config.max_episodes, desc="寻找纳什均衡", unit="episode") as episode_pbar:
            
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
        
        return self._generate_training_results()
    
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
                  desc=f"Episode {episode}", unit="step", leave=False) as step_pbar:
            
            for step in range(self.config.max_steps_per_episode):
                
                # 智能体决策
                actions = self.maddpg.take_action(observations, add_noise=True)
                
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
                    'rewards': rewards.copy(),
                    'ue_info': infos,
                    'relative_change_rate': infos.get('relative_change_rate', float('inf'))
                })
                
                # 累计UE迭代次数
                if 'ue_iterations' in infos:
                    self.total_ue_iterations += infos['ue_iterations']
                
                # 检查是否收敛（纳什均衡）
                if terminations.get('__all__', False):
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
        获取纳什均衡解（从收敛的episode中获取）
        
        Returns:
            Dict: 纳什均衡策略和统计，如果未找到收敛episode则返回None
        """
        if not self.convergence_episodes:
            return {
                'status': 'no_convergence',
                'message': '未找到收敛的episode'
            }
        
        # 获取最近一次收敛的episode
        latest_convergence_episode = self.convergence_episodes[-1]
        
        # 从step_records中找到该episode的收敛步骤（最后一步）
        convergence_steps = [
            record for record in self.step_records 
            if record['episode'] == latest_convergence_episode
        ]
        
        if not convergence_steps:
            return {
                'status': 'no_data',
                'message': f'未找到episode {latest_convergence_episode}的记录数据'
            }
        
        # 获取收敛时的最后一步数据
        final_step = convergence_steps[-1]
        equilibrium_actions = final_step['actions']
        
        # 将归一化动作转换为实际价格
        actual_prices = {}
        for agent_id, action in equilibrium_actions.items():
            actual_prices[agent_id] = self.env.actions_to_prices({agent_id: action})[int(agent_id)]
        
        return {
            'status': 'converged',
            'episode': latest_convergence_episode,
            'step': final_step['step'],
            'equilibrium_actions': equilibrium_actions,
            'equilibrium_prices': actual_prices,
            'equilibrium_rewards': final_step['rewards'],
            'environment_info': final_step['ue_info']
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
        if len(self.convergence_episodes) < 3:
            return False
        
        # 检查最近3次episode是否都收敛
        recent_episodes = list(range(len(self.episode_lengths)))[-3:]
        return all(ep in self.convergence_episodes for ep in recent_episodes)
    
    def _generate_training_results(self) -> Dict:
        """
        生成完整的训练结果统计
        
        Returns:
            Dict: 训练统计结果
        """
        total_episodes = len(self.episode_lengths)
        total_convergences = len(self.convergence_episodes)
        
        return {
            'total_episodes': total_episodes,
            'total_convergences': total_convergences,
            'convergence_rate': total_convergences / total_episodes if total_episodes > 0 else 0.0,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'total_ue_iterations': self.total_ue_iterations,
            'convergence_episodes': self.convergence_episodes,
            'final_nash_equilibrium': self.get_nash_equilibrium()
        }