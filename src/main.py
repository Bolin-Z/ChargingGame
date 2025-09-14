"""
充电站价格博弈MADDPG求解主流程

这是项目的主入口文件，实现完整的训练流程：
1. 环境初始化
2. MADDPG智能体创建  
3. 训练循环执行
4. 结果分析和保存

运行方式：
    python src/main.py

支持的算法：
    - MADDPG: 多智能体深度确定性策略梯度（当前实现）
    - 其他算法（未来扩展）
"""

import os
import sys
import torch
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, save_config
from src.utils.logger import create_experiment_logger
from src.trainer import MADDPGTrainer


def setup_device(device_arg: str) -> str:
    """设置计算设备"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"使用计算设备: {device}")
    if device == 'cuda':
        print(f"CUDA设备信息: {torch.cuda.get_device_name(0)}")
    
    return device


def create_output_structure(config):
    """创建实验输出目录结构"""
    # 创建带时间戳的实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"MADDPG_EVCS_{timestamp}"
    experiment_dir = os.path.join(config.output_dir, experiment_name)
    
    # 更新配置中的输出目录
    config.output_dir = experiment_dir
    
    # 创建目录结构
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def print_experiment_info(config, experiment_dir):
    """打印实验信息"""
    print("=" * 80)
    print("EVCS 充电站价格博弈 MADDPG 求解器")
    print("=" * 80)
    print(f"实验目录: {experiment_dir}")
    print(f"网络数据: {config.network_dir}/{config.network_name}")
    print(f"最大训练轮数: {config.max_episodes}")
    print(f"每轮最大步数: {config.max_steps_per_episode}")
    print(f"计算设备: {config.device}")
    print(f"随机种子: {config.seed}")
    print("-" * 80)
    print("MADDPG 参数:")
    print(f"  Actor学习率: {config.actor_lr}")
    print(f"  Critic学习率: {config.critic_lr}")
    print(f"  折扣因子 γ: {config.gamma}")
    print(f"  软更新系数 τ: {config.tau}")
    print(f"  经验回放容量: {config.buffer_capacity}")
    print("=" * 80)


def save_experiment_results(trainer, config, experiment_dir, training_results):
    """保存实验完整结果"""
    # 1. 保存训练配置
    config_path = os.path.join(experiment_dir, 'config.json')
    save_config(config, config_path)
    
    # 2. 保存训练统计
    stats_path = os.path.join(experiment_dir, 'results', 'training_stats.json')
    stats = {
        'training_results': training_results,
        'episode_rewards': trainer.episode_rewards,
        'convergence_episodes': trainer.convergence_episodes,
        'training_config': config.__dict__
    }
    
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 3. 获取并保存纳什均衡解
    nash_equilibrium = trainer.get_nash_equilibrium()
    nash_path = os.path.join(experiment_dir, 'results', 'nash_equilibrium.json')
    with open(nash_path, 'w', encoding='utf-8') as f:
        json.dump(nash_equilibrium, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验结果已保存到: {experiment_dir}")
    print(f"- 配置文件: config.json")
    print(f"- 训练统计: results/training_stats.json")
    print(f"- 纳什均衡解: results/nash_equilibrium.json")
    print(f"- 模型文件: models/final_model.pt")


def print_training_summary(training_results, trainer):
    """打印训练总结"""
    print("\n" + "=" * 80)
    print("训练完成总结")
    print("=" * 80)
    print(f"总训练轮数: {training_results['total_episodes']}")
    print(f"收敛次数: {training_results['total_convergences']}")
    print(f"收敛率: {training_results['convergence_rate']:.2%}")
    print(f"平均奖励: {training_results['average_reward']:.2f}")
    print(f"平均轮次长度: {training_results['average_episode_length']:.2f}")
    
    # 显示最近几次的表现
    if len(trainer.episode_rewards) >= 10:
        recent_rewards = trainer.episode_rewards[-10:]
        print(f"最近10轮平均奖励: {sum(recent_rewards)/len(recent_rewards):.2f}")
    
    if trainer.convergence_episodes:
        print(f"最后收敛轮次: {trainer.convergence_episodes[-1]}")
    
    print("=" * 80)


def run_maddpg_experiment(config):
    """运行MADDPG实验"""
    # 创建实验目录
    experiment_dir = create_output_structure(config)
    
    # 设置设备
    config.device = setup_device(config.device)
    
    # 打印实验信息
    print_experiment_info(config, experiment_dir)
    
    # 创建实验日志记录器
    logger = create_experiment_logger('MADDPG_EVCS', experiment_dir)
    
    logger.info("开始MADDPG充电站价格博弈实验")
    logger.info(f"实验配置: {config.__dict__}")
    
    try:
        # 创建MADDPG训练器
        trainer = MADDPGTrainer(config, logger)
        
        # 执行训练
        logger.info("开始训练...")
        training_results = trainer.train()
        
        # 运行评估
        logger.info("开始最终评估...")
        evaluation_results = trainer.evaluate(num_episodes=20)
        logger.info(f"评估结果: {evaluation_results}")
        
        # 保存所有结果
        training_results['evaluation'] = evaluation_results
        save_experiment_results(trainer, config, experiment_dir, training_results)
        
        # 打印总结
        print_training_summary(training_results, trainer)
        
        logger.info("实验完成成功!")
        return training_results
        
    except Exception as e:
        logger.error(f"实验过程中发生错误: {e}")
        print(f"实验失败: {e}")
        raise


def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        
        # 运行MADDPG实验
        results = run_maddpg_experiment(config)
        
        print("\n🎉 实验成功完成!")
        print("\n📊 关键结果:")
        print(f"   收敛率: {results['convergence_rate']:.2%}")
        print(f"   平均奖励: {results['average_reward']:.2f}")
        if 'evaluation' in results:
            print(f"   评估收敛率: {results['evaluation']['convergence_rate']:.2%}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断实验")
        return 1
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)