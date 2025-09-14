"""
充电站价格博弈MADDPG训练主入口程序

简化版本：
- 所有配置通过修改 src/utils/config.py 文件进行
- 直接运行 python main.py 开始训练
- 专注于求解电动汽车充电站价格博弈中的纳什均衡解
"""

import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.utils.config import get_default_configs
from src.trainer.MADDPGTrainer import MADDPGTrainer


def print_results(results):
    """
    打印训练结果摘要
    
    Args:
        results: 训练结果字典
    """
    print("=" * 60)
    print("🎯 MADDPG充电站价格博弈训练完成")
    print("=" * 60)
    
    # 基础统计
    print(f"📊 训练统计:")
    print(f"   总Episode数: {results['total_episodes']}")
    print(f"   收敛Episode数: {results['total_convergences']}")
    print(f"   收敛率: {results['convergence_rate']:.1%}")
    print(f"   平均Episode长度: {results['average_episode_length']:.1f}步")
    print(f"   总UE仿真迭代数: {results['total_ue_iterations']}")
    
    # 收敛Episode列表
    if results['convergence_episodes']:
        print(f"✅ 收敛的Episodes: {results['convergence_episodes']}")
    else:
        print("❌ 未找到收敛的Episode")
    
    # 纳什均衡解
    nash_eq = results['final_nash_equilibrium']
    if nash_eq['status'] == 'converged':
        print(f"🎉 找到纳什均衡解!")
        print(f"   收敛Episode: {nash_eq['episode']}")
        print(f"   收敛步骤: {nash_eq['step']}")
        print(f"💰 均衡价格策略:")
        for agent_id, prices in nash_eq['equilibrium_prices'].items():
            price_str = ", ".join([f"{p:.3f}" for p in prices])
            print(f"   充电站{agent_id}: [{price_str}]")
        print(f"💵 均衡收益:")
        for agent_id, reward in nash_eq['equilibrium_rewards'].items():
            print(f"   充电站{agent_id}: {reward:.2f}")
    else:
        print(f"⚠️  未找到纳什均衡: {nash_eq['message']}")
    
    print("=" * 60)


def main():
    """主入口函数"""
    try:
        print("🚀 启动MADDPG充电站价格博弈训练")
        print(f"📁 项目根目录: {project_root}")
        
        # 1. 加载默认配置
        print("⚙️  加载配置...")
        maddpg_config, training_config = get_default_configs()
        
        print(f"   训练配置: 最大{training_config.max_episodes}个Episodes, "
              f"收敛阈值{training_config.convergence_threshold}, "
              f"随机种子{training_config.seed}")
        print(f"   算法配置: Actor-LR={maddpg_config.actor_lr}, "
              f"Critic-LR={maddpg_config.critic_lr}, "
              f"噪音强度={maddpg_config.noise_sigma}")
        
        # 2. 创建训练器
        print("🏗️  初始化训练器...")
        trainer = MADDPGTrainer(maddpg_config, training_config)
        
        # 3. 执行训练
        print("🎯 开始寻找纳什均衡...")
        print()  # 空行，为训练进度条留出空间
        
        results = trainer.train()
        
        # 4. 打印结果摘要
        print()  # 空行分隔
        print_results(results)
        
        print("✅ 训练程序成功完成!")
        
    except KeyboardInterrupt:
        print("\n⚡ 用户中断训练")
        return 1
        
    except Exception as e:
        print(f"❌ 训练过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())