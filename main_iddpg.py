"""
充电站价格博弈IDDPG训练主入口程序

IDDPG (Independent DDPG):
- 完全去中心化训练
- Critic使用局部状态（48维）
- 每个agent维护独立的经验回放Buffer
- 专注于求解电动汽车充电站价格博弈中的纳什均衡解
"""

import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.utils.config import get_iddpg_config, get_training_config
from src.trainer.IDDPGTrainer import IDDPGTrainer


def print_results(results):
    """
    打印训练结果摘要

    Args:
        results: 训练结果字典
    """
    print("=" * 60)
    print("🎯 IDDPG充电站价格博弈训练完成")
    print("=" * 60)

    # 基础统计
    print(f"📊 训练统计:")
    print(f"   算法类型: {results['algorithm']}")
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
        total_equilibria = nash_eq['total_equilibria']
        print(f"🎉 找到 {total_equilibria} 个纳什均衡解!")

        # 显示所有均衡解
        for i, equilibrium in enumerate(nash_eq['equilibria'], 1):
            print(f"\n📊 均衡解 #{i}:")
            print(f"   收敛Episode: {equilibrium['episode']}")
            print(f"   收敛步骤: {equilibrium['final_step']}")
            print(f"   稳定步数: {equilibrium['stable_steps_count']}")
            print(f"💰 均衡价格策略:")
            for agent_id, prices in equilibrium['equilibrium_prices'].items():
                price_str = ", ".join([f"{p:.3f}" for p in prices])
                print(f"   充电站{agent_id}: [{price_str}]")
            print(f"💵 均衡收益:")
            for agent_id, reward in equilibrium['equilibrium_rewards'].items():
                print(f"   充电站{agent_id}: {reward:.2f}")
    else:
        print(f"⚠️  未找到纳什均衡: {nash_eq.get('message', '未知错误')}")

    print("=" * 60)


def main():
    """主入口函数"""
    try:
        print("🚀 启动IDDPG充电站价格博弈训练")
        print(f"📁 项目根目录: {project_root}")

        # 1. 加载配置
        print("⚙️  加载配置...")
        iddpg_config = get_iddpg_config()
        training_config = get_training_config()

        print(f"   训练配置: 最大{training_config.max_episodes}个Episodes, "
              f"收敛阈值{training_config.convergence_threshold}, "
              f"随机种子{training_config.seed}")
        print(f"   算法配置: Actor-LR={iddpg_config.actor_lr}, "
              f"Critic-LR={iddpg_config.critic_lr}, "
              f"噪音强度={iddpg_config.noise_sigma}")
        print(f"   IDDPG特点: 完全去中心化训练，局部状态Critic（48维）")

        # 2. 创建训练器
        print("🏗️  初始化训练器...")
        trainer = IDDPGTrainer(iddpg_config, training_config)

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