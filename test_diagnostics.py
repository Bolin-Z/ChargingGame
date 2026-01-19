"""
诊断功能快速测试脚本

运行 1 episode 300 steps，验证诊断指标是否正常工作
"""

import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from dataclasses import dataclass
from src.utils.config import MADDPGConfig, ScenarioProfile, ExperimentTask
from src.trainer.MADDPGTrainer import MADDPGTrainer


# 创建快速测试用的场景配置
TEST_PROFILE = ScenarioProfile(
    network_dir='data/siouxfalls',
    network_name='siouxfalls',
    max_episodes=1,              # 只跑 1 episode
    max_steps_per_episode=300,   # 300 steps
    convergence_threshold=0.01,
    stable_steps_required=5,
    stable_episodes_required=1,
)


def main():
    """快速测试主函数"""
    print("=" * 60)
    print("诊断功能快速测试")
    print("配置: 1 episode, 300 steps")
    print("=" * 60)

    # 创建实验任务
    task = ExperimentTask(
        name="DiagnosticsTest",
        scenario=TEST_PROFILE,
        algo_name="MADDPG",
        algo_config=MADDPGConfig(),
        seed=42
    )

    # 创建训练器并运行
    print("\n初始化训练器...")
    trainer = MADDPGTrainer(task)

    print("开始训练...\n")
    results = trainer.train()

    # 打印关键结果
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"总 Episodes: {results['total_episodes']}")
    print(f"收敛次数: {results['total_convergences']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
