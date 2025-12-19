"""
单实验运行脚本

通过命令行参数指定算法和场景，运行单个实验。

用法:
    python run_single_experiment.py --algo maddpg --scenario siouxfalls
    python run_single_experiment.py -a iddpg -s berlin --seed 123
"""

import sys
import os
import argparse

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.utils.config import (
    MADDPGConfig, IDDPGConfig, MFDDPGConfig,
    PROFILE_SIOUXFALLS, PROFILE_BERLIN, PROFILE_ANAHEIM,
    ExperimentTask
)

# 场景映射
SCENARIOS = {
    'siouxfalls': PROFILE_SIOUXFALLS,
    'sf': PROFILE_SIOUXFALLS,
    'berlin': PROFILE_BERLIN,
    'bf': PROFILE_BERLIN,
    'anaheim': PROFILE_ANAHEIM,
    'ah': PROFILE_ANAHEIM,
}

# 算法映射
ALGORITHMS = {
    'maddpg': ('MADDPG', MADDPGConfig),
    'iddpg': ('IDDPG', IDDPGConfig),
    'mfddpg': ('MFDDPG', MFDDPGConfig),
}


def get_trainer_class(algo_name: str):
    """根据算法名称获取对应的Trainer类"""
    if algo_name == 'MADDPG':
        from src.trainer.MADDPGTrainer import MADDPGTrainer
        return MADDPGTrainer
    elif algo_name == 'IDDPG':
        from src.trainer.IDDPGTrainer import IDDPGTrainer
        return IDDPGTrainer
    elif algo_name == 'MFDDPG':
        from src.trainer.MFDDPGTrainer import MFDDPGTrainer
        return MFDDPGTrainer
    else:
        raise ValueError(f"未知算法: {algo_name}")


def print_results(results, experiment_id):
    """打印训练结果摘要"""
    print("=" * 60)
    print(f"[{experiment_id}] 训练完成")
    print("=" * 60)
    print(f"  总Episode数: {results['total_episodes']}")
    print(f"  收敛Episode数: {results['total_convergences']}")
    print(f"  收敛率: {results['convergence_rate']:.1%}")
    print(f"  平均Episode长度: {results['average_episode_length']:.1f}步")

    nash_eq = results['final_nash_equilibrium']
    if nash_eq['status'] == 'converged':
        print(f"  找到 {nash_eq['total_equilibria']} 个纳什均衡解")
    else:
        print(f"  未找到纳什均衡")
    print("=" * 60)


def run_experiment(algo_key: str, scenario_key: str, seed: int):
    """运行单个实验"""
    # 解析参数
    algo_name, algo_config_class = ALGORITHMS[algo_key.lower()]
    scenario = SCENARIOS[scenario_key.lower()]

    experiment_id = f"{algo_name}_{scenario.network_name}"

    print("=" * 60)
    print(f"[{experiment_id}] 启动实验")
    print("=" * 60)
    print(f"  算法: {algo_name}")
    print(f"  场景: {scenario.network_name}")
    print(f"  种子: {seed}")
    print(f"  最大Episodes: {scenario.max_episodes}")
    print("=" * 60)

    # 创建实验任务
    task = ExperimentTask(
        name=experiment_id,
        scenario=scenario,
        algo_name=algo_name,
        algo_config=algo_config_class(),
        seed=seed
    )

    # 获取Trainer并运行
    TrainerClass = get_trainer_class(algo_name)
    trainer = TrainerClass(task)
    results = trainer.train()

    # 打印结果
    print_results(results, experiment_id)

    return results


def main():
    parser = argparse.ArgumentParser(description='运行单个充电站博弈实验')
    parser.add_argument('--algo', '-a', required=True,
                       choices=['maddpg', 'iddpg', 'mfddpg'],
                       help='算法名称')
    parser.add_argument('--scenario', '-s', required=True,
                       choices=['siouxfalls', 'sf', 'berlin', 'bf', 'anaheim', 'ah'],
                       help='场景名称')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')

    args = parser.parse_args()

    try:
        run_experiment(args.algo, args.scenario, args.seed)
    except KeyboardInterrupt:
        print("\n用户中断实验")
        return 1
    except Exception as e:
        print(f"\n实验出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
