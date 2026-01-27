"""
单次实验运行脚本

用法：
    python run_experiment.py --network siouxfalls --algo MADDPG --seed 42
    python run_experiment.py --network siouxfalls --algo IDDPG --seed 42 --no-monitor
    python run_experiment.py --help

参数说明：
    --network: 网络名称 (siouxfalls / berlin_friedrichshain / anaheim)
    --algo: 算法名称 (MADDPG / IDDPG / MFDDPG)
    --seed: 随机种子 (默认 42)
    --max-evals: 最大评估次数 (默认 1000)
    --workers: 并行 Worker 数 (默认自动)
    --no-monitor: 禁用实时监控
"""

import argparse
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行充电站价格博弈实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_experiment.py --network siouxfalls --algo MADDPG --seed 42
    python run_experiment.py --network siouxfalls --algo IDDPG --seed 123 --no-monitor
    python run_experiment.py --network berlin_friedrichshain --algo MFDDPG --max-evals 2000
        """,
    )

    parser.add_argument(
        "--network",
        type=str,
        required=True,
        choices=["siouxfalls", "berlin_friedrichshain", "anaheim"],
        help="网络名称",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["MADDPG", "IDDPG", "MFDDPG"],
        help="算法名称",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=1000,
        help="最大评估次数 (默认: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行 Worker 数，-1 表示自动 (默认: 4)",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="禁用实时监控窗口",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备 (默认: auto)",
    )

    return parser.parse_args()


def get_trainer_config(network: str, max_evals: int, workers: int, device: str):
    """根据网络名称获取训练器配置"""
    from src.utils.config_v1 import TrainerConfig

    # 网络数据目录映射
    network_dirs = {
        "siouxfalls": "data/siouxfalls",
        "berlin_friedrichshain": "data/berlin_friedrichshain",
        "anaheim": "data/anaheim",
    }

    return TrainerConfig(
        network_dir=network_dirs[network],
        network_name=network,
        max_evaluations=max_evals,
        learn_interval=5,
        device=device,
        n_workers=workers,
    )


def get_algo_config(algo: str):
    """根据算法名称获取算法配置"""
    from src.utils.config_v1 import MADDPGConfig, IDDPGConfig, MFDDPGConfig

    configs = {
        "MADDPG": MADDPGConfig(),
        "IDDPG": IDDPGConfig(),
        "MFDDPG": MFDDPGConfig(),
    }
    return configs[algo]


def main():
    args = parse_args()

    print("=" * 60)
    print("充电站价格博弈实验")
    print("=" * 60)
    print(f"网络: {args.network}")
    print(f"算法: {args.algo}")
    print(f"种子: {args.seed}")
    print(f"最大评估次数: {args.max_evals}")
    print(f"并行 Workers: {'自动' if args.workers == -1 else args.workers}")
    print(f"实时监控: {'禁用' if args.no_monitor else '启用'}")
    print(f"计算设备: {args.device}")
    print("=" * 60)

    # 导入必要模块
    from src.utils.config_v1 import ExperimentTask, NashConvConfig, MonitorConfig
    from src.trainer.game_trainer_v1 import GameTrainer, create_algorithm
    from src.evaluator.network_data import NetworkDataLoader

    # 构建配置
    trainer_config = get_trainer_config(
        args.network, args.max_evals, args.workers, args.device
    )
    algo_config = get_algo_config(args.algo)
    nashconv_config = NashConvConfig()
    monitor_config = MonitorConfig(enabled=not args.no_monitor)

    # 创建实验任务
    task = ExperimentTask(
        name=f"{args.network}_{args.algo}_seed{args.seed}",
        trainer_config=trainer_config,
        algo_name=args.algo,
        algo_config=algo_config,
        seed=args.seed,
        nashconv_config=nashconv_config,
    )

    # 加载网络数据
    print("\n加载网络数据...")
    loader = NetworkDataLoader(
        network_dir=trainer_config.network_dir,
        network_name=trainer_config.network_name,
        random_seed=args.seed,
    )
    network_data = loader.load()
    print(f"  智能体数量: {len(network_data.agent_names)}")
    print(f"  时段数量: {network_data.n_periods}")

    # 创建算法
    print("\n创建算法...")
    algorithm = create_algorithm(task, network_data)
    print(f"  算法类型: {type(algorithm).__name__}")

    # 创建训练器
    print("\n创建训练器...")
    trainer = GameTrainer(task, algorithm, monitor_config)

    # 开始训练
    print("\n开始训练...")
    print("-" * 60)
    result = trainer.train()
    print("-" * 60)

    # 打印结果
    print("\n" + "=" * 60)
    print("训练结果")
    print("=" * 60)
    print(f"收敛状态: {'是' if result.converged else '否'}")
    print(f"总评估次数: {result.metrics.total_evaluations}")
    print(f"总学习次数: {result.metrics.total_learns}")
    print(f"总耗时: {result.total_time:.1f} 秒")

    if result.final_exploitability is not None:
        print(f"最终 Exploitability: {result.final_exploitability:.4f}")

    if result.final_rewards:
        total_reward = sum(result.final_rewards.values())
        print(f"最终总收益: {total_reward:.2f}")

    print("=" * 60)

    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
