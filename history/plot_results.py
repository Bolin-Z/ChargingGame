"""
结果可视化脚本

从训练数据JSON文件中读取并绘制Episode 0的结果：
1. 各充电站奖励变化曲线
2. 各充电站在不同时段的价格演化趋势

使用方法：
    python plot_results.py
    python plot_results.py --data results/experiment_20250917_025705/step_records.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(json_path: str):
    """加载训练数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_reward_curves(data: dict, output_path: str):
    """绘制各充电站奖励变化曲线（Episode 0）"""
    episode_idx = 0
    episode_records = [r for r in data['records'] if r['episode'] == episode_idx]

    if not episode_records:
        print(f"未找到Episode {episode_idx}的数据")
        return

    # 提取充电站列表
    agents = list(episode_records[0]['rewards'].keys())

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个充电站绘制奖励曲线
    colors = ['red', 'orange', 'blue', 'green']
    for idx, agent in enumerate(agents):
        rewards = [r['rewards'][agent] for r in episode_records]
        steps = [r['step'] for r in episode_records]

        ax.plot(steps, rewards,
                color=colors[idx % len(colors)],
                label=f'充电站 {agent}',
                linewidth=1.5)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('奖励 (元)', fontsize=12)
    ax.set_title(f'Episode {episode_idx} - 各充电站奖励变化趋势', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 奖励曲线图已保存到: {output_path}")
    plt.close()


def plot_price_evolution(data: dict, output_path: str):
    """绘制各充电站在不同时段的价格演化趋势（Episode 0）"""
    episode_idx = 0
    episode_records = [r for r in data['records'] if r['episode'] == episode_idx]

    if not episode_records:
        print(f"未找到Episode {episode_idx}的数据")
        return

    # 提取充电站列表和时段数
    agents = list(episode_records[0]['actual_prices'].keys())
    n_periods = len(episode_records[0]['actual_prices'][agents[0]])

    # 创建子图 (2行4列，共8个时段)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # 为每个充电站准备颜色
    colors = ['red', 'orange', 'blue', 'green']

    # 为每个时段绘制价格演化
    for period_idx in range(n_periods):
        ax = axes[period_idx]

        for agent_idx, agent in enumerate(agents):
            prices = [r['actual_prices'][agent][period_idx] for r in episode_records]
            steps = [r['step'] for r in episode_records]

            ax.plot(steps, prices,
                   color=colors[agent_idx % len(colors)],
                   label=f'充电站 {agent}',
                   linewidth=1.0,
                   alpha=0.8)

        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('价格 (元)', fontsize=9)
        ax.set_title(f'时段 {period_idx + 1}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 添加总标题
    fig.suptitle(f'Episode {episode_idx} - 各充电站实际价格变化趋势',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 价格演化图已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制Episode 0的训练结果')
    parser.add_argument('--data', type=str, default='results/step_records.json',
                       help='训练数据JSON文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认为数据文件所在目录）')

    args = parser.parse_args()

    # 加载数据
    print(f"📊 加载训练数据: {args.data}")
    data = load_data(args.data)

    print(f"   总Episodes: {data['metadata']['total_episodes']}")
    print(f"   总Steps: {data['metadata']['total_steps']}")
    print(f"   收敛Episodes: {data['metadata']['convergence_episodes']}")
    print()

    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data)

    os.makedirs(args.output_dir, exist_ok=True)

    # 绘制Episode 0的图表
    print(f"🎨 绘制Episode 0的图表...")

    # 1. 奖励曲线图
    curve_path = os.path.join(args.output_dir, 'curve.png')
    plot_reward_curves(data, curve_path)

    # 2. 价格演化图
    tou_path = os.path.join(args.output_dir, 'tou.png')
    plot_price_evolution(data, tou_path)

    print(f"\n✨ 完成！所有图表已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
