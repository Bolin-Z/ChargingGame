"""
MADDPG vs IDDPG 算法对比可视化脚本

对比两个算法在充电站价格博弈中的表现：
1. Episode 0 奖励变化对比
2. 特定充电站的价格演化对比
3. Episode 长度和收敛速度对比

使用方法：
    python plot_comparison.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(json_path: str):
    """加载训练数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_reward_comparison(maddpg_data: dict, iddpg_data: dict, output_path: str):
    """绘制两个算法在Episode 0的奖励对比图"""
    episode_idx = 0

    # 提取 MADDPG 数据
    maddpg_records = [r for r in maddpg_data['records'] if r['episode'] == episode_idx]
    # 提取 IDDPG 数据
    iddpg_records = [r for r in iddpg_data['records'] if r['episode'] == episode_idx]

    if not maddpg_records or not iddpg_records:
        print(f"⚠️ 未找到Episode {episode_idx}的数据")
        return

    # 提取充电站列表
    agents = list(maddpg_records[0]['rewards'].keys())
    n_agents = len(agents)

    # 创建子图 (每个充电站一个子图)
    fig, axes = plt.subplots(n_agents, 1, figsize=(14, 4 * n_agents))
    if n_agents == 1:
        axes = [axes]

    for idx, agent in enumerate(agents):
        ax = axes[idx]

        # MADDPG 数据
        maddpg_rewards = [r['rewards'][agent] for r in maddpg_records]
        maddpg_steps = [r['step'] for r in maddpg_records]

        # IDDPG 数据
        iddpg_rewards = [r['rewards'][agent] for r in iddpg_records]
        iddpg_steps = [r['step'] for r in iddpg_records]

        # 绘制曲线
        ax.plot(maddpg_steps, maddpg_rewards,
                color='#2E86AB', label='MADDPG', linewidth=2.0, alpha=0.8)
        ax.plot(iddpg_steps, iddpg_rewards,
                color='#A23B72', label='IDDPG', linewidth=2.0, alpha=0.8)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('奖励 (元)', fontsize=11)
        ax.set_title(f'充电站 {agent} - 奖励变化对比', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Episode {episode_idx} - MADDPG vs IDDPG 奖励对比',
                 fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 奖励对比图已保存到: {output_path}")
    plt.close()


def plot_price_comparison(maddpg_data: dict, iddpg_data: dict, output_path: str,
                          agent_id: str = None, period_idx: int = 0):
    """绘制特定充电站在特定时段的价格演化对比"""
    episode_idx = 0

    # 提取数据
    maddpg_records = [r for r in maddpg_data['records'] if r['episode'] == episode_idx]
    iddpg_records = [r for r in iddpg_data['records'] if r['episode'] == episode_idx]

    if not maddpg_records or not iddpg_records:
        print(f"⚠️ 未找到Episode {episode_idx}的数据")
        return

    # 如果未指定充电站，选择第一个
    if agent_id is None:
        agent_id = list(maddpg_records[0]['actual_prices'].keys())[0]

    agents = list(maddpg_records[0]['actual_prices'].keys())
    n_periods = len(maddpg_records[0]['actual_prices'][agent_id])

    # 创建子图 (2行4列，8个时段)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    # 为每个充电站准备颜色
    agent_colors = {
        agents[0]: '#2E86AB',  # 蓝色
        agents[1]: '#F18F01',  # 橙色
        agents[2]: '#C73E1D',  # 红色
        agents[3]: '#6A994E',  # 绿色
    }

    # 为每个时段绘制价格演化
    for period in range(n_periods):
        ax = axes[period]

        for agent in agents:
            # MADDPG 数据
            maddpg_prices = [r['actual_prices'][agent][period] for r in maddpg_records]
            maddpg_steps = [r['step'] for r in maddpg_records]

            # IDDPG 数据
            iddpg_prices = [r['actual_prices'][agent][period] for r in iddpg_records]
            iddpg_steps = [r['step'] for r in iddpg_records]

            # 绘制 MADDPG (实线)
            ax.plot(maddpg_steps, maddpg_prices,
                   color=agent_colors[agent],
                   label=f'充电站{agent} (MADDPG)',
                   linewidth=1.5,
                   linestyle='-',
                   alpha=0.8)

            # 绘制 IDDPG (虚线)
            ax.plot(iddpg_steps, iddpg_prices,
                   color=agent_colors[agent],
                   label=f'充电站{agent} (IDDPG)',
                   linewidth=1.5,
                   linestyle='--',
                   alpha=0.8)

        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('价格 (元)', fontsize=9)
        ax.set_title(f'时段 {period + 1}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Episode {episode_idx} - MADDPG vs IDDPG 价格演化对比\n(实线: MADDPG, 虚线: IDDPG)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 价格对比图已保存到: {output_path}")
    plt.close()


def plot_convergence_comparison(maddpg_data: dict, iddpg_data: dict, output_path: str):
    """绘制收敛速度对比（Episode长度统计）"""
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # 子图1: Episode 长度对比
    ax1 = fig.add_subplot(gs[0, 0])

    maddpg_lengths = maddpg_data['metadata']['episode_lengths']
    iddpg_lengths = iddpg_data['metadata']['episode_lengths']

    episodes = range(len(maddpg_lengths))
    x = np.arange(len(episodes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, maddpg_lengths, width,
                    label='MADDPG', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, iddpg_lengths, width,
                    label='IDDPG', color='#A23B72', alpha=0.8)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('收敛所需步数', fontsize=12)
    ax1.set_title('各Episode收敛速度对比', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Ep{i}' for i in episodes])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱状图上显示数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    # 子图2: 统计信息对比
    ax2 = fig.add_subplot(gs[0, 1])

    stats = {
        'MADDPG': {
            '平均Episode长度': np.mean(maddpg_lengths),
            '总Steps': maddpg_data['metadata']['total_steps'],
            '收敛Episodes': len(maddpg_data['metadata']['convergence_episodes']),
        },
        'IDDPG': {
            '平均Episode长度': np.mean(iddpg_lengths),
            '总Steps': iddpg_data['metadata']['total_steps'],
            '收敛Episodes': len(iddpg_data['metadata']['convergence_episodes']),
        }
    }

    # 绘制统计表格
    ax2.axis('off')

    table_data = []
    table_data.append(['指标', 'MADDPG', 'IDDPG', '差异'])

    for key in ['平均Episode长度', '总Steps', '收敛Episodes']:
        maddpg_val = stats['MADDPG'][key]
        iddpg_val = stats['IDDPG'][key]

        if key == '平均Episode长度':
            diff = f'{iddpg_val - maddpg_val:+.1f}'
            maddpg_str = f'{maddpg_val:.1f}'
            iddpg_str = f'{iddpg_val:.1f}'
        else:
            diff = f'{int(iddpg_val - maddpg_val):+d}'
            maddpg_str = f'{int(maddpg_val)}'
            iddpg_str = f'{int(iddpg_val)}'

        table_data.append([key, maddpg_str, iddpg_str, diff])

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.22, 0.22, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold')

    # 设置数据行样式
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor('#F5F5F5')
            elif j == 1:
                cell.set_facecolor('#D6E9F5')  # MADDPG 蓝色
            elif j == 2:
                cell.set_facecolor('#F5D6E8')  # IDDPG 粉色

    ax2.set_title('训练统计对比', fontsize=13, fontweight='bold', pad=20)

    fig.suptitle('MADDPG vs IDDPG - 收敛性能对比',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 收敛对比图已保存到: {output_path}")
    plt.close()


def main():
    # 数据路径
    maddpg_path = 'results/experiment_20250917_025705/step_records.json'
    iddpg_path = 'results/iddpg_experiment_20251117_202408/step_records.json'

    # 输出目录
    output_dir = 'results/comparison'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("📊 加载训练数据...")
    print(f"   MADDPG: {maddpg_path}")
    maddpg_data = load_data(maddpg_path)
    print(f"   └─ Episodes: {maddpg_data['metadata']['total_episodes']}, "
          f"Steps: {maddpg_data['metadata']['total_steps']}")

    print(f"   IDDPG: {iddpg_path}")
    iddpg_data = load_data(iddpg_path)
    print(f"   └─ Episodes: {iddpg_data['metadata']['total_episodes']}, "
          f"Steps: {iddpg_data['metadata']['total_steps']}")
    print()

    # 绘制对比图表
    print("🎨 绘制对比图表...")

    # 1. 奖励对比图
    print("   1/3 绘制奖励对比图...")
    reward_path = os.path.join(output_dir, 'reward_comparison.png')
    plot_reward_comparison(maddpg_data, iddpg_data, reward_path)

    # 2. 价格演化对比图
    print("   2/3 绘制价格演化对比图...")
    price_path = os.path.join(output_dir, 'price_comparison.png')
    plot_price_comparison(maddpg_data, iddpg_data, price_path)

    # 3. 收敛速度对比图
    print("   3/3 绘制收敛速度对比图...")
    convergence_path = os.path.join(output_dir, 'convergence_comparison.png')
    plot_convergence_comparison(maddpg_data, iddpg_data, convergence_path)

    print(f"\n✨ 完成！所有对比图表已保存到: {output_dir}")
    print(f"   - 奖励对比: {reward_path}")
    print(f"   - 价格对比: {price_path}")
    print(f"   - 收敛对比: {convergence_path}")


if __name__ == '__main__':
    main()
