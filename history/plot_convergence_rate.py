"""
MADDPG vs IDDPG vs MFDDPG 收敛速度对比（基于奖励变化率）

对比三个算法的收敛速度：
1. 总奖励变化曲线
2. 奖励变化率曲线（一阶导数）
3. 累积奖励增长对比

使用方法：
    python plot_convergence_rate.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(json_path: str):
    """加载训练数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def compute_change_rate(values, window_size=5):
    """计算变化率（平滑后的一阶导数）"""
    # 先平滑
    smoothed = uniform_filter1d(values, size=window_size, mode='nearest')
    # 计算差分
    change_rate = np.diff(smoothed, prepend=smoothed[0])
    return change_rate


def plot_convergence_comparison(maddpg_data: dict, iddpg_data: dict, mfddpg_data: dict, output_path: str):
    """绘制收敛速度对比图（包含变化率分析）"""
    episode_idx = 0

    # 提取数据
    maddpg_records = [r for r in maddpg_data['records'] if r['episode'] == episode_idx]
    iddpg_records = [r for r in iddpg_data['records'] if r['episode'] == episode_idx]
    mfddpg_records = [r for r in mfddpg_data['records'] if r['episode'] == episode_idx]

    if not maddpg_records or not iddpg_records or not mfddpg_records:
        print(f"⚠️ 未找到Episode {episode_idx}的数据")
        return

    # 计算总奖励
    maddpg_total_rewards = np.array([sum(r['rewards'].values()) for r in maddpg_records])
    maddpg_steps = np.array([r['step'] for r in maddpg_records])

    iddpg_total_rewards = np.array([sum(r['rewards'].values()) for r in iddpg_records])
    iddpg_steps = np.array([r['step'] for r in iddpg_records])

    mfddpg_total_rewards = np.array([sum(r['rewards'].values()) for r in mfddpg_records])
    mfddpg_steps = np.array([r['step'] for r in mfddpg_records])

    # 归一化到[0,1]，便于对比
    maddpg_normalized = (maddpg_total_rewards - maddpg_total_rewards[0]) / (maddpg_total_rewards[-1] - maddpg_total_rewards[0] + 1e-8)
    iddpg_normalized = (iddpg_total_rewards - iddpg_total_rewards[0]) / (iddpg_total_rewards[-1] - iddpg_total_rewards[0] + 1e-8)
    mfddpg_normalized = (mfddpg_total_rewards - mfddpg_total_rewards[0]) / (mfddpg_total_rewards[-1] - mfddpg_total_rewards[0] + 1e-8)

    # 计算变化率（平滑窗口）
    window = 10
    maddpg_rate = compute_change_rate(maddpg_normalized, window_size=window)
    iddpg_rate = compute_change_rate(iddpg_normalized, window_size=window)
    mfddpg_rate = compute_change_rate(mfddpg_normalized, window_size=window)

    # 创建3个子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ========== 子图1: 总奖励对比 ==========
    ax1 = axes[0]
    ax1.plot(maddpg_steps, maddpg_total_rewards,
            color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    ax1.plot(iddpg_steps, iddpg_total_rewards,
            color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)
    ax1.plot(mfddpg_steps, mfddpg_total_rewards,
            color='#F18701', label='MFDDPG', linewidth=2.5, alpha=0.9)

    # 标注收敛点
    ax1.scatter([maddpg_steps[-1]], [maddpg_total_rewards[-1]],
               color='#2E86AB', s=120, zorder=5, edgecolors='white', linewidths=2)
    ax1.scatter([iddpg_steps[-1]], [iddpg_total_rewards[-1]],
               color='#A23B72', s=120, zorder=5, edgecolors='white', linewidths=2)
    ax1.scatter([mfddpg_steps[-1]], [mfddpg_total_rewards[-1]],
               color='#F18701', s=120, zorder=5, edgecolors='white', linewidths=2)

    ax1.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('总奖励 (元)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 总奖励变化曲线', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 添加收敛步数标注
    textstr = f'MADDPG: {len(maddpg_steps)} steps\nIDDPG: {len(iddpg_steps)} steps\nMFDDPG: {len(mfddpg_steps)} steps\n'
    textstr += f'MADDPG vs IDDPG: {(1 - len(maddpg_steps)/len(iddpg_steps))*100:.1f}%\n'
    textstr += f'MFDDPG vs IDDPG: {(1 - len(mfddpg_steps)/len(iddpg_steps))*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # ========== 子图2: 归一化奖励对比（更清晰的收敛过程）==========
    ax2 = axes[1]
    ax2.plot(maddpg_steps, maddpg_normalized,
            color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    ax2.plot(iddpg_steps, iddpg_normalized,
            color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)
    ax2.plot(mfddpg_steps, mfddpg_normalized,
            color='#F18701', label='MFDDPG', linewidth=2.5, alpha=0.9)

    # 添加50%和90%收敛线
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='50%收敛')
    ax2.axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='90%收敛')

    # 计算达到50%和90%的step数
    maddpg_50 = maddpg_steps[np.argmax(maddpg_normalized >= 0.5)] if np.any(maddpg_normalized >= 0.5) else None
    maddpg_90 = maddpg_steps[np.argmax(maddpg_normalized >= 0.9)] if np.any(maddpg_normalized >= 0.9) else None
    iddpg_50 = iddpg_steps[np.argmax(iddpg_normalized >= 0.5)] if np.any(iddpg_normalized >= 0.5) else None
    iddpg_90 = iddpg_steps[np.argmax(iddpg_normalized >= 0.9)] if np.any(iddpg_normalized >= 0.9) else None
    mfddpg_50 = mfddpg_steps[np.argmax(mfddpg_normalized >= 0.5)] if np.any(mfddpg_normalized >= 0.5) else None
    mfddpg_90 = mfddpg_steps[np.argmax(mfddpg_normalized >= 0.9)] if np.any(mfddpg_normalized >= 0.9) else None

    ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('归一化进度', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 归一化收敛进度对比', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([-0.05, 1.05])

    # 添加里程碑标注
    milestone_text = f'达到50%收敛:\n  MADDPG: {maddpg_50} steps\n  IDDPG: {iddpg_50} steps\n  MFDDPG: {mfddpg_50} steps\n\n达到90%收敛:\n  MADDPG: {maddpg_90} steps\n  IDDPG: {iddpg_90} steps\n  MFDDPG: {mfddpg_90} steps'
    ax2.text(0.98, 0.02, milestone_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # ========== 子图3: 奖励变化率绝对值对比（收敛趋势）==========
    ax3 = axes[2]

    # 计算绝对值（反映变化幅度）
    maddpg_abs_rate = np.abs(maddpg_rate)
    iddpg_abs_rate = np.abs(iddpg_rate)
    mfddpg_abs_rate = np.abs(mfddpg_rate)

    ax3.plot(maddpg_steps, maddpg_abs_rate,
            color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    ax3.plot(iddpg_steps, iddpg_abs_rate,
            color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)
    ax3.plot(mfddpg_steps, mfddpg_abs_rate,
            color='#F18701', label='MFDDPG', linewidth=2.5, alpha=0.9)

    # 添加趋势线（多项式拟合）
    if len(maddpg_steps) > 10:
        z_maddpg = np.polyfit(maddpg_steps, maddpg_abs_rate, 3)
        p_maddpg = np.poly1d(z_maddpg)
        ax3.plot(maddpg_steps, p_maddpg(maddpg_steps),
                color='#2E86AB', linestyle='--', linewidth=2, alpha=0.6, label='MADDPG趋势')

    if len(iddpg_steps) > 10:
        z_iddpg = np.polyfit(iddpg_steps, iddpg_abs_rate, 3)
        p_iddpg = np.poly1d(z_iddpg)
        ax3.plot(iddpg_steps, p_iddpg(iddpg_steps),
                color='#A23B72', linestyle='--', linewidth=2, alpha=0.6, label='IDDPG趋势')

    if len(mfddpg_steps) > 10:
        z_mfddpg = np.polyfit(mfddpg_steps, mfddpg_abs_rate, 3)
        p_mfddpg = np.poly1d(z_mfddpg)
        ax3.plot(mfddpg_steps, p_mfddpg(mfddpg_steps),
                color='#F18701', linestyle='--', linewidth=2, alpha=0.6, label='MFDDPG趋势')

    ax3.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('|奖励变化率|', fontsize=12, fontweight='bold')
    ax3.set_title(f'(c) 奖励变化率绝对值对比（收敛趋势分析，窗口={window}）', fontsize=13, fontweight='bold', loc='left')
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yscale('log')  # 使用对数坐标，更清晰展示收敛过程

    # 计算统计指标
    maddpg_avg_rate = np.mean(maddpg_abs_rate)
    iddpg_avg_rate = np.mean(iddpg_abs_rate)
    mfddpg_avg_rate = np.mean(mfddpg_abs_rate)

    # 计算前20%和后20%的平均变化率（反映初期vs后期）
    maddpg_early = np.mean(maddpg_abs_rate[:len(maddpg_abs_rate)//5])
    maddpg_late = np.mean(maddpg_abs_rate[-len(maddpg_abs_rate)//5:])
    iddpg_early = np.mean(iddpg_abs_rate[:len(iddpg_abs_rate)//5])
    iddpg_late = np.mean(iddpg_abs_rate[-len(iddpg_abs_rate)//5:])
    mfddpg_early = np.mean(mfddpg_abs_rate[:len(mfddpg_abs_rate)//5])
    mfddpg_late = np.mean(mfddpg_abs_rate[-len(mfddpg_abs_rate)//5:])

    rate_text = f'平均|变化率|:\n  MADDPG: {maddpg_avg_rate:.4f}\n  IDDPG: {iddpg_avg_rate:.4f}\n  MFDDPG: {mfddpg_avg_rate:.4f}\n\n'
    rate_text += f'初期|变化率|(前20%):\n  MADDPG: {maddpg_early:.4f}\n  IDDPG: {iddpg_early:.4f}\n  MFDDPG: {mfddpg_early:.4f}\n\n'
    rate_text += f'后期|变化率|(后20%):\n  MADDPG: {maddpg_late:.4f}\n  IDDPG: {iddpg_late:.4f}\n  MFDDPG: {mfddpg_late:.4f}\n\n'
    rate_text += f'收敛速度(早/晚比):\n  MADDPG: {maddpg_early/maddpg_late:.1f}x\n  IDDPG: {iddpg_early/iddpg_late:.1f}x\n  MFDDPG: {mfddpg_early/mfddpg_late:.1f}x'

    ax3.text(0.98, 0.98, rate_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # 总标题
    fig.suptitle(f'Episode {episode_idx} - MADDPG vs IDDPG vs MFDDPG 收敛速度深度对比',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 收敛速度对比图已保存到: {output_path}")
    plt.close()


def main():
    # 数据路径
    maddpg_path = 'results/experiment_20250917_025705/step_records.json'
    iddpg_path = 'results/iddpg_experiment_20251117_202408/step_records.json'
    mfddpg_path = 'results/mfddpg_experiment_20251119_164139/step_records.json'

    # 输出目录
    output_dir = 'results/comparison'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("📊 加载训练数据...")
    maddpg_data = load_data(maddpg_path)
    iddpg_data = load_data(iddpg_path)
    mfddpg_data = load_data(mfddpg_path)
    print(f"   MADDPG: Episodes={maddpg_data['metadata']['total_episodes']}, "
          f"Steps={maddpg_data['metadata']['total_steps']}")
    print(f"   IDDPG: Episodes={iddpg_data['metadata']['total_episodes']}, "
          f"Steps={iddpg_data['metadata']['total_steps']}")
    print(f"   MFDDPG: Episodes={mfddpg_data['metadata']['total_episodes']}, "
          f"Steps={mfddpg_data['metadata']['total_steps']}")
    print()

    # 绘制对比图
    print("🎨 绘制收敛速度对比图（含变化率分析）...")
    output_path = os.path.join(output_dir, 'convergence_rate_comparison.png')
    plot_convergence_comparison(maddpg_data, iddpg_data, mfddpg_data, output_path)

    print(f"\n✨ 完成！图表已保存到: {output_path}")


if __name__ == '__main__':
    main()
