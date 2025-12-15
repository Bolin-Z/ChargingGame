"""
MADDPG vs IDDPG 收敛速度简洁对比

使用方法：
    python plot_convergence_simple.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(json_path: str):
    """加载训练数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def smooth_curve(values, window_length=51):
    """使用Savitzky-Golay滤波器平滑曲线"""
    if len(values) < window_length:
        window_length = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window_length < 3:
        return values
    return savgol_filter(values, window_length, 3)


def plot_comparison(maddpg_data: dict, iddpg_data: dict, output_path: str):
    """绘制简洁的收敛速度对比图"""
    episode_idx = 0

    # 提取数据
    maddpg_records = [r for r in maddpg_data['records'] if r['episode'] == episode_idx]
    iddpg_records = [r for r in iddpg_data['records'] if r['episode'] == episode_idx]

    # 计算总奖励
    maddpg_total_rewards = np.array([sum(r['rewards'].values()) for r in maddpg_records])
    maddpg_steps = np.array([r['step'] for r in maddpg_records])

    iddpg_total_rewards = np.array([sum(r['rewards'].values()) for r in iddpg_records])
    iddpg_steps = np.array([r['step'] for r in iddpg_records])

    # 归一化
    maddpg_normalized = (maddpg_total_rewards - maddpg_total_rewards[0]) / (maddpg_total_rewards[-1] - maddpg_total_rewards[0] + 1e-8)
    iddpg_normalized = (iddpg_total_rewards - iddpg_total_rewards[0]) / (iddpg_total_rewards[-1] - iddpg_total_rewards[0] + 1e-8)

    # 平滑处理
    maddpg_smooth = smooth_curve(maddpg_normalized)
    iddpg_smooth = smooth_curve(iddpg_normalized)

    # 计算变化率（梯度）
    maddpg_gradient = np.gradient(maddpg_smooth)
    iddpg_gradient = np.gradient(iddpg_smooth)

    # 创建2x2子图
    fig = plt.figure(figsize=(16, 10))

    # ========== 子图1: 归一化收敛进度 ==========
    ax1 = plt.subplot(2, 2, 1)

    ax1.plot(maddpg_steps, maddpg_normalized,
            color='#2E86AB', alpha=0.3, linewidth=1)
    ax1.plot(maddpg_steps, maddpg_smooth,
            color='#2E86AB', label='MADDPG', linewidth=3, alpha=0.9)

    ax1.plot(iddpg_steps, iddpg_normalized,
            color='#A23B72', alpha=0.3, linewidth=1)
    ax1.plot(iddpg_steps, iddpg_smooth,
            color='#A23B72', label='IDDPG', linewidth=3, alpha=0.9)

    # 关键里程碑线
    for level, style in [(0.5, ':'), (0.9, '--'), (0.95, '-.')]:
        ax1.axhline(y=level, color='gray', linestyle=style, linewidth=1, alpha=0.5)
        ax1.text(0, level, f'{int(level*100)}%', fontsize=9, va='bottom', ha='right', color='gray')

    ax1.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('收敛进度', fontsize=13, fontweight='bold')
    ax1.set_title('(a) 收敛进度对比', fontsize=14, fontweight='bold', loc='left', pad=15)
    ax1.legend(loc='lower right', fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim([-0.05, 1.05])

    # ========== 子图2: 变化率绝对值（对数） ==========
    ax2 = plt.subplot(2, 2, 2)

    maddpg_abs_gradient = np.abs(maddpg_gradient)
    iddpg_abs_gradient = np.abs(iddpg_gradient)

    ax2.plot(maddpg_steps, maddpg_abs_gradient,
            color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    ax2.plot(iddpg_steps, iddpg_abs_gradient,
            color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)

    ax2.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('|变化率| (log scale)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) 变化率衰减趋势', fontsize=14, fontweight='bold', loc='left', pad=15)
    ax2.legend(loc='best', fontsize=12, framealpha=0.95)
    ax2.grid(True, alpha=0.2, which='both')
    ax2.set_yscale('log')

    # ========== 子图3: 达到关键里程碑的步数对比 ==========
    ax3 = plt.subplot(2, 2, 3)

    milestones = [0.5, 0.7, 0.9, 0.95, 0.99]
    maddpg_milestone_steps = []
    iddpg_milestone_steps = []

    for milestone in milestones:
        maddpg_idx = np.argmax(maddpg_smooth >= milestone) if np.any(maddpg_smooth >= milestone) else len(maddpg_steps) - 1
        iddpg_idx = np.argmax(iddpg_smooth >= milestone) if np.any(iddpg_smooth >= milestone) else len(iddpg_steps) - 1
        maddpg_milestone_steps.append(maddpg_steps[maddpg_idx])
        iddpg_milestone_steps.append(iddpg_steps[iddpg_idx])

    x = np.arange(len(milestones))
    width = 0.35

    bars1 = ax3.bar(x - width/2, maddpg_milestone_steps, width,
                    label='MADDPG', color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x + width/2, iddpg_milestone_steps, width,
                    label='IDDPG', color='#A23B72', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('收敛进度', fontsize=13, fontweight='bold')
    ax3.set_ylabel('所需步数', fontsize=13, fontweight='bold')
    ax3.set_title('(c) 达到关键里程碑的步数', fontsize=14, fontweight='bold', loc='left', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{int(m*100)}%' for m in milestones])
    ax3.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax3.grid(True, alpha=0.2, axis='y')

    # ========== 子图4: 综合统计对比表 ==========
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # 计算统计指标
    stats_data = []
    stats_data.append(['指标', 'MADDPG', 'IDDPG', '差异'])
    stats_data.append(['', '', '', ''])

    # 总步数
    maddpg_total = len(maddpg_steps)
    iddpg_total = len(iddpg_steps)
    speedup = (1 - maddpg_total / iddpg_total) * 100
    stats_data.append(['收敛总步数', f'{maddpg_total}', f'{iddpg_total}', f'{speedup:.1f}% 更快'])

    # 50%里程碑
    stats_data.append(['达到50%步数', f'{maddpg_milestone_steps[0]}', f'{iddpg_milestone_steps[0]}',
                      f'{(1-maddpg_milestone_steps[0]/iddpg_milestone_steps[0])*100:.1f}% 更快'])

    # 90%里程碑
    stats_data.append(['达到90%步数', f'{maddpg_milestone_steps[2]}', f'{iddpg_milestone_steps[2]}',
                      f'{(1-maddpg_milestone_steps[2]/iddpg_milestone_steps[2])*100:.1f}% 更快'])

    # 平均变化率
    maddpg_avg = np.mean(maddpg_abs_gradient)
    iddpg_avg = np.mean(iddpg_abs_gradient)
    stats_data.append(['平均变化率', f'{maddpg_avg:.5f}', f'{iddpg_avg:.5f}',
                      f'{(maddpg_avg/iddpg_avg):.2f}x'])

    # 初期vs后期
    maddpg_early = np.mean(maddpg_abs_gradient[:len(maddpg_abs_gradient)//5])
    maddpg_late = np.mean(maddpg_abs_gradient[-len(maddpg_abs_gradient)//5:])
    iddpg_early = np.mean(iddpg_abs_gradient[:len(iddpg_abs_gradient)//5])
    iddpg_late = np.mean(iddpg_abs_gradient[-len(iddpg_abs_gradient)//5:])

    stats_data.append(['早期变化率', f'{maddpg_early:.5f}', f'{iddpg_early:.5f}',
                      f'{(maddpg_early/iddpg_early):.2f}x'])
    stats_data.append(['后期变化率', f'{maddpg_late:.5f}', f'{iddpg_late:.5f}',
                      f'{(maddpg_late/iddpg_late):.2f}x'])

    # 收敛速度比
    maddpg_ratio = maddpg_early / (maddpg_late + 1e-10)
    iddpg_ratio = iddpg_early / (iddpg_late + 1e-10)
    stats_data.append(['衰减比(早/晚)', f'{maddpg_ratio:.1f}x', f'{iddpg_ratio:.1f}x',
                      f'{"更快衰减" if maddpg_ratio > iddpg_ratio else "较慢衰减"}'])

    # 创建表格
    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.22, 0.22, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)

    # 设置表头样式
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#404040')
        cell.set_text_props(weight='bold', color='white')

    # 分隔行
    for i in range(4):
        cell = table[(1, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_height(0.02)

    # 设置数据行样式
    for i in range(2, len(stats_data)):
        for j in range(4):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor('#F5F5F5')
                cell.set_text_props(weight='bold')
            elif j == 1:
                cell.set_facecolor('#D6E9F5')
            elif j == 2:
                cell.set_facecolor('#F5D6E8')
            else:
                cell.set_facecolor('#FFF9E6')
                cell.set_text_props(weight='bold', color='#CC0000')

    ax4.set_title('(d) 性能统计对比', fontsize=14, fontweight='bold', pad=20)

    # 总标题
    fig.suptitle(f'Episode {episode_idx} - MADDPG vs IDDPG 收敛速度综合对比',
                 fontsize=17, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    maddpg_data = load_data(maddpg_path)
    iddpg_data = load_data(iddpg_path)
    print(f"   MADDPG: Episodes={maddpg_data['metadata']['total_episodes']}, "
          f"Steps={maddpg_data['metadata']['total_steps']}")
    print(f"   IDDPG: Episodes={iddpg_data['metadata']['total_episodes']}, "
          f"Steps={iddpg_data['metadata']['total_steps']}")
    print()

    # 绘制对比图
    print("🎨 绘制收敛速度对比图...")
    output_path = os.path.join(output_dir, 'convergence_simple.png')
    plot_comparison(maddpg_data, iddpg_data, output_path)

    print(f"\n✨ 完成！")


if __name__ == '__main__':
    main()
