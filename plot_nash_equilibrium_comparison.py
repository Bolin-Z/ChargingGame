"""
MADDPG vs IDDPG vs MFDDPG 纳什均衡收敛能力对比

聚焦于纳什均衡的两个核心指标：
1. 价格策略收敛性（策略是否稳定）
2. 找到均衡的效率（收敛速度）

使用方法：
    python plot_nash_equilibrium_comparison.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(json_path: str):
    """加载训练数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def compute_price_stability(records, window=10):
    """
    计算价格稳定性指标（价格变化率）
    纳什均衡的关键特征：价格策略不再改变
    """
    agents = list(records[0]['actual_prices'].keys())
    n_periods = len(records[0]['actual_prices'][agents[0]])

    # 计算每个step所有充电站所有时段的平均价格变化率
    stability_metric = []

    for i in range(1, len(records)):
        total_change = 0
        count = 0

        for agent in agents:
            for period in range(n_periods):
                prev_price = records[i-1]['actual_prices'][agent][period]
                curr_price = records[i]['actual_prices'][agent][period]

                # 相对变化率
                if prev_price > 1e-6:
                    relative_change = abs(curr_price - prev_price) / prev_price
                    total_change += relative_change
                    count += 1

        avg_change = total_change / count if count > 0 else 0
        stability_metric.append(avg_change)

    # 平滑处理
    if len(stability_metric) >= window:
        stability_metric = uniform_filter1d(stability_metric, size=window, mode='nearest')

    return stability_metric


def find_convergence_point(stability_metric, steps, threshold=0.01):
    """
    找到收敛点：价格变化率持续低于阈值的第一个位置
    """
    window_size = 5  # 连续5步都低于阈值才算收敛

    for i in range(len(stability_metric) - window_size):
        if all(stability_metric[i+j] < threshold for j in range(window_size)):
            return steps[i+1]  # +1因为stability_metric从step 1开始

    return steps[-1]  # 如果没找到，返回最后一步


def compute_reward_variance(records, window=10):
    """
    计算奖励方差（衡量收益稳定性）
    纳什均衡下，收益应该稳定
    """
    agents = list(records[0]['rewards'].keys())

    variance_metric = []

    for i in range(window, len(records)):
        # 计算窗口内每个agent的奖励标准差
        window_records = records[i-window:i]

        total_std = 0
        for agent in agents:
            rewards = [r['rewards'][agent] for r in window_records]
            total_std += np.std(rewards) / (np.mean(rewards) + 1e-6)  # 归一化标准差

        variance_metric.append(total_std / len(agents))

    return variance_metric


def plot_nash_comparison(maddpg_data: dict, iddpg_data: dict, mfddpg_data: dict, output_path: str):
    """绘制纳什均衡收敛能力对比"""
    episode_idx = 0

    # 提取数据
    maddpg_records = [r for r in maddpg_data['records'] if r['episode'] == episode_idx]
    iddpg_records = [r for r in iddpg_data['records'] if r['episode'] == episode_idx]
    mfddpg_records = [r for r in mfddpg_data['records'] if r['episode'] == episode_idx]

    maddpg_steps = np.array([r['step'] for r in maddpg_records])
    iddpg_steps = np.array([r['step'] for r in iddpg_records])
    mfddpg_steps = np.array([r['step'] for r in mfddpg_records])

    # 计算关键指标
    maddpg_stability = compute_price_stability(maddpg_records)
    iddpg_stability = compute_price_stability(iddpg_records)
    mfddpg_stability = compute_price_stability(mfddpg_records)

    maddpg_variance = compute_reward_variance(maddpg_records)
    iddpg_variance = compute_reward_variance(iddpg_records)
    mfddpg_variance = compute_reward_variance(mfddpg_records)

    # 找到收敛点
    convergence_threshold = 0.01
    maddpg_convergence_step = find_convergence_point(maddpg_stability, maddpg_steps, convergence_threshold)
    iddpg_convergence_step = find_convergence_point(iddpg_stability, iddpg_steps, convergence_threshold)
    mfddpg_convergence_step = find_convergence_point(mfddpg_stability, mfddpg_steps, convergence_threshold)

    # 创建2x2子图
    fig = plt.figure(figsize=(16, 11))

    # ========== 子图1: 价格策略稳定性（核心指标）==========
    ax1 = plt.subplot(2, 2, 1)

    ax1.plot(maddpg_steps[1:], maddpg_stability,
            color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    ax1.plot(iddpg_steps[1:], iddpg_stability,
            color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)
    ax1.plot(mfddpg_steps[1:], mfddpg_stability,
            color='#F18701', label='MFDDPG', linewidth=2.5, alpha=0.9)

    # 添加收敛阈值线
    ax1.axhline(y=convergence_threshold, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'收敛阈值 ({convergence_threshold})')

    # 标注收敛点
    ax1.axvline(x=maddpg_convergence_step, color='#2E86AB',
                linestyle=':', linewidth=2, alpha=0.5)
    ax1.axvline(x=iddpg_convergence_step, color='#A23B72',
                linestyle=':', linewidth=2, alpha=0.5)
    ax1.axvline(x=mfddpg_convergence_step, color='#F18701',
                linestyle=':', linewidth=2, alpha=0.5)

    ax1.text(maddpg_convergence_step, ax1.get_ylim()[1]*0.9,
            f'MADDPG\n收敛于\nStep {maddpg_convergence_step}',
            fontsize=9, ha='center', color='#2E86AB', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#D6E9F5', alpha=0.8))

    ax1.text(iddpg_convergence_step, ax1.get_ylim()[1]*0.7,
            f'IDDPG\n收敛于\nStep {iddpg_convergence_step}',
            fontsize=9, ha='center', color='#A23B72', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#F5D6E8', alpha=0.8))

    ax1.text(mfddpg_convergence_step, ax1.get_ylim()[1]*0.5,
            f'MFDDPG\n收敛于\nStep {mfddpg_convergence_step}',
            fontsize=9, ha='center', color='#F18701', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))

    ax1.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格平均变化率', fontsize=13, fontweight='bold')
    ax1.set_title('(a) 价格策略稳定性（纳什均衡核心指标）',
                  fontsize=14, fontweight='bold', loc='left', pad=15)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    # ========== 子图2: 收益稳定性 ==========
    ax2 = plt.subplot(2, 2, 2)

    if len(maddpg_variance) > 0:
        ax2.plot(maddpg_steps[10:10+len(maddpg_variance)], maddpg_variance,
                color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    if len(iddpg_variance) > 0:
        ax2.plot(iddpg_steps[10:10+len(iddpg_variance)], iddpg_variance,
                color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)
    if len(mfddpg_variance) > 0:
        ax2.plot(mfddpg_steps[10:10+len(mfddpg_variance)], mfddpg_variance,
                color='#F18701', label='MFDDPG', linewidth=2.5, alpha=0.9)

    ax2.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('收益波动性（归一化标准差）', fontsize=13, fontweight='bold')
    ax2.set_title('(b) 收益稳定性（均衡质量指标）',
                  fontsize=14, fontweight='bold', loc='left', pad=15)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')

    # ========== 子图3: 收敛路径对比（归一化稳定性）==========
    ax3 = plt.subplot(2, 2, 3)

    # 归一化稳定性指标（从高到低）
    maddpg_stability_norm = np.array(maddpg_stability) / (maddpg_stability[0] + 1e-8)
    iddpg_stability_norm = np.array(iddpg_stability) / (iddpg_stability[0] + 1e-8)
    mfddpg_stability_norm = np.array(mfddpg_stability) / (mfddpg_stability[0] + 1e-8)

    ax3.plot(maddpg_steps[1:], maddpg_stability_norm,
            color='#2E86AB', label='MADDPG', linewidth=2.5, alpha=0.9)
    ax3.plot(iddpg_steps[1:], iddpg_stability_norm,
            color='#A23B72', label='IDDPG', linewidth=2.5, alpha=0.9)
    ax3.plot(mfddpg_steps[1:], mfddpg_stability_norm,
            color='#F18701', label='MFDDPG', linewidth=2.5, alpha=0.9)

    # 关键里程碑
    for level, label in [(0.5, '50%'), (0.1, '10%'), (0.01, '1%')]:
        ax3.axhline(y=level, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax3.text(0, level, label, fontsize=9, va='center', ha='right', color='gray')

    ax3.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax3.set_ylabel('归一化稳定性进度', fontsize=13, fontweight='bold')
    ax3.set_title('(c) 收敛路径对比',
                  fontsize=14, fontweight='bold', loc='left', pad=15)
    ax3.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yscale('log')

    # ========== 子图4: 综合性能统计表 ==========
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # 计算统计指标
    stats_data = []
    stats_data.append(['纳什均衡指标', 'MADDPG', 'IDDPG', 'MFDDPG'])
    stats_data.append(['', '', '', ''])

    # 1. 收敛步数
    stats_data.append(['达到纳什均衡步数',
                      f'{maddpg_convergence_step}',
                      f'{iddpg_convergence_step}',
                      f'{mfddpg_convergence_step}'])

    # 2. 最终稳定性
    maddpg_final_stability = np.mean(maddpg_stability[-10:])
    iddpg_final_stability = np.mean(iddpg_stability[-10:])
    mfddpg_final_stability = np.mean(mfddpg_stability[-10:])
    stats_data.append(['最终价格变化率',
                      f'{maddpg_final_stability:.6f}',
                      f'{iddpg_final_stability:.6f}',
                      f'{mfddpg_final_stability:.6f}'])

    # 3. 收敛前后期对比
    maddpg_early_stability = np.mean(maddpg_stability[:len(maddpg_stability)//5])
    maddpg_late_stability = np.mean(maddpg_stability[-len(maddpg_stability)//5:])
    iddpg_early_stability = np.mean(iddpg_stability[:len(iddpg_stability)//5])
    iddpg_late_stability = np.mean(iddpg_stability[-len(iddpg_stability)//5:])
    mfddpg_early_stability = np.mean(mfddpg_stability[:len(mfddpg_stability)//5])
    mfddpg_late_stability = np.mean(mfddpg_stability[-len(mfddpg_stability)//5:])

    maddpg_improvement = maddpg_early_stability / (maddpg_late_stability + 1e-10)
    iddpg_improvement = iddpg_early_stability / (iddpg_late_stability + 1e-10)
    mfddpg_improvement = mfddpg_early_stability / (mfddpg_late_stability + 1e-10)

    stats_data.append(['收敛改善倍数',
                      f'{maddpg_improvement:.1f}x',
                      f'{iddpg_improvement:.1f}x',
                      f'{mfddpg_improvement:.1f}x'])

    # 4. 收益方差（如果有数据）
    if len(maddpg_variance) > 0 and len(iddpg_variance) > 0 and len(mfddpg_variance) > 0:
        maddpg_final_variance = np.mean(maddpg_variance[-10:])
        iddpg_final_variance = np.mean(iddpg_variance[-10:])
        mfddpg_final_variance = np.mean(mfddpg_variance[-10:])
        stats_data.append(['最终收益波动性',
                          f'{maddpg_final_variance:.6f}',
                          f'{iddpg_final_variance:.6f}',
                          f'{mfddpg_final_variance:.6f}'])

    # 5. 总训练步数
    maddpg_total = len(maddpg_steps)
    iddpg_total = len(iddpg_steps)
    mfddpg_total = len(mfddpg_steps)
    stats_data.append(['Episode总步数',
                      f'{maddpg_total}',
                      f'{iddpg_total}',
                      f'{mfddpg_total}'])

    # 6. 算法特性描述
    stats_data.append(['', '', '', ''])
    stats_data.append(['算法特性', '中心化训练', '独立训练', '均值场'])
    stats_data.append(['信息利用', '全局观测', '局部观测', '平均场'])
    stats_data.append(['计算复杂度', '较高', '较低', '中等'])

    # 创建表格
    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.32, 0.23, 0.23, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # 设置表头样式
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#404040')
        cell.set_text_props(weight='bold', color='white')

    # 分隔行
    for i in range(4):
        for row_idx in [1, 6]:
            if row_idx < len(stats_data):
                cell = table[(row_idx, i)]
                cell.set_facecolor('#E0E0E0')
                cell.set_height(0.02)

    # 设置数据行样式
    for i in range(2, len(stats_data)):
        if i in [1, 6]:  # 跳过分隔行
            continue
        for j in range(4):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor('#F5F5F5')
                cell.set_text_props(weight='bold', fontsize=9)
            elif j == 1:
                cell.set_facecolor('#D6E9F5')
            elif j == 2:
                cell.set_facecolor('#F5D6E8')
            elif j == 3:
                cell.set_facecolor('#FFF3E0')

    ax4.set_title('(d) 纳什均衡性能统计', fontsize=14, fontweight='bold', pad=20)

    # 总标题
    fig.suptitle(f'Episode {episode_idx} - MADDPG vs IDDPG vs MFDDPG 纳什均衡求解能力对比\n'
                 f'核心关注：价格策略收敛性与均衡质量',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 纳什均衡对比图已保存到: {output_path}")
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
    print("🎨 绘制纳什均衡求解能力对比图...")
    output_path = os.path.join(output_dir, 'nash_equilibrium_comparison.png')
    plot_nash_comparison(maddpg_data, iddpg_data, mfddpg_data, output_path)

    print(f"\n✨ 完成！")
    print("\n📌 关键发现：")
    print("   - 价格变化率趋近于0 = 纳什均衡")
    print("   - 收敛步数 = 找到均衡的效率")
    print("   - 收益波动性 = 均衡质量")


if __name__ == '__main__':
    main()
