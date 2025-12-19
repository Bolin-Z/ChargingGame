"""
分析连续收敛轮数的最佳选择

基于现有的100轮实验数据，模拟不同连续轮数的收敛判断效果：
1. 首次"连续N轮GM<阈值"的迭代轮数
2. 收敛后的稳定性（是否反弹）
3. 不同阈值下的表现
"""

import json
import numpy as np
from pathlib import Path


def load_experiment_data(report_path: str) -> dict:
    """加载实验数据"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_consecutive_convergence(iterations: list, threshold: float, required_rounds: int) -> dict:
    """
    分析连续收敛的表现

    Args:
        iterations: 迭代数据列表
        threshold: GM阈值（如0.02表示2%）
        required_rounds: 需要连续达标的轮数

    Returns:
        分析结果字典
    """
    gm_values = [it['global_mean'] for it in iterations]
    n = len(gm_values)

    # 找首次连续N轮达标的位置
    first_converge_iter = None
    consecutive_count = 0

    for i, gm in enumerate(gm_values):
        if gm < threshold:
            consecutive_count += 1
            if consecutive_count >= required_rounds and first_converge_iter is None:
                first_converge_iter = i + 1  # 1-indexed
        else:
            consecutive_count = 0

    # 分析收敛后的稳定性
    stability_analysis = None
    if first_converge_iter is not None:
        # 收敛点之后的数据
        post_converge = gm_values[first_converge_iter-1:]

        # 统计反弹次数（超过阈值的次数）
        rebounds = sum(1 for gm in post_converge if gm >= threshold)
        rebound_rate = rebounds / len(post_converge) if post_converge else 0

        # 收敛后的统计
        stability_analysis = {
            'post_converge_mean': np.mean(post_converge) * 100,
            'post_converge_std': np.std(post_converge) * 100,
            'post_converge_max': np.max(post_converge) * 100,
            'rebounds': rebounds,
            'rebound_rate': rebound_rate * 100,
            'remaining_iters': len(post_converge)
        }

    return {
        'first_converge_iter': first_converge_iter,
        'converged': first_converge_iter is not None,
        'stability': stability_analysis
    }


def main():
    # 加载数据
    report_path = Path("results/metrics_comparison/comparison_report.json")
    if not report_path.exists():
        print(f"错误: 找不到 {report_path}")
        return

    data = load_experiment_data(report_path)

    # 测试参数
    thresholds = [0.02, 0.03, 0.05]  # 2%, 3%, 5%
    consecutive_rounds = [2, 3, 5, 10]

    networks = ['siouxfalls', 'berlin_friedrichshain', 'anaheim']

    print("=" * 80)
    print("连续收敛轮数分析")
    print("=" * 80)

    for network in networks:
        if network not in data['networks']:
            continue

        iterations = data['networks'][network]['iterations']

        print(f"\n{'='*60}")
        print(f"网络: {network.upper()}")
        print(f"{'='*60}")

        # 先展示GM的基本统计
        gm_values = [it['global_mean'] * 100 for it in iterations]
        print(f"\nGM统计: 均值={np.mean(gm_values):.2f}% std={np.std(gm_values):.2f}% "
              f"min={np.min(gm_values):.2f}% max={np.max(gm_values):.2f}%")

        for threshold in thresholds:
            print(f"\n阈值: {threshold*100:.0f}%")
            print("-" * 50)
            print(f"{'连续轮数':<10} {'首次收敛轮':<12} {'反弹次数':<10} {'反弹率':<10} {'收敛后均值':<12}")
            print("-" * 50)

            for rounds in consecutive_rounds:
                result = analyze_consecutive_convergence(iterations, threshold, rounds)

                if result['converged']:
                    stab = result['stability']
                    print(f"{rounds:<10} {result['first_converge_iter']:<12} "
                          f"{stab['rebounds']:<10} {stab['rebound_rate']:.1f}%{'':<5} "
                          f"{stab['post_converge_mean']:.2f}%")
                else:
                    print(f"{rounds:<10} {'未收敛':<12} {'-':<10} {'-':<10} {'-':<12}")

    # 综合建议
    print("\n" + "=" * 80)
    print("综合分析与建议")
    print("=" * 80)

    print("""
基于以上数据的建议：

1. 连续轮数选择：
   - 2轮：收敛快但反弹率高，不够稳健
   - 3轮：平衡选择，适合大多数场景
   - 5轮：更稳健，但可能难以达成（尤其是BF/Anaheim）
   - 10轮：过于严格，不推荐

2. 阈值选择建议：
   - SF网络：2% 阈值可行
   - BF/Anaheim网络：考虑放宽到 3%~5%

3. 推荐配置：
   - 严格模式：连续3轮 + 2%阈值
   - 实用模式：连续3轮 + 3%阈值（对BF/Anaheim更友好）
""")


if __name__ == "__main__":
    main()
