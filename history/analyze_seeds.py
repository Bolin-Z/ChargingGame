"""
分析三个seed的实验结果，判断哪个算法找到了真正的均衡解
"""

import json
import numpy as np
from pathlib import Path


def load_step_records(filepath):
    """加载 step_records.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['records'], data.get('metadata', {})


def analyze_experiment(records, name):
    """分析单个实验的结果"""
    last_n = min(50, len(records))
    last_records = records[-last_n:]

    last_step = records[-1]
    prices = last_step.get('actual_prices', last_step.get('prices', {}))
    rewards = last_step.get('rewards', {})

    # 边界价格比例
    price_min, price_max = 0.5, 2.0
    boundary_threshold = 0.1  # 距离边界0.1以内算边界
    total_prices = 0
    boundary_prices = 0

    for agent, p_list in prices.items():
        for p in p_list:
            total_prices += 1
            if abs(p - price_min) < boundary_threshold or abs(p - price_max) < boundary_threshold:
                boundary_prices += 1

    boundary_ratio = boundary_prices / total_prices * 100 if total_prices > 0 else 0

    # 最后50步change_rate
    change_rates = [r['relative_change_rate'] for r in last_records]

    # 价格稳定性（最后50步每个价格维度的标准差）
    price_stds = []
    agents = list(prices.keys())
    for agent in agents:
        key = 'actual_prices' if 'actual_prices' in last_records[0] else 'prices'
        agent_prices_over_time = [r[key][agent] for r in last_records]
        for period in range(len(agent_prices_over_time[0])):
            period_prices = [p[period] for p in agent_prices_over_time]
            price_stds.append(np.std(period_prices))

    avg_price_std = np.mean(price_stds)

    # 低于阈值的统计
    threshold = 0.01
    below_threshold = sum(1 for r in change_rates if r < threshold)

    # 连续低于阈值最长
    max_consecutive = 0
    current_consecutive = 0
    all_change_rates = [r['relative_change_rate'] for r in records]
    for r in all_change_rates:
        if r < threshold:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return {
        'name': name,
        'total_steps': len(records),
        'boundary_ratio': boundary_ratio,
        'avg_change_rate': np.mean(change_rates),
        'min_change_rate': min(change_rates),
        'price_stability': avg_price_std,
        'below_threshold_ratio': below_threshold / len(change_rates) * 100,
        'max_consecutive': max_consecutive,
        'final_prices': prices,
        'final_rewards': rewards,
    }


def print_result(result):
    """打印单个结果"""
    print(f"\n{'='*50}")
    print(f"  {result['name']}")
    print(f"{'='*50}")
    print(f"  总步数: {result['total_steps']}")
    print(f"  边界价格比例: {result['boundary_ratio']:.1f}%")
    print(f"  价格稳定性(avg std): {result['price_stability']:.4f}")
    print(f"  最后50步 avg change_rate: {result['avg_change_rate']:.6f}")
    print(f"  最后50步 min change_rate: {result['min_change_rate']:.6f}")
    print(f"  低于阈值(0.01)比例: {result['below_threshold_ratio']:.1f}%")
    print(f"  连续低于阈值最长: {result['max_consecutive']} steps")

    print(f"\n  最终价格:")
    for agent, p_list in sorted(result['final_prices'].items()):
        print(f"    {agent}: {[round(p, 2) for p in p_list]}")

    print(f"\n  最终收益:")
    for agent, r in sorted(result['final_rewards'].items()):
        print(f"    {agent}: {r:.1f}")


def main():
    results_dir = Path(__file__).parent / 'results' / 'siouxfalls'

    # 实验路径配置
    paths = {
        42: {
            'MADDPG': 'MADDPG/seed42/01_06_20_08',
            'IDDPG': 'IDDPG/seed42/01_06_20_07',
            'MFDDPG': 'MFDDPG/seed42/01_06_19_50',
        },
        123: {
            'MADDPG': 'MADDPG/seed123/01_07_02_12',
            'IDDPG': 'IDDPG/seed123/01_07_01_20',
            'MFDDPG': 'MFDDPG/seed123/01_07_02_12',
        },
        456: {
            'MADDPG': 'MADDPG/seed456/01_07_09_54',
            'IDDPG': 'IDDPG/seed456/01_07_09_43',
            'MFDDPG': 'MFDDPG/seed456/01_07_09_22',
        }
    }

    all_results = {}

    for seed in [42, 123, 456]:
        print(f"\n{'#'*60}")
        print(f"#  SEED {seed}")
        print(f"{'#'*60}")

        all_results[seed] = {}

        for algo in ['MADDPG', 'IDDPG', 'MFDDPG']:
            filepath = results_dir / paths[seed][algo] / 'step_records.json'

            if not filepath.exists():
                print(f"\n  {algo}: 文件不存在 - {filepath}")
                continue

            try:
                records, metadata = load_step_records(filepath)
                result = analyze_experiment(records, algo)
                all_results[seed][algo] = result
                print_result(result)
            except Exception as e:
                print(f"\n  {algo}: 加载错误 - {e}")

    # 汇总对比
    print(f"\n\n{'#'*60}")
    print(f"#  汇总对比")
    print(f"{'#'*60}")

    print(f"\n{'算法':<10} | {'Seed':<6} | {'边界%':>8} | {'稳定性':>8} | {'连续收敛':>8} | {'判断':<15}")
    print(f"{'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*15}")

    for algo in ['MADDPG', 'IDDPG', 'MFDDPG']:
        for seed in [42, 123, 456]:
            if seed in all_results and algo in all_results[seed]:
                r = all_results[seed][algo]

                # 判断逻辑
                if r['boundary_ratio'] > 90 and r['price_stability'] < 0.05:
                    judgment = "✅ 边界均衡"
                elif r['boundary_ratio'] < 10 and r['price_stability'] < 0.05:
                    judgment = "⚠️ 中间稳定?"
                elif r['price_stability'] > 0.1:
                    judgment = "❌ 仍在震荡"
                else:
                    judgment = "? 需验证"

                print(f"{algo:<10} | {seed:<6} | {r['boundary_ratio']:>7.1f}% | {r['price_stability']:>8.4f} | {r['max_consecutive']:>8} | {judgment:<15}")

    # 均衡解验证建议
    print(f"\n\n{'#'*60}")
    print(f"#  均衡解判断")
    print(f"{'#'*60}")

    print("""
根据博弈论分析（见 task_tune.md 第六节）：
- 价格成本 (25-100元) >> 时间成本 (~5元)
- 理论上应该收敛到边界解 (Bang-Bang)

判断标准：
1. 边界比例 > 90% + 价格稳定 → 可能是均衡
2. 边界比例 < 10% + 价格稳定 → 需要单边偏离测试验证
3. 价格不稳定 (std > 0.1) → 还没收敛

建议：对中间值解进行单边偏离测试，验证是否真的是均衡。
""")


if __name__ == '__main__':
    main()
