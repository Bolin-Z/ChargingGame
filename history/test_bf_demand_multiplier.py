"""
测试 Berlin Friedrichshain 网络在不同 demand_multiplier 下的表现

测试值: 2.1, 2.2, 2.3, 2.4
目标: 找到 completed_ratio 和收敛性能的平衡点
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def test_demand_multiplier(multiplier: float, gamma: float = 10.0, alpha: float = 0.05) -> dict:
    """测试指定 demand_multiplier 的表现"""

    # 临时修改配置文件
    settings_path = "data/berlin_friedrichshain/berlin_friedrichshain_settings.json"

    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    original_multiplier = settings.get('demand_multiplier', 2.0)
    settings['demand_multiplier'] = multiplier
    settings['ue_switch_gamma'] = gamma
    settings['ue_switch_alpha'] = alpha

    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

    try:
        # 创建环境
        env = EVCSChargingGameEnv(
            network_dir="data/berlin_friedrichshain",
            network_name="berlin_friedrichshain",
            random_seed=42,
            max_steps=10,
            convergence_threshold=0.01,
            stable_steps_required=3
        )
        env.reset()

        # 设置固定价格
        actions = {}
        for agent in env.agents:
            actions[agent] = np.full(env.n_periods, 0.5, dtype=np.float32)

        # 更新价格
        env._EVCSChargingGameEnv__update_prices_from_actions(actions)

        # 运行 UE-DTA
        charging_flows, ue_info = env._EVCSChargingGameEnv__run_simulation()

        result = {
            'demand_multiplier': multiplier,
            'gamma': gamma,
            'alpha': alpha,
            'converged': ue_info['ue_converged'],
            'iterations': ue_info['ue_iterations'],
            'final_gm': ue_info['ue_stats']['all_relative_gap_global_mean'] * 100,
            'final_p95': ue_info['ue_stats']['all_relative_gap_p95'] * 100,
            'final_od_max': ue_info['ue_stats']['all_relative_gap_od_max_mean'] * 100,
            'completed_ratio': ue_info['ue_stats']['completed_total_vehicles'] / ue_info['ue_stats']['total_vehicles'],
            'total_vehicles': ue_info['ue_stats']['total_vehicles'],
            'completed_vehicles': ue_info['ue_stats']['completed_total_vehicles']
        }

        env.close()
        return result

    finally:
        # 恢复原始配置
        settings['demand_multiplier'] = original_multiplier
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)


def main():
    print("=" * 70)
    print("Berlin Friedrichshain demand_multiplier 敏感性测试")
    print("=" * 70)

    multipliers = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    results = []

    for mult in multipliers:
        print(f"\n测试 demand_multiplier = {mult}...")
        result = test_demand_multiplier(mult)
        results.append(result)

        print(f"  迭代次数: {result['iterations']}")
        print(f"  收敛: {result['converged']}")
        print(f"  GM: {result['final_gm']:.2f}%")
        print(f"  P95: {result['final_p95']:.2f}%")
        print(f"  Completed: {result['completed_ratio']*100:.1f}% ({result['completed_vehicles']}/{result['total_vehicles']})")

    # 汇总表格
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"{'Multiplier':<12} {'Iterations':<12} {'GM%':<10} {'P95%':<10} {'Completed%':<12} {'Status'}")
    print("-" * 70)

    for r in results:
        status = "✅" if r['completed_ratio'] >= 0.95 else ("⚠️" if r['completed_ratio'] >= 0.80 else "❌")
        print(f"{r['demand_multiplier']:<12} {r['iterations']:<12} {r['final_gm']:<10.2f} {r['final_p95']:<10.2f} {r['completed_ratio']*100:<12.1f} {status}")

    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'network': 'berlin_friedrichshain',
        'test_type': 'demand_multiplier_sensitivity',
        'results': results
    }

    os.makedirs('results/bf_demand_test', exist_ok=True)
    output_path = 'results/bf_demand_test/demand_multiplier_test.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")

    # 推荐
    print("\n" + "=" * 70)
    print("推荐分析")
    print("=" * 70)

    valid_results = [r for r in results if r['completed_ratio'] >= 0.95]
    if valid_results:
        # 在 completed_ratio >= 95% 的结果中，选择 multiplier 最大的（最有博弈意义）
        best = max(valid_results, key=lambda x: x['demand_multiplier'])
        print(f"推荐 demand_multiplier = {best['demand_multiplier']}")
        print(f"  - completed_ratio: {best['completed_ratio']*100:.1f}%")
        print(f"  - 迭代次数: {best['iterations']}")
        print(f"  - GM: {best['final_gm']:.2f}%")
    else:
        print("警告: 没有找到 completed_ratio >= 95% 的配置")
        best_available = max(results, key=lambda x: x['completed_ratio'])
        print(f"最佳可用: demand_multiplier = {best_available['demand_multiplier']} (completed: {best_available['completed_ratio']*100:.1f}%)")


if __name__ == "__main__":
    main()
