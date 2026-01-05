"""
BF 数据集 UE-DTA 收敛配置快速测试

测试不同参数组合对 GM Gap 的影响，找出能收敛到 3% 以下的配置。
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


class ConfigTester:
    """配置测试器"""

    def __init__(self):
        self.settings_path = "data/berlin_friedrichshain/berlin_friedrichshain_settings.json"
        self.backup_settings = None

    def load_and_backup(self):
        """加载并备份原始配置"""
        with open(self.settings_path, 'r', encoding='utf-8') as f:
            self.backup_settings = json.load(f)
        return self.backup_settings.copy()

    def restore(self):
        """恢复原始配置"""
        if self.backup_settings:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.backup_settings, f, indent=4, ensure_ascii=False)

    def apply_config(self, settings: dict, config: dict):
        """应用测试配置"""
        for key, value in config.items():
            settings[key] = value
        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

    def run_test(self, config: dict, config_name: str) -> dict:
        """运行单次测试"""
        print(f"\n{'='*60}")
        print(f"测试配置: {config_name}")
        print(f"{'='*60}")
        for k, v in config.items():
            print(f"  {k}: {v}")

        # 加载并修改配置
        settings = self.load_and_backup()
        self.apply_config(settings, config)

        start_time = time.time()

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

            # 设置固定中点价格
            actions = {}
            for agent in env.agents:
                actions[agent] = np.full(env.n_periods, 0.5, dtype=np.float32)

            env._EVCSChargingGameEnv__update_prices_from_actions(actions)

            # 运行 UE-DTA
            charging_flows, ue_info = env._EVCSChargingGameEnv__run_simulation()

            elapsed = time.time() - start_time

            result = {
                'config_name': config_name,
                'config': config,
                'converged': ue_info['ue_converged'],
                'iterations': ue_info['ue_iterations'],
                'final_gm': ue_info['ue_stats']['all_relative_gap_global_mean'] * 100,
                'final_p95': ue_info['ue_stats']['all_relative_gap_p95'] * 100,
                'completed_ratio': ue_info['ue_stats']['completed_total_vehicles'] / ue_info['ue_stats']['total_vehicles'],
                'total_vehicles': ue_info['ue_stats']['total_vehicles'],
                'elapsed_seconds': elapsed
            }

            env.close()

            # 打印结果
            status = "CONVERGED" if result['converged'] else "NOT CONVERGED"
            gm_status = "< 3%" if result['final_gm'] < 3.0 else ">= 3%"
            print(f"\n结果: {status}")
            print(f"  迭代次数: {result['iterations']}")
            print(f"  GM Gap: {result['final_gm']:.2f}% ({gm_status})")
            print(f"  P95 Gap: {result['final_p95']:.2f}%")
            print(f"  完成率: {result['completed_ratio']*100:.1f}%")
            print(f"  耗时: {elapsed:.1f}s")

            return result

        finally:
            self.restore()


def main():
    print("=" * 70)
    print("BF 数据集 UE-DTA 收敛配置测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 测试配置列表
    configs = {
        # 基准：当前配置
        "A_baseline": {
            "ue_max_iterations": 100,
            "ue_switch_gamma": 10.0,
            "ue_switch_alpha": 0.05,
            "demand_multiplier": 2.5
        },
        # 增加迭代次数
        "B_more_iters": {
            "ue_max_iterations": 200,
            "ue_switch_gamma": 10.0,
            "ue_switch_alpha": 0.05,
            "demand_multiplier": 2.5
        },
        # 更激进的切换 + 更多迭代
        "C_aggressive": {
            "ue_max_iterations": 200,
            "ue_switch_gamma": 15.0,
            "ue_switch_alpha": 0.08,
            "demand_multiplier": 2.5
        },
        # 降低需求
        "D_lower_demand": {
            "ue_max_iterations": 100,
            "ue_switch_gamma": 10.0,
            "ue_switch_alpha": 0.05,
            "demand_multiplier": 2.0
        },
        # 更低需求
        "E_lowest_demand": {
            "ue_max_iterations": 100,
            "ue_switch_gamma": 10.0,
            "ue_switch_alpha": 0.05,
            "demand_multiplier": 1.8
        },
        # 组合：低需求 + 更多迭代
        "F_combined": {
            "ue_max_iterations": 150,
            "ue_switch_gamma": 12.0,
            "ue_switch_alpha": 0.06,
            "demand_multiplier": 2.2
        },
    }

    tester = ConfigTester()
    results = []

    for config_name, config in configs.items():
        try:
            result = tester.run_test(config, config_name)
            results.append(result)
        except Exception as e:
            print(f"\n错误: {config_name} 测试失败 - {e}")
            import traceback
            traceback.print_exc()

    # 汇总表格
    print("\n" + "=" * 90)
    print("汇总结果")
    print("=" * 90)
    print(f"{'配置':<18} {'iters':<6} {'gamma':<6} {'alpha':<6} {'dm':<5} {'GM%':<8} {'P95%':<8} {'完成率':<8} {'状态'}")
    print("-" * 90)

    for r in results:
        cfg = r['config']
        status = "OK" if r['final_gm'] < 3.0 else ("CLOSE" if r['final_gm'] < 4.0 else "FAR")
        print(f"{r['config_name']:<18} {cfg['ue_max_iterations']:<6} {cfg['ue_switch_gamma']:<6.1f} "
              f"{cfg['ue_switch_alpha']:<6.2f} {cfg['demand_multiplier']:<5.1f} "
              f"{r['final_gm']:<8.2f} {r['final_p95']:<8.2f} {r['completed_ratio']*100:<8.1f} {status}")

    print("-" * 90)

    # 找最佳配置
    valid_results = [r for r in results if r['completed_ratio'] >= 0.95]
    if valid_results:
        best = min(valid_results, key=lambda x: x['final_gm'])
        print(f"\n最佳配置: {best['config_name']}")
        print(f"  GM Gap: {best['final_gm']:.2f}%")
        print(f"  配置: {best['config']}")

        if best['final_gm'] < 3.0:
            print("\n SUCCESS: 找到了能收敛到 3% 以下的配置!")
        else:
            print(f"\n 最低 GM Gap 为 {best['final_gm']:.2f}%，仍高于 3% 阈值")
            print("  建议: 尝试进一步降低 demand_multiplier 或增加迭代次数")

    # 保存结果
    os.makedirs('results/bf_config_test', exist_ok=True)
    output_path = f'results/bf_config_test/config_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
