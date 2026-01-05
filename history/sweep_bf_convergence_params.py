"""
BF 数据集参数扫描：寻找使 GM < 3% 的最优配置

扫描参数：
1. demand_multiplier: 2.0, 2.2, 2.5
2. routes_per_od: 10, 15, 20
3. ue_switch_gamma/alpha 组合

执行命令: python history/sweep_bf_convergence_params.py
"""

import os
import sys
import time
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def run_ue_dta_test(env, n_iterations: int = 50) -> dict:
    """运行 UE-DTA 测试并返回收敛指标"""

    # 设置固定中点价格
    actions = {}
    for agent in env.agents:
        actions[agent] = np.full(env.n_periods, 0.5, dtype=np.float32)
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    # 获取 OD 映射
    dict_od_to_charging_vehid = defaultdict(list)
    dict_od_to_uncharging_vehid = defaultdict(list)
    W_template = env._EVCSChargingGameEnv__create_simulation_world()

    for key, veh in W_template.VEHICLES.items():
        o = veh.orig.name
        d = veh.dest.name
        if veh.attribute.get("charging_car", False):
            dict_od_to_charging_vehid[(o, d)].append(key)
        else:
            dict_od_to_uncharging_vehid[(o, d)].append(key)
    del W_template

    # 初始化路径
    current_routes = env._EVCSChargingGameEnv__initialize_routes(
        dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
    )

    # 运行迭代
    gm_history = []

    for iteration in range(n_iterations):
        W = env._EVCSChargingGameEnv__create_simulation_world()
        env._EVCSChargingGameEnv__apply_routes_to_vehicles(W, current_routes)
        W.exec_simulation()

        stats, new_routes, _ = env._EVCSChargingGameEnv__route_choice_update(
            W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
            current_routes, iteration
        )
        current_routes = new_routes

        gm = stats['all_relative_gap_global_mean'] * 100
        gm_history.append(gm)

        del W

    # 计算最终指标
    final_gm = gm_history[-1]
    min_gm = min(gm_history)
    avg_last_10 = np.mean(gm_history[-10:])

    return {
        'final_gm': final_gm,
        'min_gm': min_gm,
        'avg_last_10': avg_last_10,
        'gm_history': gm_history
    }


def modify_settings(settings_path: str, cfg: dict) -> dict:
    """临时修改配置文件，返回原始配置用于恢复"""
    with open(settings_path, 'r', encoding='utf-8') as f:
        original = json.load(f)

    modified = original.copy()
    modified['demand_multiplier'] = cfg['dm']
    modified['routes_per_od'] = cfg['routes']
    modified['ue_switch_gamma'] = cfg['gamma']
    modified['ue_switch_alpha'] = cfg['alpha']

    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(modified, f, indent=4, ensure_ascii=False)

    return original


def restore_settings(settings_path: str, original: dict):
    """恢复原始配置"""
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(original, f, indent=4, ensure_ascii=False)


def main():
    print("=" * 70)
    print("BF 数据集参数扫描")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    settings_path = "data/berlin_friedrichshain/berlin_friedrichshain_settings.json"

    # 参数组合
    configs = [
        # 基准配置
        {"name": "baseline", "dm": 2.5, "routes": 10, "gamma": 10.0, "alpha": 0.05},

        # 降低 demand_multiplier
        {"name": "dm_2.2", "dm": 2.2, "routes": 10, "gamma": 10.0, "alpha": 0.05},
        {"name": "dm_2.0", "dm": 2.0, "routes": 10, "gamma": 10.0, "alpha": 0.05},
        {"name": "dm_1.8", "dm": 1.8, "routes": 10, "gamma": 10.0, "alpha": 0.05},

        # 增加 routes_per_od
        {"name": "routes_15", "dm": 2.5, "routes": 15, "gamma": 10.0, "alpha": 0.05},
        {"name": "routes_20", "dm": 2.5, "routes": 20, "gamma": 10.0, "alpha": 0.05},

        # SF 风格参数
        {"name": "sf_style", "dm": 2.5, "routes": 10, "gamma": 5.0, "alpha": 0.08},

        # 组合优化
        {"name": "dm2.0_routes15", "dm": 2.0, "routes": 15, "gamma": 10.0, "alpha": 0.05},
        {"name": "dm2.0_sf_style", "dm": 2.0, "routes": 10, "gamma": 5.0, "alpha": 0.08},
        {"name": "best_combo", "dm": 2.0, "routes": 15, "gamma": 5.0, "alpha": 0.08},
    ]

    results = []
    n_iterations = 50  # 每个配置运行50轮

    print(f"\n共 {len(configs)} 个配置，每个运行 {n_iterations} 轮 UE-DTA")
    print("-" * 70)

    # 保存原始配置
    with open(settings_path, 'r', encoding='utf-8') as f:
        original_settings = json.load(f)

    try:
        for i, cfg in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] 测试配置: {cfg['name']}")
            print(f"  dm={cfg['dm']}, routes={cfg['routes']}, gamma={cfg['gamma']}, alpha={cfg['alpha']}")

            start_time = time.time()

            try:
                # 修改配置文件
                modify_settings(settings_path, cfg)

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

                # 运行测试
                result = run_ue_dta_test(env, n_iterations)

                elapsed = time.time() - start_time

                print(f"  结果: final_GM={result['final_gm']:.2f}%, min_GM={result['min_gm']:.2f}%, "
                      f"avg_last10={result['avg_last_10']:.2f}%, 耗时={elapsed:.1f}s")

                results.append({
                    'config': cfg,
                    'final_gm': result['final_gm'],
                    'min_gm': result['min_gm'],
                    'avg_last_10': result['avg_last_10'],
                    'elapsed': elapsed,
                    'success': True
                })

                env.close()

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'config': cfg,
                    'final_gm': None,
                    'min_gm': None,
                    'avg_last_10': None,
                    'elapsed': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                })

    finally:
        # 恢复原始配置
        restore_settings(settings_path, original_settings)
        print("\n配置文件已恢复")

    # 汇总结果
    print("\n" + "=" * 70)
    print("扫描结果汇总")
    print("=" * 70)

    print(f"\n{'配置名':<20} {'dm':<6} {'routes':<8} {'gamma':<7} {'alpha':<7} {'final_GM%':<10} {'min_GM%':<10} {'状态':<8}")
    print("-" * 90)

    successful = [r for r in results if r['success']]

    for r in results:
        cfg = r['config']
        if r['success']:
            status = "✅" if r['final_gm'] < 3.0 else "❌"
            print(f"{cfg['name']:<20} {cfg['dm']:<6} {cfg['routes']:<8} {cfg['gamma']:<7} {cfg['alpha']:<7} "
                  f"{r['final_gm']:<10.2f} {r['min_gm']:<10.2f} {status}")
        else:
            print(f"{cfg['name']:<20} {cfg['dm']:<6} {cfg['routes']:<8} {cfg['gamma']:<7} {cfg['alpha']:<7} "
                  f"{'FAIL':<10} {'FAIL':<10} ❌")

    # 找出最佳配置
    if successful:
        best = min(successful, key=lambda x: x['final_gm'])
        print(f"\n最佳配置: {best['config']['name']}")
        print(f"  demand_multiplier: {best['config']['dm']}")
        print(f"  routes_per_od: {best['config']['routes']}")
        print(f"  ue_switch_gamma: {best['config']['gamma']}")
        print(f"  ue_switch_alpha: {best['config']['alpha']}")
        print(f"  final_GM: {best['final_gm']:.2f}%")
        print(f"  min_GM: {best['min_gm']:.2f}%")

        if best['final_gm'] < 3.0:
            print(f"\n🎉 找到使 GM < 3% 的配置！")
        else:
            print(f"\n⚠️ 最佳配置仍未达到 3% 阈值，差距: {best['final_gm'] - 3.0:.2f}%")

    # 保存结果
    output_file = "history/sweep_bf_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_iterations': n_iterations,
            'results': [{
                'config': r['config'],
                'final_gm': r['final_gm'],
                'min_gm': r['min_gm'],
                'avg_last_10': r['avg_last_10'],
                'elapsed': r['elapsed'],
                'success': r['success']
            } for r in results]
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
