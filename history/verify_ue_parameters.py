"""
UE-DTA 参数验证脚本（带收敛曲线绘制）

验证三个网络在更新后参数下的 UE-DTA 收敛效果。
记录每轮迭代的统计数据并绘制收敛曲线。
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


class IterationTrackingEnv(EVCSChargingGameEnv):
    """扩展环境，支持记录每轮迭代数据"""

    def run_with_history(self, n_iterations: int = 100, fixed_price_ratio: float = 0.5) -> dict:
        """
        运行 UE-DTA 迭代并记录每轮数据

        Args:
            n_iterations: 迭代轮数
            fixed_price_ratio: 固定价格比例

        Returns:
            包含每轮统计信息的结果字典
        """
        # 重置状态
        self.price_history = []
        if hasattr(self, 'current_routes_specified'):
            delattr(self, 'current_routes_specified')

        # 设置固定价格
        actions = {}
        for agent in self.agents:
            actions[agent] = np.full(self.n_periods, fixed_price_ratio, dtype=np.float32)

        # 更新价格
        self._EVCSChargingGameEnv__update_prices_from_actions(actions)

        # 获取充电和非充电车辆的OD映射
        dict_od_to_charging_vehid = defaultdict(list)
        dict_od_to_uncharging_vehid = defaultdict(list)
        W_template = self._EVCSChargingGameEnv__create_simulation_world()

        for key, veh in W_template.VEHICLES.items():
            o = veh.orig.name
            d = veh.dest.name
            if veh.attribute["charging_car"]:
                dict_od_to_charging_vehid[(o, d)].append(key)
            else:
                dict_od_to_uncharging_vehid[(o, d)].append(key)

        # 初始化路径分配
        self.current_routes_specified = self._EVCSChargingGameEnv__initialize_routes(
            dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
        )

        # 记录每轮迭代数据
        iteration_history = []
        ue_convergence_counter = 0
        first_convergence_iter = None  # 首次收敛的迭代轮数

        with tqdm(range(n_iterations), desc="UE-DTA迭代", leave=False) as pbar:
            for iteration in pbar:
                # 创建仿真实例
                W = self._EVCSChargingGameEnv__create_simulation_world()

                # 应用当前路径分配
                self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)

                # 执行仿真
                W.exec_simulation()

                # 计算统计信息
                stats, new_routes_specified, charging_flows = self._EVCSChargingGameEnv__route_choice_update(
                    W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
                    self.current_routes_specified, iteration
                )

                # 记录本轮数据
                record = {
                    'iteration': iteration + 1,
                    'gm': stats['all_relative_gap_global_mean'] * 100,
                    'p95': stats['all_relative_gap_p95'] * 100,
                    'od_max': stats['all_relative_gap_od_max_mean'] * 100,
                    'completed_ratio': stats['completed_total_vehicles'] / stats['total_vehicles'] * 100,
                    'route_switches': stats['total_route_switches'],
                    'charging_avg_cost': stats['charging_avg_cost'],
                    'uncharging_avg_cost': stats['uncharging_avg_cost'],
                }
                iteration_history.append(record)

                # 更新路径分配
                self.current_routes_specified = new_routes_specified

                # 收敛检测
                if stats['all_relative_gap_global_mean'] < self.ue_convergence_threshold:
                    ue_convergence_counter += 1
                    if ue_convergence_counter >= self.ue_convergence_stable_rounds and first_convergence_iter is None:
                        first_convergence_iter = iteration + 1
                else:
                    ue_convergence_counter = 0

                # 更新进度条
                pbar.set_description(
                    f"迭代{iteration+1} | GM={record['gm']:.1f}% P95={record['p95']:.1f}% | 切换:{record['route_switches']}"
                )

        return {
            'history': iteration_history,
            'first_convergence_iter': first_convergence_iter,
            'total_vehicles': stats['total_vehicles'],
            'final_stats': stats
        }


def plot_convergence_curves(all_results: dict, output_dir: str):
    """
    绘制收敛曲线

    Args:
        all_results: 所有网络的结果
        output_dir: 输出目录
    """
    # 设置中文字体（Windows系统）
    import matplotlib
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

    os.makedirs(output_dir, exist_ok=True)

    networks = list(all_results.keys())
    n_networks = len(networks)

    # 创建大图：每个网络一行，4个指标（GM, P95, Completed, Switches）
    fig, axes = plt.subplots(n_networks, 4, figsize=(20, 4 * n_networks))

    if n_networks == 1:
        axes = axes.reshape(1, -1)

    colors = {'siouxfalls': '#2ecc71', 'berlin_friedrichshain': '#3498db', 'anaheim': '#e74c3c'}

    for row, network in enumerate(networks):
        result = all_results[network]
        history = result['history']
        convergence_iter = result['first_convergence_iter']
        params = result['params']

        iterations = [h['iteration'] for h in history]
        gm = [h['gm'] for h in history]
        p95 = [h['p95'] for h in history]
        completed = [h['completed_ratio'] for h in history]
        switches = [h['route_switches'] for h in history]

        color = colors.get(network, '#9b59b6')
        title_suffix = f"(γ={params['gamma']}, α={params['alpha']}, dm={params['dm']})"

        # GM 曲线
        ax = axes[row, 0]
        ax.plot(iterations, gm, color=color, linewidth=2, label='GM')
        ax.axhline(y=params['threshold'] * 100, color='red', linestyle='--', alpha=0.7, label=f'阈值 {params["threshold"]*100}%')
        if convergence_iter:
            ax.axvline(x=convergence_iter, color='green', linestyle=':', alpha=0.7, label=f'首次收敛 @{convergence_iter}')
        ax.set_xlabel('迭代轮数')
        ax.set_ylabel('Global Mean (%)')
        ax.set_title(f'{network.upper()} - GM {title_suffix}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # P95 曲线
        ax = axes[row, 1]
        ax.plot(iterations, p95, color=color, linewidth=2, label='P95')
        if convergence_iter:
            ax.axvline(x=convergence_iter, color='green', linestyle=':', alpha=0.7, label=f'首次收敛 @{convergence_iter}')
        ax.set_xlabel('迭代轮数')
        ax.set_ylabel('P95 (%)')
        ax.set_title(f'{network.upper()} - P95')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Completed Ratio 曲线
        ax = axes[row, 2]
        ax.plot(iterations, completed, color=color, linewidth=2, label='Completed Ratio')
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% 阈值')
        if convergence_iter:
            ax.axvline(x=convergence_iter, color='green', linestyle=':', alpha=0.7, label=f'首次收敛 @{convergence_iter}')
        ax.set_xlabel('迭代轮数')
        ax.set_ylabel('完成率 (%)')
        ax.set_title(f'{network.upper()} - Completed Ratio')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(90, 101)

        # Route Switches 曲线
        ax = axes[row, 3]
        ax.plot(iterations, switches, color=color, linewidth=2, label='Route Switches')
        if convergence_iter:
            ax.axvline(x=convergence_iter, color='green', linestyle=':', alpha=0.7, label=f'首次收敛 @{convergence_iter}')
        ax.set_xlabel('迭代轮数')
        ax.set_ylabel('路径切换次数')
        ax.set_title(f'{network.upper()} - Route Switches')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'convergence_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n收敛曲线已保存: {output_path}")

    # 绘制对比图（所有网络在同一图上）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for network in networks:
        result = all_results[network]
        history = result['history']
        convergence_iter = result['first_convergence_iter']

        iterations = [h['iteration'] for h in history]
        gm = [h['gm'] for h in history]
        p95 = [h['p95'] for h in history]
        completed = [h['completed_ratio'] for h in history]
        switches = [h['route_switches'] for h in history]

        color = colors.get(network, '#9b59b6')
        label = f"{network} (conv@{convergence_iter})" if convergence_iter else network

        axes[0, 0].plot(iterations, gm, color=color, linewidth=2, label=label)
        axes[0, 1].plot(iterations, p95, color=color, linewidth=2, label=label)
        axes[1, 0].plot(iterations, completed, color=color, linewidth=2, label=label)
        axes[1, 1].plot(iterations, switches, color=color, linewidth=2, label=label)

    axes[0, 0].set_xlabel('迭代轮数')
    axes[0, 0].set_ylabel('Global Mean (%)')
    axes[0, 0].set_title('GM 收敛对比')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)

    axes[0, 1].set_xlabel('迭代轮数')
    axes[0, 1].set_ylabel('P95 (%)')
    axes[0, 1].set_title('P95 收敛对比')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)

    axes[1, 0].set_xlabel('迭代轮数')
    axes[1, 0].set_ylabel('完成率 (%)')
    axes[1, 0].set_title('Completed Ratio 对比')
    axes[1, 0].axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% 阈值')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(90, 101)

    axes[1, 1].set_xlabel('迭代轮数')
    axes[1, 1].set_ylabel('路径切换次数')
    axes[1, 1].set_title('Route Switches 对比')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {output_path}")


def verify_network(network_dir: str, network_name: str,
                   n_iterations: int = 100,
                   fixed_price_ratio: float = 0.5) -> dict:
    """
    验证单个网络并返回迭代历史

    Args:
        network_dir: 网络数据目录
        network_name: 网络名称
        n_iterations: 迭代轮数
        fixed_price_ratio: 固定价格比例

    Returns:
        验证结果字典（含迭代历史）
    """
    # 读取配置
    settings_path = os.path.join(network_dir, f"{network_name}_settings.json")
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    gamma = settings.get("ue_switch_gamma", 10.0)
    alpha = settings.get("ue_switch_alpha", 0.05)
    dm = settings.get("demand_multiplier", 1.0)
    threshold = settings.get("ue_convergence_threshold", 0.02)

    print(f"\n{'='*60}")
    print(f"验证网络: {network_name.upper()}")
    print(f"{'='*60}")
    print(f"参数: γ={gamma}, α={alpha}, dm={dm}, 收敛阈值={threshold*100}%")
    print(f"迭代轮数: {n_iterations}")

    # 创建环境
    env = IterationTrackingEnv(
        network_dir=network_dir,
        network_name=network_name,
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()

    # 运行迭代
    result = env.run_with_history(n_iterations, fixed_price_ratio)

    # 提取最终结果
    final = result['history'][-1]
    convergence_iter = result['first_convergence_iter']

    print(f"\n结果:")
    print(f"  首次收敛轮数: {convergence_iter if convergence_iter else '未收敛'}")
    print(f"  最终 GM: {final['gm']:.2f}%")
    print(f"  最终 P95: {final['p95']:.2f}%")
    print(f"  最终完成率: {final['completed_ratio']:.1f}%")

    # 如果收敛，显示收敛时的数据
    if convergence_iter:
        conv_data = result['history'][convergence_iter - 1]
        print(f"\n  收敛时 (第{convergence_iter}轮):")
        print(f"    GM: {conv_data['gm']:.2f}%")
        print(f"    P95: {conv_data['p95']:.2f}%")
        print(f"    完成率: {conv_data['completed_ratio']:.1f}%")

    return {
        'network': network_name,
        'params': {'gamma': gamma, 'alpha': alpha, 'dm': dm, 'threshold': threshold},
        'history': result['history'],
        'first_convergence_iter': convergence_iter,
        'total_vehicles': result['total_vehicles']
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="UE-DTA参数验证（带收敛曲线）")
    parser.add_argument("--iterations", type=int, default=100, help="迭代轮数")
    parser.add_argument("--networks", type=str, nargs="+",
                        default=["siouxfalls", "berlin_friedrichshain", "anaheim"],
                        help="要测试的网络列表")
    parser.add_argument("--output", type=str, default="results/verification",
                        help="输出目录")

    args = parser.parse_args()

    # 网络配置
    network_configs = {
        "siouxfalls": ("data/siouxfalls", "siouxfalls"),
        "berlin_friedrichshain": ("data/berlin_friedrichshain", "berlin_friedrichshain"),
        "anaheim": ("data/anaheim", "anaheim"),
    }

    print("=" * 60)
    print("UE-DTA 参数验证测试（带收敛曲线）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"迭代轮数: {args.iterations}")
    print("=" * 60)

    all_results = {}
    for network in args.networks:
        if network not in network_configs:
            print(f"\n⚠️ 跳过未知网络: {network}")
            continue

        network_dir, network_name = network_configs[network]
        try:
            result = verify_network(network_dir, network_name, args.iterations)
            all_results[network] = result
        except Exception as e:
            print(f"\n❌ 错误: {network_name} 验证失败 - {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("没有成功验证的网络")
        return

    # 绘制收敛曲线
    plot_convergence_curves(all_results, args.output)

    # 汇总报告
    print("\n" + "=" * 95)
    print("验证汇总报告")
    print("=" * 95)
    print(f"{'网络':^25} | {'γ':^5} | {'α':^5} | {'首次收敛':^8} | {'收敛时GM':^8} | {'最终GM':^8} | {'最终完成率':^10}")
    print("-" * 95)

    for network, result in all_results.items():
        params = result['params']
        history = result['history']
        conv_iter = result['first_convergence_iter']

        final_gm = history[-1]['gm']
        final_completed = history[-1]['completed_ratio']

        if conv_iter:
            conv_gm = history[conv_iter - 1]['gm']
            conv_str = f"{conv_iter}轮"
            conv_gm_str = f"{conv_gm:.2f}%"
        else:
            conv_str = "未收敛"
            conv_gm_str = "-"

        print(f"{network:^25} | {params['gamma']:^5.0f} | {params['alpha']:^5.2f} | {conv_str:^8} | "
              f"{conv_gm_str:^8} | {final_gm:^7.2f}% | {final_completed:^9.1f}%")

    print("-" * 95)

    # 保存详细数据
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f"verification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # 转换为可序列化格式
    save_data = {}
    for network, result in all_results.items():
        save_data[network] = {
            'params': result['params'],
            'first_convergence_iter': result['first_convergence_iter'],
            'total_vehicles': result['total_vehicles'],
            'history': result['history']
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'iterations': args.iterations,
            'results': save_data
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细数据已保存: {output_file}")


if __name__ == "__main__":
    main()
