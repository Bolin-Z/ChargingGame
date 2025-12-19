"""
UE-DTA 参数扫描实验：gamma 和 alpha

测试不同的切换参数组合对收敛性能的影响：
- ue_switch_gamma: Gap敏感度系数
- ue_switch_alpha: 时间衰减速率

输出：
- 收敛轮数
- 最终GM、P95、OD Max
- completed_ratio
- 参数推荐
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


class ParameterSweepEnv(EVCSChargingGameEnv):
    """扩展环境，支持运行时修改参数"""

    def set_ue_parameters(self, gamma: float, alpha: float):
        """动态设置UE参数"""
        self.ue_switch_gamma = gamma
        self.ue_switch_alpha = alpha

    def run_single_ue(self, fixed_price_ratio: float = 0.5) -> dict:
        """运行单次UE-DTA求解，返回统计信息"""
        # 重置环境状态，避免跨参数组合的状态累积
        self.price_history = []

        # 设置固定价格
        actions = {}
        for agent in self.agents:
            actions[agent] = np.full(self.n_periods, fixed_price_ratio, dtype=np.float32)

        # 更新价格（添加到 price_history）
        self._EVCSChargingGameEnv__update_prices_from_actions(actions)

        # 运行UE-DTA
        charging_flows, ue_info = self._EVCSChargingGameEnv__run_simulation()

        return {
            'converged': ue_info['ue_converged'],
            'iterations': ue_info['ue_iterations'],
            'final_gm': ue_info['ue_stats']['all_relative_gap_global_mean'] * 100,
            'final_p95': ue_info['ue_stats']['all_relative_gap_p95'] * 100,
            'final_od_max': ue_info['ue_stats']['all_relative_gap_od_max_mean'] * 100,
            'completed_ratio': ue_info['ue_stats']['completed_total_vehicles'] / ue_info['ue_stats']['total_vehicles']
        }


def sweep_network(network_dir: str, network_name: str,
                  gamma_values: list, alpha_values: list,
                  fixed_price_ratio: float = 0.5) -> dict:
    """
    对单个网络进行参数扫描

    Args:
        network_dir: 网络数据目录
        network_name: 网络名称
        gamma_values: gamma参数列表
        alpha_values: alpha参数列表
        fixed_price_ratio: 固定价格比例

    Returns:
        扫描结果字典
    """
    print(f"\n{'='*60}")
    print(f"参数扫描: {network_name}")
    print(f"{'='*60}")
    print(f"gamma范围: {gamma_values}")
    print(f"alpha范围: {alpha_values}")
    print(f"总组合数: {len(gamma_values) * len(alpha_values)}")

    # 创建环境
    env = ParameterSweepEnv(
        network_dir=network_dir,
        network_name=network_name,
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()

    # 扫描所有参数组合
    results = []
    total_combinations = len(gamma_values) * len(alpha_values)

    with tqdm(total=total_combinations, desc=f"{network_name} 参数扫描") as pbar:
        for gamma in gamma_values:
            for alpha in alpha_values:
                # 设置参数
                env.set_ue_parameters(gamma, alpha)

                # 运行UE求解
                result = env.run_single_ue(fixed_price_ratio)
                result['gamma'] = gamma
                result['alpha'] = alpha
                results.append(result)

                # 更新进度条
                pbar.set_postfix({
                    'γ': gamma,
                    'α': alpha,
                    'iter': result['iterations'],
                    'GM': f"{result['final_gm']:.1f}%"
                })
                pbar.update(1)

                # 清理路径（确保每次都重新初始化）
                if hasattr(env, 'current_routes_specified'):
                    delattr(env, 'current_routes_specified')

    return {
        'network': network_name,
        'results': results
    }


def analyze_and_plot(all_results: dict, output_dir: str):
    """分析结果并生成可视化"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(output_dir, exist_ok=True)

    networks = list(all_results.keys())

    for network in networks:
        results = all_results[network]['results']

        # 提取gamma和alpha的唯一值
        gamma_values = sorted(list(set(r['gamma'] for r in results)))
        alpha_values = sorted(list(set(r['alpha'] for r in results)))

        # 构建矩阵
        n_gamma = len(gamma_values)
        n_alpha = len(alpha_values)

        # 创建结果矩阵
        iterations_matrix = np.zeros((n_gamma, n_alpha))
        gm_matrix = np.zeros((n_gamma, n_alpha))
        p95_matrix = np.zeros((n_gamma, n_alpha))
        completed_matrix = np.zeros((n_gamma, n_alpha))
        converged_matrix = np.zeros((n_gamma, n_alpha))

        for result in results:
            i = gamma_values.index(result['gamma'])
            j = alpha_values.index(result['alpha'])
            iterations_matrix[i, j] = result['iterations']
            gm_matrix[i, j] = result['final_gm']
            p95_matrix[i, j] = result['final_p95']
            completed_matrix[i, j] = result['completed_ratio'] * 100
            converged_matrix[i, j] = 1 if result['converged'] else 0

        # 绘制热力图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{network.upper()} - Parameter Sweep Results', fontsize=16)

        # 1. 收敛轮数
        im1 = axes[0, 0].imshow(iterations_matrix, cmap='viridis_r', aspect='auto')
        axes[0, 0].set_title('Convergence Iterations')
        axes[0, 0].set_xlabel('Alpha (α)')
        axes[0, 0].set_ylabel('Gamma (γ)')
        axes[0, 0].set_xticks(range(n_alpha))
        axes[0, 0].set_yticks(range(n_gamma))
        axes[0, 0].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[0, 0].set_yticklabels([f'{g:.0f}' for g in gamma_values])
        for i in range(n_gamma):
            for j in range(n_alpha):
                text = axes[0, 0].text(j, i, f'{int(iterations_matrix[i, j])}',
                                       ha="center", va="center", color="w", fontsize=9)
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. 最终GM
        im2 = axes[0, 1].imshow(gm_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
        axes[0, 1].set_title('Final Global Mean (%)')
        axes[0, 1].set_xlabel('Alpha (α)')
        axes[0, 1].set_ylabel('Gamma (γ)')
        axes[0, 1].set_xticks(range(n_alpha))
        axes[0, 1].set_yticks(range(n_gamma))
        axes[0, 1].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[0, 1].set_yticklabels([f'{g:.0f}' for g in gamma_values])
        for i in range(n_gamma):
            for j in range(n_alpha):
                text = axes[0, 1].text(j, i, f'{gm_matrix[i, j]:.1f}',
                                       ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. 最终P95
        im3 = axes[0, 2].imshow(p95_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=30)
        axes[0, 2].set_title('Final P95 (%)')
        axes[0, 2].set_xlabel('Alpha (α)')
        axes[0, 2].set_ylabel('Gamma (γ)')
        axes[0, 2].set_xticks(range(n_alpha))
        axes[0, 2].set_yticks(range(n_gamma))
        axes[0, 2].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[0, 2].set_yticklabels([f'{g:.0f}' for g in gamma_values])
        for i in range(n_gamma):
            for j in range(n_alpha):
                text = axes[0, 2].text(j, i, f'{p95_matrix[i, j]:.1f}',
                                       ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im3, ax=axes[0, 2])

        # 4. Completed Ratio
        im4 = axes[1, 0].imshow(completed_matrix, cmap='RdYlGn', aspect='auto', vmin=80, vmax=100)
        axes[1, 0].set_title('Completed Ratio (%)')
        axes[1, 0].set_xlabel('Alpha (α)')
        axes[1, 0].set_ylabel('Gamma (γ)')
        axes[1, 0].set_xticks(range(n_alpha))
        axes[1, 0].set_yticks(range(n_gamma))
        axes[1, 0].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[1, 0].set_yticklabels([f'{g:.0f}' for g in gamma_values])
        for i in range(n_gamma):
            for j in range(n_alpha):
                text = axes[1, 0].text(j, i, f'{completed_matrix[i, j]:.0f}',
                                       ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im4, ax=axes[1, 0])

        # 5. 收敛状态
        im5 = axes[1, 1].imshow(converged_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 1].set_title('Convergence Status (1=Yes, 0=No)')
        axes[1, 1].set_xlabel('Alpha (α)')
        axes[1, 1].set_ylabel('Gamma (γ)')
        axes[1, 1].set_xticks(range(n_alpha))
        axes[1, 1].set_yticks(range(n_gamma))
        axes[1, 1].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[1, 1].set_yticklabels([f'{g:.0f}' for g in gamma_values])
        for i in range(n_gamma):
            for j in range(n_alpha):
                symbol = '✓' if converged_matrix[i, j] == 1 else '✗'
                text = axes[1, 1].text(j, i, symbol,
                                       ha="center", va="center", color="black", fontsize=12)
        plt.colorbar(im5, ax=axes[1, 1])

        # 6. 综合得分（简化版：GM越低越好 + 迭代越少越好）
        # 归一化后的综合得分
        gm_norm = (gm_matrix - gm_matrix.min()) / (gm_matrix.max() - gm_matrix.min() + 1e-8)
        iter_norm = (iterations_matrix - iterations_matrix.min()) / (iterations_matrix.max() - iterations_matrix.min() + 1e-8)
        score_matrix = 100 * (1 - 0.7 * gm_norm - 0.3 * iter_norm)  # GM权重0.7，迭代权重0.3

        im6 = axes[1, 2].imshow(score_matrix, cmap='RdYlGn', aspect='auto')
        axes[1, 2].set_title('综合得分 (Higher=Better)')
        axes[1, 2].set_xlabel('Alpha (α)')
        axes[1, 2].set_ylabel('Gamma (γ)')
        axes[1, 2].set_xticks(range(n_alpha))
        axes[1, 2].set_yticks(range(n_gamma))
        axes[1, 2].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[1, 2].set_yticklabels([f'{g:.0f}' for g in gamma_values])

        # 标记最佳组合
        best_i, best_j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
        for i in range(n_gamma):
            for j in range(n_alpha):
                color = "white" if (i, j) == (best_i, best_j) else "black"
                weight = "bold" if (i, j) == (best_i, best_j) else "normal"
                text = axes[1, 2].text(j, i, f'{score_matrix[i, j]:.0f}',
                                       ha="center", va="center", color=color, fontsize=9, weight=weight)
        plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{network}_parameter_sweep.png'), dpi=150)
        plt.close()

        # 输出最佳参数
        print(f"\n{network.upper()} 最佳参数组合:")
        print(f"  Gamma: {gamma_values[best_i]}, Alpha: {alpha_values[best_j]}")
        print(f"  GM: {gm_matrix[best_i, best_j]:.2f}%")
        print(f"  P95: {p95_matrix[best_i, best_j]:.2f}%")
        print(f"  迭代次数: {int(iterations_matrix[best_i, best_j])}")
        print(f"  Completed: {completed_matrix[best_i, best_j]:.1f}%")


def generate_report(all_results: dict, output_file: str):
    """生成JSON报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'networks': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="UE-DTA参数扫描")
    parser.add_argument("--gamma", type=float, nargs="+", default=[5, 10, 15, 20],
                        help="Gamma值列表")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.02, 0.03, 0.05, 0.08],
                        help="Alpha值列表")
    parser.add_argument("--output", type=str, default="results/parameter_sweep",
                        help="输出目录")
    parser.add_argument("--networks", type=str, nargs="+",
                        default=["siouxfalls", "berlin_friedrichshain", "anaheim"],
                        help="要测试的网络列表")

    args = parser.parse_args()

    # 网络配置
    network_configs = {
        "siouxfalls": {
            "dir": "data/siouxfalls",
            "name": "siouxfalls"
        },
        "berlin_friedrichshain": {
            "dir": "data/berlin_friedrichshain",
            "name": "berlin_friedrichshain"
        },
        "anaheim": {
            "dir": "data/anaheim",
            "name": "anaheim"
        }
    }

    # 扫描各网络
    all_results = {}
    for network in args.networks:
        if network not in network_configs:
            print(f"警告: 未知网络 {network}，跳过")
            continue

        config = network_configs[network]
        try:
            result = sweep_network(
                network_dir=config["dir"],
                network_name=config["name"],
                gamma_values=args.gamma,
                alpha_values=args.alpha
            )
            all_results[network] = result
        except Exception as e:
            print(f"错误: 分析 {network} 失败 - {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("错误: 没有成功扫描的网络")
        return

    # 生成输出
    os.makedirs(args.output, exist_ok=True)

    # 生成可视化
    analyze_and_plot(all_results, args.output)

    # 生成报告
    generate_report(all_results, os.path.join(args.output, "sweep_report.json"))


if __name__ == "__main__":
    main()
