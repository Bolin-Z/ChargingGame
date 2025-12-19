"""
比较三种UE-DTA收敛指标在三个数据集上的表现

指标：
1. Global Mean: 全局平均相对成本差
2. OD Max Mean: 按OD分组后各组平均值的最大值
3. P95: 95百分位数

数据集：
1. Sioux Falls (小型理论网络)
2. Berlin Friedrichshain (中型真实网络)
3. Anaheim (大型真实网络)

输出：
- 各指标随迭代的收敛曲线
- 伪均衡检测能力对比
- 收敛速度与稳定性分析
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from uxsim import World


class MetricsAnalysisEnv(EVCSChargingGameEnv):
    """扩展环境类，用于收集三种聚合指标的迭代数据"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = []

    def run_analysis(self, forced_iterations: int, fixed_price_ratio: float = 0.5) -> dict:
        """
        运行UE-DTA并收集每轮的三种指标

        Args:
            forced_iterations: 强制迭代次数（忽略收敛判断）
            fixed_price_ratio: 固定价格比例（0-1）

        Returns:
            dict: 包含所有迭代数据的结果
        """
        self.metrics_history = []

        # 设置固定价格
        actions = {}
        for agent in self.agents:
            actions[agent] = np.full(self.n_periods, fixed_price_ratio, dtype=np.float32)

        prices_matrix = self.actions_to_prices_matrix(actions)
        self.current_prices = prices_matrix
        self.price_history.append(prices_matrix.copy())

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
        if not hasattr(self, 'current_routes_specified') or not self.current_routes_specified:
            self.current_routes_specified = self._EVCSChargingGameEnv__initialize_routes(
                dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, use_greedy=True
            )

        # 运行迭代
        with tqdm(range(forced_iterations), desc="指标分析", leave=True,
                  bar_format='{desc} | {n}/{total} [{elapsed}<{remaining}]') as pbar:
            for iteration in pbar:
                # 创建新的仿真实例
                W = self._EVCSChargingGameEnv__create_simulation_world()

                # 应用当前路径分配
                self._EVCSChargingGameEnv__apply_routes_to_vehicles(W, self.current_routes_specified)

                # 执行仿真
                W.exec_simulation()

                # 执行路径切换并获取统计信息
                stats, new_routes, charging_flows = self._EVCSChargingGameEnv__route_choice_update(
                    W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid, self.current_routes_specified, iteration
                )

                # 记录三种指标
                iteration_data = {
                    "iteration": iteration + 1,
                    "global_mean": stats['all_relative_gap_global_mean'],
                    "od_max_mean": stats['all_relative_gap_od_max_mean'],
                    "p95": stats['all_relative_gap_p95'],
                    "charging_global_mean": stats['charging_relative_gap_global_mean'],
                    "charging_od_max_mean": stats['charging_relative_gap_od_max_mean'],
                    "charging_p95": stats['charging_relative_gap_p95'],
                    "route_switches": stats['total_route_switches'],
                    "completed_ratio": stats['completed_total_vehicles'] / stats['total_vehicles'] if stats['total_vehicles'] > 0 else 0
                }
                self.metrics_history.append(iteration_data)

                # 更新路径分配
                self.current_routes_specified = new_routes

                # 更新进度条
                pbar.set_description(
                    f"迭代{iteration+1} | GM:{iteration_data['global_mean']*100:.1f}% "
                    f"ODMax:{iteration_data['od_max_mean']*100:.1f}% "
                    f"P95:{iteration_data['p95']*100:.1f}%"
                )

        return {
            "iterations": self.metrics_history,
            "config": {
                "forced_iterations": forced_iterations,
                "fixed_price_ratio": fixed_price_ratio,
                "time_value_coefficient": self.time_value_coefficient,
                "ue_convergence_threshold": self.ue_convergence_threshold
            }
        }


def analyze_network(network_dir: str, network_name: str, iterations: int = 100,
                    fixed_price_ratio: float = 0.5) -> dict:
    """分析单个网络的三种指标表现"""
    print(f"\n{'='*60}")
    print(f"分析网络: {network_name}")
    print(f"{'='*60}")

    env = MetricsAnalysisEnv(
        network_dir=network_dir,
        network_name=network_name,
        random_seed=42,
        max_steps=10,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    obs, info = env.reset()
    results = env.run_analysis(forced_iterations=iterations, fixed_price_ratio=fixed_price_ratio)
    results["network"] = network_name

    # 计算汇总统计
    summary = compute_summary(results["iterations"])
    results["summary"] = summary

    print(f"\n{network_name} 汇总:")
    print(f"  初始值 - GM: {summary['initial']['global_mean']*100:.2f}% | "
          f"ODMax: {summary['initial']['od_max_mean']*100:.2f}% | "
          f"P95: {summary['initial']['p95']*100:.2f}%")
    print(f"  最终值 - GM: {summary['final']['global_mean']*100:.2f}% | "
          f"ODMax: {summary['final']['od_max_mean']*100:.2f}% | "
          f"P95: {summary['final']['p95']*100:.2f}%")
    print(f"  收敛轮数 (阈值2%) - GM: {summary['convergence_iter']['global_mean']} | "
          f"ODMax: {summary['convergence_iter']['od_max_mean']} | "
          f"P95: {summary['convergence_iter']['p95']}")

    return results


def compute_summary(iterations: list) -> dict:
    """计算迭代数据的汇总统计"""
    if not iterations:
        return {}

    threshold = 0.02  # 2% 收敛阈值

    # 初始值和最终值
    initial = iterations[0]
    final = iterations[-1]

    # 计算各指标首次达到收敛阈值的轮数
    def first_converge_iter(metric_name):
        for it in iterations:
            if it[metric_name] < threshold:
                return it["iteration"]
        return None  # 未收敛

    # 计算波动性（标准差）
    def compute_stability(metric_name):
        values = [it[metric_name] for it in iterations]
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }

    return {
        "initial": {
            "global_mean": initial["global_mean"],
            "od_max_mean": initial["od_max_mean"],
            "p95": initial["p95"]
        },
        "final": {
            "global_mean": final["global_mean"],
            "od_max_mean": final["od_max_mean"],
            "p95": final["p95"]
        },
        "convergence_iter": {
            "global_mean": first_converge_iter("global_mean"),
            "od_max_mean": first_converge_iter("od_max_mean"),
            "p95": first_converge_iter("p95")
        },
        "stability": {
            "global_mean": compute_stability("global_mean"),
            "od_max_mean": compute_stability("od_max_mean"),
            "p95": compute_stability("p95")
        }
    }


def plot_comparison(all_results: dict, output_dir: str):
    """生成对比图表"""
    os.makedirs(output_dir, exist_ok=True)

    networks = list(all_results.keys())
    colors = {'global_mean': '#2196F3', 'od_max_mean': '#FF5722', 'p95': '#4CAF50'}
    labels = {'global_mean': 'Global Mean', 'od_max_mean': 'OD Max Mean', 'p95': 'P95'}

    # 1. 三个网络的收敛曲线对比（3x1子图）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Convergence Metrics Comparison Across Networks', fontsize=14)

    for idx, network in enumerate(networks):
        ax = axes[idx]
        data = all_results[network]["iterations"]
        iters = [d["iteration"] for d in data]

        for metric, color in colors.items():
            values = [d[metric] * 100 for d in data]  # 转为百分比
            ax.plot(iters, values, color=color, label=labels[metric], linewidth=2)

        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Threshold (2%)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Cost Gap (%)')
        ax.set_title(network.replace('_', ' ').title())
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'), dpi=150)
    plt.close()

    # 2. 指标差异分析（ODMax/GM 比值）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Ratio Analysis: OD Max Mean / Global Mean (Pseudo-Equilibrium Detection)', fontsize=14)

    for idx, network in enumerate(networks):
        ax = axes[idx]
        data = all_results[network]["iterations"]
        iters = [d["iteration"] for d in data]

        # 计算 ODMax/GM 比值
        ratios = []
        for d in data:
            if d["global_mean"] > 0.001:
                ratios.append(d["od_max_mean"] / d["global_mean"])
            else:
                ratios.append(1.0)

        ax.plot(iters, ratios, color='#9C27B0', linewidth=2)
        ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Warning (2x)')
        ax.axhline(y=3, color='red', linestyle=':', alpha=0.5, label='Critical (3x)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ODMax / GlobalMean Ratio')
        ax.set_title(network.replace('_', ' ').title())
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ratio_analysis.png'), dpi=150)
    plt.close()

    # 3. 收敛轮数对比柱状图
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(networks))
    width = 0.25

    for i, (metric, color) in enumerate(colors.items()):
        conv_iters = []
        for network in networks:
            ci = all_results[network]["summary"]["convergence_iter"][metric]
            conv_iters.append(ci if ci is not None else 100)  # 未收敛设为最大值

        bars = ax.bar(x + i * width, conv_iters, width, label=labels[metric], color=color)

        # 标注未收敛
        for j, ci in enumerate(conv_iters):
            if all_results[networks[j]]["summary"]["convergence_iter"][metric] is None:
                ax.annotate('N/C', (x[j] + i * width, ci), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Network')
    ax.set_ylabel('Iterations to Converge (< 2%)')
    ax.set_title('Convergence Speed Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace('_', ' ').title() for n in networks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_speed.png'), dpi=150)
    plt.close()

    # 4. 最终值对比
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, color) in enumerate(colors.items()):
        final_values = [all_results[n]["summary"]["final"][metric] * 100 for n in networks]
        ax.bar(x + i * width, final_values, width, label=labels[metric], color=color)

    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Threshold (2%)')
    ax.set_xlabel('Network')
    ax.set_ylabel('Final Relative Cost Gap (%)')
    ax.set_title('Final Convergence Values')
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace('_', ' ').title() for n in networks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_values.png'), dpi=150)
    plt.close()

    print(f"\n图表已保存到: {output_dir}")


def generate_report(all_results: dict, output_file: str):
    """生成分析报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "networks": {},
        "comparison": {},
        "recommendations": {}
    }

    # 各网络详细数据
    for network, data in all_results.items():
        report["networks"][network] = {
            "config": data["config"],
            "summary": data["summary"],
            "iterations": data["iterations"]
        }

    # 对比分析
    threshold = 0.02
    comparison = {
        "convergence_first": {},  # 哪个指标最先达到收敛
        "pseudo_equilibrium_detection": {},  # 伪均衡检测能力
        "stability": {}  # 稳定性对比
    }

    for network, data in all_results.items():
        summary = data["summary"]

        # 最先收敛的指标
        conv_iters = summary["convergence_iter"]
        valid_iters = {k: v for k, v in conv_iters.items() if v is not None}
        if valid_iters:
            first_metric = min(valid_iters, key=valid_iters.get)
            comparison["convergence_first"][network] = {
                "metric": first_metric,
                "iteration": valid_iters[first_metric]
            }
        else:
            comparison["convergence_first"][network] = {"metric": None, "iteration": None}

        # 伪均衡检测：初始 ODMax/GM 比值
        initial = summary["initial"]
        if initial["global_mean"] > 0.001:
            ratio = initial["od_max_mean"] / initial["global_mean"]
            comparison["pseudo_equilibrium_detection"][network] = {
                "initial_ratio": round(ratio, 2),
                "risk": "HIGH" if ratio > 3 else ("MEDIUM" if ratio > 2 else "LOW")
            }

        # 稳定性（最终10轮的标准差）
        final_10 = data["iterations"][-10:] if len(data["iterations"]) >= 10 else data["iterations"]
        for metric in ["global_mean", "od_max_mean", "p95"]:
            values = [d[metric] for d in final_10]
            if network not in comparison["stability"]:
                comparison["stability"][network] = {}
            comparison["stability"][network][metric] = {
                "mean": round(np.mean(values) * 100, 3),
                "std": round(np.std(values) * 100, 3)
            }

    report["comparison"] = comparison

    # 生成建议
    recommendations = []

    # 分析各网络的最佳指标
    for network, conv in comparison["convergence_first"].items():
        if conv["metric"] == "global_mean":
            pe_risk = comparison["pseudo_equilibrium_detection"].get(network, {}).get("risk", "LOW")
            if pe_risk in ["MEDIUM", "HIGH"]:
                recommendations.append(
                    f"{network}: Global Mean 最先收敛，但伪均衡风险为 {pe_risk}，"
                    f"建议使用 OD Max Mean 或双重标准"
                )
            else:
                recommendations.append(
                    f"{network}: Global Mean 表现良好，伪均衡风险低"
                )

    # 总体建议
    all_pe_risks = [v.get("risk") for v in comparison["pseudo_equilibrium_detection"].values()]
    if "HIGH" in all_pe_risks:
        recommendations.append(
            "总体建议: 存在高伪均衡风险网络，推荐使用双重收敛标准: "
            "Global Mean < 2% AND OD Max Mean < 5%"
        )
    else:
        recommendations.append(
            "总体建议: 各网络伪均衡风险可控，可使用 Global Mean < 2% 作为主要收敛判据"
        )

    report["recommendations"] = recommendations

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存到: {output_file}")

    # 打印关键结论
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)
    for rec in recommendations:
        print(f"• {rec}")

    return report


def main():
    """主函数：在三个数据集上比较三种指标"""
    import argparse

    parser = argparse.ArgumentParser(description="比较三种UE-DTA收敛指标")
    parser.add_argument("--iterations", type=int, default=100, help="每个网络的UE迭代次数")
    parser.add_argument("--price", type=float, default=0.5, help="固定价格比例")
    parser.add_argument("--output", type=str, default="results/metrics_comparison", help="输出目录")
    parser.add_argument("--networks", type=str, nargs="+",
                        default=["siouxfalls", "berlin_friedrichshain", "anaheim"],
                        help="要分析的网络列表")

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

    # 分析各网络
    all_results = {}
    for network in args.networks:
        if network not in network_configs:
            print(f"警告: 未知网络 {network}，跳过")
            continue

        config = network_configs[network]
        try:
            results = analyze_network(
                network_dir=config["dir"],
                network_name=config["name"],
                iterations=args.iterations,
                fixed_price_ratio=args.price
            )
            all_results[network] = results
        except Exception as e:
            print(f"错误: 分析 {network} 失败 - {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("错误: 没有成功分析的网络")
        return

    # 生成输出
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 生成图表
    plot_comparison(all_results, output_dir)

    # 生成报告
    report = generate_report(all_results, os.path.join(output_dir, "comparison_report.json"))


if __name__ == "__main__":
    main()
