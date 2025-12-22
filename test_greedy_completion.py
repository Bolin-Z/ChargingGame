"""
测试连续随机价格step下的UE-DTA收敛效果
模拟博弈环境中连续20个step的场景，观察完成率约束修改后的效果
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def test_continuous_steps(network_name, n_steps=20, seed=42):
    """
    测试连续随机价格step

    模拟博弈训练过程：
    1. 创建环境并reset
    2. 连续执行n_steps个step，每个step使用随机价格
    3. 使用热启动（step之间保持路径分配）
    4. 记录每个step的UE-DTA收敛情况和完成率
    """
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"连续 {n_steps} 个随机价格 Step 测试")
    print(f"网络: {network_name}, 随机种子: {seed}")
    print(f"{'='*70}")

    # 创建环境
    env = EVCSChargingGameEnv(
        network_dir=f"data/{network_name}",
        network_name=network_name,
        random_seed=seed
    )
    env.reset(seed=seed)

    print(f"充电站: {env.agents}")
    print(f"时段数: {env.n_periods}")
    print(f"UE收敛阈值: {env.ue_convergence_threshold}")
    print(f"UE最小完成率: {env.ue_min_completed_ratio}")
    print(f"{'='*70}\n")

    all_step_data = []

    for step_idx in range(n_steps):
        # 生成随机价格动作 [0, 1]
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.uniform(0.0, 1.0, env.n_periods).astype(np.float32)

        # 执行step
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # 提取UE统计信息（infos 是共享的，不是按agent分开的）
        ue_stats = infos.get('ue_stats', {})

        # 计算完成率
        completed = ue_stats.get('completed_total_vehicles', 0)
        total = ue_stats.get('total_vehicles', 1)
        completed_ratio = completed / total if total > 0 else 0

        # 获取收敛信息
        ue_converged = infos.get('ue_converged', False)
        ue_iterations = infos.get('ue_iterations', 0)
        gm = ue_stats.get('all_relative_gap_global_mean', 0)
        p90 = ue_stats.get('all_relative_gap_p90', 0)
        p95 = ue_stats.get('all_relative_gap_p95', 0)

        # 转换价格用于记录
        actual_prices = env.actions_to_prices_dict(actions)

        step_data = {
            "step": step_idx + 1,
            "prices": {k: [float(x) for x in v] for k, v in actual_prices.items()},
            "ue_converged": ue_converged,
            "ue_iterations": ue_iterations,
            "completed_ratio": completed_ratio,
            "completed_vehicles": completed,
            "total_vehicles": total,
            "gm": gm,
            "p90": p90,
            "p95": p95,
            "rewards": {k: float(v) for k, v in rewards.items()}
        }
        all_step_data.append(step_data)

        # 打印每个step的结果
        status = "✓ 收敛" if ue_converged else "✗ 未收敛"
        print(f"Step {step_idx+1:2d} | {status} | 迭代:{ue_iterations:3d} | "
              f"完成率:{completed_ratio*100:5.1f}% | "
              f"GM:{gm*100:5.2f}% | P95:{p95*100:5.2f}%")

    env.close()

    return all_step_data


def main():
    print("=" * 70)
    print("UE-DTA 完成率约束效果测试")
    print("测试目的: 验证加入完成率约束后，连续step是否能避免虚假收敛")
    print("=" * 70)

    # 运行测试
    results = test_continuous_steps("siouxfalls", n_steps=20, seed=42)

    # 汇总统计
    print(f"\n{'='*70}")
    print("汇总统计")
    print(f"{'='*70}")

    converged_count = sum(1 for r in results if r["ue_converged"])
    avg_iterations = np.mean([r["ue_iterations"] for r in results])
    avg_completed = np.mean([r["completed_ratio"] for r in results])
    min_completed = min(r["completed_ratio"] for r in results)
    max_completed = max(r["completed_ratio"] for r in results)

    print(f"收敛率: {converged_count}/{len(results)} ({converged_count/len(results)*100:.1f}%)")
    print(f"平均迭代次数: {avg_iterations:.1f}")
    print(f"完成率: 均值={avg_completed*100:.1f}%, 最小={min_completed*100:.1f}%, 最大={max_completed*100:.1f}%")

    # 检查是否有虚假收敛（收敛但完成率<95%）
    false_convergence = [r for r in results if r["ue_converged"] and r["completed_ratio"] < 0.95]
    if false_convergence:
        print(f"\n⚠️ 发现 {len(false_convergence)} 个虚假收敛（收敛但完成率<95%）:")
        for r in false_convergence:
            print(f"  Step {r['step']}: 完成率={r['completed_ratio']*100:.1f}%")
    else:
        print(f"\n✓ 未发现虚假收敛（完成率约束生效）")

    # 保存结果
    output = {
        "test_time": datetime.now().isoformat(),
        "network": "siouxfalls",
        "n_steps": len(results),
        "summary": {
            "converged_count": converged_count,
            "avg_iterations": avg_iterations,
            "avg_completed_ratio": avg_completed,
            "min_completed_ratio": min_completed,
            "max_completed_ratio": max_completed,
            "false_convergence_count": len(false_convergence)
        },
        "steps": results
    }

    output_path = "results/ue_completion_constraint_test.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
