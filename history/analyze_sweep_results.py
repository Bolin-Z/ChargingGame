"""
重新分析 sweep_report.json，使用 completed_ratio >= 95% 硬约束
"""

import json
import numpy as np

def analyze_network(network_name: str, results: list) -> dict:
    """分析单个网络的结果"""
    print(f"\n{'='*60}")
    print(f"{network_name.upper()}")
    print(f"{'='*60}")

    # 提取参数值
    gamma_values = sorted(list(set(r['gamma'] for r in results)))
    alpha_values = sorted(list(set(r['alpha'] for r in results)))

    # 构建矩阵
    n_gamma = len(gamma_values)
    n_alpha = len(alpha_values)

    gm_matrix = np.zeros((n_gamma, n_alpha))
    p95_matrix = np.zeros((n_gamma, n_alpha))
    iter_matrix = np.zeros((n_gamma, n_alpha))
    completed_matrix = np.zeros((n_gamma, n_alpha))

    for r in results:
        i = gamma_values.index(r['gamma'])
        j = alpha_values.index(r['alpha'])
        gm_matrix[i, j] = r['final_gm']
        p95_matrix[i, j] = r['final_p95']
        iter_matrix[i, j] = r['iterations']
        completed_matrix[i, j] = r['completed_ratio'] * 100

    # 打印 completed_ratio 矩阵
    print(f"\nCompleted Ratio (%) 矩阵:")
    header = "gamma\\alpha"
    print(f"{header:>12}", end="")
    for a in alpha_values:
        print(f"{a:>10.2f}", end="")
    print()

    for i, g in enumerate(gamma_values):
        print(f"{g:>12.0f}", end="")
        for j in range(n_alpha):
            val = completed_matrix[i, j]
            marker = " ⚠️" if val < 95 else ""
            print(f"{val:>8.1f}{marker}", end="")
        print()

    # 硬约束：completed_ratio >= 95%
    valid_mask = completed_matrix >= 95
    valid_count = np.sum(valid_mask)
    total_count = n_gamma * n_alpha

    print(f"\n有效配置: {valid_count}/{total_count} (completed_ratio >= 95%)")

    if valid_count == 0:
        print("⚠️ 没有满足约束的配置！")
        return None

    # 在有效配置中计算得分
    # 只对有效配置进行归一化
    valid_gm = gm_matrix[valid_mask]
    valid_iter = iter_matrix[valid_mask]

    gm_min, gm_max = valid_gm.min(), valid_gm.max()
    iter_min, iter_max = valid_iter.min(), valid_iter.max()

    # 归一化
    gm_norm = np.zeros_like(gm_matrix)
    iter_norm = np.zeros_like(iter_matrix)

    if gm_max > gm_min:
        gm_norm = (gm_matrix - gm_min) / (gm_max - gm_min)
    if iter_max > iter_min:
        iter_norm = (iter_matrix - iter_min) / (iter_max - iter_min)

    # 计算得分（GM权重0.7，迭代权重0.3）
    score_matrix = 100 * (1 - 0.7 * gm_norm - 0.3 * iter_norm)

    # 应用硬约束
    score_matrix[~valid_mask] = -1  # 无效配置标记为-1

    # 打印得分矩阵
    print(f"\n综合得分矩阵 (无效配置标记为 --):")
    print(f"{header:>12}", end="")
    for a in alpha_values:
        print(f"{a:>10.2f}", end="")
    print()

    for i, g in enumerate(gamma_values):
        print(f"{g:>12.0f}", end="")
        for j in range(n_alpha):
            if valid_mask[i, j]:
                print(f"{score_matrix[i, j]:>10.1f}", end="")
            else:
                print(f"{'--':>10}", end="")
        print()

    # 找最佳配置
    best_idx = np.argmax(score_matrix)
    best_i, best_j = np.unravel_index(best_idx, score_matrix.shape)

    best_result = {
        'gamma': gamma_values[best_i],
        'alpha': alpha_values[best_j],
        'gm': gm_matrix[best_i, best_j],
        'p95': p95_matrix[best_i, best_j],
        'iterations': int(iter_matrix[best_i, best_j]),
        'completed_ratio': completed_matrix[best_i, best_j],
        'score': score_matrix[best_i, best_j]
    }

    print(f"\n✅ 最佳参数 (满足 completed_ratio >= 95%):")
    print(f"   gamma = {best_result['gamma']}")
    print(f"   alpha = {best_result['alpha']}")
    print(f"   GM = {best_result['gm']:.2f}%")
    print(f"   P95 = {best_result['p95']:.2f}%")
    print(f"   迭代次数 = {best_result['iterations']}")
    print(f"   Completed = {best_result['completed_ratio']:.1f}%")
    print(f"   得分 = {best_result['score']:.1f}")

    return best_result


def main():
    # 读取报告
    with open('results/parameter_sweep/sweep_report.json', 'r') as f:
        report = json.load(f)

    print("=" * 60)
    print("参数扫描结果重新分析")
    print("约束条件: completed_ratio >= 95%")
    print("=" * 60)

    # 分析各网络
    best_params = {}
    for network, data in report['networks'].items():
        result = analyze_network(network, data['results'])
        if result:
            best_params[network] = result

    # 总结
    print("\n" + "=" * 60)
    print("最佳参数汇总 (满足 completed_ratio >= 95%)")
    print("=" * 60)
    print(f"{'网络':<25} {'gamma':>8} {'alpha':>8} {'GM%':>8} {'P95%':>8} {'Iter':>6} {'Completed':>10}")
    print("-" * 60)

    for network, params in best_params.items():
        print(f"{network:<25} {params['gamma']:>8.0f} {params['alpha']:>8.2f} "
              f"{params['gm']:>8.2f} {params['p95']:>8.2f} {params['iterations']:>6} "
              f"{params['completed_ratio']:>9.1f}%")

    # 输出建议的 settings.json 更新
    print("\n" + "=" * 60)
    print("建议更新 settings.json:")
    print("=" * 60)
    for network, params in best_params.items():
        print(f"\n{network}:")
        print(f'  "ue_switch_gamma": {params["gamma"]},')
        print(f'  "ue_switch_alpha": {params["alpha"]}')


if __name__ == "__main__":
    main()
