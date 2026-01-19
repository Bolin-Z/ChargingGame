"""
多Seed实验 - 验证多均衡解假设
使用5个不同seed运行三个算法
"""

import subprocess
import sys
from pathlib import Path

# 实验配置
SEEDS = [42, 123, 456, 789, 1024]
ALGORITHMS = ['maddpg', 'iddpg', 'mfddpg']

def main():
    print("=" * 60)
    print("多Seed实验 - 验证多均衡解假设")
    print(f"Seeds: {SEEDS}")
    print(f"Algorithms: {ALGORITHMS}")
    print("=" * 60)

    # 检查 test_flow_scale.py 是否存在
    test_script = Path(__file__).parent / 'test_flow_scale.py'
    if not test_script.exists():
        print(f"错误: 找不到 {test_script}")
        return

    # 生成所有实验组合
    experiments = []
    for seed in SEEDS:
        for algo in ALGORITHMS:
            experiments.append((algo, seed))

    print(f"\n总计 {len(experiments)} 个实验")
    print("\n生成启动命令...")

    # 生成 batch 文件
    bat_content = '@echo off\nchcp 65001 >nul\n'
    bat_content += 'echo ========================================\n'
    bat_content += 'echo Multi-Seed Experiment (5 seeds x 3 algos)\n'
    bat_content += 'echo ========================================\n'
    bat_content += 'echo.\n\n'

    for algo, seed in experiments:
        window_title = f"{algo.upper()}_Seed{seed}"
        cmd = f'start "{window_title}" cmd /k "conda activate drl && python test_flow_scale.py -a {algo} --seed {seed}"'
        bat_content += f'{cmd}\n'
        bat_content += 'timeout /t 3 /nobreak >nul\n'

    bat_content += '\necho.\n'
    bat_content += f'echo Started {len(experiments)} experiments\n'
    bat_content += 'pause\n'

    bat_path = Path(__file__).parent / 'test_multi_seed.bat'
    with open(bat_path, 'w', encoding='utf-8') as f:
        f.write(bat_content)

    print(f"已生成: {bat_path}")
    print("\n运行方式:")
    print("  test_multi_seed.bat")
    print("\n注意: 将同时启动 15 个窗口，请确保系统资源充足")

    # 也生成分批运行的版本（每次只运行一个seed的3个算法）
    for seed in SEEDS:
        bat_content = '@echo off\nchcp 65001 >nul\n'
        bat_content += f'echo Seed {seed} Experiment\n'
        bat_content += 'echo.\n\n'

        for algo in ALGORITHMS:
            window_title = f"{algo.upper()}_Seed{seed}"
            cmd = f'start "{window_title}" cmd /k "conda activate drl && python test_flow_scale.py -a {algo} --seed {seed}"'
            bat_content += f'{cmd}\n'
            bat_content += 'timeout /t 2 /nobreak >nul\n'

        bat_content += '\npause\n'

        bat_path = Path(__file__).parent / f'test_seed_{seed}.bat'
        with open(bat_path, 'w', encoding='utf-8') as f:
            f.write(bat_content)
        print(f"已生成: test_seed_{seed}.bat")

    print("\n分批运行方式:")
    print("  test_seed_42.bat   (seed=42)")
    print("  test_seed_123.bat  (seed=123)")
    print("  test_seed_456.bat  (seed=456)")
    print("  test_seed_789.bat  (seed=789)")
    print("  test_seed_1024.bat (seed=1024)")


if __name__ == '__main__':
    main()
