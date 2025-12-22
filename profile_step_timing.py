"""
Step执行时间分析脚本

统计一个博弈Step中各部分的执行时间和资源使用情况。
"""

import time
import sys
import os
import numpy as np
import torch
import psutil
from functools import wraps
from collections import defaultdict

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from src.algorithms.maddpg.maddpg import MADDPG


class TimingProfiler:
    """计时分析器"""

    def __init__(self):
        self.timings = defaultdict(list)
        self.current_context = []

    def start(self, name):
        """开始计时"""
        self.current_context.append((name, time.perf_counter()))

    def stop(self):
        """停止计时并记录"""
        if self.current_context:
            name, start_time = self.current_context.pop()
            elapsed = time.perf_counter() - start_time
            self.timings[name].append(elapsed)
            return elapsed
        return 0

    def summary(self):
        """生成统计摘要"""
        print("\n" + "="*70)
        print("Step执行时间分析报告")
        print("="*70)

        # 按平均时间排序
        sorted_items = sorted(
            self.timings.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )

        total_time = sum(np.mean(v) for v in self.timings.values())

        print(f"\n{'组件':<40} {'平均(ms)':<12} {'占比':<10} {'次数':<8}")
        print("-"*70)

        for name, times in sorted_items:
            avg_ms = np.mean(times) * 1000
            pct = (np.mean(times) / total_time * 100) if total_time > 0 else 0
            count = len(times)
            print(f"{name:<40} {avg_ms:<12.2f} {pct:<10.1f}% {count:<8}")

        print("-"*70)
        print(f"{'总计':<40} {total_time*1000:<12.2f}")
        print("="*70)

        return self.timings


def get_memory_usage():
    """获取当前内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def profile_step(env, maddpg, observations, profiler):
    """分析单个step的各部分耗时"""

    # 1. MADDPG决策
    profiler.start("1. MADDPG.take_action")
    actions = maddpg.take_action(observations, add_noise=True)
    profiler.stop()

    # 2. 环境step（内部会调用UE-DTA）
    profiler.start("2. env.step (总计)")
    next_observations, rewards, terminations, truncations, infos = env.step(actions)
    profiler.stop()

    # 3. 存储经验
    profiler.start("3. MADDPG.store_experience")
    maddpg.store_experience(observations, actions, rewards, next_observations, terminations)
    profiler.stop()

    # 4. 学习更新
    profiler.start("4. MADDPG.learn")
    learned = maddpg.learn()  # 返回是否真正执行了学习
    profiler.stop()

    return next_observations, rewards, terminations, truncations, infos, learned


def profile_env_step_detailed(env, actions, profiler):
    """详细分析env.step内部各阶段耗时"""

    # 1. 更新价格
    profiler.start("2.1 更新价格")
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)
    profiler.stop()

    # 2. 运行仿真（最耗时的部分）
    profiler.start("2.2 UE-DTA仿真")
    charging_flows, ue_info = env._EVCSChargingGameEnv__run_simulation()
    profiler.stop()

    # 3. 计算奖励
    profiler.start("2.3 计算奖励")
    rewards = env._EVCSChargingGameEnv__calculate_rewards(charging_flows)
    profiler.stop()

    # 4. 更新历史
    profiler.start("2.4 更新历史/状态")
    env.charging_flow_history.append(charging_flows)
    env.current_step += 1
    profiler.stop()

    # 5. 计算收敛
    profiler.start("2.5 收敛判断")
    relative_change_rate = env._EVCSChargingGameEnv__calculate_relative_change_rate()
    profiler.stop()

    # 6. 生成观测
    profiler.start("2.6 生成观测")
    observations = env._EVCSChargingGameEnv__get_observations()
    profiler.stop()

    return rewards, ue_info, observations


def main():
    print("="*70)
    print("Step执行时间分析")
    print("="*70)

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"计算设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存: {initial_memory:.1f} MB")

    # 创建环境
    print("\n正在创建环境...")
    env_start = time.perf_counter()
    env = EVCSChargingGameEnv(
        network_dir="data/siouxfalls",
        network_name="siouxfalls",
        random_seed=42,
        max_steps=50,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env_time = time.perf_counter() - env_start
    print(f"环境创建耗时: {env_time*1000:.1f} ms")

    after_env_memory = get_memory_usage()
    print(f"环境创建后内存: {after_env_memory:.1f} MB (+{after_env_memory-initial_memory:.1f} MB)")

    # 获取维度信息
    obs_space = env.observation_space(env.agents[0])
    obs_dim = sum(np.prod(space.shape) for space in obs_space.spaces.values())
    action_dim = env.action_space(env.agents[0]).shape[0]
    global_obs_dim = env.global_state_space().shape[0]

    print(f"\n环境维度: obs={obs_dim}, action={action_dim}, global_obs={global_obs_dim}")
    print(f"智能体数量: {env.n_agents}, 时段数: {env.n_periods}")

    # 创建MADDPG
    print("\n正在创建MADDPG...")
    maddpg_start = time.perf_counter()
    maddpg = MADDPG(
        agent_ids=env.agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_obs_dim=global_obs_dim,
        buffer_capacity=10000,
        max_batch_size=64,
        actor_lr=0.001,
        critic_lr=0.001,
        gamma=0.99,
        tau=0.01,
        seed=42,
        device=device,
        actor_hidden_sizes=(64, 64),
        critic_hidden_sizes=(128, 64)
    )
    maddpg_time = time.perf_counter() - maddpg_start
    print(f"MADDPG创建耗时: {maddpg_time*1000:.1f} ms")

    after_maddpg_memory = get_memory_usage()
    print(f"MADDPG创建后内存: {after_maddpg_memory:.1f} MB (+{after_maddpg_memory-after_env_memory:.1f} MB)")

    # 重置环境
    observations, _ = env.reset()

    # 创建分析器
    profiler = TimingProfiler()

    # 运行多个step进行分析
    # 注意：MADDPG需要至少8个经验才开始learn，所以需要足够多的step
    num_steps = 12
    print(f"\n开始分析 {num_steps} 个step (前8步积累经验，之后开始learn)...")
    print("-"*70)

    for step in range(num_steps):
        step_start = time.perf_counter()
        mem_before = get_memory_usage()

        # 运行并分析一个step
        next_obs, rewards, terms, truncs, infos, learned = profile_step(
            env, maddpg, observations, profiler
        )

        step_time = time.perf_counter() - step_start
        mem_after = get_memory_usage()

        # 打印当前step信息
        ue_iters = infos.get('ue_iterations', 0)
        learn_status = "学习中" if learned else "积累经验"
        print(f"Step {step+1:2d}: {step_time*1000:7.1f}ms | UE迭代: {ue_iters:2d} | "
              f"内存: {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB) | {learn_status}")

        observations = next_obs

        if all(terms.values()):
            print("环境收敛，提前结束")
            break

    # 输出统计摘要
    profiler.summary()

    # 内存使用分析
    final_memory = get_memory_usage()
    print(f"\n内存使用分析:")
    print(f"  初始: {initial_memory:.1f} MB")
    print(f"  环境创建后: {after_env_memory:.1f} MB")
    print(f"  MADDPG创建后: {after_maddpg_memory:.1f} MB")
    print(f"  最终: {final_memory:.1f} MB")
    print(f"  总增长: {final_memory-initial_memory:.1f} MB")

    # 资源使用建议
    print("\n" + "="*70)
    print("分析结论")
    print("="*70)

    avg_step_time = np.mean(profiler.timings.get("2. env.step (总计)", [0])) * 1000
    avg_take_action = np.mean(profiler.timings.get("1. MADDPG.take_action", [0])) * 1000
    avg_learn = np.mean(profiler.timings.get("4. MADDPG.learn", [0])) * 1000

    print(f"\n主要耗时组件:")
    print(f"  - env.step (UE-DTA仿真): {avg_step_time:.1f}ms")
    print(f"  - MADDPG.take_action: {avg_take_action:.1f}ms")
    print(f"  - MADDPG.learn: {avg_learn:.1f}ms")

    if avg_step_time > avg_learn + avg_take_action:
        print(f"\n瓶颈分析: UE-DTA仿真占主导 ({avg_step_time/(avg_step_time+avg_take_action+avg_learn)*100:.1f}%)")
        print("  → CPU瓶颈: UXSim是纯Python仿真，单线程执行")
        print("  → 内存占用: 每次UE迭代创建新World实例，大量Vehicle对象")
        print("  → GPU空闲: 神经网络计算量小，GPU利用率低")
    else:
        print("\n瓶颈分析: 神经网络学习占主导")

    print("\n优化建议:")
    print("  1. 减少UE迭代次数 (调整收敛阈值)")
    print("  2. 增大deltan减少Vehicle对象数")
    print("  3. 考虑多进程并行运行多个实验")


if __name__ == "__main__":
    main()
