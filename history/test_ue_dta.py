"""
UE-DTA 功能测试脚本

测试内容：使用SF数据集，价格设为区间中点，运行UE-DTA求解
输出格式：迭代编号 | 完成率 | GM | 切换次数 | 耗时
"""

import time
import numpy as np
from collections import defaultdict

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv
from src.env.patch import patch_uxsim
from uxsim import World, Vehicle


def test_ue_dta():
    """测试UE-DTA求解功能"""

    # 初始化环境
    print("初始化环境...")
    env = EVCSChargingGameEnv(
        network_dir="data/siouxfalls",
        network_name="siouxfalls",
        random_seed=42,
        max_steps=100,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    # 重置环境
    observations, infos = env.reset()

    # 设置价格为区间中点
    actions = {}
    for agent in env.agents:
        # 动作空间是[0,1]，中点就是0.5
        actions[agent] = np.full(env.n_periods, 0.5, dtype=np.float32)

    print(f"\n充电站数量: {env.n_agents}")
    print(f"时段数量: {env.n_periods}")
    print(f"价格设置: 区间中点 (归一化值=0.5)")

    # 显示实际价格
    prices_dict = env.actions_to_prices_dict(actions)
    print("\n各充电站实际价格:")
    for agent, prices in prices_dict.items():
        print(f"  站点 {agent}: {prices[0]:.2f} 元/kWh (所有时段相同)")

    # 运行UE-DTA（调用step会触发__run_simulation）
    print("\n" + "="*70)
    print("开始 UE-DTA 迭代...")
    print("="*70)

    # 执行step
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 输出结果
    print("\n" + "="*70)
    print("UE-DTA 求解完成")
    print("="*70)
    print(f"是否收敛: {'是' if infos['ue_converged'] else '否'}")
    print(f"迭代次数: {infos['ue_iterations']}")

    # 输出充电流量
    print("\n各充电站充电流量 (按时段):")
    if len(env.charging_flow_history) > 0:
        flows = env.charging_flow_history[-1]
        for agent_idx, agent in enumerate(env.agents):
            flow_per_period = flows[agent_idx]
            total_flow = np.sum(flow_per_period)
            print(f"  站点 {agent}: 总流量={total_flow:.0f}辆, 各时段={flow_per_period}")

    # 输出奖励（收益）
    print("\n各充电站收益:")
    for agent, reward in rewards.items():
        print(f"  站点 {agent}: {reward:.2f} 元")

    print(f"\n总收益: {sum(rewards.values()):.2f} 元")


def test_ue_dta_verbose():
    """详细输出版本的UE-DTA测试（自定义输出格式）"""

    # 应用patch
    patch_uxsim()

    # 初始化环境
    print("初始化环境...")
    env = EVCSChargingGameEnv(
        network_dir="data/siouxfalls",
        network_name="siouxfalls",
        random_seed=42,
        max_steps=100,
        convergence_threshold=0.01,
        stable_steps_required=3
    )

    # 重置环境
    observations, infos = env.reset()

    # 设置价格为区间中点
    actions = {}
    for agent in env.agents:
        actions[agent] = np.full(env.n_periods, 0.5, dtype=np.float32)

    # 更新价格历史（模拟step的第一步）
    new_prices = env.actions_to_prices_matrix(actions)
    env.price_history.append(new_prices)

    print(f"\n充电站数量: {env.n_agents}")
    print(f"时段数量: {env.n_periods}")
    print(f"UE-DTA 最大迭代次数: {env.ue_max_iterations}")
    print(f"收敛阈值: {env.ue_convergence_threshold*100:.1f}%")
    print(f"连续收敛轮数要求: {env.ue_convergence_stable_rounds}")

    # 显示实际价格
    prices_dict = env.actions_to_prices_dict(actions)
    print("\n各充电站实际价格:")
    for agent, prices in prices_dict.items():
        print(f"  站点 {agent}: {prices[0]:.2f} 元/kWh")

    print("\n" + "="*70)
    print("开始 UE-DTA 迭代...")
    print("="*70)

    # 手动执行UE-DTA迭代（复制__run_simulation的逻辑但自定义输出）
    _run_ue_dta_with_verbose_output(env)


def _run_ue_dta_with_verbose_output(env):
    """手动执行UE-DTA并按指定格式输出"""

    # 获取充电和非充电车辆的OD映射
    dict_od_to_charging_vehid = defaultdict(list)
    dict_od_to_uncharging_vehid = defaultdict(list)

    # 创建模板World获取车辆信息
    W_template = env._EVCSChargingGameEnv__create_simulation_world()

    for key, veh in W_template.VEHICLES.items():
        o = veh.orig.name
        d = veh.dest.name
        if veh.attribute["charging_car"]:
            dict_od_to_charging_vehid[(o, d)].append(key)
        else:
            dict_od_to_uncharging_vehid[(o, d)].append(key)

    # 初始化路径分配
    current_routes = env._EVCSChargingGameEnv__initialize_routes(
        dict_od_to_charging_vehid,
        dict_od_to_uncharging_vehid,
        use_greedy=True
    )

    ue_convergence_counter = 0

    for iteration in range(env.ue_max_iterations):
        iter_start_time = time.time()

        # 创建新的仿真实例
        W = env._EVCSChargingGameEnv__create_simulation_world()

        # 应用当前路径分配
        env._EVCSChargingGameEnv__apply_routes_to_vehicles(W, current_routes)

        # 执行仿真
        W.exec_simulation()

        # 计算成本差并执行路径切换
        stats, new_routes, charging_flows = env._EVCSChargingGameEnv__route_choice_update(
            W, dict_od_to_charging_vehid, dict_od_to_uncharging_vehid,
            current_routes, iteration
        )

        iter_elapsed = (time.time() - iter_start_time) * 1000  # 毫秒

        # 按指定格式输出
        completed_ratio = stats['completed_total_vehicles'] / stats['total_vehicles'] if stats['total_vehicles'] > 0 else 0
        gap_global = stats['all_relative_gap_global_mean']
        switches = stats['total_route_switches']

        print(f"  迭代 {iteration+1:3d} | 完成率: {completed_ratio*100:5.1f}% | GM: {gap_global*100:5.2f}% | 切换: {switches:4d} | 耗时: {iter_elapsed:.0f}ms")

        # 更新路径分配
        current_routes = new_routes

        # 收敛判断
        gap_converged = gap_global <= env.ue_convergence_threshold
        completion_ok = completed_ratio >= env.ue_min_completed_ratio

        if gap_converged and completion_ok:
            ue_convergence_counter += 1
            if ue_convergence_counter >= env.ue_convergence_stable_rounds:
                print(f"\n✓ UE-DTA 收敛！连续 {env.ue_convergence_stable_rounds} 轮满足条件")
                break
        else:
            ue_convergence_counter = 0
    else:
        print(f"\n✗ UE-DTA 未收敛，达到最大迭代次数 {env.ue_max_iterations}")

    # 输出最终统计
    print("\n" + "="*70)
    print("最终统计:")
    print("="*70)
    print(f"  完成率: {completed_ratio*100:.1f}%")
    print(f"  全局平均相对成本差 (GM): {gap_global*100:.2f}%")
    print(f"  充电车辆平均成本: {stats['charging_avg_cost']:.2f}")
    print(f"  非充电车辆平均成本: {stats['uncharging_avg_cost']:.2f}")

    # 输出充电流量
    print("\n各充电站充电流量:")
    for agent_idx, agent in enumerate(env.agents):
        total_flow = np.sum(charging_flows[agent_idx])
        print(f"  站点 {agent}: {total_flow:.0f} 辆")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        # 详细输出模式
        test_ue_dta_verbose()
    else:
        # 标准模式（使用环境的内置tqdm输出）
        test_ue_dta()
