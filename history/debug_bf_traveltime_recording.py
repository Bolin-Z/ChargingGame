"""
诊断脚本4：检查 C++ 层 traveltime_real 是否被正确更新

检查 record_travel_time 是否被调用，以及 traveltime_t/traveltime_tt 是否有数据

执行命令: python history/debug_bf_traveltime_recording.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def check_traveltime_recording():
    """检查链路的旅行时间记录"""
    print("\n" + "=" * 60)
    print("检查链路 traveltime_t 和 traveltime_tt 记录")
    print("=" * 60)

    env = EVCSChargingGameEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=5,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()
    W = env.W

    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    print(f"\n运行仿真...")
    W.exec_simulation()

    print(f"\n检查链路的旅行时间记录（前10条）:")
    for i, link in enumerate(W.LINKS[:10]):
        print(f"\n链路 {link.name}:")
        print(f"  length: {link.length:.1f}m")

        # 检查 traveltime_t 和 traveltime_tt
        tt_t = list(link.traveltime_t) if hasattr(link, 'traveltime_t') else []
        tt_tt = list(link.traveltime_tt) if hasattr(link, 'traveltime_tt') else []

        print(f"  traveltime_t 记录数: {len(tt_t)}")
        print(f"  traveltime_tt 记录数: {len(tt_tt)}")

        if tt_t:
            print(f"    前5个 traveltime_t: {tt_t[:5]}")
        if tt_tt:
            print(f"    前5个 traveltime_tt (实际旅行时间): {tt_tt[:5]}")
            print(f"    min/max traveltime_tt: {min(tt_tt):.1f}s / {max(tt_tt):.1f}s")

        # 对比 traveltime_real
        tr = list(link.traveltime_real)
        unique_values = set(tr)
        print(f"  traveltime_real unique values: {len(unique_values)}")
        if len(unique_values) <= 5:
            print(f"    values: {sorted(unique_values)}")
        else:
            print(f"    sample: {sorted(list(unique_values))[:5]}...")

        # 计算自由流时间
        free_flow_tt = link.length / link.vmax if hasattr(link, 'vmax') else "N/A"
        print(f"  free_flow_time: {free_flow_tt:.1f}s" if isinstance(free_flow_tt, float) else f"  free_flow_time: {free_flow_tt}")

    env.close()


def check_vehicle_link_transitions():
    """检查车辆的链路转移情况"""
    print("\n" + "=" * 60)
    print("检查车辆链路转移（log_t_link）")
    print("=" * 60)

    env = EVCSChargingGameEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=5,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()
    W = env.W

    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    print(f"\n运行仿真...")
    W.exec_simulation()

    print(f"\n检查完成车辆的 log_t_link（前5辆）:")
    checked = 0
    for veh in W.VEHICLES.values():
        if veh.state == "end" and checked < 5:
            print(f"\n车辆 {veh.name}:")
            print(f"  departure_time: {veh.departure_time}")
            print(f"  travel_time: {veh.travel_time}")

            log = veh.log_t_link
            print(f"  log_t_link entries: {len(log)}")
            for j, (t, link) in enumerate(log):
                link_name = link.name if link else "None"
                if j < len(log) - 1:
                    next_t = log[j + 1][0]
                    duration = (next_t - t) * W.delta_t
                    print(f"    [{j}] t={t} ({t * W.delta_t:.0f}s), link={link_name}, duration={duration:.0f}s")
                else:
                    print(f"    [{j}] t={t} ({t * W.delta_t:.0f}s), link={link_name}")

            checked += 1

    env.close()


def check_arrival_time_link():
    """检查 arrival_time_link 的值"""
    print("\n" + "=" * 60)
    print("检查车辆 arrival_time_link 属性")
    print("=" * 60)

    env = EVCSChargingGameEnv(
        network_dir="data/berlin_friedrichshain",
        network_name="berlin_friedrichshain",
        random_seed=42,
        max_steps=5,
        convergence_threshold=0.01,
        stable_steps_required=3
    )
    env.reset()
    W = env.W

    actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
    env._EVCSChargingGameEnv__update_prices_from_actions(actions)

    print(f"\n运行仿真...")
    W.exec_simulation()

    print(f"\n检查完成车辆的 arrival_time_link（前5辆）:")
    checked = 0
    for veh in W.VEHICLES.values():
        if veh.state == "end" and checked < 5:
            print(f"\n车辆 {veh.name}:")
            if hasattr(veh, 'arrival_time_link'):
                print(f"  arrival_time_link: {veh.arrival_time_link}")
            else:
                print(f"  arrival_time_link: 属性不存在")
            checked += 1

    env.close()


def compare_with_siouxfalls():
    """对比 Sioux Falls 网络的情况"""
    print("\n" + "=" * 60)
    print("对比 Sioux Falls 网络的 traveltime_real")
    print("=" * 60)

    try:
        env = EVCSChargingGameEnv(
            network_dir="data/siouxfalls",
            network_name="siouxfalls",
            random_seed=42,
            max_steps=5,
            convergence_threshold=0.01,
            stable_steps_required=3
        )
        env.reset()
        W = env.W

        actions = {agent: np.full(env.n_periods, 0.5, dtype=np.float32) for agent in env.agents}
        env._EVCSChargingGameEnv__update_prices_from_actions(actions)

        print(f"\n运行仿真...")
        W.exec_simulation()

        print(f"\n检查 Sioux Falls 链路的 traveltime_real（前5条）:")
        for i, link in enumerate(W.LINKS[:5]):
            print(f"\n链路 {link.name}:")
            print(f"  length: {link.length:.1f}m")

            tr = list(link.traveltime_real)
            unique_values = set(tr)
            print(f"  traveltime_real unique values: {len(unique_values)}")

            # 检查 traveltime_tt
            tt_tt = list(link.traveltime_tt) if hasattr(link, 'traveltime_tt') else []
            print(f"  traveltime_tt 记录数: {len(tt_tt)}")
            if tt_tt:
                print(f"    min/max: {min(tt_tt):.1f}s / {max(tt_tt):.1f}s")

        env.close()
    except Exception as e:
        print(f"Sioux Falls 测试失败: {e}")


def main():
    print("\n" + "#" * 60)
    print("# 检查 traveltime_real 记录机制")
    print("#" * 60)

    check_traveltime_recording()
    check_vehicle_link_transitions()
    check_arrival_time_link()
    compare_with_siouxfalls()

    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    print("""
检查要点:
1. traveltime_tt 是否有记录 - 如果没有，说明 record_travel_time 没被调用
2. traveltime_real 是否有多个不同的值 - 如果只有一个值，说明没有被更新
3. Sioux Falls 的情况作为对照 - 如果 SF 正常而 BF 异常，可能是网络差异

如果 traveltime_tt 有记录但 traveltime_real 没更新，检查:
- arrival_time_link 初始化是否正确
- record_travel_time 中的时步计算是否正确
""")


if __name__ == "__main__":
    main()
