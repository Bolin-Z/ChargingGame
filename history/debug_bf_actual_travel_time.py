"""
诊断脚本3：检查 actual_travel_time 估计准确性

问题：actual_travel_time 估计的旅行时间比实际偏小，
导致 relative_gap 始终较大。

执行命令: python history/debug_bf_actual_travel_time.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv


def analyze_actual_travel_time_accuracy():
    """分析 actual_travel_time 估计准确性"""
    print("\n" + "=" * 60)
    print("分析 actual_travel_time 估计准确性")
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

    print(f"\n基础参数:")
    print(f"  delta_t: {W.delta_t}")
    print(f"  t_max: {W.t_max}")

    # 收集每辆完成车辆的数据
    errors = []
    data_samples = []

    for veh in W.VEHICLES.values():
        if veh.state != "end":
            continue

        route, timestamps = veh.traveled_route(include_departure_time=True)
        actual_travel_time = timestamps[-1] - timestamps[0]

        # 使用 route.actual_travel_time 估计
        route_obj = W.defRoute([l.name for l in route.links])
        estimated_travel_time = route_obj.actual_travel_time(veh.departure_time_in_second)

        error = estimated_travel_time - actual_travel_time
        relative_error = error / actual_travel_time if actual_travel_time > 0 else 0

        errors.append(relative_error)
        data_samples.append({
            'name': veh.name,
            'departure_time': veh.departure_time,
            'actual': actual_travel_time,
            'estimated': estimated_travel_time,
            'error': error,
            'relative_error': relative_error,
            'num_links': len(route.links)
        })

    # 统计
    errors = np.array(errors)
    print(f"\n估计误差统计（{len(errors)} 辆完成车辆）:")
    print(f"  mean: {np.mean(errors):.2%} (负值表示低估)")
    print(f"  std:  {np.std(errors):.2%}")
    print(f"  min:  {np.min(errors):.2%}")
    print(f"  max:  {np.max(errors):.2%}")
    print(f"  p10:  {np.percentile(errors, 10):.2%}")
    print(f"  p50:  {np.percentile(errors, 50):.2%}")
    print(f"  p90:  {np.percentile(errors, 90):.2%}")

    # 打印一些样本
    print(f"\n误差最大的10辆车:")
    sorted_samples = sorted(data_samples, key=lambda x: abs(x['error']), reverse=True)
    for s in sorted_samples[:10]:
        print(f"  {s['name'][:30]}...")
        print(f"    departure: {s['departure_time']:.0f}s, links: {s['num_links']}")
        print(f"    actual: {s['actual']:.1f}s, estimated: {s['estimated']:.1f}s")
        print(f"    error: {s['error']:.1f}s ({s['relative_error']:.2%})")

    env.close()
    return data_samples


def analyze_link_level_estimation():
    """分析单链路级别的估计准确性"""
    print("\n" + "=" * 60)
    print("分析单链路级别的估计准确性")
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

    # 收集链路级别的估计误差
    link_errors = {}  # link_name -> list of errors

    for veh in W.VEHICLES.values():
        if veh.state != "end":
            continue

        route, timestamps = veh.traveled_route(include_departure_time=True)

        # 计算每条链路的实际旅行时间
        for i, link in enumerate(route.links):
            if i + 1 < len(timestamps) - 1:  # 不包括最后的到达时间
                entry_time = timestamps[i + 1]  # 进入这条链路的时间（跳过 departure）
                if i + 2 < len(timestamps):
                    exit_time = timestamps[i + 2]
                else:
                    exit_time = timestamps[-1]
                actual_link_tt = exit_time - entry_time

                # 估计的链路旅行时间
                estimated_link_tt = link.actual_travel_time(entry_time)

                error = estimated_link_tt - actual_link_tt

                if link.name not in link_errors:
                    link_errors[link.name] = []
                link_errors[link.name].append({
                    'actual': actual_link_tt,
                    'estimated': estimated_link_tt,
                    'error': error,
                    'entry_time': entry_time
                })

    # 统计每条链路的误差
    print(f"\n链路估计误差统计（误差最大的10条链路）:")
    link_stats = []
    for link_name, samples in link_errors.items():
        if len(samples) < 5:  # 至少5个样本
            continue
        mean_error = np.mean([s['error'] for s in samples])
        link_stats.append({
            'name': link_name,
            'samples': len(samples),
            'mean_error': mean_error,
            'mean_actual': np.mean([s['actual'] for s in samples]),
            'mean_estimated': np.mean([s['estimated'] for s in samples])
        })

    link_stats.sort(key=lambda x: abs(x['mean_error']), reverse=True)
    for ls in link_stats[:10]:
        print(f"\n  链路 {ls['name']}:")
        print(f"    样本数: {ls['samples']}")
        print(f"    mean actual: {ls['mean_actual']:.1f}s")
        print(f"    mean estimated: {ls['mean_estimated']:.1f}s")
        print(f"    mean error: {ls['mean_error']:.1f}s")

    env.close()


def check_traveltime_real_data():
    """检查 traveltime_real 数据的正确性"""
    print("\n" + "=" * 60)
    print("检查 traveltime_real 数据")
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

    # 检查几条链路的 traveltime_real
    print(f"\n链路 traveltime_real 数据检查（前5条）:")
    for i, link in enumerate(W.LINKS[:5]):
        print(f"\n  链路 {link.name}:")
        print(f"    length: {link.length:.1f}m")

        # 检查是否有缓存
        if hasattr(link, '_traveltime_cache') and link._traveltime_cache is not None:
            cache = link._traveltime_cache
            print(f"    使用缓存: 是")
            print(f"    缓存大小: {len(cache)}")
            print(f"    缓存 delta_t: {link._traveltime_cache_delta_t}")
            print(f"    前5个值: {cache[:5]}")
            print(f"    最后5个值: {cache[-5:]}")
        else:
            tt_real = link.traveltime_real
            print(f"    使用缓存: 否")
            print(f"    traveltime_real 大小: {len(tt_real)}")
            print(f"    前5个值: {tt_real[:5]}")
            print(f"    最后5个值: {tt_real[-5:]}")

        # 对比 actual_travel_time 在不同时刻的值
        print(f"    actual_travel_time 采样:")
        for t in [0, 1000, 3000, 5000, 8000]:
            att = link.actual_travel_time(t)
            print(f"      t={t}s: {att:.1f}s")

    env.close()


def compare_with_uxsim_behavior():
    """检查是否有 UXsim 行为差异的线索"""
    print("\n" + "=" * 60)
    print("检查 uxsimpp vs UXsim 可能的差异")
    print("=" * 60)

    print("""
需要检查的关键点：
1. traveltime_real 的更新逻辑是否相同
2. actual_travel_time 的缓存机制是否正确
3. 时间索引计算是否一致

建议：
- 运行 history/debug_log_t_link.py 对比 log_t_link 语义
- 检查 uxsimpp C++ 代码中 traveltime_real 的更新时机
""")

    # 检查 uxsimpp_extended 中的相关代码
    uxsimpp_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "env", "uxsimpp_extended", "uxsimpp.py"
    )

    print(f"\n检查 uxsimpp.py 中 actual_travel_time 的实现:")
    with open(uxsimpp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_function = False
    for i, line in enumerate(lines, 1):
        if 'def actual_travel_time' in line:
            in_function = True
        if in_function:
            print(f"  {i}: {line.rstrip()}")
            if line.strip().startswith('Link.actual_travel_time'):
                break


def main():
    print("\n" + "#" * 60)
    print("# actual_travel_time 估计准确性分析")
    print("#" * 60)

    analyze_actual_travel_time_accuracy()
    check_traveltime_real_data()
    analyze_link_level_estimation()
    compare_with_uxsim_behavior()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
如果 actual_travel_time 系统性低估实际旅行时间，
可能的原因：
1. traveltime_real 数据没有及时更新拥堵信息
2. 时间索引计算有偏差（秒 vs 时步）
3. 缓存机制导致使用了过期数据

下一步：检查 uxsimpp C++ 层的 traveltime_real 更新逻辑
""")


if __name__ == "__main__":
    main()
