"""
测试 uxsimpp_extended 的多进程兼容性

测试内容：
1. 多个进程能否独立创建 World 对象
2. 多个进程能否独立运行仿真
3. 进程间是否存在状态污染
"""

import multiprocessing as mp
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'env'))

from uxsimpp_extended.uxsimpp import World, Vehicle


def create_simple_world(worker_id: int, seed: int) -> World:
    """创建一个简单的测试World"""
    W = World(
        name=f"test_world_{worker_id}",
        deltan=5,
        tmax=3600,
        random_seed=seed,
        print_mode=0,
        save_mode=0,
        show_mode=0,
    )

    # 创建简单的三角形网络
    W.addNode("A", 0, 0)
    W.addNode("B", 1000, 0)
    W.addNode("C", 500, 866)

    W.addLink("AB", "A", "B", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("BC", "B", "C", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("AC", "A", "C", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("CA", "C", "A", length=1000, free_flow_speed=20, jam_density=0.2)

    # 添加一些车辆 (使用位置参数: W, orig, dest, departure_time)
    for i in range(10):
        Vehicle(W, "A", "C", i * 100, name=f"veh_{worker_id}_{i}")

    return W


def worker_task(args):
    """Worker进程的任务"""
    worker_id, seed = args
    results = {}

    try:
        # 测试1: 创建World
        start_time = time.time()
        W = create_simple_world(worker_id, seed)
        results['create_time'] = time.time() - start_time
        results['vehicle_count'] = len(W.VEHICLES)

        # 测试2: 运行仿真
        start_time = time.time()
        W.exec_simulation()
        results['sim_time'] = time.time() - start_time

        # 测试3: 检查结果
        completed = sum(1 for v in W.VEHICLES.values() if v.state == "end")
        results['completed_vehicles'] = completed

        # 测试4: 释放资源
        W.release()
        del W

        results['success'] = True
        results['error'] = None

    except Exception as e:
        results['success'] = False
        results['error'] = str(e)

    return worker_id, results


def test_sequential():
    """顺序执行测试（基准）"""
    print("=" * 60)
    print("测试1: 顺序执行（基准）")
    print("=" * 60)

    start_time = time.time()
    results = []
    for i in range(4):
        _, result = worker_task((i, 42 + i))
        results.append(result)
    total_time = time.time() - start_time

    print(f"总耗时: {total_time:.2f}s")
    for i, r in enumerate(results):
        if r['success']:
            print(f"  Worker {i}: 创建={r['create_time']:.3f}s, 仿真={r['sim_time']:.3f}s, 完成车辆={r['completed_vehicles']}")
        else:
            print(f"  Worker {i}: 失败 - {r['error']}")

    return all(r['success'] for r in results)


def test_multiprocess():
    """多进程并行测试"""
    print("\n" + "=" * 60)
    print("测试2: 多进程并行")
    print("=" * 60)

    num_workers = 4

    start_time = time.time()
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_task, [(i, 42 + i) for i in range(num_workers)])
    total_time = time.time() - start_time

    print(f"总耗时: {total_time:.2f}s (使用 {num_workers} 个进程)")
    for worker_id, r in results:
        if r['success']:
            print(f"  Worker {worker_id}: 创建={r['create_time']:.3f}s, 仿真={r['sim_time']:.3f}s, 完成车辆={r['completed_vehicles']}")
        else:
            print(f"  Worker {worker_id}: 失败 - {r['error']}")

    return all(r['success'] for _, r in results)


def test_repeated_multiprocess():
    """重复多进程测试（检查状态污染）"""
    print("\n" + "=" * 60)
    print("测试3: 重复多进程（检查状态污染）")
    print("=" * 60)

    num_workers = 4
    num_rounds = 3

    all_success = True
    for round_idx in range(num_rounds):
        start_time = time.time()
        with mp.Pool(num_workers) as pool:
            results = pool.map(worker_task, [(i, 42 + i + round_idx * 100) for i in range(num_workers)])
        total_time = time.time() - start_time

        round_success = all(r['success'] for _, r in results)
        all_success = all_success and round_success

        print(f"  第 {round_idx + 1} 轮: 耗时={total_time:.2f}s, 成功={round_success}")

        if not round_success:
            for worker_id, r in results:
                if not r['success']:
                    print(f"    Worker {worker_id} 失败: {r['error']}")

    return all_success


def test_worker_reuse():
    """测试Worker复用（模拟实际使用场景）"""
    print("\n" + "=" * 60)
    print("测试4: Worker复用（多次评估）")
    print("=" * 60)

    num_workers = 2
    num_tasks = 8

    start_time = time.time()
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_task, [(i % num_workers, 42 + i) for i in range(num_tasks)])
    total_time = time.time() - start_time

    success_count = sum(1 for _, r in results if r['success'])
    print(f"总耗时: {total_time:.2f}s")
    print(f"成功: {success_count}/{num_tasks}")

    return success_count == num_tasks


# ========== 测试5的模块级函数（Windows多进程需要在模块级别定义） ==========

# Worker进程的全局变量
_worker_world = None
_init_seed_base = None


def _init_worker_for_test5(seed_base):
    """Worker初始化函数（模块级别，用于pickle序列化）"""
    global _worker_world, _init_seed_base
    _init_seed_base = seed_base
    worker_id = mp.current_process().name
    # 每个Worker创建自己的World模板
    _worker_world = create_simple_world(hash(worker_id) % 1000, seed_base)
    print(f"  [Init] {worker_id} 初始化完成, 车辆数={len(_worker_world.VEHICLES)}")


def _evaluate_task_for_test5(task_id):
    """使用已初始化的Worker执行任务（模块级别）"""
    global _worker_world

    try:
        # 这里模拟评估：创建新World运行仿真
        # 实际实现中会基于_worker_world的网络数据创建新实例
        W = create_simple_world(task_id, 42 + task_id)
        W.exec_simulation()
        completed = sum(1 for v in W.VEHICLES.values() if v.state == "end")
        W.release()
        return task_id, True, completed, None
    except Exception as e:
        import traceback
        return task_id, False, 0, f"{str(e)}\n{traceback.format_exc()}"


def test_with_initializer():
    """测试使用initializer的Worker池（目标架构）"""
    print("\n" + "=" * 60)
    print("测试5: 使用initializer的Worker池")
    print("=" * 60)

    num_workers = 2
    num_tasks = 6

    start_time = time.time()
    with mp.Pool(num_workers, initializer=_init_worker_for_test5, initargs=(42,)) as pool:
        results = pool.map(_evaluate_task_for_test5, range(num_tasks))
    total_time = time.time() - start_time

    print(f"总耗时: {total_time:.2f}s")
    success_count = sum(1 for r in results if r[1])
    print(f"成功: {success_count}/{num_tasks}")

    for task_id, success, completed, error in results:
        if success:
            print(f"  Task {task_id}: 完成车辆={completed}")
        else:
            print(f"  Task {task_id}: 失败 - {error}")

    return success_count == num_tasks


if __name__ == "__main__":
    # Windows下需要这个保护
    mp.freeze_support()

    print("uxsimpp_extended 多进程兼容性测试")
    print("=" * 60)

    test_results = {}

    # 运行测试
    test_results['sequential'] = test_sequential()
    test_results['multiprocess'] = test_multiprocess()
    test_results['repeated'] = test_repeated_multiprocess()
    test_results['reuse'] = test_worker_reuse()
    test_results['initializer'] = test_with_initializer()

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！uxsimpp_extended 支持多进程并行。")
    else:
        print("部分测试失败，需要进一步调查。")
    print("=" * 60)
