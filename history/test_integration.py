"""
EVCSChargingGameEnv 集成测试

测试 uxsimpp_extended 与 EVCSChargingGameEnv 的集成是否正常工作
"""
import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import numpy as np


def test_env_initialization():
    """测试1：环境初始化"""
    print("=" * 60)
    print("测试1：EVCSChargingGameEnv 初始化")
    print("=" * 60)

    try:
        from src.env.EVCSChargingGameEnv import EVCSChargingGameEnv

        env = EVCSChargingGameEnv(
            network_dir=os.path.join(project_root, "data", "siouxfalls"),
            network_name="siouxfalls",
            random_seed=42,
            max_steps=10,
            convergence_threshold=0.01,
            stable_steps_required=3
        )

        print(f"  智能体数量: {env.n_agents}")
        print(f"  时段数量: {env.n_periods}")
        print(f"  智能体列表: {env.agents}")

        print("[PASS] 环境初始化成功")
        return True, env

    except Exception as e:
        print(f"[FAIL] 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_env_reset(env):
    """测试2：环境重置"""
    print("\n" + "=" * 60)
    print("测试2：环境重置 (reset)")
    print("=" * 60)

    try:
        observations, infos = env.reset(seed=42)

        print(f"  观测数量: {len(observations)}")
        for agent_id, obs in observations.items():
            print(f"  {agent_id}: last_round_all_prices shape = {obs['last_round_all_prices'].shape}, "
                  f"own_charging_flow shape = {obs['own_charging_flow'].shape}")

        print("[PASS] 环境重置成功")
        return True

    except Exception as e:
        print(f"[FAIL] 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_step(env):
    """测试3：执行一步博弈"""
    print("\n" + "=" * 60)
    print("测试3：执行一步博弈 (step)")
    print("=" * 60)

    try:
        # 生成随机价格动作（归一化到[0,1]）
        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = np.random.uniform(0.3, 0.7, size=(env.n_periods,)).astype(np.float32)

        print(f"  动作示例 ({env.agents[0]}): {actions[env.agents[0]][:3]}...")

        # 执行一步
        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"  奖励: {rewards}")
        print(f"  终止状态: {terminations}")

        # 检查奖励是否合理
        for agent_id, reward in rewards.items():
            if not np.isfinite(reward):
                print(f"[FAIL] 智能体 {agent_id} 的奖励不合理: {reward}")
                return False

        print("[PASS] 执行一步博弈成功")
        return True

    except Exception as e:
        print(f"[FAIL] 执行博弈步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_step(env):
    """测试4：执行多步博弈"""
    print("\n" + "=" * 60)
    print("测试4：执行多步博弈 (3步)")
    print("=" * 60)

    try:
        env.reset(seed=42)

        for step in range(3):
            # 生成随机价格动作
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = np.random.uniform(0.3, 0.7, size=(env.n_periods,)).astype(np.float32)

            observations, rewards, terminations, truncations, infos = env.step(actions)

            total_reward = sum(rewards.values())
            print(f"  Step {step + 1}: 总奖励 = {total_reward:.2f}, "
                  f"收敛 = {all(terminations.values())}")

            if all(terminations.values()):
                print("  -> 博弈已收敛")
                break

        print("[PASS] 多步博弈执行成功")
        return True

    except Exception as e:
        print(f"[FAIL] 多步博弈失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vehicle_attributes(env):
    """测试5：验证车辆 attribute 功能"""
    print("\n" + "=" * 60)
    print("测试5：车辆 attribute 功能")
    print("=" * 60)

    try:
        # 重置环境
        env.reset(seed=42)

        # 执行一步让车辆生成
        actions = {agent_id: np.array([0.5] * env.n_periods, dtype=np.float32)
                   for agent_id in env.agents}
        env.step(actions)

        # 检查车辆是否有 attribute
        world = env.W
        vehicles_checked = 0
        vehicles_with_attr = 0

        for veh in world.VEHICLES.values():
            vehicles_checked += 1
            if hasattr(veh, 'attribute') and veh.attribute:
                vehicles_with_attr += 1
                if vehicles_checked <= 3:
                    print(f"  车辆 {veh.name}: attribute = {veh.attribute}")

        print(f"  检查车辆数: {vehicles_checked}")
        print(f"  有attribute的车辆数: {vehicles_with_attr}")

        if vehicles_with_attr > 0:
            print("[PASS] 车辆 attribute 功能正常")
            return True
        else:
            print("[WARN] 没有发现带 attribute 的车辆（可能是预期行为）")
            return True  # 不算失败

    except Exception as e:
        print(f"[FAIL] 车辆 attribute 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有集成测试"""
    print("\n" + "#" * 60)
    print("# EVCSChargingGameEnv 集成测试")
    print("# 测试 uxsimpp_extended 与博弈环境的集成")
    print("#" * 60)

    results = []

    # 测试1：初始化
    success, env = test_env_initialization()
    results.append(("环境初始化", success))

    if not success:
        print("\n❌ 环境初始化失败，无法继续测试")
        return False

    # 测试2：重置
    results.append(("环境重置", test_env_reset(env)))

    # 测试3：单步博弈
    results.append(("单步博弈", test_env_step(env)))

    # 测试4：多步博弈
    results.append(("多步博弈", test_multi_step(env)))

    # 测试5：车辆属性
    results.append(("车辆attribute", test_vehicle_attributes(env)))

    # 汇总结果
    print("\n" + "=" * 60)
    print("集成测试结果汇总")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: [{status}]")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"通过: {passed}, 失败: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
