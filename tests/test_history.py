# tests/test_history.py
"""
GameHistory 测试脚本

验证内容：
1. 初始化正确性
2. EMA 信念更新
3. 评估记录
4. 重置功能
5. 辅助查询方法
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basic_init():
    """测试基本初始化"""
    from src.game.history import GameHistory

    print("=" * 60)
    print("测试 1: 基本初始化")
    print("=" * 60)

    agent_names = ["station_5", "station_12", "station_17"]
    n_periods = 8

    history = GameHistory(agent_names, n_periods)

    assert history.agent_names == agent_names, "agent_names 不匹配"
    assert history.n_agents == 3, "n_agents 不正确"
    assert history.n_periods == 8, "n_periods 不正确"
    assert history.ema_lambda == 0.05, "默认 ema_lambda 不正确"
    assert history.total_evaluations == 0, "初始 total_evaluations 应为 0"

    print(f"  agent_names: {history.agent_names}")
    print(f"  n_agents: {history.n_agents}")
    print(f"  n_periods: {history.n_periods}")
    print(f"  ema_lambda: {history.ema_lambda}")

    print("\n✓ 测试 1 通过")


def test_beliefs_shape_and_initial_value():
    """测试信念矩阵形状和初始值"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 2: 信念矩阵形状和初始值")
    print("=" * 60)

    agent_names = ["A", "B"]
    n_periods = 4

    history = GameHistory(agent_names, n_periods, initial_belief=0.5)

    assert history.beliefs.shape == (2, 4), f"形状不正确: {history.beliefs.shape}"
    assert np.allclose(history.beliefs, 0.5), "初始值不是 0.5"
    assert history.beliefs.dtype == np.float32, f"dtype 不正确: {history.beliefs.dtype}"

    print(f"  beliefs.shape: {history.beliefs.shape}")
    print(f"  beliefs.dtype: {history.beliefs.dtype}")
    print(f"  beliefs:\n{history.beliefs}")

    print("\n✓ 测试 2 通过")


def test_agent_index_mapping():
    """测试 Agent 名称到索引的映射"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 3: Agent 索引映射")
    print("=" * 60)

    agent_names = ["station_5", "station_12", "station_17"]
    history = GameHistory(agent_names, n_periods=8)

    assert history.get_agent_index("station_5") == 0
    assert history.get_agent_index("station_12") == 1
    assert history.get_agent_index("station_17") == 2

    print(f"  station_5 -> {history.get_agent_index('station_5')}")
    print(f"  station_12 -> {history.get_agent_index('station_12')}")
    print(f"  station_17 -> {history.get_agent_index('station_17')}")

    print("\n✓ 测试 3 通过")


def test_single_ema_update():
    """测试单次 EMA 更新"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 4: 单次 EMA 更新")
    print("=" * 60)

    history = GameHistory(["A", "B"], n_periods=2, ema_lambda=0.1, initial_belief=0.5)

    print(f"  初始 beliefs:\n{history.beliefs}")

    pure_actions = {
        "A": np.array([0.8, 0.9]),
        "B": np.array([0.2, 0.3]),
    }
    history.update_belief(pure_actions)

    # EMA: new = (1 - 0.1) * 0.5 + 0.1 * action
    expected_A = 0.9 * 0.5 + 0.1 * np.array([0.8, 0.9])  # [0.53, 0.54]
    expected_B = 0.9 * 0.5 + 0.1 * np.array([0.2, 0.3])  # [0.47, 0.48]

    print(f"  更新后 beliefs:\n{history.beliefs}")
    print(f"  期望 A: {expected_A}")
    print(f"  期望 B: {expected_B}")

    assert np.allclose(history.beliefs[0], expected_A, rtol=1e-5), "Agent A 信念更新错误"
    assert np.allclose(history.beliefs[1], expected_B, rtol=1e-5), "Agent B 信念更新错误"

    print("\n✓ 测试 4 通过")


def test_ema_convergence():
    """测试多次更新后收敛到实际值"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 5: EMA 收敛性")
    print("=" * 60)

    history = GameHistory(["A"], n_periods=1, ema_lambda=0.1, initial_belief=0.5)

    target = np.array([0.8])
    print(f"  目标值: {target[0]}")
    print(f"  初始信念: {history.beliefs[0, 0]}")

    # 持续输入相同的值
    for i in range(100):
        history.update_belief({"A": target})
        if i < 5 or i % 20 == 0:
            print(f"  第 {i+1} 轮: {history.beliefs[0, 0]:.6f}")

    # 经过足够多次更新，应该接近 target
    assert np.allclose(history.beliefs[0], target, rtol=0.01), \
        f"未收敛到目标值: {history.beliefs[0]} vs {target}"

    print(f"  最终信念: {history.beliefs[0, 0]:.6f}")
    print(f"  与目标差距: {abs(history.beliefs[0, 0] - target[0]):.6f}")

    print("\n✓ 测试 5 通过")


def test_ema_weight_decay():
    """验证 EMA 权重衰减特性"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 6: EMA 权重衰减")
    print("=" * 60)

    # λ=0.05 时，(1-0.05)^45 ≈ 0.099，约衰减到 10%
    history = GameHistory(["A"], n_periods=1, ema_lambda=0.05, initial_belief=0.0)

    # 第一轮输入 1.0
    history.update_belief({"A": np.array([1.0])})
    first_contribution = history.beliefs[0, 0]  # 0.05
    print(f"  第一轮贡献: {first_contribution}")

    # 再输入 44 轮 0.0
    for _ in range(44):
        history.update_belief({"A": np.array([0.0])})

    remaining = history.beliefs[0, 0]
    decay_ratio = remaining / first_contribution

    print(f"  45 轮后剩余: {remaining:.6f}")
    print(f"  衰减比例: {decay_ratio:.4f} (理论值 ≈ 0.099)")

    assert decay_ratio < 0.15, f"衰减不足: {decay_ratio}"

    print("\n✓ 测试 6 通过")


def test_get_beliefs_returns_copy():
    """测试 get_beliefs 返回副本"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 7: get_beliefs 返回副本")
    print("=" * 60)

    history = GameHistory(["A"], n_periods=2, initial_belief=0.5)

    beliefs = history.get_beliefs()
    original_value = history.beliefs[0, 0]
    beliefs[0, 0] = 999.0

    print(f"  修改副本为: {beliefs[0, 0]}")
    print(f"  原数据: {history.beliefs[0, 0]}")

    assert history.beliefs[0, 0] == original_value, "get_beliefs 应返回副本，不影响原数据"

    print("\n✓ 测试 7 通过")


def test_record_creation():
    """测试评估记录创建"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 8: 评估记录创建")
    print("=" * 60)

    history = GameHistory(["A", "B"], n_periods=2)

    pure_actions = {"A": np.array([0.5, 0.6]), "B": np.array([0.4, 0.5])}
    noisy_actions = {"A": np.array([0.52, 0.58]), "B": np.array([0.42, 0.48])}
    rewards = {"A": 100.0, "B": 120.0}
    flows = {"A": np.array([10.0, 15.0]), "B": np.array([12.0, 18.0])}
    ue_info = {"iterations": 5, "converged": True}

    record = history.record(pure_actions, noisy_actions, rewards, flows, ue_info)

    print(f"  eval_id: {record.eval_id}")
    print(f"  rewards: {record.rewards}")
    print(f"  ue_info: {record.ue_info}")
    print(f"  total_evaluations: {history.total_evaluations}")

    assert record.eval_id == 0, "第一条记录 eval_id 应为 0"
    assert record.rewards == rewards, "rewards 不匹配"
    assert record.ue_info == ue_info, "ue_info 不匹配"
    assert history.total_evaluations == 1, "total_evaluations 应为 1"

    print("\n✓ 测试 8 通过")


def test_multiple_records():
    """测试多次记录"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 9: 多次记录")
    print("=" * 60)

    history = GameHistory(["A"], n_periods=1)

    for i in range(5):
        record = history.record(
            pure_actions={"A": np.array([0.5])},
            noisy_actions={"A": np.array([0.5])},
            rewards={"A": float(i * 10)},
            flows={"A": np.array([float(i)])},
        )
        print(f"  记录 {i}: eval_id={record.eval_id}, reward={record.rewards['A']}")
        assert record.eval_id == i, f"eval_id 应为 {i}"

    assert history.total_evaluations == 5, "total_evaluations 应为 5"

    print("\n✓ 测试 9 通过")


def test_get_last_and_recent_records():
    """测试获取最后/最近记录"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 10: 获取最后/最近记录")
    print("=" * 60)

    history = GameHistory(["A"], n_periods=1)

    # 空历史
    assert history.get_last_record() is None, "空历史应返回 None"
    print("  空历史 get_last_record: None ✓")

    # 添加记录
    for i in range(10):
        history.record(
            {"A": np.array([0.5])},
            {"A": np.array([0.5])},
            {"A": float(i * 10)},
            {"A": np.array([float(i)])},
        )

    last = history.get_last_record()
    assert last.eval_id == 9, "最后记录 eval_id 应为 9"
    assert last.rewards["A"] == 90.0, "最后记录 reward 应为 90"
    print(f"  get_last_record: eval_id={last.eval_id}, reward={last.rewards['A']} ✓")

    recent = history.get_recent_records(3)
    assert len(recent) == 3, "应返回 3 条记录"
    assert [r.eval_id for r in recent] == [7, 8, 9], "应返回最后 3 条"
    print(f"  get_recent_records(3): eval_ids={[r.eval_id for r in recent]} ✓")

    # 请求超过总数
    recent_all = history.get_recent_records(100)
    assert len(recent_all) == 10, "应返回全部 10 条"
    print(f"  get_recent_records(100): 返回全部 {len(recent_all)} 条 ✓")

    print("\n✓ 测试 10 通过")


def test_reset():
    """测试重置功能"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 11: 重置功能")
    print("=" * 60)

    history = GameHistory(["A", "B"], n_periods=2)

    # 添加记录和更新信念
    history.record(
        {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])},
        {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])},
        {"A": 100.0, "B": 120.0},
        {"A": np.array([10.0, 10.0]), "B": np.array([12.0, 12.0])},
    )
    history.update_belief({"A": np.array([0.8, 0.9]), "B": np.array([0.2, 0.3])})

    print(f"  重置前: evaluations={history.total_evaluations}, beliefs[0,0]={history.beliefs[0,0]:.4f}")

    # 重置
    history.reset(initial_belief=0.3)

    print(f"  重置后: evaluations={history.total_evaluations}, beliefs[0,0]={history.beliefs[0,0]:.4f}")

    assert history.total_evaluations == 0, "重置后 evaluations 应为 0"
    assert np.allclose(history.beliefs, 0.3), "重置后 beliefs 应为 0.3"

    print("\n✓ 测试 11 通过")


def test_repr():
    """测试字符串表示"""
    from src.game.history import GameHistory

    print("\n" + "=" * 60)
    print("测试 12: 字符串表示")
    print("=" * 60)

    history = GameHistory(["A", "B", "C"], n_periods=8)
    history.record(
        {"A": np.array([0.5] * 8), "B": np.array([0.5] * 8), "C": np.array([0.5] * 8)},
        {"A": np.array([0.5] * 8), "B": np.array([0.5] * 8), "C": np.array([0.5] * 8)},
        {"A": 100.0, "B": 100.0, "C": 100.0},
        {"A": np.array([10.0] * 8), "B": np.array([10.0] * 8), "C": np.array([10.0] * 8)},
    )

    repr_str = repr(history)
    print(f"  repr: {repr_str}")

    assert "GameHistory" in repr_str
    assert "n_periods=8" in repr_str
    assert "total_evaluations=1" in repr_str

    print("\n✓ 测试 12 通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GameHistory 测试套件")
    print("=" * 60 + "\n")

    test_basic_init()
    test_beliefs_shape_and_initial_value()
    test_agent_index_mapping()
    test_single_ema_update()
    test_ema_convergence()
    test_ema_weight_decay()
    test_get_beliefs_returns_copy()
    test_record_creation()
    test_multiple_records()
    test_get_last_and_recent_records()
    test_reset()
    test_repr()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
