# 异步 Fictitious Play + MADDPG 静态博弈求解器设计文档

**日期**: 2026-01-26
**状态**: ✅ P9 集成测试完成，全部阶段已完成

---

## 一、系统全景架构蓝图

本系统融合了多智能体强化学习（MADRL）与传统博弈论中的虚拟博弈（Fictitious Play），针对昂贵的静态博弈（Static Game）进行了专门的异步化和收敛性改造。系统解耦了"物理仿真（慢）"与"信念更新/收敛检测（快）"。

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           GameTrainer                                    │
│                      (异步事件驱动主循环)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌────────────────────┐    │
│  │   Algorithm     │   │   GameHistory   │   │  NashConvChecker   │    │
│  │   (统一接口)     │   │                 │   │                    │    │
│  │                 │   │ • beliefs (EMA) │   │ • compute()        │    │
│  │ • MADDPG        │   │ • records       │   │ • 多起点梯度上升    │    │
│  │ • IDDPG         │   │ • policy_version│   │ • 假收敛检测        │    │
│  │ • MFDDPG        │   │                 │   │                    │    │
│  └────────┬────────┘   └─────────────────┘   └──────────┬─────────┘    │
│           │                                             │              │
│           │              get_critics()                  │              │
│           │         build_critic_input()                │              │
│           └─────────────────────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ParallelEvaluatorPool (已完成)                       │
│                        (异步提交/获取评估结果)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心设计要点（已确认）

### 2.1 状态与观测的改造 (Paradigm Shift)

| 设计项 | 决策 | 理由 |
|--------|------|------|
| γ = 0 | ✅ 已实现 | 静态博弈，Critic 直接拟合即时收益 |
| 剔除流量观测 | ✅ 采纳 | 流量是价格的确定性函数，冗余信息 |
| 剔除静态特征 | ✅ 采纳 | 每个 Agent 独立网络，隐式编码身份 |

**最终观测定义**：
- **State = 对手历史纯出价的 EMA 均值（Belief）**
- 维度：`(n_agents, n_periods)`，包含自身

### 2.2 EMA 增量信念更新 (Belief Update)

使用**指数移动平均（EMA）**以 $O(1)$ 复杂度更新信念：

$$\text{Belief}_{new} = (1 - \lambda) \times \text{Belief}_{old} + \lambda \times \text{Action}^{pure}$$

| 参数 | 值 | 说明 |
|------|-----|------|
| λ | **0.05** | 主要关注最近 ~20 轮，权重衰减到 10% 需要 ~45 轮 |
| 初始值 | **0.5** | 价格空间中点 [0, 1] |

### 2.3 策略意图与探索分离 (Pure vs. Noisy Actions)

系统严格区分两种动作：

| 类型 | 来源 | 去向 | 用途 |
|------|------|------|------|
| **纯策略** $A^{pure}$ | Actor 直接输出 | GameHistory | EMA 更新信念 |
| **噪声策略** $A^{noisy}$ | 纯策略 + 探索噪声 | UE-DTA → Replay Buffer | 试错 + DDPG 训练 |

### 2.4 收敛检测 (NashConv)

放弃使用价格变化率作为主判据，采用 **NashConv** 作为收敛指标：

$$\text{NashConv} = \sum_{i=1}^{N} \left[ Q_i(s, a_i^*, a_{-i}) - Q_i(s, a_i, a_{-i}) \right]$$

其中 $a_i^* = \arg\max_{a_i} Q_i(s, a_i, a_{-i})$ 通过**多起点梯度上升**求得。

| 设计项 | 决策 |
|--------|------|
| 主判据 | NashConv < ε |
| 计算时机 | warmup 后，每隔 K 步计算一次（早期 Critic 不稳定） |
| 假收敛处理 | 价格变化率低 + NashConv 高 → 重置探索噪声 |

**NashConvChecker 设计**：
- 独立类，不依赖具体算法实现
- 通过 `Algorithm.get_critics()` 获取 Critic 网络
- 通过 `Algorithm.build_critic_input()` 构造输入（各算法自己负责格式转换）

---

## 三、异步主循环机制 (Asynchronous Loop)

解决"UE-DTA 仿真耗时 40 秒"的瓶颈，打破木桶效应。

### 3.1 信念更新时机 (Belief Update Timing)

**设计决策**：信念在**提交时立即更新**，而非任务完成后更新。

| 方案 | 信念语义 | 顺序确定性 |
|------|----------|------------|
| 完成后更新 | 对手"实际执行"的策略 | ❌ 依赖仿真耗时 |
| **提交时更新** | 对手"宣称/计划"的策略 | ✅ 与提交顺序一致 |

**原因**：
1. 更符合 Fictitious Play 的理论语义——"观察到对手的策略声明后更新信念"
2. 消除异步带来的随机性，确保信念更新顺序与提交顺序一致
3. `next_beliefs` 可在提交时立即计算，为未来支持 γ > 0 预留接口

### 3.2 双轨记录法 (Dual-Track Recording)

任务提交时，数据兵分两路准备：

| 轨道 | 数据 | 时机 | 用途 |
|------|------|------|------|
| 轨1（博弈论） | 纯策略 $A^{pure}$ | **提交时** | EMA 更新信念 |
| 轨2（深度学习） | $(S, A^{noisy}, R, S')$ | 提交时准备 $S, S'$，完成时获取 $R$ | Replay Buffer |

### 3.3 K 步触发更新 (K-step Trigger)

解耦"数据收集"与"网络更新"：

| 参数 | 值 | 说明 |
|------|-----|------|
| K | **5** | 每 5 次环境评估触发一次网络更新 |
| 任务队列大小 | **Worker 数量** | 保持 Worker 满载 |

更新完成后：
1. `policy_version += 1`
2. 立即用新版本 Actor 生成新任务填充队列

### 3.4 事件驱动主流程伪代码

```python
# 初始化
task_queue = []

# 提交任务函数
def submit_new_task():
    beliefs = game_history.get_beliefs()           # 当前信念
    pure, noisy = algorithm.take_action(beliefs)

    # 【提交时立即更新信念】
    game_history.update_belief(pure)
    next_beliefs = game_history.get_beliefs()      # 更新后的信念

    future = pool.submit(noisy)
    task_queue.append(PendingTask(
        future=future,
        beliefs_snapshot=beliefs,                  # 更新前
        next_beliefs_snapshot=next_beliefs,        # 更新后
        pure_actions=pure,
        noisy_actions=noisy,
    ))

# 初始满负荷提交
for _ in range(n_workers):
    submit_new_task()

# 主循环
while not_converged:
    # 1. 检查是否有任务完成（非阻塞）
    completed = [t for t in task_queue if t.future.done()]

    for task in completed:
        task_queue.remove(task)
        reward = task.future.result()

        # 2. 【记录历史 + 存入 Buffer】
        # 注意：信念已在提交时更新，这里使用预存的快照
        game_history.record(task.pure_actions, task.noisy_actions, reward)
        algorithm.store_experience(
            beliefs=task.beliefs_snapshot,
            noisy_actions=task.noisy_actions,
            rewards=reward,
            next_beliefs=task.next_beliefs_snapshot,  # 使用预存值
        )

        # 3. 【K 步触发更新】
        if total_evaluations % K == 0:
            algorithm.learn()

        # 4. 【立即补充新任务】
        submit_new_task()

    # 5. 【周期性收敛检测】
    if warmup_done and (total_evals % CHECK_INTERVAL == 0):
        nashconv = nashconv_checker.compute(algorithm, game_history)
        if nashconv < epsilon:
            break
        # 假收敛检测
        if price_change_rate < threshold and nashconv > epsilon:
            algorithm.reset_noise()
```

---

## 四、GameHistory 数据结构设计

### 4.1 核心职责

1. 维护 EMA 信念（所有 Agent 的价格均值估计）
2. 记录评估历史
3. 根据算法类型提供不同格式的观测

### 4.2 数据结构

```python
class GameHistory:
    def __init__(self, agent_names: list[str], n_periods: int, ema_lambda: float = 0.05):
        self.agent_names = agent_names
        self.n_agents = len(agent_names)
        self.n_periods = n_periods
        self.ema_lambda = ema_lambda

        # EMA 信念：每个 Agent 对所有 Agent 的价格信念
        # beliefs[agent] = (n_agents, n_periods)
        self.beliefs: dict[str, np.ndarray] = {
            agent: np.full((self.n_agents, n_periods), 0.5)  # 初始化为价格空间中点
            for agent in agent_names
        }

        # 评估记录
        self.records: list[EvaluationRecord] = []

        # 策略版本
        self.policy_version: int = 0

    def update_belief(self, pure_actions: dict[str, np.ndarray]):
        """用纯策略更新 EMA 信念"""
        for agent in self.agent_names:
            for j, other_agent in enumerate(self.agent_names):
                self.beliefs[agent][j] = (
                    (1 - self.ema_lambda) * self.beliefs[agent][j]
                    + self.ema_lambda * pure_actions[other_agent]
                )

    def get_observations(self, algorithm_type: str) -> dict[str, np.ndarray]:
        """根据算法类型返回不同格式的观测"""

        if algorithm_type == "MADDPG":
            # 返回完整信念矩阵 (n_agents, n_periods)
            return {agent: self.beliefs[agent].copy() for agent in self.agent_names}

        elif algorithm_type == "MFDDPG":
            # 返回：自身信念 + 其他 Agent 的平均
            obs = {}
            for i, agent in enumerate(self.agent_names):
                own_belief = self.beliefs[agent][i]  # (n_periods,)
                others_indices = [j for j in range(self.n_agents) if j != i]
                others_mean = np.mean(self.beliefs[agent][others_indices], axis=0)  # (n_periods,)
                obs[agent] = np.stack([own_belief, others_mean])  # (2, n_periods)
            return obs

        elif algorithm_type == "IDDPG":
            # 只返回自身信念
            return {
                agent: self.beliefs[agent][i]  # (n_periods,)
                for i, agent in enumerate(self.agent_names)
            }

        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    @property
    def total_evaluations(self) -> int:
        return len(self.records)
```

### 4.3 EvaluationRecord 结构

```python
@dataclass
class EvaluationRecord:
    """单次评估记录"""
    eval_id: int                          # 全局递增计数
    policy_version: int                   # 策略版本号

    # 动作
    pure_actions: dict[str, np.ndarray]   # 纯策略动作
    noisy_actions: dict[str, np.ndarray]  # 噪声动作（实际执行）

    # 环境返回
    rewards: dict[str, float]
    flows: dict[str, np.ndarray]
    ue_info: dict
```

---

## 五、Algorithm 接口扩展

在 `task_env.md` 的 Algorithm 接口基础上，增加 NashConv 计算所需的方法：

```python
class Algorithm(Protocol):
    """算法统一接口"""

    @property
    def name(self) -> str:
        """算法名称：MADDPG / IDDPG / MFDDPG"""
        ...

    def take_action(self, observations: dict, add_noise: bool = True
                    ) -> tuple[dict, dict]:
        """
        输出动作

        Returns:
            (pure_actions, noisy_actions)
        """
        ...

    def store_experience(self, observations: dict, actions: dict,
                         rewards: dict, next_observations: dict) -> None:
        """存入 ReplayBuffer"""
        ...

    def learn(self) -> dict | None:
        """从 Buffer 采样并更新网络"""
        ...

    # === NashConv 计算所需 ===

    def get_critics(self) -> dict[str, nn.Module]:
        """返回各 Agent 的 Critic 网络"""
        ...

    def build_critic_input(self, beliefs: np.ndarray, agent: str,
                           all_actions: dict[str, np.ndarray]
                           ) -> torch.Tensor:
        """
        为指定 Agent 构造 Critic 输入（用于 NashConv 计算）

        各算法自己负责格式转换：
        - MADDPG: 全局状态 + 所有动作
        - IDDPG: 局部状态 + 自身动作
        - MFDDPG: MF 状态 + 自身动作
        """
        ...

    def reset_noise(self) -> None:
        """重置探索噪音（假收敛时调用）"""
        ...
```

---

## 六、与 task_env.md 的整合

### 6.1 组件对应关系

| task_env.md 组件 | task_play.md 调整 |
|------------------|-------------------|
| NetworkData | 无变化 |
| EVCSRewardEvaluator | 无变化 |
| ParallelEvaluatorPool | 无变化 |
| GameHistory | 增加 EMA 信念 + 观测格式转换 |
| Algorithm 接口 | 增加 `get_critics()` + `build_critic_input()` |
| GameTrainer | 改为异步事件驱动 + K 步触发 + NashConv 收敛 |

### 6.2 新增组件

| 组件 | 职责 |
|------|------|
| NashConvChecker | 独立的收敛检测器，多起点梯度上升计算 NashConv |

### 6.3 实施顺序调整

基于 `task_env.md` 的进度，调整后的实施顺序：

| 阶段 | 任务 | 状态 |
|------|------|------|
| P1 | NetworkData | ✅ 完成 |
| P2 | EVCSRewardEvaluator | ✅ 完成 |
| P3 | ParallelEvaluatorPool | ✅ 完成 |
| **P4** | **GameHistory（含 EMA 信念）** | ✅ 完成 |
| **P5** | **Algorithm 接口（含 NashConv 支持）** | ✅ 完成 |
| P6 | 重构 MADDPG/IDDPG/MFDDPG | ✅ 完成 |
| **P7** | **GameTrainer（异步主循环）** | ✅ 完成 |
| **P8** | **NashConvChecker** | ✅ 完成 |
| P9 | 集成测试 | ✅ 完成 |

### 6.4 P9 集成测试结果

**测试日期**: 2026-01-26

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 端到端功能测试 | ✅ 通过 | MADDPG/IDDPG/MFDDPG 三种算法均正常完成训练流程 |
| 并行加速比测试 | ✅ 通过 | 2 Workers 加速比 1.79x，串行/并行结果一致 |
| 收敛行为测试 | ○ 跳过 | 需要 --full 模式（长时间运行） |
| 内存稳定性测试 | ○ 跳过 | 需要 --full 模式 |
| 算法一致性测试 | ✅ 通过 | 相同种子下最终信念差异 < 0.05 |

**验收标准达成情况**：

| 验收项 | 状态 | 说明 |
|--------|------|------|
| 并行评估达到 0.8N 倍加速比 | ✅ | 2 Workers 达到 1.79x（>1.6） |
| 固定 seed 时结果确定性 | ✅ | 串行/并行评估结果完全一致 |
| 新架构训练流程可运行 | ✅ | 三种算法均完成 15 次评估的训练 |

---

## 七、关键参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| γ | 0 | 静态博弈折扣因子 |
| λ (EMA) | 0.05 | 信念更新系数，关注最近 ~20 轮 |
| K | 5 | 每 5 次评估触发一次网络更新 |
| 任务队列大小 | Worker 数量 | 保持满载 |
| 信念初始值 | 0.5 | 价格空间中点 |
| Exploitability 阈值 | 0.05 | 收敛判据，可配置 |
| NashConv 检测间隔 | 10 | 每 10 次评估计算一次 |
| NashConv warmup | 100 | 最小经验数后开始计算 |
| 多起点数量 | 5 | 梯度上升避免局部最优 |
| 优化步数 | 50 | 每个起点的梯度上升步数 |

---

## 八、协作规则

沿用 `task_env.md` 的协作规则：

1. **方案确认** → **代码实现** → **文档更新**
2. 方案确认使用简要文字表达设计核心，对齐后再实施
3. 代码实现遵循：先整体框架，再具体接口
4. 有不清楚的地方先确认再行动
5. 需要运行脚本时，提供命令由用户执行，用户反馈运行结果
