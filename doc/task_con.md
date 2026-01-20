# 博弈问题定义与求解方法讨论记录

**日期**: 2026-01-14 ~ 2026-01-20
**参与者**: User, Antigravity, Claude (Opus)

---

## 一、问题定义

用户的场景描述如下：
> N 个 Player 在连续价格区间上选择价格，通过仿真环境（DUE）计算收益，目标是最大化自身收益。

**学术定义**:
这是一个 **N人非合作静态博弈 (N-player Non-cooperative Static Game)**。
具体归类为 **空间伯特兰德寡头竞争 (Spatial Bertrand Oligopoly)**：
*   **非合作**: 追求个人利益最大化。
*   **静态**: 同时决策（Simultaneous move）。
*   **连续动作空间**: 价格是连续变量。
*   **黑盒/随机收益**: 收益由复杂的交通仿真（DUE）计算，通常不可导且含有噪声。

---

## 二、求解方法探讨

### 1. 迭代最佳响应 (IBR)
*   **方法**: 轮流固定 $N-1$ 个人的策略，优化剩下 1 个人的策略。
*   **优缺点**: 实现简单，但可能无法收敛（进入极限环）。

### 2. 多智能体强化学习 (MARL) - 当前采用
*   **算法**: MADDPG, IDDPG, MFDDPG。
*   **原理**: Actor-Critic 结构，Critic 拟合黑盒收益面，Actor 进行策略梯度下降。
*   **优势**: 能够处理连续空间和随机性，Exploration 机制有助于跳出局部最优。

### 3. 演化计算 (EC)
*   **CMA-ES**: 适合连续空间、中小规模参数的黑盒优化。
*   **SAEA**: 利用代理模型减少仿真次数，适合仿真昂贵的场景。
*   **协同演化**: 将 N 个 Player 视为不同种群进行对抗性演化。

---

## 三、纳什均衡验证方法

### 单边偏离测试 (Unilateral Deviation)
1.  **固定对手**: 保持 $\mathbf{p}_{-i}$ 不变。
2.  **寻找最佳响应**: 搜索 Player $i$ 的最优策略 $p_i^*$。
3.  **计算后悔值**: $\text{Regret}_i = U_i(p_i^*) - U_i(p_i)$。
4.  **判定**: 若 $\sum \text{Regret}_i \approx 0$，则是均衡。

### 训练过程监控
*   **策略稳定性**: Action 变化率是否趋近于 0。
*   **Exploitability Monitor**: 定期计算 Nash Gap，只有 Nash Gap 收敛才算训练完成。

---

## 四、已确认问题（2026-01-19）

### 4.1 初期现状
*   MADDPG 收敛于中间值，MFDDPG 收敛于边界值。
*   **两个解都通不过单边偏离验证**：Nash Gap 很大，Player 可通过偏离大幅获利。

### 4.2 代码验证结论

| 问题 | 验证结果 |
|------|----------|
| **Actor 梯度断开** | ✅ MADDPG/IDDPG 的 `.detach().cpu().numpy()` 导致梯度为 0；MFDDPG 正常 |
| **探索衰减过快** | ✅ 约 30 episode 后噪音就到最小值，策略过早固化 |
| **奖励动态归一化** | ✅ 使用当步 max_reward，导致 Critic 学习不稳定 |
| **价格归一化** | ✅ 正确归一化到 [0,1] |
| **flow_scale_factor** | ✅ 作用位置正确，用于观测缩放 |

### 4.3 单边偏离测试结果（修复前）

**MADDPG 中间值解（价格 ~1.0-1.4）**：

| 智能体 | 基准收益 | 最优偏离收益 | 增益 |
|--------|----------|--------------|------|
| 5 | 12,184 | 25,421 | **+109%** |
| 12 | 7,105 | 19,979 | **+181%** |
| 14 | 9,733 | 29,089 | **+199%** |
| 18 | 30,317 | 44,236 | **+46%** |

**结论**：所有智能体最优偏离都指向中间价格（~1.2），"假收敛"现象确认。

---

## 五、已完成修复（汇总）

### 5.1 修复清单

| 优先级 | 任务 | 完成日期 | 状态 |
|--------|------|----------|------|
| **P0** | 修复 MADDPG/IDDPG 的 Actor 梯度链路 | 2026-01-19 | ✅ 完成 |
| **P1** | 添加学习诊断指标 | 2026-01-19 | ✅ 完成 |
| **P2** | 固定 reward_scale | 2026-01-20 | ✅ 完成 |
| **P3** | 设置 gamma = 0 | 2026-01-20 | ✅ 完成 |
| **P4** | 记录纯策略输出 | 2026-01-20 | ✅ 完成 |

### 5.2 梯度链路修复（2026-01-19）

**修复方法**：
- MADDPG：新增 `_build_global_state_for_actor_update()` 方法，保持当前 agent 动作的梯度
- IDDPG：新增 `_build_local_state_for_actor_update()` 方法
- MFDDPG：无需修复

**修复前后对比**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Actor 梯度范数 | `0.00e+00`（全为0） | `3.64e-02 ~ 5.44e-02` |
| Critic 梯度范数 | `0.27 ~ 0.56` | `0.31 ~ 0.49` |

### 5.3 学习诊断指标（2026-01-19）

添加的诊断指标：`critic_loss`, `actor_loss`, `actor_grad_norm`, `critic_grad_norm`, `q_value_mean/std/max/min`, `noise_sigma`

三个算法的 `learn()` 方法返回统一格式的诊断字典。

### 5.4 固定 reward_scale（2026-01-20）

**问题**：动态缩放导致相同收益在不同步被缩放成不同值，Critic 学习不稳定。

**修复**：基于历史数据（P99=41,771），设定 `reward_scale = 50,000`。

### 5.5 gamma = 0 修改（2026-01-20）

**修改位置**：`src/utils/config.py`

```python
# 三个算法配置的 gamma 默认值
MADDPGConfig.gamma = 0.0  # 原 0.99
IDDPGConfig.gamma = 0.0   # 原 0.99
MFDDPGConfig.gamma = 0.0  # 原 0.99
```

**理论依据**：静态博弈中无状态转移，Critic 应直接拟合即时收益 $Q(s,a) \approx r$，而非 Bellman 方程的解。

### 5.6 纯策略输出记录（2026-01-20）

**问题**：收敛判据基于带噪音的价格，噪音衰减会导致"假收敛"。

**修改**：`take_action()` 方法返回值从单一动作改为元组 `(action, pure_action)`。

**修改的文件**：
- `src/algorithms/maddpg/maddpg.py`：`DDPG.take_action()` 和 `MADDPG.take_action()`
- `src/algorithms/iddpg/iddpg.py`：`DDPG.take_action()` 和 `IndependentDDPG.take_action()`
- `src/algorithms/mfddpg/mfddpg.py`：`DDPG.take_action()` 和 `MFDDPG.take_action()`
- `src/trainer/MADDPGTrainer.py`
- `src/trainer/IDDPGTrainer.py`
- `src/trainer/MFDDPGTrainer.py`

**新的返回格式**：
```python
# 算法层
action, pure_action = agent.take_action(obs, add_noise=True)

# 管理器层
actions, pure_actions = maddpg.take_action(observations, add_noise=True)
```

**数据记录格式**：
```python
step_record = {
    'actions': {...},           # 带噪音的动作
    'pure_actions': {...},      # 纯策略输出（不含噪音）
    'actual_prices': {...},     # 带噪音动作对应的实际价格
    'pure_prices': {...},       # 纯策略对应的价格
    ...
}
```

---

## 六、实验结果

### 6.1 修复后单边偏离测试（2026-01-20）

测试对象：修复后 MADDPG 学到的边界震荡解

| Agent | 基准收益 | 最优偏离收益 | 增益 |
|-------|---------|-------------|------|
| 5 | 16,842 | 24,052 | **+42.8%** |
| 12 | 9,574 | 10,885 | **+13.7%** |
| 14 | 18,935 | 33,750 | **+78.2%** |
| 18 | 42,826 | 45,800 | **+6.9%** |

**修复前后 Nash Gap 对比**：

| Agent | 修复前增益 | 修复后增益 | 变化 |
|-------|-----------|-----------|------|
| 5 | +220% | +42.8% | ✅ 改善 |
| 12 | +150% | +13.7% | ✅ 改善 |
| 14 | +74% | +78.2% | ❌ 无改善 |
| 18 | +26% | +6.9% | ✅ 改善 |

### 6.2 固定 reward_scale 实验结果（2026-01-20）

| 算法 | 总步数 | Episodes | 最终价格模式 | 总收益 |
|------|--------|----------|-------------|--------|
| MADDPG | 270 | 3 | 边界震荡（0.5/2.0交替） | 34,646 |
| IDDPG | 232 | 3 | 多数高价（~2.0） | 97,575 |
| MFDDPG | 206 | 3 | 多数高价（~2.0） | 42,003 |

**观察**：
1. 所有算法都倾向高价边界
2. 部分 Agent 出现边界震荡（Agent 5/12 在 0.5/2.0 交替）
3. 快速"收敛"：3 episode 内就达到收敛判据

### 6.3 当前结论

**梯度修复有效但不充分**：
- ✅ Actor 能正常学习（梯度非零）
- ✅ Nash Gap 整体有所下降（3/4 agent 改善）
- ❌ 仍未找到纳什均衡（所有 agent 都能偏离获利）
- ❌ 学到极端边界解而非经济学上合理的中间价格

---

## 七、算法设计问题深度分析（2026-01-20）

### 7.1 问题一：噪音衰减导致假收敛

**现象**：当前收敛判据基于"价格相对变化率"，而价格 = Actor输出 + 噪音。

**问题链条**：
1. 噪音按 step 衰减：`sigma = 0.1 × 0.99^step`
2. 200 步后：`sigma ≈ 0.013`（衰减 87%）
3. 价格变化自然减小，触发收敛判据（< 0.01）
4. **但这不是策略收敛，而是噪音消失**

**实验数据验证**：
| 阶段 | MADDPG sigma | IDDPG sigma | MFDDPG sigma |
|------|--------------|-------------|--------------|
| 早期 | 0.0824 | 0.0832 | 0.0840 |
| 后期 | **0.0086** | **0.0108** | **0.0134** |

**解决方案**：收敛判据应基于**纯策略输出**（Actor网络输出，不含噪音），而非最终价格。

```python
# 当前（有问题）：基于带噪音的价格
price = actor(obs) + noise
change = |price_new - price_old|  # 受噪音衰减影响

# 改进：基于纯策略输出
pure_action = actor(obs)  # 不加噪音
change = |pure_action_new - pure_action_old|  # 不受噪音影响
```

### 7.2 问题二：DDPG 的 MDP 假设不适用于静态博弈

**核心矛盾**：DDPG 为马尔可夫决策过程（MDP）设计，但本项目是静态博弈。

| 维度 | MDP（DDPG假设） | 静态博弈（本项目） |
|------|----------------|-------------------|
| 状态转移 | $s_{t+1} = f(s_t, a_t)$ | **无真正状态转移** |
| 时序结构 | 当前动作影响未来状态 | 最优价格不依赖历史 |
| 目标 | 最大化累积折扣奖励 $\sum \gamma^t r_t$ | 最大化**单次博弈收益** |
| Q函数含义 | 状态-动作的长期价值 | 给定对手策略的即时收益 |

**参考文献对比**：

原文 Mean-Field DQN 的损失函数：
```
L(θk) = Σ (rk_t - Qk(π̄k_t, πk_t))²
```

当前 MADDPG 的损失函数：
```
L = (r + γ·Q(s',a') - Q(s,a))²
```

**关键区别**：原文**没有 γQ(s',a') 项**！原文把 Q 网络当作纯粹的"奖励预测器"，而非 Bellman 方程的解。

**当前实现的问题**：
- `gamma = 0.99` 引入了对"下一状态"的价值估计
- 但静态博弈中，"下一状态"只是同一博弈的另一次观测
- Critic 学到的是被 TD 更新污染的伪值函数

### 7.3 修改方案：gamma = 0

**修改**：将 `gamma` 从 `0.99` 改为 `0.0`

```python
# 修改后的 TD 目标
target_q = reward + 0.0 * critic_target(next_state, next_action)
         = reward  # 直接拟合即时奖励
```

**修改后 Critic 学习目标**：
$$Q(s, a) \approx \mathbb{E}[r | s, a]$$

即：给定当前观测（对手历史价格、自身历史流量）和自身动作，预测**即时收益**。

**修改后 Actor 学习目标**：
$$\pi^*(s) = \arg\max_a Q(s, a)$$

即：找到在当前观测下，能最大化即时收益的动作（最佳响应）。

### 7.4 修改后算法的理论解释

修改后的 MADDPG/IDDPG/MFDDPG 本质上变成了：

1. **Critic**：一个函数逼近器，学习 $Q(s,a) \approx r$（收益面拟合）
2. **Actor**：通过 Critic 梯度进行策略优化，寻找最佳响应
3. **多智能体交互**：隐式的迭代最佳响应过程

这与原文 Mean-Field DQN 的思路一致，只是：
- 原文用 DQN（离散动作）
- 本项目用 DDPG（连续动作）

### 7.5 预期效果

| 方面 | 修改前 | 修改后预期 |
|------|--------|-----------|
| Critic 目标 | $r + \gamma Q'$（有偏） | $r$（无偏） |
| Q 值含义 | 模糊的"伪值函数" | 清晰的"即时收益预测" |
| Actor 梯度 | 基于有偏 Q 值 | 基于无偏收益估计 |
| 收敛点 | 未知 | 更可能是最佳响应 |

---

## 八、待办事项

### 8.1 高优先级

| 任务 | 原因 | 状态 |
|------|------|------|
| **重构收敛判断职责** | 环境应是纯收益计算器，收敛判断应在Trainer层基于pure_prices | ✅ 完成 |
| **调整噪音参数** | sigma=0.2, decay=0.995, min=0.02 | 🔲 待实施 |

### 8.2 中优先级

| 任务 | 原因 |
|------|------|
| 为 BF/Anaheim 配置 reward_scale | 新数据集缺少此配置 |
| 多数据集对比实验 | 验证算法在不同规模网络的表现 |

### 8.3 低优先级

| 任务 | 说明 |
|------|------|
| 引入 Nash Gap 监控 | 用于验证是否真正收敛到纳什均衡 |
| 尝试 BR-PSO 对比 | 作为 MARL 的理论基准 |

---

## 九、收敛判断重构（2026-01-20）

### 9.1 问题背景

**核心问题**：收敛判断基于带噪音的价格，噪音衰减会导致"假收敛"。

```
问题链条：
1. 噪音按 step 衰减：sigma = 0.1 × 0.99^step
2. 200 步后：sigma ≈ 0.013（衰减 87%）
3. 价格变化自然减小，触发收敛判据（< 0.01）
4. 但这不是策略收敛，而是噪音消失
```

**设计违背**：环境内置收敛判断逻辑，违背"静态博弈环境是纯收益计算器"的原则。

### 9.2 修改方案

**职责重新划分**：

| 层级 | 修改前职责 | 修改后职责 |
|------|-----------|-----------|
| **环境** | 计算收益 + 判断收敛 | **仅计算收益**（纯收益计算器） |
| **Trainer** | 被动接收环境的收敛信号 | **主动判断收敛**（基于 pure_prices） |

### 9.3 具体修改

**环境 (`EVCSChargingGameEnv.py`)**：

```python
# 修改前
def __init__(self, ..., convergence_threshold, stable_steps_required):
    self.convergence_threshold = convergence_threshold
    self.stable_steps_required = stable_steps_required
    self.convergence_counter = 0

def step(self, actions):
    ...
    # 基于带噪音价格判断收敛
    relative_change_rate = self.__calculate_relative_change_rate()
    if relative_change_rate < self.convergence_threshold:
        self.convergence_counter += 1
    terminated = self.convergence_counter >= self.stable_steps_required

# 修改后
def __init__(self, ..., max_steps):
    self.max_steps = max_steps
    # 移除 convergence_threshold, stable_steps_required, convergence_counter

def step(self, actions):
    ...
    # 环境不判断收敛，仅基于 max_steps 截断
    terminated = False  # 始终返回 False
    truncated = self.current_step >= self.max_steps
    # relative_change_rate 保留在 infos 中作为监控指标
```

**Trainer (`MADDPGTrainer.py`, `IDDPGTrainer.py`, `MFDDPGTrainer.py`)**：

```python
# 新增状态变量
self.convergence_threshold = self.config.convergence_threshold
self.stable_steps_required = self.config.stable_steps_required
self.pure_price_history = []      # 纯策略价格历史
self.convergence_counter = 0      # 连续收敛步数计数器

# 新增方法
def _calculate_pure_price_change_rate(self) -> float:
    """基于 pure_price_history 计算，不受探索噪音影响"""
    if len(self.pure_price_history) < 2:
        return float('inf')
    current = self.pure_price_history[-1]
    previous = self.pure_price_history[-2]
    # 计算相对变化率...

def _check_step_convergence(self, pure_prices) -> tuple[bool, float]:
    """检查单步是否收敛（基于纯策略价格）"""
    self.pure_price_history.append(pure_prices)
    pure_change_rate = self._calculate_pure_price_change_rate()

    if pure_change_rate < self.convergence_threshold:
        self.convergence_counter += 1
    else:
        self.convergence_counter = 0

    stable_converged = self.convergence_counter >= self.stable_steps_required
    return stable_converged, pure_change_rate

# 修改 _run_episode
def _run_episode(self, episode, observations):
    # 重置 episode 级别状态
    self.pure_price_history = []
    self.convergence_counter = 0

    for step in range(...):
        actions, pure_actions = algo.take_action(observations, add_noise=True)
        pure_prices = env.actions_to_prices_dict(pure_actions)

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # 基于纯策略价格判断收敛（Trainer 职责）
        stable_converged, pure_change_rate = self._check_step_convergence(pure_prices)

        if stable_converged:
            return True, step + 1

        if all(truncations.values()):
            break
```

### 9.4 修改效果

**进度条显示变化**：

```
# 修改前（基于带噪音价格）
Episode 0: 100%|███| 50/50 [00:30] UE迭代=5, 相对变化=0.0089

# 修改后（基于纯策略价格）
Episode 0: 100%|███| 50/50 [00:30] UE迭代=5, 纯策略变化=0.0523, 收敛计数=0/5
```

**收敛判断变化**：

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 判断依据 | `price = actor(obs) + noise` | `pure_price = actor(obs)` |
| 噪音影响 | 噪音衰减会触发假收敛 | 不受噪音影响 |
| 判断位置 | 环境层 | Trainer 层 |
| 数据记录 | `relative_change_rate` | `pure_change_rate` + `relative_change_rate` |

### 9.5 配置说明

收敛参数位于 `ScenarioProfile`（`src/utils/config.py`）：

```python
@dataclass
class ScenarioProfile:
    # 收敛控制配置
    convergence_threshold: float   # 纳什均衡价格收敛阈值
    stable_steps_required: int     # 稳定收敛所需的连续步数
    stable_episodes_required: int  # 训练提前终止所需的连续收敛episodes数
```

### 9.6 修改的文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/env/EVCSChargingGameEnv.py` | 移除收敛判断参数和逻辑，`terminations` 始终为 `False` |
| `src/trainer/MADDPGTrainer.py` | 新增基于 `pure_prices` 的收敛判断 |
| `src/trainer/IDDPGTrainer.py` | 同步修改 |
| `src/trainer/MFDDPGTrainer.py` | 同步修改 |
