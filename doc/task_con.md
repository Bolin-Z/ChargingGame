# 博弈问题定义与求解方法讨论记录

**日期**: 2026-01-14
**参与者**: User, Antigravity

---

## 一、 问题定义

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

## 二、 求解方法探讨

针对此类问题，主要的求解思路包括：

### 1. 迭代最佳响应 (Iterated Best Response, IBR)
*   **方法**: 轮流固定 $N-1$ 个人的策略，优化剩下 1 个人的策略。
*   **优缺点**: 实现简单，但在某些博弈结构下可能无法收敛（进入极限环）。

### 2. 多智能体强化学习 (MARL) - 当前采用的方法
*   **算法**: MADDPG, IDDPG, MFDDPG。
*   **原理**: Actor-Critic 结构，Critic 拟合黑盒收益面，Actor 进行策略梯度下降。
*   **优势**: 能够处理连续空间和随机性，Exploration 机制有助于跳出局部最优。

### 3. 演化计算 (Evolutionary Computation, EC)
*   **关联**: 演化博弈论 (EGT) 的复制动态方程 (Replicator Dynamics) 与强化学习的 Policy Gradient 在数学上具有深刻联系。
*   **推荐算法**:
    *   **CMA-ES (协方差矩阵自适应进化策略)**: 适合连续空间、中小规模参数的黑盒优化，样本效率优于遗传算法。
    *   **SAEA (代理模型辅助演化算法)**: 针对 **仿真昂贵** 的情况，利用 Gaussian Process 或神经网络拟合收益面，大幅减少真实仿真次数。
    *   **协同演化 (Co-evolution)**: 将 N 个 Player 视为不同的种群进行对抗性演化。

---

## 三、 纳什均衡验证与监控

### 1. 验证方法：单边偏离 (Unilateral Deviation)
要验证解 $\mathbf{p} = (p_1, ..., p_N)$ 是否为纳什均衡：
1.  **固定对手**: 保持 $\mathbf{p}_{-i}$ 不变。
2.  **寻找最佳响应**: 搜索 Player $i$ 的最优策略 $p_i^*$。
3.  **计算后悔值 (Regret)**: $\text{Regret}_i = U_i(p_i^*) - U_i(p_i)$。
4.  **判定**: 若 $\sum \text{Regret}_i \approx 0$，则是均衡；否则 Regret 越大，离均衡越远。

### 2. 训练过程监控
在 MARL 训练中，不能只看 Reward（可能震荡）。应关注：
*   **策略稳定性**: Action 变化率是否趋近于 0。
*   **Exploitability Monitor**: 每隔 X 轮暂停训练，进行一次简化的“单边偏离检查”，计算 Nash Gap。只有 Nash Gap 收敛才算训练完成。

---

## 四、 当前项目现状分析 (基于 task_nash.md)

*   **现状**: 
    *   MADDPG 收敛于中间值，MFDDPG 收敛于边界值。
    *   **双双通过不了验证**: 两个解的 Nash Gap 均很大，Player 可以通过偏离到中间价格（~1.2）大幅获利。
*   **推测原因**:
    1.  训练不充分（收敛判据太松）。
    2.  纯策略均衡不存在（需要混合策略）。
    3.  参数设置导致均衡点不稳定（如用户对价格过分敏感）。

---

## 五、 后续建议

1.  **引入 Exploitability Monitor**: 将单边偏离检查集成到训练循环中。
2.  **尝试演化计算**: 考虑使用 CMA-ES 或 SAEA 替代/辅助现有的 RL 算法，特别是应对仿真慢的问题。
3.  **参数调整**: 考虑调整 `time_value_coefficient` 等参数，改变博弈结构，看是否存在更稳定的均衡。

---

  ## 六、基于当前实现的诊断与改进建议（面向 MARL）

  结合当前代码与实验现象（“价格变化很小但单边偏离仍可大幅获利”），更可能是实现与评估口径导致“假收敛”，建议优先处理：

  ### 6.1 关键实现问题（会直接影响是否能学到均衡）
  1. **Actor 梯度被断开，导致策略学不动（或学得极慢）**
     在更新 Actor 时将 Actor 输出转换为 numpy（例如 `.detach().cpu().numpy()`）再送入 Critic 构造输入，会切断从 Critic 到 Actor 的梯度路径，Actor
  loss 对 Actor 参数的梯度趋近于 0。
     影响：非常容易得到“稳定但可被单边偏离击穿”的解。

  4. **探索衰减过快会加剧假收敛**
     `noise_decay` 每次动作调用就衰减，可能很快逼近 `min_noise`，在策略未真正学好时就停止有效探索。
     影响：早早固定在次优点/边界点/中间点。

  ### 6.2 建议的修复顺序（仍基于 MARL）
  1. **先修 Actor 更新链路（确保梯度不断）**
     Actor 更新时构造 Critic 输入应尽量保持在 torch 张量域内完成，避免将当前 agent 的动作转换为 numpy；保证 `actor_loss -> critic -> action ->
  actor` 的梯度路径存在。

  2. **将奖励缩放改为“固定常数缩放”或“运行中固定的 reward_scale”**
     避免使用依赖当前 joint action 的动态缩放因子（如当步 `max_reward`）。

  3. **引入 Exploitability / Regret Monitor 作为训练停止与对比指标**
     可复用单边偏离测试的思路：每隔若干 step/episode 固定对手策略，对每个 agent 测试一组偏离动作（或做更系统的黑盒搜索），估计 Nash Gap/Regret；只有
  该指标下降并稳定才算“接近均衡”。

  4. **调整探索与评估流程**
     训练时保持足够探索（更慢的 `noise_decay`，或按 episode 衰减）；评估时使用 `add_noise=False` 得到稳定策略表现，并用 Monitor 做可利用性检查。

  ### 6.3 下一轮建议实验
  - A：仅修复 Actor 梯度链路 + 固定 reward_scale，其他不动，观察 Nash Gap 是否显著下降。
  - B：在 A 基础上加入 Exploitability Monitor，用其替代“价格相对变化率”作为早停依据。
  - C：在 B 基础上做探索衰减扫描（`noise_decay`、`min_noise`、按 step vs 按 episode 衰减）。

  ---

  ## 七、关于 flow_scale_factor 的实现分析与建议

  ### 7.1 现有实现是什么

  当前项目的 `flow_scale_factor` 是一种**观测缩放（feature scaling）**手段：不改变环境动力学，只改变输入给神经网络的流量特征量级。

  - **流量统计口径**：环境统计充电流量时，每辆“仿真 Vehicle 对象”代表 `deltan` 辆真实车，因此每次计数是 `charging_flows += deltan`，flow 的数值天然
  会被放大（SiouxFalls 里常见到几十~几百）。
    参考：`src/env/EVCSChargingGameEnv.py:903`

  - **scale 的来源**：从各场景的 `*_settings.json` 读取 `flow_scale_factor`。
    参考：`src/env/EVCSChargingGameEnv.py:334`；`data/*/*_settings.json`

  - **scale 的作用位置**：在算法侧将 `own_charging_flow` 除以 `flow_scale_factor`，用于 Actor 输入与 Critic 输入的状态构造。
    参考：`src/algorithms/maddpg/maddpg.py:540`、`src/algorithms/maddpg/maddpg.py:568`
    参考：`src/algorithms/iddpg/iddpg.py:533`、`src/algorithms/iddpg/iddpg.py:559`
    参考：`src/algorithms/mfddpg/mfddpg.py:421`

  ### 7.2 这套做法是否合理（结论）

  - **在“数值尺度/训练稳定性”角度：合理**。
    把 flow 从“几十~几百”的量级压到“0.x~几”的量级，是标准的网络输入预处理，有助于网络同时关注价格与流量特征，减轻梯度尺度不平衡问题。

  - **但文档里“对齐到价格[0.5,2.0]量级”的目标与实际网络输入不一致**。
    网络看到的价格特征主要是 `last_round_all_prices`（环境里是将实际价格归一化到 `[0,1]` 后提供给观测），并不是 `[0.5,2.0]` 的实际价格。
    因此更合理的目标应是：让缩放后的 flow 的主质量落在 **0~1 附近**（或做标准化），而不是落在 0.5~2.0 附近。
    参考：`src/env/EVCSChargingGameEnv.py:543`（观测使用 normalize 后的价格）

  ### 7.3 为什么“flow_scale 会改变学到的解”不能直接用来判断对错

  在当前代码状态下，训练结果对很多因素敏感（例如 Actor 梯度链路、奖励缩放方式、探索衰减等）。
  在这些因素未先修复之前，仅凭“学到边界/学到中间”来反推 flow_scale 是否合理，结论可能被混杂因素误导。

  ### 7.4 更稳健的改进建议（可选方案）

  建议至少选一种做对照实验，避免只靠直觉设定 scale：

  1. **按分位数定标（推荐）**
     设定 `flow_scale_factor` 使得 `P95(flow) / scale ≈ 1`（或 `P90 ≈ 1`），而不是用 `max/scale`。
     优点：对长尾更稳健；跨场景更一致。

  2. **对 flow 做 `log1p` 再缩放/标准化**
     使用 `log1p(flow)` 处理长尾，并在大量 0 的时段保持区分度。
     优点：减少极端 flow 对网络的主导。

  3. **使用“份额/占比”流量特征**
     例如每时段 `flow_share = flow_i / (sum_j flow_j + eps)`，天然落在 0~1。
     优点：减少 `demand_multiplier`、`deltan` 等参数变化导致的尺度漂移。

  ### 7.5 建议的验证方式

  不要只看 `relative_change_rate` 是否下降；应结合“可利用性/后悔值（Regret/Nash Gap）”监控。
  在固定其他策略的条件下对单个 agent 测试偏离（或做更系统的黑盒搜索），比较不同 `flow_scale_factor` 的 Nash Gap 变化趋势，再决定保留/回滚/重设。


  • 是的，下一步应当把“梯度链路正确”放在最高优先级，同时把 Actor/Critic 的学习状态做成可观测的；不然你看到的“价格稳定”很可能只是噪声衰减后停住，而不是
  学到了均衡。

  - 先修梯度（必须）
      - MADDPG：src/algorithms/maddpg/maddpg.py:455 这里把 new_actions 转成 numpy，Actor 更新时梯度会断；应改成全程 torch 张量构造 Critic 输入（至少
        对当前 agent 的 action 这条路径不能断）。
      - IDDPG：同类问题在 src/algorithms/iddpg/iddpg.py:456、src/algorithms/iddpg/iddpg.py:457。
  - 再加学习诊断（建议同时做）
      1. 每次 learn 记录 critic_loss/actor_loss（滑动平均）；Actor loss 长期几乎不变通常就是梯度断/学习率问题。
      2. 记录梯度范数：||grad(actor)||, ||grad(critic)||（比如每 100 step 打一次）；出现长期 0 或 NaN 直接定位。
      3. 记录参数更新幅度：||theta - theta_prev||（确认 optimizer 真的在动）。
      4. 记录 Q 值统计（均值/方差/极值）+ target Q 统计（看是否爆炸/塌缩）。
      5. 记录探索强度：当前 noise sigma（你现在 noise_decay=0.99 会很快衰到 min_noise，容易“看似收敛”）。
  - 最小验收测试（强烈建议）
      - 写一个不跑环境的“小脚本/单测”：随机造一批 obs/action，跑一次 actor 更新，断言 actor 参数发生变化且 actor 的梯度范数 > 0（这能在 1 秒内证明梯
        度链路已修复）。

  等梯度确认没问题后，再去讨论 flow_scale、奖励缩放、以及"收敛判据=价格不变"这些更高层的均衡问题；否则后面的调参基本都是在噪声上做文章。

---

## 八、代码验证与现状确认（2026-01-19）

**参与者**: User, Claude (Opus)

基于对 Gemini/GPT 讨论文档的审查，结合代码实际验证，确认以下结论：

### 8.1 Actor 梯度断开问题 —— ✅ 已确认

| 算法 | 问题位置 | 梯度状态 |
|------|----------|----------|
| **MADDPG** | `maddpg.py:455` `new_actions[i].detach().cpu().numpy()` | ❌ 断开 |
| **IDDPG** | `iddpg.py:456` `current_actions[i].detach().cpu().numpy()` | ❌ 断开 |
| **MFDDPG** | `mfddpg.py:364-366` 全程 torch 张量操作 | ✅ 完整 |

**关键发现**：MFDDPG 的梯度链路是正确的，这解释了为什么三个算法表现不同——不仅是算法设计差异，还有实现正确性差异。

### 8.2 单边偏离测试结果 —— ✅ 已运行验证

运行 `test_unilateral_deviation.py` 的结果：

**MADDPG 中间值解（价格 ~1.0-1.4）**：
| 智能体 | 基准收益 | 最优偏离收益 | 增益 | 最优偏离方向 |
|--------|----------|--------------|------|--------------|
| 5 | 12,184 | 25,421 | **+109%** | 降价至 ~1.0 |
| 12 | 7,105 | 19,979 | **+181%** | 降价至 ~0.9 |
| 14 | 9,733 | 29,089 | **+199%** | 降价至 ~1.0 |
| 18 | 30,317 | 44,236 | **+46%** | 降价至 ~1.2 |

**MFDDPG 边界解（价格 ~2.0，个别 0.5）**：
| 智能体 | 基准收益 | 最优偏离收益 | 增益 | 最优偏离方向 |
|--------|----------|--------------|------|--------------|
| 5 | 9,072 | 18,188 | **+100%** | 降至中间价 1.2 |
| 12 | 7,483 | 13,453 | **+80%** | 降至中间价 1.7 |
| 14 | 6,383 | 15,971 | **+150%** | 降至中间价 1.7 |
| 18 | 33,802 | 52,646 | **+56%** | 降至中间价 1.2 |

**结论**：
1. **两个解都不是纳什均衡** —— 文档诊断正确
2. **所有智能体的最优偏离都指向中间价格（~1.2）** —— 真正的均衡可能在更低的价格区间
3. **"假收敛"现象确认** —— 价格稳定但 Nash Gap 巨大

### 8.3 探索衰减验证 —— ✅ 已确认

当前配置：`sigma=0.2`, `decay=0.9995`, `min_sigma=0.01`

计算：衰减到 min_sigma 需要 `log(0.01/0.2) / log(0.9995) ≈ 5991` 次调用
- 4 个 agent × 50 step/episode = 200 次/episode
- **约 30 个 episode 后噪音就到最小值**

这意味着在策略真正学好之前，探索就已经停止。

### 8.4 其他已验证项

| 文档观点 | 代码验证 | 状态 |
|----------|----------|------|
| 价格归一化到 [0,1] | `EVCSChargingGameEnv.py:543` | ✅ 正确 |
| flow_scale_factor 作用位置 | 三个算法的 `process_observations` 函数 | ✅ 正确 |
| 奖励动态归一化 | `normalize_rewards()` 用当步 max_reward | ✅ 确认（次要问题） |

---

## 九、修复计划

### 9.1 优先级排序

| 优先级 | 任务 | 原因 |
|--------|------|------|
| **P0** | 修复 MADDPG/IDDPG 的 Actor 梯度链路 | 根本问题，不修则 Actor 不学习 |
| **P1** | 添加学习诊断指标 | 验证修复效果，提供可观测性 |
| **P2** | 调整探索衰减策略 | 防止过早停止探索 |
| **P3** | 奖励归一化改进 | 次要问题，梯度修复后再考虑 |

### 9.2 P0：修复 Actor 梯度链路

**MADDPG 修复方案**（`maddpg.py:_update_actor`）：

当前问题代码：
```python
for i in range(batch_size):
    current_actions = batch_actions[i].copy()
    current_actions[agent_id] = new_actions[i].detach().cpu().numpy()  # ← 梯度断开
    global_state = organize_global_state(batch_obs[i], current_actions, ...)
```

修复思路：
- 保持当前 agent 的 action 在 torch 张量域
- 仅对其他 agent 的 action 转换为 numpy（它们本来就不需要梯度）
- 重构 `organize_global_state` 或在 Actor 更新时使用专门的张量版本

**IDDPG 修复方案**（`iddpg.py:_update_actor`）：同理

**MFDDPG**：无需修复，梯度链路已完整

### 9.3 P1：学习诊断指标

在 `learn()` 方法中添加返回值：
```python
{
    'critic_loss': float,
    'actor_loss': float,
    'actor_grad_norm': float,
    'critic_grad_norm': float,
    'q_value_mean': float,
    'q_value_std': float,
    'noise_sigma': float,  # 当前探索强度
}
```

### 9.4 P2：探索衰减调整

选项：
1. **按 episode 衰减**（推荐）：每个 episode 结束时衰减，而非每次 action
2. **更慢的衰减率**：`decay=0.999` → `decay=0.9999`
3. **更高的 min_sigma**：`min_sigma=0.01` → `min_sigma=0.05`

### 9.5 验收标准

1. **梯度验证测试**：随机数据跑一次 actor 更新，断言参数变化且梯度范数 > 0
2. **Nash Gap 下降**：修复后重新训练，单边偏离测试的最大增益应显著降低
3. **学习曲线合理**：actor_loss 应有明显变化趋势，而非长期平稳

---

## 十、学习诊断指标实现（2026-01-19）

**参与者**: User, Claude (Opus)

### 10.1 实现目标

在修复梯度链路之前，先添加诊断指标系统，用于：
1. **确认问题存在**：通过观察 Actor 梯度范数是否接近 0 来验证梯度断开
2. **建立基线数据**：记录修复前的指标，便于与修复后对比
3. **持续监控**：为后续训练提供可观测性

### 10.2 实现的诊断指标

| 指标类别 | 具体指标 | 诊断目的 |
|---------|---------|---------|
| **损失函数** | `critic_loss`, `actor_loss` | 确认学习在进行 |
| **梯度健康** | `actor_grad_norm`, `critic_grad_norm` | 检测梯度断开/爆炸/消失 |
| **Q值统计** | `q_value_mean`, `q_value_std`, `q_value_max`, `q_value_min` | 检测 Q 值爆炸或塌缩 |
| **探索强度** | `noise_sigma` | 监控探索是否过早停止 |

### 10.3 代码修改清单

#### 10.3.1 MADDPG 算法层 (`src/algorithms/maddpg/maddpg.py`)

1. **`_update_critic()` 方法**：返回值从 `float` 改为 `dict`
   - 新增：`critic_grad_norm`, `q_value_mean/std/max/min`, `target_q_mean`

2. **`_update_actor()` 方法**：返回值从 `float` 改为 `dict`
   - 新增：`actor_grad_norm`（关键诊断指标）

3. **新增 `_compute_grad_norm()` 方法**：计算网络参数梯度的 L2 范数

4. **`learn()` 方法**：返回值从 `bool` 改为 `dict | None`
   - 汇总所有 agent 的诊断指标
   - 包含 `batch_size`, `buffer_size`, 每个 agent 的完整指标

#### 10.3.2 Trainer 层 (`src/trainer/MADDPGTrainer.py`)

1. **`_run_episode()` 方法**：
   - 记录 `learn_metrics` 到 `step_records`
   - 进度条显示 Actor 梯度范数

2. **新增 `_generate_diagnostics_summary()` 方法**：
   - 从 `step_records` 提取学习指标
   - 计算每个 agent 的统计摘要（mean/std/min/max/first/last）

3. **新增 `_diagnose_learning_issues()` 方法**：
   - 自动检测潜在问题：
     - Actor 梯度接近 0 → 梯度断开
     - 探索噪音降至最小值 → 过早停止探索
     - Q 值过大 → Q 值爆炸
     - Actor loss 几乎不变 → Actor 没有在学习

4. **新增 `_print_diagnostics_summary()` 方法**：
   - 训练结束后打印诊断摘要到控制台

### 10.4 诊断输出示例

训练结束后自动输出：

```
============================================================
学习诊断摘要
============================================================
总学习步数: 299

--- agent_0 ---
  Actor 梯度范数: mean=X.XXe-XX, first=X.XXe-XX, last=X.XXe-XX
  Critic 梯度范数: mean=X.XXe-XX
  Q 值均值: mean=X.XXXX, first=X.XXXX, last=X.XXXX
  探索噪音 sigma: first=0.1000, last=0.XXXX

--- agent_1 ---
  ...

诊断结论:
  ⚠️ agent_0: Actor 梯度接近 0 (mean=X.XXe-XX)，可能存在梯度断开问题
  ⚠️ agent_0: 探索噪音已降至最小值 (sigma=0.0100)，可能过早停止探索
============================================================
```

### 10.5 测试脚本

创建 `test_diagnostics.py` 用于快速验证：
- 配置：1 episode, 300 steps
- 目的：快速确认诊断功能正常工作

### 10.6 预期结果

运行测试后，预期观察到：
1. **Actor 梯度范数接近 0**：确认 MADDPG 存在梯度断开问题
2. **Critic 梯度范数正常**（非零）：Critic 网络正常学习
3. **探索噪音快速下降**：确认探索衰减过快问题

### 10.7 下一步

根据诊断结果：
1. 如果 Actor 梯度确实接近 0 → 执行 P0 修复（梯度链路）
2. 如果 Actor 梯度正常 → 重新分析问题原因