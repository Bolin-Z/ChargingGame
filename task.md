# UE-DTA 参数优化任务记录

## 1. 任务背景

基于 `task_network_simplification.md` 文档中 5.2.10 节提出的**方案C（Gap敏感 + 时间衰减）**切换机制，已在 `EVCSChargingGameEnv.py` 中实现。

切换概率公式：
$$P_{switch}(n) = \min(1, \gamma \cdot gap_{rel}) \cdot \frac{1}{1 + \alpha \cdot n}$$

- $\gamma$：Gap敏感度系数
- $\alpha$：时间衰减速率
- $n$：当前迭代轮数
- $gap_{rel}$：相对成本差

## 2. 参数扫描实验

### 2.1 实验配置

- **扫描脚本**：`sweep_ue_parameters.py`
- **执行时间**：2025-12-18
- **参数范围**：
  - gamma: [5, 10, 15, 20]
  - alpha: [0.02, 0.03, 0.05, 0.08]
- **测试网络**：siouxfalls, berlin_friedrichshain, anaheim
- **总组合数**：16 × 3 = 48

### 2.2 实验结果（原始，有缺陷）

| 网络 | 最佳 γ | 最佳 α | GM | P95 | 迭代次数 | Completed | 状态 |
|-----|--------|--------|-----|-----|---------|-----------|------|
| **Sioux Falls** | 15 | 0.03 | 0.59% | 3.04% | 10 | **53.9%** | ⚠️ 虚假收敛 |
| **Berlin** | 10 | 0.05 | 1.60% | 10.26% | 4 | 100% | ✅ 良好 |
| **Anaheim** | 15 | 0.08 | 2.69% | 16.55% | 53 | 96.5% | ✅ 良好 |

### 2.3 问题分析与修复

#### 2.3.1 问题根源

原评分函数只考虑 GM 和迭代次数，**未考虑 completed_ratio**：
```python
score = 100 * (1 - 0.7 * gm_norm - 0.3 * iter_norm)
```

导致 Sioux Falls 的 γ=15, α=0.03 配置被错误选为"最佳"：
- GM=0.59%（看似很低）+ iterations=10（很少）→ 高分
- 但 completed_ratio=53.9%，实际是**虚假收敛**

#### 2.3.2 虚假收敛机制

1. 高 γ + 低 α → 早期迭代切换概率接近 100%（对比固定概率 5%）
2. 大量车辆切换到"最优"路径 → 该路径严重拥堵
3. 车辆无法在 simulation_time 内完成 → completed_ratio 下降
4. 未完成车辆不参与 gap 计算 → 少量完成车辆的 gap 很小
5. 快速"收敛"（实为假收敛）

#### 2.3.3 修复方案

添加硬约束：**completed_ratio < 95% 的配置直接排除**

分析脚本：`analyze_sweep_results.py`

### 2.4 修正后的最佳参数

| 网络 | 最佳 γ | 最佳 α | GM | P95 | 迭代次数 | Completed | 状态 |
|-----|--------|--------|-----|-----|---------|-----------|------|
| **Sioux Falls** | 5 | 0.08 | 0.79% | 4.63% | 24 | 100% | ✅ 修正 |
| **Berlin** | 10 | 0.05 | 1.60% | 10.26% | 4 | 100% | ✅ 不变 |
| **Anaheim** | 15 | 0.08 | 2.69% | 16.55% | 53 | 96.5% | ✅ 不变 |

### 2.5 关键发现

1. **Berlin 表现优异**
   - 100% completed_ratio
   - 仅需 4 轮迭代即收敛
   - GM < 2%，满足收敛标准

2. **Sioux Falls 问题已修复**
   - 原 γ=15, α=0.03 为虚假收敛
   - 修正为 γ=5, α=0.08，completed_ratio=100%
   - 低 γ 值避免早期过度切换

3. **Anaheim 基本正常**
   - 96.5% completed_ratio
   - 迭代次数较多（53轮），但最终收敛

4. **参数规律**
   - 较小网络（Sioux Falls）需要较低的 γ
   - 较大 α（0.08）普遍表现较好（快速衰减避免过度切换）

## 3. 已解决问题

### 3.1 ✅ Sioux Falls completed_ratio 异常

**问题描述**：Sioux Falls 作为 24 节点的理论网络，completed_ratio 应接近 100%，但实测仅 53.9%。

**根因分析**：
- ~~`simulation_time` 设置过短~~ ❌ 不是根因
- ~~`demand_multiplier` 过高导致网络拥堵~~ ❌ 不是根因
- ✅ **评分函数缺陷**：未考虑 completed_ratio，导致选出"虚假收敛"参数
- ✅ **虚假收敛**：高 γ + 低 α 导致早期过度切换，车辆无法按时完成

**解决方案**：
- 添加 completed_ratio ≥ 95% 硬约束
- Sioux Falls 参数从 γ=15, α=0.03 修正为 γ=5, α=0.08

### 3.2 ✅ 更新各网络 settings.json

已更新配置文件：

| 网络 | ue_switch_gamma | ue_switch_alpha | 状态 |
|-----|-----------------|-----------------|------|
| Sioux Falls | 5 | 0.08 | ✅ 已更新 |
| Berlin | 10 | 0.05 | ✅ 已确认 |
| Anaheim | 15 | 0.08 | ✅ 已更新 |

### 3.3 ✅ Berlin demand_multiplier 优化

**问题发现**：参数扫描中 Berlin 网络所有 16 个 γ/α 组合都在 **4 轮**迭代内收敛，参数敏感性极低。

**原因分析**：`demand_multiplier=2.0` 网络压力过低，拥堵效应不明显，削弱博弈的空间差异化意义。

**敏感性测试** (`test_bf_demand_multiplier.py`)：

| Multiplier | 迭代次数 | GM | P95 | Completed |
|------------|---------|-----|-----|-----------|
| 2.0 | 4 | 1.60% | 10.34% | 100% |
| 2.1 | 5 | 1.76% | 11.11% | 100% |
| 2.2 | 5 | 1.96% | 12.50% | 100% |
| 2.3 | 7 | 2.70% | 16.13% | 100% |
| 2.4 | 9 | 2.49% | 14.29% | 100% |
| **2.5** | **25** | **2.75%** | **16.28%** | **100%** |

**结论**：在当前 γ=10, α=0.05 配置下，demand_multiplier=2.5 表现良好：
- 100% completed_ratio
- 25 轮迭代（有博弈意义）
- GM=2.75%, P95=16.28%

**已更新**：`berlin_friedrichshain_settings.json` → `demand_multiplier: 2.5`

### 3.4 更新 task_network_simplification.md

记录本次实验结果到文档 5.2.10.6 节。

## 4. 下一步工作计划

- [x] **Step 1**：检查 Sioux Falls 配置，分析 completed_ratio 异常原因
- [x] **Step 2**：修复评分函数，重新选择最佳参数
- [x] **Step 3**：更新三个网络的 `settings.json` 配置
- [x] **Step 4**：运行验证测试，确认改进效果
- [x] **Step 5**：更新 `task_network_simplification.md` 文档

## 5. 相关文件

- 参数扫描脚本：`sweep_ue_parameters.py`
- 扫描结果分析：`analyze_sweep_results.py`
- 扫描结果报告：`results/parameter_sweep/sweep_report.json`
- 可视化图表：`results/parameter_sweep/{network}_parameter_sweep.png`
- 环境实现：`src/env/EVCSChargingGameEnv.py`
- 网络配置：
  - `data/siouxfalls/siouxfalls_settings.json`
  - `data/berlin_friedrichshain/berlin_friedrichshain_settings.json`
  - `data/anaheim/anaheim_settings.json`

## 6. 协作记录

| 日期 | 进展 |
|-----|------|
| 2025-12-18 | 完成参数扫描实验，发现 Sioux Falls completed_ratio 异常 |
| 2025-12-18 | 分析问题根因：评分函数未考虑 completed_ratio，导致选出"虚假收敛"参数 |
| 2025-12-18 | 创建 `analyze_sweep_results.py`，添加 completed_ratio ≥ 95% 硬约束 |
| 2025-12-18 | 更新 Sioux Falls (γ=5, α=0.08) 和 Anaheim (γ=15, α=0.08) 参数 |
| 2025-12-18 | Berlin demand_multiplier 敏感性测试，从 2.0 提升到 2.5，迭代次数从 4 增加到 25 |
