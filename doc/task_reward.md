# Reward持续下降问题诊断记录

**日期**: 2026-01-21
**问题**: 所有agent的reward随着迭代持续下降

---

## 一、问题现象

### 1.1 观察到的数据

三个算法（MADDPG、IDDPG、MFDDPG）都出现相同问题：

| 算法 | 前10步总Reward均值 | 后10步总Reward均值 | 变化 |
|------|-------------------|-------------------|------|
| MADDPG | 69,335 | 15,289 | **-77.9%** |
| IDDPG | 57,326 | 10,961 | **-80.9%** |
| MFDDPG | 82,404 | 20,904 | **-74.6%** |

### 1.2 各Agent的Reward变化（MADDPG为例）

| Agent | 前10步均值 | 后10步均值 | 变化 |
|-------|-----------|-----------|------|
| 5 | 7,942 | 1,609 | -79.7% |
| 12 | 11,456 | 1,471 | -87.2% |
| 14 | 15,763 | 4,362 | -72.3% |
| 18 | 34,174 | 7,847 | -77.0% |

---

## 二、排除的假设

### 2.1 ❌ 价格下降导致

**数据**：价格几乎没有变化
- 初期均价: 1.21
- 后期均价: 1.23
- 变化: +0.02（基本不变）

**结论**：价格不是reward下降的原因

### 2.2 ❌ 车辆无法完成行程

**数据**：完成率始终为100%
```
Step | 充电完成/总数 | 完成率
0    | 1780/1780    | 100.0%
76   | 1780/1780    | 100.0%
```

**结论**：所有充电车辆都完成了行程

### 2.3 ❌ 充电车辆总数减少

**数据**：充电车辆数稳定
- 充电车辆总数: ~1,780（始终不变）
- 非充电车辆总数: ~31,225（始终不变）

**结论**：车辆数量没有问题

---

## 三、确认的问题

### 3.1 ✅ 充电成本大幅下降

| 指标 | 初期 | 后期 | 变化 |
|------|------|------|------|
| 充电均成本 | 63.4 | 19.4 | **-69%** |
| 非充电均成本 | 7.8 | 7.7 | 不变 |

**分析**：
- 充电成本 = 出行时间成本 + 充电价格
- 价格不变，说明是**出行时间成本**大幅下降
- 出行时间下降说明**路网拥堵减轻**
- 拥堵减轻说明**充电路径上的车辆减少了**

### 3.2 ✅ 隐含流量大幅下降

通过 `flow = reward / (price × charging_demand_per_vehicle)` 反推流量：

| 指标 | Step 0 | Step 76 | 变化 |
|------|--------|---------|------|
| 实际充电车辆数 | ~1,780 | ~1,780 | 不变 |
| **隐含流量（从reward反推）** | **1,494** | **308** | **-79%** |

**关键发现**：
- 1,780辆车完成了充电
- 但只有308辆被统计到reward计算中
- **约83%的充电车辆"消失"了**

### 3.3 ✅ 流量在Agent间重新分配

| Agent | Step 0 流量份额 | Step 76 流量份额 | 变化 |
|-------|----------------|-----------------|------|
| 5 | 2.2% | 9.3% | +7.1% |
| 12 | 20.0% | 10.9% | -9.1% |
| 14 | 27.5% | 51.0% | +23.5% |
| 18 | 50.2% | 28.7% | -21.5% |

### 3.4 ✅ 价格边界震荡

Step 76的价格呈现0.5/2.0边界震荡模式：
```
Agent 5:  [1.79, 0.50, 1.02, 2.00, 1.99, 0.52, 1.73, 0.50]
Agent 12: [0.69, 2.00, 1.87, 1.75, 0.50, 0.50, 0.50, 0.50]
Agent 14: [1.63, 1.97, 1.55, 0.51, 0.54, 1.20, 0.50, 0.50]
Agent 18: [0.50, 2.00, 1.87, 2.00, 0.74, 2.00, 1.61, 2.00]
```

---

## 四、问题根因分析

### 4.1 设计背景

当前设计：**每个step继承上一步的路径分配**

```
Step 0: UE-DTA迭代 → 路径分配A
Step 1: 继承路径分配A → UE-DTA迭代 → 路径分配B
Step N: 继承路径分配(N-1) → UE-DTA迭代 → 路径分配N
```

### 4.2 问题链条

```
Step间继承路径分配
    ↓
路径分配逐步演化
    ↓
车辆选择充电站基于总成本（时间+价格）
    ↓
UE均衡使车辆流向低成本路径
    ↓
某些充电站/时段的流量趋近于0
    ↓
Reward大幅下降
```

### 4.3 流量"消失"的可能原因

**假设A：流量统计逻辑bug**
- 某些完成行程的车辆未被统计到 `charging_flows`
- 需要验证 `__calculate_actual_vehicle_cost_and_flow` 的逻辑

**假设B：时段/充电站匹配问题**
- 车辆充电时间超出仿真时间范围
- 充电站节点名称匹配失败

**假设C：路径演化导致充电链路未被经过**
- 路径分配变化导致车辆绕过充电链路
- 但这与"充电车辆必须经过充电链路"的设计矛盾

---

## 五、待验证

### 5.1 调试脚本

已创建 `debug_flow.py` 用于验证流量统计：

```bash
python D:\MyCode\ChargingGame\debug_flow.py
```

该脚本会：
1. 创建环境并执行一个step
2. 对比实际充电车辆数和隐含流量
3. 直接检查 `charging_flow_history` 矩阵

### 5.2 需要确认的问题

1. `charging_flows` 矩阵的实际值是多少？
2. 每个时段的流量分布如何？
3. 是否有车辆的充电链路未被正确识别？

---

## 六、分析脚本清单

| 脚本 | 用途 |
|------|------|
| `analyze_rewards.py` | 分析reward变化趋势 |
| `analyze_flow.py` | 分析流量和UE迭代信息 |
| `analyze_ue.py` | 分析UE迭代中的成本变化 |
| `analyze_completed.py` | 分析完成车辆数变化 |
| `analyze_breakdown.py` | 分析reward的组成 |
| `analyze_redistribution.py` | 分析流量在Agent间的重新分配 |
| `debug_flow.py` | 调试流量统计逻辑 |

---

## 七、深度诊断（2026-01-21续）

### 7.1 调试结果汇总

#### 测试1：单次UE迭代流量统计（`debug_flow_detail.py`）

| 指标 | 数值 |
|------|------|
| 分配路径有充电链路 | 356（100%）|
| 实际路径有充电链路 | 355 |
| 流量统计 | 1775 |
| **流量与完成数一致** | ✅ |

**结论**：单次迭代时，流量统计逻辑正确。

#### 测试2：调用 `env.step()` 后的流量统计

| 指标 | 数值 |
|------|------|
| UE统计完成充电车辆 | 1780 |
| 充电流量矩阵总计 | 1095 |
| **差距** | **685辆（38.5%丢失）** |

**结论**：`env.step()` 中存在流量丢失问题。

#### 测试3：多次创建World的车辆ID一致性（`debug_id_mismatch.py`）

```
模板中有但新World没有: 131
新World有但模板没有: 148

丢失的ID示例:
  21-16-4440.0-4494: 在新World中存在, charging_car=False  ← 属性变了！
```

**关键发现**：同一个车辆ID，在不同World中的 `charging_car` 属性不同！

#### 测试4：连续创建World的充电车辆变化（`debug_random_effect.py`）

```
连续创建World检查:
World 1: 356 个充电车辆
World 2: 356 个充电车辆，与World 1差异 6 个ID
World 3: 363 个充电车辆，与World 1差异 95 个ID
World 4: 334 个充电车辆，与World 1差异 134 个ID
World 5: 352 个充电车辆，与World 1差异 16 个ID
```

**关键发现**：每次调用 `__create_simulation_world()` 创建的World中，充电车辆的ID集合会变化！

### 7.2 根因确认

**问题根因**：`__create_simulation_world()` 每次创建的World中，车辆的 `charging_car` 属性会随机变化。

**问题链条**：

```
__run_simulation() 开始
    ↓
创建 W_template，构建 dict_od_to_charging_vehid（基于当时的charging_car属性）
    ↓
删除 W_template
    ↓
UE-DTA迭代开始
    ↓
每轮迭代创建新的 W（此时charging_car属性可能已变化！）
    ↓
用 dict_od_to_charging_vehid（旧的充电车辆ID）去统计流量
    ↓
部分充电车辆不在 dict 中 → 流量未被统计
    ↓
reward下降
```

**核心问题**：
- `dict_od_to_charging_vehid` 基于 `W_template` 中的充电车辆ID构建
- 但后续每轮UE迭代创建的 `W` 中，相同ID的车辆可能已经不是充电车辆了
- 同时，新的充电车辆ID不在 `dict_od_to_charging_vehid` 中
- 导致这些充电车辆的流量不会被统计

### 7.3 待验证的根本原因

虽然已经确认了"每次创建World时charging_car属性会变化"这一现象，但还需要确认**为什么会变化**：

**可能原因A：`self.W.VEHICLES` 的属性不稳定**
- 测试显示 `self.W.VEHICLES` 多次迭代是稳定的
- 但 `__create_simulation_world()` 中的 `veh.attribute.copy()` 可能有问题

**可能原因B：`uxsimpp_extended` 的随机性**
- C++ 层面可能存在随机性影响车辆属性
- 需要检查 `Vehicle` 构造函数和 `VehiclesDict` 的行为

**可能原因C：迭代顺序问题**
- `self.W.VEHICLES.values()` 的迭代顺序可能不稳定
- 导致车辆ID与属性的对应关系混乱

### 7.4 调试脚本清单（新增）

| 脚本 | 用途 |
|------|------|
| `debug_flow_detail.py` | 检查单次UE迭代的流量统计 |
| `debug_ue_iteration.py` | 检查UE多轮迭代中的流量统计 |
| `debug_id_mismatch.py` | 检查充电车辆ID不匹配问题 |
| `debug_random_effect.py` | 检查随机状态对充电车辆的影响 |
| `debug_iteration_order.py` | 检查VEHICLES迭代顺序 |
| `debug_attribute_changes.py` | 检查车辆属性变化 |
| `debug_direct_comparison.py` | 直接对比连续创建的World |

---

## 八、最新诊断进展（2026-01-21续2）

### 8.1 验证流量丢失（`debug_step_flow.py`）

执行 `env.step()` 后观察流量统计：

| Step | 总流量（观测） | 期望流量 | 丢失比例 |
|------|---------------|---------|---------|
| 0 | 840 | 1780 | 52.8% |
| 1 | 495 | 1780 | 72.2% |
| 2 | 585 | 1780 | 67.1% |
| 3 | 490 | 1780 | 72.5% |
| 4 | 500 | 1780 | 71.9% |

**结论**：流量丢失问题确实存在，丢失比例在50%-72%之间。

### 8.2 手动仿真对比（`debug_flow_loss.py`）

| 指标 | env.W (reset后) | 手动创建World |
|------|----------------|--------------|
| 充电Vehicle对象数 | 356 | 371 |
| 完成行程 | - | 369/371 (99.5%) |
| 经过充电链路 | - | 369 (100%) |
| 预期流量 | 1780 | 1845 |

**关键发现**：手动创建的World中充电车辆数(371)与env.W中(356)不同！

### 8.3 World对比测试（`debug_world_compare.py`）

#### 测试1：env.W vs 新创建的World

```
env.W 充电车辆: 356
新World 充电车辆: 356
差异: 0
属性变化的车辆: 0
```

**结论**：env.W 与第一次创建的World完全一致。

#### 测试2：多次调用 `__create_simulation_world()` 的稳定性

```
第1次: 356 个充电车辆
第2次: 356 个充电车辆
第3次: 356 个充电车辆

三次调用的充电车辆ID是否一致:
  1 vs 2: True
  2 vs 3: False  ← 从第3次开始不一致！
  1 vs 3: False
```

**关键发现**：
- 充电车辆**数量**始终是356
- 但**哪些车辆**是充电车辆从第3次开始会变化
- 第1次和第2次一致，第3次开始出现差异

### 8.4 更新的根因分析

**问题链条（更新版）**：

```
env.reset()
    ↓
创建 env.W，充电车辆ID集合 = Set_A (356个)
    ↓
env.step() 调用 __run_simulation()
    ↓
__run_simulation() 创建 W_template，构建 dict_od_to_charging_vehid（基于Set_A）
    ↓
删除 W_template
    ↓
UE-DTA迭代开始，每轮创建新的W
    ↓
第1-2轮：新W的充电车辆ID = Set_A（一致）
    ↓
第3轮开始：新W的充电车辆ID = Set_B ≠ Set_A（不一致！）
    ↓
用 Set_A 的ID去统计 Set_B 的流量 → 部分车辆流量丢失
```

**核心问题**：
- `__create_simulation_world()` 多次调用后，充电车辆ID集合会变化
- 可能与World对象的创建/销毁、内存状态、或C++层随机性有关
- 前2次调用稳定，第3次开始不稳定

### 8.5 待调查

1. **为什么第3次开始不稳定**：
   - 是否与前两个World的资源释放有关？
   - 是否有全局状态被修改？
   - 是否与随机数生成器状态有关？

2. **`__create_simulation_world()` 内部逻辑**：
   - 检查 `veh.attribute.copy()` 的行为
   - 检查车辆创建顺序和ID分配逻辑

### 8.6 调试脚本清单（更新）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `debug_step_flow.py` | 验证env.step()后的流量统计 | ✅ 完成 |
| `debug_flow_loss.py` | 追踪流量丢失原因 | ✅ 完成 |
| `debug_world_compare.py` | 对比env.W和新创建World | ✅ 完成 |
| `debug_flow_detail.py` | 检查单次UE迭代的流量统计 | 已有 |
| `debug_direct_comparison.py` | 直接对比连续创建的World | 已有 |

---

## 九、根因定位完成（2026-01-21续3）

### 9.1 不稳定性来源调查（`debug_instability_source.py`）

#### 测试1：释放资源后的稳定性（C++扩展版）

```
第1次: 356 个充电车辆
第2次: 356 个充电车辆
第3次: 356 个充电车辆
第4次: 389 个充电车辆  ← 数量变化！
第5次: 347 个充电车辆  ← 数量变化！

一致性检查:
  1 vs 2: False, 差异ID数: 6
  1 vs 3: False, 差异ID数: 6
  1 vs 4: False, 差异ID数: 61
  1 vs 5: False, 差异ID数: 99
```

#### 测试2：不释放资源时的行为（C++扩展版）

```
第1次: 344 个充电车辆
第2次: 344 个充电车辆
第3次: 344 个充电车辆
第4次: 344 个充电车辆
第5次: 344 个充电车辆

一致性检查:
  1 vs 2: True, 差异ID数: 0
  1 vs 3: True, 差异ID数: 0
  1 vs 4: True, 差异ID数: 0
  1 vs 5: True, 差异ID数: 0  ← 完全一致！
```

**关键发现**：
- **释放资源后**：充电车辆ID不稳定，数量和ID都会变化
- **不释放资源**：完全稳定，5次调用结果完全一致

### 9.2 Python版UXsim对比测试（`debug_python_uxsim.py`）

#### 简单测试（释放资源）

```
第1次: 34 个充电车辆
第2次: 34 个充电车辆
第3次: 34 个充电车辆
第4次: 34 个充电车辆
第5次: 34 个充电车辆

一致性检查 (释放资源):
  1 vs 2: True, 差异ID数: 0
  1 vs 3: True, 差异ID数: 0
  1 vs 4: True, 差异ID数: 0
  1 vs 5: True, 差异ID数: 0  ← 完全一致！
```

#### 真实数据测试（释放资源）

```
模板World: 6601个Vehicle对象, 356个充电车辆

第1次: 6601个Vehicle, 356个充电车辆
第2次: 6601个Vehicle, 356个充电车辆
第3次: 6601个Vehicle, 356个充电车辆
第4次: 6601个Vehicle, 356个充电车辆
第5次: 6601个Vehicle, 356个充电车辆

充电车辆数量一致性检查:
  所有次数数量相同: True  ← 完全一致！
```

### 9.3 根因确认

| 测试环境 | 释放资源 | 结果 |
|---------|---------|------|
| C++扩展版（uxsimpp_extended） | 是 | ❌ 不稳定 |
| C++扩展版（uxsimpp_extended） | 否 | ✅ 稳定 |
| Python版（uxsim+patch） | 是 | ✅ 稳定 |
| Python版（uxsim+patch） | 否 | ✅ 稳定 |

**根因确认**：**问题出在C++扩展层（uxsimpp_extended）的资源释放逻辑！**

当调用 `W._cpp_world.release()` 释放C++层资源后，后续创建的World中车辆的 `charging_car` 属性会出现不一致。这是C++扩展特有的问题，Python版UXsim不存在此问题。

### 9.4 问题机制

```
World.release() 被调用
    ↓
C++ 层释放内存/重置某些全局状态
    ↓
下次创建 World 时，某些内部状态已被污染
    ↓
导致 Vehicle 的 attribute 分配出现不一致
    ↓
charging_car 属性与预期不符
    ↓
流量统计丢失
```

### 9.5 调试脚本清单（最终）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `debug_step_flow.py` | 验证env.step()后的流量统计 | ✅ 完成 |
| `debug_flow_loss.py` | 追踪流量丢失原因 | ✅ 完成 |
| `debug_world_compare.py` | 对比env.W和新创建World | ✅ 完成 |
| `debug_instability_source.py` | 调查不稳定性来源 | ✅ 完成 |
| `debug_python_uxsim.py` | Python版UXsim对比测试 | ✅ 完成 |

---

## 十、修复方案

### 10.1 推荐方案：固定充电车辆ID集合（方案A）

**思路**：在 `env.reset()` 或 `__load_network()` 时，基于 `self.W.VEHICLES` 构建并存储充电车辆ID集合，后续所有操作都使用这个固定的集合。

**实现要点**：
1. 在 `__load_network()` 完成后，遍历 `self.W.VEHICLES` 构建 `self.charging_vehicle_ids: Set[str]`
2. 在 `__run_simulation()` 中，使用 `self.charging_vehicle_ids` 而非每次从新World获取
3. 流量统计时，只统计 `self.charging_vehicle_ids` 中的车辆

**优点**：
- 不依赖C++扩展层的稳定性
- 一次确定，全程使用
- 改动最小，风险最低

### 10.2 备选方案

**方案B**：修复C++扩展层的release逻辑
- 需要深入检查 `uxsimpp_extended` 的C++代码
- 找出release后影响后续World创建的全局状态
- 工作量大，风险较高

**方案C**：避免释放World资源
- 在UE-DTA迭代中不调用 `release()`
- 可能导致内存持续增长
- 仅作为临时方案

### 10.3 实施优先级

1. ✅ 方案A：固定充电车辆ID集合（推荐，立即实施）
2. ⏸️ 方案B：修复C++扩展层（后续优化）
3. ❌ 方案C：避免释放资源（不推荐）

---

## 十一、根因定位完成（2026-01-22）

### 11.1 根因确认：pybind11动态属性因内存地址复用被继承

通过 `debug_memory_address.py` 实验验证：

```
⚠️ veh_0 的id=2752424613488 与旧 veh_3 相同
   旧marker: world_A_veh_3, 新marker: world_A_veh_3  ← 新Vehicle继承了旧attribute！

❌ veh_4: 期望charging=True, 实际=False  ← veh_4继承了旧veh_7的False值
```

**Vehicle对象id复用数: 7 / 10** —— 70%的新Vehicle复用了旧Vehicle的内存地址。

### 11.2 问题机制

```
1. 创建World A的veh_7，设置attribute={"charging_car": False}
   - C++分配内存地址: 0x280XXXXXX
   - pybind11在内部字典记录: dict[0x280XXXXXX] = {"charging_car": False}

2. release() World A
   - C++执行 delete veh_7，内存地址被释放
   - 但pybind11的属性字典仍保留该条目

3. 创建World B的veh_4
   - C++重新分配内存，恰好得到相同地址（内存复用）
   - Python层设置: veh_4.attribute = {"charging_car": True}
   - 但pybind11发现该地址已有attribute，返回旧值

4. 访问veh_4.attribute时，得到旧veh_7的值 → charging_car错误
```

### 11.3 修复方案

在`World.release()`中先删除动态属性再释放C++资源。

**修改文件**：
- `src/env/uxsimpp_extended/uxsimpp.py` - 添加`World.release()`方法
- `src/env/EVCSChargingGameEnv.py` - 改用`W.release()`

### 11.4 验证结果

| 测试项 | 修复前 | 修复后 |
|--------|--------|--------|
| attribute继承 | ❌ 70%复用致错 | ✅ 全部正确 |
| 充电车辆ID | ❌ 差异6~134个 | ✅ 5次完全一致 |
| UE-DTA流量 | ❌ 丢失50%~72% | ✅ 356/356匹配 |
