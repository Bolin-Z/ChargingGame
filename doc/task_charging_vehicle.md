# SF网络充电车辆数量与Reward分析

## 1. 问题背景

在分析实验结果 `results/mfddpg_experiment_20251119_164139/step_records.json` 时发现以下问题：

1. SF网络的充电车辆数只有理论值的一半
2. 新实验中节点5和12基本没有流量，所有充电站的reward只有三位数（而非五位数）

## 2. 问题1：充电车辆只有理论值的一半

### 2.1 数据对比

| 指标 | 理论值 | 实际值 | 差异 |
|------|--------|--------|------|
| 总车辆数 | 37,501 | 33,005 | -12.0% |
| 充电车辆数 | 3,750 | 1,780 | **-52.5%** |
| 无法生成充电车辆的OD对 | 0 | **848个 (80%)** | - |

### 2.2 根因分析

**根本原因：deltan离散化 + 低充电比例的叠加效应**

当前配置：
- `deltan = 5`（每个Vehicle对象代表5辆实际车辆）
- `charging_car_rate = 0.1`（10%的车辆需要充电）
- `DELTAT = 5s`（UXSim时间步长）

**车辆生成机制**（UXSim的adddemand逻辑）：
```
每时间步累积流量 = flow × DELTAT
当累积 >= deltan 时，创建一个Vehicle对象
```

**问题机制**：
```
每个OD对的充电流量 = 原始流量 × charging_car_rate(0.1)
每时间步累积 = 充电流量 × DELTAT(5s)
需要累积达到 deltan(5) 才能生成一个Vehicle

例如 OD对 2->10:
- 原始流量 q = 0.01 veh/s
- 充电流量 = 0.01 × 0.1 = 0.001 veh/s
- 每步累积 = 0.001 × 5 = 0.005 辆
- 4800秒总累积 = 0.005 × (4800/5) = 4.8 辆 < 5
- 结果：无法生成任何充电Vehicle！
```

**关键发现**：
- 848个OD对（占80%）的充电流量太小，整个发车期累积不足5辆
- 这些OD对理论上应产生的充电车辆被完全丢弃

### 2.3 验证

诊断脚本 `diagnose_paths.py` 模拟UXSim生成逻辑：
- 模拟生成充电车辆数：1780
- 实验数据中的充电车辆数：1780
- **完全吻合**，证实诊断正确

### 2.4 解决方案

这不是bug，是UXSim的设计特性。如需提高充电车辆生成率，可选：

| 方案 | 操作 | 效果 | 副作用 |
|------|------|------|--------|
| A | 增大 `charging_car_rate` (如0.2~0.3) | 每OD对充电流量翻倍 | 改变博弈场景 |
| B | 减小 `deltan` (如1或2) | 降低生成阈值 | 计算时间增加 |
| C | 增大 `demand_multiplier` | 整体流量放大 | 可能导致拥堵 |

**推荐方案**：根据研究目的选择。如果关注真实充电比例，可考虑方案B（减小deltan）。

---

## 3. 问题2：节点5和12无流量，Reward三位数

### 3.1 问题现象

新实验观察到：
- 充电站5和12基本没有充电流量
- 所有充电站的reward都是三位数（之前实验是五位数）
- 完成率是100%

### 3.2 Reward计算公式

```python
reward = Σ (price × flow × charging_demand_per_vehicle)
```

其中：
- `price`: 充电价格（元/kWh），范围[0.5, 2.0]
- `flow`: 充电车辆数
- `charging_demand_per_vehicle = 50` kWh

### 3.3 数量级分析

**五位数Reward的情况**（旧实验）：
- 总充电车辆：1780辆
- 平均每车贡献：50kWh × 1.2元/kWh = 60元
- 总Reward：1780 × 60 ≈ 107,000元
- 4个充电站平均：~27,000元/站

**三位数Reward的情况**（新实验）：
- 假设Reward = 500元/站
- 反推流量 = 500 / (1.2 × 50) ≈ 8辆/站
- 总流量 ≈ 32辆（vs 理论1780辆）

### 3.4 可能原因（待验证）

1. **配置差异**：新实验可能使用了不同的配置参数
2. **价格竞争失衡**：某些站价格过高导致流量集中到其他站
3. **路径选择问题**：UE-DTA收敛到只经过14和18的路径
4. **充电车辆生成问题**：需要检查新实验的`total_charging_vehicles`

### 3.5 待分析

需要新实验的 `step_records.json` 文件来进一步分析：
- [ ] 确认 `total_charging_vehicles` 数量
- [ ] 检查各充电站的 `charging_flows` 分布
- [ ] 对比价格策略与流量分配的关系
- [ ] 验证节点5和12的地理位置是否处于主要通勤路径上

---

## 4. 问题3：uxsimpp_extended的Vehicle命名冲突Bug（已修复 ✓）

### 4.1 问题现象

使用 `uxsimpp_extended` 版本运行实验时：
- 旧实验（uxsim + patch）：充电车辆 **1780辆**
- 新实验（uxsimpp_extended）：充电车辆 **70辆**
- 损失率：**96%**

### 4.2 根因分析

**问题出在 `VehiclesDict` 类的 `items()` 方法与 `values()` 方法行为不一致：**

| 方法 | 返回数量 | 说明 |
|------|----------|------|
| `values()` | 6601 | 直接返回C++ vector所有元素 |
| `items()` | 6112 | 返回字典，同名Vehicle被覆盖 |

**命名冲突机制**：

```python
# uxsimpp_extended/uxsimpp.py 第563行
veh_name = f"{origin}-{destination}-{t}"
```

当同一OD对在同一时间步同时生成充电和非充电Vehicle时：
1. 充电需求先调用 `adddemand`，生成 `1-2-100.0` (charging_car=True)
2. 非充电需求后调用 `adddemand`，生成 `1-2-100.0` (charging_car=False)
3. 两个Vehicle名称相同！

**字典覆盖**：
```python
# VehiclesDict._get_dict() 第62行
self._cached_dict = {v.name: v for v in cpp_vehicles}
# 同名Vehicle后者覆盖前者，充电Vehicle被非充电Vehicle覆盖
```

### 4.3 验证数据

诊断脚本 `check_name_conflict.py` 确认：

| 统计项 | 修复前 | 修复后 |
|--------|--------|--------|
| 总Vehicle数 | 6601 | 6601 |
| 唯一名称数 | 6112 | **6601** ✓ |
| 重复名称数 | 469 | **0** ✓ |
| 被覆盖的充电Vehicle | 342 | **0** ✓ |

### 4.4 修复方案（已实施）

**采用方案A + 性能优化**：在 `World` 类中维护 Python 层计数器，避免跨 Python-C++ 边界访问。

**修改文件**：`src/env/uxsimpp_extended/uxsimpp.py`

**修改内容**：

1. `World.__init__` 添加计数器：
```python
self._vehicle_counter = 0  # Vehicle 命名计数器
```

2. `adddemand` 函数（第566行）：
```python
# 修改前
veh_name = f"{origin}-{destination}-{t}"

# 修改后
veh_name = f"{origin}-{destination}-{t}-{W._vehicle_counter}"
W._vehicle_counter += 1
```

3. `adddemand_predefined_route` 函数（第638行）：同上修改

**性能考虑**：使用 Python 层计数器而非 `len(cpp_world.VEHICLES)`，避免每次创建车辆都跨 Python-C++ 边界访问。

### 4.5 诊断脚本

本次诊断创建的脚本（可删除）：
- `diagnose_charging_issue.py` - 初步诊断
- `diagnose_charging_issue_v2.py` - 完整UE-DTA验证（已更新）
- `diagnose_charging_issue_v3.py` - 检查attribute
- `diagnose_adddemand.py` - 测试adddemand
- `check_deltat.py` - 检查时间步长
- `compare_uxsim.py` - 对比uxsim版本
- `check_uxsim_adddemand.py` - 检查原生adddemand
- `verify_charging_count.py` - 验证充电车辆数
- `check_sim_world.py` - 检查仿真World
- `check_vehicles_iteration.py` - 检查遍历不一致
- `check_name_conflict.py` - 验证命名冲突

---

## 5. 问题4：充电流量异常低（已修复 ✓）

### 5.1 问题现象

命名冲突修复后，运行 `diagnose_charging_issue_v2.py` 发现：

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| 充电Vehicle对象数 | 356 ✓ | 356 ✓ | - |
| 实际充电车辆数 | 1780 ✓ | 1780 ✓ | - |
| UE-DTA充电车辆总数 | 70辆 | **1745辆** | **25x** |
| 充电流量总计 | 60辆 | **1380辆** | **23x** |
| 总收益 | 4375元 | **86250元** | **20x** |

### 5.2 各充电站流量分布（修复后）

```
站点 5:  总流量=255辆
站点 12: 总流量=200辆
站点 14: 总流量=300辆
站点 18: 总流量=625辆
所有站点总充电流量: 1380辆
```

**关键改善**：
- 四个充电站都有流量了（之前12和14没有流量）
- 总收益从三位数恢复到五位数
- 充电车辆完成率 99.7% (1740/1745)

### 5.3 根因分析

**问题出在 `__create_simulation_world()` 方法**：

该方法使用 `Vehicle()` 构造函数直接创建车辆，但构造函数的自动命名逻辑没有使用计数器：

```python
# Vehicle.__new__ 第336行（修复前）
veh_name = name if name else f"{orig}-{dest}-{departure_time}"
```

而 `adddemand` 函数（之前修复过）使用了计数器：

```python
# adddemand（已修复）
veh_name = f"{origin}-{destination}-{t}-{W._vehicle_counter}"
```

**问题链条**：
1. `env.W` 中的车辆通过 `adddemand` 创建 → 名称唯一 → 356个充电Vehicle
2. `__create_simulation_world()` 通过 `Vehicle()` 创建 → 名称冲突 → `items()` 只能访问到45个
3. UE-DTA 遍历使用 `items()` → 大量充电车辆被忽略 → 流量只有60辆

### 5.4 修复方案（已实施）

**修改文件**：`src/env/EVCSChargingGameEnv.py` 第671-674行

```python
# 修改前
Vehicle(W, veh.orig.name, veh.dest.name, veh.departure_time, ...)

# 修改后
unique_name = f"{veh.orig.name}-{veh.dest.name}-{veh.departure_time}-{vehicle_objects_created}"
Vehicle(W, veh.orig.name, veh.dest.name, veh.departure_time, ..., name=unique_name)
```

**设计考虑**：使用局部计数器 `vehicle_objects_created` 而非 `W._vehicle_counter`，确保兼容 uxsimpp_extended 和 UXsim+Patch 两种后端。

### 5.5 诊断脚本

- `diagnose_low_charging_flow.py` - 详细分析车辆完成率和路径分配
- `verify_fix_charging_flow.py` - 验证修复效果

---

## 6. 相关文件

- 诊断脚本：`diagnose_paths.py`
- 旧实验数据：`results/mfddpg_experiment_20251119_164139/step_records.json`
- SF网络配置：`data/siouxfalls/siouxfalls_settings.json`
- 环境实现：`src/env/EVCSChargingGameEnv.py`

## 7. 协作记录

| 日期 | 进展 |
|-----|------|
| 2025-12-30 | 分析旧实验数据，发现充电车辆损失52.5%的问题 |
| 2025-12-30 | 确认根因：deltan离散化 + 低充电比例叠加效应 |
| 2025-12-30 | 创建诊断脚本验证，结果与实验数据完全吻合 |
| 2025-12-30 | 记录问题2（节点5/12无流量），待新实验数据分析 |
| 2026-01-02 | 发现新实验充电车辆只有70辆（vs旧实验1780辆） |
| 2026-01-02 | 排除deltan问题：env.W中确实有356个充电Vehicle |
| 2026-01-02 | 发现VehiclesDict.items()与values()返回数量不一致 |
| 2026-01-02 | **确认根因：Vehicle命名冲突导致342个充电Vehicle被覆盖** |
| 2026-01-02 | 提出3个解决方案，推荐方案A（修改adddemand命名） |
| 2026-01-02 | **实施修复：采用方案A + Python层计数器优化** |
| 2026-01-02 | **验证修复成功：命名冲突从469→0，充电Vehicle从14→356** |
| 2026-01-02 | 发现新问题4：充电流量异常低（60辆 vs 预期1780辆） |
| 2026-01-02 | 更新 `diagnose_charging_issue_v2.py` 为完整UE-DTA验证脚本 |
| 2026-01-02 | 创建 `diagnose_low_charging_flow.py` 详细分析车辆完成率和路径分配 |
| 2026-01-02 | **确认根因：`__create_simulation_world()` 中 Vehicle 命名冲突** |
| 2026-01-02 | **实施修复：在创建 Vehicle 时传入唯一名称（使用局部计数器）** |
| 2026-01-02 | **验证修复成功：充电流量 60→1380辆，总收益 4375→86250元** |
