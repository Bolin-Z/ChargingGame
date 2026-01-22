# Vehicle.attribute 不存在问题诊断记录

**日期**: 2026-01-22
**问题**: 运行训练时出现 `vehicle.attribute` 不存在的 AttributeError

---

## 一、问题现象

在修复了 `task_reward.md` 中记录的流量丢失问题后，运行 `run_single_experiment.py` 进行训练时，出现 `vehicle.attribute` 不存在的错误。

错误发生在 `EVCSChargingGameEnv.py:681`：
```python
for veh in self.W.VEHICLES.values():
    Vehicle(..., attribute=veh.attribute.copy(), ...)  # AttributeError: 'Vehicle' object has no attribute 'attribute'
```

---

## 二、架构背景

### 2.1 Python-C++ 双层架构

uxsimpp_extended 采用 pybind11 实现 Python-C++ 绑定：

```
┌─────────────────────────────────────────────────────────┐
│                    Python 包装层                         │
│  World (Python class)                                   │
│    └── self._cpp_world  ──────► 真正的 C++ World 对象    │
│    └── self._vehicle_refs ────► 持有 Python Vehicle 引用 │
│    └── self._link_refs ───────► 缓存 Python Link 引用    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    C++ 仿真核心                          │
│  World (C++)                                            │
│    └── vehicles[]  ───► C++ Vehicle 指针                 │
│    └── links[]                                          │
│    └── nodes[]                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 pybind11 动态属性机制

- Vehicle 类使用 `py::dynamic_attr()` 支持 Python 层动态属性（如 `attribute`）
- 动态属性存储在 pybind11 内部的**全局字典**中，key 是 **Python 包装对象的 id**
- 每次从 C++ vector 获取对象时，pybind11 **可能创建新的 Python 包装对象**

### 2.3 当前的引用管理设计

```python
# uxsimpp.py 中的设计意图
class World:
    def __init__(self):
        self._vehicle_refs = []  # 保存所有 Vehicle 的 Python 引用

def adddemand(W, ...):
    veh = _CppVehicle(...)       # 创建 Vehicle
    W._vehicle_refs.append(veh)  # 保存引用
    veh.attribute = {...}        # 设置动态属性
```

---

## 三、根因分析

### 3.1 问题代码路径

```python
# VehiclesDict.values() 的实现 (uxsimpp.py:98-101)
def values(self):
    return list(self._get_cpp_vehicles())  # 直接遍历 C++ vector

def _get_cpp_vehicles(self):
    return self._world._cpp_world.VEHICLES  # 返回 C++ vector
```

### 3.2 问题机制

```
1. adddemand() 创建 Vehicle:
   veh = _CppVehicle(...)           # Python 包装对象 A (id=0x1000)
   W._vehicle_refs.append(veh)      # 保存对象 A
   veh.attribute = {"charging_car": True}  # 在对象 A 上设置 attribute

   pybind11 动态属性字典: {id(A): {"attribute": {...}}}

2. 后续访问 W.VEHICLES.values():
   def values(self):
       return list(self._get_cpp_vehicles())
       # 遍历 C++ vector 时，pybind11 为每个 C++ 指针创建新的 Python 包装对象
       # 返回的是对象 B (id=0x2000)，不是之前的对象 A！

   pybind11 查找 id(B) → 找不到 → attribute 不存在！

3. EVCSChargingGameEnv 访问:
   for veh in self.W.VEHICLES.values():  # veh 是对象 B
       veh.attribute.copy()              # AttributeError!
```

### 3.3 根因总结

**`VehiclesDict.values()` 返回的是从 C++ vector 新创建的 Python 包装对象，而不是 `_vehicle_refs` 中设置了 `attribute` 的原始对象。**

这是因为：
1. pybind11 每次访问 C++ 对象时可能返回不同的 Python 包装对象
2. 动态属性绑定在特定的 Python 包装对象上
3. `VehiclesDict.values()` 没有使用 `_vehicle_refs` 中缓存的对象

---

## 四、设计缺陷

### 4.1 `_vehicle_refs` 未被正确使用

`_vehicle_refs` 的设计意图是保存设置了动态属性的 Python 对象引用，但 `VehiclesDict` 完全绕过了它：

| 方法 | 当前实现 | 问题 |
|------|---------|------|
| `values()` | `list(cpp_vehicles)` | 返回新对象，无 attribute |
| `__getitem__` | `cpp_vehicles[key]` 或 `_get_dict()[key]` | 返回新对象，无 attribute |
| `_get_dict()` | `{v.name: v for v in cpp_vehicles}` | 构建时创建新对象 |

### 4.2 缓存一致性问题

`VehiclesDict` 有自己的缓存 `_cached_dict`，但：
- 缓存的是遍历 C++ vector 时创建的新 Python 对象
- 与 `_vehicle_refs` 中的对象不是同一批
- 导致动态属性丢失

---

## 五、修复方向

### 5.1 方案 A：让 VehiclesDict 使用 _vehicle_refs

```python
def values(self):
    # 返回 _vehicle_refs 而不是遍历 C++ vector
    return list(self._world._vehicle_refs)
```

**优点**：改动小，直接使用已有的正确引用
**缺点**：需要确保 `_vehicle_refs` 顺序与 C++ vector 一致

### 5.2 方案 B：重新设计缓存机制

统一使用 `_vehicle_refs` 作为唯一的 Vehicle Python 对象来源，所有访问都通过它。

### 5.3 方案 C：在 C++ 层存储 attribute

将 `attribute` 作为 C++ Vehicle 的成员变量，避免依赖 Python 动态属性。

**优点**：彻底解决 pybind11 包装对象不稳定的问题
**缺点**：需要修改 C++ 代码，attribute 类型需要固定

---

## 六、完整错误信息

```
(drl) E:\hakurinn\code\ChargingGame>python run_single_experiment.py -a maddpg -s siouxfalls --seed 42
============================================================
[MADDPG_siouxfalls] 启动实验
============================================================
  算法: MADDPG
  场景: siouxfalls
  种子: 42
  最大Episodes: 10
  实时监控: 启用
============================================================
使用设备: cuda
GPU: NVIDIA GeForce RTX 5060
寻找纳什均衡:   0%|          | 0/10 [05:32<?, ?episode/s]

实验出错: 'uxsimpp_extended.trafficppy.Vehicle' object has no attribute 'attribute'
Traceback (most recent call last):
  File "run_single_experiment.py", line 131, in main
    run_experiment(args.algo, args.scenario, args.seed)
  File "run_single_experiment.py", line 109, in run_experiment
    results = trainer.train()
  File "src\trainer\MADDPGTrainer.py", line 150, in train
    converged_in_episode, episode_length = self._run_episode(episode, observations)
  File "src\trainer\MADDPGTrainer.py", line 268, in _run_episode
    next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
  File "src\env\EVCSChargingGameEnv.py", line 226, in step
    charging_flows, ue_info = self.__run_simulation()
  File "src\env\EVCSChargingGameEnv.py", line 1206, in __run_simulation
    stats, new_routes_specified, charging_flows = self.__route_choice_update(...)
  File "src\env\EVCSChargingGameEnv.py", line 1009, in __route_choice_update
    current_cost = self.__calculate_actual_vehicle_cost_and_flow(veh, W, charging_flows)
  File "src\env\EVCSChargingGameEnv.py", line 884, in __calculate_actual_vehicle_cost_and_flow
    if veh.attribute["charging_car"]:
AttributeError: 'uxsimpp_extended.trafficppy.Vehicle' object has no attribute 'attribute'
```

**错误调用链**：
1. `__run_simulation()` 创建临时 World：`W = self.__create_simulation_world()`
2. `__route_choice_update()` 中通过 `W.VEHICLES[veh_id]` 获取车辆
3. `__calculate_actual_vehicle_cost_and_flow()` 访问 `veh.attribute["charging_car"]`
4. 由于 `VehiclesDict` 返回的是新创建的 Python 包装对象，没有 `attribute` 属性

---

## 七、修复方案实施

### 7.1 采用方案 B：重新设计 VehiclesDict

修改 `VehiclesDict` 类，使其完全基于 `_vehicle_refs` 而非 C++ vector：

**修改文件**：`src/env/uxsimpp_extended/uxsimpp.py`

**修改前**：
```python
class VehiclesDict:
    """
    将 C++ vector<Vehicle*> 包装为类字典访问
    ...
    """
    def __init__(self, world):
        self._world = world
        self._cached_dict = None
        self._cache_len = -1
        self._cpp_vehicles_ref = None  # 缓存 C++ vehicles 引用

    def _get_cpp_vehicles(self):
        """获取 C++ vehicles vector"""
        if self._cpp_vehicles_ref is None:
            self._cpp_vehicles_ref = self._world._cpp_world.VEHICLES
        return self._cpp_vehicles_ref

    def _get_dict(self):
        cpp_vehicles = self._get_cpp_vehicles()
        current_len = len(cpp_vehicles)
        if self._cached_dict is None or self._cache_len != current_len:
            self._cached_dict = {v.name: v for v in cpp_vehicles}  # 问题：创建新对象
            self._cache_len = current_len
        return self._cached_dict

    def values(self):
        return list(self._get_cpp_vehicles())  # 问题：返回新对象
```

**修改后**：
```python
class VehiclesDict:
    """
    将 Vehicle 对象包装为类字典访问

    关键设计：使用 _vehicle_refs 作为唯一数据源
    - _vehicle_refs 保存了设置了动态属性（如 attribute）的 Python 包装对象
    - 直接访问 C++ vector 会创建新的 Python 包装对象，丢失动态属性
    - 所有访问都通过 _vehicle_refs，确保动态属性可用
    """
    def __init__(self, world):
        self._world = world
        self._cached_dict = None
        self._cache_len = -1

    def _get_vehicle_refs(self):
        """获取 _vehicle_refs 列表（保存了动态属性的 Python 对象）"""
        return self._world._vehicle_refs

    def _get_dict(self):
        vehicle_refs = self._get_vehicle_refs()
        current_len = len(vehicle_refs)
        if self._cached_dict is None or self._cache_len != current_len:
            self._cached_dict = {v.name: v for v in vehicle_refs}  # 使用 _vehicle_refs
            self._cache_len = current_len
        return self._cached_dict

    def values(self):
        """返回所有 Vehicle 对象（从 _vehicle_refs，保留动态属性）"""
        return list(self._get_vehicle_refs())  # 使用 _vehicle_refs
```

### 7.2 关键修改点

| 方法 | 修改前 | 修改后 |
|------|--------|--------|
| `_get_cpp_vehicles()` | 访问 C++ vector | 删除，改用 `_get_vehicle_refs()` |
| `_get_vehicle_refs()` | 不存在 | 新增，返回 `_vehicle_refs` |
| `_get_dict()` | 遍历 C++ vector | 遍历 `_vehicle_refs` |
| `values()` | `list(cpp_vehicles)` | `list(_vehicle_refs)` |
| `__getitem__(int)` | 访问 C++ vector | 访问 `_vehicle_refs` |
| `__len__()` | `len(cpp_vehicles)` | `len(_vehicle_refs)` |

### 7.3 Link 对象分析

Link 对象已正确使用 `_link_refs` 缓存：

```python
@property
def LINKS(self):
    for cpp_link in self._cpp_world.LINKS:
        if cpp_link.name not in self._link_refs:
            self._link_refs[cpp_link.name] = cpp_link
    return [self._link_refs.get(l.name, l) for l in self._cpp_world.LINKS]

def get_link(self, name):
    if name in self._link_refs:
        return self._link_refs[name]
    link = self._cpp_world.get_link(name)
    if link is not None:
        self._link_refs[name] = link
    return link
```

`addLink()` 通过 `get_link()` 获取 Link 并设置 `attribute`，缓存机制保证后续访问返回同一对象。

---

## 八、待验证

修复后需要验证：
1. `W.VEHICLES.values()` 返回的对象都有 `attribute`
2. `W.VEHICLES[name]` 和 `W.VEHICLES[index]` 返回的对象都有 `attribute`
3. 训练流程能够正常运行
4. 流量统计正确（不再有丢失）
