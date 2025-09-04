# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于Python的电动汽车充电站(EVCS)博弈论仿真项目。系统建模用户均衡动态交通分配(UE-DTA)与充电站定价竞争，使用多智能体强化学习。

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主仿真
python main.py

# 运行示例/遗留实现
python example.py
```

## 核心架构

### 主要组件

- **`src/env.py`**: 核心游戏环境 (`EVCSGameEnv` 类)
  - 管理多智能体充电站定价博弈
  - 处理网络加载、路径计算和价格管理
  - 与UXSim集成进行交通仿真
  - 实现MADRL接口：`reset()`, `step()`, `reward()` 方法

- **`main.py`**: 简单入口点，初始化环境
- **`example.py`**: 扩展UXSim的自定义DUE求解器的遗留实现

### 网络数据结构

仿真使用存储在 `siouxfalls/` 中的Sioux Falls网络数据：
- `siouxfalls_nodes.csv`: 带坐标的网络节点
- `siouxfalls_links.csv`: 带容量和速度参数的道路链路
- `siouxfalls_demand.csv`: 起终点对之间的交通需求
- `siouxfalls_settings.json`: 配置文件，包括充电节点和价格边界

### 关键算法

**路径枚举**：
- `_enumerate_k_shortest_routes()`: 非充电车辆的标准k最短路径
- `_enumerate_k_shortest_charge_routes()`: 充电车辆的双层图方法
  - 第0层：充电前，第1层：充电后
  - 充电链路连接各层以建模强制充电行为

**充电站建模**：
- 在选定节点(5, 12, 14, 18)添加虚拟充电链路
- 基于时间的定价周期(7200仿真秒内8个周期)
- 设置中每个充电节点定义价格边界

## 依赖项

- **UXSim**: 交通微观仿真框架 (v1.8.2)
- **NetworkX**: 路径计算的图算法
- **PyTorch**: MADRL的深度学习框架
- **Pandas/NumPy**: 数据操作和分析
- **Matplotlib**: 可视化和绘图

## 配置

环境从 `siouxfalls_settings.json` 读取配置：
- `charging_nodes`: 节点ID映射到[最低价格, 最高价格]边界的字典
- `charging_periods`: 动态定价的时间周期数
- `charging_car_rate`: 需要充电的车辆比例
- `routes_per_od`: 每个OD对计算的替代路径数量

## MADRL博弈环境设计方案

基于 PettingZoo ParallelEnv 和 Gymnasium 环境设计指南的关键设计问题框架：

### 1. What skill should the agent learn? (智能体需要学习的技能)

**核心技能**: 多时段静态定价策略优化
- **博弈理论技能**: 学习在多智能体竞争环境中制定最优的多时段价格组合
- **时段间平衡技能**: 在多个时段之间分配价格，考虑不同时段的需求差异
- **市场响应技能**: 根据交通流量反馈和竞争对手的价格向量调整下轮策略
- **纳什均衡收敛**: 最终目标是所有智能体的价格向量收敛到纳什均衡解

### 2. What information does the agent need? (智能体需要的信息)

**观测空间设计** (PettingZoo ParallelEnv接口):
```python
def observation_space(self, agent):
    return Dict({
        "last_round_all_prices": Box(low=0.0, high=1.0, shape=(self.n_agents, self.n_periods)),
        "own_charging_flow": Box(low=0, high=np.inf, shape=(self.n_periods,))
    })
```

**信息详解**:
- `last_round_all_prices`: 上一轮所有充电站的归一化报价向量 (全局信息，用于了解竞争态势)
- `own_charging_flow`: 自己各时段的充电车辆数量 (个体反馈信息，用于评估定价效果)

**设计原则**:
- **部分可观测**: 只能观测到价格信息，无法直接观测其他agent的内部状态
- **信息对称**: 所有agent都能观测到相同的价格历史信息
- **及时反馈**: 每轮都能获得自己的市场反馈(充电流量)

### 3. What actions can the agent take? (智能体可以采取的行动)

**动作空间设计** (PettingZoo ParallelEnv接口):
```python
def action_space(self, agent):
    return Box(low=0.0, high=1.0, shape=(self.n_periods,))
```

**动作映射机制**:
```python
实际价格 = min_price + 动作值 × (max_price - min_price)
```

**设计特点**:
- **归一化输出**: Agent输出(0,1)区间值，便于神经网络训练和探索噪声添加
- **连续动作空间**: 支持精细的价格调节，符合现实定价场景
- **多维决策**: 每个时段独立定价，体现时间维度的策略复杂性
- **约束映射**: 环境自动将归一化动作映射到配置文件定义的价格边界内

### 4. How do we measure success? (如何衡量成功)

**奖励函数设计**:
```python
reward_i = Σ(时段_j的价格 × 时段_j的充电车辆数量)  # 对于智能体i
```

**成功衡量标准**:
- **个体层面**: 每个agent最大化自己的累积收益
- **系统层面**: 整个系统收敛到纳什均衡状态
- **收敛判定**: 所有agent的报价向量变化小于设定阈值

**奖励设计原则**:
- **即时反馈**: 每轮给出当轮收益作为即时奖励
- **真实映射**: 直接对应现实中的充电站收益计算
- **竞争激励**: agent需要在吸引流量(低价)和获得收益(高价)间平衡

### 5. When should episodes end? (何时结束episode)

**终止条件设计**:
```python
# 收敛终止
terminated = all(||price_vector_i(t) - price_vector_i(t-1)|| < convergence_threshold)

# 截断终止  
truncated = current_step >= max_steps
```

**Episode结构**:
- **一个episode**: 从随机初始化到收敛(或达到最大轮数)的完整博弈过程
- **一个step**: 所有agent同时出价 → UE-DTA计算 → 返回奖励和观测
- **长episode设计**: 有利于学习长期策略和纳什均衡收敛行为

### PettingZoo ParallelEnv 实现架构

**基础继承结构**:
```python
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

class EVCSGameEnv(ParallelEnv):
    metadata = {"name": "evcs_game_v1"}
    
    def __init__(self, network_config_path: str, random_seed: int, 
                 max_steps: int, convergence_threshold: float):
        super().__init__()
        # 智能体管理
        self.possible_agents = list(self._charging_nodes.keys())  # ["5", "12", "14", "18"]
        self.agents = self.possible_agents.copy()
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
```

**核心接口实现**:
```python
def step(self, actions):
    # actions = {"5": np.array([0.3, 0.5, ...]), "12": np.array([0.2, 0.8, ...])}
    # 返回: observations, rewards, terminations, truncations, infos
    
def reset(self, seed=None, options=None):
    # 返回: observations, infos
    
def observation_space(self, agent):
    # 返回单个agent的观测空间
    
def action_space(self, agent):
    # 返回单个agent的动作空间
```

### 环境实现技术细节

**博弈流程**:
1. 环境初始化：使用环境初始化参数中的随机种子随机初始化所有充电站报价
2. 每轮博弈：所有agent同时提交归一化报价向量
3. 价格映射：环境将归一化动作映射到实际价格区间
4. 交通仿真：执行UE-DTA计算获得交通流分配结果
5. 收益计算：基于流量和价格计算各agent收益
6. 状态更新：更新观测信息，判断终止条件

**UE-DTA集成**:
- 每轮运行完整UXSim仿真确保交通流分配准确性
- 每轮创建新World实例，更新充电价格
- 基于车辆traveled_route()方法统计充电流量
- 保持现有双层图路径枚举算法(`_enumerate_k_shortest_charge_routes`)
- 后续优化方向：训练代理模型加速仿真计算

**充电量建模**:
- 每辆车充电量设为固定数值
- 收益计算基于通过车辆数量，非充电时长

**数据集通用化**:
- 支持加载不同网络数据集，不限于Siouxfalls
- 统一配置文件格式：充电节点、价格边界、时段数等
- 抽象化网络数据加载逻辑，提高环境可复用性

### 实现关键点总结

**1. 环境初始化**:
- 继承`ParallelEnv`，设置`possible_agents`和`agents`属性
- 初始化参数：网络配置路径、随机种子、最大步数、收敛阈值
- 预计算路径集合，一次计算重复使用

**2. 状态管理**:
- 维护当前步数和价格历史
- 跟踪上一轮观测信息用于构造当前观测
- 收敛判断基于价格向量变化的L2范数

**3. 仿真执行**:
- 每step创建新World实例确保价格更新
- 复用预计算的路径集合提高效率
- 从车辆轨迹中提取充电站流量统计

**4. 多智能体接口**:
- `step()`接收和返回字典格式的多智能体数据
- `observation_space()`和`action_space()`返回单个agent的空间定义
- 动作归一化映射到实际价格区间

**5. 收敛与终止**:
- `terminated`: 所有agent价格向量收敛
- `truncated`: 达到最大步数限制
- 支持早停提高训练效率

## 当前开发进度与下步计划

### ✅ 已完成任务
1. **需求分析与设计**: 确定了多智能体充电站竞价博弈的MADRL解决方案
2. **框架选择**: 决定使用PettingZoo ParallelEnv替代单智能体Gymnasium
3. **接口设计**: 完成观测空间、动作空间、奖励函数、终止条件设计
4. **技术方案**: 明确UXSim集成方式和关键实现点
5. **PettingZoo环境实现**: ✅ **已完成多智能体环境的完整实现**
6. **UE-DTA仿真集成**: ✅ **已完成基于day-to-day动态均衡的完整实现**

#### 详细实现进度
- ✅ **环境继承结构**: EVCSGameEnv继承ParallelEnv，添加必要依赖
- ✅ **智能体管理**: 实现固定智能体列表和映射关系  
- ✅ **空间定义**: 实现observation_space()和action_space()方法
- ✅ **环境接口**: 实现reset()和step()核心方法
- ✅ **动作映射**: 归一化动作到实际价格区间的转换
- ✅ **价格管理**: 随机初始化和历史记录功能
- ✅ **观测生成**: 价格归一化和观测字典构建
- ✅ **收敛判断**: 基于L2范数的价格收敛检测逻辑
- ✅ **奖励计算**: 基于价格×流量的收益计算框架
- ✅ **随机种子统一**: 环境初始化参数控制所有随机操作
- ✅ **UE-DTA完整实现**: 基于day-to-day动态均衡的仿真集成方案
- ✅ **成本函数实现**: 时间成本+充电成本的综合成本函数
- ✅ **路径选择算法**: 实现SolverDUE风格的路径切换逻辑
- ✅ **充电流量统计**: 正确处理DELTAN platoon大小的流量计算
- ✅ **网络复制机制**: 实现World实例复制以支持迭代仿真
- ✅ **参数配置完整性**: 所有UE-DTA参数正确加载到环境

### 🧪 测试结果与调试进度

#### ✅ 已验证功能
1. **环境基础功能**: reset()和step()方法运行正常
2. **车辆生成**: 正确生成充电和非充电车辆（347/356充电车辆完成行程）
3. **UE-DTA迭代**: Day-to-day算法运行，但第1轮就收敛
4. **路径分配**: 成功初始化182个充电OD对，528个非充电OD对

#### 🐛 发现的问题
1. **充电流量为0**: 347辆完成的充电车辆中，0辆找到充电链路
   - 原因分析：充电车辆路径可能未包含充电链路
   - 双层图设计正确（强制经过充电链路），但路径计算或分配有问题
2. **奖励全为0**: 由于充电流量为0导致所有智能体收益为0
3. **UE收敛过快**: 第1轮就收敛（阈值3.0元可能过高）
4. **未完成车辆多**: 1100/33005车辆未完成行程

#### 🔍 待调试任务
1. **充电路径验证**: 检查充电车辆的路径计算是否正确包含充电链路
2. **路径分配调试**: 验证`_apply_routes_to_vehicles`是否正确分配充电路径
3. **收敛阈值调整**: 降低UE收敛阈值以获得更精确解
4. **仿真时间优化**: 检查未完成车辆是否由于仿真时间不足

### 🎯 后续开发方向
1. **MADRL算法集成**: 选择并集成具体的多智能体强化学习算法
2. **性能优化**: UE-DTA求解效率优化
3. **超参数调优**: 收敛阈值、学习率等参数优化
4. **可视化工具**: 添加训练过程和结果可视化

## UXSim仿真集成详细方案

### 核心设计思路
参考UXSim SolverDUE的day-to-day动态均衡求解方法，实现考虑充电成本的UE-DTA问题求解。

### Day-to-Day动态均衡算法框架

**算法流程**：
1. **初始化**: 为所有车辆随机分配初始路径
2. **迭代求解**: 
   - 执行完整交通仿真
   - 根据实际成本进行路径调整
   - 重复直到收敛或达到最大迭代次数
3. **收敛判断**: 基于平均成本差的收敛标准

**具体实现步骤**：
```python
def _run_simulation(self) -> Dict[str, np.ndarray]:
    # 1. 初始化路径分配（首次）或使用上轮路径分配
    # 2. Day-to-day迭代循环
    for iter in range(ue_max_iterations):
        # 2.1 创建新World实例，应用当前充电价格
        # 2.2 为车辆分配路径（基于上轮决策）
        # 2.3 执行仿真: W.exec_simulation()
        # 2.4 计算所有车辆的实际总成本
        # 2.5 路径选择与切换逻辑
        # 2.6 收敛判断
    # 3. 统计各充电站的流量分布
    # 4. 返回按时段分组的充电流量
```

### 成本函数设计

**充电车辆总成本**：
```python
总成本 = time_value_coefficient × 实际旅行时间 + 充电价格(进入时刻) × charging_demand_per_vehicle
```

**非充电车辆总成本**：
```python
总成本 = time_value_coefficient × 实际旅行时间
```

**关键技术点**：
- **充电价格获取**: 使用车辆进入充电链路时刻的价格
- **时刻确定**: 从`veh.traveled_route()`获取进入充电链路的时间戳
- **时段映射**: `_get_period(t)`将时间映射到价格时段

### 路径选择与切换逻辑

**参考SolverDUE实现**：
```python
for each vehicle:
    current_cost = 计算当前路径实际总成本
    best_cost = current_cost
    best_route = current_route
    
    for each alternative_route:
        alt_cost = 计算替代路径预期总成本
        if alt_cost < best_cost:
            best_cost = alt_cost
            best_route = alternative_route
    
    cost_gap = current_cost - best_cost
    total_cost_gap += cost_gap
    
    # 随机切换机制（有限理性建模）
    if cost_gap > 0 and random.random() < ue_swap_probability:
        routes_specified[vehicle_id] = best_route
    else:
        routes_specified[vehicle_id] = current_route
```

### 收敛判断标准

**收敛条件**：
```python
average_cost_gap = total_cost_gap / len(vehicles)
converged = average_cost_gap < ue_convergence_threshold
```

**逻辑解释**：
- **cost_gap**: 每辆车当前路径与最优路径的成本差
- **average_cost_gap**: 所有车辆平均成本差
- **收敛**: 当平均成本差小于阈值时认为达到准动态均衡

### 配置参数

**新增配置项**：
```json
{
    "time_value_coefficient": 0.005,      // 元/秒 (18元/小时)
    "charging_demand_per_vehicle": 50,    // kWh，每辆充电车固定充电量
    "ue_convergence_threshold": 3.0,      // 元，平均成本差收敛阈值
    "ue_max_iterations": 100,             // UE求解最大迭代次数
    "ue_swap_probability": 0.05           // 路径切换概率，建模有限理性
}
```

### 实现技术要点

**1. 成本计算时机**：
- **当前路径成本**: 基于仿真完成后的实际旅行时间和充电价格
- **替代路径成本**: 基于替代路径的预期旅行时间（`route.actual_travel_time()`）

**2. 充电价格获取**：
```python
# 获取车辆进入充电链路的时刻
route, timestamps = veh.traveled_route()
for i, link in enumerate(route):
    if link.attribute["charging_link"]:
        charging_entry_time = timestamps[i]
        charging_price = self._get_price(charging_entry_time, charging_node)
```

**3. 流量统计**：
- 按充电站和时段统计车辆数量
- 返回`{agent: np.array([period0_flow, period1_flow, ...])}`格式

**4. 性能优化考虑**：
- 复用预计算的路径集合
- 仿真模式设置：关闭不必要的日志和可视化
- 考虑收敛加速技术（如自适应swap_prob）

### 🔧 关键技术决策记录
- **智能体ID**: 使用充电节点名称("5", "12", "14", "18")
- **动作空间**: 归一化到[0,1]，环境内部映射到价格区间
- **观测空间**: 包含`last_round_all_prices`和`own_charging_flow`
- **仿真方式**: 每step创建新World实例，复用预计算路径集合
- **参数化**: 随机种子、收敛阈值、最大步数作为环境初始化参数
- **随机种子管理**: 
  - 环境初始化时传入统一随机种子
  - 所有随机操作(价格初始化、UXSim仿真等)使用同一种子
  - 配置文件不再包含随机种子设置
- **max_steps设置**: 
  - 初始开发/调试：1000-5000步
  - 正式求解：10000+步
  - 作为"安全网"防止不收敛，应设置比预期收敛步数大2-3倍
  - 重点通过convergence_threshold控制收敛精度

## 充电链路设计v2.0 - 节点分离方案

### 设计思路
由于UXSim对环路支持有限，采用**节点分离设计**来实现充电功能：

### 节点分离架构
```
原充电节点 5 → 拆分为：
├─ 5-in  (处理所有进入的交通)
└─ 5-out (处理所有离开的交通)

四条内部链路：
├─ 5-in-bypass-5-out     (非充电旁路：in→out)
├─ 5-out-bypass-5-in     (非充电旁路：out→in)  
├─ 5-in-charging-5-out   (充电链路：in→out)
└─ 5-out-charging-5-in   (充电链路：out→in)
```

### 多状态图路径算法
为确保充电车辆**恰好充电一次**，使用多状态图替代双层图：

**状态定义**：
- `uncharged_节点`: 未充电状态
- `charged_节点`: 已充电状态

**路径构建规则**：
```python
def _enumerate_k_shortest_charge_routes(self, source, target, k):
    """基于多状态图的充电路径枚举"""
    G = nx.DiGraph()
    
    for link in self.W.LINKS:
        start, end = link.start_node.name, link.end_node.name
        
        if "bypass" in link.name:
            # 旁路：状态保持
            G.add_edge(f"uncharged_{start}", f"uncharged_{end}")
            G.add_edge(f"charged_{start}", f"charged_{end}")
            
        elif "charging" in link.name:
            # 充电链路：未充电→已充电（状态转换）
            G.add_edge(f"uncharged_{start}", f"charged_{end}")
            
        else:
            # 普通链路：状态保持
            G.add_edge(f"uncharged_{start}", f"uncharged_{end}")  
            G.add_edge(f"charged_{start}", f"charged_{end}")
    
    # 搜索路径：从uncharged_source到charged_target
    return nx.shortest_simple_paths(G, f"uncharged_{source}", f"charged_{target}")
```

### 技术优势
1. **强制单次充电**：状态转换机制天然保证充电车辆只充电一次
2. **完整连通性**：双向链路保证任意方向交通都能通过
3. **UXSim兼容**：所有链路都是标准有向边，无环路问题
4. **清晰语义**：充电/非充电路径选择明确
5. **易于扩展**：可添加更多状态（如电量等级）

## 节点分离方案实施进度 (2025-09-04)

### 🎯 当前开发阶段：基于节点分离的EVCSChargingGameEnv重构

#### ✅ 第一阶段：网络重构 (已完成 4/4)
1. **✅ 节点分离加载**: 充电节点 → `{name}_i` 和 `{name}_o`
   - 命名规范：`5` → `5_i`, `5_o` 
   - 坐标偏移：为可视化设置不同坐标
   - 普通节点保持原状
   
2. **✅ 链路重定向**: 原始链路自动重定向到分离节点
   - `A → 5` → `A → 5_i` (指向充电节点)
   - `5 → B` → `5_o → B` (来自充电节点)
   - `A → B` → `A → B` (普通链路保持)
   
3. **✅ 内部链路创建**: 每个充电节点创建两条内部链路
   - 旁路链路：`{node}_i_bypass_{node}_o` (`charging_link: False`)
   - 充电链路：`{node}_i_charging_{node}_o` (`charging_link: True`)
   - 使用配置文件参数：`charging_link_length`, `charging_link_free_flow_speed`
   
4. **✅ 交通需求重定向**: OD需求适配分离节点
   - 起点：`origin_o` (从充电节点出口出发)
   - 终点：`destination_i` (到达充电节点入口)
   - 车辆分类：`charging_car: True/False` 按比例分配

#### 🔄 第二阶段：路径算法重构 (进行中 0/4)
5. **⏳ 网络结构验证**: 验证分离网络正确性
6. **⏳ 非充电路径枚举**: 普通链路 + 旁路，排除充电链路
7. **⏳ 多状态图构建**: uncharged/charged 状态转换机制
8. **⏳ 充电路径枚举**: 基于多状态图的k最短路径

#### 🔮 第三阶段：环境接口集成 (待开始 0/14)
9. **⏳ PettingZoo接口**: possible_agents, observation_space, action_space
10. **⏳ 环境管理**: reset, step, 价格管理, MADRL参数
11. **⏳ 仿真集成**: 网络复制, 路径分配, UE-DTA, 流量统计
12. **⏳ 验证测试**: 充电约束, 基础测试, 性能优化

### 📋 当前Todo状态 (4/24 完成)

**✅ 已完成** (4项):
- ✅ 直接节点分离加载
- ✅ 链路直接重定向加载 
- ✅ 内部链路直接创建
- ✅ 交通需求直接重定向

**⏳ 待完成** (20项):
- 网络结构验证 + 路径算法重构 (4项)
- PettingZoo环境接口 (9项) 
- 仿真集成与测试 (7项)

### 🎯 下一步计划
1. **网络结构验证**: 检查分离后的节点数量、链路连通性
2. **路径算法重构**: 实现基于节点分离的多状态图路径枚举
3. **接口集成**: 适配PettingZoo接口到新的网络结构

### 🔧 技术实现要点
- **实施策略**: 策略B - 直接构建分离网络（运行时重构）
- **节点命名**: `{name}_i` (入口) 和 `{name}_o` (出口)
- **链路属性**: 仅使用 `charging_link: True/False` 区分
- **OD重定向**: 起点用 `_o`，终点用 `_i`
- **兼容性**: 新实现位于 `src/EVCSChargingGameEnv.py`

## 开发注意事项

- 项目全程使用中文注释和变量名
- 默认配置INFO级别日志
- 环境支持充电和非充电车辆流
- 使用环境初始化参数中的随机种子进行可复现的随机初始化
- 新实现基于节点分离方案，替代原有双层图方法
- 保持与原有配置文件格式的兼容性