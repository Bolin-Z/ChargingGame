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
5. **PettingZoo环境实现**: 已完成多智能体环境的核心架构

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

### 🔄 待完善任务
1. **UXSim仿真集成**: `_run_simulation()`方法的具体实现
   - 需要讨论每步仿真的World实例创建策略
   - 确定路径分配和充电流量统计方法
   - 优化仿真性能（考虑代理模型等）

2. **环境验证与测试**: 完整的环境功能测试
3. **MADRL算法集成**: 选择并集成具体的多智能体强化学习算法
4. **超参数调优**: 收敛阈值、学习率等参数优化

### 🔧 关键技术决策记录
- **智能体ID**: 使用充电节点名称("5", "12", "14", "18")
- **动作空间**: 归一化到[0,1]，环境内部映射到价格区间
- **观测空间**: 包含`last_round_all_prices`和`own_charging_flow`
- **仿真方式**: 每step创建新World实例，复用预计算路径集合
- **参数化**: 随机种子、收敛阈值、最大步数作为环境初始化参数
- **max_steps设置**: 
  - 初始开发/调试：1000-5000步
  - 正式求解：10000+步
  - 作为"安全网"防止不收敛，应设置比预期收敛步数大2-3倍
  - 重点通过convergence_threshold控制收敛精度

## 开发注意事项

- 项目全程使用中文注释和变量名
- 默认配置INFO级别日志
- 环境支持充电和非充电车辆流
- 使用环境初始化参数中的随机种子进行可复现的随机初始化