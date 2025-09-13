# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于Python的电动汽车充电站(EVCS)博弈论仿真项目。**核心目标是求解电动汽车充电站价格博弈中的均衡解**。

项目包含两个主要组成部分：
1. **博弈环境实现**：建模用户均衡动态交通分配(UE-DTA)与充电站定价竞争的仿真环境
2. **MADRL算法实现**：基于MADDPG改造的多智能体强化学习算法，用于求解价格均衡

## 🎯 博弈论定义与分类

### 博弈类型

本项目实现的是**单步静态不完全信息空间差异化定价博弈**，具有以下特征：

#### 博弈结构
- **智能体集合**：N = {充电站5, 充电站12, 充电站14, 充电站18}
- **策略空间**：S_i = [0.1, 1.0]^8 （每个充电站在8个时段的价格选择）
- **收益函数**：R_i = Σ(p_i,j × q_i,j)，其中p_i,j为充电站i在时段j的价格，q_i,j为对应的充电流量

#### 关键特征
1. **空间差异化**：每个充电站具有不同的地理位置，形成产品差异化
2. **间接需求响应**：用户基于总成本选择充电路径
   ```
   总成本 = 出行时间成本 + 充电价格
   ```
3. **不完全信息**：智能体只观测到`last_round_all_prices`和`own_charging_flow`
4. **静态同时博弈**：所有充电站同时设定价格策略

#### 博弈分类对比

**vs. 传统Bertrand竞争**：
- ❌ 同质产品价格竞争
- ✅ 空间差异化产品定价竞争
- ❌ 消费者选择最低价格
- ✅ 消费者考虑总成本（价格+出行成本）

**vs. Hotelling空间竞争**：
- ✅ 空间位置差异化
- ✅ 每个厂商具有局部垄断力
- ✅ 消费者考虑运输成本

#### 多轮训练与均衡求解

**重要澄清**：MADRL中的"重复博弈"是**算法求解工具**，不是博弈本身的性质

- **博弈定义**：单次价格设定博弈
- **多轮训练目的**：通过梯度学习逼近纳什均衡
- **每轮reset()作用**：重新初始化同一博弈，尝试找到更好的策略
- **收敛目标**：单步博弈的纳什均衡解

```python
# 伪代码说明
博弈定义 = 单步价格设定博弈
for episode in range(max_episodes):
    # 不是"下一个博弈"，而是对同一博弈的求解尝试
    observations = env.reset()  # 重新初始化同一博弈
    actions = agents.act(observations)  # 尝试策略
    rewards = env.step(actions)  # 计算收益
    agents.learn()  # 更新策略逼近均衡
```

### MADRL算法设计目的

#### 🎯 核心目标

**求解复杂博弈的纳什均衡**：找到4个充电站的最优定价策略组合，使得没有任何充电站能通过单方面改变价格来获得更高收益。

#### 🚫 传统博弈论方法失效原因

1. **需求函数不可解析**：充电流量q_i,j依赖复杂的UE-DTA仿真，无法写出p_i,j → q_i,j的解析表达式

2. **策略空间巨大**：4个智能体 × 8个时段 × 连续价格空间 = 32维连续策略空间

3. **收益函数非凸**：基于交通仿真的收益函数不可微、多峰

4. **耦合复杂性**：每个充电站的收益不仅依赖自己的价格，还受其他3个充电站价格影响

#### 📊 算法预期输出

```python
# 每个充电站的最优策略
optimal_strategy = {
    "充电站5": [p_5_1, p_5_2, ..., p_5_8],   # 8个时段的价格
    "充电站12": [p_12_1, p_12_2, ..., p_12_8],
    "充电站14": [p_14_1, p_14_2, ..., p_14_8], 
    "充电站18": [p_18_1, p_18_2, ..., p_18_8]
}

# 对应的均衡结果
equilibrium_result = {
    "charging_flows": ...,    # 各站充电流量
    "total_revenues": ...,    # 各站收益
    "user_costs": ...,       # 用户总成本
    "system_efficiency": ... # 系统效率指标
}
```

#### 🎯 应用价值

1. **政策制定**：为政府制定充电桩建设和定价监管政策提供依据
2. **商业决策**：为充电站运营商提供最优定价策略
3. **城市规划**：优化充电基础设施的空间布局
4. **理论贡献**：为交通-能源耦合系统的博弈分析提供方法论

**核心意义**：通过AI求解传统数学方法无法处理的复杂博弈均衡问题。

## 📋 MADDPG算法设计Todo列表

### 🧠 网络架构设计
1. **Actor网络结构设计**：MLP vs LSTM（处理时序相关性）
   - ✅ **确定方案：选用MLP**
   - **理由**：静态同时博弈本质 - 一次性输出8个时段价格，等价于8维向量输出，无时序依赖
   - **架构设计**：
     ```python
     class MLPActor(nn.Module):
         def __init__(self, obs_dim, action_dim=8, hidden_sizes=[256, 128]):
             # Input: flatten([last_round_all_prices, own_charging_flow]) 
             # Output: 8个时段价格向量 [0,1]^8
             # 激活函数: Sigmoid确保输出范围[0,1]
     ```

2. **Critic网络整合方式**：简单拼接 vs 注意力机制
   - ✅ **确定方案：简单拼接（首选）+ 注意力机制（对比）**
   - **核心洞察**：空间差异化博弈中，智能体间影响是非对称的（如地理位置相近的充电站竞争更激烈）
   - **设计思考**：显式建模（注意力机制）vs 隐式学习（MLP拼接）哪种方式更好？
   - **理由**：MLP的权重矩阵能隐含地实现智能体重要性差异，学习能力等价但更稳定
   - **创新对比点**：实验对比两种方式在多智能体价格博弈中的效果差异
   - **输入组织方式**：
     ```python
     # 每个智能体都有独立的Critic网络，无需显式身份标识
     critic_input = torch.cat([
         last_round_all_prices.flatten(),    # (32,) - 全局价格历史
         all_charging_flows.flatten(),       # (32,) - 所有智能体流量  
         all_current_actions.flatten()       # (32,) - 所有智能体动作
     ], dim=0)  # 总计96维
     ```
   
   - **方案A：简单拼接架构**：
     ```python
     class SimpleCritic(nn.Module):
         def __init__(self, input_dim=96, hidden_sizes=[512, 256]):
             self.mlp = nn.Sequential(
                 nn.Linear(96, 512), nn.ReLU(),
                 nn.Linear(512, 256), nn.ReLU(), 
                 nn.Linear(256, 1)  # 输出对应智能体的Q值
             )
     ```
   
   - **方案B：注意力机制架构**：
     ```python
     class AttentionCritic(nn.Module):
         def __init__(self):
             self.agent_encoder = nn.Linear(24, 64)  # 编码每个智能体
             self.attention = nn.MultiheadAttention(64, num_heads=4)
             self.global_encoder = nn.Linear(32, 64)
             self.q_head = nn.Linear(128, 1)
         
         def forward(self, global_info, agent_infos):
             # 编码所有智能体特征 (4, 64)
             agent_features = torch.stack([
                 self.agent_encoder(info) for info in agent_infos
             ])
             # 注意力机制自动学习智能体间非对称影响权重
             attended, weights = self.attention(agent_features, agent_features, agent_features)
             attended_global = attended.mean(dim=0)
             # 整合全局和注意力特征
             global_feature = self.global_encoder(global_info)
             combined = torch.cat([attended_global, global_feature])
             return self.q_head(combined)
     ```

3. **网络层数和隐藏单元数**：平衡表达能力和训练效率
   - ✅ **确定方案：小网络起步，渐进优化策略**
   - **设计原则**：从简单网络开始验证可行性，避免过拟合和过度复杂化
   - **基于参考实现分析**：参考MADDPG (3智能体，输入8-10维，隐藏64维) vs 我们的场景 (4智能体，输入40/96维)
   - **起始架构配置**：
     ```python
     class Actor(nn.Module):
         def __init__(self, obs_dim=40, action_dim=8):
             self.net = nn.Sequential(
                 nn.Linear(40, 64),    # 1.6倍扩展，合理表达空间
                 nn.ReLU(),
                 nn.Linear(64, 64),    # 等宽保持，参考实现风格
                 nn.ReLU(),
                 nn.Linear(64, 8),     # 输出8个时段价格
                 nn.Sigmoid()          # 确保[0,1]范围
             )
             
     class Critic(nn.Module):
         def __init__(self, input_dim=96):
             self.net = nn.Sequential(
                 nn.Linear(96, 128),   # 1.33倍扩展，处理复杂多智能体输入
                 nn.ReLU(),
                 nn.Linear(128, 64),   # 收敛到参考实现规模
                 nn.ReLU(),
                 nn.Linear(64, 1)      # Q值输出
             )
     ```
   - **参数量评估**：
     - Actor: ≈6.4K参数 (40×64 + 64×64 + 64×8)
     - Critic: ≈20.8K参数 (96×128 + 128×64 + 64×1)
     - 单智能体: ≈27K，4智能体总计≈216K（含target网络）
   - **渐进优化路径**：表达能力不足时可升级到更深/更宽网络

### 🎯 训练策略设计
4. **噪音探索策略**：Gaussian噪音 vs OU噪音 vs 参数空间噪音
   - ✅ **确定方案：Gaussian噪音 + 指数衰减**
   - **理由**：静态博弈特征 - 8个时段价格独立决策，无需时序相关噪音
   - **参数设置**：
     ```python
     class GaussianNoise:
         def __init__(self, action_dim=8, sigma=0.2, sigma_decay=0.9995, min_sigma=0.01):
             self.sigma = sigma          # 初始探索强度20%
             self.sigma_decay = sigma_decay  # 指数衰减率
             self.min_sigma = min_sigma      # 保持最小探索1%
         
         def __call__(self, action):
             noise = np.random.normal(0, self.sigma, self.action_dim)
             self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)
             return np.clip(action + noise, 0.0, 1.0)
     ```
   - **设计优势**：简单高效、各时段独立探索、逐步收敛到稳定策略
   - **参考启发**：Boyuai实现使用explore参数模式切换，可作为未来优化方向

5. **奖励工程**：原始收益 vs 归一化 vs 基准比较
   - ✅ **确定方案：正仿射变换（当轮最大值归一化）**
   - **理论依据**：正仿射变换保持纳什均衡不变，确保博弈等价性
   - **实现方式**：
     ```python
     def normalize_rewards(self, raw_rewards):
         """使用当轮最大值进行正仿射变换"""
         current_max = max(raw_rewards.values())
         if current_max > 0:
             return {agent: reward / current_max for agent, reward in raw_rewards.items()}
         else:
             return {agent: 0.0 for agent in raw_rewards.keys()}
     
     # 实际数据示例：
     # raw_rewards = {'5': 18067.83, '12': 13634.79, '14': 21977.31, '18': 52348.59}
     # normalized = {'5': 0.345, '12': 0.260, '14': 0.420, '18': 1.000}
     ```
   - **设计优势**：保持真实博弈结构、数值友好[0,1]范围、实现简单
   - **适用性**：完美匹配单步静态博弈的纳什均衡求解需求

6. **学习率调度**：固定 vs 衰减 vs 自适应
   - ✅ **确定方案：固定学习率**
   - **参考依据**：Boyuai MADDPG实现使用固定学习率策略验证可行性
   - **参数设置**：
     ```python
     # 基于参考实现的经验，适配我们的奖励范围[0,1]
     actor_lr = 0.001     # Actor网络学习率
     critic_lr = 0.001    # Critic网络学习率
     
     self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
     ```
   - **设计理由**：单步博弈收敛后策略相对稳定，固定学习率简单有效
   - **优化考虑**：相比参考实现的0.01，选择0.001确保稳定收敛到纳什均衡

### 🔧 稳定性增强设计
7. **多次仿真平均**：是否对每个动作进行多次UE-DTA仿真求平均
   - ✅ **确定方案：单次仿真**
   - **实现方式**：
     ```python
     def step(self, actions):
         # 设置价格
         self._set_agent_prices(actions)
         # 单次UE-DTA仿真
         rewards = self._run_single_simulation()
         return observations, rewards, done, info
     ```
   - **设计理由**：计算效率优先，避免训练时间成倍增长
   - **风险控制**：通过固定随机种子确保仿真可复现性
   - **优化路径**：如发现训练不稳定，可后续实现多次平均作为可选功能

8. **经验回放改进**：标准回放 vs 优先级回放 vs 多样性采样
   - ✅ **确定方案：标准经验回放（单Episode版本）**
   - **设计策略**：专注于单episode内的均衡收敛，而非跨episode泛化
   - **实现方式**：
     ```python
     class ReplayBuffer:
         def __init__(self, capacity=10000):  # 存储单episode的经验
             self.buffer = deque(maxlen=capacity)
         
         def reset_episode(self):
             # 可选：每个episode开始时清空或保留少量经验
             pass
             
         def sample(self, batch_size):
             return random.sample(self.buffer, batch_size)
     
     # 参数设置
     buffer_size = 10000       # 单episode经验容量
     batch_size = 64           # 固定批次大小，与最小缓冲区匹配
     min_buffer_size = 64      # 开始学习的最小经验数（尽早开始学习）
     ```
   - **核心理念**：一旦在单episode中收敛到纳什均衡即达成项目目标
   - **扩展路径**：如需验证均衡稳定性，可后续实现跨episode版本

9. **目标网络更新**：软更新频率和tau参数选择
   - ✅ **确定方案：软更新 tau=0.01**
   - **实现方式**：
     ```python
     def soft_update(target_net, main_net, tau=0.01):
         """每步进行软更新"""
         for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
             target_param.data.copy_(
                 tau * main_param.data + (1.0 - tau) * target_param.data
             )
     
     # 每次学习时调用
     def learn(self):
         # ... 网络训练代码 ...
         
         # 更新目标网络
         soft_update(self.actor_target, self.actor_main, tau=0.01)
         soft_update(self.critic_target, self.critic_main, tau=0.01)
     ```
   - **设计优势**：平滑更新保证训练稳定性，适合纳什均衡的精确收敛
   - **参数选择**：tau=0.01为MADDPG经典参数，平衡更新速度与稳定性

### 📊 收敛判断设计
10. **均衡检测指标**：价格变化阈值 vs 收益稳定性 vs 策略梯度范数
    - ✅ **确定方案：相对变化率价格阈值**
    - **核心理念**：单步静态博弈中，纳什均衡表现为价格策略稳定
    - **设计理由**：相对变化率能自适应不同价格水平，比L2范数更科学
    - **实现方式**：
      ```python
      def __check_convergence(self) -> bool:
          """检查价格是否收敛（基于相对变化率）"""
          if len(self.price_history) < 2:
              return False
          
          current_prices = self.price_history[-1]
          previous_prices = self.price_history[-2]
          
          # 使用相对变化率，避免除零
          relative_changes = np.abs(current_prices - previous_prices) / (previous_prices + 1e-8)
          avg_relative_change = np.mean(relative_changes)
          
          converged = avg_relative_change < self.convergence_threshold
          return converged
      ```
    - **优势对比**：
      - ❌ L2范数：对价格矩阵维度敏感，不同价格水平下阈值难设定
      - ❌ 收益稳定性：受UE-DTA仿真随机性影响，可能误判
      - ❌ 策略梯度范数：实现复杂，需要访问网络内部信息
      - ✅ 相对变化率：维度无关、自适应、实现简单、理论清晰

11. **训练终止条件**：固定episodes vs 动态收敛检测
    - ✅ **确定方案：仅需环境层收敛检测**
    - **核心洞察**：单步静态博弈求解不需要MADRL算法层收敛检测
    - **理论依据**：
      ```python
      # 每个episode都是同一博弈的求解尝试
      博弈定义 = 单步价格设定博弈  # 固定不变
      for episode in range(max_episodes):
          observations = env.reset()    # 重新初始化同一博弈
          actions = agents.act(obs)     # 尝试策略
          rewards = env.step(actions)   # 计算收益
          if env.converged:             # 环境内收敛 = 找到纳什均衡
              break                     # 任务完成
          agents.learn()                # 学习如何更快找到均衡
      ```
    - **终止条件设计**：
      - **环境内收敛**：`avg_relative_change < convergence_threshold` → 找到纳什均衡
      - **Episode截断**：`current_step >= max_steps` → 继续下个episode
      - **MADRL评估**：不是"是否收敛"，而是"平均多少episodes找到均衡"
    - **实现位置**：
      ```python
      # 环境层（EVCSChargingGameEnv.step()）
      terminated = self.__check_convergence()  # 价格博弈均衡检测
      truncated = self.current_step >= self.max_steps  # 单episode截断
      
      # MADRL层：无需收敛检测，专注策略学习
      ```

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主仿真
python main.py

# 运行技术验证测试
python src/technical_validation.py
```

## 核心架构与文件结构

### 主要组件

- **`src/EVCSChargingGameEnv.py`**: **v3.0核心环境实现** ✅ **已完成**
  - 基于v2框架 + 自环充电链路 + 预定路径的最终设计方案
  - 完整PettingZoo ParallelEnv接口
  - 集成Monkey Patch增强的UXSim + 多状态图路径算法
  - 包含完整的UE-DTA仿真循环和奖励计算系统

- **`src/patch.py`**: **UXSim Monkey Patch模块** ✅ **已完成**
  - 统一的UXSim增强补丁，解决所有UXSim相关问题
  - Analyzer文件夹创建问题修复
  - Vehicle预定路径功能增强（转移确认机制）
  - World增强功能（addVehicle和adddemand方法）

- **`src/EVCSChargingGameEnv_v1.py`**: v1.0版本实现
  - 基于充电链路复制的初始方案
  - 完整的UE-DTA和多智能体接口
  - 已验证的day-to-day动态均衡算法

- **`src/EVCSChargingGameEnv_v2.py`**: v2.0版本实现
  - 基于节点分离的完整框架设计
  - v3.0继承其PettingZoo接口和UE-DTA算法框架
  - 提供v3.0的整体结构基础

- **`src/technical_validation.py`**: 技术验证脚本
  - PredefinedRouteVehicle和PredefinedRouteWorld类实现
  - UXSim自环链路兼容性验证
  - 路径执行完整性测试

- **`main.py`**: 简单入口点，初始化环境

### 网络数据结构

仿真使用存储在 `siouxfalls/` 中的Sioux Falls网络数据：
- `siouxfalls_nodes.csv`: 带坐标的网络节点
- `siouxfalls_links.csv`: 带容量和速度参数的道路链路
- `siouxfalls_demand.csv`: 起终点对之间的交通需求
- `siouxfalls_settings.json`: 配置文件，包括充电节点和价格边界

## EVCSChargingGameEnv v3.0 设计方案

### 🎯 核心设计理念

**整体架构**: 继承v2框架（PettingZoo接口 + UE-DTA算法 + 类结构）
**充电建模**: 自环充电链路 + 多状态图路径算法 + Monkey Patch增强UXSim严格执行

**设计原则**:
1. **框架继承**: 采用v2版本的成熟PettingZoo接口和UE-DTA框架
2. **拓扑简化**: 使用自环充电链路，避免节点分离的网络复杂性
3. **路径精确控制**: 多状态图确保充电车辆恰好充电一次
4. **执行保证**: Monkey Patch增强的Vehicle确保严格按预定路径行驶
5. **车辆数统计**: deltan概念确保所有统计显示实际车辆数量

### 🚗 关键概念：deltan（车辆批次大小）

**deltan**是UXSim中的关键参数，定义了每个Vehicle对象代表多少辆实际车辆。这是一个"platoon"（车队）概念：

- **定义**: `deltan` = 每个Vehicle对象代表的实际车辆数量
- **作用**: 减少仿真计算量，提高大规模交通网络仿真效率
- **统计影响**: 所有车辆数统计都需要乘以deltan才是实际车辆数量

**v3.0中的deltan处理**:
- ✅ 所有显示的车辆数统计都是实际车辆数（Vehicle对象数 × deltan）
- ✅ tqdm进度条显示实际车辆数
- ✅ 收敛/未收敛消息显示实际车辆数
- ✅ 最终统计表格显示实际车辆数
- ✅ 充电流量统计考虑deltan影响

**示例**:
```python
# 如果创建了100个Vehicle对象，deltan=5
vehicle_objects = 100
deltan = 5
actual_vehicles = vehicle_objects * deltan  # 500辆实际车辆

# v3.0中所有地方显示的都是500，而不是100
```

### 🧪 技术验证结果

#### ✅ UXSim自环链路完全兼容
基于 `src/technical_validation.py` 的验证测试表明：

```python
# 测试路径（包含自环A-A）
predefined_route = ["A-B", "B-A", "A-A", "A-B", "B-A", "A-C"]

# 验证结果
Actual route:     ['A-B', 'B-A', 'A-A', 'A-B', 'B-A', 'A-C'] 
Predefined route: ['A-B', 'B-A', 'A-A', 'A-B', 'B-A', 'A-C']
Completion progress: 100.0%
Unfinished vehicles: 0
```

#### ✅ PredefinedRouteVehicle关键特性
1. **100%路径匹配**: 实际行驶路径与预定路径完全一致
2. **自环支持**: UXSim原生支持自环链路（X-X格式）
3. **完成率保证**: 所有车辆按预定路径完成行程
4. **路径灵活性**: 支持任意复杂的路径序列，包括自环

### 🏗️ v3.0架构设计

#### 1. 自环充电链路创建
```python
# 为每个充电节点创建自环充电链路
def _create_charging_links(self):
    for charging_node_idx in self._charging_nodes.keys():
        charging_link_name = f"charging_{charging_node_idx}"
        self.W.addLink(
            name=charging_link_name,
            start_node=charging_node_idx,
            end_node=charging_node_idx,  # 自环：起点=终点
            length=self._charging_link_length,
            free_flow_speed=self._charging_link_free_flow_speed,
            attribute={"charging_link": True}
        )
```

#### 2. 多状态图路径算法
```python
def _enumerate_k_shortest_charge_routes(self, source, target, k):
    """基于多状态图的充电路径枚举，确保恰好一次充电"""
    G = nx.DiGraph()
    
    # 构建多状态图
    for link in self.W.LINKS:
        start, end = link.start_node.name, link.end_node.name
        
        if link.attribute.get("charging_link", False):
            # 自环充电链路：状态转换 uncharged -> charged
            node_idx = start  # 自环起点=终点
            G.add_edge(f"uncharged_{node_idx}", f"charged_{node_idx}", 
                      weight=link.length/link.u, link_name=link.name)
        else:
            # 普通链路：状态保持
            G.add_edge(f"uncharged_{start}", f"uncharged_{end}", 
                      weight=link.length/link.u, link_name=link.name)
            G.add_edge(f"charged_{start}", f"charged_{end}", 
                      weight=link.length/link.u, link_name=link.name)
    
    # 路径搜索：uncharged_source -> charged_target
    paths = nx.shortest_simple_paths(G, f"uncharged_{source}", f"charged_{target}", weight='weight')
    
    # 转换为实际链路序列，供PredefinedRouteVehicle使用
    routes = []
    for path in islice(paths, k):
        route = [G[path[i]][path[i+1]]['link_name'] for i in range(len(path)-1)]
        routes.append(route)
    
    return routes
```

#### 3. PredefinedRouteVehicle集成
```python
# 继承v2框架，集成预定路径车辆
class EVCSGameEnv(ParallelEnv):  # 继承v2的基础结构
    def _create_simulation_world(self):
        """创建支持预定路径的仿真世界"""
        W = PredefinedRouteWorld(name="EVCS_Simulation", **simulation_params)
        
        # 复制网络结构（包括自环充电链路）
        self._copy_network_to_world(W)
        
        # 为每辆车分配预定路径
        for vehicle_id, route_links in self.routes_specified.items():
            veh = W.addVehicle(
                predefined_route=route_links,  # 完整链路序列，包含自环charging_links
                departure_time=self.departure_times[vehicle_id],
                name=vehicle_id
            )
```

#### 4. 充电行为物理建模
```python
# 通过自环链路参数控制充电时间（仅使用settings.json中配置的参数）
charging_link_parameters = {
    "length": 3000,              # 3km，来自settings.json的charging_link_length
    "free_flow_speed": 10,       # 10m/s，来自settings.json的charging_link_free_flow_speed
}
```

### 🎮 PettingZoo环境接口设计（继承v2框架）

#### 观测空间
```python
def observation_space(self, agent):
    return spaces.Dict({
        "last_round_all_prices": spaces.Box(low=0.0, high=1.0, shape=(self.n_agents, self.n_periods)),
        "own_charging_flow": spaces.Box(low=0, high=np.inf, shape=(self.n_periods,))
    })
```

#### 动作空间
```python
def action_space(self, agent):
    return spaces.Box(low=0.0, high=1.0, shape=(self.n_periods,))
```

#### 奖励函数
```python
reward_i = Σ(时段_j的价格 × 时段_j的充电车辆数量)  # 对于智能体i
```

### 📊 Day-to-Day动态均衡算法（继承v2框架）

#### UE-DTA求解流程
1. **路径预计算**: 使用多状态图为所有OD对预计算k条充电/非充电路径
2. **路径初始化**: 为车辆分配初始预定路径（充电车辆确保包含恰好一条自环充电链路）
3. **仿真执行**: 使用PredefinedRouteVehicle严格按预定路径仿真
4. **成本计算**: 基于实际旅行时间和充电价格计算总成本
5. **路径切换**: 根据成本差异和随机概率进行路径调整
6. **收敛判断**: 基于平均成本差判断是否达到均衡

#### 成本函数
```python
# 充电车辆（检测到自环charging_链路）
总成本 = time_value_coefficient × 实际旅行时间 + 充电价格 × charging_demand_per_vehicle

# 非充电车辆  
总成本 = time_value_coefficient × 实际旅行时间
```

## 🏃 v3.0当前实现状态

### ✅ 已完成的核心功能（截至2025年1月）

**✅ 完整的v3.0实现（Phase 1-8全部完成）：**

**PettingZoo环境接口层（Phase 4）：**
- ✅ 价格初始化系统：支持随机和中点初始化模式
- ✅ 观测空间实现：last_round_all_prices + own_charging_flow
- ✅ 动作空间实现：[0,1]归一化价格动作
- ✅ 价格归一化：实际价格 ↔ [0,1]区间映射  
- ✅ 环境生命周期：reset()和step()方法完整实现
- ✅ 收敛检测：基于L2范数的价格变化检测

**网络和路径基础设施（Phase 1-3）：**
- ✅ 自环充电链路创建：charging_{node_id} (node→node)
- ✅ 多状态图路径算法：uncharged/charged状态转换
- ✅ Monkey Patch Vehicle增强：技术验证完成
- ✅ 充电/非充电路径枚举：k最短路径计算

**路径分配与车辆管理（Phase 5）：**
- ✅ __create_simulation_world：创建支持预定路径的仿真世界
- ✅ __initialize_routes：车辆分类与初始路径分配
- ✅ __apply_routes_to_vehicles：增强Vehicle车辆创建

**UE-DTA仿真集成（Phase 6）：**
- ✅ __run_simulation：完整UE-DTA仿真循环集成
- ✅ __route_choice_update：路径选择更新和统计
- ✅ __estimate_route_cost：路径成本估算
- ✅ __calculate_actual_vehicle_cost_and_flow：成本计算和充电流量统计

**充电流量统计与奖励计算（Phase 7）：**
- ✅ __calculate_rewards：基于价格×流量的奖励计算
- ✅ 自环充电链路检测：从traveled_route识别充电行为
- ✅ 按时段统计各充电站流量：支持deltan车辆数量概念

**用户体验优化（Phase 8）：**
- ✅ tqdm进度条：详细的UE-DTA收敛过程显示
- ✅ 收敛/未收敛消息：使用pbar.write()显示详细统计
- ✅ 最终统计表格：Texttable显示实际车辆数量统计
- ✅ 实际车辆数统计：所有显示均为Vehicle对象数×deltan

### 🎉 v3.0实现完成！

EVCSChargingGameEnv v3.0已完全实现，包括：
- 完整PettingZoo ParallelEnv接口
- 自环充电链路建模
- 多状态图路径算法
- Monkey Patch增强Vehicle严格路径执行
- UE-DTA动态均衡仿真
- 充电流量统计和奖励计算
- 详细的进度显示和统计信息

## 🎉 Monkey Patch重构完成！

### ✅ 重构完成状态（2025年1月更新）

**核心重构**：成功将基于继承的`PredefinedRouteVehicle`和`PredefinedRouteWorld`实现重构为**Monkey Patch**方式，统一解决了所有UXSim相关问题。

### ✅ 已完成的重构任务

**Phase 1: 统一UXSim补丁设计** ✅
- ✅ 创建统一的`patch_uxsim()`方法在`src/patch.py`
- ✅ 补丁应用时机：在`EVCSChargingGameEnv.__init__()`中调用
- ✅ 分析并实现所有必要的UXSim组件补丁

**Phase 2: Analyzer文件夹问题修复** ✅
- ✅ Monkey patch `uxsim.Analyzer.__init__()`方法
- ✅ 跳过`os.makedirs(f"out{s.W.name}", exist_ok=True)`调用
- ✅ 保留必要属性并确保兼容性

**Phase 3: 预定路径功能补丁** ✅
- ✅ Monkey patch `uxsim.Vehicle.__init__()`添加预定路径支持
- ✅ 新增`assign_route(route_names)`方法
- ✅ 实现转移确认机制的`route_next_link_choice()`补丁
- ✅ 完善预定路径状态管理的`update()`补丁

**Phase 4: World类增强** ✅
- ✅ 新增`addVehicle()`方法支持预定路径车辆创建
- ✅ 重写`adddemand()`方法使用增强的Vehicle

**Phase 5: 清理原有实现** ✅
- ✅ 删除`PredefinedRouteVehicle`和`PredefinedRouteWorld`类定义（约200行代码）
- ✅ 更新所有调用代码使用标准`uxsim.World`和`uxsim.Vehicle`

**Phase 6: 测试验证** ✅
- ✅ 验证补丁后的环境初始化正常
- ✅ 测试预定路径车辆严格按路径行驶
- ✅ 确认转移确认机制解决自环问题
- ✅ 验证不再创建输出文件夹
- ✅ 多轮reset-step循环测试通过

### 🎯 重构效果

**代码简化**：
- ✅ 删除了约200行的继承类代码
- ✅ 统一的补丁管理，维护更简单

**功能统一**：
- ✅ 在`src/patch.py`中统一解决所有UXSim相关问题
- ✅ 更透明的实现，直接使用标准`uxsim.World`和`uxsim.Vehicle`

**性能提升**：
- ✅ 避免文件夹创建的I/O开销
- ✅ 更轻量的对象创建过程

**架构改进**：
- ✅ 模块化设计：补丁功能独立于主要环境代码
- ✅ 运行时增强：通过Monkey Patch而非继承实现功能扩展
- ✅ 维护便利：所有UXSim增强集中在单个文件中

## 依赖项

- **UXSim**: 交通微观仿真框架 (v1.8.2)
- **NetworkX**: 路径计算的图算法
- **PettingZoo**: 多智能体强化学习环境框架
- **Gymnasium**: 环境接口标准
- **NumPy/Pandas**: 数据操作和分析
- **PyTorch**: MADRL的深度学习框架
- **Matplotlib**: 可视化和绘图

## 配置文件参数

环境从 `siouxfalls_settings.json` 读取配置：

```json
{
    // 基础博弈配置
    "charging_nodes": {"5": [0.1, 1.0], "12": [0.1, 1.0], "14": [0.1, 1.0], "18": [0.1, 1.0]},
    "charging_periods": 8,
    "charging_car_rate": 0.3,
    "routes_per_od": 5,
    
    // 自环充电链路物理参数
    "charging_link_length": 3000,
    "charging_link_free_flow_speed": 10,
    
    // UE-DTA求解参数
    "time_value_coefficient": 0.005,
    "charging_demand_per_vehicle": 50,
    "ue_convergence_threshold": 1.0,
    "ue_max_iterations": 100,
    "ue_swap_probability": 0.05,
    
    // 仿真控制参数
    "simulation_time": 7200,
    "deltan": 5  // Platoon大小：每个Vehicle对象代表的实际车辆数
}
```

### 重要概念说明

**deltan（Platoon概念）**：
- UXsim中的关键参数，定义了每个Vehicle对象代表多少辆实际车辆
- 例如deltan=5时，一个Vehicle对象代表5辆实际车辆
- 在统计车辆数量、充电流量时需要乘以deltan获得实际数值
- 影响仿真精度与计算效率的平衡
```

## 关键技术决策记录

### v3.0 vs v1.0/v2.0 设计对比

| 方面 | v1.0 (充电链路复制) | v2.0 (节点分离) | v3.0 (v2框架+自环+预定路径) |
|------|-------------------|----------------|------------------------------|
| **整体架构** | 基础实现 | 完整PettingZoo框架 | 继承v2框架 |
| **充电链路** | 复制现有链路为charging_版本 | 内部链路连接分离节点 | 真正自环：idx->idx |
| **路径控制** | 依赖UXSim路径选择 | 多状态图路径控制 | 多状态图+Monkey Patch增强Vehicle |
| **网络拓扑** | 链路数翻倍 | 节点数翻倍+内部链路 | 仅增加自环，最简洁 |
| **充电保证** | 双层图概率保证 | 强制恰好一次 | 预定路径100%保证 |
| **实现复杂度** | 中等 | 高 | 中等（继承v2框架） |

### v3.0核心优势

1. **架构成熟**: 继承v2的完整PettingZoo接口和UE-DTA框架，避免重复开发
2. **拓扑最简**: 自环充电链路是所有方案中网络拓扑最简洁的
3. **控制精确**: 多状态图+预定路径双重保证充电行为的确定性
4. **扩展便利**: 基于v2框架，易于后续功能扩展和算法集成
5. **验证充分**: Monkey Patch增强Vehicle已通过技术验证，确保路径执行可靠性

### 随机种子管理策略

- **环境初始化**: 传入统一random_seed参数控制所有随机操作
- **价格初始化**: 使用种子控制初始价格生成
- **UXSim仿真**: 每轮设置相同种子确保可复现性
- **路径切换**: UE算法中的随机切换使用种子控制

## 开发注意事项

- 项目全程使用中文注释和变量名
- 默认配置INFO级别日志，关键步骤输出调试信息
- 环境支持充电和非充电车辆混合仿真
- 使用环境初始化参数中的随机种子进行完全可复现的仿真
- v3.0实现位于 `src/EVCSChargingGameEnv.py`，使用`src/patch.py`提供UXSim增强，保持与配置文件格式完全兼容
- 严格遵循PettingZoo ParallelEnv接口规范，确保MADRL算法兼容性
- 充电链路命名规范：`charging_{node_idx}` 用于自环链路 `{node_idx} -> {node_idx}`

## ✅ 预定路径Vehicle问题已解决（2025年1月更新）

### 🎯 问题解决状态

**核心问题**：预定路径Vehicle的路径执行不准确问题已通过实施**转移确认机制**完全解决，现在通过Monkey Patch方式提供该功能。

### 🚀 最终解决方案：转移确认机制

基于对UXSim内部转移机制的深度理解，成功实施了最优解决方案：

#### 核心思想
通过比较`route_next_link`与当前`link`来确认转移是否成功，只有确认转移成功后才递增`route_index`

#### 实现关键（已在src/patch.py中实现）
```python
def route_next_link_choice(s):
    """
    选择下一个链路 - 使用转移确认机制确保严格按照预定路径执行
    
    核心思想：通过比较route_next_link与当前link来确认转移是否成功，
    只有确认转移成功后才递增route_index
    """

    # 步骤1：检查转移确认
    if (s.route_next_link is not None and 
        s.link is not None and 
        s.route_next_link == s.link):
        # 当前路径已完成, 递增 route_index
        s.route_index += 1
    
    # 步骤2：选择下一个目标链路
    if s.route_index >= len(s.predefined_route_links):
        # 路径完成, 设置route_next_link为None
        s.route_next_link = None
        return
    
    # 设置下一个预定链路作为目标
    s.route_next_link = s.predefined_route_links[s.route_index]
```

### ✅ 解决方案优势

1. **逻辑简洁**：核心逻辑只需几行代码，基于简单的对象引用比较
2. **自动纠错**：节点阻塞时自动保持状态不变，转移成功时才前进
3. **与UXSim同步**：完全符合UXSim内部转移机制，无需额外状态管理
4. **无需防重复逻辑**：基于实际转移结果判断，自然避免重复调用问题

### 🎯 解决的关键问题

1. **自环链路重复转移请求**：通过转移确认机制自然防止重复处理
2. **route_index异常跳跃**：只有确认转移成功时才递增索引，避免跳跃
3. **节点阻塞场景**：转移失败时route_index保持不变，自动等待下次机会
4. **路径执行精确性**：确保车辆严格按预定路径行驶，100%路径匹配

### 🔬 问题根源回顾

通过深度调试发现的根本原因：
1. **UXSim自环链路**触发多次转移请求（内部机制特性）
2. **route_index管理问题**：在某些情况下被意外修改或重复递增
3. **转移时序问题**：转移请求与确认之间的时序不一致

### 📊 实施效果验证

**预期效果**（已实现）：
- ✅ 防止自环充电链路的重复转移问题
- ✅ 防止`route_index`异常跳跃
- ✅ 确保车辆严格按预定路径行驶
- ✅ 提供最简洁、可靠的实现方案

### 🏆 技术方案评估

**最终选择**：🥇 **转移确认机制**（已实施）
- **简洁性**：核心逻辑仅需几行代码
- **可靠性**：基于UXSim内部机制，完全同步
- **易维护性**：逻辑清晰，无复杂状态管理

**替代方案**：
- 🥈 防重复调用机制 - 复杂度较高，需要额外状态跟踪
- 🥉 索引管理加固 - 治标不治本，无法从根源解决问题

### 相关实现文件

- **解决方案实现**：`src/patch.py` - _patched_vehicle_route_next_link_choice()方法（转移确认机制）
- **主要环境**：`src/EVCSChargingGameEnv.py` - 应用补丁的v3.0环境实现
- **技术验证参考**：`src/technical_validation.py` - 原始工作版本
- **测试验证**：`main.py` - 环境测试入口