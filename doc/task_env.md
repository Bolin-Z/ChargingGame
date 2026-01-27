# 环境重构：无状态收益评估器

**日期**: 2026-01-24
**目标**: 将环境改造为支持并行评估的无状态收益计算器

---

## 一、背景

### 1.1 当前设计

当前 `EVCSChargingGameEnv` 是一个有状态的 PettingZoo 环境：
- 每个 `step()` 调用执行一次完整的 UE-DTA 求解
- Step 之间存在状态依赖：路径分配从上一个 step 继承
- 观测依赖历史：`price_history` 和 `charging_flow_history`

### 1.2 问题

- 无法并行：同一环境实例的多次 `step()` 调用必须串行
- 经验积累慢：每次只能评估一个价格组合
- 状态耦合：路径分配继承导致结果依赖调用顺序

### 1.3 目标

将环境改造为**无状态的收益评估函数**：
- 输入：价格组合
- 输出：收益、流量、UE统计信息
- 特性：每次调用独立，可并行执行

---

## 二、技术验证

### 2.1 多进程兼容性测试（已完成）

测试 `uxsimpp_extended` 是否支持多进程并行：

| 测试项 | 结果 |
|--------|------|
| 多进程独立创建 World | ✅ 通过 |
| 多进程独立运行仿真 | ✅ 通过 |
| 重复多进程无状态污染 | ✅ 通过 |
| Worker 复用执行多任务 | ✅ 通过 |
| initializer 模式 | ✅ 通过 |

**结论**：`uxsimpp_extended` 完全支持多进程并行。

---

## 三、设计约束

### 3.1 数据加载开销

- 网络数据从文件加载耗时较小
- **路径集合计算耗时较大**（k最短路径算法，OD对数量×k）
- 这些数据是只读的，应该**只计算一次，多次复用**

### 3.2 多进程数据共享

- Python 多进程使用 pickle 序列化传递数据
- C++ 对象（如 World）无法直接序列化
- 需要将网络数据转换为**纯 Python 数据结构**

---

## 四、架构设计（已确认）

### 4.1 核心决策

| 问题 | 决策 |
|------|------|
| 并行评估场景 | 批量/异步并行评估多个价格组合 |
| 路径分配策略 | 放弃继承，每次从贪心分配开始 UE-DTA，确保确定性 |
| 接口设计 | 放弃 PettingZoo，采用 Fictitious Play 风格 |
| Trainer 设计 | 统一 GameTrainer + 算法插件，消除三个独立 Trainer 的重复代码 |

### 4.2 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            系统架构                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌─────────────────────┐                              │
│                        │    GameTrainer      │                              │
│                        │  ┌───────────────┐  │                              │
│                        │  │ GameHistory   │  │  ← 管理策略/收益历史          │
│                        │  │ Algorithm     │  │  ← 可插拔：MADDPG/IDDPG/MFDDPG│
│                        │  └───────────────┘  │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                         │
│                   evaluate(prices) / evaluate_batch(prices_list)            │
│                                   │                                         │
│                                   ▼                                         │
│                        ┌─────────────────────┐                              │
│                        │ParallelEvaluatorPool│  ← 管理多个 Worker 进程       │
│                        └──────────┬──────────┘                              │
│                                   │                                         │
│            ┌──────────────────────┼──────────────────────┐                  │
│            │                      │                      │                  │
│            ▼                      ▼                      ▼                  │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │    Worker 1     │   │    Worker 2     │   │    Worker 3     │   ...    │
│   │  ┌───────────┐  │   │  ┌───────────┐  │   │  ┌───────────┐  │          │
│   │  │ Evaluator │  │   │  │ Evaluator │  │   │  │ Evaluator │  │          │
│   │  │NetworkData│  │   │  │NetworkData│  │   │  │NetworkData│  │          │
│   │  └───────────┘  │   │  └───────────┘  │   │  └───────────┘  │          │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 核心组件

#### 1. NetworkData（可序列化的网络数据）

主进程加载一次，序列化传递给各 Worker。

```python
@dataclass
class NodeData:
    name: str
    x: float
    y: float

@dataclass
class LinkData:
    name: str
    start_node: str
    end_node: str
    length: float              # 米
    free_flow_speed: float     # 米/秒
    jam_density: float         # 车辆/米/车道
    merge_priority: float
    is_charging_link: bool     # 是否为充电链路

@dataclass
class DemandData:
    origin: str
    destination: str
    start_t: float
    end_t: float
    flow: float                # 车辆数/单位时间
    is_charging: bool          # 是否为充电需求

@dataclass
class RouteInfo:
    """预处理的路径信息"""
    links: list[str]              # 链路名称列表
    charging_link_idx: int | None # 充电链路在路径中的索引（或None）
    charging_node: str | None     # 充电节点名称（或None）

@dataclass
class NetworkData:
    # 参数配置
    settings: dict

    # 网络拓扑
    nodes: list[NodeData]
    links: list[LinkData]
    demands: list[DemandData]

    # 预计算路径集合（计算开销大，只算一次）
    # RouteInfo 包含链路名称列表 + 预提取的充电信息
    routes: dict[tuple[str, str], list[RouteInfo]]
```

**设计说明**：
- 存储链路名称列表而非 Link 对象引用（C++ 对象不可序列化）
- RouteInfo 预提取充电链路信息，避免运行时重复解析
- 需求数据只存储原始信息，Evaluator 每次评估时调用 `W.adddemand()` 生成车辆

#### 2. EVCSRewardEvaluator（无状态收益评估器）

**设计原则**：
1. 纯函数：prices + seed → EvalResult
2. 只返回原始数据（rewards、flows），不构造 observations
3. observations 由上层 Trainer 根据需要构造

**数据结构**：

```python
@dataclass
class EvalResult:
    """评估结果"""
    rewards: dict[str, float]           # 各充电站收益
    flows: dict[str, np.ndarray]        # 各充电站各时段流量 (n_periods,)
    ue_iterations: int                  # UE-DTA 迭代次数
    converged: bool                     # 是否收敛
    ue_stats: dict | None = None        # 可选的 UE 统计信息
```

**Evaluator 实现**：

```python
class EVCSRewardEvaluator:
    """无状态收益评估器"""

    def __init__(self, network_data: NetworkData):
        self.network_data = network_data

    def evaluate(self, prices: dict[str, np.ndarray], seed: int = None) -> EvalResult:
        """
        单次评估：价格 -> 收益

        Args:
            prices: 各充电站各时段价格 {agent_name: (n_periods,)}
            seed: 随机种子（固定则结果确定性）

        Returns:
            EvalResult: 收益、流量、UE 统计信息
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. 创建 World 实例
        W = self._create_world()

        # 2. 贪心路径分配作为初始状态
        routes_specified = self._initialize_routes_greedy(W)

        # 3. UE-DTA 迭代直到收敛
        for iteration in range(self.network_data.settings['ue_max_iterations']):
            # 3.1 创建新 World（每轮迭代需要新实例）
            W = self._create_world()
            self._apply_routes(W, routes_specified)

            # 3.2 执行仿真
            W.exec_simulation()

            # 3.3 计算成本、统计流量、路径切换
            flows, stats, new_routes = self._route_choice_update(
                W, routes_specified, prices, iteration
            )

            # 3.4 收敛检测
            if self._check_convergence(stats):
                break

            routes_specified = new_routes
            W.release()

        # 4. 计算收益
        rewards = self._calculate_rewards(prices, flows)

        return EvalResult(
            rewards=rewards,
            flows=flows,
            ue_iterations=iteration + 1,
            converged=(iteration < self.network_data.settings['ue_max_iterations'] - 1),
            ue_stats=stats
        )

    def _create_world(self) -> World:
        """用 NetworkData 创建 World 实例"""
        ...

    def _initialize_routes_greedy(self, W) -> dict:
        """贪心路径分配（最短路径）"""
        ...

    def _apply_routes(self, W, routes_specified):
        """应用路径分配到车辆"""
        ...

    def _route_choice_update(self, W, routes_specified, prices, iteration):
        """路径选择更新（UE-DTA 核心逻辑）"""
        ...

    def _check_convergence(self, stats) -> bool:
        """检查 UE-DTA 是否收敛"""
        ...

    def _calculate_rewards(self, prices, flows) -> dict:
        """计算各充电站收益"""
        ...
```

**Trainer 层构造 observations**：

```python
class GameTrainer:
    def _build_observations(self, prices: dict, result: EvalResult) -> dict:
        """根据 prices 和 flows 构造环境观测格式"""
        # 归一化价格到 [0, 1]
        normalized_prices = self._normalize_prices(prices)
        price_matrix = np.array([normalized_prices[name] for name in self.agent_names])

        observations = {}
        for agent in self.agent_names:
            observations[agent] = {
                "last_round_all_prices": price_matrix.astype(np.float32),
                "own_charging_flow": result.flows[agent].astype(np.float32)
            }
        return observations
```

特性：**无状态 | 确定性(固定seed) | 可并行**

#### 3. ParallelEvaluatorPool（并行评估池）

管理多进程 Worker，提供批量评估接口。

```python
class ParallelEvaluatorPool:
    def __init__(self, network_data: NetworkData, n_workers: int = None):
        """
        启动 n_workers 个进程，每个进程用 network_data 初始化自己的 Evaluator
        n_workers 默认为 CPU 核心数
        """
        pass

    def evaluate(self, prices: dict, seed: int = None) -> EvalResult:
        """单次评估，分配给空闲 Worker"""
        pass

    def evaluate_batch(self, prices_list: list[dict], seeds: list[int] = None) -> list[EvalResult]:
        """批量并行评估，自动分配任务"""
        pass

    def submit(self, prices: dict, seed: int = None) -> Future:
        """异步提交，返回 Future 对象"""
        pass

    def shutdown(self):
        """关闭所有 Worker"""
        pass
```

#### 4. GameTrainer（统一训练器）

管理学习过程，维护历史状态，支持插拔不同算法。

```python
class GameTrainer:
    def __init__(self,
                 evaluator_pool: ParallelEvaluatorPool,
                 algorithm: Algorithm,  # MADDPG / IDDPG / MFDDPG
                 config: TrainingConfig):
        self.pool = evaluator_pool
        self.algorithm = algorithm
        self.history = GameHistory()
        self.config = config

    def train(self):
        for episode in range(self.config.max_episodes):
            self.history.reset()

            for step in range(self.config.max_steps):
                # 1. 获取观测，选择动作
                obs = self.history.get_observations()
                prices = self.algorithm.take_action(obs)

                # 2. 评估收益
                result = self.pool.evaluate(prices)

                # 3. 更新历史和算法
                self.history.record(prices, result)
                self.algorithm.update(self.history)

                # 4. 收敛检测
                if self._check_convergence():
                    break
```

#### 5. Algorithm 接口（算法抽象）

三个算法实现相同接口，可插拔使用。

```python
class Algorithm(Protocol):
    def take_action(self, observations: dict) -> dict[str, np.ndarray]:
        """根据观测选择动作（价格）"""
        ...

    def update(self, history: GameHistory) -> None:
        """根据历史更新策略"""
        ...

    def save(self, path: str) -> None:
        """保存模型"""
        ...

    def load(self, path: str) -> None:
        """加载模型"""
        ...
```

### 4.4 数据流

```
启动阶段:
┌────────────┐      ┌─────────────┐      ┌──────────────────┐
│ 网络文件   │ ───► │ 主进程加载   │ ───► │ NetworkData      │
│ (csv/json) │      │ + 路径预计算 │      │ (可序列化)       │
└────────────┘      └─────────────┘      └────────┬─────────┘
                                                   │
                                       序列化传递给各 Worker
                                                   │
                    ┌──────────────────────────────┼───────────────┐
                    ▼                              ▼               ▼
               ┌─────────┐                   ┌─────────┐    ┌─────────┐
               │Worker 1 │                   │Worker 2 │    │Worker N │
               │Evaluator│                   │Evaluator│    │Evaluator│
               └─────────┘                   └─────────┘    └─────────┘

运行阶段:
┌─────────┐  prices   ┌──────────┐  分配任务  ┌─────────┐
│ Trainer │ ────────► │   Pool   │ ─────────► │ Workers │
└─────────┘           └──────────┘            └────┬────┘
     ▲                                             │
     │               EvalResult                    │
     └─────────────────────────────────────────────┘
```

### 4.5 与当前代码的对应关系

| 新组件 | 当前代码 | 变化 |
|--------|----------|------|
| `NetworkData` | 分散在 `EVCSChargingGameEnv.__init__` | 提取为独立数据类 |
| `EVCSRewardEvaluator` | `EVCSChargingGameEnv.step` 内部逻辑 | 提取为无状态函数 |
| `ParallelEvaluatorPool` | 无 | 新增 |
| `GameTrainer` | `MADDPGTrainer` / `IDDPGTrainer` / `MFDDPGTrainer` | 合并为统一 Trainer |
| `GameHistory` | `EVCSChargingGameEnv` 的状态 | 移至 Trainer |
| `Algorithm` | `MADDPG` / `IDDPG` / `MFDDPG` | 统一接口 |

---

## 五、详细设计讨论（已确认）

### 5.1 NetworkData 设计

**已确认**：
- 存储链路名称列表，UXSim API 内部处理名称→Link 对象转换
- RouteInfo 预提取充电链路信息（charging_node, charging_link_idx）
- 需求数据存储原始信息，每次评估时通过 `W.adddemand()` 生成车辆

### 5.2 评估器设计

**已确认**：
- 每次评估需要重新创建 World（World 没有 reset 方法，仿真会修改状态）
- 每轮 UE-DTA 迭代也需要新建 World
- 路径数据（NetworkData.routes）可跨 World 复用

**单次评估流程**：
```
for ue_iteration in range(max_iterations):
    1. 创建新 World
       - addNode(), addLink(), adddemand()
       - Vehicle 此时没有 predefined_route

    2. 为每辆车分配路径
       - veh.assign_route(route_info.links)  # 链路名称列表
       - 内部转换为 Link* 列表

    3. 执行仿真
       - W.exec_simulation()

    4. 收集结果，检查收敛
       - 如果收敛，退出循环

返回 EvalResult
```

### 5.3 路径成本计算与缓存优化

**背景**：仿真核心在 C++ 层，Python 层是包装。pybind11 访问 `std::vector` 每次都复制整个数组。

**问题**：计算路径成本需要频繁访问 `link.traveltime_real[t]`，每次访问复制 1920 元素。

**解决方案**：每轮仿真后，一次性缓存所有链路的 traveltime_real 到 Python 层。

```python
def _build_traveltime_cache(W) -> dict:
    """仿真后调用一次，缓存所有链路的通行时间"""
    cache = {}
    delta_t = W.delta_t
    for link in W.LINKS:
        cache[link.name] = {
            'traveltime_real': list(link.traveltime_real),  # 只复制一次
            'max_idx': len(link.traveltime_real) - 1
        }
    return cache, delta_t

def compute_route_cost(route_info: RouteInfo, departure_time: float,
                       prices: dict, link_cache: dict, delta_t: float,
                       time_value_coefficient: float) -> float:
    """使用缓存计算路径成本（纯 Python 操作，无边界跨越）"""
    tt = 0
    current_t = departure_time

    for link_name in route_info.links:
        cache = link_cache[link_name]
        tt_idx = min(int(current_t // delta_t), cache['max_idx'])
        link_tt = cache['traveltime_real'][tt_idx]
        tt += link_tt
        current_t += link_tt

    time_cost = time_value_coefficient * tt

    # 充电成本（直接用预存的信息，无需查询 attribute）
    charging_cost = 0
    if route_info.charging_node:
        period = get_period(departure_time + ...)  # 计算到达充电站的时段
        charging_cost = prices[route_info.charging_node][period] * charging_demand

    return time_cost + charging_cost
```

**性能对比**：
| 操作 | 耗时 (μs/次) | 说明 |
|------|-------------|------|
| 直接访问 `link.traveltime_real` | 26.36 | 每次复制整个 vector |
| 缓存后访问 | 0.10 | 纯 Python list 操作 |
| **加速比** | **263x** | |

### 5.4 路径分配优化（可选）

**当前实现**：每轮 UE-DTA 迭代，为每辆车调用 `assign_route(["link1", "link2", ...])`，C++ 层每次都做字符串→Link* 哈希查找。

**优化方案：路径索引预注册**

```python
# World 创建后，一次性注册所有路径
route_registry = {}  # (od, route_idx) -> route_id
for od, routes in network_data.routes.items():
    for i, route_info in enumerate(routes):
        route_id = W.register_route(route_info.links)  # C++ 层缓存转换结果
        route_registry[(od, i)] = route_id

# 每轮 UE-DTA 分配路径时，只传整数
assignments = [(veh.id, route_registry[(od, chosen_idx)]), ...]
W.batch_assign_routes_by_id(assignments)  # 无字符串查找
```

**收益**：
- 字符串→Link* 转换只做一次（注册时）
- 分配时只传整数对，数据量小
- 可在后续优化阶段实现

### 5.5 GameHistory 设计

**设计原则**：
1. 以单次环境评估为记录单位，并行只是加速手段
2. 记录环境返回的原始信息，算法自己组装观测
3. 算法内部自己管理 ReplayBuffer，GameHistory 只负责训练历史记录

**数据结构**：

```python
@dataclass
class EvaluationRecord:
    """单次评估记录"""
    eval_id: int                          # 全局递增计数
    batch_id: int                         # 属于哪个 iteration

    # 动作相关
    prices: dict[str, np.ndarray]         # 实际价格
    actions: dict[str, np.ndarray]        # 归一化动作 [0,1]（含噪音）
    pure_actions: dict[str, np.ndarray]   # 纯策略动作（不含噪音）

    # 环境返回
    rewards: dict[str, float]
    flows: dict[str, np.ndarray]          # 各充电站各时段流量
    observations: dict[str, dict]         # 环境返回的原始观测
    ue_info: dict                         # ue_converged, ue_iterations, ue_stats


@dataclass
class BatchSummary:
    """每个 iteration 结束后的汇总信息"""
    batch_id: int
    eval_id_range: tuple[int, int]        # 这批评估的 id 范围 [start, end)
    learn_metrics: dict | None            # 学习指标
    nashconv: float | None                # NashConv（如果这轮做了评估）
    strategy_change_rate: float | None    # 策略变化率


class GameHistory:
    """训练历史管理器"""

    def __init__(self):
        self.records: list[EvaluationRecord] = []
        self.batch_summaries: list[BatchSummary] = []

    def record(self, batch_id: int, prices: dict, actions: dict,
               pure_actions: dict, result: EvalResult, observations: dict):
        """记录单次评估"""
        self.records.append(EvaluationRecord(
            eval_id=len(self.records),
            batch_id=batch_id,
            prices=prices,
            actions=actions,
            pure_actions=pure_actions,
            rewards=result.rewards,
            flows=result.flows,
            observations=observations,
            ue_info=result.ue_info
        ))

    def record_batch(self, batch_id: int, prices_list: list, actions_list: list,
                     pure_actions_list: list, results: list, observations_list: list):
        """批量记录（并行评估后调用）"""
        for prices, actions, pure_actions, result, obs in zip(
            prices_list, actions_list, pure_actions_list, results, observations_list
        ):
            self.record(batch_id, prices, actions, pure_actions, result, obs)

    def add_batch_summary(self, batch_id: int, learn_metrics: dict = None,
                          nashconv: float = None, strategy_change_rate: float = None):
        """添加 iteration 汇总"""
        # 计算这批评估的 id 范围
        batch_records = [r for r in self.records if r.batch_id == batch_id]
        if batch_records:
            start_id = batch_records[0].eval_id
            end_id = batch_records[-1].eval_id + 1
        else:
            start_id = end_id = len(self.records)

        self.batch_summaries.append(BatchSummary(
            batch_id=batch_id,
            eval_id_range=(start_id, end_id),
            learn_metrics=learn_metrics,
            nashconv=nashconv,
            strategy_change_rate=strategy_change_rate
        ))

    def get_last_observations(self) -> dict | None:
        """获取最后一次评估的观测（用于下一轮决策）"""
        if not self.records:
            return None
        return self.records[-1].observations

    @property
    def total_evaluations(self) -> int:
        return len(self.records)

    @property
    def total_iterations(self) -> int:
        return len(self.batch_summaries)
```

**数据关系**：
```
batch_summaries[0] ──► records[0:4]   (batch_id=0, M=4)
batch_summaries[1] ──► records[4:8]   (batch_id=1, M=4)
...
```

**并行等价性**：
- 并行 M=4 运行 250 个 iteration = 1000 次评估
- 串行 M=1 运行 1000 个 iteration = 1000 次评估
- 两者的 `records` 数量相同，格式一致

### 5.6 Algorithm 接口统一

**设计原则**：
1. Trainer 传递环境原始观测，算法内部处理格式转换
2. 算法内部自己管理 ReplayBuffer
3. 不需要 save/load（目标是求解均衡价格，不是保存模型）

**统一接口**：

```python
class Algorithm(Protocol):
    """算法统一接口"""

    @property
    def name(self) -> str:
        """算法名称：MADDPG / IDDPG / MFDDPG"""
        ...

    def take_action(self,
                    observations: dict[str, dict],  # 环境原始格式
                    add_noise: bool = True
                    ) -> dict[str, np.ndarray]:
        """
        输出动作

        Args:
            observations: 环境原始观测格式
            add_noise: 是否添加探索噪音

        Returns:
            各智能体的动作 {agent_name: action_array}

        Note:
            内部处理观测格式转换（MADDPG/IDDPG/MFDDPG 各自不同）
        """
        ...

    def sample_actions_batch(self,
                             observations: dict[str, dict],
                             batch_size: int
                             ) -> list[dict[str, np.ndarray]]:
        """
        批量采样动作（并行评估用）

        Args:
            observations: 环境原始观测格式
            batch_size: 采样数量 M

        Returns:
            M 个动作字典的列表，每个动作有独立噪音
        """
        ...

    def get_pure_actions(self,
                         observations: dict[str, dict]
                         ) -> dict[str, np.ndarray]:
        """
        获取纯策略动作（不含噪音）

        用途：收敛检测 / NashConv 计算
        """
        ...

    def store_experience(self,
                         observations: dict,
                         actions: dict,
                         rewards: dict,
                         next_observations: dict) -> None:
        """
        存入 ReplayBuffer

        Note:
            内部处理格式转换
        """
        ...

    def learn(self) -> dict | None:
        """
        从 ReplayBuffer 采样并更新网络

        Returns:
            学习指标字典（actor_loss, critic_loss 等），
            或 None（缓冲区不足时）
        """
        ...

    def get_critics(self) -> dict[str, nn.Module]:
        """
        获取 Critic 网络

        用途：NashConv 计算（多起点梯度上升寻找最佳响应）

        Returns:
            {agent_name: critic_network}
        """
        ...

    def reset_noise(self) -> None:
        """
        重置探索噪音到初始值

        用途：假收敛时强制重新探索
        """
        ...
```

**各算法内部观测处理**：

| 算法 | 内部转换方法 | 输出格式 |
|------|-------------|----------|
| MADDPG | `_organize_global_state()` | 全局状态 = 去重价格 + 所有流量 + 所有动作 |
| IDDPG | `_flatten_local_obs()` | 局部状态 = 自身价格历史 + 自身流量 |
| MFDDPG | `_compute_mean_field()` | MF 状态 = 自身 + 其他智能体平均动作 |

**Trainer 调用示例**：

```python
class GameTrainer:
    def _run_iteration(self, batch_id: int):
        # 1. 获取观测（环境原始格式）
        observations = self.history.get_last_observations()

        # 2. 批量采样动作（算法内部处理观测转换）
        actions_list = self.algorithm.sample_actions_batch(observations, self.batch_size)
        pure_actions = self.algorithm.get_pure_actions(observations)

        # 3. 并行评估
        results = self.pool.evaluate_batch(actions_list)

        # 4. 记录历史 & 存入经验池
        next_observations = ...  # 从 results 构造
        for actions, result in zip(actions_list, results):
            self.history.record(batch_id, ...)
            self.algorithm.store_experience(observations, actions, result.rewards, next_observations)

        # 5. 学习
        learn_metrics = self.algorithm.learn()

        # 6. 周期性 NashConv 评估
        if batch_id % self.eval_interval == 0:
            nashconv = self._compute_nashconv(observations, pure_actions)
            ...
```

---

## 六、实施计划

### 6.1 依赖关系与实施顺序

```
NetworkData
    ↓
EVCSRewardEvaluator (依赖 NetworkData)
    ↓
ParallelEvaluatorPool (依赖 Evaluator)
    ↓                                    ┐
GameHistory (独立)                        ├── 可并行
Algorithm 接口 (独立)                     ┘
    ↓
重构三个算法 (依赖 Algorithm 接口)
    ↓
GameTrainer (依赖 Pool + History + Algorithm)
    ↓
测试验证
    ↓
清理旧代码（最后）
```

**分阶段实施**：

| 阶段 | 任务 | 依赖 | 输出 |
|------|------|------|------|
| P1 | NetworkData + 数据加载 | 无 | `src/evaluator/network_data.py` |
| P2 | EVCSRewardEvaluator | P1 | `src/evaluator/evaluator.py` |
| P3 | ParallelEvaluatorPool | P2 | `src/evaluator/pool.py` |
| P4 | GameHistory | 无 | `src/game/history.py` |
| P5 | Algorithm 接口 | 无 | `src/algorithms/base.py` |
| P6 | 重构 MADDPG/IDDPG/MFDDPG | P5 | 原地改造 `src/algorithms/*/` |
| P7 | GameTrainer | P3 + P4 + P6 | `src/game/trainer.py` |
| P8 | 集成测试 | P7 | 验收通过 |
| P9 | 清理旧代码 | P8 | 可选 |

### 6.2 阶段测试脚本

| 阶段 | 测试脚本 | 验证内容 |
|------|----------|----------|
| P1 | `tests/test_network_data.py` | 数据加载正确、可 pickle 序列化、路径预计算正确 |
| P2 | `tests/test_evaluator.py` | 单次评估正确、与旧环境 `step()` 结果对比一致 |
| P3 | `tests/test_pool.py` | 并行评估正确、多 Worker 稳定性、无内存泄漏 |
| P4 | `tests/test_history.py` | 记录/读取正确、批量记录、eval_id 计数 |
| P5-P6 | `tests/test_algorithms.py` | 接口一致性、与旧算法输出对比 |
| P7 | `tests/test_trainer.py` | 完整训练流程可运行 |
| P8 | `tests/test_integration.py` | 端到端对比：新旧架构收敛行为一致 |

### 6.3 迁移策略

**原则**：保留原有代码，通过创建新代码的形式实现。

**目录结构**：

```
src/
├── env/                          # 旧代码（保留）
│   ├── EVCSChargingGameEnv.py
│   └── uxsimpp_extended/
├── trainer/                      # 旧代码（保留）
│   ├── MADDPGTrainer.py
│   ├── IDDPGTrainer.py
│   └── MFDDPGTrainer.py
├── algorithms/                   # 原地改造
│   ├── base.py                   # 新增：Algorithm 接口
│   ├── maddpg/
│   │   └── maddpg.py            # 改造：实现统一接口
│   ├── iddpg/
│   │   └── iddpg.py
│   └── mfddpg/
│       └── mfddpg.py
│
├── evaluator/                    # 新代码
│   ├── network_data.py           # NetworkData + 加载器
│   ├── evaluator.py              # EVCSRewardEvaluator
│   └── pool.py                   # ParallelEvaluatorPool
├── game/                         # 新代码
│   ├── history.py                # GameHistory
│   └── trainer.py                # GameTrainer
│
└── tests/                        # 测试脚本
    ├── test_network_data.py
    ├── test_evaluator.py
    ├── test_pool.py
    ├── test_history.py
    ├── test_algorithms.py
    ├── test_trainer.py
    └── test_integration.py
```

**入口切换**：

```python
# main.py
USE_NEW_ARCHITECTURE = False  # 开关

if USE_NEW_ARCHITECTURE:
    from src.game.trainer import GameTrainer
    from src.evaluator.pool import ParallelEvaluatorPool
    from src.evaluator.network_data import NetworkDataLoader
    # 新架构启动
else:
    from src.trainer.MADDPGTrainer import MADDPGTrainer
    # 旧架构启动
```

**算法改造范围**：

| 部分 | 变化 |
|------|------|
| 核心网络（Actor/Critic） | 不变 |
| ReplayBuffer | 不变 |
| 噪音机制 | 不变 |
| `learn()` 逻辑 | 不变 |
| `take_action()` | 签名微调（返回格式统一） |
| 新增方法 | `sample_actions_batch()`, `get_pure_actions()`, `get_critics()`, `reset_noise()` |
| 观测处理 | 内聚到算法内部 |

### 6.4 验收标准

**功能正确性**：
- [x] Evaluator 输出与旧环境 `step()` 结果数值一致（误差 < 1e-6）
- [x] 固定 seed 时，多次评估结果完全相同
- [ ] 新架构训练收敛行为与旧架构一致

**性能指标**：
- [ ] 并行评估：N 个 Worker 达到 0.8N 倍以上加速比
- [ ] 单次评估耗时无明显退化（< 10%）
- [ ] 无内存泄漏（连续运行 1000 次评估）

---

## 七、协作规则

### 7.1 工作流程

1. **方案确认** → **代码实现** → **文档更新**
2. 方案确认使用简要文字表达设计核心，对齐后再实施
3. 代码实现遵循：先整体框架，再具体接口
4. 有不清楚的地方先确认再行动
5. 需要运行脚本时，提供命令由用户执行，用户反馈运行结果

### 7.2 进度追踪

| 阶段 | 状态 | 备注 |
|------|------|------|
| P1: NetworkData | ✅ 完成 | 24节点, 80链路, 4充电站, 序列化786KB |
| P2: EVCSRewardEvaluator | ✅ 完成 | 与旧环境总收益完全一致，确定性验证通过，平均耗时40秒/次 |
| P3: ParallelEvaluatorPool | ✅ 完成 | 2 Worker 加速比 1.63x，7 项测试通过，串行/并行结果一致 |
| P4: GameHistory | ✅ 完成 | EMA 信念更新，12 项测试通过 |
| P5: Algorithm 接口 | ✅ 完成 | AlgorithmBase 抽象基类，7 项测试通过 |
| P6: 重构算法 | ✅ 完成 | MADDPGv1/IDDPGv1/MFDDPGv1，20 项测试通过 |
| P7: GameTrainer | ✅ 完成 | 异步事件驱动主循环，8 项测试通过 |
| P8: NashConvChecker | ✅ 完成 | 多起点梯度上升，7 项测试通过 |
| P9: 集成测试 | ✅ 完成 | 端到端测试通过，加速比 1.79x，结果一致性验证通过 |