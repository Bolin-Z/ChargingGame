# Anaheim 数据集转换方案

## 概述

将 `construct_dataset/Anaheim/` 下的 TNTP 格式数据转换为项目标准格式，输出到 `data/anaheim/`。

## 源数据分析

### 文件清单
| 文件 | 内容 | 用途 |
|------|------|------|
| `anaheim_nodes.geojson` | 节点坐标（WGS84经纬度） | 提取节点位置 |
| `Anaheim_net.tntp` | 链路数据（容量、长度、速度等） | 提取链路信息 |
| `Anaheim_trips.tntp` | OD需求矩阵 | 提取交通需求 |

### 网络规模
- Zones: 38
- Nodes: 416
- Links: 914
- Total Trips: 104,694.40

### 单位
- 距离: feet
- 时间: minutes
- 速度: feet/minute

## 目标格式

参考 `data/siouxfalls/` 的标准格式：

```
data/anaheim/
├── anaheim_nodes.csv      # name, x, y
├── anaheim_links.csv      # name, start, end, length, u, kappa
├── anaheim_demand.csv     # orig, dest, start_t, end_t, q
└── anaheim_settings.json  # 仿真和博弈配置
```

## 转换规则

### 单位转换
| 原始 | 目标 | 转换系数 |
|------|------|----------|
| feet → meters | length | ×0.3048 |
| ft/min → m/s | speed (u) | ×0.00508 (0.3048/60) |

### 参数设定
- **kappa (jam_density)**: 统一设为 0.2（与 siouxfalls 一致）
- **坐标**: 保留原始经纬度

### 充电站配置
- **数量**: 30 个（约占节点数 7.2%）
- **选址方法**: 基于节点流量统计，选取 Top 30 高流量节点
- **价格范围**: [0.5, 2.0]（与其他数据集一致）

## 转换流程

### 第一步：基础格式转换
- [ ] 创建 `convert_anaheim.py` 脚本
- [ ] 解析 geojson 提取节点坐标
- [ ] 解析 TNTP 网络文件提取链路
- [ ] 解析 TNTP 需求文件提取 OD 矩阵
- [ ] 输出标准 CSV 文件

### 第二步：初始配置
- [ ] 创建 `anaheim_settings.json`
- [ ] 设定基础仿真参数
- [ ] charging_nodes 暂时留空

### 第三步：UXSim 验证
- [ ] 创建 `test_anaheim.py` 测试脚本
- [ ] 验证网络连通性
- [ ] 调整 demand_multiplier 获得合适拥堵水平
- [ ] 目标：完成率 80-100%，有明显拥堵效果

### 第四步：充电站选址
- [ ] 运行纯交通仿真统计流量
- [ ] 选取 Top 30 高流量节点
- [ ] 更新 charging_nodes 配置

### 第五步：最终验证
- [ ] 完整环境加载测试
- [ ] 验证路径计算正常
- [ ] 验证充电仿真正常

## 初始 settings.json 模板

```json
{
    "network_name": "anaheim",
    "simulation_time": 9600,
    "deltan": 5,
    "demand_multiplier": 1.0,

    "charging_car_rate": 0.1,
    "charging_link_length": 3000,
    "charging_link_free_flow_speed": 10,

    "charging_periods": 8,
    "charging_nodes": {},

    "routes_per_od": 10,
    "time_value_coefficient": 0.005,
    "charging_demand_per_vehicle": 50,
    "ue_convergence_threshold": 1.0,
    "ue_max_iterations": 100,
    "ue_swap_probability": 0.05
}
```

## 对比参考

| 参数 | Sioux Falls | Berlin | Anaheim (计划) |
|------|-------------|--------|----------------|
| 节点数 | 24 | 224 | 416 |
| 链路数 | 76 | 523 | 914 |
| 充电站数 | 4 | 20 | 30 |
| 充电站比例 | 16.7% | 8.9% | 7.2% |
| demand_multiplier | 1.0 | 3.5 | 待定 |

## 相关脚本

### 新建脚本
- `convert_anaheim.py` - TNTP → CSV 格式转换
- `test_anaheim.py` - UXSim 加载和仿真测试

### 可复用脚本（需适配）
- `analyze_station_placement.py` - 充电站选址分析

## 进度记录

| 日期 | 步骤 | 状态 | 备注 |
|------|------|------|------|
| - | 第一步：格式转换 | 待开始 | |
| - | 第二步：初始配置 | 待开始 | |
| - | 第三步：UXSim验证 | 待开始 | |
| - | 第四步：充电站选址 | 待开始 | |
| - | 第五步：最终验证 | 待开始 | |
