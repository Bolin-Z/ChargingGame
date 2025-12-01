# 柏林路网简化数据集 - 完成记录

## 1. 数据集概览

已完成 `berlin_friedrichshain_simplified` 数据集构建。

| 指标 | 数值 |
|-----|------|
| 节点数 | 191 |
| 链路数 | 451 |
| OD对数 | 506 |
| 充电站数 | 20 |
| 路网总长度 | 167.8 km |

## 2. 关键参数

### 2.1 空间聚合
- **聚合阈值**: 50.0m
- **节点缩减**: 224 → 191 (减少 14.7%)

### 2.2 需求倍数
经过仿真测试，确定最佳需求倍数：

| 倍数 | 完成率 | 平均速度 | 延误比 | 评价 |
|-----|-------|---------|-------|------|
| 1x | 100% | 28.0 m/s | 14% | 无拥堵 |
| 3x | 100% | 10.5 m/s | 78% | 轻度拥堵 |
| **3.5x** | **100%** | **7.4 m/s** | **88%** | **推荐** |
| 4x | 81% | 4.1 m/s | 93% | 较重拥堵 |
| 5x | 42% | 2.5 m/s | 92% | 严重拥堵 |

**推荐配置**: `demand_multiplier = 3.5`
- 100% 完成率
- 明显的高峰拥堵效果
- 仿真结束前网络可恢复

## 3. 文件结构

```
berlin_friedrichshain_simplified/
├── nodes.csv          # 节点坐标
├── links.csv          # 链路数据 (name, start, end, length, u, kappa)
├── demand.csv         # OD需求 (orig, dest, start_t, end_t, q)
├── settings.json      # 仿真配置 (含20个充电站)
└── uxsim.py           # UXSim源码参考
```

## 4. 使用方式

```python
# 测试脚本
python test_berlin_simplified.py

# 在博弈环境中使用时，需要放大需求
demand_multiplier = 3.5
W.adddemand(orig, dest, start_t, end_t, q * demand_multiplier)
```
