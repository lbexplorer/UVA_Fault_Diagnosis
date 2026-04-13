# UAV Fault Diagnosis

本仓库用于开展无人机多传感器故障诊断实验，当前重点任务是基于 BASiC 数据集完成 `stage-2` 六类故障分类。

## 当前实验目标

- 任务: 基于飞行日志窗口，对故障飞行样本进行六分类
- 类别: `GPS`、`RC`、`Accelerometer`、`Gyroscope`、`Compass`、`Barometer`
- 数据范围: 当前实验使用 60 个故障飞行样本
- 数据源: `datasets/BASiC/Processed Data`
- 输出目录: `outputs/`

当前实验关注两个问题:

1. 故障发生附近的时序窗口应该如何截取
2. 在小样本条件下，统计特征配合哪类分类器更有效

## 仓库结构

```text
.
├── datasets/
│   └── BASiC/
│       └── Processed Data/          # 飞行日志主数据
├── outputs/                         # 各实验输出结果
├── uav_fault_new_method_v2.py       # MLP 方案
├── uav_fault_tree_baseline.py       # RF / XGB 树模型基线
└── README.md
```

## 当前进行中的实验主线

### 1. 统计特征 + 轻量 MLP

脚本: `uav_fault_new_method_v2.py`

核心流程:

- 自动遍历 `Processed Data` 下每个飞行样本主 CSV
- 根据故障标签名推断类别，忽略 `No Failure`
- 结合 `Status` 跳变与可选 `metadata.csv` 定位故障锚点
- 以锚点为中心截取时间窗口
- 对精选传感器列构造统计特征
- 使用分层随机划分进行 10 次重复评估

### 2. 树模型基线对比

脚本: `uav_fault_tree_baseline.py`

目的:

- 在相同锚点、窗口与统计特征设置下，验证树模型是否比 MLP 更适合当前小样本任务
- 当前已完成 `Random Forest` 与 `XGBoost` 两组基线

## 已进行的关键改进

相较于只做简单窗口分类的初始思路，当前实验已经做了这些改进:

### 1. 故障锚点定位更稳定

- 优先使用 `metadata.csv` 中记录的故障时间/行号
- 若缺失元数据，则回退到 `Status` 从正常到故障的跳变点
- 对所有样本导出 `sample_anchor_check.csv`，便于人工检查锚点是否合理

这一步的意义是尽量让特征窗口覆盖真正的故障触发附近，而不是任意时间段。

### 2. 使用人工筛选的关键传感器特征

当前不是直接使用所有数值列，而是优先采用一组与故障诊断更相关的精选特征，包括:

- GPS / 位置与速度相关列
- IMU 加速度与角速度相关列
- 磁罗盘与航向相关列
- 气压计与高度相关列
- RC / 姿态控制相关列

当前精选列数为 51 个，对应:

- `enhanced` 统计特征时，总特征维度为 663
- `mean_std` 统计特征时，总特征维度为 102

### 3. 从简单统计量扩展到增强统计量

当前支持两组统计方式:

- `mean_std`: 均值、标准差
- `enhanced`: 均值、标准差、最大值、最小值、极差、RMS、中位数、四分位数、绝对均值、差分统计、斜率

这让模型不仅看到窗口内的平均水平，也能看到波动、趋势和动态变化。

### 4. 小样本不平衡处理

在 MLP 方案中，已经加入:

- 类别加权损失
- `WeightedRandomSampler`
- 可选窗口平移增强 `--use-augmentation`

其中数据增强通过对锚点做毫秒级左右偏移，生成多个训练窗口，降低模型对单一定位点的过拟合。

### 5. 增加窗口长度对比实验

当前已经系统比较:

- `1.0s` 窗口
- `3.0s` 窗口

实验表明，仅从 `1s` 扩展到 `3s` 对 MLP 提升有限，但为树模型提供了更稳定的统计模式。

### 6. 从单一神经网络扩展到树模型基线

这是当前最重要的一步改进。仓库已经从单一 MLP 实验扩展为:

- MLP
- Random Forest
- XGBoost

结果显示，在当前 60 个故障样本、小样本多分类设定下，树模型明显更强。

## 当前实验结果

结果明细见 [outputs/README.md](/home/liubo/software_falut/outputs/README.md)。

### 汇总对比

| 实验 | 模型 | 窗口 | 特征 | 准确率 | 宏 F1 |
| --- | --- | --- | --- | --- | --- |
| `esf_csmlp_v2` | MLP | 1s | curated + enhanced | 42.78% | 41.21% |
| `esf_csmlp_v2_3sec` | MLP | 3s | curated + enhanced | 43.89% | 42.71% |
| `mlp_mean_std_3sec` | MLP | 3s | curated + mean/std | 43.89% | 42.29% |
| `enhanced_rf_3sec` | RF | 3s | curated + enhanced | 55.56% | 54.39% |
| `enhanced_xgb_3sec` | XGBoost | 3s | curated + enhanced | 75.00% | 74.23% |

### 当前结论

- 在当前实验设置下，`XGBoost` 是最优方案
- 在同样的 `3s + curated + enhanced` 设定下:
  - `RF` 相比 `MLP`，准确率从 `43.89%` 提升到 `55.56%`
  - `XGBoost` 相比 `RF`，准确率进一步提升到 `75.00%`
  - `XGBoost` 相比 `MLP`，宏 F1 从 `42.71%` 提升到 `74.23%`

这说明当前任务更适合基于统计特征的树模型，而不是直接用轻量 MLP 做小样本分类。

## 如何复现实验

建议先准备 Python 3.10+ 环境，并安装常用依赖:

```bash
pip install numpy pandas scikit-learn torch xgboost
```

### 1. 运行 MLP 主实验

```bash
python uav_fault_new_method_v2.py \
  --data-root "datasets/BASiC/Processed Data" \
  --output-dir outputs/esf_csmlp_v2 \
  --window-sec 1.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --loss weighted_ce \
  --repeats 10
```

### 2. 运行 3 秒窗口 MLP 对比实验

```bash
python uav_fault_new_method_v2.py \
  --data-root "datasets/BASiC/Processed Data" \
  --output-dir outputs/esf_csmlp_v2_3sec \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --loss weighted_ce \
  --repeats 10
```

### 3. 运行带增强的 mean/std MLP 实验

```bash
python uav_fault_new_method_v2.py \
  --data-root "datasets/BASiC/Processed Data" \
  --output-dir outputs/mlp_mean_std_3sec \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode mean_std \
  --loss weighted_ce \
  --use-augmentation \
  --augment-shifts-ms -150 -75 75 150 \
  --repeats 10
```

### 4. 运行 Random Forest 基线

```bash
python uav_fault_tree_baseline.py \
  --data-root "datasets/BASiC/Processed Data" \
  --output-dir outputs/enhanced_rf_3sec \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --model rf \
  --repeats 10
```

### 5. 运行 XGBoost 基线

```bash
python uav_fault_tree_baseline.py \
  --data-root "datasets/BASiC/Processed Data" \
  --output-dir outputs/enhanced_xgb_3sec \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --model xgb \
  --repeats 10
```

## 输出文件说明

每个实验目录通常包含:

- `summary.json`: 实验总览、平均指标和各 split 详情
- `split_metrics.csv`: 每次分层划分的指标
- `confusion_matrix_mean.csv`: 平均混淆矩阵
- `sample_anchor_check.csv`: 锚点定位检查表
- `feature_names.csv`: 实际使用的统计特征名
- `features_used.json`: 选择的原始列与特征模式
- `run_config.json`: 运行参数

## 下一步可继续改进的方向

- 引入留一飞行架次或更严格的组划分，进一步验证泛化能力
- 对 XGBoost 做系统调参，而不只是当前基线参数
- 分析混淆最严重的类别组合，如 `Compass` 与 `Barometer`
- 尝试多窗口拼接、故障前后不对称窗口或序列模型
- 补充无故障样本，用于扩展到故障检测 + 故障分类两阶段任务

## 备注

- 当前根目录下的 `数据集介绍.md` 仍为空，可后续补充原始数据字段说明
- `outputs/README.md` 已维护实验结果索引，根目录 README 侧重项目整体说明与阶段性结论
