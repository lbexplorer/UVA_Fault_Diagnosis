# UAV Fault Diagnosis

本仓库用于开展无人机多传感器故障诊断实验，基于 BASiC 数据集实现完整的**两阶段故障检测与归因框架**。

## 🎯 项目目标

构建完整的无人机故障诊断体系：

**第一阶段：故障检测（二分类）**
- 任务：判断局部飞控窗口是否包含故障症状
- 输入：时间窗口内的多传感器遥测数据
- 输出：Failure vs No Failure
- 数据规模：70 个 flight（10 No Failure + 60 Failure）

**第二阶段：故障归因（六分类）**
- 任务：在确认存在故障的前提下，识别故障类型
- 输入：故障点附近的时序窗口数据
- 输出：6 类传感器故障（GPS、RC、Accelerometer、Gyroscope、Compass、Barometer）
- 数据规模：60 个故障 flight（每类 10 个）

## 📊 最新实验成果

### 六分类任务性能对比（3秒窗口，增强统计特征）

| 模型 | 准确率 | 宏平均 F1 | 标准差 | 相对改进 |
|-----|------|----------|------|---------|
| MLP | 43.89% | 42.71% | 7.64% | - |
| Random Forest | 55.56% | 54.39% | 9.74% | +26.6% |
| **XGBoost** | **75.00%** | **74.23%** | 10.18% | **+70.7%** |

### 二分类任务性能（3秒窗口，增强统计特征）

| 模型 | 准确率 | F1 分数 | ROC-AUC |
|-----|------|--------|---------|
| MLP | 81.90% | 89.20% | 0.665 |

**关键发现：**
- XGBoost 在六分类任务上展现出显著优势，准确率达到 75%
- 二分类任务相对简单，MLP 即可达到 82% 准确率
- 树模型在小样本数据集上表现更稳定

## 📁 仓库结构

```text
.
├── datasets/
│   └── BASiC/
│       └── Processed Data/          # 飞行日志主数据
│           ├── 2022-07-26 05-49-12 (No Failure)/
│           ├── 2022-07-26 06-25-08 (RC Failure)/
│           └── ...
├── outputs/                         # 实验结果输出目录
│   ├── esf_csmlp_v2/               # MLP 六分类结果
│   ├── enhanced_rf_3sec/           # RF 六分类结果
│   ├── enhanced_xgb_3sec/          # XGBoost 六分类结果
│   ├── failure_vs_nofailure_mlp_3sec/  # MLP 二分类结果
│   └── README.md                   # 实验结果索引
├── 实验方案/                        # 实验方案文档
│   └── failure_vs_nofailure_experiment_plan.md
├── uav_fault_detection_binary.py    # 二分类故障检测脚本
├── uav_fault_new_method_v2.py       # 六分类 MLP 方案
├── uav_fault_tree_baseline.py       # 树模型基线脚本
├── 研究方案v1.md                     # 原始研究方案
├── 研究方案v2.md                     # 扩展研究方案
├── 数据集介绍.md                     # 数据集说明
└── README.md                        # 项目说明（本文件）
```

## 🚀 核心实验脚本

### 1. 二分类故障检测
**脚本**: `uav_fault_detection_binary.py`
- 任务：Failure vs No Failure 二分类
- 模型支持：MLP、Random Forest、XGBoost
- 特征：精选传感器 + 增强统计量
- 窗口：3秒（主设定）

### 2. 六分类故障归因
**脚本**: `uav_fault_new_method_v2.py`
- 任务：6 类传感器故障分类
- 模型：轻量 MLP（256-128 隐藏层）
- 特征：精选传感器 + 增强统计量
- 窗口：1秒 / 3秒 对比

### 3. 树模型基线对比
**脚本**: `uav_fault_tree_baseline.py`
- 任务：六分类故障归因
- 模型：Random Forest、XGBoost
- 特征：与 MLP 方案相同设置
- 目的：验证树模型在小样本上的优势

## 🔧 技术特色

### 1. 稳定的故障锚点定位
- 优先使用 `metadata.csv` 中的故障时间/行号
- 自动回退到 `Status` 跳变检测
- 导出 `sample_anchor_check.csv` 用于人工验证

### 2. 精选传感器特征集
基于无人机飞行控制原理，优先选择与故障诊断相关的 51 个关键特征：
- **GPS/导航**：位置、速度、高度、精度等
- **IMU/姿态**：加速度、角速度、EG/EA 等
- **磁罗盘**：磁场强度、航向等
- **气压计**：气压、温度、高度等
- **RC/控制**：遥控信号、姿态控制等

### 3. 分层统计特征工程
- **基础模式** (mean_std)：均值、标准差（102 维）
- **增强模式** (enhanced)：13 类统计量（663 维）
  - 位置统计：mean, std, max, min, range
  - 分布统计：median, q25, q75
  - 动态统计：rms, abs_mean, diff_mean, diff_std, slope

### 4. 小样本学习优化
- **类别不平衡处理**：加权交叉熵 + Focal Loss
- **数据增强**：时间窗口偏移（±150ms, ±75ms）
- **稳定评估**：10 折分层交叉验证

## 📈 实验配置快速开始

### 六分类实验（推荐 XGBoost）
```bash
python uav_fault_tree_baseline.py \
  --data-root datasets/BASiC/Processed\ Data \
  --output-dir outputs/xgb_best \
  --model xgb \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced
```

### 二分类实验
```bash
python uav_fault_detection_binary.py \
  --data-root datasets/BASiC/Processed\ Data \
  --output-dir outputs/binary_best \
  --model xgb \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced
```

## 📋 研究方案演进

- **研究方案 v1**：聚焦六分类任务，提出统计特征增强 + MLP 改进
- **研究方案 v2**：扩展为完整两阶段框架，系统对比多模型架构

## 🔄 版本控制与协作

项目采用 Git 进行版本控制，所有实验脚本、结果和文档均已推送到 GitHub：

- **仓库地址**：https://github.com/lbexplorer/UVA_Fault_Diagnosis
- **持续跟踪**：所有 Python 脚本和关键文档纳入版本控制
- **实验复现**：完整的运行配置和结果保存在 `outputs/` 目录

## 🎯 后续研究方向

### 短期目标
- [ ] 补全二分类任务的 RF/XGBoost 实验
- [ ] 特征重要性分析与消融实验
- [ ] 超参数网格搜索优化

### 中期目标
- [ ] 直接时序模型探索（1D-CNN + 注意力）
- [ ] 跨 flight 泛化性验证
- [ ] 多源数据融合策略

### 长期愿景
- [ ] 实时推理系统部署
- [ ] 故障生命周期预测
- [ ] 多飞行器协同诊断

---

**最后更新**：2026-04-14
**版本**：v2.0
**数据集**：BASiC Processed Data
**实验状态**：活跃进行中

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
