# Failure vs No Failure 补充实验方案

## 1. 目的

该实验用于在现有六类故障归因实验之外，补充一层“故障检测”证据，即判断局部飞控遥测窗口是否包含故障症状。这样可以在论文中更自然地体现“软件故障检测/运行故障检测”的主题。

## 2. 任务定义

- 输入：BASiC Processed Data 主 CSV 中截取的局部时序窗口。
- 输出：二分类标签：
  - `NoFailure`：无故障症状
  - `Failure`：存在故障症状

该任务是运行层故障检测支撑实验；六类分类任务仍然是主实验中的故障归因任务。

## 3. 数据使用方式

- 使用 BASiC Processed Data 的全部 70 个 flight：
  - 10 个 `No Failure`
  - 60 个 `Failure`
- 主 CSV 中包含 `Time` 与 `Status`，可用于定位故障锚点与构造窗口。

## 4. 样本构造规则

### 4.1 Failure 样本

- 以 `Status` 首次 `0 -> 1` 跳变作为故障锚点；
- 若提供 `metadata.csv`，则优先用 `fault_time / fault_row_index` 做校验；
- 以故障锚点为中心截取固定长度窗口。

### 4.2 No Failure 样本

- 因为 No Failure flight 不存在故障锚点，因此采用固定位置构造窗口：
  - 测试样本默认取 flight 中部窗口（`middle`）；
  - 训练增强可额外取 1/4 位置（`q1`）和 3/4 位置（`q3`）窗口。

### 4.3 窗口长度

建议：
- 主设定：`3 s`
- 可选对比：`1 s`

选择 3 s 的原因：
- baseline 第一阶段最终以 3 s 持续窗口触发故障判定；
- 你当前 3 s 六分类实验比 1 s 更稳；
- 对二分类检测来说，更长局部动态通常更容易聚合出异常症状。

## 5. 特征与模型建议

### 5.1 推荐主设定

- 特征：`curated + enhanced`
- 模型：`MLP`
- 损失：`weighted_ce`

### 5.2 推荐对比组

- `curated + enhanced + RF`
- `curated + enhanced + XGBoost`
- （可选）`curated + mean_std + MLP`

## 6. 数据划分与评价口径

- 严格按 `flight-level` 划分，避免同一 flight 的不同窗口同时进入训练和测试；
- 使用分层 `70:30` 划分；
- 重复 `10` 次随机实验并报告均值与标准差。

## 7. 指标

推荐指标：
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC（可选）
- Confusion Matrix

其中：
- Recall 用于衡量故障窗口是否容易漏检；
- Precision 用于衡量是否把正常窗口误报为故障；
- F1 用于综合平衡两者。

## 8. 建议的最小实验组合

### E0-1 主实验：Binary Detection + MLP

```bash
python uav_fault_detection_binary.py \
  --data-root "datasets/BASiC/Processed Data" \
  --metadata-csv "datasets/BASiC/small_data_analysis_pack/metadata.csv" \
  --output-dir "outputs/failure_vs_nofailure_mlp_3sec" \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --model mlp \
  --loss weighted_ce \
  --use-augmentation
```

### E0-2 对比实验：Binary Detection + RF

```bash
python uav_fault_detection_binary.py \
  --data-root "datasets/BASiC/Processed Data" \
  --metadata-csv "datasets/BASiC/small_data_analysis_pack/metadata.csv" \
  --output-dir "outputs/failure_vs_nofailure_rf_3sec" \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --model rf
```

### E0-3 对比实验：Binary Detection + XGBoost

```bash
python uav_fault_detection_binary.py \
  --data-root "datasets/BASiC/Processed Data" \
  --metadata-csv "datasets/BASiC/small_data_analysis_pack/metadata.csv" \
  --output-dir "outputs/failure_vs_nofailure_xgb_3sec" \
  --window-sec 3.0 \
  --feature-profile curated \
  --stats-mode enhanced \
  --model xgb
```

## 9. 论文写法建议

### 9.1 在方法章节中如何定位

建议写成：

- 二分类实验：验证飞控遥测中是否存在可检测的运行故障症状；
- 六分类实验：验证在检测到异常后，能否进一步完成故障归因。

### 9.2 在结果章节中如何安排

建议把二分类放在六分类之前，结构如下：

1. 运行故障检测结果（Failure vs No Failure）
2. 六类故障归因结果
3. 窗口、特征、分类器对比分析

### 9.3 在讨论章节中如何解释

可写：

- 二分类结果说明飞控运行故障可在多源遥测层面形成可观测症状；
- 六分类结果进一步说明这些症状具有一定的类型可分性；
- 因此，本文方法可作为后续更深入故障诊断与定位分析的运行层支撑模块。
