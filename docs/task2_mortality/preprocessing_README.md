# 院内死亡任务：数据预处理

本目录只负责数据处理，不训练、选择或评估机器学习模型。

## 冻结的主任务定义

使用入 ICU 后 `0 <= Time < 48h` 的记录，在第 48 小时时点预测最终 `In-hospital_death`。`outcomes.csv` 中的 `SOFA`、`SAPS-I`、`Length_of_stay` 和 `Survival` 不进入特征矩阵；`RecordID` 只作连接键。

`Time=48:00` 与 README 声明的 `47:59` 上限冲突，因此主版本排除，并为受影响患者单独输出含边界记录的敏感性版本。

## 一键运行

在项目根目录运行：

```powershell
& 'C:\Users\在九华山升旗的外交官\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' preprocessing\mortality_preprocess.py
& 'C:\Users\在九华山升旗的外交官\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s preprocessing\tests -v
& 'C:\Users\在九华山升旗的外交官\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' preprocessing\verify_outputs.py
& 'C:\Users\在九华山升旗的外交官\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' preprocessing\validate_phase3_outputs.py
```

规则集中在 `config.json`。原始 `release/` 不会被修改。

## 交付物

- `patient_features_unimputed.csv.gz`：每个 ICU stay 一行的未填补特征；缺失保留为 NA。
- `labels_and_splits.csv`：标签及固定、分层的 train/validation/test 划分。
- `X_*_imputed.csv.gz`：仅用训练集拟合中位数后得到的三个 X；未缩放，`RecordID` 仍只作连接键。
- `patient_features_48h_inclusive_affected_rows.csv.gz`：只含受 `48:00` 影响患者的替换行，用于敏感性分析。
- `feature_dictionary.csv`：特征来源、时间窗、统计量、单位与缺失语义。统计量 `delta` 即实验方案 §4.1 要求的（末值−首值）趋势，`slope_per_hour` 是同窗口最小二乘斜率。
- `preprocessing_state.json`：训练集填补值、列顺序及 schema 哈希。
- `quality_flags_by_stay.csv`：逐 stay 的空参数、重复、时间边界等质量标记。
- `feature_missingness.csv`、`parameter_distributions.csv`、`split_summary.csv`：质量统计。
- `source_manifest.json`、`preprocessing_summary.json`：来源哈希、输出哈希和自动 QA。
- `independent_verification.json`：重新读取全部交付物后的独立端到端验收结果。
- `mortality_data_preprocessing.ipynb`：已执行的 16 项 QA 笔记本；不含模型训练。
- `mortality_preprocessing_report_artifact.json`：可浏览技术报告的数据与布局定义。

## 派生特征（Phase 3）

以下特征为**新增**，六个原始血压列与静态 Height/Weight 全部原样保留，便于做消融对比。

- **合并血压** `bp_sys / bp_dias / bp_mean`：逐时间窗判断，该窗内只要有一条有创读数就用有创流，否则回落到无创袖带流。理由是全队 58.7% 的病人两种血压并存、28.6% 只有无创，分开建模会让逻辑回归对只有袖带的病人用全体中位数去填一个不存在的动脉压。
- **动脉导管指示** `bp__<window>__has_arterial_line`：该窗内是否存在有创读数（0/1）。是否置管本身反映病情严重度。
- **静态 BMI** `static_bmi` 与 `static_bmi_missing`：由 00:00 的 Height/Weight 推出；两者任一缺失则为 NA。Height 覆盖率仅 52%，因此 BMI 覆盖率上限同为 52%。
- 三条合并血压流与 BMI **不计入** `record__<window>__distinct_parameter_count` 等记录级计数器，这些计数器仍只描述 37 个原始动态变量，保证跨版本可比。

## 重要边界

- `-1` 按 README 转为缺失；“没有测”不等于“正常”或 0。
- 未因缺胆红素或其他检查删除患者；用 `measured`、`count` 及 NA 表达信息性缺失。
- 精确重复和同时间同变量多值都会单独审计，并同时报告“冲突键数”和“超出首值的额外不同值数”。为避免动态特征的 first/last 依赖原始行序，同一分钟的连续指标取中位数、MechVent 取最大值、Urine 求和后再做纵向统计；`count` 仍记录原始有效行数。静态入院 Weight 明确保留 00:00 第一条有效描述符，冲突另留标记。该描述符**有意**同时留在 Weight 时间序列内，使 48 小时体重 `delta`/`slope` 保留入院锚点（体重变化反映液体平衡）；这是别名而非重复测量，故不做去重。Urine 同时值究竟是独立尿量还是修订值仍需医学组确认。
- “无有效动态观测”提供两个口径：特征工程口径把 00:00 入院 Weight 也纳入 Weight 时间序列；临床审计口径排除这条描述符，避免把静态体重误称为动态测量。
- **生理范围校验已启用（Phase 2）。** `PHYSIOLOGICAL_RANGES` 为 30 余个变量给出临床上下界，越界值按 `-1` 记为缺失，不做截断到边界（避免在边界堆积假值）。两条定向修正先于范围判断执行：pH > 14 视为小数点错位除以 100；Height < 100 cm 视为单位错误乘以 100。`Urine = 0` 与 `MechVent` 缺记录不受影响，仍按真实语义处理。阈值取自临床常识而非训练集拟合，因此不构成数据泄漏；具体数值仍建议医学组复核。
- 没有做标准化、特征筛选、SMOTE 或模型训练；这些属于模型流水线，必须只在训练折内完成。
- 数据只有 stay 级 `RecordID`，无法核查同一患者是否多次住院。
