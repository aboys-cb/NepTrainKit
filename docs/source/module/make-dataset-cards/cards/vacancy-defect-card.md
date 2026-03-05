<!-- card-schema: {"card_name": "Vacancy Defect Generation", "source_file": "src/NepTrainKit/ui/views/_card/vacancy_defect_card.py", "serialized_keys": ["engine_type", "num_condition", "num_radio_button", "concentration_radio_button", "concentration_condition", "max_atoms_condition", "use_seed", "seed"]} -->

# 空位缺陷生成（Vacancy Defect Generation）

`Group`: `Defect`  
`Class`: `VacancyDefectCard`  
`Source`: `src/NepTrainKit/ui/views/_card/vacancy_defect_card.py`

## 功能说明
按数量或浓度随机生成空位缺陷（vacancy sampling），快速覆盖缺陷强度分布。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 缺陷密度维度不足，模型对空位数敏感。
- 目标任务 (Target objective): 快速构建低-中-高缺陷强度样本。
- 建议添加条件 (Add-it trigger): 需要高通量空位数据且不需复杂规则。
- 不建议添加条件 (Avoid trigger): 需要按元素/group 精细控制空位位置。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先选 count 或 concentration 单一主模式。
- 控制 `max_atoms_condition` 先小后大。


## 参数说明（完整）
### `engine_type` (Random Engine)
- UI Label: `Random Engine`
- 字段映射 (Field mapping): 序列化键 `engine_type` <-> 界面标签 `Random Engine`。
- 控件标签 (Caption): `Random Engine`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=Sobol`, `1=Uniform`
- 默认值 (Default): `1`
- 含义 (Meaning): 随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。
- 对输出规模/物理性的影响: Uniform 在大批量生成时更快；Sobol 在样本较少时对空位数与位置的覆盖更均衡。样本很多时两者统计差异通常不大。
- 推荐范围 (Recommended range):
  - 保守：小样本覆盖优先 Sobol
  - 平衡：先 Uniform 提速，再用 Sobol 抽样复核分布
  - 探索：超大样本阶段优先速度，固定引擎避免混杂偏差

### `num_condition` (Vacancy Num)
- UI Label: `Vacancy Num`
- 字段映射 (Field mapping): 序列化键 `num_condition` <-> 界面标签 `Vacancy Num`。
- 控件标签 (Caption): `Vacancy Num`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 采样数量控制 (sample count control)。
- 对输出规模/物理性的影响: 主要影响输出规模与耗时，不是幅度主控参数。
- 推荐范围 (Recommended range):
  - 保守：1-1
  - 平衡：1-2
  - 探索：2-5

### `num_radio_button` (Vacancy Num Mode)
- UI Label: `Vacancy Num Mode`
- 字段映射 (Field mapping): 序列化键 `num_radio_button` <-> 界面标签 `Vacancy Num Mode`。
- 控件标签 (Caption): `Vacancy Num Mode`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 计数模式开关 (count mode switch)。
- 对输出规模/物理性的影响: 按绝对数量控制缺陷强度。
- 配置建议 (Practical note):
  - 开启：需要启用 `Vacancy Num Mode` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `concentration_radio_button` (Vacancy Concentration Mode)
- UI Label: `Vacancy Concentration Mode`
- 字段映射 (Field mapping): 序列化键 `concentration_radio_button` <-> 界面标签 `Vacancy Concentration Mode`。
- 控件标签 (Caption): `Vacancy Concentration Mode`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 浓度模式开关 (concentration mode switch)。
- 对输出规模/物理性的影响: 按比例控制缺陷强度。
- 配置建议 (Practical note):
  - 开启：需要启用 `Vacancy Concentration Mode` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `concentration_condition` (Vacancy Concentration)
- UI Label: `Vacancy Concentration`
- 字段映射 (Field mapping): 序列化键 `concentration_condition` <-> 界面标签 `Vacancy Concentration`。
- 控件标签 (Caption): `Vacancy Concentration`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): 空位浓度比例 (vacancy concentration ratio)。取值 `0-1`，可按百分比理解为 `0%-100%`。
- 对输出规模/物理性的影响: 在浓度模式下，最大空位数按 `int(concentration * n_atoms)` 计算；例如 `0.02` 表示约 `2%` 原子被删。
- 推荐范围 (Recommended range):
  - 保守：0.005-0.02（约 0.5%-2%）
  - 平衡：0.02-0.08（约 2%-8%）
  - 探索：0.08-0.20（约 8%-20%，需稳定性评估）

### `max_atoms_condition` (Max Num)
- UI Label: `Max Num`
- 字段映射 (Field mapping): 序列化键 `max_atoms_condition` <-> 界面标签 `Max Num`。
- 控件标签 (Caption): `Max Num`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 每帧最大生成数 (max generated structures per frame)。
- 对输出规模/物理性的影响: 主要控制数据量和运行时间。
- 推荐范围 (Recommended range):
  - 保守：10-50
  - 平衡：50-200
  - 探索：200+ 需 FPS

### `use_seed` (Use Seed)
- UI Label: `Use Seed`
- 字段映射 (Field mapping): 序列化键 `use_seed` <-> 界面标签 `Use Seed`。
- 控件标签 (Caption): `Use Seed`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 是否启用固定随机种子 (deterministic seed switch)。
- 对输出规模/物理性的影响: 开启后可复现实验；关闭后每次采样分布会变化。
- 配置建议 (Practical note):
  - 开启：需要可复现对比时开启。
  - 关闭：探索阶段可关闭以增加随机覆盖。

### `seed` (Seed)
- UI Label: `Seed`
- 字段映射 (Field mapping): 序列化键 `seed` <-> 界面标签 `Seed`。
- 控件标签 (Caption): `Seed`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[0]`
- 含义 (Meaning): 随机种子值 (random seed value)。
- 对输出规模/物理性的影响: 只影响随机路径，不改变物理模型本身。
- 推荐范围 (Recommended range):
  - 保守：0（随机）
  - 平衡：1-99（可复现）
  - 探索：100-9999（多 seed 对比）


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 1,
  "num_condition": [
    2
  ],
  "num_radio_button": false,
  "concentration_radio_button": true,
  "concentration_condition": [
    0.02
  ],
  "max_atoms_condition": [
    100
  ],
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 1,
  "num_condition": [
    4
  ],
  "num_radio_button": false,
  "concentration_radio_button": true,
  "concentration_condition": [
    0.05
  ],
  "max_atoms_condition": [
    100
  ],
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 0,
  "num_condition": [
    8
  ],
  "num_radio_button": false,
  "concentration_radio_button": true,
  "concentration_condition": [
    0.1
  ],
  "max_atoms_condition": [
    100
  ],
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Vacancy Defect Generation -> Insert Defect: 联合采样空位与插隙缺陷族。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 强度超预期：检查模式开关与参数冲突。
- 重复度高：调整引擎或 seed 策略。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Vac(n={...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
