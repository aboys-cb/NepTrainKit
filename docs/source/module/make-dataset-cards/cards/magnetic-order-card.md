<!-- card-schema: {"card_name": "Magnetic Order", "source_file": "src/NepTrainKit/ui/views/_card/magnetic_order_card.py", "serialized_keys": ["format", "axis", "magmom_map", "use_element_dirs", "default_moment", "apply_elements", "gen_fm", "gen_afm", "afm_mode", "afm_kvec", "afm_group_a", "afm_group_b", "afm_zero_unknown", "gen_pm", "pm_count", "pm_direction", "pm_cone_angle", "pm_balanced", "use_seed", "seed"]} -->

# 磁有序初始化（Magnetic Order）

`Group`: `Magnetism`  
`Class`: `MagneticOrderCard`  
`Source`: `src/NepTrainKit/ui/views/_card/magnetic_order_card.py`

## 功能说明
生成 FM/AFM/PM 初始磁序（magnetic order initialization），用于构建多磁态训练数据基础集。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 模型只会单一磁序，跨磁序泛化差。
- 目标任务 (Target objective): 系统覆盖 FM/AFM/PM 分支。
- 建议添加条件 (Add-it trigger): 研究磁性材料且需多磁序联合训练。
- 不建议添加条件 (Avoid trigger): 非磁任务或固定单一磁序。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先定义 `magmom_map` 或合理 `default_moment`。
- AFM group 模式需先有 Group Label 结果。


## 参数说明（完整）
### `format` (Format)
- UI Label: `Format`
- 字段映射 (Field mapping): 序列化键 `format` <-> 界面标签 `Format`。
- 控件标签 (Caption): `Format`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Collinear (scalar)"`
- 含义 (Meaning): 磁矩格式 (magmom format)。
- 对输出规模/物理性的影响: 决定共线标量还是非共线向量表示。
- 配置建议 (Practical note): `Format` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `axis` (Axis)
- UI Label: `Axis`
- 字段映射 (Field mapping): 序列化键 `axis` <-> 界面标签 `Axis`。
- 控件标签 (Caption): `Axis`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 作用轴/方向 (axis)。
- 对输出规模/物理性的影响: 改变操作方向定义，直接影响输出分布。
- 推荐范围 (Recommended range):
  - 保守：0 到 0，step 1
  - 平衡：0 到 0，step 0.5
  - 探索：0 到 0，step 2

### `magmom_map` (Magmom Map)
- UI Label: `Magmom Map`
- 字段映射 (Field mapping): 序列化键 `magmom_map` <-> 界面标签 `Magmom Map`。
- 控件标签 (Caption): `Magmom Map`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 元素磁矩映射 (element moment map)。
- 对输出规模/物理性的影响: 定义元素到初始磁矩的映射关系。
- 配置建议 (Practical note): `Magmom Map` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `use_element_dirs` (Use Element Dirs)
- UI Label: `Use Element Dirs`
- 字段映射 (Field mapping): 序列化键 `use_element_dirs` <-> 界面标签 `Use Element Dirs`。
- 控件标签 (Caption): `Use Element Dirs`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 元素方向模板开关 (use element directions)。
- 对输出规模/物理性的影响: 允许不同元素采用不同方向先验。
- 配置建议 (Practical note):
  - 开启：需要启用 `Use Element Dirs` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `default_moment` (Default Moment)
- UI Label: `Default Moment`
- 字段映射 (Field mapping): 序列化键 `default_moment` <-> 界面标签 `Default Moment`。
- 控件标签 (Caption): `Default Moment`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): 默认磁矩 (default moment)。
- 对输出规模/物理性的影响: 未命中映射元素时的后备值。
- 推荐范围 (Recommended range):
  - 保守：0.1-0.5
  - 平衡：0.5-1.5
  - 探索：1.5-3.0

### `apply_elements` (Apply Elements)
- UI Label: `Apply Elements`
- 字段映射 (Field mapping): 序列化键 `apply_elements` <-> 界面标签 `Apply Elements`。
- 控件标签 (Caption): `Apply Elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 应用元素列表 (apply elements)。
- 对输出规模/物理性的影响: 限制哪些元素执行当前磁序策略。
- 配置建议 (Practical note): `Apply Elements` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `gen_fm` (Gen Fm)
- UI Label: `Gen Fm`
- 字段映射 (Field mapping): 序列化键 `gen_fm` <-> 界面标签 `Gen Fm`。
- 控件标签 (Caption): `Gen Fm`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 生成 FM 分支 (generate FM)。
- 对输出规模/物理性的影响: 控制是否输出铁磁样本。
- 配置建议 (Practical note):
  - 开启：需要启用 `Gen Fm` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `gen_afm` (Gen AFM)
- UI Label: `Gen AFM`
- 字段映射 (Field mapping): 序列化键 `gen_afm` <-> 界面标签 `Gen AFM`。
- 控件标签 (Caption): `Gen AFM`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 生成 AFM 分支 (generate AFM)。
- 对输出规模/物理性的影响: 控制是否输出反铁磁样本。
- 配置建议 (Practical note):
  - 开启：需要启用 `Gen AFM` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `afm_mode` (AFM Mode)
- UI Label: `AFM Mode`
- 字段映射 (Field mapping): 序列化键 `afm_mode` <-> 界面标签 `AFM Mode`。
- 控件标签 (Caption): `AFM Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"k-vector"`
- 含义 (Meaning): AFM 构造模式 (AFM mode)。
- 对输出规模/物理性的影响: 决定用 k-vector 还是 group 构造正负子晶格。
- 推荐范围 (Recommended range):
  - 保守：k-vector 先跑通
  - 平衡：按结构切 group
  - 探索：混合模式需审计

### `afm_kvec` (AFM Kvec)
- UI Label: `AFM Kvec`
- 字段映射 (Field mapping): 序列化键 `afm_kvec` <-> 界面标签 `AFM Kvec`。
- 控件标签 (Caption): `AFM Kvec`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"111"`
- 含义 (Meaning): AFM k 向量 (AFM k-vector)。
- 对输出规模/物理性的影响: 控制 AFM 反转周期方向。
- 配置建议 (Practical note): `AFM Kvec` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `afm_group_a` (AFM Group A)
- UI Label: `AFM Group A`
- 字段映射 (Field mapping): 序列化键 `afm_group_a` <-> 界面标签 `AFM Group A`。
- 控件标签 (Caption): `AFM Group A`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"A"`
- 含义 (Meaning): AFM A 组标签 (AFM group A)。
- 对输出规模/物理性的影响: group 模式下正向子晶格标签。
- 配置建议 (Practical note): `AFM Group A` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `afm_group_b` (AFM Group B)
- UI Label: `AFM Group B`
- 字段映射 (Field mapping): 序列化键 `afm_group_b` <-> 界面标签 `AFM Group B`。
- 控件标签 (Caption): `AFM Group B`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"B"`
- 含义 (Meaning): AFM B 组标签 (AFM group B)。
- 对输出规模/物理性的影响: group 模式下反向子晶格标签。
- 配置建议 (Practical note): `AFM Group B` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `afm_zero_unknown` (AFM Zero Unknown)
- UI Label: `AFM Zero Unknown`
- 字段映射 (Field mapping): 序列化键 `afm_zero_unknown` <-> 界面标签 `AFM Zero Unknown`。
- 控件标签 (Caption): `AFM Zero Unknown`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 未知元素置零 (zero unknown moments)。
- 对输出规模/物理性的影响: 防止未配置元素引入噪声磁矩。
- 配置建议 (Practical note):
  - 开启：需要启用 `AFM Zero Unknown` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `gen_pm` (Gen PM)
- UI Label: `Gen PM`
- 字段映射 (Field mapping): 序列化键 `gen_pm` <-> 界面标签 `Gen PM`。
- 控件标签 (Caption): `Gen PM`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 生成 PM 分支 (generate PM)。
- 对输出规模/物理性的影响: 控制是否输出顺磁样本。
- 配置建议 (Practical note):
  - 开启：需要启用 `Gen PM` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `pm_count` (PM Count)
- UI Label: `PM Count`
- 字段映射 (Field mapping): 序列化键 `pm_count` <-> 界面标签 `PM Count`。
- 控件标签 (Caption): `PM Count`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[10]`
- 含义 (Meaning): PM 样本数 (PM sample count)。
- 对输出规模/物理性的影响: 控制 PM 分支输出规模。
- 推荐范围 (Recommended range):
  - 保守：5-10
  - 平衡：10-30
  - 探索：30+ 配过滤

### `pm_direction` (PM Direction)
- UI Label: `PM Direction`
- 字段映射 (Field mapping): 序列化键 `pm_direction` <-> 界面标签 `PM Direction`。
- 控件标签 (Caption): `PM Direction`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"sphere"`
- 含义 (Meaning): PM 方向分布 (PM direction distribution)。
- 对输出规模/物理性的影响: 决定顺磁方向采样模式。
- 推荐范围 (Recommended range):
  - 保守：sphere
  - 平衡：cone
  - 探索：定向分布仅专题研究

### `pm_cone_angle` (PM Cone Angle)
- UI Label: `PM Cone Angle`
- 字段映射 (Field mapping): 序列化键 `pm_cone_angle` <-> 界面标签 `PM Cone Angle`。
- 控件标签 (Caption): `PM Cone Angle`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[30.0]`
- 含义 (Meaning): PM 锥角 (PM cone angle)。
- 对输出规模/物理性的影响: 仅在 cone 模式控制偏离主轴幅度。
- 推荐范围 (Recommended range):
  - 保守：15-30
  - 平衡：30-60
  - 探索：60-150

### `pm_balanced` (PM Balanced)
- UI Label: `PM Balanced`
- 字段映射 (Field mapping): 序列化键 `pm_balanced` <-> 界面标签 `PM Balanced`。
- 控件标签 (Caption): `PM Balanced`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): PM 平衡开关 (PM balanced switch)。
- 对输出规模/物理性的影响: 控制方向采样是否保持正负平衡。
- 配置建议 (Practical note):
  - 开启：需要启用 `PM Balanced` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

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
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [
    0.0,
    0.0,
    1.0
  ],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [
    0.0
  ],
  "apply_elements": "",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "afm_group_a": "A",
  "afm_group_b": "B",
  "afm_zero_unknown": true,
  "gen_pm": false,
  "pm_count": [
    10
  ],
  "pm_direction": "sphere",
  "pm_cone_angle": [
    30.0
  ],
  "pm_balanced": true,
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [
    0.0,
    0.0,
    1.0
  ],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [
    0.0
  ],
  "apply_elements": "",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "afm_group_a": "A",
  "afm_group_b": "B",
  "afm_zero_unknown": true,
  "gen_pm": false,
  "pm_count": [
    10
  ],
  "pm_direction": "sphere",
  "pm_cone_angle": [
    30.0
  ],
  "pm_balanced": true,
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [
    0.0,
    0.0,
    1.0
  ],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [
    0.0
  ],
  "apply_elements": "",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "afm_group_a": "A",
  "afm_group_b": "B",
  "afm_zero_unknown": true,
  "gen_pm": true,
  "pm_count": [
    30
  ],
  "pm_direction": "sphere",
  "pm_cone_angle": [
    30.0
  ],
  "pm_balanced": true,
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Group Label -> Magnetic Order -> Magmom Rotation: 形成覆盖 AFM/FM/PM 的完整磁性流程。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- AFM 分配异常：检查 `afm_mode`、`afm_kvec`、group 标签一致性。
- PM 分布偏置：调整 `pm_direction` 与 `pm_balanced`。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `MagFM` / `MagFMnc`
  - `MagAFM...` / `MagAFM...nc` (k-vector or group mode variants)
  - `MagPM...` / `MagPM...nc` (seed suffix may be appended)
- Writes `initial_magmoms` array via `set_initial_magnetic_moments(...)`.


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
