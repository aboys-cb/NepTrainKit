<!-- card-schema: {"card_name": "Vib Mode Perturb", "source_file": "src/NepTrainKit/ui/views/_card/vibration_perturb_card.py", "serialized_keys": ["distribution", "amplitude", "modes_per_sample", "min_frequency", "max_num", "scale_by_frequency", "exclude_near_zero", "use_seed", "seed"]} -->

# 振动模态扰动（Vib Mode Perturb）

`Group`: `Perturbation`  
`Class`: `VibrationModePerturbCard`  
`Source`: `src/NepTrainKit/ui/views/_card/vibration_perturb_card.py`

## 功能说明
沿振动模方向施加位移扰动（vibrational mode perturbation），比纯随机扰动更贴近动力学自由度。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\mathbf{r}'=\mathbf{r}+A\sum_{k\in\mathcal{K}} c_k\mathbf{u}_k$$
$$c_k\sim\mathcal{N}(0,1)\ \text{or}\ \mathcal{U}(-1,1),\quad c_k\leftarrow\frac{c_k}{\sqrt{|\omega_k|}}\ (\text{when scale\_by\_frequency=true})$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 纯随机扰动不足以覆盖模态方向。
- 目标任务 (Target objective): 强化声子/振动相关结构覆盖。
- 建议添加条件 (Add-it trigger): 需要模态驱动的位移样本。
- 不建议添加条件 (Avoid trigger): 缺少可信振动模式输入。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 确认模态输入质量和单位一致。
- 先小 `amplitude` + 低 `modes_per_sample` 验证。


## 参数说明（完整）
### `distribution` (Distribution)
- UI Label: `Distribution`
- 字段映射 (Field mapping): 序列化键 `distribution` <-> 界面标签 `Distribution`。
- 控件标签 (Caption): `Distribution`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `0`
- 含义 (Meaning): 采样分布类型 (distribution type)。
- 对输出规模/物理性的影响: 决定随机变量分布形状。
- 推荐范围 (Recommended range):
  - 保守：均匀分布
  - 平衡：高斯分布
  - 探索：重尾分布仅探索

### `amplitude` (Amplitude)
- UI Label: `Amplitude`
- 字段映射 (Field mapping): 序列化键 `amplitude` <-> 界面标签 `Amplitude`。
- 控件标签 (Caption): `Amplitude`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.05]`
- 含义 (Meaning): 模态位移幅度 (mode displacement amplitude)。
- 对输出规模/物理性的影响: 主控振动扰动强度，过大易进入高能异常区。
- 推荐范围 (Recommended range):
  - 保守：0.01-0.03
  - 平衡：0.04-0.07
  - 探索：0.1+ 需后筛

### `modes_per_sample` (Modes Per Sample)
- UI Label: `Modes Per Sample`
- 字段映射 (Field mapping): 序列化键 `modes_per_sample` <-> 界面标签 `Modes Per Sample`。
- 控件标签 (Caption): `Modes Per Sample`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[2]`
- 含义 (Meaning): 每样本叠加模态数 (modes per sample)。
- 对输出规模/物理性的影响: 模态数越高，扰动方向组合越复杂。
- 推荐范围 (Recommended range):
  - 保守：1-2
  - 平衡：3-4
  - 探索：5+

### `min_frequency` (Min Frequency)
- UI Label: `Min Frequency`
- 字段映射 (Field mapping): 序列化键 `min_frequency` <-> 界面标签 `Min Frequency`。
- 控件标签 (Caption): `Min Frequency`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[10.0]`
- 含义 (Meaning): 最小频率阈值 (minimum frequency)。
- 对输出规模/物理性的影响: 过滤过低频模式，减少软模异常。
- 推荐范围 (Recommended range):
  - 保守：5-10
  - 平衡：10-20
  - 探索：20-50

### `max_num` (Max Num)
- UI Label: `Max Num`
- 字段映射 (Field mapping): 序列化键 `max_num` <-> 界面标签 `Max Num`。
- 控件标签 (Caption): `Max Num`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[32]`
- 含义 (Meaning): 每帧最大样本数 (max samples per frame)。
- 对输出规模/物理性的影响: 控制输出规模。
- 推荐范围 (Recommended range):
  - 保守：10-20
  - 平衡：20-60
  - 探索：100+

### `scale_by_frequency` (Scale By Frequency)
- UI Label: `Scale By Frequency`
- 字段映射 (Field mapping): 序列化键 `scale_by_frequency` <-> 界面标签 `Scale By Frequency`。
- 控件标签 (Caption): `Scale By Frequency`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 按频率缩放开关 (scale by frequency)。
- 对输出规模/物理性的影响: 开启后高频模位移更小，通常更物理。
- 配置建议 (Practical note):
  - 开启：需要启用 `Scale By Frequency` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `exclude_near_zero` (Exclude Near Zero)
- UI Label: `Exclude Near Zero`
- 字段映射 (Field mapping): 序列化键 `exclude_near_zero` <-> 界面标签 `Exclude Near Zero`。
- 控件标签 (Caption): `Exclude Near Zero`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 排除近零频开关 (exclude near-zero)。
- 对输出规模/物理性的影响: 减少平移/旋转伪模引起的异常位移。
- 配置建议 (Practical note):
  - 开启：需要启用 `Exclude Near Zero` 对应行为时开启。
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
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "distribution": 0,
  "amplitude": [
    0.02
  ],
  "modes_per_sample": [
    2
  ],
  "min_frequency": [
    0.0
  ],
  "max_num": [
    20
  ],
  "scale_by_frequency": true,
  "exclude_near_zero": true,
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "distribution": 0,
  "amplitude": [
    0.05
  ],
  "modes_per_sample": [
    3
  ],
  "min_frequency": [
    0.0
  ],
  "max_num": [
    20
  ],
  "scale_by_frequency": true,
  "exclude_near_zero": true,
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "distribution": 0,
  "amplitude": [
    0.12
  ],
  "modes_per_sample": [
    4
  ],
  "min_frequency": [
    0.0
  ],
  "max_num": [
    20
  ],
  "scale_by_frequency": true,
  "exclude_near_zero": true,
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Vib Mode Perturb -> Lattice Perturb: 在模态位移后追加轻量晶胞变化。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 软模导致异常位移：提高 `min_frequency` 或排除近零频。
- 样本差异不足：先增加模态数，再考虑增大幅度。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Vib(a={...},m={...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
