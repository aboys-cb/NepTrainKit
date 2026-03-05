<!-- card-schema: {"card_name": "Magmom Rotation", "source_file": "src/NepTrainKit/ui/views/_card/magmom_rotation_card.py", "serialized_keys": ["elements", "max_angle", "num_structures", "lift_scalar", "axis", "disturb_magnitude", "magnitude_factor", "use_seed", "seed"]} -->

# 磁矩旋转（Magmom Rotation）

`Group`: `Perturbation`  
`Class`: `MagneticMomentRotationCard`  
`Source`: `src/NepTrainKit/ui/views/_card/magmom_rotation_card.py`

## 功能说明
旋转指定元素的磁矩方向并可扰动模长，构建连续磁构型邻域数据。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\mathbf{m}'=\lambda\,\mathbf{R}(\hat{\mathbf{n}},\theta)\,\mathbf{m},\quad \lambda\in[f_{\min},f_{\max}]$$
$$\mathbf{R}(\hat{\mathbf{n}},\theta)=\cos\theta\,\mathbf{I}+(1-\cos\theta)\hat{\mathbf{n}}\hat{\mathbf{n}}^\top+\sin\theta\,[\hat{\mathbf{n}}]_{\times}$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 非共线磁方向相关误差高。
- 目标任务 (Target objective): 在已有磁序附近扩展方向和模长自由度。
- 建议添加条件 (Add-it trigger): 关注磁各向异性或自旋动力学相关任务。
- 不建议添加条件 (Avoid trigger): 非磁体系或无磁矩训练目标。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 输入结构需包含可用初始磁矩。
- 先从小角度 `max_angle` 开始。


## 参数说明（完整）
### `elements` (Elements)
- UI Label: `Elements`
- 字段映射 (Field mapping): 序列化键 `elements` <-> 界面标签 `Elements`。
- 控件标签 (Caption): `Elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 元素集合输入 (element set)。
- 对输出规模/物理性的影响: 决定参与操作的元素子集。
- 配置建议 (Practical note): `Elements` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `max_angle` (Max Angle)
- UI Label: `Max Angle`
- 字段映射 (Field mapping): 序列化键 `max_angle` <-> 界面标签 `Max Angle`。
- 控件标签 (Caption): `Max Angle`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[10.0]`
- 含义 (Meaning): 最大旋转角 (max rotation angle)。
- 对输出规模/物理性的影响: 主控磁方向扰动强度，角度越大偏离基态越远。
- 推荐范围 (Recommended range):
  - 保守：2-5°
  - 平衡：8-15°
  - 探索：20°+ 需重点筛查

### `num_structures` (Num Structures)
- UI Label: `Num Structures`
- 字段映射 (Field mapping): 序列化键 `num_structures` <-> 界面标签 `Num Structures`。
- 控件标签 (Caption): `Num Structures`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[5]`
- 含义 (Meaning): 每帧输出结构数 (structures per frame)。
- 对输出规模/物理性的影响: 影响数据体量，不直接决定单样本幅度。
- 推荐范围 (Recommended range):
  - 保守：5-10
  - 平衡：10-30
  - 探索：30+ 配过滤

### `lift_scalar` (Lift Scalar)
- UI Label: `Lift Scalar`
- 字段映射 (Field mapping): 序列化键 `lift_scalar` <-> 界面标签 `Lift Scalar`。
- 控件标签 (Caption): `Lift Scalar`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 标量提升开关 (lift scalar)。
- 对输出规模/物理性的影响: 控制标量输入是否映射到向量表示。
- 配置建议 (Practical note):
  - 开启：需要启用 `Lift Scalar` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

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

### `disturb_magnitude` (Disturb Magnitude)
- UI Label: `Disturb Magnitude`
- 字段映射 (Field mapping): 序列化键 `disturb_magnitude` <-> 界面标签 `Disturb Magnitude`。
- 控件标签 (Caption): `Disturb Magnitude`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 模长扰动开关 (disturb magnitude)。
- 对输出规模/物理性的影响: 开启后会拓宽磁矩长度分布。
- 配置建议 (Practical note):
  - 开启：需要启用 `Disturb Magnitude` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `magnitude_factor` (Magnitude Factor)
- UI Label: `Magnitude Factor`
- 字段映射 (Field mapping): 序列化键 `magnitude_factor` <-> 界面标签 `Magnitude Factor`。
- 控件标签 (Caption): `Magnitude Factor`。
- 控件解释 (Widget): 按字段类型解析。
- 类型/范围 (Type/Range): list[2]
- 默认值 (Default): `[0.95, 1.05]`
- 含义 (Meaning): 模长缩放区间 (magnitude factor range)。
- 对输出规模/物理性的影响: 区间越宽，磁矩长度分布越发散。
- 推荐范围 (Recommended range):
  - 保守：0.98-1.02
  - 平衡：0.95-1.05
  - 探索：0.85-1.15

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
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "elements": "",
  "max_angle": [
    3.0
  ],
  "num_structures": [
    5
  ],
  "lift_scalar": true,
  "axis": [
    0.0,
    0.0,
    1.0
  ],
  "disturb_magnitude": true,
  "magnitude_factor": [
    0.98,
    1.02
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
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "elements": "",
  "max_angle": [
    10.0
  ],
  "num_structures": [
    5
  ],
  "lift_scalar": true,
  "axis": [
    0.0,
    0.0,
    1.0
  ],
  "disturb_magnitude": true,
  "magnitude_factor": [
    0.95,
    1.05
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
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "elements": "",
  "max_angle": [
    25.0
  ],
  "num_structures": [
    5
  ],
  "lift_scalar": true,
  "axis": [
    0.0,
    0.0,
    1.0
  ],
  "disturb_magnitude": true,
  "magnitude_factor": [
    0.85,
    1.15
  ],
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Magnetic Order -> Magmom Rotation: 先生成有序种子，再采样方向变化。
- Group Label -> Magnetic Order -> Magmom Rotation: 在磁矩扰动前保留子晶格上下文。


## 常见问题与排查
- 磁矩异常跳变：降低 `max_angle` 与 `magnitude_factor` 范围。
- 样本膨胀：保持 `num_structures`，优先调幅度而非数量。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `MMR(a=...,s=...)` when vector rotation is active.
  - `MMS(s=...)` when only magnitude scaling is active.
- Writes `initial_magmoms` array via `set_initial_magnetic_moments(...)`.


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
