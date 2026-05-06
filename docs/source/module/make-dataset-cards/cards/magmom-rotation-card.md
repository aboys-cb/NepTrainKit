<!-- card-schema: {"card_name": "Magmom Rotation", "source_file": "src/NepTrainKit/ui/views/_card/magmom_rotation_card.py", "serialized_keys": ["params"]} -->

# 磁矩旋转（Magmom Rotation）

`Group`: `Perturbation`  
`Class`: `MagneticMomentRotationCard`  
`Source`: `src/NepTrainKit/ui/views/_card/magmom_rotation_card.py`

## 功能说明
旋转指定元素的磁矩方向并可扰动模长，构建连续磁构型邻域数据。

它最适合的场景是：在已有磁矩基础上做小角度旋转，补充局部自旋偏转样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\mathbf{m}'=\lambda\,\mathbf{R}(\hat{\mathbf{n}},\theta)\,\mathbf{m},\quad \lambda\in[f_{\min},f_{\max}]$$
$$\mathbf{R}(\hat{\mathbf{n}},\theta)=\cos\theta\,\mathbf{I}+(1-\cos\theta)\hat{\mathbf{n}}\hat{\mathbf{n}}^\top+\sin\theta\,[\hat{\mathbf{n}}]_{\times}$$

## 操作示例
### 场景：在已有磁矩基础上做小角度旋转，补充局部自旋偏转样本

**输入：** 已经写入初始磁矩的磁性结构

**目标：** 围绕参考轴生成一批旋转角不同的磁矩构型，而不重新定义 FM/AFM 分支

**参数设置：**
- `max_angle` 先从 5-15 度这种小角度开始
- `num_structures` 决定每个输入额外扩出多少个旋转版本
- `lift_scalar=true` 只在输入是共线标量磁矩时需要开启

**输出：** 多份磁矩方向稍有差异的结构；原子位置不变，变化主要在 `initial_magmoms`

**怎么验证结果合理：**
- 检查磁矩模长是否基本保持
- 确认旋转轴与 `axis` 一致
- 若输出没有变化，先检查输入结构是否真的已有初始磁矩

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 非共线磁方向相关误差高。
- 目标任务 (Target objective): 在已有磁序附近扩展方向和模长自由度。
- 建议添加条件 (Add-it trigger): 关注磁各向异性或自旋动力学相关任务。
- 不建议添加条件 (Avoid trigger): 非磁体系或无磁矩训练目标。
> 物理提示 (Physics caution): 重点检查位移后最短键长和局部角度；幅度先小后大，比一次性追求大覆盖更稳妥。

## 输入前提
- 输入结构需包含可用初始磁矩。
- 先从小角度 `max_angle` 开始。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由元素、旋转角、输出数量、标量抬升、参考轴、模长扰动和 seed 控件组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"elements": "", "max_angle": 10.0, "num_structures": 5, "lift_scalar": true, "axis": [0.0, 0.0, 1.0], "disturb_magnitude": true, "magnitude_factor": [0.95, 1.05], "use_seed": false, "seed": 0}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组磁矩旋转参数。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

### `elements` (Elements)
- UI Label: `Elements`
- 字段映射 (Field mapping): 序列化键 `elements` <-> 界面标签 `Elements`。
- 控件标签 (Caption): `Elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 元素集合输入 (element set)。
- 对输出规模/物理性的影响: 决定参与操作的元素子集。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
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
- 物理直觉 / 典型值: 它通常是控制变化幅度的主旋钮；先从能看清趋势的小幅度起步，再决定是否扩到探索档。
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
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 怎么判断该开还是该关: 做可复现实验或要对比不同参数时开启；纯探索阶段可以先关闭以增加随机覆盖。
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
- 参数联动 / 生效条件: `seed` 只有在 `use_seed=true` 时才真正固定随机路径。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 与晶格类卡片串联时，先做晶格变化，再补局部位移噪声。

## 常见问题与排查
- 输出为空时，优先检查输入是否满足这张卡的前提，例如是否已有振动模态、是否启用了正确的模式。
- 如果出现短键、断键或明显高能构型，先降低主控位移幅度，再缩小每帧样本数做抽样检查。
- 随机种子只控制采样路径，不会自动修正非物理参数；参数过激时程序仍会按当前配置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `MMR(a=...,s=...)` when vector rotation is active.
  - `MMS(s=...)` when only magnitude scaling is active.
- Writes `initial_magmoms` array via `set_initial_magnetic_moments(...)`.

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
