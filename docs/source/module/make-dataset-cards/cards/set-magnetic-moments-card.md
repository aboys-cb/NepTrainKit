<!-- card-schema: {"card_name": "Set Magnetic Moments", "source_file": "src/NepTrainKit/ui/views/_card/set_magnetic_moments_card.py", "serialized_keys": ["source", "format", "axis", "magmom_map", "use_element_dirs", "default_moment", "constant_moment", "lift_scalar", "apply_elements"]} -->

# 设置磁矩（Set Magnetic Moments）

`Group`: `Magnetism`  
`Class`: `SetMagneticMomentsCard`  
`Source`: `src/NepTrainKit/ui/views/_card/set_magnetic_moments_card.py`

## 功能说明
这是一张基础磁性卡，专门负责把输入结构写成统一的 `initial_magmoms` 表示。它只处理“磁矩从哪里来、写成标量还是向量、沿哪个轴初始化、哪些元素参与”这些基础设置，不负责 AFM、PM、spin spiral 或局部偏转等后续磁性变换。

最小可运行示例：把 `Source` 设为 `Map/default magnitude`、`Format` 设为 `Non-collinear (vector)`、在 `Magmom map` 里填 `Fe:2.2`，对一帧含 Fe 的结构运行后检查 `initial_magmoms` 是否变成三列向量，并确认 `Config_type` 出现 `MagSet(map,vec)`。

:::{tip}
高通量示例：在磁性数据流程最前面先放一张本卡，统一把所有输入结构初始化成相同的磁矩格式，再串接 `Magnetic Order`、`Small-Angle Spin Tilt`、`Spin Spiral` 或 `Magmom Rotation`，这样后续磁性卡就可以优先消费已有磁矩，减少重复参数和分支逻辑。
:::

### 关键公式 (Core equations)
若目标输出为向量格式，代码按

$$
\mathbf{m}_i = |\mathbf{m}_i|\,\hat{\mathbf{a}}
$$

初始化参考 FM 方向，其中 $\hat{\mathbf{a}}$ 由 `axis` 给出；若元素映射里提供了向量并开启 `use_element_dirs`，则改为保留其归一化方向。若输入是标量磁矩且开启 `lift_scalar`，则使用

$$
\mathbf{m}_i = m_i\,\hat{\mathbf{a}}
$$

把标量 `initial_magmoms` 抬升为向量。

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 多张磁性卡都在重复设置 `magmom_map`、`axis`、标量抬升等基础参数，导致流程冗余且容易配错。
- 目标任务 (Target objective): 在磁性流程最前面统一初始化 `initial_magmoms`，让后续磁性卡优先基于已有磁矩做变换。
- 建议添加条件 (Add-it trigger): 输入结构还没有磁矩，或者不同上游来源的磁矩格式不统一，需要先标准化成统一的 scalar/vector 形式。
- 不建议添加条件 (Avoid trigger): 你当前只需要一次性的 AFM/PM/spiral 构造，而且并不想单独维护一层基础磁矩初始化卡。
> 设计提示 (Workflow caution): 推荐把本卡作为“基础层”，而不是要求所有后续卡强依赖它；后续磁性卡最好仍保留对缺失磁矩的温和回退。

## 输入前提
- 若 `Source=Existing initial magmoms`，输入结构最好已经带有可解析的 `initial_magmoms`。
- 若 `Source=Map/default magnitude`，请提供 `magmom_map` 或合理的 `default_moment`。
- 若只想初始化某些元素，可通过 `apply_elements` 收缩作用范围；未选中的元素会被写成零磁矩。

## 参数说明（完整）
### `source` (Source)
- UI Label: `Source`
- 字段映射 (Field mapping): 序列化键 `source` <-> 界面标签 `Source`。
- 控件标签 (Caption): `Source`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Map/default magnitude"`
- 含义 (Meaning): 选择磁矩来源：沿用已有磁矩、按元素映射生成，或用统一常数幅值生成。
- 对输出规模/物理性的影响: 不改变输出条数，但决定磁矩初始化的来源与参考假设。
- 推荐范围 (Recommended range):
- 保守：`Map/default magnitude`
- 均衡：`Existing initial magmoms` 或 `Map/default magnitude`
- 探索：`Constant magnitude`，用于快速扫描统一参考磁矩幅值

### `format` (Format)
- UI Label: `Format`
- 字段映射 (Field mapping): 序列化键 `format` <-> 界面标签 `Format`。
- 控件标签 (Caption): `Format`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Non-collinear (vector)"`
- 含义 (Meaning): 选择输出写成标量 `MAGMOM` 还是三列向量 `MAGMOM`。
- 对输出规模/物理性的影响: 不改变样本数，但会决定后续磁性卡接收到的是共线表示还是非共线表示。
- 配置建议 (Practical note): 如果后续会做 `Small-Angle Spin Tilt`、`Spin Spiral` 或向量旋转，优先使用 `Non-collinear (vector)`；若只做共线磁序，可保留 `Collinear (scalar)`。

### `axis` (Axis)
- UI Label: `Axis (x,y,z)`
- 字段映射 (Field mapping): 序列化键 `axis` <-> 界面标签 `Axis (x,y,z)`。
- 控件标签 (Caption): `Axis (x,y,z)`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（3 个输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 向量输出和标量抬升时使用的参考轴。
- 对输出规模/物理性的影响: 会改变 FM 参考方向，以及向量转标量时的投影方向。
- 推荐范围 (Recommended range):
- 保守：`[0.0, 0.0, 1.0]`
- 均衡：`[1.0, 0.0, 0.0]`、`[0.0, 1.0, 0.0]`、`[0.0, 0.0, 1.0]`
- 探索：按研究问题选择其他晶向，如 `[1.0, 1.0, 0.0]`

### `magmom_map` (Magmom Map)
- UI Label: `Magmom map`
- 字段映射 (Field mapping): 序列化键 `magmom_map` <-> 界面标签 `Magmom map`。
- 控件标签 (Caption): `Magmom map`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 元素到磁矩的映射，可写标量如 `Fe:2.2`，也可写 JSON 向量如 `{"Cr":[0,0,1.0]}`。
- 对输出规模/物理性的影响: 不改变样本数，但直接决定每个元素的参考磁矩幅值与可选方向。
- 配置建议 (Practical note): 建议显式写出主要磁性元素；如果想保留元素专属方向，请配合 `use_element_dirs=true` 且使用向量形式的 map。

### `use_element_dirs` (Use Element Dirs)
- UI Label: `Use element vector directions`
- 字段映射 (Field mapping): 序列化键 `use_element_dirs` <-> 界面标签 `Use element vector directions`。
- 控件标签 (Caption): `Use element vector directions`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 当 `magmom_map` 提供向量时，是否保留这些元素自带的方向，而不是统一投到 `axis`。
- 对输出规模/物理性的影响: 不改变样本数，但会改变不同元素的参考磁矩方向分布。
- 配置建议 (Practical note):
  - 开启：元素映射里已经给出有物理意义的向量方向时开启。
  - 关闭：只想统一生成同轴 FM 参考态时关闭。

### `default_moment` (Default Moment)
- UI Label: `Default |m|`
- 字段映射 (Field mapping): 序列化键 `default_moment` <-> 界面标签 `Default |m|`。
- 控件标签 (Caption): `Default |m|`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): `magmom_map` 未命中的元素默认使用的磁矩幅值。
- 对输出规模/物理性的影响: 不改变输出条数，但会决定遗漏元素是否被视为无磁或带磁。
- 推荐范围 (Recommended range):
- 保守：`0.0`
- 均衡：`0.5` 到 `2.5`
- 探索：`0.0` 到 `5.0`

### `constant_moment` (Constant Moment)
- UI Label: `Constant |m|`
- 字段映射 (Field mapping): 序列化键 `constant_moment` <-> 界面标签 `Constant |m|`。
- 控件标签 (Caption): `Constant |m|`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[2.0]`
- 含义 (Meaning): 当 `Source=Constant magnitude` 时，对所有目标元素统一赋予的磁矩幅值。
- 对输出规模/物理性的影响: 只影响参考磁矩大小，不影响样本数。
- 推荐范围 (Recommended range):
- 保守：`1.0`
- 均衡：`1.5` 到 `3.0`
- 探索：`0.5` 到 `5.0`

### `lift_scalar` (Lift Scalar)
- UI Label: `Lift scalar magmoms to vectors`
- 字段映射 (Field mapping): 序列化键 `lift_scalar` <-> 界面标签 `Lift scalar magmoms to vectors`。
- 控件标签 (Caption): `Lift scalar magmoms to vectors`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 当来源是已有标量磁矩且目标格式是向量时，是否沿 `axis` 抬升为向量。
- 对输出规模/物理性的影响: 关闭后，已有标量磁矩将无法直接产出向量格式。
- 配置建议 (Practical note):
  - 开启：已有输入是共线标量磁矩，但后续流程需要向量磁矩时开启。
  - 关闭：只有在你明确不希望自动抬升标量磁矩时才关闭。

### `apply_elements` (Apply Elements)
- UI Label: `Apply elements`
- 字段映射 (Field mapping): 序列化键 `apply_elements` <-> 界面标签 `Apply elements`。
- 控件标签 (Caption): `Apply elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 可选元素白名单；非空时只对这些元素写入磁矩，其余元素清零。
- 对输出规模/物理性的影响: 不改变样本条数，但会改变哪些原子拥有非零磁矩。
- 配置建议 (Practical note): 多元素体系中，若只想初始化磁性子晶格，可填如 `Fe,Co`；留空则默认全部元素都参与初始化。

## 推荐预设（可直接复制 JSON）
### Safe
```json
{
  "class": "SetMagneticMomentsCard",
  "check_state": true,
  "source": "Map/default magnitude",
  "format": "Non-collinear (vector)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2",
  "use_element_dirs": false,
  "default_moment": [0.0],
  "constant_moment": [2.0],
  "lift_scalar": true,
  "apply_elements": ""
}
```

### Balanced
```json
{
  "class": "SetMagneticMomentsCard",
  "check_state": true,
  "source": "Existing initial magmoms",
  "format": "Non-collinear (vector)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [0.0],
  "constant_moment": [2.0],
  "lift_scalar": true,
  "apply_elements": ""
}
```

### Aggressive/Exploration
```json
{
  "class": "SetMagneticMomentsCard",
  "check_state": true,
  "source": "Constant magnitude",
  "format": "Non-collinear (vector)",
  "axis": [1.0, 0.0, 0.0],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [0.0],
  "constant_moment": [3.0],
  "lift_scalar": true,
  "apply_elements": "Fe,Co"
}
```

## 推荐组合
- `Set Magnetic Moments -> Magnetic Order`: 先统一初始化磁矩格式，再生成 FM/AFM/PM 等磁序。
- `Set Magnetic Moments -> Small-Angle Spin Tilt`: 先生成统一的 FM 向量磁矩，再做单自旋小角度偏转。
- `Set Magnetic Moments -> Spin Spiral`: 先写入稳定的磁矩模长，再生成 helix / spiral 初态。

## 常见问题与排查
- 运行后仍然没有磁矩：检查 `Source` 是否选错，或 `apply_elements` 是否把目标元素过滤掉了。
- 向量输出方向不符合预期：检查 `axis`，以及在 map 向量输入场景下是否误开了 `use_element_dirs`。
- 从已有磁矩转向量失败：通常是输入只有标量磁矩且 `lift_scalar=false`。
- 某些元素被意外清零：确认它们是否不在 `apply_elements` 中，或没有在 `magmom_map` 中命中且 `default_moment=0.0`。

## 输出标签 / 元数据变更
- 该卡片会通过 `set_initial_magnetic_moments(...)` 写入统一格式的 `initial_magmoms`。
- `Config_type` 会追加 `MagSet(existing,sca)`、`MagSet(map,vec)`、`MagSet(const,vec)` 这类标签，用来标记磁矩来源与输出格式。

## 可复现性说明
- 本卡没有随机采样；相同输入、相同 `source` 和相同参数会得到完全一致的输出。
- 若来源是 `Existing initial magmoms` 且目标格式是标量，向量输入会按 `axis` 做确定性投影。
- 本卡只负责基础磁矩初始化或规范化，不引入 AFM/PM/spiral 等更高层磁性结构假设。
