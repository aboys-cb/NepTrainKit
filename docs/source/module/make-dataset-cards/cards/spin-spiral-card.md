<!-- card-schema: {"card_name": "Spin Spiral", "source_file": "src/NepTrainKit/ui/views/_card/spin_spiral_card.py", "serialized_keys": ["axis", "spiral_parameter_mode", "period_range", "angle_gradient_range", "phase_range", "mz", "chirality", "magnitude_source", "magmom_map", "default_moment", "apply_elements", "max_outputs"]} -->

# 自旋螺旋初始化（Spin Spiral）

`Group`: `Magnetism`  
`Class`: `SpinSpiralCard`  
`Source`: `src/NepTrainKit/ui/views/_card/spin_spiral_card.py`

## 功能说明
对输入结构按一维相位场写入非共线 `initial_magmoms`，生成一批不同周期、不同起始相位、不同手性的 spin spiral / helix 初始构型。卡片支持两种等价的主控方式：直接扫周期 `L_D`，或直接扫每 Å 转角 `angle gradient`，二者只需要选一个。

最小可运行示例：先把 **保守预设（Safe）** 应用到一帧含磁性元素的结构，只扫描一个 `period_range` 或一个 `angle_gradient_range`，确认导出的 `initial_magmoms` 是三列向量，并检查 `Config_type` 是否带有 `Helix(...)` 或 `Spiral(...)` 标签。

:::{tip}
高通量示例：固定一组局域磁矩模长，只扫描 `spiral_parameter_mode + period_range/angle_gradient_range + chirality` 生成成对构型；导出后先用快速单点或已有势函数筛掉明显异常的磁矩初态，再进入大规模训练或筛选流程。
:::

### 关键公式 (Core equations)
设传播轴单位向量为 $\hat{\mathbf{n}}$，其正交基为 $(\mathbf{e}_1,\mathbf{e}_2,\hat{\mathbf{n}})$，则代码中采用

$$
\mathbf{m}(u)=\sqrt{1-m_z^2}\left[\cos\phi(u)\,\mathbf{e}_1+\sin\phi(u)\,\mathbf{e}_2\right] + m_z\,\hat{\mathbf{n}}
$$

$$
\phi(u)=s\cdot \frac{2\pi u}{L_D}+\phi_0,\qquad s\in\{-1,+1\}
$$

并且

$$
g=\frac{360^\circ}{L_D}
$$

其中 $u$ 是原子坐标在传播轴上的投影，$L_D$ 对应 `period_range`，$g$ 对应 `angle_gradient_range`，$\phi_0$ 对应 `phase_range`。当 `mz=0` 时，该构型退化为平面 helix。

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 已有数据只覆盖单一磁矩方向，模型对螺旋态、手性成对态或 conical spiral 初态不敏感。
- 目标任务 (Target objective): 在固定晶体结构上系统扫描 spiral 周期、每 Å 转角或 DMI 相关手性分支。
- 建议添加条件 (Add-it trigger): 你已经知道几何结构不需要扩胞复制，只想在输入结构上生成一批非共线磁矩初态。
- 不建议添加条件 (Avoid trigger): 体系没有磁矩自由度，或任务本质上需要真实长周期超胞而不是仅改变初始自旋纹理。
> 物理提示 (Physics caution): 该卡片只写入初始磁矩，不会自动保证该 spiral 与交换参数、晶格尺度或实际基态严格相容。

## 输入前提
- 输入结构应至少包含一类有意义的磁性元素。
- 若 `magnitude_source` 选 `Existing initial magmoms`，输入结构最好已经带有 `initial_magmoms`。
- 若 `magnitude_source` 选 `Map/default magnitude`，需要提供 `magmom_map` 或合理的 `default_moment`。

## 参数说明（完整）
### `axis` (Propagation Axis)
- UI Label: `Propagation axis`
- 字段映射 (Field mapping): 序列化键 `axis` <-> 界面标签 `Propagation axis`。
- 控件标签 (Caption): `Propagation axis`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（3 个输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 自旋螺旋相位的传播方向。
- 对输出规模/物理性的影响: 改变沿哪个方向计算相位投影，也就改变了同一结构上自旋旋转的空间分布。
- 推荐范围 (Recommended range):
  - 保守：`[0, 0, 1]`
  - 平衡：`[1, 0, 0]`、`[0, 1, 0]`、`[0, 0, 1]`
  - 探索：按研究问题扫描任意归一化方向，如 `[1, 1, 0]`

### `spiral_parameter_mode` (Spiral Parameter)
- UI Label: `Spiral parameter`
- 字段映射 (Field mapping): 序列化键 `spiral_parameter_mode` <-> 界面标签 `Spiral parameter`。
- 控件标签 (Caption): `Spiral parameter`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Period (L_D)"`
- 含义 (Meaning): 选择用周期 `L_D` 还是用每 Å 转角来定义 spiral。
- 对输出规模/物理性的影响: 不改变结果数，但决定你在界面里操作的是哪一个等价参数。
- 配置建议 (Practical note): 二选一即可，不需要同时把 `period_range` 和 `angle_gradient_range` 都当主控量来调。

### `period_range` (Period Range)
- UI Label: `Period range`
- 字段映射 (Field mapping): 序列化键 `period_range` <-> 界面标签 `Period range`。
- 控件标签 (Caption): `Period range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[20.0, 40.0, 10.0]`
- 含义 (Meaning): spiral 周期 $L_D$ 的扫描区间，单位为 Å；仅在 `spiral_parameter_mode="Period (L_D)"` 时生效。
- 对输出规模/物理性的影响: 周期越短，相邻位置的相位变化越快；同时也会线性增加输出数量。
- 推荐范围 (Recommended range):
  - 保守：`[20.0, 20.0, 5.0]`
  - 平衡：`[10.0, 40.0, 10.0]`
  - 探索：`[4.0, 80.0, 4.0]`

### `angle_gradient_range` (Angle Gradient Range)
- UI Label: `Angle gradient range`
- 字段映射 (Field mapping): 序列化键 `angle_gradient_range` <-> 界面标签 `Angle gradient range`。
- 控件标签 (Caption): `Angle gradient range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[18.0, 18.0, 1.0]`
- 含义 (Meaning): 每 Å 的转角梯度 $g$，单位为 deg/Å；仅在 `spiral_parameter_mode="Angle gradient (deg/A)"` 时生效。
- 对输出规模/物理性的影响: 这是 `period_range` 的等价写法，数值越大表示旋转越快，对应更短的周期。
- 推荐范围 (Recommended range):
  - 保守：`[9.0, 9.0, 1.0]`
  - 平衡：`[9.0, 36.0, 9.0]`
  - 探索：`[4.5, 90.0, 4.5]`

### `phase_range` (Phase Range)
- UI Label: `Phase range`
- 字段映射 (Field mapping): 序列化键 `phase_range` <-> 界面标签 `Phase range`。
- 控件标签 (Caption): `Phase range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 15.0]`
- 含义 (Meaning): 全局相位偏移 $\phi_0$ 的扫描区间，单位为度。
- 对输出规模/物理性的影响: 不改变周期或转角梯度，只平移整条自旋纹理的起始角。
- 推荐范围 (Recommended range):
  - 保守：`[0.0, 0.0, 15.0]`
  - 平衡：`[0.0, 90.0, 30.0]`
  - 探索：`[-180.0, 180.0, 30.0]`

### `mz` (Constant M Parallel)
- UI Label: `Constant m_parallel`
- 字段映射 (Field mapping): 序列化键 `mz` <-> 界面标签 `Constant m_parallel`。
- 控件标签 (Caption): `Constant m_parallel`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): 沿传播轴的常数分量；`0` 对应纯 helix，非零时为 conical spiral。
- 对输出规模/物理性的影响: `|mz|` 越大，磁矩越靠近传播轴方向，平面内旋转分量越小。
- 推荐范围 (Recommended range):
  - 保守：`0.0`
  - 平衡：`0.0` 到 `0.5`
  - 探索：`-0.9` 到 `0.9`

### `chirality` (Chirality)
- UI Label: `Chirality`
- 字段映射 (Field mapping): 序列化键 `chirality` <-> 界面标签 `Chirality`。
- 控件标签 (Caption): `Chirality`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Both"`
- 含义 (Meaning): 选择顺时针、逆时针或同时输出成对手性构型。
- 对输出规模/物理性的影响: 选 `Both` 会对同一组参数额外生成一对手性相反的结果。
- 配置建议 (Practical note): `Both` 适合直接构造成对数据；只想固定某个 DMI 手性分支时再改成单一方向。

### `magnitude_source` (Magnitude Source)
- UI Label: `Magnitude source`
- 字段映射 (Field mapping): 序列化键 `magnitude_source` <-> 界面标签 `Magnitude source`。
- 控件标签 (Caption): `Magnitude source`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Existing initial magmoms"`
- 含义 (Meaning): 选择磁矩大小来自现有 `initial_magmoms`，还是来自下方的元素映射/默认值。
- 对输出规模/物理性的影响: 不改变输出数量，但会决定写入的磁矩模长是否继承上游结构。
- 配置建议 (Practical note): 上游已存在可信局域磁矩时优先用 `Existing initial magmoms`；否则用 `Map/default magnitude` 明确指定模长。

### `magmom_map` (Magmom Map)
- UI Label: `Magmom map`
- 字段映射 (Field mapping): 序列化键 `magmom_map` <-> 界面标签 `Magmom map`。
- 控件标签 (Caption): `Magmom map`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `magnitude_source=Map/default magnitude` 时，用于指定元素到磁矩模长的映射。
- 对输出规模/物理性的影响: 不改变输出数量，但直接决定每个元素写入的局域磁矩大小。
- 配置建议 (Practical note): 使用如 `Fe:2.2,Co:1.7,Ni:0.6` 的简洁格式；未知元素会退回 `default_moment`。

### `default_moment` (Default Moment)
- UI Label: `Default |m|`
- 字段映射 (Field mapping): 序列化键 `default_moment` <-> 界面标签 `Default |m|`。
- 控件标签 (Caption): `Default |m|`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): `magmom_map` 未命中元素时使用的默认磁矩模长。
- 对输出规模/物理性的影响: 可以作为保守回退值，防止遗漏元素直接变成无磁态或乱填模长。
- 推荐范围 (Recommended range):
  - 保守：`0.0`
  - 平衡：`0.5` 到 `2.5`
  - 探索：`0.0` 到 `5.0`

### `apply_elements` (Apply Elements)
- UI Label: `Apply elements`
- 字段映射 (Field mapping): 序列化键 `apply_elements` <-> 界面标签 `Apply elements`。
- 控件标签 (Caption): `Apply elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 可选元素白名单；留空时对所有原子写入 spiral 磁矩。
- 对输出规模/物理性的影响: 不改变样本数，但可把自旋纹理限制到一部分元素上，其余元素磁矩会置零。
- 配置建议 (Practical note): 多子晶格体系中，如果只想在磁性子晶格上施加 spiral，可填写如 `Fe,Co`。

### `max_outputs` (Max Outputs)
- UI Label: `Max outputs`
- 字段映射 (Field mapping): 序列化键 `max_outputs` <-> 界面标签 `Max outputs`。
- 控件标签 (Caption): `Max outputs`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int
- 默认值 (Default): `[100]`
- 含义 (Meaning): 当参数组合过多时，用于截断输出数量。
- 对输出规模/物理性的影响: 直接限制生成样本总数，避免一次扫描产生过大的数据集。
- 推荐范围 (Recommended range):
  - 保守：`16`
  - 平衡：`50` 到 `200`
  - 探索：`500` 以上，仅在批量筛选流程中使用

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "SpinSpiralCard",
  "check_state": true,
  "axis": [0.0, 0.0, 1.0],
  "spiral_parameter_mode": "Period (L_D)",
  "period_range": [20.0, 20.0, 10.0],
  "angle_gradient_range": [18.0, 18.0, 1.0],
  "phase_range": [0.0, 0.0, 15.0],
  "mz": [0.0],
  "chirality": "Both",
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [16]
}
```

### 平衡（Balanced）
```json
{
  "class": "SpinSpiralCard",
  "check_state": true,
  "axis": [0.0, 0.0, 1.0],
  "spiral_parameter_mode": "Period (L_D)",
  "period_range": [10.0, 40.0, 10.0],
  "angle_gradient_range": [18.0, 18.0, 1.0],
  "phase_range": [0.0, 90.0, 30.0],
  "mz": [0.0],
  "chirality": "Both",
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [100]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "SpinSpiralCard",
  "check_state": true,
  "axis": [0.0, 0.0, 1.0],
  "spiral_parameter_mode": "Angle gradient (deg/A)",
  "period_range": [20.0, 40.0, 10.0],
  "angle_gradient_range": [4.5, 90.0, 4.5],
  "phase_range": [-180.0, 180.0, 30.0],
  "mz": [0.5],
  "chirality": "Both",
  "magnitude_source": "Map/default magnitude",
  "magmom_map": "Fe:2.2",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [500]
}
```

## 推荐组合
- `Magnetic Order -> Spin Spiral`: 先用 `Magnetic Order` 生成合理的局域磁矩模长，再用本卡片扫描 spiral 周期或转角梯度。
- `Group Label -> Magnetic Order -> Spin Spiral`: 先对不同子晶格做标记和磁矩初始化，再只对目标磁性元素施加 spiral。

## 常见问题与排查
- 需要同时设 `period_range` 和 `angle_gradient_range` 吗：不需要，`spiral_parameter_mode` 只会启用其中一个，另一个只是保留在配置里。
- 输出全是零磁矩：检查 `magnitude_source` 是否选错，或 `magmom_map/default_moment` 是否没有给到有效模长。
- 顺时针和逆时针看起来一样：小胞和特定周期下可能只采到对称点，先改更长的周期或加 `phase_range` 扫描。
- 想要 helix 但出现轴向分量：确认 `mz` 设置为 `0.0`。

## 输出标签 / 元数据变更
- 该卡片会通过 `set_initial_magnetic_moments(...)` 写入三列向量型 `initial_magmoms`。
- `Config_type` 会追加：
  - `Helix(L=...,ph=...,mz=...,chi=...,ax=...)` 当 `mz=0`
  - `Spiral(L=...,ph=...,mz=...,chi=...,ax=...)` 当 `mz!=0`

## 可复现性说明
- 本卡片没有随机采样；相同输入结构与相同参数会得到完全相同的输出。
- 代码默认以传播轴最小投影位置作为相位原点，因此结构整体平移不会改变生成的 spiral 纹理。
- 若上游卡片已经随机化了结构或磁矩模长，本卡片会忠实继承那些上游差异。
