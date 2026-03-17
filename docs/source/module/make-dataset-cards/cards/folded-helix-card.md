<!-- card-schema: {"card_name": "Folded Helix", "source_file": "src/NepTrainKit/ui/views/_card/folded_helix_card.py", "serialized_keys": ["layer_axis", "plane_normal", "layer_tolerance", "half_period_mode", "half_period_layers", "angle_step_range", "phase_range", "sequence_mode", "magnitude_source", "magmom_map", "default_moment", "apply_elements", "max_outputs"]} -->

# 折返螺旋初始磁矩（Folded Helix）

`Group`: `Magnetism`  
`Class`: `FoldedHelixCard`  
`Source`: `src/NepTrainKit/ui/views/_card/folded_helix_card.py`

## 功能说明
这张卡片用于生成按层离散定义的对称 folded helix 初始磁矩纹理。它先沿 `layer_axis` 对原子位置做投影并按 `layer_tolerance` 分层，再把磁矩限制在 `plane_normal` 垂直的平面内，使其在前半周期逐层按固定角度旋转、到转折层后按相同步长反向旋转，并按 `2 * half_period_layers` 周期重复。默认情况下 `half_period_mode=Auto from layer count`，会在当前层范围上构造一个首尾闭合的三角相位轮廓；奇数层时中心层为峰值，偶数层时中间两层共享峰值。

最小可运行示例：把 `layer_axis` 和 `plane_normal` 都设为 `[0, 0, 1]`，保持 `half_period_mode="Auto from layer count"`，再令 `angle_step_range=[15,15,15]`、`sequence_mode="Clockwise then counterclockwise"`，即可对当前结构的现有层数自动铺满一个“磁矩在 `xy` 面内、沿 `z` 分层、先顺时针后逆时针”的周期初态。

:::{tip}
高通量示例：先用 `Magnetic Order` 或 `Set Magnetic Moments` 生成稳定的磁矩模长，再固定 `layer_axis=[0,0,1]` 与 `plane_normal=[0,0,1]`，只扫描 `half_period_layers + angle_step_range + sequence_mode`，批量构造不同折返周期和转角的 layered helix 数据，再结合快速单点或筛选流程剔除明显异常的初态。
:::

### 关键公式 (Core equations)
设层号为 `k`，半周期层步数为 `h`，层间转角为 `Δφ`，全局相位为 `φ0`，顺/逆时针序符号为 `σ∈{-1,+1}`，则一周期内的折返层步数写为

$$
s(k)=
\begin{cases}
k, & 0 \le k \le h \\
2h-k, & h < k < 2h
\end{cases}
$$

并按 `k mod (2h)` 周期重复，对应相位为

$$
\phi(k)=\phi_0 + \sigma \cdot s(k)\cdot \Delta \phi
$$

若旋转平面的正交基为 $(\mathbf{e}_1,\mathbf{e}_2)$，则单位磁矩方向为

$$
\mathbf{m}(k)=\cos\phi(k)\,\mathbf{e}_1+\sin\phi(k)\,\mathbf{e}_2
$$

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 现有数据只有标准 FM/AFM 或线性 spin spiral，缺少“先顺时针、到中间再逆时针”的分层磁矩纹理，或者你不想每次手工计算半周期层数。
- 目标任务 (Target objective): 构造磁矩固定在某个面内、沿层方向折返旋转的非共线初态，用于 layered spin texture、折返 helix 或手性翻转型初态数据扩充。
- 建议添加条件 (Add-it trigger): 你关心的是按层离散的磁矩纹理，并且希望同层原子共享相位，而不是让相位随连续坐标线性漂移。
- 不建议添加条件 (Avoid trigger): 你需要的是标准连续 `q·r` spin spiral、圆锥 spiral，或者需要通过真实长周期超胞显式表示的磁基态，此时更适合使用 `Spin Spiral`。
> 物理提示 (Physics caution): 这张卡片只写入初始磁矩，不保证该折返纹理一定与交换参数、各向异性或真实基态严格相容。

## 输入前提
- 输入结构最好已经带有可用的 `initial_magmoms`；如果没有，可切换到 `Map/default magnitude` 提供元素磁矩模长。
- 输入结构需要沿 `layer_axis` 存在可区分的层状投影；如果层内投影有小幅噪声，请把 `layer_tolerance` 设得略大于该噪声。
- 如果你只想对部分磁性元素施加 folded helix，可以通过 `apply_elements` 限定元素集合。

## 参数说明（完整）
### `layer_axis` (Layer axis)
- UI Label: `Layer axis`
- 字段映射 (Field mapping): 序列化键 `layer_axis` <-> 界面标签 `Layer axis`。
- 控件标签 (Caption): `Layer axis`。
- 控件解释 (Widget): 三分量向量输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 用于投影坐标并给原子分层的方向。
- 对输出规模/物理性的影响: 改变“沿哪个方向看作相邻层”，直接影响层编号与相位分布。
- 推荐范围 (Recommended range):
  - 保守：`[0.0, 0.0, 1.0]`
  - 平衡：`[1.0, 0.0, 0.0]`、`[0.0, 1.0, 0.0]`、`[0.0, 0.0, 1.0]`
  - 探索：任意归一化方向，如 `[1.0, 1.0, 0.0]`

### `plane_normal` (Rotation-plane normal)
- UI Label: `Rotation-plane normal`
- 字段映射 (Field mapping): 序列化键 `plane_normal` <-> 界面标签 `Rotation-plane normal`。
- 控件标签 (Caption): `Rotation-plane normal`。
- 控件解释 (Widget): 三分量向量输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 磁矩旋转平面的法向；例如 `[0,0,1]` 表示磁矩在 `xy` 面内。
- 对输出规模/物理性的影响: 决定磁矩被限制在哪个平面内旋转。
- 推荐范围 (Recommended range):
  - 保守：`[0.0, 0.0, 1.0]`
  - 平衡：`[1.0, 0.0, 0.0]`、`[0.0, 1.0, 0.0]`、`[0.0, 0.0, 1.0]`
  - 探索：如 `[1.0, 1.0, 0.0]`、`[1.0, 1.0, 1.0]`

### `layer_tolerance` (Layer tolerance)
- UI Label: `Layer tolerance`
- 字段映射 (Field mapping): 序列化键 `layer_tolerance` <-> 界面标签 `Layer tolerance`。
- 控件标签 (Caption): `Layer tolerance`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[0.05]`
- 含义 (Meaning): 投影坐标差不超过该阈值的原子会被归为同一层。
- 对输出规模/物理性的影响: 阈值过小会把同层拆散，阈值过大又可能把相邻层并到一起。
- 推荐范围 (Recommended range):
  - 保守：`[0.01]`
  - 平衡：`[0.03]` 到 `[0.10]`
  - 探索：按层内起伏大小调整到 `[0.20]` 或更高

### `half_period_mode` (Half-period mode)
- UI Label: `Half-period mode`
- 字段映射 (Field mapping): 序列化键 `half_period_mode` <-> 界面标签 `Half-period mode`。
- 控件标签 (Caption): `Half-period mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Auto from layer count"`
- 含义 (Meaning): 选择自动由层数生成闭合折返相位，或手动指定扫描范围。
- 对输出规模/物理性的影响: `Auto` 会让当前结构默认铺满一个首尾闭合的折返周期；`Manual` 则允许在固定结构上比较不同折返周期。
- 配置建议 (Practical note): 大多数固定层数结构直接用 `Auto`；只有在想比较多个周期长度或构造不满铺周期时再切到 `Manual`。

### `half_period_layers` (Half-period layers)
- UI Label: `Half-period layers`
- 字段映射 (Field mapping): 序列化键 `half_period_layers` <-> 界面标签 `Half-period layers`。
- 控件标签 (Caption): `Half-period layers`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame` (`min/max/step`)。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[2, 4, 1]`
- 含义 (Meaning): 当 `half_period_mode="Manual"` 时，表示从边界层走到转折层需要经历的层间步进数；一个完整周期共有 `2 * half_period_layers` 层步。若处于 `Auto`，该字段只作为回退配置保存，实际相位由当前层数决定。
- 对输出规模/物理性的影响: 在手动模式下，数值越大，折返周期越长；自动模式则总是优先保证当前结构首尾闭合。
- 推荐范围 (Recommended range):
  - 保守：`[2, 2, 1]`
  - 平衡：`[2, 6, 1]`
  - 探索：`[4, 12, 2]`

### `angle_step_range` (Layer angle step)
- UI Label: `Layer angle step`
- 字段映射 (Field mapping): 序列化键 `angle_step_range` <-> 界面标签 `Layer angle step`。
- 控件标签 (Caption): `Layer angle step`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame` (`min/max/step`)。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[15.0, 45.0, 15.0]`
- 含义 (Meaning): 相邻层之间的固定面内转角，单位为度。
- 对输出规模/物理性的影响: 转角越大，相邻层磁矩差异越强。
- 推荐范围 (Recommended range):
  - 保守：`[5.0, 15.0, 5.0]`
  - 平衡：`[15.0, 45.0, 15.0]`
  - 探索：`[30.0, 90.0, 15.0]`

### `phase_range` (Phase range)
- UI Label: `Phase range`
- 字段映射 (Field mapping): 序列化键 `phase_range` <-> 界面标签 `Phase range`。
- 控件标签 (Caption): `Phase range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame` (`min/max/step`)。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 15.0]`
- 含义 (Meaning): 全局相位偏移扫描范围，单位为度。
- 对输出规模/物理性的影响: 不改变折返周期，只平移整个纹理在旋转平面内的起始方向。
- 推荐范围 (Recommended range):
  - 保守：`[0.0, 0.0, 15.0]`
  - 平衡：`[0.0, 90.0, 30.0]`
  - 探索：`[-180.0, 180.0, 30.0]`

### `sequence_mode` (Sequence)
- UI Label: `Sequence`
- 字段映射 (Field mapping): 序列化键 `sequence_mode` <-> 界面标签 `Sequence`。
- 控件标签 (Caption): `Sequence`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Clockwise then counterclockwise"`
- 含义 (Meaning): 选择一周期内先顺时针再逆时针、先逆时针再顺时针，或两者都输出。
- 对输出规模/物理性的影响: 选 `Both` 会对同一组参数生成一对镜像序列。
- 配置建议 (Practical note): 只想构造单一 folded helix 时选单方向；想做成对手性数据时选 `Both`。

### `magnitude_source` (Magnitude source)
- UI Label: `Magnitude source`
- 字段映射 (Field mapping): 序列化键 `magnitude_source` <-> 界面标签 `Magnitude source`。
- 控件标签 (Caption): `Magnitude source`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Existing initial magmoms"`
- 含义 (Meaning): 磁矩模长来自已有 `initial_magmoms`，或来自 `magmom_map/default_moment`。
- 对输出规模/物理性的影响: 不改变样本数，但决定每个原子的磁矩模长如何赋值。
- 配置建议 (Practical note): 上游已有可信局域磁矩时优先用 `Existing initial magmoms`，否则切换到 `Map/default magnitude`。

### `magmom_map` (Magmom map)
- UI Label: `Magmom map`
- 字段映射 (Field mapping): 序列化键 `magmom_map` <-> 界面标签 `Magmom map`。
- 控件标签 (Caption): `Magmom map`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `magnitude_source=Map/default magnitude` 时，用于给元素指定磁矩模长。
- 对输出规模/物理性的影响: 不影响样本数，但直接影响各元素磁矩大小。
- 配置建议 (Practical note): 可使用如 `Fe:2.2,Co:1.7,Ni:0.6` 的格式；未命中的元素回退到 `default_moment`。

### `default_moment` (Default moment)
- UI Label: `Default |m|`
- 字段映射 (Field mapping): 序列化键 `default_moment` <-> 界面标签 `Default |m|`。
- 控件标签 (Caption): `Default |m|`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): `magmom_map` 未命中元素时使用的默认磁矩模长。
- 对输出规模/物理性的影响: 作为兜底值，避免遗漏元素直接变成零磁矩。
- 推荐范围 (Recommended range):
  - 保守：`[0.0]`
  - 平衡：`[0.5]` 到 `[2.5]`
  - 探索：`[0.0]` 到 `[5.0]`

### `apply_elements` (Apply elements)
- UI Label: `Apply elements`
- 字段映射 (Field mapping): 序列化键 `apply_elements` <-> 界面标签 `Apply elements`。
- 控件标签 (Caption): `Apply elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 只对这些元素施加 folded helix；留空表示所有原子。
- 对输出规模/物理性的影响: 不改变样本数，但可把折返纹理限制到目标磁性元素子集。
- 配置建议 (Practical note): 多子晶格体系中常用 `Fe,Co` 之类的白名单来限定施加对象。

### `max_outputs` (Max outputs)
- UI Label: `Max outputs`
- 字段映射 (Field mapping): 序列化键 `max_outputs` <-> 界面标签 `Max outputs`。
- 控件标签 (Caption): `Max outputs`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[100]`
- 含义 (Meaning): 限制参数组合生成的最大结构数量。
- 对输出规模/物理性的影响: 直接截断样本总数，避免一次性生成过多组合。
- 推荐范围 (Recommended range):
  - 保守：`[16]`
  - 平衡：`[50]` 到 `[200]`
  - 探索：`[500]` 以上，仅在批量筛选流程中使用

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "FoldedHelixCard",
  "check_state": true,
  "layer_axis": [0.0, 0.0, 1.0],
  "plane_normal": [0.0, 0.0, 1.0],
  "layer_tolerance": [0.03],
  "half_period_mode": "Auto from layer count",
  "half_period_layers": [2, 2, 1],
  "angle_step_range": [10.0, 10.0, 5.0],
  "phase_range": [0.0, 0.0, 15.0],
  "sequence_mode": "Clockwise then counterclockwise",
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
  "class": "FoldedHelixCard",
  "check_state": true,
  "layer_axis": [0.0, 0.0, 1.0],
  "plane_normal": [0.0, 0.0, 1.0],
  "layer_tolerance": [0.05],
  "half_period_mode": "Auto from layer count",
  "half_period_layers": [2, 6, 1],
  "angle_step_range": [15.0, 45.0, 15.0],
  "phase_range": [0.0, 90.0, 30.0],
  "sequence_mode": "Both",
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [100]
}
```

### 激进 / 探索（Aggressive/Exploration）
```json
{
  "class": "FoldedHelixCard",
  "check_state": true,
  "layer_axis": [0.0, 0.0, 1.0],
  "plane_normal": [0.0, 0.0, 1.0],
  "layer_tolerance": [0.10],
  "half_period_mode": "Manual",
  "half_period_layers": [4, 12, 2],
  "angle_step_range": [30.0, 90.0, 15.0],
  "phase_range": [-180.0, 180.0, 30.0],
  "sequence_mode": "Both",
  "magnitude_source": "Map/default magnitude",
  "magmom_map": "Fe:2.2",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [500]
}
```

## 推荐组合
- `Magnetic Order -> Folded Helix`: 先用 `Magnetic Order` 建立稳定的 FM/AFM 局域磁矩模长和方向，再用本卡片按层折返旋转，适合 layered magnetic texture 数据生成。
- `Set Magnetic Moments -> Folded Helix`: 当你只想手动指定元素模长而不引入额外磁序扫描时，可先用 `Set Magnetic Moments` 写入参考磁矩，再用本卡片生成折返纹理。

## 常见问题与排查
- 同一层原子的方向不一致：通常是 `layer_tolerance` 太小，导致本应同层的原子被拆成不同层。
- 自动模式下周期看起来不符合预期：先确认检测到的层数是否就是你想要的层定义；如果不是，调大 `layer_tolerance` 或改用 `Manual`。
- 输出几乎看不出旋转：先检查 `angle_step_range` 是否过小，或 `half_period_layers` 是否远大于实际层数。
- 只有部分原子被赋值：检查 `apply_elements` 是否限制过严，或 `magnitude_source=Existing initial magmoms` 时上游是否真的存在有效磁矩。
- 想要标准连续 spiral：这不是 `Folded Helix` 的目标场景，应改用 `Spin Spiral`。

## 输出标签 / 元数据变更
- 该卡片会通过 `set_initial_magnetic_moments(...)` 写入三列向量型 `initial_magmoms`。
- `Config_type` 会追加 `FoldedHelix(h=...,da=...,ph=...,seq=...,ax=...,pn=...)`，其中：
  - `h` 是半周期层步数
  - `da` 是层间转角
  - `ph` 是全局相位
  - `seq` 是手性顺序
  - `ax` 是分层轴标签
  - `pn` 是旋转平面法向标签

## 可复现性说明
- 这张卡片没有随机采样；相同输入结构和相同参数会得到完全一致的输出。
- 分层原点固定为沿 `layer_axis` 投影后的最小层号开始，因此同一份结构在不改变相对层序的情况下结果可复现。
- 若上游卡片已经随机化了结构坐标或磁矩模长，本卡片会忠实继承那些上游差异。
