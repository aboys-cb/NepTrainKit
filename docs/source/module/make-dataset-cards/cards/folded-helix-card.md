<!-- card-schema: {"card_name": "Folded Helix", "source_file": "src/NepTrainKit/ui/views/_card/folded_helix_card.py", "serialized_keys": ["params"]} -->

# 折返螺旋初始磁矩（Folded Helix）

`Group`: `Magnetism`  
`Class`: `FoldedHelixCard`  
`Source`: `src/NepTrainKit/ui/views/_card/folded_helix_card.py`

## 功能说明
这张卡片用于生成按层离散定义的对称 folded helix 初始磁矩纹理。它先沿 `layer_axis` 对原子位置做投影并按 `layer_tolerance` 分层，再把磁矩限制在 `plane_normal` 垂直的平面内，使其在前半周期逐层按固定角度旋转、到转折层后按相同步长反向旋转，并按 `2 * half_period_layers` 周期重复。默认情况下 `half_period_mode=Auto from layer count`，会在当前层范围上构造一个首尾闭合的三角相位轮廓；奇数层时中心层为峰值，偶数层时中间两层共享峰值。

它最适合的场景是：为层状磁结构构造折返螺旋式初始磁矩分布。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

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

## 操作示例
### 场景：为层状磁结构构造折返螺旋式初始磁矩分布

**输入：** 已经具备初始磁矩幅值的磁性结构

**目标：** 生成一组分层翻转、方向连续变化的非共线初始态，供后续磁性计算采样

**参数设置：**
- `layer_axis` 定义分层方向
- `half_period_mode` 先决定半周期由程序推断还是手工指定
- `angle_step_range` 和 `phase_range` 先用小步长检查纹理是否合理

**输出：** 带折返螺旋磁矩纹理的结构，磁矩方向按层周期性变化

**怎么验证结果合理：**
- 检查磁矩模长是否保持稳定
- 抽查相邻层相位变化是否符合设定
- 若结构本身没有初始磁矩幅值，先用 `Set Magnetic Moments` 或 `Magnetic Order` 补上

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 现有数据只有标准 FM/AFM 或线性 spin spiral，缺少“先顺时针、到中间再逆时针”的分层磁矩纹理，或者你不想每次手工计算半周期层数。
- 目标任务 (Target objective): 构造磁矩固定在某个面内、沿层方向折返旋转的非共线初态，用于 layered spin texture、折返 helix 或手性翻转型初态数据扩充。
- 建议添加条件 (Add-it trigger): 你关心的是按层离散的磁矩纹理，并且希望同层原子共享相位，而不是让相位随连续坐标线性漂移。
- 不建议添加条件 (Avoid trigger): 你需要的是标准连续 `q·r` spin spiral、圆锥 spiral，或者需要通过真实长周期超胞显式表示的磁基态，此时更适合使用 `Spin Spiral`。
> 物理提示 (Physics caution): 这组卡片主要写入初始磁矩，不自动保证磁序就是最低能态；后验能量和磁矩收敛仍需另行判断。

## 输入前提
- 输入结构最好已经带有可用的 `initial_magmoms`；如果没有，可切换到 `Map/default magnitude` 提供元素磁矩模长。
- 输入结构需要沿 `layer_axis` 存在可区分的层状投影；如果层内投影有小幅噪声，请把 `layer_tolerance` 设得略大于该噪声。
- 如果你只想对部分磁性元素施加 folded helix，可以通过 `apply_elements` 限定元素集合。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由分层轴、旋转平面法向、层容差、半周期模式、角步长、相位、手性序列、磁矩来源和输出上限组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"layer_axis": [0.0, 0.0, 1.0], "plane_normal": [0.0, 0.0, 1.0], "layer_tolerance": 0.05, "half_period_mode": "Auto from layer count", "half_period_layers": [2, 4, 1], "angle_step_range": [15.0, 45.0, 15.0], "phase_range": [0.0, 0.0, 15.0], "sequence_mode": "Clockwise then counterclockwise", "magnitude_source": "Existing initial magmoms", "magmom_map": "", "default_moment": 0.0, "apply_elements": "", "max_outputs": 100}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组 folded helix 参数。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

### `layer_axis` (Layer axis)
- UI Label: `Layer axis`
- 字段映射 (Field mapping): 序列化键 `layer_axis` <-> 界面标签 `Layer axis`。
- 控件标签 (Caption): `Layer axis`。
- 控件解释 (Widget): 三分量向量输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 用于投影坐标并给原子分层的方向。
- 对输出规模/物理性的影响: 改变“沿哪个方向看作相邻层”，直接影响层编号与相位分布。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
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
- 物理直觉 / 典型值: 它通常是控制变化幅度的主旋钮；先从能看清趋势的小幅度起步，再决定是否扩到探索档。
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
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 参数联动 / 生效条件: 它决定当前工作流走哪条主分支；先选模式，再填写与该模式对应的字段。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
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
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
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
- 磁矩写入类卡片应放在真正依赖 `initial_magmoms` 的旋转、螺旋或 canting 卡片之前。

## 常见问题与排查
- 输出没有明显变化时，先检查输入是否已有初始磁矩，或当前是否真的开启了 FM/AFM/PM/旋转分支。
- 如果结果不合理，先看磁矩模长是否被意外改坏，再检查方向参数、group 标签或 k-vector 是否匹配结构。
- 这些卡片主要写入初始磁矩，不保证一定对应真实磁基态；是否物理合理仍需要结合后续计算结果判断。

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
