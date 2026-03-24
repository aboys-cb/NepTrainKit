<!-- card-schema: {"card_name": "Small-Angle Spin Tilt", "source_file": "src/NepTrainKit/ui/views/_card/small_angle_spin_tilt_card.py", "serialized_keys": ["canting_mode", "target_mode", "target_indices", "pair_left_indices", "pair_right_indices", "pair_source", "pair_shell", "pair_shell_tolerance", "pair_element_filter", "pair_group_filter", "bond_filter_mode", "bond_filter_axis", "bond_filter_tolerance", "group_a", "group_b", "angle_list", "tilt_signs", "include_reference", "magnitude_source", "magmom_map", "default_moment", "lift_scalar", "axis", "reference_direction", "apply_elements", "max_outputs"]} -->

# 小角度单自旋/成对 Canting（Small-Angle Spin Tilt）

`Group`: `Magnetism`  
`Class`: `SmallAngleSpinTiltCard`  
`Source`: `src/NepTrainKit/ui/views/_card/small_angle_spin_tilt_card.py`

## 功能说明
这张卡用于生成近参考磁态的小角度非共线扰动样本。它支持三种几何模式：单自旋偏转、显式原子对 canting、两组原子的 group-pair canting。对 DMI 训练集来说，后两种模式更直接，因为它们能明确构造 `S_i × S_j` 的正负手性对。

它最适合的场景是：对单个原子、自旋对或 group 对做小角度 canting，补充局部磁无序样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
设基准方向为 $\hat{\mathbf{m}}_0$，参考侧向方向为 $\hat{\mathbf{t}}$，则单自旋偏转使用

$$
\hat{\mathbf{m}}(\theta)=\cos\theta\,\hat{\mathbf{m}}_0+\sin\theta\,\hat{\mathbf{t}}
$$

对原子对或两组原子时，代码对左/右两侧分别施加

$$
\theta_L=+\theta/2,\qquad \theta_R=-\theta/2
$$

并在 `tilt_signs` 取反时整体翻转正负手性，因此更适合显式构造 `S_i \times S_j` 的正/负样本。

## 操作示例
### 场景：对单个原子、自旋对或 group 对做小角度 canting，补充局部磁无序样本

**输入：** 一个已具备初始磁矩的磁性结构，最好还带有可选的 group 标签

**目标：** 围绕参考方向生成细粒度小角度偏转，而不是整体旋转所有磁矩

**参数设置：**
- `canting_mode` 先决定改单原子、原子对还是 group 对
- `angle_list` 建议先从 1-10 度的小角度起步
- `include_reference=true` 适合保留一个未偏转的基准帧

**输出：** 输出会带有局部 canting 标签，适合补充细粒度磁涨落样本

**怎么验证结果合理：**
- 检查只有目标原子或目标 group 被旋转
- 确认旋转后磁矩模长没有被意外改坏
- 若自动找邻居失败，先检查 pair shell 与容差

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 已有磁性数据主要覆盖严格共线的 FM/AFM 初态，模型对小角度非共线扰动与手性项不敏感。
- 目标任务 (Target objective): 补充近参考态的小角度单自旋或成对 canting 样本，用于约束交换、DMI 或相关非共线响应。
- 建议添加条件 (Add-it trigger): 你已经有合理的参考磁态，或者能通过 `magmom_map/default_moment` 快速生成一套参考 FM 磁矩，并且需要系统扫描 1-10° 的正/负手性小角度样本。
- 不建议添加条件 (Avoid trigger): 体系没有可用磁矩自由度，或研究目标明确要求真实长周期 spin spiral / 多自旋复杂纹理，此时应改用 `Spin Spiral` 等更直接的构造方式。
> 物理提示 (Physics caution): 这组卡片主要写入初始磁矩，不自动保证磁序就是最低能态；后验能量和磁矩收敛仍需另行判断。

## 输入前提
- 输入结构最好已经带有合理的 `initial_magmoms`；如果没有，可切换到 `Map/default magnitude` 用元素磁矩映射生成参考 FM 态。
- 若输入是标量磁矩而不是三列向量，通常应保持 `lift_scalar=true`，让代码先沿 `axis` 抬升为向量再做 canting。
- 若使用 `Group pair canting`，输入结构需要带有 `arrays['group']`；通常可先用 `Group Label` 生成。

## 参数说明（完整）
### `canting_mode` (Canting Mode)
- UI Label: `Canting mode`
- 字段映射 (Field mapping): 序列化键 `canting_mode` <-> 界面标签 `Canting mode`。
- 控件标签 (Caption): `Canting mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Single-spin tilt"`
- 含义 (Meaning): 选择单自旋偏转、显式原子对 canting，或两组原子的 group-pair canting。
- 对输出规模/物理性的影响: 该模式决定样本的几何含义和 DMI 相关性。单自旋模式更局部，原子对模式最直接对应一对原子的手性，group-pair 模式适合两子晶格整体 canting。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 初步验证流程时先用 `Single-spin tilt`；做 bond-resolved DMI 时优先切到 `Atom pair canting`；做子晶格级别的整体 canting 时再用 `Group pair canting`。

### `target_mode` (Target Mode)
- UI Label: `Target atoms`
- 字段映射 (Field mapping): 序列化键 `target_mode` <-> 界面标签 `Target atoms`。
- 控件标签 (Caption): `Target atoms`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"First eligible atom"`
- 含义 (Meaning): 在 `canting_mode="Single-spin tilt"` 时，控制单自旋偏转施加到哪个目标原子集合。
- 对输出规模/物理性的影响: `First eligible atom` 最保守，`All eligible atoms` 会按站点展开输出，`Explicit indices (1-based)` 适合精确指定站点。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - `First eligible atom` 适合先验证流程和标签是否正确。
  - `All eligible atoms` 适合系统扫描所有磁性位点，但应同步检查 `max_outputs`。
  - `Explicit indices (1-based)` 适合只研究特定位点或缺陷邻域。

### `target_indices` (Target Indices)
- UI Label: `Atom indices`
- 字段映射 (Field mapping): 序列化键 `target_indices` <-> 界面标签 `Atom indices`。
- 控件标签 (Caption): `Atom indices`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `canting_mode="Single-spin tilt"` 且 `target_mode="Explicit indices (1-based)"` 时，用逗号或区间语法指定目标原子，例如 `1,3-5`。
- 对输出规模/物理性的影响: 直接限定哪些站点被单独偏转，常用于锁定特定磁性位点、表面位或缺陷邻近位点。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 只有在单自旋显式索引模式下才会生效；建议使用 1-based 索引并先用单帧结构检查结果。

### `pair_left_indices` (Pair Left Indices)
- UI Label: `Pair left indices`
- 字段映射 (Field mapping): 序列化键 `pair_left_indices` <-> 界面标签 `Pair left indices`。
- 控件标签 (Caption): `Pair left indices`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `canting_mode="Atom pair canting"` 时，给出每个原子对左侧的 1-based 索引列表。
- 对输出规模/物理性的影响: 与 `pair_right_indices` 按顺序一一配对，形成明确的原子对 canting 样本。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 建议左右两侧列表长度一致，例如 `1,3` 对应 `2,4`；若长度不同，代码会自动截断到较短一侧。

### `pair_right_indices` (Pair Right Indices)
- UI Label: `Pair right indices`
- 字段映射 (Field mapping): 序列化键 `pair_right_indices` <-> 界面标签 `Pair right indices`。
- 控件标签 (Caption): `Pair right indices`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `canting_mode="Atom pair canting"` 时，给出每个原子对右侧的 1-based 索引列表。
- 对输出规模/物理性的影响: 决定每个 pair canting 样本中的右侧原子集合，从而直接影响 `S_i × S_j` 的对象。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 适合直接构造你关心的 bond 或邻近原子对；若想看更大尺度的两子晶格 canting，应改用 `Group pair canting`。

### `pair_source` (Pair Source)
- UI Label: `Pair source`
- 字段映射 (Field mapping): 序列化键 `pair_source` <-> 界面标签 `Pair source`。
- 控件标签 (Caption): `Pair source`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Manual indices"`
- 含义 (Meaning): 在 `canting_mode="Atom pair canting"` 时，选择显式索引输入，或自动按邻居壳层查找唯一原子对。
- 对输出规模/物理性的影响: `Manual indices` 适合精确指定目标 pair；`Auto by neighbor shell` 适合批量构造近邻 DMI 训练集。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 研究特定 bond 时用 `Manual indices`；想批量扫第一近邻或第二近邻时用 `Auto by neighbor shell`。

### `pair_shell` (Pair Shell)
- UI Label: `Neighbor shell`
- 字段映射 (Field mapping): 序列化键 `pair_shell` <-> 界面标签 `Neighbor shell`。
- 控件标签 (Caption): `Neighbor shell`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[1]`
- 含义 (Meaning): 当 `pair_source="Auto by neighbor shell"` 时，指定要选取第几近邻壳层。
- 对输出规模/物理性的影响: 壳层编号越大，通常 pair 距离越长、候选 pair 越多，也更可能超出最主要的 DMI 相互作用范围。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：`1`
  - 均衡：`1` 到 `2`
  - 探索：`3` 及以上，仅在你明确关注更远相互作用时使用

### `pair_shell_tolerance` (Pair Shell Tolerance)
- UI Label: `Shell tolerance`
- 字段映射 (Field mapping): 序列化键 `pair_shell_tolerance` <-> 界面标签 `Shell tolerance`。
- 控件标签 (Caption): `Shell tolerance`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[0.05]`
- 含义 (Meaning): 自动分壳层时，用于把距离相近的原子对归并到同一近邻壳层的距离容差，单位为 Å。
- 对输出规模/物理性的影响: 容差越大，越容易把不同但接近的距离并到同一壳层；容差过小则可能把本应同一壳层的 pair 拆开。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 推荐范围 (Recommended range):
  - 保守：`0.02`
  - 均衡：`0.05`
  - 探索：`0.1` 到 `0.2`

### `pair_element_filter` (Pair Element Filter)
- UI Label: `Pair elements`
- 字段映射 (Field mapping): 序列化键 `pair_element_filter` <-> 界面标签 `Pair elements`。
- 控件标签 (Caption): `Pair elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `pair_source="Auto by neighbor shell"` 时，可用 `Fe-Fe,Fe-Co` 这类写法只保留指定元素组合。
- 对输出规模/物理性的影响: 可显著减少不相关的 pair，让自动找对更贴近你真正想拟合的交换/DMI 通道。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 多元素体系里建议优先显式填入你关心的元素对；留空表示不过滤元素组合。

### `pair_group_filter` (Pair Group Filter)
- UI Label: `Pair groups`
- 字段映射 (Field mapping): 序列化键 `pair_group_filter` <-> 界面标签 `Pair groups`。
- 控件标签 (Caption): `Pair groups`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当结构带有 `arrays['group']` 时，可用 `A-B,A-A` 这类写法只保留指定分组组合。
- 对输出规模/物理性的影响: 适合把数据限定到某个子晶格耦合通道，避免不同 group 的 pair 混在一起。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 如果你已经用 `Group Label` 做过子晶格标记，这个过滤器通常比手动填索引更稳。

### `bond_filter_mode` (Bond Filter Mode)
- UI Label: `Bond filter`
- 字段映射 (Field mapping): 序列化键 `bond_filter_mode` <-> 界面标签 `Bond filter`。
- 控件标签 (Caption): `Bond filter`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Any"`
- 含义 (Meaning): 在自动找对模式下，决定是否进一步按键方向筛选 pair，可选 `Any`、`Near axis`、`In plane (normal)`。
- 对输出规模/物理性的影响: 适合二维材料、表面或界面体系，把层内键、层间键或特定晶向键拆开处理。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 不需要方向筛选时保持 `Any`；只有当你明确关心某个键向通道时再启用其它模式。

### `bond_filter_axis` (Bond Filter Axis)
- UI Label: `Bond reference`
- 字段映射 (Field mapping): 序列化键 `bond_filter_axis` <-> 界面标签 `Bond reference`。
- 控件标签 (Caption): `Bond reference`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（3 个输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 给出 `Near axis` 的参考轴，或 `In plane (normal)` 的平面法向。
- 对输出规模/物理性的影响: 决定哪些 bond 被认为“接近某轴”或“位于某平面内”。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 推荐范围 (Recommended range):
  - 保守：`[0.0, 0.0, 1.0]`
  - 均衡：`[1.0, 0.0, 0.0]`、`[0.0, 1.0, 0.0]`、`[0.0, 0.0, 1.0]`
  - 探索：`[1.0, 1.0, 0.0]`、`[1.0, 1.0, 1.0]`

### `bond_filter_tolerance` (Bond Filter Tolerance)
- UI Label: `Bond angle tol`
- 字段映射 (Field mapping): 序列化键 `bond_filter_tolerance` <-> 界面标签 `Bond angle tol`。
- 控件标签 (Caption): `Bond angle tol`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[20.0]`
- 含义 (Meaning): 键方向筛选的角度容差，单位为度。
- 对输出规模/物理性的影响: 容差越小，选中的键方向越“纯”；容差越大，保留下来的 pair 越多。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 推荐范围 (Recommended range):
  - 保守：`5.0`
  - 均衡：`10.0` 到 `20.0`
  - 探索：`30.0` 及以上，仅在结构畸变较明显时使用

### `group_a` (Group A)
- UI Label: `Group A`
- 字段映射 (Field mapping): 序列化键 `group_a` <-> 界面标签 `Group A`。
- 控件标签 (Caption): `Group A`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"A"`
- 含义 (Meaning): 当 `canting_mode="Group pair canting"` 时，`arrays['group']` 中属于 Group A 的原子整体旋转 `+theta/2`。
- 对输出规模/物理性的影响: 决定 group-pair canting 的左侧子群。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 通常与 `Group Label` 卡联用；若输入结构没有 `arrays['group']`，本模式不会生效。

### `group_b` (Group B)
- UI Label: `Group B`
- 字段映射 (Field mapping): 序列化键 `group_b` <-> 界面标签 `Group B`。
- 控件标签 (Caption): `Group B`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"B"`
- 含义 (Meaning): 当 `canting_mode="Group pair canting"` 时，`arrays['group']` 中属于 Group B 的原子整体旋转 `-theta/2`。
- 对输出规模/物理性的影响: 与 `group_a` 一起定义两组原子的对称 canting。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 若你的分组不是默认的 `A/B`，请在这里改成对应标签。

### `angle_list` (Angle List)
- UI Label: `Tilt angles`
- 字段映射 (Field mapping): 序列化键 `angle_list` <-> 界面标签 `Tilt angles`。
- 控件标签 (Caption): `Tilt angles`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"1,2,5,10"`
- 含义 (Meaning): 逗号分隔的偏转角列表，单位为度。
- 对输出规模/物理性的影响: 每增加一个角度，就会为每个目标原子、原子对或组对多生成一批样本；角度越大，偏离参考态越明显。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 推荐从 `1,2,5,10` 这样的低角度列表开始；如果只想拟合更严格的 $q\to 0$ 区域，可收缩到 `1,2,3,5`。

### `tilt_signs` (Tilt Signs)
- UI Label: `Tilt signs`
- 字段映射 (Field mapping): 序列化键 `tilt_signs` <-> 界面标签 `Tilt signs`。
- 控件标签 (Caption): `Tilt signs`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Positive only"`
- 含义 (Meaning): 选择只输出 `+theta`、只输出 `-theta`，还是为同一角度成对输出 `+/-theta` 变体。
- 对输出规模/物理性的影响: `Both (+/- pair)` 会把每个目标角度的样本数翻倍，但更适合直接构造成对的 DMI/手性训练样本。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 常规小角度扰动可先用 `Positive only`；若目标是 DMI 训练集，优先用 `Both (+/- pair)`；`Negative only` 更适合补单侧对照。

### `include_reference` (Include Reference)
- UI Label: `Include reference state`
- 字段映射 (Field mapping): 序列化键 `include_reference` <-> 界面标签 `Include reference state`。
- 控件标签 (Caption): `Include reference state`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 是否在 canting 样本之前额外输出一帧未偏转的参考磁态。
- 对输出规模/物理性的影响: 开启后更利于直接比较参考态与 canting 态之间的能量差，但会额外增加 1 个输出。
- 怎么判断该开还是该关: 只有当你明确知道这个开关会改变当前工作流目标时才开启；不确定时先保持默认并用小样本验证。
- 配置建议 (Practical note):
  - 开启：需要显式保留参考磁态，方便后续做能量差分与标签追踪时使用。
  - 关闭：上游已经单独保存了参考态，或者只想导出纯 canting 样本时关闭。

### `magnitude_source` (Magnitude Source)
- UI Label: `Magnitude source`
- 字段映射 (Field mapping): 序列化键 `magnitude_source` <-> 界面标签 `Magnitude source`。
- 控件标签 (Caption): `Magnitude source`。
- 控件解释 (Widget): 下拉选择 `ComboBox`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Existing initial magmoms"`
- 含义 (Meaning): 决定参考磁矩来自输入结构已有的 `initial_magmoms`，还是来自 `magmom_map/default_moment` 生成的 FM 参考态。
- 对输出规模/物理性的影响: 不改变输出条数，但会改变参考磁矩的来源与可靠性。
- 参数联动 / 生效条件: 它决定当前工作流走哪条主分支；先选模式，再填写与该模式对应的字段。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 上游已经生成可信磁矩时优先用 `Existing initial magmoms`；如果输入结构还没有磁矩，就切到 `Map/default magnitude` 明确指定元素磁矩。

### `magmom_map` (Magmom Map)
- UI Label: `Magmom map`
- 字段映射 (Field mapping): 序列化键 `magmom_map` <-> 界面标签 `Magmom map`。
- 控件标签 (Caption): `Magmom map`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 当 `magnitude_source="Map/default magnitude"` 时，用于指定元素到磁矩幅值的映射，如 `Fe:2.2,Ni:0.6`。
- 对输出规模/物理性的影响: 不改变样本数，但直接决定生成的参考磁矩幅值。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): 只在 `Map/default magnitude` 模式下生效；建议显式列出主要磁性元素，避免关键元素退回到 `default_moment`。

### `default_moment` (Default Moment)
- UI Label: `Default |m|`
- 字段映射 (Field mapping): 序列化键 `default_moment` <-> 界面标签 `Default |m|`。
- 控件标签 (Caption): `Default |m|`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): `magmom_map` 未覆盖到的元素默认使用的磁矩幅值。
- 对输出规模/物理性的影响: 不改变输出条数，但会影响未命中元素是否带磁以及其参考态幅值。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：`0.0`
  - 均衡：`0.5` 到 `2.5`
  - 探索：`0.0` 到 `5.0`

### `lift_scalar` (Lift Scalar)
- UI Label: `Lift scalar magmoms to vectors`
- 字段映射 (Field mapping): 序列化键 `lift_scalar` <-> 界面标签 `Lift scalar magmoms to vectors`。
- 控件标签 (Caption): `Lift scalar magmoms to vectors`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 当输入磁矩是标量时，是否先沿 `axis` 抬升为向量再施加 canting。
- 对输出规模/物理性的影响: 关闭后，标量磁矩输入将无法直接做非共线 canting；开启则能把共线输入稳定转换成向量形式。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：输入是标量磁矩、而你又需要生成非共线 canting 样本时应保持开启。
  - 关闭：只有在你明确不希望自动抬升标量磁矩时才关闭。

### `axis` (Axis)
- UI Label: `Base axis`
- 字段映射 (Field mapping): 序列化键 `axis` <-> 界面标签 `Base axis`。
- 控件标签 (Caption): `Base axis`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（3 个输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0.0, 0.0, 1.0]`
- 含义 (Meaning): 标量磁矩抬升和 `Map/default magnitude` 参考 FM 态所采用的基准方向。
- 对输出规模/物理性的影响: 会改变参考磁态的方向定义，并影响 canting 围绕哪个基准方向展开。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 推荐范围 (Recommended range):
  - 保守：`[0.0, 0.0, 1.0]`
  - 均衡：`[1.0, 0.0, 0.0]`、`[0.0, 1.0, 0.0]`、`[0.0, 0.0, 1.0]`
  - 探索：`[1.0, 1.0, 0.0]` 或其他归一化前的方向候选

### `reference_direction` (Reference Direction)
- UI Label: `Tilt reference`
- 字段映射 (Field mapping): 序列化键 `reference_direction` <-> 界面标签 `Tilt reference`。
- 控件标签 (Caption): `Tilt reference`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（3 个输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[1.0, 0.0, 0.0]`
- 含义 (Meaning): 定义 canting 平面的首选参考方向；代码会先把它对基准磁矩方向做正交化。
- 对输出规模/物理性的影响: 不改变输出条数，但会改变“朝哪个侧向”进行 canting，因此影响非共线向量的具体分布。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：`[1.0, 0.0, 0.0]`
  - 均衡：`[1.0, 0.0, 0.0]` 或 `[0.0, 1.0, 0.0]`
  - 探索：按研究需求扫描不同晶向，例如 `[1.0, 1.0, 0.0]`

### `apply_elements` (Apply Elements)
- UI Label: `Apply elements`
- 字段映射 (Field mapping): 序列化键 `apply_elements` <-> 界面标签 `Apply elements`。
- 控件标签 (Caption): `Apply elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 可选元素白名单；非空时，只让这些元素参与目标位点筛选，在 map/default 模式下也只给这些元素赋磁矩。
- 对输出规模/物理性的影响: 能显著减少目标位点数量，避免把 canting 施加到不关心的元素上。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): 多子晶格体系中，若只想针对磁性子晶格或特定元素做 canting，可填如 `Fe,Co`；留空则默认所有可用原子都可能成为目标。

### `max_outputs` (Max Outputs)
- UI Label: `Max outputs`
- 字段映射 (Field mapping): 序列化键 `max_outputs` <-> 界面标签 `Max outputs`。
- 控件标签 (Caption): `Max outputs`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[1]
- 默认值 (Default): `[100]`
- 含义 (Meaning): 对最终导出的样本数设置上限，防止“目标数 × 角度数 × 手性数”组合膨胀。
- 对输出规模/物理性的影响: 直接限制生成样本总数；若开启 `include_reference`，参考态也计入这个上限。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：`16`
  - 均衡：`50` 到 `200`
  - 探索：`500` 以上，仅建议在批量筛选流程中使用

## 推荐预设（可直接复制 JSON）
### Safe
```json
{
  "class": "SmallAngleSpinTiltCard",
  "check_state": true,
  "canting_mode": "Single-spin tilt",
  "target_mode": "First eligible atom",
  "target_indices": "",
  "pair_left_indices": "",
  "pair_right_indices": "",
  "pair_source": "Manual indices",
  "pair_shell": [1],
  "pair_shell_tolerance": [0.05],
  "pair_element_filter": "",
  "pair_group_filter": "",
  "bond_filter_mode": "Any",
  "bond_filter_axis": [0.0, 0.0, 1.0],
  "bond_filter_tolerance": [20.0],
  "group_a": "A",
  "group_b": "B",
  "angle_list": "1,2,5,10",
  "tilt_signs": "Positive only",
  "include_reference": true,
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "lift_scalar": true,
  "axis": [0.0, 0.0, 1.0],
  "reference_direction": [1.0, 0.0, 0.0],
  "apply_elements": "",
  "max_outputs": [16]
}
```

### Balanced
```json
{
  "class": "SmallAngleSpinTiltCard",
  "check_state": true,
  "canting_mode": "Atom pair canting",
  "target_mode": "First eligible atom",
  "target_indices": "",
  "pair_left_indices": "",
  "pair_right_indices": "",
  "pair_source": "Auto by neighbor shell",
  "pair_shell": [1],
  "pair_shell_tolerance": [0.05],
  "pair_element_filter": "",
  "pair_group_filter": "",
  "bond_filter_mode": "Any",
  "bond_filter_axis": [0.0, 0.0, 1.0],
  "bond_filter_tolerance": [20.0],
  "group_a": "A",
  "group_b": "B",
  "angle_list": "1,2,5,10",
  "tilt_signs": "Both (+/- pair)",
  "include_reference": true,
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "lift_scalar": true,
  "axis": [0.0, 0.0, 1.0],
  "reference_direction": [1.0, 0.0, 0.0],
  "apply_elements": "",
  "max_outputs": [100]
}
```

### Aggressive/Exploration
```json
{
  "class": "SmallAngleSpinTiltCard",
  "check_state": true,
  "canting_mode": "Group pair canting",
  "target_mode": "First eligible atom",
  "target_indices": "",
  "pair_left_indices": "",
  "pair_right_indices": "",
  "pair_source": "Auto by neighbor shell",
  "pair_shell": [2],
  "pair_shell_tolerance": [0.1],
  "pair_element_filter": "Fe-Co",
  "pair_group_filter": "A-B",
  "bond_filter_mode": "In plane (normal)",
  "bond_filter_axis": [0.0, 0.0, 1.0],
  "bond_filter_tolerance": [15.0],
  "group_a": "A",
  "group_b": "B",
  "angle_list": "1,2,5,10,15",
  "tilt_signs": "Both (+/- pair)",
  "include_reference": false,
  "magnitude_source": "Map/default magnitude",
  "magmom_map": "Fe:2.2,Co:1.7",
  "default_moment": [0.5],
  "lift_scalar": true,
  "axis": [0.0, 0.0, 1.0],
  "reference_direction": [1.0, 0.0, 0.0],
  "apply_elements": "Fe,Co",
  "max_outputs": [500]
}
```

## 推荐组合
- `Set Magnetic Moments -> Small-Angle Spin Tilt`: 先统一把磁矩标准化成向量参考态，再批量生成 DMI 更友好的成对 canting 样本。
- `Magnetic Order -> Small-Angle Spin Tilt`: 先生成稳定的参考磁态，再做单自旋、原子对或组对 canting。
- `Group Label -> Magnetic Order -> Small-Angle Spin Tilt`: 当你只想对某个子晶格或两组原子做对称 canting 时，先分组再切到 `Group pair canting`。

## 常见问题与排查
- 输出没有明显变化时，先检查输入是否已有初始磁矩，或当前是否真的开启了 FM/AFM/PM/旋转分支。
- 如果结果不合理，先看磁矩模长是否被意外改坏，再检查方向参数、group 标签或 k-vector 是否匹配结构。
- 这些卡片主要写入初始磁矩，不保证一定对应真实磁基态；是否物理合理仍需要结合后续计算结果判断。

## 输出标签 / 元数据变更
- 该卡片会通过 `set_initial_magnetic_moments(...)` 写入三列向量形式的 `initial_magmoms`。
- `Config_type` 会追加：
  - `SpinTiltRef`：当 `include_reference=true` 时写入未偏转参考态。
  - `SpinTilt(i=...,a=...,sg=...)`：单自旋偏转。
  - `SpinPair(i=...,j=...,a=...,sg=...)`：显式原子对 canting。
  - `SpinPairG(A=...,B=...,a=...,sg=...)`：两组原子的 group-pair canting。

## 可复现性说明
- 本卡没有随机采样；相同输入结构、相同磁矩来源和相同参数设置会得到完全一致的输出。
- `reference_direction` 会先对基准磁矩方向正交化，因此即使输入磁矩方向与 `axis` 不完全一致，输出仍是确定性的。
- 如果上游卡片本身包含随机操作，应在上游统一控制 seed；本卡只负责确定性地构造单自旋或成对 canting 样本。
