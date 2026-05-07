<!-- card-schema: {"card_name": "Small-Angle Spin Tilt", "source_file": "src/NepTrainKit/ui/views/_card/small_angle_spin_tilt_card.py", "serialized_keys": ["params"]} -->

# 小角度自旋倾斜（Small-Angle Spin Tilt）

`Group`: `Magnetism` | `Class`: `SmallAngleSpinTiltCard`

## 功能说明

对选定目标原子、全局磁序或原子对做确定性小角度磁矩偏转（canting）。支持四种模式：单自旋偏转、Global tilt、显式原子对 canting、两组原子的 group-pair canting。pair 模式显式构造 S_i x S_j 的正/负手性对，直接服务于 DMI 训练集；Global tilt 用于外场下的集体偏转角扫描。

$$\hat{\mathbf{m}}(\theta)=\cos\theta\,\hat{\mathbf{m}}_0+\sin\theta\,\hat{\mathbf{t}}$$

$$\theta_L=+\theta/2,\qquad \theta_R=-\theta/2$$

**关键限制：** 这是一张确定性卡片——没有随机采样。每个角度和每个目标都会生成确定性的输出。需要随机方向扰动时用 `Magmom Rotation`。

## 操作示例

### 场景：模型在 DMI 体系上手性相关的能量差完全预测反了

你在一个非中心对称磁性体系上训练了 NEP 模型，所有训练数据来自 FM 和 AFM 共线构型。模型拟合的交换作用还不错，但计算左手螺旋和右手螺旋的能量差时，符号反了——它根本不知道 DMI 的方向偏好。

**诊断思路：** DMI 的微观来源是 S_i x S_j 项。如果训练集里所有构型的 S_i 和 S_j 都是完美共线（夹角 0 或 180 度），交叉积恒为零，模型无法学到 DMI。需要显式加入相邻自旋有小角度偏差的构型，并且正负手性成对出现，让模型看到不同的 S_i x S_j 值。

**输入：** 一个已有 `initial_magmoms` 的 bcc Fe 结构，或通过 `magnitude_source = Map/default magnitude` 生成 FM 参考态

**目标：** 对所有第一近邻 Fe-Fe 原子对做 ±1°/2°/5°/10° 的成对 canting，正负手性成对输出，覆盖常见的 DMI 强度区间

**参数设置：**
- `Canting Mode` = `Atom pair canting`
- `Pair Source` = `Auto by neighbor shell`
- `Pair Shell` = `[1]`
- `Angle List` = `1,2,5,10`
- `Tilt Signs` = `Both (+/- pair)`
- `Magnitude Source` = `Map/default magnitude`，`Magmom Map` = `Fe:2.2`

**输出：** 对每个第一近邻 Fe-Fe 对，4 个角度 x 2 种手性 = 8 个 canting 构型。如果自动找到 4 个近邻对，共 ~32 个输出（含 reference 则为 33）。

**怎么验证训练集质量改善：**
- 重训后计算左手和右手螺旋构型的能量差，符号和趋势应该接近 DFT 参考
- 抽查一对 canting 输出：左侧原子磁矩偏 +theta/2，右侧偏 -theta/2
- 如果 DMI 仍不准，扩大 `Angle List` 到 `1,2,3,5,7,10,15`，增加第二近邻 `Pair Shell = [2]`
- 如果只想研究特定键，切到 `Manual indices` 精确指定

### 什么时候加这张卡、什么时候不加

**加：**
- 训练 DMI 或手性相关的磁性模型
- 需要确定性、可对比的小角度偏转样本（不是随机方向扰动）
- 需要外场或 metamagnetic 路径附近的整体磁序偏转
- 分子动力学显示特定原子对的磁矩夹角出现非物理振荡

**不加：**
- 只需要随机方向扰动覆盖 → 用 `Magmom Rotation`
- 需要离散比例翻转或从有序到无序的梯度 → 用 `Spin Disorder`
- 需要整体磁序翻转（不是局部 canting）→ 用 `Magnetic Order` 的 AFM 分支
- 需要连续螺旋调制 → 用 `Spin Spiral`

## 参数说明

### 倾斜目标

#### Canting Mode（canting_mode）
类型：`str`。默认：`'Single-spin tilt'`。选择小角度自旋倾斜的目标类型。

| 模式 | 含义 | 适用 |
|------|------|------|
| `Single-spin tilt` | 单独偏转选定原子的磁矩 | 验证流程、研究特定位点 |
| `Global tilt` | 所有 eligible 磁矩按同一角度偏转 | 外场下整体偏转、spin-flop 近似路径 |
| `Atom pair canting` | 左右两侧原子分别偏转 ±θ/2 | DMI 训练集首选 |
| `Group pair canting` | 两组原子整体分别偏转 ±θ/2 | 子晶格级别 canting |

#### Target Mode（target_mode）
类型：`str`。默认：`'First eligible atom'`。选择目标原子的解释方式。

物理直觉：单原子验证用 first/manual；需要系统性覆盖局部环境时用 all eligible。全量目标会显著放大输出数量。

#### Target Indices（target_indices）
类型：`str`。默认：`''`。手动指定要倾斜的原子索引列表。

仅显式索引模式生效。格式 `1,3-5`。

生效条件：`target_mode` 使用手动索引时。

### 原子对目标

#### Pair Left Indices（pair_left_indices）
类型：`str`。默认：`''`。指定原子对左侧索引列表。

手动模式时，左右两侧 1-based 索引列表，一一配对。

生效条件：`pair_source` 选择手动索引时。

#### Pair Right Indices（pair_right_indices）
类型：`str`。默认：`''`。指定原子对右侧索引列表。

手动模式时，左右两侧 1-based 索引列表，一一配对。

生效条件：`pair_source` 选择手动索引时。

#### Pair Source（pair_source）
类型：`str`。默认：`'Manual indices'`。选择原子对来自手动列表还是自动近邻搜索。

物理直觉：手动索引适合可控验证；自动近邻壳层适合批量生成 DMI/交换路径样本，但必须检查元素对和键方向筛选。

#### Pair Shell（pair_shell）
类型：`int`。默认：`1`。选择第几近邻壳层。

自动模式时，第几近邻壳层。1 为第一近邻，2 为第二近邻。

生效条件：`pair_source` 选择近邻自动搜索时。

#### Pair Shell Tolerance（pair_shell_tolerance）
类型：`float`。默认：`0.05`。设置近邻壳层距离容差。

自动分壳层的距离容差，单位 Angstrom。

生效条件：`pair_source` 选择近邻自动搜索时。

#### Pair Element Filter（pair_element_filter）
类型：`str`。默认：`''`。按元素对筛选原子对。

物理直觉：按元素对限制自动近邻 pair，例如只看 Fe-Fe 或 Fe-Co。DMI/交换路径分析必须匹配目标相互作用元素对。

生效条件：自动生成原子对后需要按元素筛选时。

#### Pair Group Filter（pair_group_filter）
类型：`str`。默认：`''`。按 group 对筛选原子对。

物理直觉：按 group 对限制自动近邻 pair，适合层状 AFM、界面或已标记子晶格。没有上游 group 标签时不要使用。

生效条件：自动生成原子对后需要按 group 筛选时。

#### Bond Filter Mode（bond_filter_mode）
类型：`str`。默认：`'Any'`。选择键方向筛选方式。

物理直觉：`Any` 保留所有候选键；`Near axis` 选接近某方向的键；`Near plane` 选接近某晶面的键，用于区分面内/面外相互作用。

#### Bond Filter Axis（bond_filter_axis）
类型：`list[float] | tuple[float, float, float]`。默认：`(0.0, 0.0, 1.0)`。设置键方向筛选参考轴。

物理直觉：用于区分接近某方向或某平面的键。研究面内/面外 DMI 时必须和晶体取向一致。

生效条件：`bond_filter_mode` 不是关闭状态时。

#### Bond Filter Tolerance（bond_filter_tolerance）
类型：`float`。默认：`20.0`。设置键方向筛选角度或投影容差。

物理直觉：方向筛选容差越小，选出的键越接近目标轴或目标平面；容差太小可能没有 pair，太大会混入不该比较的交换路径。

生效条件：`bond_filter_mode` 不是关闭状态时。

### Group Pair 模式

#### Group A（group_a）
类型：`str`。默认：`'A'`。指定 A 组原子或 group 标签。

`arrays['group']` 中的标签名。需要输入已有 group 标签。

生效条件：需要 group pair、手动 group 或 AFM group 模式时。

#### Group B（group_b）
类型：`str`。默认：`'B'`。指定 B 组原子或 group 标签。

`arrays['group']` 中的标签名。需要输入已有 group 标签。

生效条件：需要 group pair、手动 group 或 AFM group 模式时。

### 角度和手性

#### Angle List（angle_list）
类型：`str`。默认：`'1,2,5,10'`。设置需要扫描的小角度列表。

逗号分隔的偏转角列表，单位度。推荐从 `1,2,5,10` 开始。

#### Tilt Signs（tilt_signs）
类型：`str`。默认：`'Positive only'`。设置角度正负手性组合。

物理直觉：只做 +θ 可以验证局部响应；同时做 ±θ 才能提取手性不对称项，DMI 数据建议用成对输出。

#### Include Reference（include_reference）
类型：`bool`。默认：`True`。决定是否额外输出未倾斜的参考磁构型。

是否额外输出一帧未偏转的参考磁态，方便做 energy difference 对比。

### 磁矩与参考态

#### Magnitude Source（magnitude_source）
类型：`str`。默认：`'Existing initial magmoms'`。选择磁矩幅值来自元素映射、默认值还是已有结构。

物理直觉：已有 `initial_magmoms` 时复用它最安全；没有磁矩输入时用 `magmom_map`/`default_moment` 构造幅值。不要用默认幅值替代已知元素磁矩。

#### Magmom Map（magmom_map）
类型：`str`。默认：`''`。按元素指定磁矩幅值或方向，例如 `Fe:2.2, Ni:0.6`。

物理直觉：已知元素局域磁矩时显式写入，例如 `Fe:2.2,Ni:0.6`。未知元素不要用默认值伪造先验。

#### Default Moment（default_moment）
类型：`float`。默认：`0.0`。为没有显式元素映射的原子提供默认磁矩幅值。

物理直觉：只作为 `magmom_map` 未命中的兜底幅值。关键磁性元素应显式列出，非磁元素通常保持 0。

#### Lift Scalar（lift_scalar）
类型：`bool`。默认：`True`。决定是否把标量磁矩提升为非共线向量。

物理直觉：输入是标量磁矩但下游需要非共线向量时打开；如果原始数据已有方向，不要重新提升覆盖方向信息。

#### Axis（axis）
类型：`list[float] | tuple[float, float, float]`。默认：`(0.0, 0.0, 1.0)`。设置共线输入提升为非共线磁矩时的参考轴。

物理直觉：这是方向参考，不是普通数值。改它会改变分层、表面法向或磁矩方向；使用前先确认 cell 取向和目标物理方向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

#### Reference Direction（reference_direction）
类型：`list[float] | tuple[float, float, float]`。默认：`(1.0, 0.0, 0.0)`。定义参考自旋方向，用于构造相对倾斜或 cone。

canting 平面的首选侧向参考方向。

#### Apply Elements（apply_elements）
类型：`str`。默认：`''`。限制只处理指定元素，留空表示处理所有元素。

限制哪些元素参与目标筛选。

### 输出预算

#### Max Outputs（max_outputs）
类型：`int`。默认：`100`。限制这张卡最多输出多少个结构。

物理直觉：自动 pair、角度列表和正负手性会相乘。先用 20-100 检查 pair 选择，再扩大到完整 DMI 扫描。

## 推荐预设

### 单自旋验证（~5 个输出，先确认流程正确）
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

### 近邻对 DMI 训练集（~100 个输出，常规用途）
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

### 筛选键方向 + 元素对的深度 DMI（~500 个输出，研究级）
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

- `Set Magnetic Moments` → `Small-Angle Spin Tilt`：先统一向量磁矩，再批量生成成对 canting
- `Magnetic Order` → `Small-Angle Spin Tilt`：先生成稳定参考磁态，再做局部 canting
- `Group Label` → `Magnetic Order` → `Small-Angle Spin Tilt`：先分组再切到 `Group pair canting`

## 常见问题

**输出只有参考态没有 canting 帧。** 检查是否有符合条件的磁性原子（有非零磁矩的 eligible 元素）。`apply_elements` 是否过滤掉了所有目标？pair 模式下找到的对是否为空？

**pair 自动找对结果不对。** 调整 `pair_shell_tolerance`——太大把不同壳层并到一起，太小把同一壳层拆开。检查 `pair_element_filter` 和 `pair_group_filter` 是否过紧。

**group pair canting 没生效。** 输入需要 `arrays['group']` 且包含 `group_a` 和 `group_b` 指定的标签。用 `Group Label` 先生成。

**输出数量多于预期。** `Target Mode = All eligible atoms` + `Both (+/- pair)` 会快速膨胀。设 `max_outputs` 上限或改用 `First eligible atom`。

## 输出标签

- `SpinTiltRef`：参考态（`include_reference=true` 时）
- `SpinTilt(i=...,a=...,sg=...)`：单自旋偏转
- `SpinPair(i=...,j=...,a=...,sg=...)`：原子对 canting
- `SpinPairG(A=...,B=...,a=...,sg=...)`：group pair canting

所有输出写入 `initial_magmoms` 向量数组。

## 可复现性

无随机性。相同输入、相同参数 → 严格一致输出。`reference_direction` 会先对基准磁矩方向正交化，结果是确定性的。
