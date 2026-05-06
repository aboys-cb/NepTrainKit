<!-- card-schema: {"card_name": "Small-Angle Spin Tilt", "source_file": "src/NepTrainKit/ui/views/_card/small_angle_spin_tilt_card.py", "serialized_keys": ["params"]} -->

# 小角度自旋倾斜（Small-Angle Spin Tilt）

`Group`: `Magnetism` | `Class`: `SmallAngleSpinTiltCard`

## 功能说明

对选定目标原子或原子对做确定性小角度磁矩偏转（canting）。支持三种模式：单自旋偏转、显式原子对 canting、两组原子的 group-pair canting。后两种模式显式构造 S_i x S_j 的正/负手性对，直接服务于 DMI 训练集。

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
- 分子动力学显示特定原子对的磁矩夹角出现非物理振荡

**不加：**
- 只需要随机方向扰动覆盖 → 用 `Magmom Rotation`
- 需要整体磁序翻转（不是局部 canting）→ 用 `Magnetic Order` 的 AFM 分支
- 需要连续螺旋调制 → 用 `Spin Spiral`

## 参数说明

### Canting 模式

**`Canting Mode`**（canting_mode）：

| 模式 | 含义 | 适用 |
|------|------|------|
| `Single-spin tilt` | 单独偏转选定原子的磁矩 | 验证流程、研究特定位点 |
| `Atom pair canting` | 左右两侧原子分别偏转 ±θ/2 | DMI 训练集首选 |
| `Group pair canting` | 两组原子整体分别偏转 ±θ/2 | 子晶格级别 canting |

### 单自旋模式的目标选择

**`Target Mode`**（target_mode）：`First eligible atom`（最保守）、`All eligible atoms`（按站点展开）、`Explicit indices (1-based)`（精确指定）。

**`Target Indices`**（target_indices）：仅显式索引模式生效。格式 `1,3-5`。

### 原子对模式的目标选择

**`Pair Source`**（pair_source）：`Manual indices`（手动指定左右索引）或 `Auto by neighbor shell`（按近邻壳层自动找对）。

**`Pair Left/Right Indices`**（pair_left_indices / pair_right_indices）：手动模式时，左右两侧 1-based 索引列表，一一配对。

**`Pair Shell`**（pair_shell）：自动模式时，第几近邻壳层。1 为第一近邻，2 为第二近邻。

**`Pair Shell Tolerance`**（pair_shell_tolerance）：自动分壳层的距离容差，单位 Angstrom。

**过滤条件（仅在自动模式下生效）：**

- **`Pair Element Filter`**（pair_element_filter）：`Fe-Fe,Fe-Co` 格式，只保留指定元素组合的对
- **`Pair Group Filter`**（pair_group_filter）：`A-B,A-A` 格式，需要输入有 `arrays['group']`
- **`Bond Filter Mode`**（bond_filter_mode）：`Any` / `Near axis` / `In plane (normal)`，按键方向筛选
- **`Bond Filter Axis`** / **`Bond Filter Tolerance`**：键方向筛选的参考轴和角度容差

### Group pair 模式

**`Group A/B`**（group_a / group_b）：`arrays['group']` 中的标签名。需要输入已有 group 标签。

### 角度和手性

**`Angle List`**（angle_list）：逗号分隔的偏转角列表，单位度。推荐从 `1,2,5,10` 开始。

**`Tilt Signs`**（tilt_signs）：`Positive only`（只 +θ）、`Negative only`（只 -θ）、`Both (+/- pair)`（成对输出，DMI 必选）。

### 参考态

**`Include Reference`**（include_reference）：是否额外输出一帧未偏转的参考磁态，方便做 energy difference 对比。

### 磁矩来源

**`Magnitude Source`**（magnitude_source）：`Existing initial magmoms`（优先）或 `Map/default magnitude`。

**`Magmom Map`** / **`Default Moment`**：仅在 `Map/default magnitude` 模式下生效。

**`Lift Scalar`** / **`Axis`**：标量磁矩抬升为向量的开关和参考轴。

**`Reference Direction`**（reference_direction）：canting 平面的首选侧向参考方向。

**`Apply Elements`**（apply_elements）：限制哪些元素参与目标筛选。

### 输出上限

**`Max Outputs`**（max_outputs）：防止"目标数 x 角度数 x 手性数"组合膨胀。16（保守），50~200（常规），500+（需配合筛选）。

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
