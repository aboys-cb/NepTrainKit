<!-- card-schema: {"card_name": "Folded Helix", "source_file": "src/NepTrainKit/ui/views/_card/folded_helix_card.py", "serialized_keys": ["params"]} -->

# 折返螺旋（Folded Helix）

`Group`: `Magnetism` | `Class`: `FoldedHelixCard`

## 功能说明

按层离散定义对称折返螺旋磁矩纹理。磁矩被限制在垂直于 `plane_normal` 的平面内，沿 `layer_axis` 方向在前半周期逐层按固定角度旋转、到转折层后按相同步长反向旋转，形成一个"先顺时针、到中间再逆时针"的三角相位轮廓。

$$s(k)=\begin{cases}k,&0\le k\le h\\2h-k,&h\lt k\lt2h\end{cases}$$

$$\phi(k)=\phi_0+\sigma\cdot s(k)\cdot\Delta\phi$$

默认 `half_period_mode = Auto from layer count`，会在当前层范围上自动构造首尾闭合的折返周期。

**和 `Spin Spiral` 的区别：** `Spin Spiral` 沿传播轴连续旋转（单向、不折返）。本卡生成的是"走一段、折回来"的分层离散纹理，更适合二维磁体、异质结界面或需要镜像对称磁矩分布的场景。

## 操作示例

### 场景：层状反铁磁模型在非均匀自旋纹理上预测崩塌

你训练了一个层状磁性模型（如 CrI3 双层），训练数据里层内是 FM、层间是 AFM——这是完美均匀的磁序。但实验发现施加电场后，顶层磁矩偏了 15 度而底层偏了 -15 度——这是一个非单调的、分层的自旋纹理。模型对此完全无法预测。

**诊断思路：** 训练集里的自旋变化是单调的（要么全同向、要么交替翻转）。模型从未见过磁矩方向在空间中"先转过去、再转回来"的纹理。需要生成按层离散的折返螺旋构型，让模型学习分层磁矩分布的能量面。

**输入：** 一个已有 `initial_magmoms` 的层状磁性结构（或通过 `magnitude_source = Map/default magnitude` 指定磁矩模长）

**目标：** 用 auto 模式自动适配当前层数，生成 3 个层间转角（15/30/45 度）、4 个全局相位（0/30/60/90 度）、2 种手性顺序，共 3x4x2 = 24 个折返螺旋构型

**参数设置：**
- `Layer Axis` = `[0, 0, 1]`（沿 z 轴分层）
- `Plane Normal` = `[0, 0, 1]`（磁矩在 xy 面内旋转）
- `Half-Period Mode` = `Auto from layer count`
- `Angle Step Range` = `[15, 45, 15]`
- `Phase Range` = `[0, 90, 30]`
- `Sequence` = `Both`

**输出：** 24 个结构，磁矩在 xy 面内按层折返旋转，带 `FoldedHelix(h=...,da=...,ph=...,seq=...,ax=...,pn=...)` 标签。

**怎么验证训练集质量改善：**
- 重训后跑几组不同层间转角的测试构型，能量排序应合理（近 FM 的转角能量低，近 180 度的高）
- 抽查输出：层 0 磁矩方向 = 全局相位，中间层达到最大转角，顶层回到接近全局相位
- 如果 auto 模式给出的半周期不合适——比如你的结构有 10 层但 auto 给了 h=4——切到 manual 模式手动指定 `half_period_layers`

### 什么时候加这张卡、什么时候不加

**加：**
- 层状磁体（vdW 磁体、异质结、超晶格）需要非单调的分层磁矩纹理
- 想构造镜像对称的自旋分布（两半周期互为镜像）
- 需要同层原子共享磁矩方向（layer-locked 行为是内置的）

**不加：**
- 需要标准单向传播的螺旋 → 用 `Spin Spiral`
- 需要连续坐标依赖的相位（非分层）→ 用 `Spin Spiral` 的 continuous 模式
- 体系没有层状结构 → 分层无意义

## 参数说明

### 分层定义

**`Layer Axis`**（layer_axis）：`[x, y, z]`。原子坐标沿此方向投影后用于分层。

**`Plane Normal`**（plane_normal）：`[x, y, z]`。磁矩旋转平面的法向。例如 `[0, 0, 1]` 表示磁矩在 xy 面内旋转。

**`Layer Tolerance`**（layer_tolerance）：投影坐标差不超过此阈值的原子归为同一层，单位 Angstrom。
- 保守：`0.01`（严格分层）
- 平衡：`0.03~0.10`（容忍小幅层内起伏）
- 如果层内有明显 rumpling，适当放宽

### 半周期控制

**`Half-Period Mode`**（half_period_mode）：
- `Auto from layer count`：由当前层数自动推导半周期，保证首尾闭合。大多数情况下用这个。
- `Manual`：手动指定 `half_period_layers` 扫描范围，适合比较不同折返周期。

**`Half-Period Layers`**（half_period_layers）：仅在 manual 模式下生效。`[min, max, step]`，半周期的层步进数。
- 保守：`[2, 2, 1]`（最短折返）
- 平衡：`[2, 6, 1]`
- 探索：`[4, 12, 2]`

### 旋转参数

**`Angle Step Range`**（angle_step_range）：`[min, max, step]`，相邻层之间的面内转角，单位度。
- 保守：`[5, 15, 5]`（小转角）
- 平衡：`[15, 45, 15]`（中转角）
- 探索：`[30, 90, 15]`（大转角）

**`Phase Range`**（phase_range）：`[min, max, step]`，全局相位偏移，单位度。
- 保守：`[0, 0, 15]`
- 平衡：`[0, 90, 30]`
- 探索：`[-180, 180, 30]`

### 手性顺序

**`Sequence`**（sequence_mode）：
- `Clockwise then counterclockwise`：前半周期顺时针、后半周期逆时针
- `Counterclockwise then clockwise`：前半周期逆时针、后半周期顺时针
- `Both`：两种都生成

### 磁矩幅值

**`Magnitude Source`**（magnitude_source）：`Existing initial magmoms` 或 `Map/default magnitude`。

**`Magmom Map`** / **`Default Moment`**：仅在 `Map/default magnitude` 模式下生效。

**`Apply Elements`**（apply_elements）：限制哪些元素施加折返纹理。

### 输出上限

**`Max Outputs`**（max_outputs）：控制总输出数。16（保守），50~200（常规），500+（配合筛选）。

## 推荐预设

### auto 模式单折返验证（~2 个输出，先确认分层和方向正确）
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

### 多转角多相位常规覆盖（~24 个输出，适合层状磁体训练）
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

### 手动多周期 + 全相位探索（~500 个输出，研究级）
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

- `Magnetic Order` → `Folded Helix`：先建立 FM/AFM 局域磁矩模长，再按层折返旋转
- `Set Magnetic Moments` → `Folded Helix`：手动指定元素模长，再生成折返纹理
- `Group Label` → `Folded Helix`：先打 group 标签，再在特定子晶格上施加折返

## 常见问题

**输出只有原始输入。** 磁矩幅值全为 0。检查 `magnitude_source` 和 `magmom_map`。结构只有 1 层时折返无意义——至少需要 2 层。

**分层结果不符合预期。** `layer_axis` 方向是否正确？调整 `layer_tolerance`——太小把同层拆散，太大把不同层合并。

**auto 模式的半周期太小或太大。** auto 模式取 `(总层数 - 1) // 2`。如果结构有偶数层且你期望一个层作为转折峰，切到 manual 模式手动指定 `half_period_layers`。

**输出相位看起来没有折返。** 如果层数太少（如 3 层），折返曲线不明显。用 6 层以上的结构效果更直观。

## 输出标签

- `FoldedHelix(h=...,da=...,ph=...,seq=...,ax=...,pn=...)`
  - `h`：半周期层步数
  - `da`：层间转角
  - `ph`：全局相位
  - `seq`：手性顺序（`cw-ccw` 或 `ccw-cw`）
  - `ax`：分层轴标签
  - `pn`：旋转平面法向标签

所有输出写入 `initial_magmoms` 三列向量。

## 可复现性

无随机性。相同输入结构和相同参数 → 严格一致输出。分层原点固定为沿 `layer_axis` 投影后的最小层号。
