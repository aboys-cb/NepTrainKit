<!-- card-schema: {"card_name": "Spin Spiral", "source_file": "src/NepTrainKit/ui/views/_card/spin_spiral_card.py", "serialized_keys": ["params"]} -->

# 自旋螺旋（Spin Spiral）

`Group`: `Magnetism` | `Class`: `SpinSpiralCard`

## 功能说明

对输入结构按一维相位场写入非共线 `initial_magmoms`，生成一批不同周期、相位、手性和轴向分量的 spin spiral / helix / conical spiral 初始构型。支持两种等价的主控方式：直接扫周期 L_D 或扫每 Angstrom 转角梯度。

$$\mathbf{m}(u)=\sqrt{1-m_z^2}\left[\cos\phi(u)\,\mathbf{e}_1+\sin\phi(u)\,\mathbf{e}_2\right] + m_z\,\hat{\mathbf{n}}$$

$$\phi(u)=s\cdot \frac{2\pi u}{L_D}+\phi_0,\qquad s\in\{-1,+1\}$$

其中 m_z = m_parallel / |m| 是无量纲轴向分量。m_z=0 为平面 helix，m_z=0 为 conical spiral。

**关键限制：** 这张卡只写入初始磁矩纹理，不改变晶体结构。如果当前晶格与目标周期不相容，需要先扩胞。

## 操作示例

### 场景：模型在螺旋磁结构上能量曲线完全不对

你训练了一个 NEP 模型，训练集里有 FM 和 AFM 构型。模型在共线磁序上表现尚可，但一跑到具有螺旋磁序的构型（如 CrNb3S6 或 MnSi 中的 chiral helimagnet），能量曲线形状完全错误——周期依赖的能量极小值位置偏离 DFT 结果超过 30%。

**诊断思路：** 模型从未见过磁矩在空间中连续旋转的构型。FM/AFM 训练集隐式告诉了模型"磁矩方向在空间中是分段常数"的假设。需要在训练集里加入不同周期的螺旋磁纹理，让模型学习 q 空间不同波矢下的能量响应。

**输入：** 一个已知磁矩幅值的磁性结构（如 MnSi 单胞，Mn 磁矩约 2.0 μB，螺旋沿 [111] 方向）

**目标：** 沿 [111] 方向扫描 3 个周期（20/30/40 Angstrom），3 个全局相位（0/30/60 度），纯 helix（mz=0），正反手性成对。共 3x3x2 = 18 个螺旋构型

**参数设置：**
- `Propagation Axis` = `[1, 1, 1]`
- `Spiral Parameter` = `Period (L_D)`
- `Period Range` = `[20, 40, 10]`
- `Phase Range` = `[0, 60, 30]`
- `m_parallel Range` = `[0, 0, 0.1]`
- `Chirality` = `Both`

**输出：** 18 个结构，磁矩沿 [111] 方向逐层旋转，带 `Helix(L=...,ph=...,mz=0,chi=...,ax=...)` 标签。

**怎么验证训练集质量改善：**
- 重训后用 DFT 计算几个螺旋构型的能量作为参考，对比模型预测的 E(q) 曲线
- 如果极小值位置仍偏离，加密 `period_range` 扫描（例如 `[10, 40, 5]`）
- 如果发现 conical spiral（mz != 0）的能量比纯 helix 更低，加入 `mz = [0, 0.5, 0.1]` 覆盖锥面螺旋
- 如果需要晶格相容的周期，勾选 `Period Filter` 开启整周期约束

### 什么时候加这张卡、什么时候不加

**加：**
- 研究体系存在螺旋/非共线磁基态（如 chiral magnet、skyrmion 宿主材料）
- 模型在非共线磁序上泛化失败
- 需要系统覆盖 q 空间的磁激发

**不加：**
- 体系只有共线磁序 → `Magnetic Order` 够用
- 需要局部小角度偏转而非长程螺旋 → 用 `Small-Angle Spin Tilt`
- 晶体结构本身需要扩胞才能容纳目标周期 → 先扩胞再回来

## 参数说明

### 传播方向

**`Propagation Axis`**（axis）：`[x, y, z]`。螺旋相位沿这个方向传播。原子坐标在该方向的投影用于计算相位。

### 主控参数（二选一）

**`Spiral Parameter`**（spiral_parameter_mode）：选 `Period (L_D)` 扫周期（Angstrom），或 `Angle gradient (deg/A)` 扫每 Angstrom 转角。两者等价。

**`Period Range`**（period_range）：`[min, max, step]`，单位 Angstrom。
- 保守：`[20, 20, 5]`（单周期）
- 平衡：`[10, 40, 10]`（4 个周期）
- 探索：`[4, 80, 4]`（宽范围）

**`Angle Gradient Range`**（angle_gradient_range）：`[min, max, step]`，单位 deg/A。值越大旋转越快。360/L_D = 梯度。

### 相位和轴向分量

**`Phase Range`**（phase_range）：`[min, max, step]`，全局相位偏移，单位度。
- 保守：`[0, 0, 15]`（单个相位）
- 平衡：`[0, 90, 30]`（4 个相位）
- 探索：`[-180, 180, 30]`（全相位空间）

**`m_parallel Range`**（mz）：`[min, max, step]`，沿传播轴的单位化分量，范围 [-1, 1]。0 = 纯 helix，非零 = conical spiral。
- 保守：`[0, 0, 0.1]`（纯 helix）
- 平衡：`[0, 0.5, 0.1]`（弱 conical）
- 探索：`[-0.9, 0.9, 0.1]`（全 conical 空间）

### 手性

**`Chirality`**（chirality）：`Clockwise` / `Counterclockwise` / `Both`。选 `Both` 会对同一组参数生成一对手性相反的构型。

### 相位模式

**`Phase Mode`**（phase_mode）：
- `Continuous by position`：每个原子按自身投影坐标独立计算相位。标准连续螺旋。
- `Layer-locked`：先按投影坐标分层，同层原子共享相位。适合层状体系。

**`Layer Tolerance`**（layer_tolerance）：仅在 `Layer-locked` 模式下生效。投影差小于此阈值的原子归为同一层。

### 整周期约束

**`Period Filter`**（only_commensurate_periods）：勾选后只保留与当前晶格周期边界相容的周期。开启后程序在指定区间内自动搜索相容周期。如果没有相容周期，卡片会给出建议的超胞倍数。

### 磁矩幅值

**`Magnitude Source`**（magnitude_source）：`Existing initial magmoms` 或 `Map/default magnitude`。

**`Magmom Map`** / **`Default Moment`**：仅在 `Map/default magnitude` 模式下生效。

**`Apply Elements`**（apply_elements）：限制哪些元素施加螺旋纹理。

### 输出上限

**`Max Outputs`**（max_outputs）：防止"周期数 x 相位数 x mz 数 x 手性数"组合膨胀。16（保守），50~200（常规），500+（需配合筛选）。

## 推荐预设

### 单周期纯 helix 验证（2 个输出，先确认方向正确）
```json
{
  "class": "SpinSpiralCard",
  "check_state": true,
  "axis": [0.0, 0.0, 1.0],
  "spiral_parameter_mode": "Period (L_D)",
  "period_range": [20.0, 20.0, 10.0],
  "angle_gradient_range": [18.0, 18.0, 1.0],
  "phase_range": [0.0, 0.0, 15.0],
  "mz": [0.0, 0.0, 0.1],
  "chirality": "Both",
  "phase_mode": "Continuous by position",
  "layer_tolerance": [0.05],
  "only_commensurate_periods": false,
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [16]
}
```

### 多周期多相位螺旋（~30 个输出，常规 E(q) 曲线拟合）
```json
{
  "class": "SpinSpiralCard",
  "check_state": true,
  "axis": [0.0, 0.0, 1.0],
  "spiral_parameter_mode": "Period (L_D)",
  "period_range": [10.0, 40.0, 10.0],
  "angle_gradient_range": [18.0, 18.0, 1.0],
  "phase_range": [0.0, 90.0, 30.0],
  "mz": [0.0, 0.3, 0.1],
  "chirality": "Both",
  "phase_mode": "Layer-locked",
  "layer_tolerance": [0.05],
  "only_commensurate_periods": false,
  "magnitude_source": "Existing initial magmoms",
  "magmom_map": "",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [100]
}
```

### 全参数空间扫描（~500 个输出，研究级，含 conical + 整周期约束）
```json
{
  "class": "SpinSpiralCard",
  "check_state": true,
  "axis": [0.0, 0.0, 1.0],
  "spiral_parameter_mode": "Angle gradient (deg/A)",
  "period_range": [20.0, 40.0, 10.0],
  "angle_gradient_range": [4.5, 90.0, 4.5],
  "phase_range": [-180.0, 180.0, 30.0],
  "mz": [0.0, 0.8, 0.2],
  "chirality": "Both",
  "phase_mode": "Layer-locked",
  "layer_tolerance": [0.08],
  "only_commensurate_periods": true,
  "magnitude_source": "Map/default magnitude",
  "magmom_map": "Fe:2.2",
  "default_moment": [0.0],
  "apply_elements": "",
  "max_outputs": [500]
}
```

## 推荐组合

- `Magnetic Order` → `Spin Spiral`：先确定局域磁矩模长，再扫螺旋周期和 mz
- `Set Magnetic Moments` → `Spin Spiral`：先统一写入磁矩幅值，再做螺旋
- `Super Cell` → `Spin Spiral`：扩胞后容纳更长周期，再用整周期约束锁相

## 常见问题

**输出为空 / 只有原始输入。** 磁矩幅值全为 0（检查 `magnitude_source` 和 `magmom_map`）。`Period Filter` 开启且区间内没有相容周期——卡片会打印建议超胞倍数。

**相邻层相位递进不对。** 检查 `Propagation Axis` 是否指向预期的传播方向。`Phase Mode = Layer-locked` 时调整 `layer_tolerance`。

**输出比预期的多很多。** `period_range` 步长太小，或 `mz` 范围太宽。设 `max_outputs` 上限，先小步长试跑再扩大。

**conical spiral 的 mz 值看不懂。** mz 是无量纲比值 m_parallel / |m|，不是以 mu_B 计的绝对磁矩。物理上它只能取 [-1, 1]。

## 输出标签

- `Helix(L=...,ph=...,mz=0,chi=...,ax=...)`：纯平面 helix
- `Spiral(L=...,ph=...,mz=...,chi=...,ax=...)`：conical spiral
- 附加 `pm=layer,ltol=...`：仅 layer-locked 模式

所有输出写入 `initial_magmoms` 三列向量。

## 可复现性

无随机性。相同输入结构与相同参数 → 严格一致输出。相位原点固定为传播轴最小投影位置。
