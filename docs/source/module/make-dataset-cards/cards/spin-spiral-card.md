:orphan:

<!-- card-schema: {"card_name": "Spin Spiral", "source_file": "src/NepTrainKit/ui/views/_card/spin_spiral_card.py", "serialized_keys": ["params"]} -->

# 旧版自旋螺旋（Spin Spiral）

**分类：** 磁性

> [!WARNING]
> 这是一张仅用于加载旧工作流的兼容卡片，已从“新建卡片”目录隐藏。新建有限 q、Bloch、cycloidal 或一般螺旋路径请使用 [SOC / 纹理响应](soc-texture-response-card.md)。

## 功能说明

旧版卡片按原子在传播轴上的投影坐标写入螺旋磁矩，并枚举周期、相位、手性和锥分量。它保留每个原子的磁矩模长，只改变方向。

该能力与 `SOC / Texture Response` 的有限 q 路径重复；新版卡片还提供完整路径、共格失败和响应元数据合同，因此不再为旧版卡片新增功能。

## 原理与公式

对第 $i$ 个原子，旧版卡片生成

$$
\mathbf m_i=M_i\left\{\sqrt{1-m_\parallel^2}
[\cos\phi_i\,\mathbf e_1+\sin\phi_i\,\mathbf e_2]
+m_\parallel\hat{\mathbf n}\right\},
$$

$$
\phi_i=s\frac{2\pi u_i}{L}+\phi_0,
$$

其中 $M_i$ 是输入磁矩模长，$u_i$ 是原子坐标在笛卡尔传播轴 $\hat{\mathbf n}$ 上的投影，$s=\pm1$ 表示两种手性。$m_\parallel=0$ 为平面螺旋，$0<|m_\parallel|<1$ 为锥形螺旋。

勾选公度筛选后，周期必须让所有周期晶格矢量上的相位增量都是 $360°$ 的整数倍；找不到相容周期时会明确失败，不再把原结构当作生成结果。

## 操作示例

旧工作流若保存了 $L=20$ Å、双手性和 $m_\parallel=0$，仍可加载并复现两份平面螺旋结构。若要新建同类路径，在 `SOC / Texture Response` 中选择 `General spiral`，把传播方向设为原 `axis`，并使用

$$|\mathbf q|=2\pi/L$$

换算基准波矢。运行前应先确认 q 与晶胞共格。

## 迁移到 SOC / 纹理响应

| 旧版设置 | 新版对应设置 |
| --- | --- |
| `axis` | `Propagation direction q (Cartesian)` |
| `period_range` | 将每个 $L$ 换算为 $|q|=2\pi/L$；需要多段时拆成明确路径 |
| `angle_gradient_range` | 先用 $L=360/g$ 换算周期，再换算 q |
| `phase_range` | `Phase Deg`；多个相位使用多张明确配置 |
| `mz` | `Cone Component` |
| `chirality` | 使用正负 q 扫描 |
| `only_commensurate_periods` | `Require Commensurate`，新卡默认开启 |
| `max_outputs` | `Max Outputs`；新卡只接受完整路径 |

`Layer-locked` 没有直接迁移项；如果确实需要离散分层且折返的纹理，使用“折返螺旋磁序”。

## 参数说明

以下参数仅用于理解和复现旧 JSON。

### 传播轴（axis）

笛卡尔三维向量，默认 `(0,0,1)`。原子坐标在该方向上的投影决定螺旋相位。

### 螺旋参数方式（spiral_parameter_mode）

默认 `Period (L_D)`。也可通过 `Angle gradient (deg/A)` 按每 Å 转角定义周期。

### 周期范围（period_range）

默认 `(20,40,10)` Å，格式为 `[最小值, 最大值, 步长]`。

### 角度梯度范围（angle_gradient_range）

默认 `(18,18,1)` 度/Å，格式为 `[最小值, 最大值, 步长]`，与周期满足 $g=360/L$。

### 相位范围（phase_range）

默认 `(0,0,15)` 度。每个相位偏移都会增加一组输出。

### 轴向分量范围（mz）

默认 `(0,0,0.1)`，表示归一化轴向分量 $m_\parallel$，范围 `[-1,1]`。

### 手性（chirality）

默认 `Both`。`Clockwise` 和 `Counterclockwise` 分别生成一个旋向；`Both` 生成一对。

### 相位模式（phase_mode）

默认 `Continuous by position`。`Layer-locked` 会把投影距离足够接近的原子设为相同相位。

### 层容差（layer_tolerance）

默认 0.05 Å，仅在 `Layer-locked` 模式下用于合并相邻投影层。

### 仅保留公度周期（only_commensurate_periods）

默认关闭。开启后只生成与当前晶胞和 PBC 相容的周期；找不到时明确报错并尽可能给出超胞倍数建议。

### 磁矩来源（magnitude_source）

默认 `Existing initial magmoms`，要求输入已有至少一个非零磁矩。`Map/default magnitude` 才会读取元素表和默认幅值。

### 元素磁矩表（magmom_map）

元素表模式下使用，格式如 `Fe:2.2,Ni:0.6`。

### 默认磁矩（default_moment）

默认 0.0。元素表未列出的元素使用该模长。

### 应用元素（apply_elements）

旧实现仅在元素表模式下用它筛选元素；已有磁矩模式不会据此筛选。迁移时应在上游建立明确的参考磁矩。

### 最大输出数（max_outputs）

默认 100。旧版按周期 × 相位 × 轴向分量 × 手性依次生成，并在达到此数量时截断；新版响应卡只接受完整路径。

## 常见问题

**为什么旧工作流现在会报“没有非零磁矩”？** 旧实现曾原样返回输入并显示成功，这并不是螺旋输出。请在上游写入磁矩，或在旧卡中明确选择元素表来源。

**为什么公度筛选会失败？** 请求区间内没有与当前周期晶胞相容的周期。按错误中的建议扩胞，或迁移到 `SOC / Texture Response` 后重新设置 q。

## 输出

成功输出写入 `spin:R:3` 并同步 ASE 初始磁矩；`Config_type` 使用 `Helix(...)` 或 `Spiral(...)`。该操作没有随机性，相同输入与参数产生相同结果。
