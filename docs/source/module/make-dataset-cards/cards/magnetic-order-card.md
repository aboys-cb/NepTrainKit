<!-- card-schema: {"card_name": "Magnetic Order", "source_file": "src/NepTrainKit/ui/views/_card/magnetic_order_card.py", "serialized_keys": ["params"]} -->

# 磁有序初始化（Magnetic Order）

`Group`: `Magnetism` | `Class`: `MagneticOrderCard`

## 功能说明

从一个磁性母相结构出发，一次性生成 FM（铁磁）、AFM（反铁磁）、PM（顺磁）多种磁序分支。每个输出结构都写入对应的 `initial_magmoms` 数组和 `Config_type` 标签。

**关键限制：** 这张卡只写入初始磁矩，不保证生成的就是磁基态。磁矩是否收敛到合理值仍需要用 NEP/DFT 后续计算验证。

## 操作示例

### 场景：FM 模型在 AFM 相上预测崩了

你在 bcc Fe 上训练了一个 NEP 模型，所有训练数据来自 FM 构型。模型对铁磁态的能量和力预测很好，但一跑 AFM 构型，能量误差是 FM 的 5 倍。诊断结果是：训练集里只有一种磁序，模型把"所有磁矩同向"当成了默认假设，碰到磁矩翻转的结构只能瞎猜。

**诊断思路：** 磁性体系的训练集至少应该覆盖 FM 和 AFM 两种极端磁序，如果体系可能有顺磁相，PM 也要加。不同磁序下，同一种原子的局域磁环境完全不同——模型需要见过这些环境才能泛化。

**输入：** 一个含 Fe 的 bcc 晶体，已知 Fe 的磁矩约 2.2 μB（来自 DFT 或文献）

**目标：** 从一个结构出发，生成 1 个 FM + 1 个 AFM（k-vector 111）+ 8 个 PM，共 10 个不同磁序的训练样本

**参数设置：**
- `Format` = `Collinear (scalar)` （先从不共线开始，复杂度低）
- `Magmom Map` = `Fe:2.2`
- `Gen FM` = 勾选，`Gen AFM` = 勾选
- `AFM Mode` = `k-vector`，`AFM Kvec` = `111`
- `Gen PM` = 勾选，`PM Count` = `[8]`

**输出：** 1 个 FM + 1 个 AFM + 8 个 PM，全部带 `initial_magmoms` 和对应的 `Config_type` 标签

**怎么验证训练集质量改善：**
- 重训后用这组数据里的 FM 和 AFM 分别做推理，能量误差差异应该显著缩小
- 抽查 FM/AFM 的 `initial_magmoms`：FM 全正、AFM 在 111 方向上相邻层符号翻转
- 如果 AFM 效果仍差，换 `AFM Kvec` = `110` 或 `100`，覆盖不同翻转周期
- 如果 PM 结构能量分布异常宽，可能是磁矩幅值设置不合理——检查 `magmom_map` 或 `default_moment`

### 什么时候加这张卡、什么时候不加

**加：**
- 模型在不同磁序之间泛化差（FM 推理好、AFM 崩）
- 研究体系本身存在多种磁有序相（如 Fe、NiO、钙钛矿磁性材料）
- 需要模型能区分磁基态和激发态

**不加：**
- 体系是非磁的（纯 Si、Al 等），加了只会引入无意义的磁矩标签
- 只需要静态磁矩初始化，不需要多磁序分支（用 `Set Magnetic Moments`）

## 参数说明

### 磁矩格式和幅值

先把"磁矩写多大、什么格式"定下来，后面 FM/AFM/PM 分支都在这个基础上操作。

#### Format（format）

`str`，默认 `Collinear (scalar)`。`Collinear` = 每个原子只有一个标量 ± 磁矩，沿 `Axis` 方向；`Non-collinear` = 每个原子一个 3D 矢量。

大多数共线磁序用标量就够了，简单且训练数据量小。如果你后续要做非共线螺旋、canting 或 DMI 相关训练，这里必须先切成矢量模式。

#### Axis（axis）

`[x, y, z]`，默认 `[0, 0, 1]`（z 轴）。共线模式下磁矩沿这个方向；非共线模式下 FM/AFM 的默认方向也以它为参考。

改它之前先确认结构的晶胞取向——如果你的 slab 法向是 z，通常保持默认；如果研究面内各向异性，可能要改到 x 或 y。

#### Magmom Map（magmom_map）

`str`，默认空。`Fe:2.2, Ni:0.6` 这种格式，显式指定每个元素的磁矩大小。这是你最精确的控制方式——用已知的 DFT 或文献值填。

#### Use Element Dirs（use_element_dirs）

`bool`，默认 false。打开后不同元素可以有不同的磁矩方向（从 `magmom_map` 中读取方向信息），而不是全沿 `Axis`。非共线多元素体系（比如 Fe↑ Mn↓）才会用到。标量模式下不需要开。

#### Default Moment（default_moment）

`float`，默认 0.0。没在 `magmom_map` 里列出的元素统一用这个值。设 0 意味着非磁性元素不写磁矩。

#### Apply Elements（apply_elements）

`str`，默认空（全部生效）。逗号分隔如 `Fe, Co`。只给这些元素写磁矩，其他的写 0。

> 优先级：`magmom_map` 命中的 → 用映射值；没命中但被 `apply_elements` 包含的 → 用 `default_moment`；都不在的 → 写 0。

### FM / AFM 分支

这两张复选框决定输出里有没有铁磁和反铁磁构型。通常至少保留 FM ——它是一个重要的能量参考态。

#### Gen FM（gen_fm）

`bool`，默认 true。打开 → 输出一个铁磁结构（所有磁矩同向）。关掉它意味着你只要 AFM/PM，比较少见。

#### Gen AFM（gen_afm）

`bool`，默认 true。打开 → 输出反铁磁结构。需要选一种 AFM 构造方式：

#### AFM Mode（afm_mode）

`str`，默认 `k-vector`。`k-vector` 用波矢自动翻转——简单、不需要上游准备。`group A/B` 用手动标注的 group 标签决定正负——更灵活，但需要先用 `Group Label` 卡打好标签。

你很可能先从 k-vector 试起。只有当你发现 k-vector 的翻转面切到了不等价原子（比如 NiO 里 Ni 和 O 两种子晶格被混在一起了），再切到 group A/B。

#### AFM Kvec（afm_kvec）

`str`，默认 `111`。仅 k-vector 模式。`100` / `010` / `001` / `110` / `111`。沿这个方向相邻原子层磁矩正负交替。

#### AFM Group A（afm_group_a）

`str`，默认 `A`。仅 group A/B 模式。`atoms.arrays['group']` 中等于这个标签的原子取正号磁矩。它必须和上游 `Group Label` 卡写入的标签完全一致，包括大小写。

#### AFM Group B（afm_group_b）

`str`，默认 `B`。仅 group A/B 模式。`atoms.arrays['group']` 中等于这个标签的原子取负号磁矩。通常和 `afm_group_a` 成对使用，用来显式指定两套反平行子晶格。

#### AFM Zero Unknown（afm_zero_unknown）

`bool`，默认 true。仅 group A/B 模式。打开 → 不属于 A 也不属于 B 的原子磁矩置零。关掉 → 这些原子默认为正。

### PM 分支

PM 生成大量随机磁序样本。默认关闭的——因为开了输出数量会乘上 `pm_count + 2`。

#### Gen PM（gen_pm）

`bool`，默认 false。打开后才生成顺磁构型。

#### PM Count（pm_count）

`int`，默认 10。生成几个 PM 样本。5-10 覆盖基本无序态，20+ 做统计训练。注意每帧输出 = 1 FM（如果开了）+ 1 AFM（如果开了）+ `pm_count` 个 PM。

#### PM Direction（pm_direction）

`str`，默认 `sphere`。`sphere` — 磁矩方向在球面上均匀撒。`cone` — 限制在以 `Axis` 为中心的锥面内，锥角由下面的 `PM Cone Angle` 控制。如果你要模拟有限温度下自旋偏离平衡方向不太远的情况，用 cone；要模拟完全无序的顺磁态，用 sphere。

#### PM Cone Angle（pm_cone_angle）

`float`，默认 30°。仅 cone 模式。偏离 `Axis` 的最大角度。30° 表示有限温近平衡扰动，60° 接近强无序。

#### PM Balanced（pm_balanced）

`bool`，默认 true。打开后正负方向数量尽量平衡。一般建议保持——不勾的话可能出现某一方向系统性偏多。

### 随机性

#### Use Seed（use_seed）

`bool`，默认 false。打开后固定随机种子 → 同参数同输入可复现。对比实验时开，纯探索可以关。

#### Seed（seed）

`int`，默认 0。仅在 `use_seed` 打开时生效。

## 推荐预设

### FM + AFM 基础集（2 个输出，快速验证用）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2",
  "default_moment": [0.0],
  "apply_elements": "Fe",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "gen_pm": false,
  "use_seed": true,
  "seed": [42]
}
```

### FM + AFM + PM 全磁序（~12 个输出，常规训练用）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2,Ni:0.6",
  "default_moment": [0.0],
  "apply_elements": "",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "gen_pm": true,
  "pm_count": [10],
  "pm_direction": "sphere",
  "pm_balanced": true,
  "use_seed": true,
  "seed": [42]
}
```

### 非共线多磁序（~17 个输出，研究级）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Non-collinear (vector)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2,Cr:1.5",
  "default_moment": [0.5],
  "apply_elements": "Fe,Cr,Mn",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "110",
  "gen_pm": true,
  "pm_count": [15],
  "pm_direction": "cone",
  "pm_cone_angle": [45.0],
  "pm_balanced": true,
  "use_seed": true,
  "seed": [42]
}
```

## 推荐组合

- `Group Label` → `Magnetic Order`：先打 group 标签，再用 group A/B 模式生成细粒度 AFM
- `Magnetic Order` → `Magmom Rotation`：基础磁序 → 小角度旋转补样
- `Magnetic Order` → `Spin Spiral`：磁序初始化幅值 → 螺旋调制

## 常见问题

**输出没有磁矩。** `magmom_map` 和 `default_moment` 是否都为空/0？`apply_elements` 是否过滤掉了所有有磁矩的元素？

**AFM 没翻转。** 如果用 `group A/B` 模式，检查输入是否有 `atoms.arrays['group']` 且包含 A/B 标签。没有的话换 `k-vector` 模式。

**PM 没生成。** `gen_pm` 默认关闭，确认已勾选。

**非共线磁矩方向不符合预期。** 检查 `Axis` 是否正确。非共线 FM/AFM 模式下，每个原子的磁矩方向沿 `Axis`，只有 PM 才会随机偏离。

## 输出标签

- `MagFM` / `MagFMnc`：铁磁（共线/非共线）
- `MagAFM111` / `MagAFM110`：反铁磁，后缀为 k-vector
- `MagAFMg`：group A/B 模式反铁磁
- `MagPM` / `MagPMnc`：顺磁，可能带 seed 后缀

所有输出写入 `initial_magmoms` 数组。

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。PM 随机性受 seed + 结构稳定 ID 联合控制。
