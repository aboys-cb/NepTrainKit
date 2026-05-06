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

### 磁矩幅值来源（至少填一个）

**`Magmom Map`**（magmom_map）：字符串。每个元素的磁矩大小。格式 `Fe:2.2,Ni:0.6`。最精确的控制方式。

**`Default Moment`**（default_moment）：浮点数。`magmom_map` 中未列出的元素用此默认值。设为 0 则不给未配置元素写磁矩。

**`Apply Elements`**（apply_elements）：逗号分隔，如 `Fe,Co`。只对列出的元素写磁矩，其余写 0。留空 = 全部生效。

> 优先级：`magmom_map` 命中 → `default_moment` 兜底 → 未命中 `apply_elements` 的写 0。

### 磁矩格式与方向

**`Format`**（format）：`Collinear (scalar)` 或 `Non-collinear (vector)`。
- Collinear：磁矩只有大小和 ± 号，沿 `Axis` 方向。适合大多数共线磁序。
- Non-collinear：3D 矢量磁矩。适合螺旋、canting 等非共线体系。

**`Axis`**（axis）：`[x, y, z]`，collinear 模式下磁矩的参考轴方向。默认 `[0, 0, 1]`（z 轴）。

**`Use Element Dirs`**（use_element_dirs）：勾选后，非共线模式下不同元素可以使用 `magmom_map` 中各自的方向先验，而不是全部沿 `Axis`。适合多元素磁性体系（如 Fe↑ Mn↓）。仅在非共线格式下有意义。

### FM 分支

**`Gen FM`**（gen_fm）：勾选 → 输出一个铁磁结构（所有磁矩同向）。通常建议保持开启，FM 是重要的磁序参考态。

### AFM 分支

**`Gen AFM`**（gen_afm）：勾选 → 输出反铁磁结构。

**`AFM Mode`**（afm_mode）：
- `k-vector`：用波矢决定正负翻转。适合周期性子晶格。需配置 `AFM Kvec`。
- `group A/B`：用 `atoms.arrays['group']` 中的标签决定正负。需上游先跑 `Group Label`，需配置 `AFM Group A/B`。

**`AFM Kvec`**（afm_kvec）：仅 k-vector 模式。`100` / `010` / `001` / `110` / `111`。

**`AFM Group A / B`**（afm_group_a / afm_group_b）：仅 group A/B 模式。指定哪个 group 取正、哪个取负。

**`AFM Zero Unknown`**（afm_zero_unknown）：仅 group A/B 模式。勾选 → 不属 A 也不属 B 的原子磁矩置零；不勾选 → 默认正号。

### PM 分支

**`Gen PM`**（gen_pm）：勾选 → 生成顺磁结构。默认关闭，因为 PM 会显著增大输出规模。

**`PM Count`**（pm_count）：PM 样本数，建议 5-20。

**`PM Direction`**（pm_direction）：
- `sphere`：方向在球面上均匀采样
- `cone`：在以 `Axis` 为中心的锥面内采样，锥角由 `PM Cone Angle` 控制

**`PM Cone Angle`**（pm_cone_angle）：仅 cone 模式，单位度。30° 表示偏离 Axis 最多 30°。

**`PM Balanced`**（pm_balanced）：勾选 → 正负方向数量尽量平衡。建议保持。

### 随机性

**`Use Seed`**（use_seed）：勾选 → 固定种子可复现。

**`Seed`**（seed）：种子值。仅 `use_seed` 勾选时生效。不同值产生不同 PM 随机分布。

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
