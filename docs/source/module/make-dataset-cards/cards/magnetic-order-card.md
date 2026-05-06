<!-- card-schema: {"card_name": "Magnetic Order", "source_file": "src/NepTrainKit/ui/views/_card/magnetic_order_card.py", "serialized_keys": ["params"]} -->

# 磁有序初始化（Magnetic Order）

`Group`: `Magnetism` | `Class`: `MagneticOrderCard`

## 功能说明

从一个磁性母相结构出发，一次性生成 FM（铁磁）、AFM（反铁磁）、PM（顺磁）多种磁序分支。每个输出结构都写入对应的 `initial_magmoms` 数组和 `Config_type` 标签。

**关键限制：** 这张卡只写入初始磁矩，不保证生成的就是磁基态。磁矩是否收敛到合理值仍需要用 NEP/DFT 后续计算验证。

## 操作示例

### 场景：为 Fe 体系生成 FM + AFM + PM 训练集

**输入：** 一个含 Fe 的晶体结构，已从文献知道 Fe 磁矩约 2.2 μB

**目标：** 生成 1 个 FM、1 个 AFM（k-vector 111）、8 个 PM，覆盖不同磁序

**参数设置：**
- `Format` = `Collinear (scalar)` （不关心非共线的话）
- `Magmom Map` = `Fe:2.2`
- `Gen FM` = 勾选
- `Gen AFM` = 勾选，`AFM Mode` = `k-vector`，`AFM Kvec` = `111`
- `Gen PM` = 勾选，`PM Count` = `[8]`，`PM Direction` = `sphere`

**输出：** 1 个 FM + 1 个 AFM + 8 个 PM，全部带 `initial_magmoms`

**怎么验证结果合理：**
- FM：所有带磁矩的原子符号相同
- AFM k-vector 111：沿 111 方向相邻层磁矩翻转
- PM：随机正负，没有明显规律
- 所有输出磁矩模长约 2.2

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
