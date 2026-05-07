<!-- card-schema: {"card_name": "Geometry Filter", "source_file": "src/NepTrainKit/ui/views/_card/geometry_filter_card.py", "serialized_keys": ["params"]} -->

# 几何过滤（Geometry Filter）

`Group`: `Filter` | `Class`: `GeometryFilterCard`

## 功能说明

`Geometry Filter` 按显式阈值过滤候选结构：最短原子间距、单原子体积、质量密度和有限晶胞。它不计算能量，不修复结构，也不替代 `FPS Filter`。它的职责是在代表性采样和 DFT 前挡掉几何上已经明显不可用的结构。

## 操作示例

### 场景：随机占位和强扰动后出现短键结构

合金候选池经过 `Random Occupancy -> Atomic Perturb` 后，少量结构出现 < 1.0 A 的原子重叠。它们进入 DFT 会浪费队列，进入 FPS 会污染 descriptor 空间。

**输入：** 已生成的候选结构池。
**目标：** 在 FPS 前删除短键和异常体积结构。
**参数设置：** `min_pair_distance=1.2`，`min_volume_per_atom=5.0`，`max_volume_per_atom=40.0`，密度阈值保持关闭。
**输出：** 只保留满足所有开启阈值的结构。
**怎么验证训练集质量改善：** 导入 `NEP Dataset Display` 后，最短键分布不再有低端离群点；FPS 选出的结构不应再包含明显重叠帧。

## 参数说明
### Min Pair Distance（min_pair_distance）
类型：`float`。默认：`1.0`。设置任意原子对允许的最小距离。

物理直觉：这是硬短键门槛。应低于目标体系正常最近邻键长，但高于明显原子重叠距离；例如金属/氧化物常从 1.0-1.5 A 起步。

### Min Volume Per Atom（min_volume_per_atom）
类型：`float`。默认：`0.0`。设置允许的最小单原子体积。

物理直觉：过滤过度压缩结构。阈值应来自同材料平衡体积的下界；多孔、slab 或含真空体系不要随便打开。

### Max Volume Per Atom（max_volume_per_atom）
类型：`float`。默认：`0.0`。设置允许的最大单原子体积。

物理直觉：过滤过度拉伸或异常大 cell。体相候选池可以设置；slab/分子体系有真空时，这个阈值容易误杀。

### Min Density（min_density）
类型：`float`。默认：`0.0`。设置允许的最小质量密度。

物理直觉：按质量密度过滤低密度异常结构。只适合体相或近体相候选；含真空、孔洞或表面模型应保持 0 关闭。

### Max Density（max_density）
类型：`float`。默认：`0.0`。设置允许的最大质量密度。

物理直觉：按质量密度过滤过度压缩结构。阈值应参考材料真实密度上界，不要跨元素体系套统一数值。

### Require Finite Cell（require_finite_cell）
类型：`bool`。默认：`False`。决定是否要求晶胞体积有限。

物理直觉：晶体、slab、界面候选打开；孤立分子或非周期构型关闭。打开后零体积 cell 会被直接删除。

## 推荐预设

### 短键门槛

```json
{
  "class": "GeometryFilterCard",
  "params": {
    "min_pair_distance": 1.2,
    "min_volume_per_atom": 0.0,
    "max_volume_per_atom": 0.0,
    "min_density": 0.0,
    "max_density": 0.0,
    "require_finite_cell": false
  }
}
```

用于随机扰动、插入缺陷或随机占位后的第一道硬检查。

### 晶体候选池清洗

```json
{
  "class": "GeometryFilterCard",
  "params": {
    "min_pair_distance": 1.1,
    "min_volume_per_atom": 5.0,
    "max_volume_per_atom": 60.0,
    "min_density": 0.0,
    "max_density": 0.0,
    "require_finite_cell": true
  }
}
```

用于晶体、缺陷和表面候选池，防止零体积或极端体积结构进入后续流程。

## 推荐组合

- `Atomic Perturb -> Geometry Filter -> FPS Filter`：先生成位移扰动，再删除短键，最后做代表性采样。
- `Random Occupancy -> Geometry Filter -> FPS Filter`：合金占位后先做几何门槛，再选代表结构去 DFT。
- `Random Slab -> Insert Defect -> Geometry Filter`：表面插入后检查吸附物和基底是否重叠。

## 常见问题

**输出为空。** 至少一个阈值过严。先看最短键和体积分布，确认阈值是否超出了当前体系的真实范围。

**密度阈值不生效。** `min_density=0.0` 和 `max_density=0.0` 都表示关闭密度检查。只有大于 0 的阈值参与判定。

**非周期分子被删掉。** 如果开启了体积、密度或 `require_finite_cell`，零体积分子会被删除。分子构象清洗通常只开 `min_pair_distance`。

## 输出标签

本卡是过滤卡，不修改保留下来的 `Config_type`。

## 可复现性

本卡没有随机性。同一输入和同一阈值会得到相同输出。
