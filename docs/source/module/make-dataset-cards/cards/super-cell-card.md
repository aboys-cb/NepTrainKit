<!-- card-schema: {"card_name": "Super Cell", "source_file": "src/NepTrainKit/ui/views/_card/super_cell_card.py", "serialized_keys": ["params"]} -->

# 超胞生成（Super Cell）

`Group`: `Lattice` | `Class`: `SuperCellCard`

## 功能说明

按倍率、目标胞长或原子数上限扩胞，为缺陷/表面/磁性操作提供足够空间。三种扩胞策略三选一：固定倍率（`scale`）、目标胞长（`cell`）、原子数上限（`max_atoms`）。可锁定特定轴向不扩胞——适合 slab 等需要维持法向层间距的场景。

$$\mathbf{T}=\mathrm{diag}(n_a,n_b,n_c),\quad \mathbf{C}'=\mathbf{C}\mathbf{T},\quad N'=N\cdot n_a n_b n_c$$

## 操作示例

### 场景：模型在小胞上训练后跑空位计算，周期镜像干扰导致空位形成能偏高 0.5 eV

你在 bcc Fe 上用 2x2x2 超胞（16 原子）训练了一个 NEP 模型。拿这个模型算单空位形成能，结果比 DFT 高了 0.5 eV。诊断发现：2x2x2 太小的胞里，空位和它的周期镜像之间距离只有约 5A，空位-空位镜像相互作用不可忽略，模型学到的实际上是"带镜像相互作用的空位"而非孤立空位。

**诊断思路：** 缺陷计算的黄金法则是"超胞要大到使缺陷-缺陷镜像相互作用可忽略"。2x2x2 = 16 原子的胞对空位来说太小。解法是在训练集里用更大的超胞（至少 3x3x3 = 54 原子或 4x4x4 = 128 原子）做空位计算，这样模型至少见过大胞里的空位环境。同时，下游做空位操作时也需要大胞作为母结构。

**输入：** 一个 bcc Fe 原胞（2 原子）

**目标：** 扩到 4x4x4 超胞（128 原子），为后续空位缺陷生成提供母结构

**参数设置：**
- mode = `scale`
- `super_scale` = `[4, 4, 4]`

**输出：** 1 个 4x4x4 超胞结构，128 原子

**怎么验证结果合理：**
- 检查原子数：原胞 2 原子 * 4 * 4 * 4 = 128，核对输出
- 检查晶格矢量：每个方向拉长 4 倍，体积增大 64 倍
- 如果空位形成能仍偏高，继续增大到 5x5x5 = 250 原子，直到 DFT 和 NEP 的形成能收敛
- 如果超胞太大计算不起，用 `max_atoms` 模式设置预算上限

### 什么时候加这张卡、什么时候不加

**加：**
- 下游要做缺陷（空位、间隙）、表面（slab）、磁性操作，需要超胞作为母结构
- 当前训练集胞太小，周期镜像干扰不可忽略
- 需要扩胞以使 defect-defect 距离 > 10A

**不加：**
- 下游任务是体相性质（弹性常数、声子），原胞/小胞就够 —— 扩胞只会白增计算量
- 计算预算严重受限，扩胞后 DFT 跑不动 —— 这种情况下维持小胞，接受周期镜像有残留误差
- 需要的是带应变的晶格变化而非单纯的重复复制 —— 用 `Lattice Strain` 或 `Cell Scaling`

## 参数说明

### 扩胞策略（三选一）

通过 `mode` 字段选择扩胞策略。每种模式下的实际行为受 `behavior_type` 控制（见下文）。

**`mode`**（string，默认 `"scale"`）：扩胞主模式。
- `"scale"`：按 `super_scale` 指定的固定倍数扩胞。适合你明确知道需要多大超胞。
- `"cell"`：按 `target_cell` 指定的目标胞长（单位 A）自动计算倍数。每个方向的倍数 = floor(target_length / original_length) 或 ceil(target_length / original_length)，取决于 `behavior_type`。适合你想让胞长达到特定数值（如所有方向 ≥ 20A）。
- `"max_atoms"`：按 `max_atoms` 指定的原子数上限枚举所有可能的倍数组合。输出所有总原子数不超过此上限的超胞。适合预算受限时找最大可用超胞。

**`behavior_type`**（int，默认 0）：输出行为。
- `0`（单输出）：scale 模式下输出一个超胞；cell 模式下输出最大/最小倍数的超胞；max_atoms 模式下输出原子数最大的那个超胞。
- `1`（枚举）：scale 和 cell 模式下枚举从 1x1x1 到目标倍数的所有组合；max_atoms 模式下枚举所有在上限内的组合。适合想一次生成多个不同大小的超胞。
- `2`（最小满足）：cell 模式下使用 ceil 取整（至少达到目标长度）；max_atoms 模式下使用 ceil 取整。适合"至少多大才够"的场景。

### 各模式的数值参数

**`super_scale`**（tuple[int,int,int]，默认 `[3,3,3]`）：仅 scale 模式。a/b/c 方向各复制几倍。最小 1（不复制），典型值 2~4。3x3x3 = 27 倍原子数。

**`target_cell`**（tuple[float,float,float]，默认 `[20,20,20]` A）：仅 cell 模式。各方向目标胞长。如果原胞 a 长度 = 5A，target=20，倍数 = 4（或 5，取决于 behavior_type）。

**`max_atoms`**（int，默认 100）：仅 max_atoms 模式。超胞总原子数上限。实际输出可能包含多个不超过此上限的超胞。

### 轴向锁定

**`fixed_axis_flags`**（tuple[bool,bool,bool]，默认 `[false,false,false]`）：锁定 a/b/c 方向的扩胞倍数。适合 slab 场景：锁住法向（通常 c 轴）不让扩胞，只扩面内方向。被锁定的轴使用 `fixed_axis_scale` 中对应的固定倍数。

**`fixed_axis_scale`**（tuple[int,int,int]，默认 `[1,1,1]`）：被锁定方向的固定扩胞倍数。仅对 `fixed_axis_flags=true` 的方向生效。通常设为 1（不扩胞）。

## 推荐预设

### 快速扩胞（固定 3x3x3，单输出）
```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "mode": "scale",
  "behavior_type": 0,
  "super_scale": [3, 3, 3],
  "target_cell": [20, 20, 20],
  "max_atoms": 100,
  "fixed_axis_flags": [false, false, false],
  "fixed_axis_scale": [1, 1, 1]
}
```

### Slab 扩胞（面内扩到 20A，法向锁 1x，枚举所有组合）
```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "mode": "cell",
  "behavior_type": 1,
  "super_scale": [3, 3, 3],
  "target_cell": [20, 20, 20],
  "max_atoms": 100,
  "fixed_axis_flags": [false, false, true],
  "fixed_axis_scale": [1, 1, 1]
}
```

### 预算限制下的最大超胞（≤ 300 原子，取最大）
```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "mode": "max_atoms",
  "behavior_type": 0,
  "super_scale": [3, 3, 3],
  "target_cell": [20, 20, 20],
  "max_atoms": 300,
  "fixed_axis_flags": [false, false, false],
  "fixed_axis_scale": [1, 1, 1]
}
```

## 推荐组合

- `Super Cell` -> `Vacancy Defect`：先扩胞再删原子，保证缺陷-镜像距离
- `Super Cell` -> `Lattice Strain`：扩胞后做应变扫描
- `Super Cell` -> `Surface Warp`：扩胞后做表面起伏
- `Super Cell` -> `FPS Filter`：枚举模式产生大量超胞时，用 FPS 去重（相同大小的超胞只有 1 个有用）

## 常见问题

**输出只有 1x1x1（等于没扩胞）。** 如果原胞自身已经满足目标——例如原胞 3A、target_cell=20、behavior_type=0 时倍数=6，但原胞 a 长度已经 25A —— 倍数被截断为 1。检查 target_cell 是否设得比原胞还小。或者在 max_atoms 模式下，原胞原子数已超上限，只输出原胞。

**Slab 面内方向扩胞不均匀。** `fixed_axis_flags` 只锁了法向，面内两个方向各走各的倍数。如果面内需要正方形胞，确保 target_cell 的 a/b 目标值一致，或者用 scale 模式手动统一倍数。

**原子数暴增超出计算预算。** 4x4x4 = 64 倍原子数。先心算：倍数 = (na*nb*nc)，原子数 = 原胞原子数 * 倍数。谨慎选择 behavior_type=1（枚举模式），因为从 1x1x1 到目标倍数会产生多个不同大小的超胞。

**枚举模式产生过多超胞。** behavior_type=1 + max_atoms=500 + 原胞=2 原子 → 所有 ≤250 倍原子数的组合都会被输出。如果不需要所有中间大小，改用 behavior_type=0 只取最大。

## 输出标签

`SC({na}x{nb}x{nc})` —— 如 `SC(4x4x4)`。

## 可复现性

无随机性。同参数同输入 → 严格一致输出。
