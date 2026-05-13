<!-- card-schema: {"card_name": "Solvent Box Fill", "source_file": "src/NepTrainKit/ui/views/_card/solvent_box_fill_card.py", "serialized_keys": ["params"]} -->

# Solvent Box Fill

`Group`: `Organic` | `Class`: `SolventBoxFillCard`

## 功能说明

`Solvent Box Fill` 在已有周期性晶胞中插入溶剂分子，适合从干结构或稀疏溶剂结构生成整盒溶剂候选初态。输入结构必须有非奇异 cell，并且至少一个周期方向开启。

它解决的是“训练集缺少周期溶剂环境”的覆盖问题，不替代分子动力学平衡或量化优化。输出仍需要做几何筛查和后续 DFT/MD 处理。

## 操作示例

### 场景：模型只见过真空表面，没有见过周期溶剂盒

你要训练一个界面或溶液体系的 NEP，但已有训练集主要是干表面和单分子吸附构型。先准备带 cell 的周期输入结构，用 `Solvent Box Fill` 插入固定数量或按密度估算的水分子，生成一批整盒初态。

参数设置：`count_mode="fixed"`，`solvent_count=200`，`sampling_mode="auto"`，`min_distance=0.8`，`max_attempts_per_solvent=500`，打开 `use_seed`。

检查输出时重点看是否插满、是否存在明显短接触、溶剂是否都包回 cell 内，以及后续预松弛是否大面积崩坏。

## 参数说明

### 溶剂输入

#### Solvent XYZ（solvent_xyz）

`str`，默认是一个三原子水分子 XYZ。这里保存溶剂分子文本，而不是文件路径。

#### Structures（structures）

`int`，默认 1。每个输入结构生成多少个独立填充版本。

### 数量控制

#### Count Mode（count_mode）

`str`，默认 `fixed`。可选 `fixed`、`density`。`fixed` 直接使用 `solvent_count`；`density` 根据 cell 体积、溶剂分子质量、`density` 和 `fill_packing` 估算数量。

#### Solvent Count（solvent_count）

`int`，默认 100。`count_mode="fixed"` 时生效。

#### Density（density）

`float`，默认 1.0，单位 g/cm3。`count_mode="density"` 时用于估算溶剂分子数量。

#### Fill Packing（fill_packing）

`float`，默认 1.0。`count_mode="density"` 时作为密度缩放因子；小于 1 会降低目标分子数。

### 采样和几何约束

#### Sampling Mode（sampling_mode）

`str`，默认 `auto`。可选 `auto`、`general`、`water`、`loose`、`dense`。整盒填充不会因为盒子里有离子就自动切到 `ion-water`，因为它没有局部中心原子。

#### Min Distance（min_distance）

`float`，默认 0。大于 0 时作为全局原子-原子最小距离；等于 0 时使用元素碰撞半径和 `collision_scale`。

#### Collision Scale（collision_scale）

`float`，默认 0。等于 0 时使用 `sampling_mode` 的内置半径缩放；大于 0 时覆盖模式默认值。

#### Max Attempts Per Solvent（max_attempts_per_solvent）

`int`，默认 500。每个目标溶剂分子的最大随机放置尝试次数。高密度盒子触顶时，优先检查目标数量和 `min_distance`，再考虑增加它。

#### Strict Count（strict_count）

`bool`，默认 true。打开后，未插满目标数量就失败；关闭后允许输出部分填充结果。

### 柔性溶剂

#### Flex Solvent（flex_solvent）

`bool`，默认 false。打开后复用仓库已有 torsion-guard core 生成溶剂构象池，再用于填盒。

#### Flex Pool（flex_pool）

`int`，默认 32。柔性溶剂构象池大小。

#### Flex Torsion Range（flex_torsion_range）

`tuple[float, float]`，默认 `(-180.0, 180.0)`，单位 degree。柔性构象生成时的扭转角范围。

#### Flex Max Torsions（flex_max_torsions）

`int`，默认 5。每个柔性构象最多扰动多少个可旋转键。

#### Flex Gaussian Sigma（flex_gaussian_sigma）

`float`，默认 0.03，单位 A。柔性构象生成时叠加的坐标噪声。

### 随机性

#### Use Seed（use_seed）

`bool`，默认 false。打开后，同一输入结构、参数和 seed 会生成相同输出。

#### Seed（seed）

`int`，默认 0。`use_seed=True` 时生效。

## 推荐组合

- `Solvent Box Fill -> Geometry Filter`：先剔除短接触和明显异常盒子。
- `Solvent Box Fill -> FPS Filter`：整盒初态很多时，用代表性采样降低 DFT 数量。

## 输出标签

`SolvBox(mode={mode},n={placed})`
