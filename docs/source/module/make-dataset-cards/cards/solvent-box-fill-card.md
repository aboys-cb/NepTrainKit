<!-- card-schema: {"card_name": "Solvent Box Fill", "source_file": "src/NepTrainKit/ui/views/_card/solvent_box_fill_card.py", "serialized_keys": ["params"]} -->

# 周期溶剂盒（Solvent Box Fill）

`Group`: `Organic` | `Class`: `SolventBoxFillCard`

## 功能说明

`Solvent Box Fill` 在已有周期性晶胞的整个 cell 中随机插入溶剂分子，适合从空盒、干结构或稀疏溶剂结构生成周期溶剂候选初态。输入可以没有原子，但必须有有限且非奇异的 cell，并且至少一个周期方向开启。

它解决的是“训练集缺少周期溶剂环境”的覆盖问题，不替代分子动力学平衡或量化优化。各模式下分子位置和整体取向都采用随机采样；`water`、`loose`、`dense` 主要改变默认碰撞半径缩放，并不会生成局部离子水合取向。

## 操作示例

### 场景：模型只见过真空表面，没有见过周期溶剂盒

你要训练一个界面或溶液体系的 NEP，但已有训练集主要是干表面和单分子吸附构型。先准备带 cell 的周期输入结构，用 `Solvent Box Fill` 插入固定数量或按密度估算的水分子，生成一批整盒初态。

参数设置：`count_mode="fixed"`，`solvent_count=200`，`sampling_mode="auto"`，`min_distance=0.8`，`max_attempts_per_solvent=500`，打开 `use_seed`。

检查输出时重点看是否插满、是否存在明显短接触、溶剂是否都包回 cell 内，以及后续预松弛是否大面积崩坏。

## 参数说明

### 溶剂输入

#### Solvent XYZ（solvent_xyz）

`str`，默认是一个三原子水分子 XYZ。这里保存溶剂分子文本，而不是文件路径。

#### Independent outputs per input（structures）

`int`，默认 1。每个输入结构生成多少个独立填充版本。

### 数量控制

#### Target amount（count_mode）

`str`，默认 `fixed`。可选 `fixed`、`density`。`fixed` 直接使用目标分子数；`density` 根据**完整 cell 体积**、溶剂分子质量、目标密度和数量乘数估算一个名义分子数。

#### Target solvent molecules（solvent_count）

`int`，默认 100。`count_mode="fixed"` 时生效。

#### Density（density）

`float`，默认 1.0，单位 g/cm³。`count_mode="density"` 时用于估算溶剂分子数量。

#### Density count multiplier（fill_packing）

`float`，默认 1.0，范围 `(0, 1]`。`count_mode="density"` 时乘到名义纯溶剂分子数上；小于 1 会降低目标数量。大于 1 没有清楚的“填充比例”含义，因此程序不再静默截断，而是直接报错。

:::{warning}
密度模式不会扣除宿主原子、已有溶剂或真空层占据的体积，也不会根据溶质质量反算最终溶液密度。对含表面、溶质或大块固体的 cell，它只是按完整 cell 体积给出的初始数量估计；请看卡片预览中的解析数量，再用 `fill_packing` 或固定数量修正。
:::

### 采样和几何约束

#### Collision profile（sampling_mode）

`str`，默认 `auto`。可选 `auto`、`general`、`water`、`loose`、`dense`。`auto` 只根据溶剂分子是否恰好为 H₂O 解析成 `water` 或 `general`；整盒填充没有局部中心，因此不会进入 `ion-water`。这些配置主要控制 `min_distance=0` 时的默认元素半径缩放，分子整体取向始终是随机的。

#### Uniform minimum-distance override（min_distance）

`float`，默认 0。大于 0 时，所有试放溶剂原子与已有原子之间使用同一个最小距离，并忽略 `collision_scale`；等于 0 时使用元素碰撞半径。

#### Element-radius collision scale（collision_scale）

`float`，默认 0。等于 0 时使用碰撞配置的内置缩放；大于 0 时覆盖配置默认值。仅在 `min_distance=0` 时生效。

#### Max Attempts Per Solvent（max_attempts_per_solvent）

`int`，默认 500。总尝试上限是“该值 × 目标分子数”；若连续拒绝次数达到内部停滞阈值，也会提前停止。高密度盒子触顶时，优先检查目标数量和碰撞规则，再考虑增加它。

#### Strict Count（strict_count）

`bool`，默认 true。打开后，未插满目标数量就失败；关闭后允许输出至少放入 1 个分子的部分结果。若一个分子都放不进去，仍会报错，避免把未变化的输入标成“已填盒”。

### 柔性溶剂

#### Flex Solvent（flex_solvent）

`bool`，默认 false。打开后复用“有机构象采样”的几何启发式拓扑生成溶剂构象池，再用于填盒。水没有可旋转单键，因此普通水盒不需要打开；打开后主要只增加高斯坐标噪声。

#### Flex Pool（flex_pool）

`int`，默认 32。柔性溶剂构象池大小。

#### Flex Torsion Range（flex_torsion_range）

`tuple[float, float]`，默认 `(-180.0, 180.0)`，单位 degree。柔性构象生成时附加的扭转角增量范围。

#### Flex Max Torsions（flex_max_torsions）

`int`，默认 5。每个柔性构象最多扰动多少个可旋转键。

#### Flex Gaussian Sigma（flex_gaussian_sigma）

`float`，默认 0.03，单位 Å。柔性构象生成时叠加的坐标噪声。

### 随机性

#### Use Seed（use_seed）

`bool`，默认 false。打开后，同一输入结构、参数和 seed 会生成相同输出。

#### Seed（seed）

`int`，默认 0。`use_seed=True` 时生效。

## 推荐预设

### 固定数量水盒（200 个水，固定 seed）

```json
{
  "class": "SolventBoxFillCard",
  "check_state": true,
  "params": {
    "structures": 1,
    "count_mode": "fixed",
    "solvent_count": 200,
    "sampling_mode": "auto",
    "min_distance": 0.8,
    "max_attempts_per_solvent": 500,
    "strict_count": true,
    "use_seed": true,
    "seed": 42
  }
}
```

### 按密度填充水盒（允许部分输出）

```json
{
  "class": "SolventBoxFillCard",
  "check_state": true,
  "params": {
    "structures": 3,
    "count_mode": "density",
    "density": 1.0,
    "fill_packing": 0.7,
    "sampling_mode": "water",
    "min_distance": 0.85,
    "max_attempts_per_solvent": 800,
    "strict_count": false,
    "use_seed": true,
    "seed": 7
  }
}
```

## 推荐组合

- `Solvent Box Fill → Geometry Filter`：先剔除短接触和明显异常盒子。
- `Solvent Box Fill → FPS Filter`：整盒初态很多时，用代表性采样降低 DFT 数量。

## 常见问题

- 输入没有有效周期 cell：这张卡需要非奇异 cell 和至少一个周期方向；非周期局部溶剂环境请用 `Local Solvation`。
- 无法插满目标数量：降低固定数量或密度数量乘数，减小统一最小距离或元素半径缩放，增大 cell；也可以关闭严格数量先保留非零的部分结果。
- 按密度估算的数量不符合预期：数量由完整 cell 体积、溶剂分子质量、目标密度和数量乘数共同决定，不会扣除宿主占据体积。先检查 cell 单位、真空层、已有原子和 `solvent_xyz`。
- 想生成离子第一水合壳：改用 `Local Solvation`。整盒卡不会按离子元素选择 ion–O 距离或水分子取向。

## 输出标签

`SolvBox(mode={mode},req={目标数},ok={实际放入数})`

## 可复现性

打开 `use_seed` 后，相同输入结构、结构顺序、参数和 `seed` 会生成相同填盒结果。`count_mode="density"` 时，cell 体积或 `solvent_xyz` 改变会改变目标分子数，因此输出也会改变。
