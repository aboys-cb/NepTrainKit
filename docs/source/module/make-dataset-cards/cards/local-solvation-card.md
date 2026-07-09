<!-- card-schema: {"card_name": "Local Solvation", "source_file": "src/NepTrainKit/ui/views/_card/local_solvation_card.py", "serialized_keys": ["params"]} -->

# Local Solvation

`Group`: `Organic` | `Class`: `LocalSolvationCard`

## 功能说明

`Local Solvation` 在选中原子周围插入一批溶剂分子，用来补充离子水合壳、局部极性环境、溶质附近初始溶剂排布等候选构型。它只生成几何初态，不做力场或 DFT 优化。

这张卡保留输入结构，并追加溶剂原子；输出会写入 `SolvLocal(...)` 标签，便于后续在 `NEP Dataset Display` 中筛选。

## 操作示例

### 场景：模型没有见过离子第一水合壳

你训练了含 Ca 的体系，但训练集里 Ca 周围几乎都是干结构。模型在含水环境里预测 Ca-O 局部相互作用时异常。先用 `Local Solvation` 选中 Ca 原子，插入 4-8 个水分子，生成候选水合壳，再经过几何检查和 DFT 松弛。

参数设置：`center_mode="elements"`，`center_elements="Ca"`，`sampling_mode="auto"`，`solvent_count=6`，`shell=(2.6, 3.4)`，`min_distance=0.8`，打开 `use_seed`。

检查输出时重点看 Ca-O 距离、短 H-H/O-H 非键接触、以及 DFT 松弛后是否仍保持合理水合结构。

## 参数说明

### 溶剂输入

#### Solvent XYZ（solvent_xyz）

`str`，默认是一个三原子水分子 XYZ。这里保存的是溶剂分子文本，而不是文件路径，因此 card JSON 可以脱离原始文件复现。

#### Structures（structures）

`int`，默认 1。每个输入结构生成多少个独立溶剂化版本。

#### Solvent Count（solvent_count）

`int`，默认 30。每个输出结构插入多少个溶剂分子。离子第一壳通常从 4-8 开始；大分子局部溶剂环境再按中心原子数量扩大。

#### Sampling Mode（sampling_mode）

`str`，默认 `auto`。可选 `auto`、`general`、`water`、`ion-water`、`loose`、`dense`。`auto` 会在水分子和离子中心同时出现时使用 `ion-water`，只有水分子时使用 `water`，否则使用 `general`。

### 中心原子选择

#### Center Mode（center_mode）

`str`，默认 `all`。可选 `all`、`elements`、`indices`、`z_range`。这决定哪些原子作为局部溶剂化中心。

#### Center Elements（center_elements）

`str`，默认空。`center_mode="elements"` 时生效，例如 `Ca,Na,O`。

#### Center Indices（center_indices）

`str`，默认空。`center_mode="indices"` 时生效，使用 1-based 索引和范围，例如 `1,3,5-8`。

#### Z Range（z_range）

`tuple[float, float]`，默认 `(0.0, 0.0)`。`center_mode="z_range"` 时按笛卡尔 z 坐标选择中心原子。

### 几何约束

#### Shell（shell）

`tuple[float, float]`，默认 `(2.2, 4.5)`，单位 Å。局部溶剂中心到溶剂分子参考位置的采样壳层。外半径必须大于内半径。

#### Min Distance（min_distance）

`float`，默认 0。大于 0 时作为全局原子-原子最小距离；等于 0 时使用元素碰撞半径和 `collision_scale`。

#### Collision Scale（collision_scale）

`float`，默认 0。等于 0 时使用 `sampling_mode` 的内置半径缩放；大于 0 时覆盖模式默认值。它控制插入溶剂与已有原子之间的非键接触下限。

#### Max Attempts（max_attempts）

`int`，默认 3000。每个输出结构的最大插入尝试次数。达到上限仍插不满时，`strict_count=True` 会报错。

#### Strict Count（strict_count）

`bool`，默认 true。打开后，未插满 `solvent_count` 就失败；关闭后允许输出部分插入结果。

### 非周期输出

#### Auto Box（auto_box）

`bool`，默认 false。打开后按输出坐标自动生成非周期盒子。

#### Fixed Box Size（box_size）

`float`，默认 100.0，单位 Å。输入没有有效 cell 且 `auto_box=False` 时，输出会居中放入这个固定盒子；这对应原脚本的 `-box` 默认行为。

#### Box Padding（box_padding）

`float`，默认 8.0，单位 Å。`auto_box=True` 时在坐标包围盒外增加的边距。

#### Min Box（min_box）

`float`，默认 0.0，单位 Å。`auto_box=True` 时每条盒边的最小长度。

### 柔性溶剂

#### Flex Solvent（flex_solvent）

`bool`，默认 false。打开后复用仓库已有 torsion-guard core 生成溶剂构象池，再用于插入。

#### Flex Pool（flex_pool）

`int`，默认 32。柔性溶剂构象池大小。

#### Flex Torsion Range（flex_torsion_range）

`tuple[float, float]`，默认 `(-180.0, 180.0)`，单位 degree。柔性构象生成时的扭转角范围。

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

### Ca 第一水合壳（6 个水，固定 seed）

```json
{
  "class": "Local Solvation",
  "check_state": true,
  "params": {
    "structures": 1,
    "solvent_count": 6,
    "sampling_mode": "auto",
    "center_mode": "elements",
    "center_elements": "Ca",
    "shell": [2.6, 3.4],
    "min_distance": 0.8,
    "strict_count": true,
    "use_seed": true,
    "seed": 42
  }
}
```

### 表面含水局部环境（按 z 选中心，允许部分输出）

```json
{
  "class": "Local Solvation",
  "check_state": true,
  "params": {
    "structures": 3,
    "solvent_count": 12,
    "sampling_mode": "water",
    "center_mode": "z_range",
    "z_range": [8.0, 14.0],
    "shell": [2.4, 5.0],
    "min_distance": 0.9,
    "strict_count": false,
    "auto_box": false,
    "use_seed": true,
    "seed": 7
  }
}
```

## 推荐组合

- `Local Solvation → Geometry Filter`：先生成局部溶剂环境，再剔除短接触或异常体积结构。
- `Local Solvation → FPS Filter`：局部溶剂化批量生成后抽代表结构送 DFT。

## 常见问题

- 没有选到中心原子：检查 `center_mode` 是否和 `center_elements`、`center_indices` 或 `z_range` 匹配。
- 一直插不满目标数量：降低 `solvent_count`，放宽 `shell`，减小 `min_distance` 或 `collision_scale`；如果目标是周期整盒溶剂，优先改用 `Solvent Box Fill`。
- 输出有明显短接触：增大 `min_distance` 或 `collision_scale`，再接 `Geometry Filter` 和 DFT/MD 预松弛。

## 输出标签

`SolvLocal(mode={mode},n={placed},sel={center_count})`

## 可复现性

打开 `use_seed` 后，相同输入结构、结构顺序、参数和 `seed` 会生成相同插入结果。若输入结构顺序、中心原子集合或溶剂分子文本发生变化，输出也会随之变化。
