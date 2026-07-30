<!-- card-schema: {"card_name": "Local Solvation", "source_file": "src/NepTrainKit/ui/views/_card/local_solvation_card.py", "serialized_keys": ["params"]} -->

# 局部溶剂壳（Local Solvation）

**分类：** 分子与溶剂

## 功能说明

`Local Solvation` 在选中原子周围插入一批溶剂分子，用来补充离子水合壳、局部极性环境、溶质附近初始溶剂排布等候选构型。它只生成几何初态，不做力场或 DFT 优化，也不保证得到平衡溶剂化数或平衡取向。

这张卡保留输入结构，并追加溶剂原子；碰撞检查只负责排除明显短接触。`Auto` 会根据溶剂是否为一水分子以及中心元素是否包含内置离子来解析实际放置模式。卡片首帧预览会显示中心数、溶剂分子式、解析后的模式和真正采用的碰撞规则。

## 原理与公式

程序先按中心选择规则得到中心原子集合 $\mathcal C$，再在球壳
$r_{\min}\le r\le r_{\max}$ 中抽样溶剂质心。若按体积均匀采样，半径可写为

$$
r=\left[r_{\min}^3+u\left(r_{\max}^3-r_{\min}^3\right)\right]^{1/3},
\qquad u\sim U(0,1).
$$

每个候选分子还会随机取向，并对宿主、已放置溶剂及周期镜像执行碰撞检查。`自动/水/疏松/
致密`模式选择不同的预设碰撞尺度和采样策略；用户显式填写碰撞尺度时覆盖预设。柔性构象
选项会先生成有限构象池再参与放置，不是在放置过程中做分子动力学。达到尝试上限仍无合法
位置时会失败，不会把重叠结构当作成功结果。

## 操作示例

### 场景：模型没有见过离子第一水合壳

你训练了含 Ca 的体系，但训练集里 Ca 周围几乎都是干结构。模型在含水环境里预测 Ca-O 局部相互作用时异常。先用 `Local Solvation` 选中 Ca 原子，插入 4-8 个水分子，生成候选水合壳，再经过几何检查和 DFT 松弛。

参数设置：中心选择“按元素”并填 `Ca`，放置模式选“自动识别溶剂 / 离子”，每个输出放 6 个水分子，统一最小距离可先设为 `0.8 Å`，打开固定种子。

在 `ion-water` 模式下，Ca 第一配位壳的 O 原子距离来自内置 `2.30–2.60 Å` 区间；界面上的“备用质心壳层”只在该离子的内置配位数填满之后使用。不要把备用壳层误认为第一壳 Ca–O 距离。

检查输出时重点看 Ca-O 距离、短 H-H/O-H 非键接触、以及 DFT 松弛后是否仍保持合理水合结构。

## 参数说明

### 溶剂输入

#### 溶剂 XYZ（solvent_xyz）

`str`，默认是一个三原子水分子 XYZ。这里保存的是溶剂分子文本，而不是文件路径，因此 card JSON 可以脱离原始文件复现。

#### 每个输入的独立输出数（structures）

`int`，默认 1。每个输入结构生成多少个独立溶剂化版本。

#### 每个输出的溶剂分子数（solvent_count）

`int`，默认 6。每个输出结构插入多少个溶剂分子。离子第一壳通常从 4–8 开始；大分子局部溶剂环境再按中心原子数量扩大。

#### 放置配置（sampling_mode）

`str`，默认 `auto`。可选项：

- `auto`：水 + 内置离子中心时解析为 `ion-water`；只有水时解析为 `water`；其他溶剂解析为 `general`。
- `general`：在备用质心壳层内随机放置和取向。
- `water`：在备用质心壳层内放置水，并让 O→H 中点的方向大致背离局部中心。
- `ion-water`：内置离子的前若干个水使用元素专属 ion–O 区间并让 H 侧背离离子；达到内置配位数后回到备用质心壳层。
- `loose` / `dense`：主要改变默认碰撞半径缩放；它们不会自动改写界面中的备用壳层数值。

### 中心原子选择

#### 溶剂化中心（center_mode）

`str`，默认 `all`。可选 `all`、`elements`、`indices`、`Z 轴范围`（`z_range`）。这决定哪些原子作为局部溶剂化中心。

#### 中心元素（center_elements）

`str`，默认空。`center_mode="elements"` 时生效，例如 `Ca,Na,O`。

#### 中心索引（center_indices）

`str`，默认空。`center_mode="indices"` 时生效，使用 1-based 索引和范围，例如 `1,3,5-8`。

#### Z 轴范围（z_range）

`tuple[float, float]`，默认 `(0.0, 0.0)`。`center_mode="z_range"` 时按笛卡尔 z 坐标选择中心原子。

### 几何约束

#### 备用中心-质心壳层（shell）

`tuple[float, float]`，默认 `(2.2, 4.5)`，单位 Å。普通放置时是局部中心到溶剂质心的采样范围；`ion-water` 的第一配位壳不使用它，而是使用元素专属 ion–O 区间。外半径必须大于内半径。

#### 统一最小距离覆盖值（min_distance）

`float`，默认 0。大于 0 时，所有试放溶剂原子与已有原子之间都使用同一个最小距离，且会忽略 `元素半径碰撞缩放`（`collision_scale`）；等于 0 时才使用元素碰撞半径和 `元素半径碰撞缩放`（`collision_scale`）。

#### 元素半径碰撞缩放（collision_scale）

`float`，默认 0。等于 0 时使用解析后放置模式的内置半径缩放；大于 0 时覆盖模式默认值。仅在 `min_distance=0` 时生效。

#### 最大尝试次数（max_attempts）

`int`，默认 3000。每个输出结构全部溶剂分子共享的总尝试次数，不是“每个分子 3000 次”。

#### 严格数量（strict_count）

`bool`，默认 true。打开后，未插满 `每个输出的溶剂分子数`（`solvent_count`）就失败；关闭后允许输出至少放入 1 个分子的部分结果。若一个分子都放不进去，即使关闭严格数量也会报错，避免输出带标签但实际未溶剂化的原结构。

### 非周期输出

#### 自动设置晶胞（auto_box）

`bool`，默认 false。仅当输入没有有效 cell 时生效，打开后按输出坐标生成非周期显示盒。输入已有有效 cell（包括周期晶胞）时，卡片始终保留原 cell 和 PBC，不会因为这个选项把周期体系改成非周期体系。

#### 固定盒尺寸（box_size）

`float`，默认 100.0，单位 Å。输入没有有效 cell 且 `auto_box=False` 时，输出会居中放入这个固定的非周期显示盒。

#### 晶胞留白（box_padding）

`float`，默认 8.0，单位 Å。`auto_box=True` 时在坐标包围盒外增加的边距。

#### 最小晶胞尺寸（min_box）

`float`，默认 0.0，单位 Å。`auto_box=True` 时每条盒边的最小长度。

### 柔性溶剂

#### 启用柔性溶剂（flex_solvent）

`bool`，默认 false。打开后复用“有机构象采样”的几何启发式拓扑，先生成溶剂构象池，再用于插入。刚性水分子没有可旋转单键，因此打开后主要只会得到高斯坐标噪声；普通水无需打开。

#### 柔性构象池（flex_pool）

`int`，默认 32。柔性溶剂构象池大小。

#### 柔性扭转角范围（flex_torsion_range）

`tuple[float, float]`，默认 `(-180.0, 180.0)`，单位 degree。柔性构象生成时附加的扭转角增量范围。

#### 柔性采样最大扭转键数（flex_max_torsions）

`int`，默认 5。每个柔性构象最多扰动多少个可旋转键。

#### 柔性采样高斯宽度（flex_gaussian_sigma）

`float`，默认 0.03，单位 Å。柔性构象生成时叠加的坐标噪声。

### 随机性

#### 使用随机种子（use_seed）

`bool`，默认 false。打开后，同一输入结构、参数和 seed 会生成相同输出。

#### 随机种子（seed）

`int`，默认 0。`use_seed=True` 时生效。

## 推荐预设

### Ca 第一水合壳（6 个水，固定 seed）

```json
{
  "class": "LocalSolvationCard",
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
  "class": "LocalSolvationCard",
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

- 没有选到中心原子：检查 `溶剂化中心`（`center_mode`）是否和 `中心元素`（`center_elements`）、`中心索引`（`center_indices`）或 `Z 轴范围`（`z_range`）匹配。
- 一直插不满目标数量：降低 `每个输出的溶剂分子数`（`solvent_count`），放宽备用壳层，减小统一最小距离或碰撞缩放；如果目标是周期整盒溶剂，优先改用 `Solvent Box Fill`。
- 调了备用壳层但 Ca–O 距离没变化：`ion-water` 第一配位壳使用内置 ion–O 区间；卡片预览会列出实际范围。
- 输出有明显短接触：增大 `统一最小距离覆盖值`（`min_distance`）或 `元素半径碰撞缩放`（`collision_scale`），再接 `Geometry Filter` 和 DFT/MD 预松弛。

## 输出标签

`SolvLocal(mode={mode},n={placed},sel={center_count})`

## 可复现性

打开 `使用随机种子`（`use_seed`）后，相同输入结构、结构顺序、参数和 `随机种子`（`seed`）会生成相同插入结果。若输入结构顺序、中心原子集合或溶剂分子文本发生变化，输出也会随之变化。
