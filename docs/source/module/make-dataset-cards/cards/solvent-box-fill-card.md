<!-- card-schema: {"card_name": "Solvent Box Fill", "source_file": "src/NepTrainKit/ui/views/_card/solvent_box_fill_card.py", "serialized_keys": ["params"]} -->

# 周期溶剂盒（Solvent Box Fill）

**分类：** 分子与溶剂

## 功能说明

`Solvent Box Fill` 保留宿主结构，并在已有 cell 的整个体积内随机插入同一种溶剂分子。输入可以没有原子，但必须有有限、非奇异的 cell，并且至少开启一个周期方向。

它用于补充整盒溶剂环境；若目标是围绕离子或指定原子构造第一溶剂壳，应使用 `Local Solvation`。

## 原理与公式

`固定分子数`直接使用用户给定数量。`按密度估算`根据晶胞体积 $V$、目标密度 $\rho$、
填充比例 $p$ 和单个溶剂分子摩尔质量 $M$ 计算

$$
N_{\mathrm{solv}}=
\max\!\left(1,\operatorname{round}
\frac{\rho\,(V\times10^{-24})\,p}{M/N_A}\right),
$$

其中 $V$ 用 Å³、$\rho$ 用 g/cm³、$M$ 用 g/mol。这里使用完整 cell 体积，不扣除宿主、已有溶剂或真空层占据的体积。

每次试放会随机选择分子中心和整体取向。原子对 $i,j$ 的接受条件是

$$
d_{ij}^{\mathrm{MIC}} \ge
\begin{cases}
d_{\min}, & d_{\min}>0,\\
s(R_i+R_j), & d_{\min}=0,
\end{cases}
$$

其中 $d_{ij}^{\mathrm{MIC}}$ 是只沿已开启周期方向计算的最小镜像距离，$R_i$、$R_j$ 是元素碰撞半径，$s$ 是间距倍率。统一最小距离和元素半径规则是**二选一**，不会叠加。

卡片只生成未经平衡的初始结构；密度模式中的 $\rho$ 是数量估算输入，不代表输出已达到液体平衡密度。

## 操作示例

### 场景：模型只见过真空表面，没有见过周期溶剂盒

你要训练一个界面或溶液体系的 NEP，但已有训练集主要是干表面和单分子吸附构型。先准备带 cell 的周期输入结构，用 `Solvent Box Fill` 插入固定数量或按密度估算的水分子，生成一批整盒初态。

参数设置：`count_mode="fixed"`，`solvent_count=200`，`sampling_mode="general"`，`min_distance=0.8`，`max_attempts_per_solvent=500`，打开 `使用随机种子`（`use_seed`）。

检查输出时重点看是否插满、是否存在明显短接触、溶剂是否都包回 cell 内，以及后续预松弛是否大面积崩坏。

## 参数说明

### 溶剂输入

#### 溶剂 XYZ（solvent_xyz）

`str`，默认是一个三原子水分子 XYZ。这里保存溶剂分子文本，而不是文件路径。

#### 每个输入的独立输出数（structures）

`int`，默认 1。每个输入结构生成多少个独立填充版本。

### 数量控制

#### 目标用量（count_mode）

`str`，默认 `density`。可选 `fixed`、`按密度`（`density`）。`fixed` 直接使用目标分子数；`按密度`（`density`）根据**完整 cell 体积**、溶剂分子质量、目标密度和数量乘数估算一个名义分子数。

#### 目标溶剂分子数（solvent_count）

`int`，默认 100，仅在主动切换到 `count_mode="fixed"` 后生效。

#### 按密度（density）

`float`，默认 1.0，单位 g/cm³。`count_mode="density"` 时用于估算溶剂分子数量。

#### 完整 cell 计数系数（fill_packing）

`float`，默认 1.0，范围 `(0, 1]`。`count_mode="density"` 时乘到完整 cell 的名义纯溶剂分子数上；小于 1 会降低目标数量。

:::{warning}
密度模式不会扣除宿主原子、已有溶剂或真空层占据的体积，也不会根据溶质质量反算最终溶液密度。对含表面、溶质或大块固体的 cell，它只是按完整 cell 体积给出的初始数量估计；请看卡片预览中的解析数量，再用 `完整 cell 计数系数`（`fill_packing`）或固定数量修正。
:::

### 采样和几何约束

#### 碰撞配置（sampling_mode）

`str`，默认 `general`。界面提供三档真实有差异的间距倍率：

| 界面选项 | 序列化值 | $s$ | 效果 |
| --- | --- | ---: | --- |
| 紧凑间距 | `loose` | 0.62 | 允许分子靠得更近，更容易插满 |
| 标准间距 | `general` | 0.70 | 默认选择 |
| 保守间距 | `dense` | 0.78 | 留出更大原子间距，更可能提前耗尽尝试 |

旧工作流中的 `auto` 和 `water` 仍可加载，并按标准倍率 0.70 运行；重新在界面保存后会归一为 `general`。三档都只改变碰撞间距，分子中心和整体取向始终随机。

#### 统一最小距离覆盖值（min_distance）

`float`，默认 0，单位 Å。大于 0 时，所有原子对都使用该统一距离，并禁用间距预设和元素半径倍率；等于 0 时使用元素半径规则。

#### 元素半径碰撞缩放（collision_scale）

`float`，默认 0。等于 0 时使用间距预设；大于 0 时覆盖预设。仅在 `min_distance=0` 时生效。

#### 每个溶剂分子最大尝试次数（max_attempts_per_solvent）

`int`，默认 500。总尝试上限是“该值 × 目标分子数”；若连续拒绝次数达到内部停滞阈值，也会提前停止。高密度盒子触顶时，优先检查目标数量和碰撞规则，再考虑增加它。

#### 严格数量（strict_count）

`bool`，默认 true。打开后，未插满目标数量就失败；关闭后允许输出至少放入 1 个分子的部分结果。若一个分子都放不进去，仍会报错，避免把未变化的输入标成“已填盒”。

### 柔性溶剂

#### 启用柔性溶剂（flex_solvent）

`bool`，默认 false。打开后复用“分子构象”的几何启发式拓扑生成溶剂构象池，再用于填盒。水没有可旋转单键，因此普通水盒不需要打开；打开后主要只增加高斯坐标噪声。

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

### 固定数量水盒（200 个水，固定 seed）

```json
{
  "class": "SolventBoxFillCard",
  "check_state": true,
  "params": {
    "structures": 1,
    "count_mode": "fixed",
    "solvent_count": 200,
    "sampling_mode": "general",
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
    "sampling_mode": "general",
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
- 无法插满目标数量：先看预览中的目标分子数和名义纯溶剂密度；再降低数量或完整 cell 计数系数、减小碰撞距离，或者增大 cell。关闭严格数量后可保留非零的部分结果。
- 按密度估算的数量不符合预期：数量由完整 cell 体积、溶剂分子质量、目标密度和数量乘数共同决定，不会扣除宿主占据体积。先检查 cell 单位、真空层、已有原子和 `溶剂 XYZ`（`solvent_xyz`）。
- 想生成离子第一水合壳：改用 `Local Solvation`。整盒卡不会按离子元素选择 ion–O 距离或水分子取向。

## 输出标签

`SolvBox(mode={mode},req={目标数},ok={实际放入数})`

## 可复现性

打开 `使用随机种子`（`use_seed`）后，相同输入结构、结构顺序、参数和 `随机种子`（`seed`）会生成相同填盒结果。`count_mode="density"` 时，cell 体积或 `溶剂 XYZ`（`solvent_xyz`）改变会改变目标分子数，因此输出也会改变。
