<!-- card-schema: {"card_name": "Lattice Strain", "source_file": "src/NepTrainKit/ui/views/_card/cell_strain_card.py", "serialized_keys": ["params"]} -->

# 晶格应变（Lattice Strain）

`Group`: `Lattice` | `Class`: `CellStrainCard`

## 功能说明

对晶胞轴向做受控应变扫描。选定方向组合（单轴/双轴/三轴/各向同性），给定拉伸百分比和步长，程序按比例缩放晶格矢量，生成一组不同应变的结构。

$$\epsilon_i=\frac{s_i}{100},\quad \mathbf{C}'=\mathbf{D}\mathbf{C},\quad \mathbf{D}=\mathrm{diag}(1+\epsilon_x,1+\epsilon_y,1+\epsilon_z)$$

## 操作示例

### 场景：模型预测的弹性常数偏差太大

你在 Si 上训练了一个 NEP 模型，能量-体积曲线拟合得很好，但算出来的 C11 比 DFT 参考值高了 15%。这说明训练集里所有结构都处在平衡体积附近，模型没见过晶格被拉伸/压缩的构型，只能外推——外推不准。

**诊断思路：** 弹性常数是能量对应变的二阶导数。如果训练集在平衡点附近的应变成分太稀疏，导数拟合就不可靠。解决办法是往训练集里加入一组已知应变的晶格，让模型"见过"被拉伸和压缩的晶胞。

**输入：** 一个弛豫好的 Si 单胞（当前训练集里唯一的结构）

**目标：** 沿 x 方向做 -2% 到 +2% 的单轴应变，步长 1%，共 5 个结构，让模型在 ±2% 范围内有内插依据

**参数设置：**
- `Axes` = `uniaxial`
- `x_range` = `[-2, 2, 1]` （-2% 到 +2%，每 1% 一步）

**输出：** 5 个结构（-2%, -1%, 0%, +1%, +2%），晶格 x 分量逐步拉长，原子分数坐标保持不变

**怎么验证训练集质量改善：**
- 重训后重新计算 C11，应该比之前更接近 DFT 参考值
- 如果改善不够，把范围扩大到 ±5%，加 biaxial 方向，增加采样密度
- 如果应变 +5% 后最近邻键长超过物理合理区间（和 DFT 参考键长差 > 0.2Å），说明到了非物理变形区，上限回调

### 什么时候加这张卡、什么时候不加

**加：**
- 模型弹性常数、体模量、应力-应变响应系统性偏差大
- 训练集中所有结构体积/形状过于集中在平衡态附近
- 需要模型在变形路径上有内插能力而非外推

**不加：**
- 只需要局部原子坐标噪声（用 `Atomic Perturb` 更合适）
- 体系本身对应变不敏感（如刚性分子晶体），加了只是增大训练集体积没帮助

## 参数说明
### Axes（axes）

`str`，默认 `'uniaxial'`。选 `uniaxial` 就只沿一个方向拉/压，补 C11/C22/C33 这类单轴弹性响应；`biaxial` 同时动两个方向，面内耦合和泊松效应都能覆盖；`isotropic` 三个方向等比例缩放，更接近 EOS 和体模量采样。注意，别在只想做体积曲线的时候打开三轴独立扫描（`triaxial`）——那会触发热爆炸级的组合数。

### X Range（x_range）

`tuple[float, float, float]`，默认 `(-5.0, 5.0, 1.0)`。格式是 `[最小值, 最大值, 步长]`，单位 %。只有 `Axes` 激活的方向才生效；isotropic 模式只认 x_range。步长越小、范围越宽，输出的结构就越多。

典型参考值：±1~2% 是大多数无机晶体的弹性区，键长变化一般不到 0.1A，补弹性常数首选这个区间。推到 ±3~5% 就进入非谐区了，适合做应力-应变曲线覆盖，抽检输出确认没有非物理键长。如果你设到 ±6% 以上，那是极端变形，很可能拉断键——每批跑完务必抽查最近邻距离。

### Y Range（y_range）

`tuple[float, float, float]`，默认 `(-5.0, 5.0, 1.0)`。语法和 x_range 完全一样，仅作用在 y 轴。只有 `Axes` 激活的方向才生效。isotropic 模式不单独看 y_range。典型参考值和 x_range 一致：±1~2% 弹性区，±3~5% 非谐区，±6%+ 需要警惕非物理键长。

### Z Range（z_range）

`tuple[float, float, float]`，默认 `(-5.0, 5.0, 1.0)`。同上，只作用于 z 轴。isotropic 模式不单独看 z_range。典型参考值同上：±1~2% 弹性区，±3~5% 非谐区，±6%+ 注意抽查最近邻距离。

### Identify Organic（identify_organic）

`bool`，默认 `false`。分子晶体、MOF、有机半导体这几种场景必须打开——打开后程序会识别有机分子团簇，晶格应变只作用在分子间，分子内部的键长尽量保持刚性。纯无机晶体关着就行，开了反而多算一堆团簇检测。

## 推荐预设

### 近平衡弹性（isotropic ±1%，适合补体模量）
```json
{
  "class": "CellStrainCard",
  "check_state": true,
  "organic": false,
  "engine_type": "isotropic",
  "x_range": [-1, 1, 1],
  "y_range": [-1, 1, 1],
  "z_range": [-1, 1, 1]
}
```

### 各向异性弹性（uniaxial ±3%，三个方向各 7 个点）
```json
{
  "class": "CellStrainCard",
  "check_state": true,
  "organic": false,
  "engine_type": "uniaxial",
  "x_range": [-3, 3, 1],
  "y_range": [-3, 3, 1],
  "z_range": [-3, 3, 1]
}
```

### 全方向覆盖（biaxial ±6%，适合怀疑模型在大变形下崩溃）
```json
{
  "class": "CellStrainCard",
  "check_state": true,
  "organic": false,
  "engine_type": "biaxial",
  "x_range": [-6, 6, 2],
  "y_range": [-6, 6, 2],
  "z_range": [-6, 6, 2]
}
```

## 推荐组合

- `Lattice Strain` → `Atomic Perturb`：应变 + 坐标噪声，同时覆盖几何变形和热扰动
- `Lattice Strain` → `Shear Matrix Strain`：先补轴向应变，再补剪切分量
- `Super Cell` → `Lattice Strain`：先扩胞到目标尺寸，再做应变扫描

## 常见问题

**输出为空。** 步长 ≤ 0，或者最大值 < 最小值。检查 x/y/z_range。

**结构角度异常变化。** 轴向应变不改变晶格角。如果变了，检查上游结构是否已经带非正交晶格。

**组合爆炸。** `triaxial` + 步长 0.5% + ±5% = 21³ = 9261 个结构。先估算 Nx × Ny × Nz 再跑。

**有机分子键被拉断。** `Identify organic` 没开。

## 输出标签

`Str(x=-1%,y=2%)` / `Str(all=3%)` 等。

## 可复现性

无随机性。同参数同输入 → 严格一致输出。
