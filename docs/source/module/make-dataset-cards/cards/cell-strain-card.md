<!-- card-schema: {"card_name": "Lattice Strain", "source_file": "src/NepTrainKit/ui/views/_card/cell_strain_card.py", "serialized_keys": ["params"]} -->

# 晶格应变（Lattice Strain）

`Group`: `Lattice` | `Class`: `CellStrainCard`

## 功能说明

对晶胞的轴向做受控应变扫描。选一个方向组合（单轴/双轴/三轴/各向同性），给定拉伸范围和步长，程序按百分比改变晶格矢量，生成一组应变结构。

$$\epsilon_i=\frac{s_i}{100},\quad \mathbf{C}'=\mathbf{D}\mathbf{C},\quad \mathbf{D}=\mathrm{diag}(1+\epsilon_x,1+\epsilon_y,1+\epsilon_z)$$

## 操作示例

### 场景：测一个已弛豫晶体的弹性响应

**输入：** 一个弛豫好的 Si 单胞

**目标：** 沿 x 方向做 -2% 到 +2% 的单轴应变，步长 1%，看看模型能不能正确预测应力-应变趋势

**参数设置：**
- `Axes` = `uniaxial`
- `x_range` = `[-2, 2, 1]` （-2% 到 +2%，每 1% 一步）
- `y_range` / `z_range` 用默认值即可（uniaxial 模式只生效 x 方向）

**输出：** 5 个结构（-2%, -1%, 0%, +1%, +2%），晶格 x 分量逐步拉长

**怎么验证结果合理：**
- 用可视化工具检查晶格 x 长度是否按百分比线性变化
- y、z 方向晶格长度不应变化
- 原子分数坐标应保持不变（`scale_atoms=True` 下绝对坐标自动缩放）

## 参数说明

### `Axes`（engine_type）

下拉选择。决定同时对哪些轴施加应变。

| 选项 | 含义 | 什么时候用 |
|------|------|-----------|
| `uniaxial` | 只拉压一个轴，对 x/y/z 分别扫描 | 测各向异性弹性常数 |
| `biaxial` | 两两组合（xy, xz, yz），同时拉压两个轴 | 覆盖泊松耦合效应 |
| `triaxial` | 三轴同时独立扫描 | 预算充足时做全空间覆盖 |
| `isotropic` | 三轴等比例缩放 | 只要体积变化，不改变晶格形状 |

自由输入单个轴也可以，比如输入 `X` 等价于 uniaxial 且只扫 x 方向。

**输出规模：** isotropic 最少（每个 strain 值 1 个结构），triaxial 最多（三维网格，组合数 = Nx × Ny × Nz）。

### `X` / `Y` / `Z`（x_range / y_range / z_range）

三个独立的 `[最小值, 最大值, 步长]`，单位是百分比。

- `[-3, 3, 1]` 表示从 -3% 压缩到 +3% 拉伸，每隔 1% 取一个点
- 只有当前 `Axes` 模式激活的轴才生效。其余轴的方向即使填了值也被忽略

**推荐幅度：**
- ±1~2%：大多数晶体的弹性区，键长变化 < 0.1Å，适合近平衡训练
- ±3~5%：开始进入非谐区域，适合覆盖中等变形
- ±6%+：极端变形，可能产生非物理键长，建议先抽查输出再批量跑

### `Identify organic`（organic）

复选框。开启后程序先识别有机分子团簇，应变时将分子作为刚性整体移动，而不是对分子内每个原子单独缩放。

- 输入是分子晶体（MOF、有机半导体等）：**开启**
- 输入是无机晶体（金属、氧化物等）：**关闭**，开了只会增加不必要计算

## 推荐预设

### 近平衡弹性（isotropic, ±1%）
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

### 各向异性覆盖（uniaxial, ±3%, 三个方向）
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

### 大变形探索（biaxial, ±6%）
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

- `Lattice Strain` → `Atomic Perturb`：应变后再加原子坐标噪声，覆盖几何+热扰动
- `Lattice Strain` → `Shear Matrix Strain`：轴向应变后补剪切分量
- 作为 `Random Slab` / `Insert Defect` / 磁性卡片的母胞准备步骤

## 常见问题

**输出为空。** 检查各轴的 `步长` 是否 > 0，`最大值` 是否 >= `最小值`。如果某个轴的步长为 0，该方向不会产生扫描点。

**输出结构晶格角明显变化。** 这是纯轴向应变，不应该改变角度。如果角度变了，检查是否上游结构已经带有非正交晶格，或误用了其他卡片。

**组合爆炸。** `triaxial` + 细步长会很难产。比如 3 个轴各 10 个点 = 1000 个结构。先算一下 Nx × Ny × Nz 再跑。

**有机分子键被拉断。** 确认 `Identify organic` 已开启。

## 输出标签

输出结构的 `Config_type` 追加标签：`Str(x=-2%,y=1%)` / `Str(all=3%)` 等。

## 可复现性

无随机性。相同参数 + 相同输入结构 → 输出严格一致。
