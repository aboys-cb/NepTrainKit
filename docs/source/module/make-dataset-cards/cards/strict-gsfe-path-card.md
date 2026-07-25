<!-- card-schema: {"card_name": "Strict GSFE Path", "source_file": "src/NepTrainKit/ui/views/_card/strict_gsfe_path_card.py", "serialized_keys": ["params"]} -->

# 严格 GSFE 路径（Strict GSFE Path）

`Group`: `Defect` | `Class`: `StrictGSFEPathCard`

## 功能说明

按显式晶面 `hkl` 和滑移方向 `uvw` 生成未弛豫的广义层错能（GSFE）路径结构。输入必须已经是 slab-oriented cell：第三晶胞方向要垂直于 `plane_hkl`，否则周期边界会把断层面错接成重合或过近原子，程序会直接报错。

相比 `Stacking Fault`，这张卡把滑移方向写成明确的晶向，并提供 `middle`、`fractional`、`layer_index` 三种切面选择。它适合你已经知道目标滑移系统，想要严格控制 GSFE 路径几何的场景。

## 操作示例

### 场景：模型在指定滑移系的 GSFE 峰值严重偏低

你在 fcc Ni 上训练了 NEP，体相和弹性响应都不错，但 (111)<112> 的 GSFE 峰值比 DFT 低很多。训练集里虽然有随机扰动和表面，但没有沿这个滑移系系统移动半晶体的结构，模型没学到层间错排的能量代价。

**输入：** 一个 fcc111 slab-oriented 周期超胞，例如 `Crystal Prototype Builder` 的 `fcc111` 原型。

**目标：** 沿 slab cell 的 `(001)` 面和面内 `[100]` 周期方向生成 0 到 1 个滑移向量的路径，步长 0.1；这个 `(001)` 就是原始 fcc 的 `(111)` 面。

**参数设置：**
- `plane_hkl = [0, 0, 1]`
- `slip_uvw = [1, 0, 0]`
- `displacement_range = [0.0, 1.0, 0.1]`
- `displacement_unit = "fraction_of_vector"`
- `cut_mode = "middle"`

**输出：** 11 个未弛豫层错结构，带 `GSFE(hkl=001,uvw=100,d=...)` 标签。

**怎么验证训练集质量改善：** 对输出结构做 DFT 单点或短弛豫后加入训练集，重训再扫同一 GSFE 路径。峰值、局部极小位置和曲线对称性应更接近 DFT；如果只有峰值改善但曲线仍有锯齿，把步长从 0.1 缩到 0.05。

## 参数说明

### Plane HKL（plane_hkl）

`Sequence[int]`，默认 `(0, 0, 1)`。Miller 指数，定义哪一个晶面作为层错切面；该面的法向必须平行于第三晶胞方向。

低指数面最常用：fcc111 slab cell 里用 `(001)` 表示原始 fcc `(111)` 面。`(0, 0, 0)` 没有物理意义，程序会直接报错。普通 cubic fcc cell 里的 `(111)` 不满足 slab-oriented 要求，不能直接使用。

### Slip UVW（slip_uvw）

`Sequence[int]`，默认 `(1, 0, 0)`。晶向指数，定义滑移方向；程序会把它投影到 `plane_hkl` 所在平面内。

如果 `slip_uvw` 平行于晶面法向，投影后长度为零，程序会报错。对于 fcc111 slab cell，先用 `[100]` 或另一个面内周期方向做完整周期路径；如果要扫特定 partial 位移，需要把 slab cell 的面内基矢先构造成对应路径。

### Displacement Range（displacement_range）

`Sequence[float]`，默认 `(0.0, 1.0, 0.5)`。格式是 `[起点, 终点, 步长]`，扫描滑移位移。

如果 `displacement_unit = "fraction_of_vector"`，`1.0` 表示完整投影滑移向量；如果单位是 `angstrom`，数值就是 A。GSFE 曲线通常需要至少 0.05-0.1 的分辨率；只做 smoke 时可以用 0、0.5、1.0 三个点快速确认几何没错。

### Displacement Unit（displacement_unit）

`str`，默认 `"fraction_of_vector"`。决定 `displacement_range` 里的数值怎么转成真实位移。

| 选项 | 含义 | 什么时候选 |
|------|------|-------------|
| `fraction_of_vector` | 位移 = 数值乘以投影后的 `slip_uvw` 向量 | 扫完整 GSFE 周期或按晶向分数描述路径 |
| `angstrom` | 位移 = 数值 A，方向为投影滑移方向的单位向量 | 已经知道实际位移长度，或只想小范围推开层间结构 |

### Cut Mode（cut_mode）

`str`，默认 `"middle"`。决定哪一侧原子会被整体滑移。

| 选项 | 含义 | 什么时候选 |
|------|------|-------------|
| `middle` | 按投影坐标中位数切开 | 普通超胞快速生成上下两半 |
| `fractional` | 按投影坐标范围的 `cut_fraction` 位置切开 | 想明确控制切面在胞内的相对高度 |
| `layer_index` | 按排序后的离散层位置切开 | 层状结构或 slab 中想从第几层上方开始滑移 |

### Cut Fraction（cut_fraction）

`float`，默认 `0.5`。当 `cut_mode = "fractional"` 时使用，范围是 0 到 1。

`0.5` 接近中间切面；`0.25` 会让更多原子处在上半部分并被滑移；`0.75` 则只移动靠近上侧的原子。这个值越靠近 0 或 1，移动的原子数越不平衡，适合 slab 或非均匀层结构，不适合普通体相路径的第一轮扫描。

生效条件：`cut_mode` 必须是 `fractional`。

### Layer Index（layer_index）

`int`，默认 `0`。当 `cut_mode = "layer_index"` 时使用，选择哪一层下面保持不动、上面的层整体滑移。

索引从 0 开始，必须选到顶层以下的位置；如果选到最后一层，程序会报错，因为那样没有上半部分可移动。层状材料、slab 或你已经知道滑移面夹在哪两层之间时，用它比 `middle` 更明确。

生效条件：`cut_mode` 必须是 `layer_index`。

### Wrap（wrap）

`bool`，默认 `true`。打开后把滑移后的原子坐标包回周期晶胞内；关闭则保留真实 Cartesian 位移。

训练集导出通常建议打开，避免原子跑出可视化盒子导致后续工具误判。做几何调试或想直接看半晶体移动了多远时可以关闭；关闭后输出仍然是同一个周期 cell，但坐标可能超过原胞边界。

## 推荐预设

### fcc111 slab 完整分数路径

适合 GSFE 曲线第一轮覆盖，输出 11 个结构。

```json
{
  "class": "StrictGSFEPathCard",
  "check_state": true,
  "params": {
    "plane_hkl": [0, 0, 1],
    "slip_uvw": [1, 0, 0],
    "displacement_range": [0.0, 1.0, 0.1],
    "displacement_unit": "fraction_of_vector",
    "cut_mode": "middle",
    "cut_fraction": 0.5,
    "layer_index": 0,
    "wrap": true
  }
}
```

### 小位移 Angstrom 探针

适合只想补层间微小错排，不扫完整滑移周期。

```json
{
  "class": "StrictGSFEPathCard",
  "check_state": true,
  "params": {
    "plane_hkl": [0, 0, 1],
    "slip_uvw": [1, 0, 0],
    "displacement_range": [0.0, 0.5, 0.1],
    "displacement_unit": "angstrom",
    "cut_mode": "middle",
    "cut_fraction": 0.5,
    "layer_index": 0,
    "wrap": true
  }
}
```

### 指定层间切面

适合 slab 或层状材料，希望从第 2 层上方开始滑移。

```json
{
  "class": "StrictGSFEPathCard",
  "check_state": true,
  "params": {
    "plane_hkl": [0, 0, 1],
    "slip_uvw": [1, 0, 0],
    "displacement_range": [0.0, 1.0, 0.25],
    "displacement_unit": "angstrom",
    "cut_mode": "layer_index",
    "cut_fraction": 0.5,
    "layer_index": 1,
    "wrap": false
  }
}
```

## 推荐组合

- `Super Cell` → `Strict GSFE Path`：先扩胞，再切面滑移，降低小胞周期重复带来的假相互作用。
- `Strict GSFE Path` → `Atomic Perturb`：在每个位移点附近加入小扰动，让模型不仅见过理想路径，也见过热扰动后的层错环境。
- `Bain Path` → `Strict GSFE Path`：先改变晶胞形状，再扫层错路径，用于相变或应变下的 GSFE 数据。

## 常见问题

**程序提示 slip_uvw 平行于 plane normal。** 这个方向投影到晶面内后长度为零，不能作为滑移方向。换一个面内晶向。

**移动的原子层不是预期那一半。** 先切到 `layer_index` 模式，用离散层索引明确切面；普通 `middle` 适合均匀体相，不一定适合 slab。

**输出坐标超过晶胞边界。** 打开 `wrap`。如果你为了检查真实位移故意关闭了 `wrap`，这不是错误，但后续导出和可视化要知道坐标未包回。

## 输出标签

`GSFE(hkl={hkl},uvw={uvw},d={位移量})`

## 可复现性

无随机性。同一输入结构和同一参数会生成完全一致的结构列表。
