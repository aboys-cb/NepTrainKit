<!-- card-schema: {"card_name": "Stacking Fault / GSFE Path", "source_file": "src/NepTrainKit/ui/views/_card/strict_gsfe_path_card.py", "serialized_keys": ["params"]} -->

# 层错 / GSFE 路径（Stacking Fault / GSFE Path）

**分类：** 缺陷

## 功能说明

按显式晶面 `hkl` 和滑移方向 `uvw` 生成层错结构或未弛豫的广义层错（GSF）位移路径。卡片先在原子层之间放置切面，再把切面上方的原子作为一个整体沿面内方向移动；它只生成结构，不计算能量，也不做原子弛豫。

它不限定 fcc、(111) 面或某一种 Burgers 矢量，但输入必须已经按目标晶面定向：第三晶胞矢量要垂直于 `层错面 (h k l)`（`plane_hkl`），否则程序会直接报错。`hkl` 和 `uvw` 都是相对于**当前晶胞基矢**的指数，不会自动识别或重定向原始体相晶胞里的 `(111)`、`[11-2]` 等晶向。

滑移方向必须本来就在层错面内；若含有面外分量，卡片会报错，不会静默投影成另一个方向。若只是为了补充训练集，可以直接使用输出结构；若要从能量差换算 GSFE，还必须自行确认周期边界中有一个还是两个层错界面，并使用正确的界面面积和界面数。

## 原理与公式

当前晶胞基矢为 $\mathbf a,\mathbf b,\mathbf c$。用户给出的滑移指数定义

$$
\mathbf b_{\mathrm{slip}}=u\mathbf a+v\mathbf b+w\mathbf c,
\qquad hu+kv+lw=0,
$$

第二式是滑移方向位于 $(hkl)$ 面内的必要检查。对位移参数 $\lambda$，分数向量模式使用
$\Delta\mathbf r=\lambda\mathbf b_{\mathrm{slip}}$；Å 模式使用
\(\Delta\mathbf r=d\,\mathbf b_{\mathrm{slip}}/
\|\mathbf b_{\mathrm{slip}}\|\)。切面上方原子统一加该位移，下方保持不动。

若用能量差计算广义层错能，应另行使用

$$
\gamma(\lambda)=
\frac{E(\lambda)-E(0)}{n_{\mathrm{interface}}A},
$$

其中 $A$ 是层错面面积，$n_{\mathrm{interface}}$ 是周期晶胞中的界面数。本卡只生成路径，
不会替用户判断 $n_{\mathrm{interface}}$ 或计算 $\gamma$。

## 操作示例

### 场景：模型在指定滑移系的 GSFE 峰值严重偏低

你在 fcc Ni 上训练了 NEP，体相和弹性响应都不错，但 (111)<112> 的 GSFE 峰值比 DFT 低很多。训练集里虽然有随机扰动和表面，但没有沿这个滑移系系统移动半晶体的结构，模型没学到层间错排的能量代价。

**输入：** 一个 fcc111 slab-oriented 周期超胞，例如 `Crystal Prototype Builder` 的 `fcc111` 原型。

**目标：** 沿当前 slab cell 的 `(001)` 面和面内 `[100]` 周期方向生成 0 到 1 个滑移向量的路径，步长 0.1；在这个特定定向晶胞里，`(001)` 对应原始 fcc 的 `(111)` 面。

**参数设置：**
- `层错面 (h k l)`填 `[0, 0, 1]`；
- `滑移方向 [u v w]`填 `[1, 0, 0]`；
- `位移范围`填 `[0.0, 1.0, 0.1]`；
- `位移单位`选择“滑移向量分数”；
- `切面位置`选择“中间原子层”。

**输出：** 11 个未弛豫层错结构，带 `GSFE(hkl=001,uvw=100,d=...)` 标签。

**怎么验证训练集质量改善：** 对输出结构做 DFT 单点或短弛豫后加入训练集，重训再扫同一 GSFE 路径。峰值、局部极小位置和曲线对称性应更接近 DFT；如果只有峰值改善但曲线仍有锯齿，把步长从 0.1 缩到 0.05。

## 参数说明

### 层错面 (h k l)（plane_hkl）

`Sequence[int]`，默认 `(0, 0, 1)`。当前晶胞基矢下的 Miller 指数，定义层错面的法向；该法向必须平行于第三晶胞矢量。

低指数面最常用：fcc111 slab cell 里用 `(001)` 表示原始 fcc `(111)` 面。`(0, 0, 0)` 没有物理意义，程序会直接报错。普通 cubic fcc cell 里的 `(111)` 不满足 slab-oriented 要求，不能直接使用。

### 滑移方向 [u v w]（slip_uvw）

`Sequence[int]`，默认 `(1, 0, 0)`。当前晶胞基矢下的方向指数，真实滑移向量为 `u*a + v*b + w*c`。

方向必须位于 `层错面 (h k l)`（`plane_hkl`）面内；在晶格指数下等价于 `h*u + k*v + l*w = 0`。不满足时程序会报错，避免用户填写 `[u v w]`，实际却得到另一个投影方向。对于 fcc111 slab cell，可以用 `[100]`、`[010]` 或另一个面内方向；如果要扫特定 partial 位移，需要让当前晶胞的面内基矢或 Å 位移范围对应目标 Burgers 矢量。

### 位移范围（displacement_range）

`Sequence[float]`，默认 `(0.0, 1.0, 0.5)`。格式是 `[起点, 终点, 步长]`，扫描滑移位移。

如果 `displacement_unit = "fraction_of_vector"`，`1.0` 表示完整 `滑移方向 [u v w]`（`slip_uvw`）向量；如果单位是 `angstrom`，数值就是沿该方向的实际距离。起点和终点都包含在输出中，步长必须为正。GSFE 曲线通常需要至少 0.05–0.1 的分辨率；只做 smoke 时可以用 0、0.5、1.0 三个点快速确认几何没错。

### 位移单位（displacement_unit）

`str`，默认 `"fraction_of_vector"`。决定 `位移范围`（`displacement_range`）里的数值怎么转成真实位移。

| 选项 | 含义 | 什么时候选 |
|------|------|-------------|
| `fraction_of_vector` | 位移 = 数值乘以 `滑移方向 [u v w]`（`slip_uvw`）向量 | 扫完整周期或按滑移向量分数描述路径 |
| `angstrom` | 位移 = 数值 Å，方向为 `滑移方向 [u v w]`（`slip_uvw`）的单位向量 | 已经知道实际位移长度，或只想小范围推开层间结构 |

### 切面位置（cut_mode）

`str`，默认 `"middle"`。决定哪一侧原子会被整体滑移。

| 选项 | 含义 | 什么时候选 |
|------|------|-------------|
| `middle` | 在最接近结构中部的两个离散原子层之间切开 | 普通超胞快速生成上下两部分 |
| `fractional` | 按最低与最高投影原子层之间的厚度分数切开 | 想明确控制切面在胞内的相对高度 |
| `下方原子层索引`（`layer_index`） | 在指定离散层和下一层之间切开 | 层状结构或 slab 中想从第几层上方开始滑移 |

无论选择哪种模式，切面两侧都必须至少有一个原子；卡片不会接受“所有原子一起移动”或“没有原子移动”的伪层错。

### 厚度分数（cut_fraction）

`float`，默认 `0.5`。当 `cut_mode = "fractional"` 时使用，界面范围是 0 到 0.9999；核心参数必须满足 `0 <= cut_fraction <= 1`，但 `1` 会因为上方没有原子而报错。

`0.5` 接近中间切面；`0.25` 会让更多原子处在上半部分并被滑移；`0.75` 则只移动靠近上侧的原子。这个值越靠近 0 或 1，移动的原子数越不平衡，适合 slab 或非均匀层结构，不适合普通体相路径的第一轮扫描。

生效条件：`切面位置`（`cut_mode`）必须是 `fractional`。

### 下方原子层索引（layer_index）

`int`，默认 `0`。当 `cut_mode = "layer_index"` 时使用。索引指定切面下方的原子层，切面实际放在该层和下一层的中间。

索引从 0 开始，必须选到顶层以下的位置；如果选到最后一层，程序会报错，因为那样没有上半部分可移动。层状材料、slab 或你已经知道滑移面夹在哪两层之间时，用它比 `middle` 更明确。

生效条件：`切面位置`（`cut_mode`）必须是 `下方原子层索引`（`layer_index`）。

### 将移动后的原子包回晶胞（wrap）

`bool`，默认 `true`。打开后把滑移后的原子坐标包回周期晶胞内；关闭则保留真实 Cartesian 位移。

训练集导出通常建议打开，避免原子跑出可视化盒子导致后续工具误判。做几何调试或想直接看上半部分移动了多远时可以关闭；关闭后输出仍然是同一个周期 cell，但坐标可能超过原胞边界。

## 推荐预设

### 对应 Atomsk 的 fcc (111) 半晶体滑移

参考脚本先把原始 fcc 晶体定向为 `x=[1-10]`、`y=[11-2]`、`z=[111]`，再移动 `z > 0.5*box` 的上半部分，位移沿当前 y 方向。把同一已定向结构送入本卡片时，对应参数是：

- `plane_hkl = [0, 0, 1]`
- `slip_uvw = [0, 1, 0]`
- `displacement_range = [0.0, 1.47786, 0.073893]`
- `displacement_unit = "angstrom"`
- `cut_mode = "fractional"`，`cut_fraction = 0.5`

对晶格常数 $a=3.62$ Å，终点 $1.47786 \approx a/\sqrt{6}$ Å，正好扫描一个 fcc Shockley partial 位移。参考脚本随后增加真空层；本卡片不自动改晶胞，因此若目标是含真空的单内部层错 slab，应在上游先准备好真空和定向。

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

- `Super Cell` → `Stacking Fault / GSFE Path`：先扩胞，再切面滑移，降低小胞周期重复带来的假相互作用。
- `Stacking Fault / GSFE Path` → `Atomic Perturb`：在每个位移点附近加入小扰动，让模型不仅见过理想路径，也见过热扰动后的层错环境。
- `Bain Path` → `Stacking Fault / GSFE Path`：先改变晶胞形状，再扫层错路径，用于相变或应变下的 GSFE 数据。

## 常见问题

**程序提示 slip_uvw 不在层错面内。** 检查 `h*u + k*v + l*w` 是否为 0。卡片不会把面外分量静默投影掉，请填写真实的面内晶向。

**移动的原子层不是预期那一半。** 先切到 `下方原子层索引`（`layer_index`）模式，用离散层索引明确切面；普通 `middle` 适合均匀体相，不一定适合 slab。

**输出坐标超过晶胞边界。** 打开 `将移动后的原子包回晶胞`（`wrap`）。如果你为了检查真实位移故意关闭了 `将移动后的原子包回晶胞`（`wrap`），这不是错误，但后续导出和可视化要知道坐标未包回。

**能量差除以面积后为什么和参考 GSFE 差一倍。** 先检查周期方向上有几个层错界面。纯周期体相超胞把上半部分整体平移时，胞内切面和周期边界都可能形成错配；若两处界面等价，换算单界面 GSFE 时还要除以 2。含真空 slab 通常只有一个内部切面，但仍应检查几何而不是按卡片名称直接假设。

## 输出标签

`GSFE(hkl={hkl},uvw={uvw},d={位移量})`

## 可复现性

无随机性。同一输入结构和同一参数会生成完全一致的结构列表。
