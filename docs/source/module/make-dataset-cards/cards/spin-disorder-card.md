<!-- card-schema: {"card_name": "Spin Disorder", "source_file": "src/NepTrainKit/ui/views/_card/spin_disorder_card.py", "serialized_keys": ["params"]} -->

# 自旋无序（Spin Disorder）

**分类：** 磁性

## 功能说明

`Spin Disorder` 从已有磁矩或元素磁矩表出发，按指定无序比例生成中间磁态。它覆盖 FM/AFM 到 PM 之间的离散翻转、随机方向和 cone disorder，不把这些离散无序混进 `Spin Perturb` 的局部连续扰动语义里。

## 原理与公式

设符合元素筛选且磁矩非零的原子数为 $N_m$，无序比例为 $f$。每个样本实际改动

$$
N_{\mathrm{change}}=\min\!\left(N_m,\max\left[1,
\operatorname{round}(fN_m)\right]\right)
$$

个不重复原子。`翻转一定比例`使用
$\mathbf m_i'=-\mathbf m_i$；`圆锥无序`在原方向周围半角
$\theta_{\max}$ 的球冠上均匀抽样，其中
$\cos\theta\sim U(\cos\theta_{\max},1)$；`随机化一定比例`在整个单位球面上抽取新方向。
三种模式都保留被改动磁矩的模长，只改变方向。固定随机种子后，结构编号、比例和样本编号
共同派生每个输出的随机路径。

## 操作示例

### 场景：模型只见过 FM/AFM 和完全随机 PM

FeCo 训练集包含 FM、AFM 和完全随机 PM，但缺少 10%-70% 局部翻转的中间无序态。模型在有限温度磁态上能量排序不稳定。

**输入：** 已经通过 `Magnetic Order` 或 `Set Magnetic Moments` 写入磁矩的结构。
**目标：** 生成 `0.1,0.3,0.5,0.7` 四档局部翻转，让训练集覆盖有序到无序的连续路径。
**参数设置：** `无序方式`选择“翻转一定比例”，`无序比例`填
`0.1,0.3,0.5,0.7`，`每个比例的样本数`填 `3`，并开启`使用随机种子`。
**输出：** 每个输入结构生成 12 个自旋无序结构。
**怎么验证训练集质量改善：** 重训后，中间翻转比例测试集的能量/磁力误差应不再显著高于 FM/AFM 端点。

## 参数说明

### 无序模型

#### 模式（mode）

`str`，默认 `'Flip fraction'`。`Flip fraction` 保持共线轴只翻转符号，适合 FM/AFM 到 PM 的离散无序梯度；`Randomize fraction` 把选中自旋方向在完整球面上随机化；`Cone disorder` 保持围绕参考方向的有限温非共线扰动。这三个值与下拉框和序列化参数完全一致。

#### 比例（fractions）

`str`，默认 `'0.1,0.3,0.5,0.7'`。界面使用比例的最小值、最大值和步长控件，非等距扫描可切换到自定义列表。每个值必须位于 `(0, 1]`；非法值会明确报错，不会被忽略或截断。0.1/0.3/0.5/0.7 可覆盖 FM/AFM 到 PM 之间的无序梯度。

#### 每个无序比例的样本数（samples_per_fraction）

`int`，默认 `1`。同一无序度下不同随机选择会给不同局域环境。1 个用于路径扫描，3-10 个用于统计训练。

#### 锥角（cone_angle）

`float`，默认 `30.0`。Cone disorder 中限制随机方向偏离参考轴的最大角。10-30° 表示有序态附近有限温扰动；接近 90° 时已接近强无序。

生效条件：`模式`（`mode`）或方向模型选择 cone/noncollinear 随机化时。

### 磁矩幅值

#### 磁矩大小来源（magnitude_source）

`str`，默认 `'Existing initial magmoms'`。输入有 `spin:R:3` 时优先复用；没有 `spin` 时兼容旧 `initial_magmoms`。两者都没有时用 `元素磁矩表`（`magmom_map`）/`默认磁矩`（`default_moment`）构造幅值。不要用默认幅值替代已知元素磁矩。

#### 元素磁矩表（magmom_map）

`str`，默认 `''`。已知元素局域磁矩时显式写入，如 `Fe:2.2,Ni:0.6`。未知元素不要用默认值伪造先验。

#### 默认磁矩（default_moment）

`float`，默认 `0.0`。只作为 `元素磁矩表`（`magmom_map`）未命中元素的兜底幅值。关键磁性元素应显式列出，非磁元素通常保持 0。

#### 标量磁矩转为矢量（lift_scalar）

`bool`，默认 `True`。输入是标量磁矩但下游需要非共线向量时打开；如果原始数据已有方向信息，不要重新提升覆盖它。

#### 轴（axis）

`list[float] | tuple[float, float, float]`，默认 `(0.0, 0.0, 1.0)`。这是方向参考，不是普通数值——改它会改变分层、表面法向或磁矩方向。使用前先确认 cell 取向和目标物理方向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

#### 应用元素（apply_elements）

`str`，默认 `''`。只对列出的磁性元素翻转或随机化。合金/界面里应显式列出磁性元素，避免给非磁原子写入无意义磁矩。留空则全部参与。

### 随机性和预算

#### 使用随机种子（use_seed）

`bool`，默认 `False`。勾选后固定种子可复现。对比实验时开，最终大规模随机探索可以关——但关后结果不能逐帧复现。

#### 随机种子（seed）

`int`，默认 `0`。同一输入、同一参数和同一 seed 应生成同一批候选。

生效条件：`use_seed=True`。

#### 最大输出数（max_outputs）

`int`，默认 `100`。总输出约等于 fractions 数量乘以 samples_per_fraction。链式输入多时必须设上限，避免磁无序样本淹没结构样本。

## 推荐预设

### 共线翻转梯度

```json
{
  "class": "SpinDisorderCard",
  "params": {
    "mode": "Flip fraction",
    "fractions": "0.1,0.3,0.5,0.7",
    "samples_per_fraction": 3,
    "cone_angle": 30.0,
    "magnitude_source": "Existing initial magmoms",
    "magmom_map": "",
    "default_moment": 0.0,
    "lift_scalar": true,
    "axis": [0.0, 0.0, 1.0],
    "apply_elements": "",
    "use_seed": true,
    "seed": 42,
    "max_outputs": 100
  }
}
```

用于从 FM/AFM 参考态生成局部翻转比例扫描。

### 非共线 cone disorder

```json
{
  "class": "SpinDisorderCard",
  "params": {
    "mode": "Cone disorder",
    "fractions": "0.25,0.5,0.75",
    "samples_per_fraction": 2,
    "cone_angle": 20.0,
    "magnitude_source": "Existing initial magmoms",
    "magmom_map": "",
    "default_moment": 0.0,
    "lift_scalar": true,
    "axis": [0.0, 0.0, 1.0],
    "apply_elements": "Fe,Co",
    "use_seed": true,
    "seed": 7,
    "max_outputs": 50
  }
}
```

用于有限温度附近的非共线方向扰动。

## 推荐组合

- `Magnetic Order → Spin Disorder`：先建立 FM/AFM 参考态，再生成中间无序比例。
- `Set Magnetic Moments → Spin Disorder → Canting Scan`：先统一磁矩模长，再做无序化和可控偏转。
- `Spin Disorder → Geometry Filter`：磁性结构如果同时经过强几何扰动，后面接几何清洗。

## 常见问题

**运行报错：找不到 eligible magnetic moments。** 输入没有可用磁矩，或 `应用元素`（`apply_elements`）没有匹配任何非零磁矩。先用 `Set Magnetic Moments` 或 `Magnetic Order` 初始化。

**翻转数量不是精确小数比例。** 原子数是离散的，程序按 fraction 转成整数个原子，至少选 1 个且不超过 eligible atom 数。

**`Cone disorder` 看起来和 PM 不一样。** cone disorder 只在参考方向附近随机，不是全空间 PM。全空间随机用 `Randomize fraction`。

## 输出标签

`SpinDis(f={fraction},n={count},mode={flip|rand|cone},s={seed},a={cone_angle})`。`s` 只在 `use_seed=True` 时出现，`a` 只在 cone disorder 输出中出现。

所有导出输出写入 `spin:R:3`；内部同步维护 ASE `initial_magmoms` 向量别名。

## 可复现性

开启 `使用随机种子`（`use_seed`）后，随机原子选择和方向采样由 `随机种子`（`seed`）、输入结构标识、fraction 序号和 sample 序号共同决定。
