<!-- card-schema: {"card_name": "Spin Disorder", "source_file": "src/NepTrainKit/ui/views/_card/spin_disorder_card.py", "serialized_keys": ["params"]} -->

# 自旋无序（Spin Disorder）

`Group`: `Magnetism` | `Class`: `SpinDisorderCard`

## 功能说明

`Spin Disorder` 从已有磁矩或元素磁矩表出发，按指定无序比例生成中间磁态。它覆盖 FM/AFM 到 PM 之间的离散翻转、随机方向和 cone disorder，不把这些离散无序混进 `Magmom Rotation` 的连续小角扰动语义里。

## 操作示例

### 场景：模型只见过 FM/AFM 和完全随机 PM

FeCo 训练集包含 FM、AFM 和完全随机 PM，但缺少 10%-70% 局部翻转的中间无序态。模型在有限温度磁态上能量排序不稳定。

**输入：** 已经通过 `Magnetic Order` 或 `Set Magnetic Moments` 写入磁矩的结构。
**目标：** 生成 `0.1,0.3,0.5,0.7` 四档局部翻转，让训练集覆盖有序到无序的连续路径。
**参数设置：** `mode=Flip fraction`，`fractions=0.1,0.3,0.5,0.7`，`samples_per_fraction=3`，开启 `use_seed`。
**输出：** 每个输入结构生成 12 个自旋无序结构。
**怎么验证训练集质量改善：** 重训后，中间翻转比例测试集的能量/磁力误差应不再显著高于 FM/AFM 端点。

## 参数说明

### 无序模型

#### Mode（mode）

`str`，默认 `'Flip fraction'`。`Flip fraction` 保持共线轴只翻转符号，适合 FM/AFM 到 PM 的离散无序梯度；`Randomize fraction` 把选中自旋完全随机化；`Cone disorder` 保持围绕参考方向的有限温非共线扰动。

#### Fractions（fractions）

`str`，默认 `'0.1,0.3,0.5,0.7'`。被翻转或随机化的自旋比例，从 0 到 1。0.1/0.3/0.5/0.7 可覆盖 FM/AFM 到 PM 之间的无序梯度。

#### Samples Per Fraction（samples_per_fraction）

`int`，默认 `1`。同一无序度下不同随机选择会给不同局域环境。1 个用于路径扫描，3-10 个用于统计训练。

#### Cone Angle（cone_angle）

`float`，默认 `30.0`。Cone disorder 中限制随机方向偏离参考轴的最大角。10-30° 表示有序态附近有限温扰动；接近 90° 时已接近强无序。

生效条件：`mode` 或方向模型选择 cone/noncollinear 随机化时。

### 磁矩幅值

#### Magnitude Source（magnitude_source）

`str`，默认 `'Existing initial magmoms'`。已有 `initial_magmoms` 时复用它最安全；没有磁矩输入时用 `magmom_map`/`default_moment` 构造幅值。不要用默认幅值替代已知元素磁矩。

#### Magmom Map（magmom_map）

`str`，默认 `''`。已知元素局域磁矩时显式写入，如 `Fe:2.2,Ni:0.6`。未知元素不要用默认值伪造先验。

#### Default Moment（default_moment）

`float`，默认 `0.0`。只作为 `magmom_map` 未命中元素的兜底幅值。关键磁性元素应显式列出，非磁元素通常保持 0。

#### Lift Scalar（lift_scalar）

`bool`，默认 `True`。输入是标量磁矩但下游需要非共线向量时打开；如果原始数据已有方向信息，不要重新提升覆盖它。

#### Axis（axis）

`list[float] | tuple[float, float, float]`，默认 `(0.0, 0.0, 1.0)`。这是方向参考，不是普通数值——改它会改变分层、表面法向或磁矩方向。使用前先确认 cell 取向和目标物理方向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

#### Apply Elements（apply_elements）

`str`，默认 `''`。只对列出的磁性元素翻转或随机化。合金/界面里应显式列出磁性元素，避免给非磁原子写入无意义磁矩。留空则全部参与。

### 随机性和预算

#### Use Seed（use_seed）

`bool`，默认 `False`。勾选后固定种子可复现。对比实验时开，最终大规模随机探索可以关——但关后结果不能逐帧复现。

#### Seed（seed）

`int`，默认 `0`。同一输入、同一参数和同一 seed 应生成同一批候选。

生效条件：`use_seed=True`。

#### Max Outputs（max_outputs）

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

- `Magnetic Order -> Spin Disorder`：先建立 FM/AFM 参考态，再生成中间无序比例。
- `Set Magnetic Moments -> Spin Disorder -> Small-Angle Spin Tilt`：先统一磁矩模长，再做无序化和小角偏转。
- `Spin Disorder -> Geometry Filter`：磁性结构如果同时经过强几何扰动，后面接几何清洗。

## 常见问题

**运行报错：找不到 eligible magnetic moments。** 输入没有可用磁矩，或 `apply_elements` 没有匹配任何非零磁矩。先用 `Set Magnetic Moments` 或 `Magnetic Order` 初始化。

**翻转数量不是精确小数比例。** 原子数是离散的，程序按 fraction 转成整数个原子，至少选 1 个且不超过 eligible atom 数。

**`Cone disorder` 看起来和 PM 不一样。** cone disorder 只在参考方向附近随机，不是全空间 PM。全空间随机用 `Randomize fraction`。

## 输出标签

`SpinDis(f={fraction},n={count},mode={flip|rand|cone},s={seed},a={cone_angle})`。`s` 只在 `use_seed=True` 时出现，`a` 只在 cone disorder 输出中出现。

## 可复现性

开启 `use_seed` 后，随机原子选择和方向采样由 `seed`、输入结构标识、fraction 序号和 sample 序号共同决定。
