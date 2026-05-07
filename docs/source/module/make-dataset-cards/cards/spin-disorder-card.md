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

`mode`：枚举，默认 `Flip fraction`。`Flip fraction` 取反选中磁矩；`Randomize fraction` 保持模长并随机方向；`Cone disorder` 在原方向 cone 内随机。

`fractions`：string，默认 `0.1,0.3,0.5,0.7`。每个数表示参与无序化的 eligible magnetic atoms 比例。

`samples_per_fraction`：int，默认 `1`。每个 fraction 输出多少个随机样本。

`cone_angle`：float，默认 `30.0` deg。仅 `mode=Cone disorder` 时使用。

`magnitude_source`：枚举，默认 `Existing initial magmoms`。可选 `Existing initial magmoms` 或 `Map/default magnitude`。

`magmom_map`：string，默认空。`magnitude_source=Map/default magnitude` 时使用，例如 `Fe:2.2,Co:1.7`。

`default_moment`：float，默认 `0.0`。元素不在 `magmom_map` 中时使用的磁矩模长。

`lift_scalar`：bool，默认 `True`。输入是一维标量磁矩时，是否沿 `axis` 提升成三维向量后处理。

`axis`：三维 float list，默认 `(0.0, 0.0, 1.0)`。用于提升标量磁矩和 map/default 参考态。

`apply_elements`：string，默认空。只对列出的元素施加无序化，空表示所有非零磁矩都可选。

`use_seed`：bool，默认 `False`。开启后随机选择和方向采样可复现。

`seed`：int，默认 `0`。`use_seed=True` 时作为基础 seed。

`max_outputs`：int，默认 `100`。每个输入结构最多输出多少个无序结构。

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
