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
类型：`str`。默认：`'Flip fraction'`。选择这张卡的主操作模式。

物理直觉：主模式会改变生成语义；不要只为了多出结构而混用物理含义不同的模式。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### Fractions（fractions）
类型：`str`。默认：`'0.1,0.3,0.5,0.7'`。设置离散翻转或随机化的无序度扫描列表。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Samples Per Fraction（samples_per_fraction）
类型：`int`。默认：`1`。设置每个无序度取值生成的样本数量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Cone Angle（cone_angle）
类型：`float`。默认：`30.0`。设置非共线 cone disorder 的最大偏转角。

物理直觉：小 cone 对应有限温扰动，大 cone 更接近强无序态；不要把它当作真实温度。

生效条件：`mode` 或方向模型选择 cone/noncollinear 随机化时。


### 磁矩幅值

#### Magnitude Source（magnitude_source）
类型：`str`。默认：`'Existing initial magmoms'`。选择磁矩幅值来自元素映射、默认值还是已有结构。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### Magmom Map（magmom_map）
类型：`str`。默认：`''`。按元素指定磁矩幅值或方向，例如 `Fe:2.2, Ni:0.6`。

物理直觉：已有元素磁矩先验时使用；未知体系不要用它伪造不存在的元素差异。


#### Default Moment（default_moment）
类型：`float`。默认：`0.0`。为没有显式元素映射的原子提供默认磁矩幅值。

物理直觉：只适合作为兜底幅值；关键磁性元素应在 `magmom_map` 中显式给出。


#### Lift Scalar（lift_scalar）
类型：`bool`。默认：`True`。决定是否把标量磁矩提升为非共线向量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Axis（axis）
类型：`list[float] | tuple[float, float, float]`。默认：`(0.0, 0.0, 1.0)`。选择操作沿哪个空间轴或磁矩参考轴定义。

物理直觉：坐标轴参数会改变空间分层、吸附方向或磁矩方向；使用前先确认结构取向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。


#### Apply Elements（apply_elements）
类型：`str`。默认：`''`。限制只处理指定元素，留空表示处理所有元素。

物理直觉：用于只扰动磁性元素或只替换目标元素；留空代表全元素参与。


### 随机性和预算

#### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现的训练集生成或测试时打开；做最终大规模探索且希望保留随机多样性时可关闭。


#### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一 seed 应产生同一批候选；只有在 `use_seed` 打开时才改变结果。

生效条件：`use_seed=True`。


#### Max Outputs（max_outputs）
类型：`int`。默认：`100`。限制这张卡最多输出多少个结构。

物理直觉：这是防止链式卡片数量爆炸的预算阀；上游结构很多时应先按计算预算设上限。

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
