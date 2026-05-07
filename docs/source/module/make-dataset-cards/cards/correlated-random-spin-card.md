<!-- card-schema: {"card_name": "Correlated Random Spin", "source_file": "src/NepTrainKit/ui/views/_card/correlated_random_spin_card.py", "serialized_keys": ["params"]} -->

# Correlated Random Spin

`Group`: `Magnetism` | `Class`: `CorrelatedRandomSpinCard`

## 功能说明

`Correlated Random Spin` 从已有磁矩或元素磁矩表出发，生成带空间相关长度的非共线随机磁矩。它保持每个原子的磁矩模长，只改变方向；相关性由距离核函数和 `correlation_length` 控制。

这张卡不叫 `Spin Glass`，因为它没有引入交换哈密顿量、frustration 约束或动力学冷却过程。它只表达一个清楚的训练集状态：空间相关的随机非共线自旋场。

## 操作示例

### 场景：训练集有 PM 随机态，但缺少有限相关长度的非共线态

模型在完全随机 PM 和有序 FM/AFM 上都能跑，但在短程相关的高温磁态上磁力误差明显偏大。用 `Correlated Random Spin` 可以生成相关长度为 3 Angstrom 的非共线扰动，让相邻原子磁矩方向更相近，远距离方向逐渐失相关。

参数设置：`mode=Cone around reference`，`correlation_kernel=exponential`，`correlation_length=3.0`，`samples=5`，`cone_angle=30.0`，`magnitude_source=Existing initial magmoms`，`max_atoms_for_full=200`。

输出结构的 `initial_magmoms` 为 `(N, 3)`，`Config_type` 追加 `CorrSpin(...)`。检查时看磁矩模长是否保持、cone 角度是否符合设定，以及不同 `correlation_length` 下相邻自旋相似度是否变化。

## 参数说明

### 相关随机场

#### Mode（mode）
类型：`str`。默认：`'Cone around reference'`。选择相关随机场围绕参考磁矩生成，还是生成 PM-like 随机场。

物理直觉：`Cone around reference` 保留有序态方向，只加入有限温非共线涨落；`PM-like random field` 不再围绕参考方向，适合补顺磁附近但带空间相关长度的样本。

#### Correlation Kernel（correlation_kernel）
类型：`str`。默认：`'exponential'`。选择自旋空间相关函数。

物理直觉：`exponential` 衰减慢，保留更长尾的自旋相关；`squared_exponential` 更平滑但远距离衰减更快。有限相关长度磁无序优先 exponential。

#### Correlation Length（correlation_length）
类型：`float`。默认：`3.0`。设置自旋随机场的空间相关长度。

物理直觉：小于最近邻距离时近似独立随机；几倍晶格常数会形成平滑磁畴。用于有限温磁无序时不要直接跳到无限长相关。

#### Max Atoms For Full（max_atoms_for_full）
类型：`int`。默认：`200`。限制 full covariance 方法允许处理的最大原子数。

物理直觉：full covariance 需要 O(N^3) 分解。小体系可保持默认，大于几百个磁性原子时应先缩小结构或等待近似方法，而不是盲目调大。

### 输出和角度

#### Samples（samples）
类型：`int`。默认：`1`。设置每个输入结构生成的样本数量。

物理直觉：每个相关长度生成多少个随机场。1-3 个用于覆盖趋势，5+ 才开始给同一相关长度做统计。

#### Cone Angle（cone_angle）
类型：`float`。默认：`30.0`。设置非共线 cone disorder 的最大偏转角。

物理直觉：只在 cone 模式下限制偏离参考方向的角度。10-30° 是有序态附近扰动，60° 以上更接近强非共线无序。

生效条件：`mode` 或方向模型选择 cone/noncollinear 随机化时。

### 磁矩幅值

#### Magnitude Source（magnitude_source）
类型：`str`。默认：`'Existing initial magmoms'`。选择磁矩幅值来自元素映射、默认值还是已有结构。

物理直觉：已有 `initial_magmoms` 时复用它最安全；没有磁矩输入时用 `magmom_map`/`default_moment` 构造幅值。不要用默认幅值替代已知元素磁矩。

#### Magmom Map（magmom_map）
类型：`str`。默认：`''`。按元素指定磁矩幅值或方向，例如 `Fe:2.2, Ni:0.6`。

物理直觉：已知元素局域磁矩时显式写入，例如 `Fe:2.2,Ni:0.6`。未知元素不要用默认值伪造先验。

#### Default Moment（default_moment）
类型：`float`。默认：`0.0`。为没有显式元素映射的原子提供默认磁矩幅值。

物理直觉：只作为 `magmom_map` 未命中的兜底幅值。关键磁性元素应显式列出，非磁元素通常保持 0。

#### Lift Scalar（lift_scalar）
类型：`bool`。默认：`True`。决定是否把标量磁矩提升为非共线向量。

物理直觉：输入是标量磁矩但下游需要非共线向量时打开；如果原始数据已有方向，不要重新提升覆盖方向信息。

#### Axis（axis）
类型：`list[float] | tuple[float, float, float]`。默认：`(0.0, 0.0, 1.0)`。设置 cone 模式下的参考磁矩方向。

物理直觉：这是方向参考，不是普通数值。改它会改变分层、表面法向或磁矩方向；使用前先确认 cell 取向和目标物理方向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

#### Apply Elements（apply_elements）
类型：`str`。默认：`''`。限制只处理指定元素，留空表示处理所有元素。

物理直觉：只对磁性元素施加相关随机方向。含非磁基底或轻元素时显式列出 Fe/Co/Ni/Mn 等目标元素，避免给无关原子写磁矩。

### 随机性

#### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现训练集、测试或对比实验时打开；最终大规模随机探索可以关闭，但结果不能逐帧复现。

#### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一输入、同一参数和同一 seed 应生成同一批候选；只有 `use_seed=True` 时改变它才会改变随机输出。

生效条件：`use_seed=True`。

## 推荐预设

### 有序态附近的有限温扰动

```json
{
  "class": "CorrelatedRandomSpinCard",
  "params": {
    "mode": "Cone around reference",
    "correlation_kernel": "exponential",
    "correlation_length": 3.0,
    "samples": 5,
    "cone_angle": 25.0,
    "magnitude_source": "Existing initial magmoms",
    "magmom_map": "",
    "default_moment": 0.0,
    "lift_scalar": true,
    "axis": [0.0, 0.0, 1.0],
    "apply_elements": "",
    "max_atoms_for_full": 200,
    "use_seed": true,
    "seed": 42
  }
}
```

### PM-like 相关随机方向

```json
{
  "class": "CorrelatedRandomSpinCard",
  "params": {
    "mode": "Full random directions",
    "correlation_kernel": "exponential",
    "correlation_length": 5.0,
    "samples": 3,
    "cone_angle": 30.0,
    "magnitude_source": "Map/default magnitude",
    "magmom_map": "Fe:2.2,Co:1.7",
    "default_moment": 0.0,
    "lift_scalar": true,
    "axis": [0.0, 0.0, 1.0],
    "apply_elements": "Fe,Co",
    "max_atoms_for_full": 200,
    "use_seed": true,
    "seed": 7
  }
}
```

## 推荐组合

- `Set Magnetic Moments -> Correlated Random Spin`：先统一磁矩模长来源，再生成相关非共线态。
- `Magnetic Order -> Correlated Random Spin`：从 FM/AFM 参考态生成有限温 cone disorder。
- `Spin Disorder -> Correlated Random Spin`：先做离散翻转比例，再对局部方向加空间相关扰动。

## 常见问题

**为什么超过 `max_atoms_for_full` 直接失败？** 当前实现是精确协方差采样，需要 full matrix 和 Cholesky，复杂度随原子数快速增长。自动切换近似算法会让同一张卡在不同体系大小下改变物理语义，所以 v1 明确失败。

**`exponential` 和 `squared_exponential` 怎么选？** 默认用 `exponential`。如果你明确希望更平滑的随机场，再选 `squared_exponential`。

**没有磁矩会怎样？** 输入没有可用 `initial_magmoms`，且没有通过 `magmom_map/default_moment` 提供非零模长时，卡片会失败，不会生成零磁矩伪结果。

## 输出标签

`CorrSpin(xi={correlation_length},ker={kernel},mode={cone|full},n={eligible_atoms},s={seed},a={cone_angle})`。`s` 只在 `use_seed=True` 时出现，`a` 只在 cone 模式出现。

## 可复现性

开启 `use_seed` 后，相关随机场由 `seed`、输入结构稳定 ID 和 sample 序号共同决定。相同输入、相同参数、相同 seed 会生成相同磁矩方向。
