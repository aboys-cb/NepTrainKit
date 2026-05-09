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
`str`，默认 `'Cone around reference'`。`Cone around reference` 保留有序态的方向，只叠加有限温非共线涨落；`PM-like random field` 不再围绕参考方向，适合补 PM 附近但带空间相关长度的样本。

#### Correlation Kernel（correlation_kernel）
`str`，默认 `'exponential'`。自旋空间相关函数。`exponential` 衰减更慢，保留更长尾的自旋关联；`squared_exponential` 更平滑但远距离衰减更快。有限相关长度磁无序优先选 exponential。

#### Correlation Length（correlation_length）
`float`，默认 3.0 A。自旋随机场的空间相关长度。设得比最近邻距离还小 → 近似独立随机；放大到几倍晶格常数 → 形成平滑磁畴。做有限温磁无序时不要一上来就跳到无限长相关。

#### Max Atoms For Full（max_atoms_for_full）
`int`，默认 200。full covariance 方法能处理的最大原子数，因为需要 O(N³) 的 Cholesky 分解。小体系保持默认即可；超过几百个磁性原子时先缩小结构，不要盲目调大这个值。

### 输出和角度

#### Samples（samples）
`int`，默认 1。每个相关长度下生成几个随机场样本。1~3 个覆盖趋势，5 个以上才开始给同一相关长度做统计。

#### Cone Angle（cone_angle）
`float`，默认 30.0°。cone 模式下偏离参考方向的最大角度。10~30° 是有序态附近的扰动，60° 以上更接近强非共线无序。

生效条件：`mode` 为 cone 模式时。

### 磁矩幅值

#### Magnitude Source（magnitude_source）
`str`，默认 `'Existing initial magmoms'`。磁矩幅值的来源。有 `initial_magmoms` 时优先复用，最安全；没有磁矩输入时用 `magmom_map` 和 `default_moment` 构造。不要用默认幅值替代已知元素的实测磁矩。

#### Magmom Map（magmom_map）
`str`，默认空。按元素显式指定磁矩幅值，格式如 `Fe:2.2, Ni:0.6`。已知元素局域磁矩就填进去，未知元素别用默认值伪造先验。

#### Default Moment（default_moment）
`float`，默认 0.0。作为 `magmom_map` 没命中的元素的兜底幅值。关键磁性元素应显式列在 magmom_map 里，非磁元素通常保持 0。

#### Lift Scalar（lift_scalar）
`bool`，默认 true。把标量磁矩提升为非共线向量。输入是标量但下游需要向量时打开；如果原始数据本身已有方向信息，不要重新提升覆盖它。

#### Axis（axis）
`list[float] | tuple[float, float, float]`，默认 `(0.0, 0.0, 1.0)`。cone 模式下的参考磁矩方向。这不是普通数值——改它会影响分层、表面法向或磁矩取向；使用前先确认 cell 取向和目标物理方向。

生效条件：涉及方向、分层、表面或向量初始化的模式。

#### Apply Elements（apply_elements）
`str`，默认空（全部生效）。逗号分隔限定只处理指定元素。含非磁基底时显式列出 Fe/Co/Ni/Mn 等磁性元素，避免给无关原子写磁矩。

### 随机性

#### Use Seed（use_seed）
`bool`，默认 false。需要可复现训练集、测试或对比实验时打开；最终大规模随机探索可以关，但结果不能逐帧复现。

#### Seed（seed）
`int`，默认 0。固定随机种子值。相同输入、相同参数和相同 seed 生成同一批候选；只有 `use_seed=True` 时改它才会改变输出。

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
