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

`mode`：string，默认 `Cone around reference`。`Cone around reference` 在原始磁矩方向周围做相关 cone disorder；`Full random directions` 直接用相关随机场方向覆盖整球。

`correlation_kernel`：string，默认 `exponential`。可选 `exponential` 或 `squared_exponential`。默认指数核 `exp(-r/xi)` 更接近常见自旋相关长度直觉；平方指数核更平滑、短程衰减更快。

`correlation_length`：float，默认 `3.0`。空间相关长度 `xi`，单位 Angstrom，必须大于 0。

`samples`：int，默认 `1`。每个输入结构生成多少个独立相关随机自旋场。

`cone_angle`：float，默认 `30.0`。`mode=Cone around reference` 时使用，单位 degree。

`magnitude_source`：string，默认 `Existing initial magmoms`。可选 `Existing initial magmoms` 或 `Map/default magnitude`。

`magmom_map`：string，默认 `""`。`magnitude_source=Map/default magnitude` 时使用，例如 `Fe:2.2,Co:1.7`。

`default_moment`：float，默认 `0.0`。元素不在 `magmom_map` 中时使用的磁矩模长。

`lift_scalar`：bool，默认 `True`。输入是一维标量磁矩时，是否沿 `axis` 提升为三维向量。

`axis`：三维 float list，默认 `(0.0, 0.0, 1.0)`。用于标量磁矩提升和 map/default 参考方向。

`apply_elements`：string，默认 `""`。只对列出的元素施加相关随机化；空字符串表示所有非零磁矩原子。

`max_atoms_for_full`：int，默认 `200`。精确 full-covariance 采样的 eligible 原子数上限。超过后直接失败，不隐式切换近似算法。

`use_seed`：bool，默认 `False`。开启后相关随机场可复现。

`seed`：int，默认 `0`。`use_seed=True` 时作为基础 seed。

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
