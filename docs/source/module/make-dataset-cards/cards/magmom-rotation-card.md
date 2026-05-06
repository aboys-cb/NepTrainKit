<!-- card-schema: {"card_name": "Magmom Rotation", "source_file": "src/NepTrainKit/ui/views/_card/magmom_rotation_card.py", "serialized_keys": ["params"]} -->

# 磁矩旋转（Magmom Rotation）

`Group`: `Perturbation` | `Class`: `MagneticMomentRotationCard`

## 功能说明

对已存在的磁矩做随机小角度旋转，并可选地扰动模长。每个输入结构输出多个旋转版本，用于补充自旋方向附近的连续构型邻域数据。

$$\mathbf{m}'=\lambda\,\mathbf{R}(\hat{\mathbf{n}},\theta)\,\mathbf{m},\quad \lambda\in[f_{\min},f_{\max}]$$

**关键限制：** 这张卡只做随机旋转和模长缩放，不改变晶体结构，不定义新的磁序。它依赖输入已有 `initial_magmoms`。

## 操作示例

### 场景：模型在 MD 中遇到自旋方向涨落就崩

你训练了一个 NEP 磁性模型，所有训练数据里磁矩方向都是严格对齐的（FM 全同向、AFM 严格翻转）。静态推理表现完美，但一跑有限温度 MD——原子磁矩受热涨落发生 5~10 度的小角度偏转——力的误差直接翻倍。

**诊断思路：** 训练集只覆盖了磁矩方向空间的离散点，模型在这些点之间只能外推。需要在每个参考磁态周围加一批小角度旋转样本，让模型"见过"自旋在参考方向附近连续变化的环境。

**输入：** 一个已写入 `initial_magmoms` 的磁性结构（来自 `Magnetic Order` 或 `Set Magnetic Moments` 的上游输出）

**目标：** 对每个输入帧生成 5 个旋转版本，最大旋转角 10 度，模长扰动 ±5%，覆盖自旋方向附近的连续邻域

**参数设置：**
- `Max Angle` = `10.0`
- `Num Structures` = `5`
- `Disturb Magnitude` = 勾选，`Magnitude Factor` = `[0.95, 1.05]`

**输出：** 每输入帧 5 个结构，磁矩方向在 0~10 度范围内随机偏离，模长在 95%~105% 之间浮动。原子位置不变。

**怎么验证训练集质量改善：**
- 重训后跑有限温度 MD，检查力的 MAE 是否比之前更稳定
- 如果仍有方向敏感性问题，增大 `Max Angle` 到 15~20 度，或增加 `Num Structures`
- 如果特定元素的磁矩扰动后偏离物理合理范围，用 `Elements` 限制只扰动目标元素

### 什么时候加这张卡、什么时候不加

**加：**
- 训练集只有固定方向的磁态，模型对方向涨落敏感
- 做有限温度 MD 或自旋动力学，需要连续的方向覆盖
- 上游已有可信磁矩，只需在附近做小幅度采样

**不加：**
- 输入没有初始磁矩 → 先用 `Set Magnetic Moments` 或 `Magnetic Order`
- 需要方向大范围覆盖（40°+）→ 用 `Small-Angle Spin Tilt` 确定性扫描
- 只需要模长扰动不需要方向变化 → 这张卡的旋转功能白费了

## 参数说明

### 核心控制

**`Elements`**（elements）：逗号分隔的元素列表，如 `Fe,Co`。只对列出的元素做旋转和模长扰动。留空 = 全部磁性原子参与。

**`Max Angle`**（max_angle）：最大旋转角，单位度。每次随机采样角度在 [0, max_angle] 内均匀分布。
- 保守：2~5°（验证方向变化确实有帮助）
- 平衡：8~15°（常规有限温度覆盖）
- 探索：20°+（宽温度区间，需重点抽查）

**`Num Structures`**（num_structures）：每输入帧输出的旋转版本数。5~10 用于轻量补样，10~30 用于常规覆盖，30+ 建议后接过滤。

### 标量抬升

**`Lift Scalar`**（lift_scalar）：输入是共线标量磁矩时，是否先沿 `Axis` 抬升为向量再做旋转。输入是标量且需要旋转时必须开启。

### 参考轴

**`Axis`**（axis）：`lift_scalar` 抬升时使用的参考方向，默认 `[0, 0, 1]`。输入已经是向量磁矩时此参数对初始方向无影响。

### 模长扰动

**`Disturb Magnitude`**（disturb_magnitude）：是否在旋转的同时随机缩放磁矩模长。

**`Magnitude Factor`**（magnitude_factor）：`[min, max]`，模长缩放因子区间。
- 保守：`[0.98, 1.02]`
- 平衡：`[0.95, 1.05]`
- 探索：`[0.85, 1.15]`

### 随机性

**`Use Seed`**（use_seed）：勾选 → 固定种子可复现。

**`Seed`**（seed）：种子值。仅 `use_seed` 勾选时生效。

## 推荐预设

### 方向微扰验证（5 个输出/帧，适合初步诊断）
```json
{
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "elements": "",
  "max_angle": [5.0],
  "num_structures": [5],
  "lift_scalar": true,
  "axis": [0.0, 0.0, 1.0],
  "disturb_magnitude": false,
  "magnitude_factor": [0.95, 1.05],
  "use_seed": false,
  "seed": [0]
}
```

### 方向 + 模长常规覆盖（10 个输出/帧，适合有限温度 MD）
```json
{
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "elements": "",
  "max_angle": [10.0],
  "num_structures": [10],
  "lift_scalar": true,
  "axis": [0.0, 0.0, 1.0],
  "disturb_magnitude": true,
  "magnitude_factor": [0.95, 1.05],
  "use_seed": true,
  "seed": [42]
}
```

### 宽温区探索（30 个输出/帧，可复现）
```json
{
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "elements": "Fe,Co",
  "max_angle": [25.0],
  "num_structures": [30],
  "lift_scalar": true,
  "axis": [0.0, 0.0, 1.0],
  "disturb_magnitude": true,
  "magnitude_factor": [0.85, 1.15],
  "use_seed": true,
  "seed": [42]
}
```

## 推荐组合

- `Magnetic Order` → `Magmom Rotation`：先生成 FM/AFM 参考态，再补方向邻域
- `Set Magnetic Moments` → `Magmom Rotation`：先统一磁矩格式，再旋转
- `Magmom Rotation` → `FPS Filter`：生成大量版本后用 FPS 筛选代表性样本

## 常见问题

**输出和输入一样。** 检查输入是否有 `initial_magmoms`。标量磁矩需开 `Lift Scalar`。`Max Angle` 不能为 0。

**输出磁矩全是零。** `Elements` 过滤掉了所有有磁矩的元素，或 `Magnitude Factor` 下限为 0 且被采样到。

**每次运行结果不同。** 未开启 `Use Seed`。需要可复现时勾选并固定 `Seed`。

## 输出标签

- `MMR(a=10.0,s=0.95-1.05)`：向量旋转 + 模长扰动同时开启
- `MMS(s=0.95-1.05)`：仅模长缩放（旋转角度为 0 或无向量输入）

所有输出写入 `initial_magmoms` 数组。

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。每次旋转的随机轴和角度由 seed + 结构序号联合控制。
