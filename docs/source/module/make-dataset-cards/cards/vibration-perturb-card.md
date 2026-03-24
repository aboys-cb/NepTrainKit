<!-- card-schema: {"card_name": "Vib Mode Perturb", "source_file": "src/NepTrainKit/ui/views/_card/vibration_perturb_card.py", "serialized_keys": ["distribution", "amplitude", "modes_per_sample", "min_frequency", "max_num", "scale_by_frequency", "exclude_near_zero", "use_seed", "seed"]} -->

# 振动模态扰动（Vib Mode Perturb）

`Group`: `Perturbation`  
`Class`: `VibrationModePerturbCard`  
`Source`: `src/NepTrainKit/ui/views/_card/vibration_perturb_card.py`

## 功能说明
沿振动模方向施加位移扰动（vibrational mode perturbation），比纯随机扰动更贴近动力学自由度。

它最适合的场景是：基于已有振动模态为结构添加模态相关位移，而不是纯随机位移。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\mathbf{r}'=\mathbf{r}+A\sum_{k\in\mathcal{K}} c_k\mathbf{u}_k$$
$$c_k\sim\mathcal{N}(0,1)\ \text{or}\ \mathcal{U}(-1,1),\quad c_k\leftarrow\frac{c_k}{\sqrt{|\omega_k|}}\ (\text{when scale\_by\_frequency=true})$$

## 操作示例
### 场景：基于已有振动模态为结构添加模态相关位移，而不是纯随机位移

**输入：** 一个已经携带振动模态和频率数组的结构；如果来自 `.xyz`，通常需要用 EXTXYZ 风格追加 `mode_1_x/y/z`、`frequency_1` 这类列

**目标：** 围绕指定频率窗口和模态数生成更接近振动空间的扰动样本

**参数设置：**
- `amplitude` 先从较小位移幅度开始
- `modes_per_sample` 控制每个样本叠加多少个模态
- `scale_by_frequency` 只在你明确要按频率缩放模态贡献时开启

**输出：** 多份沿振动模态方向偏移的结构；与纯随机扰动相比更依赖输入模态信息

**怎么验证结果合理：**
- 确认输入结构确实带有模态数组
- 检查低频或近零频模态是否按 `exclude_near_zero` 处理
- 若输出近乎不变，先检查 `amplitude` 和频率窗口

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 纯随机扰动不足以覆盖模态方向。
- 目标任务 (Target objective): 强化声子/振动相关结构覆盖。
- 建议添加条件 (Add-it trigger): 需要模态驱动的位移样本。
- 不建议添加条件 (Avoid trigger): 缺少可信振动模式输入。
> 物理提示 (Physics caution): 重点检查位移后最短键长和局部角度；幅度先小后大，比一次性追求大覆盖更稳妥。

## 输入前提
- 确认模态输入质量和单位一致。
- 先小 `amplitude` + 低 `modes_per_sample` 验证。
- 如果输入来自 `.xyz`，请使用 EXTXYZ 风格追加模态列；普通三列 XYZ 只有元素和坐标，不能提供这张卡所需的额外数组。

### 额外输入模板（振动模态数组）
这张卡除了控件参数外，还要求输入结构本身带有振动模态数据；如果缺少这些数组，程序会返回空结果并发出 warning。

最容易手写的是 EXTXYZ 按列拆开存储模式。下面这个 2 原子、1 个模态的最小模板可以直接参考：

```text
2
Properties=species:S:1:pos:R:3:mode_1_x:R:1:mode_1_y:R:1:mode_1_z:R:1:frequency_1:R:1 pbc="F F F"
H 0.000 0.000 0.000 0.10 0.00 0.00 1500.0
H 0.740 0.000 0.000 -0.10 0.00 0.00 1500.0
```

- `mode_1_x` / `mode_1_y` / `mode_1_z`: 第 1 个振动模态在每个原子上的位移分量。若有更多模态，继续追加 `mode_2_x/y/z`、`mode_3_x/y/z`。
- `frequency_1`: 第 1 个模态的频率。因为 EXTXYZ 是逐原子列，最简单的写法是每个原子都重复同一个频率值；`min_frequency` 会按这个单位直接筛选，不做自动单位换算。
- 如果你想写紧凑一点，也可以直接提供整块数组 `modes` 与 `frequencies`。单模态最小模板如下：

```text
2
Properties=species:S:1:pos:R:3:modes:R:3:frequencies:R:1 pbc="F F F"
H 0.000 0.000 0.000 0.10 0.00 0.00 1500.0
H 0.740 0.000 0.000 -0.10 0.00 0.00 1500.0
```

- `modes:R:3` 表示每个原子附带 3 个实数，也就是 1 个模态的 `(dx,dy,dz)`；如果有 2 个模态，就写成 `modes:R:6`，按 `mode1_xyz + mode2_xyz + ...` 顺序展开。
- `frequencies:R:1` 表示 1 个模态频率；如果有多个模态，就写成 `frequencies:R:N`，并在每个原子行重复同一组频率值。
- 代码当前接受的整块键名是 `vibration_modes` / `normal_modes` / `modes`，以及 `vibration_frequencies` / `normal_mode_frequencies` / `frequencies` / `freqs`。
- 不建议写 `mode:R:3` 这种单数键名；当前解析器不识别它。
- 列名写错或数组形状不对时，这张卡不会自动修复；它会把该结构视为“没有可用振动模态”。

## 参数说明（完整）
### `distribution` (Distribution)
- UI Label: `Distribution`
- 字段映射 (Field mapping): 序列化键 `distribution` <-> 界面标签 `Distribution`。
- 控件标签 (Caption): `Distribution`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `0`
- 含义 (Meaning): 采样分布类型 (distribution type)。
- 对输出规模/物理性的影响: 决定随机变量分布形状。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：均匀分布
  - 平衡：高斯分布
  - 探索：重尾分布仅探索

### `amplitude` (Amplitude)
- UI Label: `Amplitude`
- 字段映射 (Field mapping): 序列化键 `amplitude` <-> 界面标签 `Amplitude`。
- 控件标签 (Caption): `Amplitude`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.05]`
- 含义 (Meaning): 模态位移幅度 (mode displacement amplitude)。
- 对输出规模/物理性的影响: 主控振动扰动强度，过大易进入高能异常区。
- 物理直觉 / 典型值: 它通常是控制变化幅度的主旋钮；先从能看清趋势的小幅度起步，再决定是否扩到探索档。
- 推荐范围 (Recommended range):
  - 保守：0.01-0.03
  - 平衡：0.04-0.07
  - 探索：0.1+ 需后筛

### `modes_per_sample` (Modes Per Sample)
- UI Label: `Modes Per Sample`
- 字段映射 (Field mapping): 序列化键 `modes_per_sample` <-> 界面标签 `Modes Per Sample`。
- 控件标签 (Caption): `Modes Per Sample`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[2]`
- 含义 (Meaning): 每样本叠加模态数 (modes per sample)。
- 对输出规模/物理性的影响: 模态数越高，扰动方向组合越复杂。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：1-2
  - 平衡：3-4
  - 探索：5+

### `min_frequency` (Min Frequency)
- UI Label: `Min Frequency`
- 字段映射 (Field mapping): 序列化键 `min_frequency` <-> 界面标签 `Min Frequency`。
- 控件标签 (Caption): `Min Frequency`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[10.0]`
- 含义 (Meaning): 最小频率阈值 (minimum frequency)。
- 对输出规模/物理性的影响: 过滤过低频模式，减少软模异常。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：5-10
  - 平衡：10-20
  - 探索：20-50

### `max_num` (Max Num)
- UI Label: `Max Num`
- 字段映射 (Field mapping): 序列化键 `max_num` <-> 界面标签 `Max Num`。
- 控件标签 (Caption): `Max Num`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[32]`
- 含义 (Meaning): 每帧最大样本数 (max samples per frame)。
- 对输出规模/物理性的影响: 控制输出规模。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：10-20
  - 平衡：20-60
  - 探索：100+

### `scale_by_frequency` (Scale By Frequency)
- UI Label: `Scale By Frequency`
- 字段映射 (Field mapping): 序列化键 `scale_by_frequency` <-> 界面标签 `Scale By Frequency`。
- 控件标签 (Caption): `Scale By Frequency`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 按频率缩放开关 (scale by frequency)。
- 对输出规模/物理性的影响: 开启后高频模位移更小，通常更物理。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Scale By Frequency` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `exclude_near_zero` (Exclude Near Zero)
- UI Label: `Exclude Near Zero`
- 字段映射 (Field mapping): 序列化键 `exclude_near_zero` <-> 界面标签 `Exclude Near Zero`。
- 控件标签 (Caption): `Exclude Near Zero`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 排除近零频开关 (exclude near-zero)。
- 对输出规模/物理性的影响: 减少平移/旋转伪模引起的异常位移。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Exclude Near Zero` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `use_seed` (Use Seed)
- UI Label: `Use Seed`
- 字段映射 (Field mapping): 序列化键 `use_seed` <-> 界面标签 `Use Seed`。
- 控件标签 (Caption): `Use Seed`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 是否启用固定随机种子 (deterministic seed switch)。
- 对输出规模/物理性的影响: 开启后可复现实验；关闭后每次采样分布会变化。
- 怎么判断该开还是该关: 做可复现实验或要对比不同参数时开启；纯探索阶段可以先关闭以增加随机覆盖。
- 配置建议 (Practical note):
  - 开启：需要可复现对比时开启。
  - 关闭：探索阶段可关闭以增加随机覆盖。

### `seed` (Seed)
- UI Label: `Seed`
- 字段映射 (Field mapping): 序列化键 `seed` <-> 界面标签 `Seed`。
- 控件标签 (Caption): `Seed`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[0]`
- 含义 (Meaning): 随机种子值 (random seed value)。
- 对输出规模/物理性的影响: 只影响随机路径，不改变物理模型本身。
- 参数联动 / 生效条件: `seed` 只有在 `use_seed=true` 时才真正固定随机路径。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：0（随机）
  - 平衡：1-99（可复现）
  - 探索：100-9999（多 seed 对比）

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "distribution": 0,
  "amplitude": [
    0.02
  ],
  "modes_per_sample": [
    2
  ],
  "min_frequency": [
    0.0
  ],
  "max_num": [
    20
  ],
  "scale_by_frequency": true,
  "exclude_near_zero": true,
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "distribution": 0,
  "amplitude": [
    0.05
  ],
  "modes_per_sample": [
    3
  ],
  "min_frequency": [
    0.0
  ],
  "max_num": [
    20
  ],
  "scale_by_frequency": true,
  "exclude_near_zero": true,
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "distribution": 0,
  "amplitude": [
    0.12
  ],
  "modes_per_sample": [
    4
  ],
  "min_frequency": [
    0.0
  ],
  "max_num": [
    20
  ],
  "scale_by_frequency": true,
  "exclude_near_zero": true,
  "use_seed": true,
  "seed": [
    0
  ]
}
```

## 推荐组合
- Vib Mode Perturb -> Lattice Perturb: 在模态位移后追加轻量晶胞变化。
- 与晶格类卡片串联时，先做晶格变化，再补局部位移噪声。
- 大批量生成后可在流程末端接 `FPS Filter` 去掉重复样本。

## 常见问题与排查
- 输出为空时，优先检查输入是否真的带有可识别的模态字段，例如 `mode_1_x/y/z` 与 `frequency_1`，或 `vibration_modes` 与 `vibration_frequencies`；列名写错、数组形状不对时这张卡会直接返回空结果。
- 如果出现短键、断键或明显高能构型，先降低主控位移幅度，再缩小每帧样本数做抽样检查。
- 随机种子只控制采样路径，不会自动修正非物理参数；参数过激时程序仍会按当前配置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Vib(a={...},m={...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
