<!-- card-schema: {"card_name": "Vib Mode Perturb", "source_file": "src/NepTrainKit/ui/views/_card/vibration_perturb_card.py", "serialized_keys": ["params"]} -->

# 振动模态扰动（Vib Mode Perturb）

`Group`: `Perturbation` | `Class`: `VibrationModePerturbCard`

## 功能说明

沿声子振动模态方向施加位移扰动，比纯随机扰动更贴近动力学自由度。每个输出结构随机选取若干振动模态，按幅值叠加位移。可选按频率缩放（高频模态贡献更小）和排除近零频模态（避免平移/旋转伪模）。

$$\mathbf{r}'=\mathbf{r}+A\sum_{k\in\mathcal{K}} c_k\mathbf{u}_k,\quad c_k\sim\mathcal{N}(0,1)\ \text{or}\ \mathcal{U}(-1,1)$$

**关键前置条件：** 输入结构必须携带振动模态数据。如果来自 EXTXYZ 文件，需要包含 `mode_N_x/y/z` 和 `frequency_N` 列，或者整块数组 `modes` / `vibration_modes` 和 `frequencies` / `vibration_frequencies`。没有模态数据时，卡片返回空结果。

## 操作示例

### 场景：加了 Atomic Perturb 后力 MAE 改善有限，低频声子支的预测仍然很差

你在 Si 上训练了一个 NEP 模型。加了 `Atomic Perturb`（0.2A 随机位移）后，力的 MAE 从 150 meV/A 降到了 80 meV/A，但计算出的声子色散在低频声学支上误差仍然很大。诊断发现：纯随机扰动在所有方向上均匀采样，但低频声子对应的是长波长的协同原子运动——纯随机扰动几乎不可能恰好对齐这些方向。

**诊断思路：** 声子频率对位移的灵敏度在各个模态方向上差异巨大。低频声学模涉及大范围原子协同位移，但在局部键长上变化很小——纯随机扰动倾向于产生"局域键长变化大"的构型，这恰好是高频光学模的特征，不是低频模的特征。需要沿准确的振动模本征方向施加位移，让模型在模态坐标上也有训练点。

**输入：** 一个 Si 超胞，EXTXYZ 文件已包含 phonopy 计算出的振动模态数据（`mode_1_x/y/z` 到 `mode_N_x/y/z` 和 `frequency_1` 到 `frequency_N`）

**目标：** 生成 32 个扰动结构，每次随机选 3 个模态叠加，幅值 0.05，只使用频率 > 10 THz 的模态（或 < 10 的具体看体系）

**参数设置：**
- `amplitude` = `0.05`
- `modes_per_sample` = `3`
- `max_num` = `32`
- `min_frequency` = `0.0` （或设一个合理的过滤值）
- `distribution` = `0` （高斯分布）
- `scale_by_frequency` = `true` （高频模自动衰减）

**输出：** 32 个结构，每个沿 3 个随机选取的振动模态方向偏移，位移幅值按频率缩放

**怎么验证训练集质量改善：**
- 重训后重新计算声子色散，低频声学支应更接近 DFT 结果
- 检查输出中哪些模态被选中——如果低频模态总是被 `min_frequency` 排除，降低此阈值
- 如果模型对高频光学支仍然不准，增大 `modes_per_sample` 到 4~5，或增大 `max_num`
- 如果 `scale_by_frequency` 导致高频模位移太小（几乎看不到变化），关闭此开关试一批

### 什么时候加这张卡、什么时候不加

**加：**
- 已有可信的振动模态数据，需要沿模态方向补充训练样本
- 纯随机扰动改进有限，声子或振动相关性质仍有偏差
- 需要模型学到模态方向上的势能面曲率

**不加：**
- 输入结构没有模态数据——卡片直接返回空，先跑 phonopy/DFPT 生成模态
- 模态是从低精度方法（如经验势）算出来的——沿错误模态方向扰动只会引入噪声
- 只需要覆盖高频局域环境——用 `Atomic Perturb` 更简单直接

### 输入格式：振动模态数据

输入结构需要以下数组之一（EXTXYZ 格式优先）：

**方式一：按模态拆列（推荐，手写最友好）**
```
2
Properties=species:S:1:pos:R:3:mode_1_x:R:1:mode_1_y:R:1:mode_1_z:R:1:frequency_1:R:1 pbc="F F F"
H 0.000 0.000 0.000 0.10 0.00 0.00 1500.0
H 0.740 0.000 0.000 -0.10 0.00 0.00 1500.0
```
`mode_N_x/y/z`：第 N 个模态在每个原子上的位移分量。`frequency_N`：第 N 个模态的频率，每个原子行重复同一值。

**方式二：整块数组（适合多模态）**
```
2
Properties=species:S:1:pos:R:3:modes:R:3N:frequencies:R:N pbc="F F F"
```
`modes:R:3N` 表示 N 个模态的位移分量，按 mode1_xyz + mode2_xyz + ... 顺序。`frequencies:R:N` 表示 N 个频率值，每行重复同一组。

代码接受的键名：`vibration_modes` / `normal_modes` / `modes`，以及 `vibration_frequencies` / `normal_mode_frequencies` / `frequencies` / `freqs`。列名或形状不对时，卡片将该结构视为"没有可用模态"并跳过。

## 参数说明


### Distribution（distribution）

类型：`int`。默认：`0`。选择振动模态采样分布。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |

### Amplitude（amplitude）

类型：`float`。默认：`0.05`。设置振动模态位移幅值。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Modes Per Sample（modes_per_sample）

类型：`int`。默认：`2`。设置每个样本叠加的振动模态数量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Min Frequency（min_frequency）

类型：`float`。默认：`10.0`。设置参与采样的最低频率。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Max Num（max_num）

类型：`int`。默认：`32`。设置扰动或采样结构的输出数量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Scale BY Frequency（scale_by_frequency）

类型：`bool`。默认：`True`。决定是否按频率缩放模态位移。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Exclude Near Zero（exclude_near_zero）

类型：`bool`。默认：`True`。决定是否排除近零频模态。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Use Seed（use_seed）

类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现的训练集生成或测试时打开；做最终大规模探索且希望保留随机多样性时可关闭。

### Seed（seed）

类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一 seed 应产生同一批候选；只有在 `use_seed` 打开时才改变结果。

生效条件：`use_seed=True`。

## 推荐预设

### 微扰验证（幅值 0.02，2 模态，20 样本，高斯）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "params": {
    "distribution": 0,
    "amplitude": 0.02,
    "modes_per_sample": 2,
    "min_frequency": 0.1,
    "max_num": 20,
    "scale_by_frequency": true,
    "exclude_near_zero": true,
    "use_seed": true,
    "seed": 42
  }
}
```

### 常规声子覆盖（幅值 0.05，3 模态，32 样本，高斯）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "params": {
    "distribution": 0,
    "amplitude": 0.05,
    "modes_per_sample": 3,
    "min_frequency": 0.1,
    "max_num": 32,
    "scale_by_frequency": true,
    "exclude_near_zero": true,
    "use_seed": true,
    "seed": 42
  }
}
```

### 大位移非谐探索（幅值 0.1，5 模态，50 样本，不缩放）
```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "params": {
    "distribution": 1,
    "amplitude": 0.1,
    "modes_per_sample": 5,
    "min_frequency": 0.0,
    "max_num": 50,
    "scale_by_frequency": false,
    "exclude_near_zero": false,
    "use_seed": true,
    "seed": 42
  }
}
```

## 推荐组合

- `Vib Mode Perturb` -> `Atomic Perturb`：先沿模态方向扰动，再补充纯随机方向
- `phonopy/DFPT` 生成模态 -> `Vib Mode Perturb`：计算完模态后直接喂入生成训练数据
- `Vib Mode Perturb` -> `FPS Filter`：大批量生成后去重

## 常见问题

**输出为空（无结构生成）。** 输入结构没有可识别的模态数据。检查：列名是否正确（`mode_1_x` 不是 `mode1_x`），频率列是否存在且命名正确，数组形状是否和原子数一致。可以用 ASE 读取 EXTXYZ 文件后查看 `arrays` 确认键名。

**部分模态始终没被选中。** 这是随机采样的正常现象。`modes_per_sample` 较小时（1~2），样本数不够覆盖所有模态。增大 `max_num` 或 `modes_per_sample` 提高覆盖率。

**关闭 `scale_by_frequency` 后高频模位移过大。** 这是预期行为——`1/sqrt(frequency)` 的缩放会显著衰减高频贡献。需要研究高频非谐效应时关闭，但注意高频模方向上的大幅位移很容易进入非物理区。建议关闭后降低 `amplitude` 补偿。

**`min_frequency` 过滤后模态太少。** 如果体系存在大量软模（近零频），高阶阈值会把模态池缩到很小。先检查软模是否物理（有可能是没弛豫好），然后适当降低 `min_frequency` 或关闭 `exclude_near_zero`。

## 输出标签

`Vib(a={amplitude},m={modes_per_sample})`

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。模态选择、系数采样均受 seed 控制。
