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

`int`，默认 0。模态系数的采样分布。`0` = 高斯（均值 0，大幅位移概率更低，更接近热振动统计）；`1` = 均匀。通常高斯更物理——你不希望模型被极端大位移的训练点带偏。

### Amplitude（amplitude）

`float`，默认 `0.05`。模态叠加的总幅值系数，实际位移 = `amplitude * sum(coefficient * mode_vector)`。注意这个幅值是乘在归一化后的模态矢量上的——如果上游的模态归一化方式不同（mass-weighted vs. 不归一化），同样的 amplitude 产生的实际原子位移也不一样。0.01~0.03 适合微扰验证；0.05~0.08 常规补充；设到 0.1 以上建议后筛检查最近邻距离。

### Modes Per Sample（modes_per_sample）

`int`，默认 2。每个输出结构随机选几个模态来叠加。1~2 个适合单模态方向验证，能清楚看到每个模态的贡献；3~4 个做多模态混合覆盖；5 个以上单个模态的贡献会被稀释，而且组合数指数增长——通常 2~4 就够了。

### Min Frequency（min_frequency）

`float`，默认 `10.0`。频率低于这个值的模态不参与采样。仅在 `exclude_near_zero=true` 时生效。设 0 则不过滤。典型值：排除平移/旋转伪模设 0.1~1.0 THz；排除软模设 1~10 THz，具体看你的体系。

### Max Num（max_num）

`int`，默认 32。每个输入结构生成多少个扰动样本。20~40 常规覆盖，50~100 高密度覆盖。建议后面接一张 `FPS Filter` 去重。

### Scale BY Frequency（scale_by_frequency）

`bool`，默认 `true`。打开后每个模态系数除以 sqrt(frequency)，高频模态的贡献自动衰减。这是物理上更合理的设置——同样的位移幅值，高频模态需要的能量远大于低频模。如果你在研究大位移的非谐效应，可以临时关掉，但注意关掉后高频模方向可能被过度采样，建议同时降低 `amplitude`。

### Exclude Near Zero（exclude_near_zero）

`bool`，默认 `true`。打开后用 `min_frequency` 做阈值过滤，排除近零频模态。一般建议保持打开——平移和旋转伪模的频率接近 0，沿这些"模态"做位移会产生异常大的整体漂移。

### Use Seed（use_seed）

`bool`，默认 `false`。打开后每次同一输入 + 同一参数 + 同一 seed 得到完全相同的扰动结果。对比实验或需要复现时打开，纯探索阶段可以关着。

### Seed（seed）

`int`，默认 0。随机种子值。仅在 `use_seed` 打开时生效。

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
