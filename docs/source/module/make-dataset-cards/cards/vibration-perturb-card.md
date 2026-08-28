<!-- card-schema: {"card_name": "Vib Mode Perturb", "source_file": "src/NepTrainKit/ui/views/_card/vibration_perturb_card.py", "serialized_keys": ["params"]} -->

# 振动模式扰动（Vib Mode Perturb）

**分类：** 扰动

## 功能说明

沿输入结构已有的振动模式生成协同原子位移。它适合已有可信模态、希望在这些模态张成的子空间内补样的场景；没有模态数据时应使用“原子扰动”。

每个输入结构都必须带有可识别的模态数组。开启频率筛选或频率加权时，还必须为所有模态提供有限频率值；参与频率加权的频率不能为零。

## 输入数据

推荐把每个模式保存为一个三列向量属性：

```text
vibration_mode_0       # shape: (原子数, 3)
vibration_frequency_0  # shape: (原子数,)
vibration_mode_1       # shape: (原子数, 3)
vibration_frequency_1  # shape: (原子数,)
...
```

在 EXTXYZ 的 `Properties` 中，一个模式对应 `vibration_mode_0:R:3`，后面的 `3` 表示每个原子直接保存 x、y、z 三个分量。频率对应 `vibration_frequency_0:R:1`；频率属于整个模式，但需要按原子重复同一个值。

程序仍兼容拆分的 `vibration_mode_0_x/y/z`、`normal_mode_*`、`mode_*`，以及聚合模态数组 `vibration_modes`、`normal_modes`、`modes`。聚合频率数组可命名为 `vibration_frequencies`、`normal_mode_frequencies`、`frequencies` 或 `freqs`。

频率没有内置固定单位。频率截止值与输入频率使用同一数值单位；同一批数据必须保持一致，不能混用 `cm⁻¹` 与 THz。

### 一个完整的两原子示例

下面的 Si₂ 结构带有两个模式。每个模式都为每个原子保存一个三维矢量；这里假设频率统一使用 `cm⁻¹`：

| 原子 | 坐标（Å） | 模式 0 矢量 | 频率 0 | 模式 1 矢量 | 频率 1 |
|---|---|---|---:|---|---:|
| Si 0 | `(0.0, 0.0, 0.0)` | `(+0.10, 0.00, 0.00)` | 100 | `(0.00, +0.20, 0.00)` | 250 |
| Si 1 | `(2.3, 0.0, 0.0)` | `(-0.10, 0.00, 0.00)` | 100 | `(0.00, -0.20, 0.00)` | 250 |

对应的 ASE 结构可以这样构造：

```python
import numpy as np
from ase import Atoms

atoms = Atoms(
    "Si2",
    positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]],
    cell=[10.0, 10.0, 10.0],
    pbc=True,
)

# 模式 0：两个原子沿 x 方向相向运动
atoms.new_array(
    "vibration_mode_0",
    np.array([[+0.10, 0.00, 0.00], [-0.10, 0.00, 0.00]]),
)
atoms.new_array("vibration_frequency_0", np.array([100.0, 100.0]))

# 模式 1：两个原子沿 y 方向反向运动
atoms.new_array(
    "vibration_mode_1",
    np.array([[0.00, +0.20, 0.00], [0.00, -0.20, 0.00]]),
)
atoms.new_array("vibration_frequency_1", np.array([250.0, 250.0]))
```

以 `vibration_mode_0` 为例，第一行 `[0.10, 0.00, 0.00]` 属于 Si 0，第二行 `[-0.10, 0.00, 0.00]` 属于 Si 1。每行已经是该原子完整的三维模态矢量。

## 原理与公式

设筛选后共有 $M$ 个模式，第 $j$ 个模式为 $\mathbf e_j$，频率为 $\nu_j$。每个输出先无放回抽取 $K$ 个模式，再采样系数 $z_j$：

$$
z_j\sim\mathcal N(0,1)
\quad\text{或}\quad
z_j\sim U(-1,1).
$$

开启频率加权时：

$$
c_j=\frac{z_j}{\sqrt{|\nu_j|}};
$$

关闭时 $c_j=z_j$。最终坐标为：

$$
\mathbf R'=\mathbf R+a\sum_{j\in S}c_j\mathbf e_j,
$$

其中 $a$ 是“模态系数尺度”，$S$ 是本次选中的模式集合。周期方向上的新坐标会回到原晶胞内。

`a` 不是原子最大位移。实际位移还取决于模态矢量的归一化、抽到的随机系数、组合模式数和频率加权。正态分布无界，因此也不存在由 `a` 给出的硬位移上限。

## 操作示例

### 用已有模态生成近平衡样本

导入频率单位统一的模态结构，将系数尺度设为 `0.02`、每个样本组合 2 个模式、每个输入生成 8 个结构，并启用 seed。先检查这 8 个结构的最大位移和最短原子间距；尺度合适后再增加输出数。

## 参数说明

### 系数分布（distribution）

`int`，默认 `0`。`0` 为标准正态分布，系数无界；`1` 为 $[-1,1]$ 均匀分布。

### 模态系数尺度（amplitude）

`float`，默认 `0.05`。对应公式中的 $a$，直接乘在组合模态上，不代表最大原子位移。

### 每个样本组合模式数（modes_per_sample）

`int`，默认 `2`，至少为 1。每个输出无放回选择这么多个模式，不能超过筛选后的可用模式数。

### 绝对频率截止值（min_frequency）

`float`，默认 `10.0`。对应 $\nu_\min$，单位跟随输入频率；关闭频率截止后不生效。

### 每个输入生成（max_num）

`int`，默认 `32`，至少为 1。表示每个输入结构各自生成的输出数。

### 频率加权（scale_by_frequency）

`bool`，默认 `true`。开启后将系数除以 $\sqrt{|\nu|}$，相对增强低频模式。

### 频率截止（exclude_near_zero）

`bool`，默认 `true`。开启后仅保留 $|\nu|\ge\nu_\min$ 的模式。

### 使用随机种子（use_seed）

`bool`，默认 `false`。开启后，相同结构与参数可重复得到相同结果。

### 随机种子（seed）

`int`，默认 `0`。它与结构内容共同派生该结构的随机流。

总输出数为：

$$
N_\mathrm{out}=N_\mathrm{input}\times\texttt{max\_num}.
$$

## 使用步骤

1. 导入带振动模式的结构；
2. 确认频率单位一致，并据此设置频率截止值；
3. 先用较小的模态系数尺度和少量输出试算；
4. 检查位移幅度、最短原子间距和所覆盖的模式，再扩大样本量。

如果筛选后没有模式、频率相关选项缺少有限频率，或组合模式数超过可用数，卡片会停止并说明原因，不会返回成功的空结果。

## 常见问题

**为什么尺度是 0.05，最大位移却不是 0.05 Å？** 尺度乘在模态矢量及随机系数的组合上；位移还取决于模态归一化、组合模式数和频率加权。

**截止值 10 的单位是什么？** 与输入保存的频率单位相同。程序不自动判断或换算 `cm⁻¹` 与 THz。

**为什么提示可用模式不足？** 频率筛选后的模式数小于“每个样本组合模式数”。降低组合数或调整截止值。

## 输出与复现

- 每个输入严格生成 `max_num` 个结构；
- `Config_type` 追加 `Vib(a=<尺度>,m=<模式数>)`；
- `vibration_mode_perturb` metadata 记录参数、候选模式数、所选模式位置、对应频率、seed 和样本序号；
- 相同结构、参数和 seed 得到相同结果；不同结构从同一基准 seed 派生不同随机流。

metadata 中的模式位置是“筛选后候选模式列表”的零基序号。

<details>
<summary>示例配置</summary>

```json
{
  "class": "VibrationModePerturbCard",
  "check_state": true,
  "params": {
    "distribution": 0,
    "amplitude": 0.05,
    "modes_per_sample": 2,
    "min_frequency": 10.0,
    "max_num": 32,
    "scale_by_frequency": true,
    "exclude_near_zero": true,
    "use_seed": true,
    "seed": 42
  }
}
```

</details>
