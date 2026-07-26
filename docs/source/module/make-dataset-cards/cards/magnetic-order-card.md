<!-- card-schema: {"card_name": "Magnetic Order", "source_file": "src/NepTrainKit/ui/views/_card/magnetic_order_card.py", "serialized_keys": ["params"]} -->

# 磁序（Magnetic Order）

`Group`: `Magnetism` | `Class`: `MagneticOrderCard`

## 功能说明

根据元素磁矩幅值，从每个输入结构生成 FM、AFM 和随机 PM 初始自旋构型。卡片不改变原子坐标、元素或晶胞；输出统一写入三分量 `spin:R:3`，并同步维护 ASE `initial_magmoms`。

这张卡生成的是用于 DFT/NEP 数据准备的**初始自旋端点**，不是磁基态判断器，也不能证明某个输出是真正的热力学 FM、AFM 或 PM 相。后续仍需通过电子结构计算、能量比较和收敛后磁矩验证。

`Magnetic Order` 与 `Spin Disorder` 的职责不同：

- `Magnetic Order`：建立 FM、AFM 或完全随机 PM 端点。
- `Spin Disorder`：从已有磁态出发，扫描局部翻转比例、随机化比例或锥角无序等中间状态。

## 操作示例

### 场景：BCC Fe 模型只见过 FM，AFM 测试误差明显更高

训练集中的 BCC Fe 全是 FM 初态，模型在分层 AFM 构型上的能量和磁力误差显著增大。需要对同一批几何结构补充 FM、AFM 和随机 PM 自旋端点，分离“几何覆盖不足”和“磁环境覆盖不足”。

**输入：** 2×2×2 BCC Fe 周期超胞。

**参数设置：**

- `Spin model` = `collinear`
- `Reference axis` = `[0, 0, 1]`
- `Element moments` = `Fe:2.2`
- 开启 FM、AFM 和 PM
- `AFM assignment` = `k_vector`
- `AFM layer vector` = `111`
- `PM structures per input` = `8`
- 开启随机种子并设置 `seed=42`

**运行前检查：** 首输入预览应显示 16 个磁性原子、AFM `+8/-8`，并给出每个输入及整个数据集的预计输出数量。如果 AFM 只有一个符号，先扩胞或更换 layer vector。

**输出：** 每个输入生成 1 个 FM、1 个 AFM 和 8 个随机 PM，共 10 个结构。

**验证改善：**

- 检查 FM 自旋沿参考轴同向，AFM 同时包含正负号。
- 固定 seed 后重复运行，PM 自旋应逐帧一致。
- 重训后分别比较 FM、AFM 和随机 PM 测试切片，不能只看混合总 RMSE。

## 参数说明

### 自旋模型和磁矩幅值

#### 自旋模型（format）

`str`，默认 `collinear`。

- `collinear`：所有输出沿参考轴，只改变正负符号。
- `noncollinear`：FM/AFM 可使用三维参考方向，PM 可在球面、锥面、平面或参考轴上取向。

旧项目中的 `Collinear (scalar)` 和 `Non-collinear (vector)` 会继续读取。这里的“模型”描述允许的自旋方向，不是文件格式；导出的规范字段始终是 `spin:R:3`。

#### 参考轴（axis）

`[x, y, z]`，默认 `[0, 0, 1]`。必须是有限的非零三分量向量。程序内部会归一化：

- 共线 FM/AFM/PM 使用 $\pm\hat{\mathbf a}$。
- 非共线 FM/AFM 默认以它为方向。
- 非共线 PM 的 cone、plane 和 axis 分布以它为参考。

#### 元素磁矩（magmom_map）

`str`，默认空。用于指定逐元素磁矩，单位为 $\mu_\mathrm{B}$：

```text
Fe:2.2,Co:1.7,Ni:0.6
```

非共线模式也接受向量值；向量的模长同时决定磁矩大小：

```text
Fe:[0,0,2.2],Cr:[0,0,-1.5]
```

也可使用 JSON：

```json
{"Fe": [0, 0, 2.2], "Cr": [0, 0, -1.5]}
```

#### 使用元素向量方向（use_element_dirs）

`bool`，默认 false，仅非共线 FM/AFM 使用。开启后，向量形式的元素条目会成为该元素的参考方向，AFM 再在这些方向上施加正负号。

开启此项后，`MagFMnc` 表示所有原子使用 FM 正号分支，并不保证不同元素的向量彼此平行。普通 FM/AFM 数据准备建议保持关闭；需要自定义多元素方向时应仔细检查输出。

#### 未列元素磁矩（default_moment）

`float`，默认 `0.0`。对作用范围内、但未出现在 `magmom_map` 中的元素使用该非负磁矩幅值。

#### 仅作用于元素（apply_elements）

`str`，默认空。逗号分隔，例如 `Fe,Co`。非空时，只有列出的元素参与磁矩映射；其他元素磁矩为 0。留空时考虑全部元素。

执行顺序是：先按 `apply_elements` 选择元素，再从 `magmom_map` 读取幅值，未命中时使用 `default_moment`。如果最终没有任何非零磁矩，卡片明确失败，不生成带错误磁序标签的全零结构。

### 输出分支

#### 生成 FM（gen_fm）

`bool`，默认 true。生成一个正号 FM 分支。

#### 生成 AFM（gen_afm）

`bool`，默认 false。开启后生成一个 AFM 分支，并显示 AFM 分配参数。

#### 生成随机 PM（gen_pm）

`bool`，默认 false。开启后生成 `pm_count` 个随机 PM 初始态。

FM、AFM 和 PM 至少要开启一个；全部关闭会明确失败。

### AFM 分配

#### AFM 分配方式（afm_mode）

`str`，默认 `k_vector`。

- `k_vector`：按当前晶胞的分数坐标层交替分配正负号。
- `group_ab`：读取已有的 `atoms.arrays["group"]`，由两个标签指定正负号。

旧值 `k-vector` 和 `group A/B` 会继续读取。group 模式缺少输入标签时不会再静默退回 k-vector。

#### AFM 层向量（afm_kvec）

`str`，默认 `111`，仅 `k_vector` 使用。可选 `100`、`010`、`001`、`110` 和 `111`，分别对应当前晶胞分数坐标中的 a、b、c、a+b 和 a+b+c 相位方向。

该规则依赖当前晶胞和坐标原点。若磁性原子只产生一种符号，卡片会要求扩胞或更换向量，而不是把 FM-like 结果标成 AFM。

#### AFM 正号组（afm_group_a）

`str`，默认 `A`，仅 `group_ab` 使用。必须非空，并在磁性原子中至少匹配一个输入 `group`。

#### AFM 负号组（afm_group_b）

`str`，默认 `B`，仅 `group_ab` 使用。必须非空、与正号组不同，并在磁性原子中至少匹配一个输入 `group`。

#### 其他分组置零（afm_zero_unknown）

`bool`，默认 true。开启时，不属于正号组或负号组的原子磁矩置零；关闭时，这些原子使用正号。该设置不会把 `group` 当成晶体学 `sublattice`。

### 随机 PM

#### 每个输入的 PM 数量（pm_count）

`int`，默认 `10`，最小为 1。总输出数为已开启的 FM/AFM 分支数加 `pm_count`。

#### PM 方向分布（pm_direction）

`str`，默认 `sphere`，仅非共线 PM 生效。

- `sphere`：整个球面均匀随机。
- `cone`：参考轴周围的锥内随机。
- `plane`：垂直于参考轴的平面内随机。
- `axis`：沿参考轴随机取正负方向。

共线 PM 始终沿参考轴随机取正负号，不使用该参数。

#### PM 锥半角（pm_cone_angle）

`float`，默认 `30.0`，范围 0–180°，仅非共线 `cone` 使用。

#### 按模长配对反向 PM（pm_balanced）

`bool`，默认 true。程序按相同磁矩模长分别生成反向方向对；完整配对会精确抵消，某一模长对应奇数个原子时可能留下一个残余磁矩。

在 `cone` 模式下，开启配对意味着方向分布在 $+\hat{\mathbf a}$ 和 $-\hat{\mathbf a}$ 两个锥内，而不是通过减均值破坏锥角限制。

### 随机性和输出预算

#### 使用随机种子（use_seed）

`bool`，默认 false，仅 PM 分支使用。开启后，seed、输入结构稳定标识和 PM 序号共同决定随机状态。

#### 随机种子（seed）

`int`，默认 `0`，必须非负。

#### 每个输入最大输出（max_outputs）

`int`，默认 `100`，最小为 1。如果 FM、AFM 和 PM 的预计输出总数超过该值，卡片明确失败，防止误设 `pm_count` 后生成过量结构。

## 推荐预设

### FM 单端点

```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "params": {
    "format": "collinear",
    "axis": [0.0, 0.0, 1.0],
    "magmom_map": "Fe:2.2",
    "use_element_dirs": false,
    "default_moment": 0.0,
    "apply_elements": "Fe",
    "gen_fm": true,
    "gen_afm": false,
    "afm_mode": "k_vector",
    "afm_kvec": "111",
    "afm_group_a": "A",
    "afm_group_b": "B",
    "afm_zero_unknown": true,
    "gen_pm": false,
    "pm_count": 10,
    "pm_direction": "sphere",
    "pm_cone_angle": 30.0,
    "pm_balanced": true,
    "use_seed": false,
    "seed": 0,
    "max_outputs": 100
  }
}
```

### FM + 分层 AFM + 共线 PM

```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "params": {
    "format": "collinear",
    "axis": [0.0, 0.0, 1.0],
    "magmom_map": "Fe:2.2",
    "use_element_dirs": false,
    "default_moment": 0.0,
    "apply_elements": "Fe",
    "gen_fm": true,
    "gen_afm": true,
    "afm_mode": "k_vector",
    "afm_kvec": "111",
    "afm_group_a": "A",
    "afm_group_b": "B",
    "afm_zero_unknown": true,
    "gen_pm": true,
    "pm_count": 8,
    "pm_direction": "sphere",
    "pm_cone_angle": 30.0,
    "pm_balanced": true,
    "use_seed": true,
    "seed": 42,
    "max_outputs": 20
  }
}
```

### 非共线球面 PM

```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "params": {
    "format": "noncollinear",
    "axis": [0.0, 0.0, 1.0],
    "magmom_map": "Fe:2.2,Co:1.7",
    "use_element_dirs": false,
    "default_moment": 0.0,
    "apply_elements": "Fe,Co",
    "gen_fm": false,
    "gen_afm": false,
    "afm_mode": "k_vector",
    "afm_kvec": "111",
    "afm_group_a": "A",
    "afm_group_b": "B",
    "afm_zero_unknown": true,
    "gen_pm": true,
    "pm_count": 16,
    "pm_direction": "sphere",
    "pm_cone_angle": 30.0,
    "pm_balanced": true,
    "use_seed": true,
    "seed": 7,
    "max_outputs": 20
  }
}
```

## 推荐组合

- `Group Label` → `Magnetic Order`：已有 coordinate-based group 时，用 `group_ab` 生成对应 AFM 正负号。
- `Magnetic Order` → `Spin Disorder`：先建立 FM/AFM 端点，再扫描局部无序比例。
- `Magnetic Order` → `Small-Angle Spin Tilt`：在端点附近补充确定性小角偏转。
- `Magnetic Order` → `Spin Spiral`：复用元素磁矩幅值，生成位置连续的螺旋磁序。

## 常见问题

**预览提示没有非零磁矩。** `magmom_map` 为空、磁矩都是 0，或 `apply_elements` 排除了所有映射元素。卡片不会把全零自旋标成 FM/AFM/PM。

**AFM 提示只有一个符号。** 当前晶胞在所选 k-vector 下没有足够的交替相位。先扩胞或更换 layer vector。

**group 模式提示缺少 A/B。** 输入必须已有 `atoms.arrays["group"]`，且正负标签各自至少匹配一个非零磁矩原子。不会自动识别晶体学子晶格，也不会静默回退。

**配对后的 PM 净磁矩仍不完全为零。** 某种磁矩模长对应奇数个原子时会留下一个未配对方向；不同模长之间不会通过改变模长强行抵消。

**PM 和 Spin Disorder 是否重复。** PM 分支把全部磁性原子直接放到一个随机端点；`Spin Disorder` 能按比例只扰动部分已有自旋并扫描中间无序度。

## 输出标签

- `MagFM` / `MagFMnc`：共线 / 非共线 FM 正号分支。
- `MagAFM111` / `MagAFM111nc`：按 k-vector 生成的 AFM。
- `MagAFMg` / `MagAFMgnc`：按现有 group 标签生成的 AFM。
- `MagPM` / `MagPMnc`：随机 PM；固定 seed 时标签带派生 seed。

所有导出输出写入 `spin:R:3`。共线模式内部保留标量 `initial_magmoms`，导出时仍按参考轴转换为三分量 spin。

## 可复现性

FM 和 AFM 没有随机性。PM 开启 `use_seed` 后，同一输入、参数和 seed 会得到相同输出。
