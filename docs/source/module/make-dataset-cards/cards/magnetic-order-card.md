<!-- card-schema: {"card_name": "Magnetic Order", "source_file": "src/NepTrainKit/ui/views/_card/magnetic_order_card.py", "serialized_keys": ["params"]} -->

# 磁有序初始化（Magnetic Order）

`Group`: `Magnetism` | `Class`: `MagneticOrderCard`

## 功能说明

从一个磁性母相结构出发，一次性生成 FM（铁磁）、AFM（反铁磁）、PM（顺磁）多种磁序分支。每个输出结构都写入对应的 `initial_magmoms` 数组和 `Config_type` 标签。

**关键限制：** 这张卡只写入初始磁矩，不保证生成的就是磁基态。磁矩是否收敛到合理值仍需要用 NEP/DFT 后续计算验证。

## 操作示例

### 场景：FM 模型在 AFM 相上预测崩了

你在 bcc Fe 上训练了一个 NEP 模型，所有训练数据来自 FM 构型。模型对铁磁态的能量和力预测很好，但一跑 AFM 构型，能量误差是 FM 的 5 倍。诊断结果是：训练集里只有一种磁序，模型把"所有磁矩同向"当成了默认假设，碰到磁矩翻转的结构只能瞎猜。

**诊断思路：** 磁性体系的训练集至少应该覆盖 FM 和 AFM 两种极端磁序，如果体系可能有顺磁相，PM 也要加。不同磁序下，同一种原子的局域磁环境完全不同——模型需要见过这些环境才能泛化。

**输入：** 一个含 Fe 的 bcc 晶体，已知 Fe 的磁矩约 2.2 μB（来自 DFT 或文献）

**目标：** 从一个结构出发，生成 1 个 FM + 1 个 AFM（k-vector 111）+ 8 个 PM，共 10 个不同磁序的训练样本

**参数设置：**
- `Format` = `Collinear (scalar)` （先从不共线开始，复杂度低）
- `Magmom Map` = `Fe:2.2`
- `Gen FM` = 勾选，`Gen AFM` = 勾选
- `AFM Mode` = `k-vector`，`AFM Kvec` = `111`
- `Gen PM` = 勾选，`PM Count` = `[8]`

**输出：** 1 个 FM + 1 个 AFM + 8 个 PM，全部带 `initial_magmoms` 和对应的 `Config_type` 标签

**怎么验证训练集质量改善：**
- 重训后用这组数据里的 FM 和 AFM 分别做推理，能量误差差异应该显著缩小
- 抽查 FM/AFM 的 `initial_magmoms`：FM 全正、AFM 在 111 方向上相邻层符号翻转
- 如果 AFM 效果仍差，换 `AFM Kvec` = `110` 或 `100`，覆盖不同翻转周期
- 如果 PM 结构能量分布异常宽，可能是磁矩幅值设置不合理——检查 `magmom_map` 或 `default_moment`

### 什么时候加这张卡、什么时候不加

**加：**
- 模型在不同磁序之间泛化差（FM 推理好、AFM 崩）
- 研究体系本身存在多种磁有序相（如 Fe、NiO、钙钛矿磁性材料）
- 需要模型能区分磁基态和激发态

**不加：**
- 体系是非磁的（纯 Si、Al 等），加了只会引入无意义的磁矩标签
- 只需要静态磁矩初始化，不需要多磁序分支（用 `Set Magnetic Moments`）

## 参数说明



### 磁矩格式与幅值

#### Format（format）
类型：`str`。默认：`'Collinear (scalar)'`。选择磁矩写入格式。

物理直觉：标量适合共线近似；向量适合非共线、自旋螺旋和 canting 数据。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### Axis（axis）
类型：`list[float] | tuple[float, float, float]`。默认：`(0.0, 0.0, 1.0)`。选择操作沿哪个空间轴或磁矩参考轴定义。

物理直觉：坐标轴参数会改变空间分层、吸附方向或磁矩方向；使用前先确认结构取向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。


#### Magmom Map（magmom_map）
类型：`str`。默认：`''`。按元素指定磁矩幅值或方向，例如 `Fe:2.2, Ni:0.6`。

物理直觉：已有元素磁矩先验时使用；未知体系不要用它伪造不存在的元素差异。


#### Use Element Dirs（use_element_dirs）
类型：`bool`。默认：`False`。决定是否使用元素映射中的方向信息。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Default Moment（default_moment）
类型：`float`。默认：`0.0`。为没有显式元素映射的原子提供默认磁矩幅值。

物理直觉：只适合作为兜底幅值；关键磁性元素应在 `magmom_map` 中显式给出。


#### Apply Elements（apply_elements）
类型：`str`。默认：`''`。限制只处理指定元素，留空表示处理所有元素。

物理直觉：用于只扰动磁性元素或只替换目标元素；留空代表全元素参与。


### FM / AFM 分支

#### Gen FM（gen_fm）
类型：`bool`。默认：`True`。决定是否生成 FM 磁序结构。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Gen AFM（gen_afm）
类型：`bool`。默认：`True`。决定是否生成 AFM 磁序结构。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### AFM Mode（afm_mode）
类型：`str`。默认：`'k-vector'`。选择 AFM 生成方式。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### AFM Kvec（afm_kvec）
类型：`str`。默认：`'111'`。控制 `afm_kvec` 对应的生成或过滤行为。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`gen_afm=True` 且 `afm_mode` 使用 k-vector。


#### AFM Group A（afm_group_a）
类型：`str`。默认：`'A'`。指定 AFM A 子晶格。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`gen_afm=True` 且 `afm_mode` 使用 group。


#### AFM Group B（afm_group_b）
类型：`str`。默认：`'B'`。指定 AFM B 子晶格。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`gen_afm=True` 且 `afm_mode` 使用 group。


#### AFM Zero Unknown（afm_zero_unknown）
类型：`bool`。默认：`True`。决定未知 group 原子是否置零磁矩。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


### PM 分支

#### Gen PM（gen_pm）
类型：`bool`。默认：`False`。决定是否生成 PM 随机磁矩结构。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### PM Count（pm_count）
类型：`int`。默认：`10`。设置 PM 随机结构数量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`gen_pm=True`。


#### PM Direction（pm_direction）
类型：`str`。默认：`'sphere'`。选择 PM 随机方向模型。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |

生效条件：`gen_pm=True`。


#### PM Cone Angle（pm_cone_angle）
类型：`float`。默认：`30.0`。设置 PM cone 随机方向的最大偏转角。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`pm_direction` 选择 cone 类方向时。


#### PM Balanced（pm_balanced）
类型：`bool`。默认：`True`。决定 PM 随机方向是否强制总体平衡。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`gen_pm=True`。


### 随机性

#### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现的训练集生成或测试时打开；做最终大规模探索且希望保留随机多样性时可关闭。


#### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一 seed 应产生同一批候选；只有在 `use_seed` 打开时才改变结果。

生效条件：`use_seed=True`。

## 推荐预设

### FM + AFM 基础集（2 个输出，快速验证用）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2",
  "default_moment": [0.0],
  "apply_elements": "Fe",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "gen_pm": false,
  "use_seed": true,
  "seed": [42]
}
```

### FM + AFM + PM 全磁序（~12 个输出，常规训练用）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Collinear (scalar)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2,Ni:0.6",
  "default_moment": [0.0],
  "apply_elements": "",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "111",
  "gen_pm": true,
  "pm_count": [10],
  "pm_direction": "sphere",
  "pm_balanced": true,
  "use_seed": true,
  "seed": [42]
}
```

### 非共线多磁序（~17 个输出，研究级）
```json
{
  "class": "MagneticOrderCard",
  "check_state": true,
  "format": "Non-collinear (vector)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2,Cr:1.5",
  "default_moment": [0.5],
  "apply_elements": "Fe,Cr,Mn",
  "gen_fm": true,
  "gen_afm": true,
  "afm_mode": "k-vector",
  "afm_kvec": "110",
  "gen_pm": true,
  "pm_count": [15],
  "pm_direction": "cone",
  "pm_cone_angle": [45.0],
  "pm_balanced": true,
  "use_seed": true,
  "seed": [42]
}
```

## 推荐组合

- `Group Label` → `Magnetic Order`：先打 group 标签，再用 group A/B 模式生成细粒度 AFM
- `Magnetic Order` → `Magmom Rotation`：基础磁序 → 小角度旋转补样
- `Magnetic Order` → `Spin Spiral`：磁序初始化幅值 → 螺旋调制

## 常见问题

**输出没有磁矩。** `magmom_map` 和 `default_moment` 是否都为空/0？`apply_elements` 是否过滤掉了所有有磁矩的元素？

**AFM 没翻转。** 如果用 `group A/B` 模式，检查输入是否有 `atoms.arrays['group']` 且包含 A/B 标签。没有的话换 `k-vector` 模式。

**PM 没生成。** `gen_pm` 默认关闭，确认已勾选。

**非共线磁矩方向不符合预期。** 检查 `Axis` 是否正确。非共线 FM/AFM 模式下，每个原子的磁矩方向沿 `Axis`，只有 PM 才会随机偏离。

## 输出标签

- `MagFM` / `MagFMnc`：铁磁（共线/非共线）
- `MagAFM111` / `MagAFM110`：反铁磁，后缀为 k-vector
- `MagAFMg`：group A/B 模式反铁磁
- `MagPM` / `MagPMnc`：顺磁，可能带 seed 后缀

所有输出写入 `initial_magmoms` 数组。

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。PM 随机性受 seed + 结构稳定 ID 联合控制。
