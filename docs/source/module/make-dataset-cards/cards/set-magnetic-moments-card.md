<!-- card-schema: {"card_name": "Set Magnetic Moments", "source_file": "src/NepTrainKit/ui/views/_card/set_magnetic_moments_card.py", "serialized_keys": ["params"]} -->

# 设置磁矩（Set Magnetic Moments）

`Group`: `Magnetism` | `Class`: `SetMagneticMomentsCard`

## 功能说明

把输入结构写成统一的 `initial_magmoms` 表示。选择磁矩来源（已有磁矩 / 元素映射 / 常数幅值）、输出格式（标量 / 向量）、参考轴方向和生效元素范围。这是一张基础层卡片——不生成多磁序分支，只为后续磁性卡片提供干净、统一的初始磁矩。

**和 `Magnetic Order` 的区别：** `Magnetic Order` 从一个母相出发生成 FM+AFM+PM 多分支。本卡只标准化磁矩格式，不创建新磁序。如果你的流程里多张磁性卡都在重复配 `magmom_map` 和 `axis`，用本卡统一初始化一次可以消除重复。

## 操作示例

### 场景：多张磁性卡各自配 magmom_map，配错了一处

你的训练流水线里有 `Magnetic Order` → `Small-Angle Spin Tilt` → `Spin Spiral` 三张磁性卡。每张卡都有一个 `magmom_map` 输入框，你填了三遍 `Fe:2.2,Co:1.7`。两周后改 Co 的磁矩为 1.5，只改了两张卡，第三张忘了——生成的数据里 Co 的磁矩默默用了旧值。训练出来的模型在含 Co 的构型上能量系统性偏高。

**诊断思路：** 磁性流水线里，磁矩幅值的定义应该只在一处。用本卡在最前面做一次统一初始化，后续磁性卡全部选 "Existing initial magmoms" 模式，不再各自定义 magmom_map。

**输入：** 一个磁性晶体结构，还没写 `initial_magmoms`

**目标：** 统一写入向量格式的初始磁矩，Fe 取 2.2 μB，Co 取 1.7 μB，沿 z 轴，后续卡片全部复用

**参数设置：**
- `Source` = `Map/default magnitude`
- `Format` = `Non-collinear (vector)`
- `Magmom Map` = `Fe:2.2,Co:1.7`
- `Apply Elements` = `Fe,Co`

**输出：** 1 个结构，带 `MagSet(map,vec)` 标签和 `initial_magmoms` 数组。原子位置和数量不变。

**怎么验证训练集质量改善：**
- 后续磁性卡全部切到 "Existing initial magmoms"，检查输出磁矩幅值是否一致
- 如果 Co 的磁矩需要精确到实验值，直接用 `Magmom Map` 的精确定义
- 如果后续要同时跑标量和向量流程，分别用本卡输出两种格式的版本，而不是在后面卡片里切换

### 什么时候加这张卡、什么时候不加

**加：**
- 流水线里多张磁性卡都在重复配置磁矩幅值、格式、axis
- 输入结构的磁矩格式不统一（有的标量有的向量），需要标准化
- 只想初始化参考 FM 磁矩，不想生成多磁序分支

**不加：**
- 只有一张磁性卡且不共享磁矩配置 → 直接在目标卡上配也可以
- 需要一次性生成 FM/AFM/PM 多磁序 → 用 `Magnetic Order` 一步完成

## 参数说明



### 磁矩来源

#### Source（source）
类型：`str`。默认：`'Map/default magnitude'`。选择输入信息的来源。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### Magmom Map（magmom_map）
类型：`str`。默认：`''`。按元素指定磁矩幅值或方向，例如 `Fe:2.2, Ni:0.6`。

物理直觉：已有元素磁矩先验时使用；未知体系不要用它伪造不存在的元素差异。


#### Use Element Dirs（use_element_dirs）
类型：`bool`。默认：`False`。决定是否使用元素映射中的方向信息。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Default Moment（default_moment）
类型：`float`。默认：`0.0`。为没有显式元素映射的原子提供默认磁矩幅值。

物理直觉：只适合作为兜底幅值；关键磁性元素应在 `magmom_map` 中显式给出。


#### Constant Moment（constant_moment）
类型：`float`。默认：`2.0`。用统一磁矩幅值覆盖目标原子。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


### 输出格式

#### Format（format）
类型：`str`。默认：`'Non-collinear (vector)'`。选择磁矩写入格式。

物理直觉：标量适合共线近似；向量适合非共线、自旋螺旋和 canting 数据。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### Lift Scalar（lift_scalar）
类型：`bool`。默认：`True`。决定是否把标量磁矩提升为非共线向量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Axis（axis）
类型：`list[float] | tuple[float, float, float]`。默认：`(0.0, 0.0, 1.0)`。选择操作沿哪个空间轴或磁矩参考轴定义。

物理直觉：坐标轴参数会改变空间分层、吸附方向或磁矩方向；使用前先确认结构取向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。


### 元素范围

#### Apply Elements（apply_elements）
类型：`str`。默认：`''`。限制只处理指定元素，留空表示处理所有元素。

物理直觉：用于只扰动磁性元素或只替换目标元素；留空代表全元素参与。

## 推荐预设

### 标准化现有磁矩为向量（不改变幅值，适合接入已有数据）
```json
{
  "class": "SetMagneticMomentsCard",
  "check_state": true,
  "source": "Existing initial magmoms",
  "format": "Non-collinear (vector)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [0.0],
  "constant_moment": [2.0],
  "lift_scalar": true,
  "apply_elements": ""
}
```

### 按元素映射统一初始化（适合流水线第一张磁性卡）
```json
{
  "class": "SetMagneticMomentsCard",
  "check_state": true,
  "source": "Map/default magnitude",
  "format": "Non-collinear (vector)",
  "axis": [0.0, 0.0, 1.0],
  "magmom_map": "Fe:2.2,Co:1.7",
  "use_element_dirs": false,
  "default_moment": [0.0],
  "constant_moment": [2.0],
  "lift_scalar": true,
  "apply_elements": "Fe,Co"
}
```

### 常数幅值快速探索（扫不同参考磁矩的影响）
```json
{
  "class": "SetMagneticMomentsCard",
  "check_state": true,
  "source": "Constant magnitude",
  "format": "Non-collinear (vector)",
  "axis": [1.0, 0.0, 0.0],
  "magmom_map": "",
  "use_element_dirs": false,
  "default_moment": [0.0],
  "constant_moment": [3.0],
  "lift_scalar": true,
  "apply_elements": "Fe,Co,Ni"
}
```

## 推荐组合

- `Set Magnetic Moments` → `Magnetic Order`：先统一磁矩格式，再生成 FM/AFM/PM 磁序
- `Set Magnetic Moments` → `Small-Angle Spin Tilt`：先生成统一 FM 向量磁矩，再做 canting
- `Set Magnetic Moments` → `Spin Spiral`：先写入稳定模长，再生成螺旋初态

## 常见问题

**输出没有磁矩。** 检查 `Source` 模式是否匹配输入。`Existing` 模式要求输入已有 `initial_magmoms`。`Map/default` 需要填写 `magmom_map` 或合理的 `default_moment`。

**标量磁矩没变成向量。** `Lift Scalar` 没开，且 `Format` 选了向量但输入只有标量。开启 `lift_scalar`。

**某些元素磁矩被意外清零。** 检查 `apply_elements` 是否为非空且遗漏了目标元素。

## 输出标签

- `MagSet(existing,sca)` / `MagSet(map,vec)` / `MagSet(const,vec)` 等：标记磁矩来源和输出格式

所有输出写入 `initial_magmoms` 数组。

## 可复现性

无随机性。相同输入、相同 `Source` 和相同参数 → 严格一致输出。
