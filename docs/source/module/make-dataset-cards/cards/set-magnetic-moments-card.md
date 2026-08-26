<!-- card-schema: {"card_name": "Set Magnetic Moments", "source_file": "src/NepTrainKit/ui/views/_card/set_magnetic_moments_card.py", "serialized_keys": ["params"]} -->

# 设置磁矩（Set Magnetic Moments）

**分类：** 磁性

## 功能说明

把输入结构写成统一的 `spin:R:3` 表示。选择磁矩来源（已有磁矩 / 元素映射 / 常数幅值）、输出格式（标量 / 向量）、参考轴方向和生效元素范围。这是一张基础层卡片——不生成多磁序分支，只为后续磁性卡片提供干净、统一的初始磁矩。内部仍同步维护 ASE `initial_magmoms` 兼容别名。

**和 `Magnetic Order` 的区别：** `Magnetic Order` 从一个母相出发生成 FM+AFM+PM 多分支。本卡只标准化磁矩格式，不创建新磁序。如果你的流程里多张磁性卡都在重复配 `元素磁矩表`（`magmom_map`）和 `轴`（`axis`），用本卡统一初始化一次可以消除重复。

## 原理与公式

这张卡执行的是确定性映射而不是磁结构求解。对元素为 $e$ 的原子 $i$，标量模式写入

$$
m_i=M(e),
$$

非共线模式写入

$$
\mathbf m_i=M(e)\,
\frac{\mathbf a_e}{\|\mathbf a_e\|},
$$

其中 $M(e)$ 来自元素-磁矩表，$\mathbf a_e$ 来自元素方向表；没有独立方向时使用全局
参考轴。未列元素按“默认磁矩”处理。数值单位为 $\mu_B$。它只设置
`initial_magmoms` 初始标签，不会计算交换作用、自动判断磁性元素或保证 DFT 收敛到该磁态。

## 操作示例

### 场景：多张磁性卡各自配 magmom_map，配错了一处

你的训练流水线里有 `Magnetic Order` → `Local Magnetic Response` → `Spin Spiral` 三张磁性卡。每张卡都有一个 `元素磁矩表`（`magmom_map`）输入框，你填了三遍 `Fe:2.2,Co:1.7`。两周后改 Co 的磁矩为 1.5，只改了两张卡，第三张忘了——生成的数据里 Co 的磁矩默默用了旧值。训练出来的模型在含 Co 的构型上能量系统性偏高。

**诊断思路：** 磁性流水线里，磁矩幅值的定义应该只在一处。用本卡在最前面做一次统一初始化，后续磁性卡全部选 "Existing initial magmoms" 模式，不再各自定义 magmom_map。

**输入：** 一个磁性晶体结构，还没写 `spin` 或 `initial_magmoms`

**目标：** 统一写入向量格式的初始磁矩，Fe 取 2.2 μB，Co 取 1.7 μB，沿 z 轴，后续卡片全部复用

**参数设置：**
- `来源`（`source`） = `Map/default magnitude`
- `格式`（`format`） = `Non-collinear (vector)`
- `元素磁矩表`（`magmom_map`） = `Fe:2.2,Co:1.7`
- `Apply Elements` = `Fe,Co`

**输出：** 1 个结构，带 `MagSet(map,vec)` 标签和 `spin:R:3` 数组。原子位置和数量不变。

**怎么验证训练集质量改善：**
- 后续磁性卡全部切到 "Existing initial magmoms"，检查输出磁矩幅值是否一致
- 如果 Co 的磁矩需要精确到实验值，直接用 `元素磁矩表`（`magmom_map`）的精确定义
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

#### 来源（source）

`str`，默认 `'Map/default magnitude'`。磁矩从哪里来。

| 选项 | 含义 |
|------|------|
| `Map/default magnitude` | 用 `元素磁矩表`（`magmom_map`） / `默认磁矩`（`default_moment`）生成。输入不需要已有磁矩 |
| `Existing initial magmoms` | 保留输入已有的磁矩，只做格式转换。输入需要已有 `initial_magmoms` |
| `Constant magnitude` | 所有目标元素统一用一个常数幅值，适合快速扫参考值 |

#### 元素磁矩表（magmom_map）

`str`，默认 `''`。界面使用“元素 / 磁矩或向量”表格；标量可填 `2.2`，向量可填 `[0,0,1.0]`。序列化仍兼容 `Fe:2.2,Ni:0.6` 和 `{"Cr":[0,0,1.0]}`。

#### 使用元素方向（use_element_dirs）

`bool`，默认 `False`。勾选后 `元素磁矩表`（`magmom_map`）中提供的向量方向会被保留，而不是统一投到 `轴`（`axis`）。仅向量格式下有意义。

#### 默认磁矩（default_moment）

`float`，默认 `0.0`。`元素磁矩表`（`magmom_map`）未命中元素的默认幅值。设 0 则不给未配置元素写磁矩。

#### 固定磁矩（constant_moment）

`float`，默认 `2.0`。`Source = Constant magnitude` 时所有目标元素的统一幅值。

### 输出格式

#### 格式（format）

`str`，默认 `'Non-collinear (vector)'`。`Collinear (scalar)` 或 `Non-collinear (vector)`。后续要做旋转、螺旋、canting 时优先选向量格式。

#### 标量磁矩转为矢量（lift_scalar）

`bool`，默认 `True`。输入是标量且目标格式是向量时，是否沿 `轴`（`axis`）抬升。通常保持开启。

#### 轴（axis）

`list[float] | tuple[float, float, float]`，默认 `(0.0, 0.0, 1.0)`。向量输出的参考方向、标量抬升的方向，格式为 `[x, y, z]`。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

### 元素范围

#### 应用元素（apply_elements）

`str`，默认 `''`。逗号分隔如 `Fe,Co`，只对列出的元素写磁矩，其余写 0。留空则全部生效。

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
- `Set Magnetic Moments` → `Local Magnetic Response`：先生成统一 FM 向量磁矩，再做局域响应扫描
- `Set Magnetic Moments` → `Spin Spiral`：先写入稳定模长，再生成螺旋初态

## 常见问题

**输出没有磁矩。** 检查 `来源`（`source`）模式是否匹配输入。`Existing` 模式优先读取 `spin:R:3`，没有时才读取旧 `initial_magmoms`。`Map/default` 需要填写 `元素磁矩表`（`magmom_map`）或合理的 `默认磁矩`（`default_moment`）。

**标量磁矩没变成向量。** `标量磁矩转为矢量`（`lift_scalar`）没开，且 `格式`（`format`）选了向量但输入只有标量。开启 `标量磁矩转为矢量`（`lift_scalar`）。

**某些元素磁矩被意外清零。** 检查 `应用元素`（`apply_elements`）是否为非空且遗漏了目标元素。

## 输出标签

- `MagSet(existing,sca)` / `MagSet(map,vec)` / `MagSet(const,vec)` 等：标记磁矩来源和输出格式

所有导出输出写入 `spin:R:3`。标量格式必须提供非零 `轴`（`axis`），程序据此构造三分量自旋，不会静默假设方向。

## 可复现性

无随机性。相同输入、相同 `来源`（`source`）和相同参数 → 严格一致输出。
