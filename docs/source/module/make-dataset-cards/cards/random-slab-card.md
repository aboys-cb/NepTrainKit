<!-- card-schema: {"card_name": "Random Slab", "source_file": "src/NepTrainKit/ui/views/_card/random_slab_card.py", "serialized_keys": ["h_range", "k_range", "l_range", "layer_range", "vacuum_range"]} -->

# 随机表面切片（Random Slab）

`Group`: `Surface`  
`Class`: `RandomSlabCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_slab_card.py`

## 功能说明
按 Miller 指数、层数和真空范围随机生成 slab，构建表面取向与厚度覆盖样本。

它最适合的场景是：从体相结构切出多取向、多厚度的 slab，用于表面相关训练。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：从体相结构切出多取向、多厚度的 slab，用于表面相关训练

**输入：** 一个体相晶体结构，最好已先扩成适合切片的母胞

**目标：** 覆盖不同 Miller 指数、层数和真空厚度，而不是只做单一表面

**参数设置：**
- `h_range/k_range/l_range` 先从低指数开始
- `layer_range` 先保证 slab 足够厚，不要一开始就追求极薄表面
- `vacuum_range` 通常先从 10-15 Å 起步

**输出：** 一批表面取向、厚度和真空层不同的 slab 结构

**怎么验证结果合理：**
- 检查 slab 上下表面没有明显重叠
- 确认真空层足够隔离镜像
- 若输出过多，先收窄 Miller 指数范围

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 表面相关任务误差显著高于体相。
- 目标任务 (Target objective): 扩展表面几何和边界条件分布。
- 建议添加条件 (Add-it trigger): 吸附、表面反应、界面任务。
- 不建议添加条件 (Avoid trigger): 只做体相性质训练。
> 物理提示 (Physics caution): 重点检查真空层是否足够、上下表面是否重叠，以及表面附近是否出现异常短键。

## 输入前提
- 先用窄 h/k/l 范围试跑。
- 保证真空层下限足够避免镜像相互作用。

## 参数说明（完整）
### `h_range` (H Range)
- UI Label: `H Range`
- 字段映射 (Field mapping): 序列化键 `h_range` <-> 界面标签 `H Range`。
- 控件标签 (Caption): `H Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0, 1, 1]`
- 含义 (Meaning): h 指数范围 (h range)。
- 对输出规模/物理性的影响: 控制表面取向扫描维度之一。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `k_range` (K Range)
- UI Label: `K Range`
- 字段映射 (Field mapping): 序列化键 `k_range` <-> 界面标签 `K Range`。
- 控件标签 (Caption): `K Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0, 1, 1]`
- 含义 (Meaning): k 指数范围 (k range)。
- 对输出规模/物理性的影响: 控制表面取向扫描维度之一。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `l_range` (L Range)
- UI Label: `L Range`
- 字段映射 (Field mapping): 序列化键 `l_range` <-> 界面标签 `L Range`。
- 控件标签 (Caption): `L Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[1, 3, 1]`
- 含义 (Meaning): l 指数范围 (l range)。
- 对输出规模/物理性的影响: 控制表面取向扫描维度之一。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `layer_range` (Layer Range)
- UI Label: `Layer Range`
- 字段映射 (Field mapping): 序列化键 `layer_range` <-> 界面标签 `Layer Range`。
- 控件标签 (Caption): `Layer Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[3, 6, 1]`
- 含义 (Meaning): 层数范围 (layer range)。
- 对输出规模/物理性的影响: 主控 slab 厚度分布。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 推荐范围 (Recommended range):
  - 保守：3 到 6，step 1
  - 平衡：3 到 6，step 0.5
  - 探索：3 到 6，step 2

### `vacuum_range` (Vacuum Range)
- UI Label: `Vacuum Range`
- 字段映射 (Field mapping): 序列化键 `vacuum_range` <-> 界面标签 `Vacuum Range`。
- 控件标签 (Caption): `Vacuum Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[10, 10, 1]`
- 含义 (Meaning): 真空层范围 (vacuum range)。
- 对输出规模/物理性的影响: 影响表面镜像相互作用强度。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 推荐范围 (Recommended range):
  - 保守：10Å 左右
  - 平衡：12-20Å
  - 探索：20Å+

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [
    0,
    1,
    1
  ],
  "k_range": [
    0,
    1,
    1
  ],
  "l_range": [
    1,
    3,
    1
  ],
  "layer_range": [
    3,
    6,
    1
  ],
  "vacuum_range": [
    10,
    10,
    1
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [
    0,
    1,
    1
  ],
  "k_range": [
    0,
    1,
    1
  ],
  "l_range": [
    1,
    3,
    1
  ],
  "layer_range": [
    3,
    6,
    1
  ],
  "vacuum_range": [
    10,
    10,
    1
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [
    0,
    1,
    1
  ],
  "k_range": [
    0,
    1,
    1
  ],
  "l_range": [
    1,
    3,
    1
  ],
  "layer_range": [
    3,
    6,
    1
  ],
  "vacuum_range": [
    10,
    10,
    1
  ]
}
```

## 推荐组合
- Random Slab -> Insert Defect -> Random Vacancy: 生成更接近实际的缺陷表面样本。
- 表面任务通常先 `Super Cell` 再 `Random Slab`，最后再接吸附或空位卡。
- 若想保留表面多样性但减少冗余，可在最后再做 `FPS Filter`。

## 常见问题与排查
- 输出为空时，通常是切片范围太窄、最小距离太严，或输入母胞尺寸不适合当前表面任务。
- 如果 slab 或吸附结构不合理，先检查真空层、层数和插入距离，再抽查表面上下是否发生重叠。
- 这类卡片不会自动判断你想做“表面”还是“体相缺陷”；需要表面缺陷时应先切 slab，再叠加空位或插入。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Slab(hkl={...}{...}{...},L={...},vac={...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
