<!-- card-schema: {"card_name": "Layer Copy", "source_file": "src/NepTrainKit/ui/views/_card/layer_copy_card.py", "serialized_keys": ["operation_params", "preset_index", "dz_expr", "params", "apply_mode", "elements", "z_range", "wrap", "extend_cell_z", "extra_vacuum", "layers", "distance"]} -->

# 层复制（Layer Copy）

`Group`: `Structure`  
`Class`: `LayerCopyCard`  
`Source`: `src/NepTrainKit/ui/views/_card/layer_copy_card.py`

## 功能说明
复制层并按 `dz_expr` 施加位移调制，生成层间错位、起伏和堆叠变化数据。

它最适合的场景是：复制指定层并沿 z 方向平移，用于构造多层界面、重复层或表面加厚样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：复制指定层并沿 z 方向平移，用于构造多层界面、重复层或表面加厚样本

**输入：** 一个分层明显的 slab 或层状材料结构

**目标：** 把某一层或某几层复制到新位置，快速扩展层数或构造分层组合

**参数设置：**
- `preset_index` 先决定常用复制模式还是自定义模式
- `dz_expr` 或 `distance` 控制复制后层间距
- `apply_mode` 先选复制到顶部、底部还是全部

**输出：** 结构层数增加，新增层相对原层有明确的 z 方向平移

**怎么验证结果合理：**
- 检查复制层没有与原层重叠
- 确认 `extend_cell_z` 与 `extra_vacuum` 是否同步更新了盒子高度
- 若只想局部复制，记得限制 `elements` 或 `z_range`

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 层状体系样本单一，层间自由度覆盖不足。
- 目标任务 (Target objective): 增强层间几何变化与形貌多样性。
- 建议添加条件 (Add-it trigger): 研究二维材料、多层异质结、层间耦合。
- 不建议添加条件 (Avoid trigger): 非层状体相体系。
> 物理提示 (Physics caution): 重点检查层间距离、是否发生层重叠，以及是否需要同步增加真空或盒子高度。

## 输入前提
- 先在单帧验证 `dz_expr` 与 `params`。
- 按边界条件选择 `wrap/extend_cell_z/extra_vacuum`。

## 参数说明（完整）
### `operation_params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `operation_params` <-> 核心操作参数 `LayerCopyParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"preset_index": 1, "dz_expr": "sin(x/pi) + sin(y/pi)", "expression_params": "", "apply_mode": 0, "elements": "", "z_range": [-1000000.0, 1000000.0], "wrap": false, "extend_cell_z": true, "extra_vacuum": 0.0, "layers": 3, "distance": 3.0}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的表达式、作用范围、层数、层间距和晶胞扩展字段一致。
- 配置建议 (Practical note): `params` 已用于表达式参数，因此核心快照使用 `operation_params` 避免覆盖旧 workflow 语义。

### `preset_index` (Preset Index)
- UI Label: `Preset Index`
- 字段映射 (Field mapping): 序列化键 `preset_index` <-> 界面标签 `Preset Index`。
- 控件标签 (Caption): `Preset Index`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `1`
- 含义 (Meaning): 预设索引 (preset index)。
- 对输出规模/物理性的影响: 选择内置变换模板。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：默认预设
  - 平衡：按体系切预设
  - 探索：复杂预设先单帧验证

### `dz_expr` (Dz Expr)
- UI Label: `Dz Expr`
- 字段映射 (Field mapping): 序列化键 `dz_expr` <-> 界面标签 `Dz Expr`。
- 控件标签 (Caption): `Dz Expr`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"sin(x/pi) + sin(y/pi)"`
- 含义 (Meaning): 位移表达式 (displacement expression)。
- 对输出规模/物理性的影响: 决定空间相关层位移函数形式。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 按表达式语法填写，可替换为自定义位移函数。

### `params` (Params)
- UI Label: `Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 界面标签 `Params`。
- 控件标签 (Caption): `Params`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 表达式参数 (expression parameters)。
- 对输出规模/物理性的影响: 用于调节 `dz_expr` 形状和幅度。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 按表达式参数表填写，建议先小范围试跑确认语义。

### `apply_mode` (Apply Mode)
- UI Label: `Apply Mode`
- 字段映射 (Field mapping): 序列化键 `apply_mode` <-> 界面标签 `Apply Mode`。
- 控件标签 (Caption): `Apply Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `0`
- 含义 (Meaning): 应用模式 (apply mode)。
- 对输出规模/物理性的影响: 决定操作作用对象和范围。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：局部模式
  - 平衡：分层模式
  - 探索：全局模式仅探索

### `elements` (Elements)
- UI Label: `Elements`
- 字段映射 (Field mapping): 序列化键 `elements` <-> 界面标签 `Elements`。
- 控件标签 (Caption): `Elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 元素集合输入 (element set)。
- 对输出规模/物理性的影响: 决定参与操作的元素子集。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): `Elements` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `z_range` (Z Range)
- UI Label: `Z Range`
- 字段映射 (Field mapping): 序列化键 `z_range` <-> 界面标签 `Z Range`。
- 控件标签 (Caption): `Z Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[-1000000.0, 1000000.0]`
- 含义 (Meaning): Z 向范围 (z range)。
- 对输出规模/物理性的影响: 控制沿 z 方向的作用区间或幅度。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：1-2
  - 平衡：2-5
  - 探索：5-10

### `wrap` (Wrap)
- UI Label: `Wrap`
- 字段映射 (Field mapping): 序列化键 `wrap` <-> 界面标签 `Wrap`。
- 控件标签 (Caption): `Wrap`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 回卷边界 (wrap to cell)。
- 对输出规模/物理性的影响: 开启后坐标会映射回周期胞内。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Wrap` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `extend_cell_z` (Extend Cell Z)
- UI Label: `Extend Cell Z`
- 字段映射 (Field mapping): 序列化键 `extend_cell_z` <-> 界面标签 `Extend Cell Z`。
- 控件标签 (Caption): `Extend Cell Z`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 扩展 z 晶胞 (extend cell in z)。
- 对输出规模/物理性的影响: 用于避免层复制后跨边界冲突。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Extend Cell Z` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `extra_vacuum` (Extra Vacuum)
- UI Label: `Extra Vacuum`
- 字段映射 (Field mapping): 序列化键 `extra_vacuum` <-> 界面标签 `Extra Vacuum`。
- 控件标签 (Caption): `Extra Vacuum`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): 额外真空层 (extra vacuum)。
- 对输出规模/物理性的影响: 增大可降低镜像相互作用。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 推荐范围 (Recommended range):
  - 保守：0-5Å
  - 平衡：5-15Å
  - 探索：20Å+ 强隔离

### `layers` (Layers)
- UI Label: `Layers`
- 字段映射 (Field mapping): 序列化键 `layers` <-> 界面标签 `Layers`。
- 控件标签 (Caption): `Layers`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[3]`
- 含义 (Meaning): 层参数 (layer index/count)。
- 对输出规模/物理性的影响: 控制操作层位或层数覆盖。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 推荐范围 (Recommended range):
  - 保守：2-3
  - 平衡：3-6
  - 探索：6-15

### `distance` (Distance)
- UI Label: `Distance`
- 字段映射 (Field mapping): 序列化键 `distance` <-> 界面标签 `Distance`。
- 控件标签 (Caption): `Distance`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[3.0]`
- 含义 (Meaning): 距离参数 (distance parameter)。
- 对输出规模/物理性的影响: 过小会导致碰撞，过大会稀释作用强度。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 推荐范围 (Recommended range):
  - 保守：2-3
  - 平衡：3-6
  - 探索：6-15

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "LayerCopyCard",
  "check_state": true,
  "preset_index": 1,
  "dz_expr": "sin(x/pi) + sin(y/pi)",
  "params": "",
  "apply_mode": 0,
  "elements": "",
  "z_range": [
    -5,
    5,
    1
  ],
  "wrap": false,
  "extend_cell_z": true,
  "extra_vacuum": [
    0.0
  ],
  "layers": [
    3
  ],
  "distance": [
    3.0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "LayerCopyCard",
  "check_state": true,
  "preset_index": 1,
  "dz_expr": "sin(x/pi) + sin(y/pi)",
  "params": "",
  "apply_mode": 0,
  "elements": "",
  "z_range": [
    -5,
    5,
    1
  ],
  "wrap": false,
  "extend_cell_z": true,
  "extra_vacuum": [
    0.0
  ],
  "layers": [
    3
  ],
  "distance": [
    3.0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "LayerCopyCard",
  "check_state": true,
  "preset_index": 1,
  "dz_expr": "sin(x/pi) + sin(y/pi)",
  "params": "",
  "apply_mode": 0,
  "elements": "",
  "z_range": [
    -5,
    5,
    1
  ],
  "wrap": false,
  "extend_cell_z": true,
  "extra_vacuum": [
    0.0
  ],
  "layers": [
    3
  ],
  "distance": [
    3.0
  ]
}
```

## 推荐组合
- Layer Copy -> Insert Defect: 先构建层间形变结构，再采样插隙/吸附位点。
- 层操作前先确认输入真的具有明确层状或几何分区特征。
- 涉及表面或界面时，复制或重排层之后要重新检查真空层和层间重叠。

## 常见问题与排查
- 输出为空时，先检查输入是否真的有可识别的层或区域，以及表达式或过滤范围是否命中了这些层。
- 如果复制或重排后的结构不合理，优先检查层间距离、真空高度和是否出现层重叠。
- 结构类卡片通常不会替你自动修复重叠或真空不足；参数激进时仍会按当前配置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `SWC(L={...},dz={...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
