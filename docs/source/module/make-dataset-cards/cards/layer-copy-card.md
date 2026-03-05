<!-- card-schema: {"card_name": "Layer Copy", "source_file": "src/NepTrainKit/ui/views/_card/layer_copy_card.py", "serialized_keys": ["preset_index", "dz_expr", "params", "apply_mode", "elements", "z_range", "wrap", "extend_cell_z", "extra_vacuum", "layers", "distance"]} -->

# 层复制（Layer Copy）

`Group`: `Structure`  
`Class`: `LayerCopyCard`  
`Source`: `src/NepTrainKit/ui/views/_card/layer_copy_card.py`

## 功能说明
复制层并按 `dz_expr` 施加位移调制，生成层间错位、起伏和堆叠变化数据。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 层状体系样本单一，层间自由度覆盖不足。
- 目标任务 (Target objective): 增强层间几何变化与形貌多样性。
- 建议添加条件 (Add-it trigger): 研究二维材料、多层异质结、层间耦合。
- 不建议添加条件 (Avoid trigger): 非层状体相体系。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先在单帧验证 `dz_expr` 与 `params`。
- 按边界条件选择 `wrap/extend_cell_z/extra_vacuum`。


## 参数说明（完整）
### `preset_index` (Preset Index)
- UI Label: `Preset Index`
- 字段映射 (Field mapping): 序列化键 `preset_index` <-> 界面标签 `Preset Index`。
- 控件标签 (Caption): `Preset Index`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `1`
- 含义 (Meaning): 预设索引 (preset index)。
- 对输出规模/物理性的影响: 选择内置变换模板。
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
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 层间冲突：收窄 `z_range` 并增大 `distance`。
- 边界异常：核对 `wrap` 与 `extend_cell_z` 配置。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `SWC(L={...},dz={...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
