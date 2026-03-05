<!-- card-schema: {"card_name": "Crystal Prototype Builder", "source_file": "src/NepTrainKit/ui/views/_card/crystal_prototype_builder_card.py", "serialized_keys": ["lattice", "element", "a_range", "covera", "auto_supercell", "max_atoms", "rep", "max_outputs"]} -->

# 晶体原型构建（Crystal Prototype Builder）

`Group`: `Lattice`  
`Class`: `CrystalPrototypeBuilderCard`  
`Source`: `src/NepTrainKit/ui/views/_card/crystal_prototype_builder_card.py`

## 功能说明
按晶型原型和晶格常数范围生成标准晶体起始结构，快速搭建可控的基础结构库。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 缺少标准原型结构，训练集拓扑单一。
- 目标任务 (Target objective): 构建 clean prototype baseline。
- 建议添加条件 (Add-it trigger): 需要系统对比 fcc/bcc/hcp 等晶型。
- 不建议添加条件 (Avoid trigger): 已有充分真实结构且无需原型补充。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 确认 `lattice` 与元素组合物理可行。
- 设好 `max_outputs` 避免网格过密。


## 参数说明（完整）
### `lattice` (Lattice)
- UI Label: `Lattice`
- 字段映射 (Field mapping): 序列化键 `lattice` <-> 界面标签 `Lattice`。
- 控件标签 (Caption): `Lattice`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"fcc"`
- 含义 (Meaning): 晶型模板 (lattice prototype)。
- 对输出规模/物理性的影响: 决定生成结构的拓扑基底。
- 推荐范围 (Recommended range):
  - 保守：主晶型
  - 平衡：主+次晶型
  - 探索：全晶型需预算限制

### `element` (Element)
- UI Label: `Element`
- 字段映射 (Field mapping): 序列化键 `element` <-> 界面标签 `Element`。
- 控件标签 (Caption): `Element`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Cu"`
- 含义 (Meaning): 元素类型 (element type)。
- 对输出规模/物理性的影响: 定义当前操作核心元素。
- 配置建议 (Practical note): `Element` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `a_range` (A Range)
- UI Label: `A Range`
- 字段映射 (Field mapping): 序列化键 `a_range` <-> 界面标签 `A Range`。
- 控件标签 (Caption): `A Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[3.6, 3.6, 0.1]`
- 含义 (Meaning): 晶格常数范围 (lattice constant range)。
- 对输出规模/物理性的影响: 控制结构尺寸扫描范围。
- 推荐范围 (Recommended range):
  - 保守：3.6 到 3.6，step 0.1
  - 平衡：3.6 到 3.6，step 0.05
  - 探索：3.6 到 3.6，step 0.2

### `covera` (Covera)
- UI Label: `Covera`
- 字段映射 (Field mapping): 序列化键 `covera` <-> 界面标签 `Covera`。
- 控件标签 (Caption): `Covera`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[1.633]`
- 含义 (Meaning): c/a 比例 (c over a ratio)。
- 对输出规模/物理性的影响: 影响非立方晶型的几何各向异性。
- 推荐范围 (Recommended range):
  - 保守：1.14-1.63
  - 平衡：1.63-2.45
  - 探索：2.45-4.08

### `auto_supercell` (Auto Supercell)
- UI Label: `Auto Supercell`
- 字段映射 (Field mapping): 序列化键 `auto_supercell` <-> 界面标签 `Auto Supercell`。
- 控件标签 (Caption): `Auto Supercell`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 自动扩胞开关 (auto supercell)。
- 对输出规模/物理性的影响: 自动根据目标规模调整复制参数。
- 配置建议 (Practical note):
  - 开启：需要启用 `Auto Supercell` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `max_atoms` (Max Atoms)
- UI Label: `Max Atoms`
- 字段映射 (Field mapping): 序列化键 `max_atoms` <-> 界面标签 `Max Atoms`。
- 控件标签 (Caption): `Max Atoms`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[512]`
- 含义 (Meaning): 最大原子数 (maximum atoms)。
- 对输出规模/物理性的影响: 限制单结构规模，避免超算力预算。
- 推荐范围 (Recommended range):
  - 保守：256-512
  - 平衡：512-1024
  - 探索：1024-2560

### `rep` (Rep)
- UI Label: `Rep`
- 字段映射 (Field mapping): 序列化键 `rep` <-> 界面标签 `Rep`。
- 控件标签 (Caption): `Rep`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[4, 4, 4]`
- 含义 (Meaning): 复制倍率向量 (replication vector)。
- 对输出规模/物理性的影响: 直接决定扩胞倍数和原子数增长。
- 推荐范围 (Recommended range):
  - 保守：1x1x1 到 2x2x2
  - 平衡：2x2x2 到 3x3x3
  - 探索：3x3x3 到 5x5x5

### `max_outputs` (Max Outputs)
- UI Label: `Max Outputs`
- 字段映射 (Field mapping): 序列化键 `max_outputs` <-> 界面标签 `Max Outputs`。
- 控件标签 (Caption): `Max Outputs`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[200]`
- 含义 (Meaning): 输出上限 (maximum outputs)。
- 对输出规模/物理性的影响: 限制样本规模，防止组合爆炸。
- 推荐范围 (Recommended range):
  - 保守：100-200
  - 平衡：200-400
  - 探索：400-1000


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "CrystalPrototypeBuilderCard",
  "check_state": true,
  "lattice": "fcc",
  "element": "Cu",
  "a_range": [
    3.6,
    3.6,
    0.1
  ],
  "covera": [
    1.633
  ],
  "auto_supercell": true,
  "max_atoms": [
    512
  ],
  "rep": [
    4,
    4,
    4
  ],
  "max_outputs": [
    10
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "CrystalPrototypeBuilderCard",
  "check_state": true,
  "lattice": "fcc",
  "element": "Cu",
  "a_range": [
    3.6,
    3.6,
    0.1
  ],
  "covera": [
    1.633
  ],
  "auto_supercell": true,
  "max_atoms": [
    512
  ],
  "rep": [
    4,
    4,
    4
  ],
  "max_outputs": [
    200
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "CrystalPrototypeBuilderCard",
  "check_state": true,
  "lattice": "fcc",
  "element": "Cu",
  "a_range": [
    3.6,
    3.6,
    0.1
  ],
  "covera": [
    1.633
  ],
  "auto_supercell": true,
  "max_atoms": [
    512
  ],
  "rep": [
    4,
    4,
    4
  ],
  "max_outputs": [
    600
  ]
}
```


## 推荐组合
- Crystal Prototype Builder -> Composition Sweep -> Random Occupancy: 先生成干净模板，再进行成分修饰。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 生成失败：检查 `a_range`、`rep` 是否产生非法晶胞。
- 规模过大：降低 `max_atoms` 或启用自动扩胞策略。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Proto({...},a={...},rep={...}x{...}x{...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
