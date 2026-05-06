<!-- card-schema: {"card_name": "Crystal Prototype Builder", "source_file": "src/NepTrainKit/ui/views/_card/crystal_prototype_builder_card.py", "serialized_keys": ["params"]} -->

# 晶体原型构建（Crystal Prototype Builder）

`Group`: `Lattice`  
`Class`: `CrystalPrototypeBuilderCard`  
`Source`: `src/NepTrainKit/ui/views/_card/crystal_prototype_builder_card.py`

## 功能说明
按晶型原型和晶格常数范围生成标准晶体起始结构，快速搭建可控的基础结构库。

它最适合的场景是：从 fcc / bcc / hcp 等标准原型直接生成一批统一格式的母相结构。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：从 fcc / bcc / hcp 等标准原型直接生成一批统一格式的母相结构

**输入：** 目标晶型、元素和晶格常数范围，例如 Cu-fcc 或 Fe-bcc

**目标：** 先构造一批可控的基础结构，再把它们作为缺陷、表面或磁性流程的输入

**参数设置：**
- `lattice` 先选原型类型
- `a_range` 控制晶格常数扫描
- `auto_supercell` 与 `max_atoms` 一起决定输出尺寸

**输出：** 一批标准化原型结构，适合作为后续 Make Dataset 流程的起点

**怎么验证结果合理：**
- 检查原子数和对称性是否符合原型预期
- 确认 `auto_supercell` 没有把尺寸扩到超预算
- 下游要做表面或缺陷时，优先先在这里把母相建对

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 缺少标准原型结构，训练集拓扑单一。
- 目标任务 (Target objective): 构建 clean prototype baseline。
- 建议添加条件 (Add-it trigger): 需要系统对比 fcc/bcc/hcp 等晶型。
- 不建议添加条件 (Avoid trigger): 已有充分真实结构且无需原型补充。
> 物理提示 (Physics caution): 重点检查体积变化、晶胞条件数和最近邻距离，避免把几何畸变直接放大到非物理区间。

## 输入前提
- 确认 `lattice` 与元素组合物理可行。
- 设好 `max_outputs` 避免网格过密。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 核心操作参数 `CrystalPrototypeBuilderParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"lattice": "fcc", "element": "Cu", "a_range": [3.6, 3.6, 0.1], "covera": 1.633, "auto_supercell": true, "max_atoms": 512, "rep": [4, 4, 4], "max_outputs": 200}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的晶格类型、元素、晶格常数、扩胞和最大输出字段一致。
- 配置建议 (Practical note): 新版本优先读取 `params`，旧字段仍保留用于兼容已有 workflow。

### `lattice` (Lattice)
- UI Label: `Lattice`
- 字段映射 (Field mapping): 序列化键 `lattice` <-> 界面标签 `Lattice`。
- 控件标签 (Caption): `Lattice`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"fcc"`
- 含义 (Meaning): 晶型模板 (lattice prototype)。
- 对输出规模/物理性的影响: 决定生成结构的拓扑基底。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
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
- 作为后续缺陷、表面或磁性卡片的母胞准备步骤。
- 若扩胞后结构规模明显上升，建议在流程末端再接 `FPS Filter` 控制代表性样本数。

## 常见问题与排查
- 输出为空或远少于预期时，先检查各范围参数是否真的形成了有效扫描组合；很多这类卡片在参数只给定单点时只会输出很少的结构。
- 如果结构明显不合理，先看体积、晶胞角和最近邻距离，再把主控幅度或步长回调到更小的量级。
- 模式冲突时以当前 UI 状态和代码分支为准；导入旧 JSON 后如果发现多个主模式字段同时存在，建议手工确认只保留一条主路径。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Proto({...},a={...},rep={...}x{...}x{...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
