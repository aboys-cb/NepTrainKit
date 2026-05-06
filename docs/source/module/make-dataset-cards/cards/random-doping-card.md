<!-- card-schema: {"card_name": "Random Doping", "source_file": "src/NepTrainKit/ui/views/_card/random_doping_card.py", "serialized_keys": ["params", "rules", "doping_type", "max_atoms_condition", "use_seed", "seed"]} -->

# 随机掺杂（Random Doping）

`Group`: `Alloy`  
`Class`: `RandomDopingCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_doping_card.py`

## 功能说明
依据规则表执行替位掺杂（substitutional doping），可选随机采样或比例精确分配。

它最适合的场景是：对指定元素位点做一次或多次随机掺杂，补充局部化学环境差异。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：对指定元素位点做一次或多次随机掺杂，补充局部化学环境差异

**输入：** 一个母相结构和明确的掺杂规则

**目标：** 快速得到具体的随机合金/掺杂落点，而不是系统扫描完整配比空间

**参数设置：**
- `rules` 写清 target、dopants 和 count/percent 规则
- `doping_type="Exact"` 更适合可复现实验
- `max_atoms_condition` 控制每帧额外生成多少个掺杂版本

**输出：** 每个输入结构会扩出若干真实落位的掺杂版本，并追加掺杂标签

**怎么验证结果合理：**
- 检查被替换元素的数量是否与规则一致
- 确认元素统计和 `Config_type` 标签匹配
- 若结果没有发生变化，先检查 `rules` 是否为空或 target 是否没命中

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 掺杂浓度和成分覆盖不足。
- 目标任务 (Target objective): 构建可控掺杂比例和位点分布样本。
- 建议添加条件 (Add-it trigger): 已明确 target 元素与 dopant 组合。
- 不建议添加条件 (Avoid trigger): 只需全局占位随机化。
> 物理提示 (Physics caution): 重点检查目标配比、实际元素统计和标签是否一致，避免“标签写对了、占位落错了”。

## 输入前提
- 至少配置一条可解析规则。
- 先用窄浓度区间做正确性验证。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由 rules、doping mode、输出数量和 seed 控件组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"rules": [], "doping_type": "Random", "max_structures": 1, "use_seed": false, "seed": 0}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组随机掺杂规则。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

### `rules` (Rules)
- UI Label: `Rules`
- 字段映射 (Field mapping): 序列化键 `rules` <-> 界面标签 `Rules`。
- 控件标签 (Caption): `Rules`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string (JSON list)
- 默认值 (Default): `"[]"`
- 含义 (Meaning): 掺杂规则表 (doping rules)，字段含 `target/dopants/use/percent/count/group`。
- 对输出规模/物理性的影响: 决定替换对象、替换比例和局域范围，是化学分布主控参数。
- 参数联动 / 生效条件: 这张卡的主控输入就是规则列表；规则为空时通常不会产生真正的替换结果。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): 在界面里优先把每条 rule 的 `dopants` 写成 `Ge:0.7,C:0.3` 这类逗号+冒号字符串；程序会先解析成内部 dict，再写入导出的 card JSON。

### `doping_type` (Doping Type)
- UI Label: `Doping Type`
- 字段映射 (Field mapping): 序列化键 `doping_type` <-> 界面标签 `Doping Type`。
- 控件标签 (Caption): `Doping Type`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"Random"`
- 含义 (Meaning): 掺杂采样类型 (doping type)。
- 对输出规模/物理性的影响: Random 强随机性，Exact 更接近目标比例。
- 参数联动 / 生效条件: `Random` 更偏向统计分布，`Exact` 更适合希望每次落点数量更稳定的对比实验。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：Exact 基线
  - 平衡：Exact+Random 对比
  - 探索：Random 探索扩展

### `max_atoms_condition` (Max Atoms Condition)
- UI Label: `Max Atoms Condition`
- 字段映射 (Field mapping): 序列化键 `max_atoms_condition` <-> 界面标签 `Max Atoms Condition`。
- 控件标签 (Caption): `Max Atoms Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 每帧最大生成数 (max generated structures per frame)。
- 对输出规模/物理性的影响: 主要控制数据量和运行时间。
- 参数联动 / 生效条件: 这里的含义是“每帧生成多少个结构版本”，不是控制结构中的最大原子数。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：10-50
  - 平衡：50-200
  - 探索：200+ 需 FPS

### `use_seed` (Use Seed)
- UI Label: `Use Seed`
- 字段映射 (Field mapping): 序列化键 `use_seed` <-> 界面标签 `Use Seed`。
- 控件标签 (Caption): `Use Seed`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 是否启用固定随机种子 (deterministic seed switch)。
- 对输出规模/物理性的影响: 开启后可复现实验；关闭后每次采样分布会变化。
- 怎么判断该开还是该关: 做可复现实验或要对比不同参数时开启；纯探索阶段可以先关闭以增加随机覆盖。
- 配置建议 (Practical note):
  - 开启：需要可复现对比时开启。
  - 关闭：探索阶段可关闭以增加随机覆盖。

### `seed` (Seed)
- UI Label: `Seed`
- 字段映射 (Field mapping): 序列化键 `seed` <-> 界面标签 `Seed`。
- 控件标签 (Caption): `Seed`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[0]`
- 含义 (Meaning): 随机种子值 (random seed value)。
- 对输出规模/物理性的影响: 只影响随机路径，不改变物理模型本身。
- 参数联动 / 生效条件: `seed` 只有在 `use_seed=true` 时才真正固定随机路径。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：0（随机）
  - 平衡：1-99（可复现）
  - 探索：100-9999（多 seed 对比）

### 规则输入 Schema (Rule input schema)
`rules` 在 card JSON 中仍保存为 JSON 字符串，但界面里每条规则建议按“容易输入、不容易写坏”的字符串语法填写，而不是手写 JSON。
- `target` (string): 被替换元素，例如 `Si`。
- `dopants` (string, recommended): 推荐写成 `Ge:0.7,C:0.3`；逗号分隔元素，冒号分隔元素与权重。解析成功后，导出的 card JSON 会把它存成内部 object。
- `use` (string): `atomic_percent`、`mass_percent` 或 `count`。
- `percent` (list[2]): 百分比区间，单位是 `%`，例如 `[3,8]` 表示从候选位点中抽取约 3% 到 8%。
- `count` (list[2]): 替换数量区间。
- `group` (string or list[string], optional): 界面里推荐写成 `surface_top,surface_bottom` 这种逗号分隔名称；运行时会匹配 `atoms.arrays['group']`。如果输入来自 `.xyz`，请使用 EXTXYZ 风格的 `group` 列，而不是普通三列 XYZ。

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "rules": "[{\"target\":\"Si\",\"dopants\":{\"Ge\":1.0},\"use\":\"concentration\",\"concentration\":[0.01,0.02],\"count\":[1,1]}]",
  "doping_type": "Exact",
  "max_atoms_condition": [
    20
  ],
  "use_seed": true,
  "seed": [
    101
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "rules": "[{\"target\":\"Si\",\"dopants\":{\"Ge\":0.7,\"C\":0.3},\"use\":\"concentration\",\"concentration\":[0.03,0.08],\"count\":[1,1]}]",
  "doping_type": "Exact",
  "max_atoms_condition": [
    20
  ],
  "use_seed": true,
  "seed": [
    101
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "rules": "[{\"target\":\"Si\",\"dopants\":{\"Ge\":0.5,\"C\":0.3,\"Sn\":0.2},\"use\":\"concentration\",\"concentration\":[0.08,0.2],\"count\":[1,1]}]",
  "doping_type": "Random",
  "max_atoms_condition": [
    20
  ],
  "use_seed": true,
  "seed": [
    101
  ]
}
```

## 推荐组合
- Group Label -> Random Doping: 按 group 定向作用于特定子晶格/层。
- Composition Sweep -> Random Occupancy -> Random Doping: 先做成分展开，再做位点定向替位。
- 先明确“目标配比”还是“具体落位”，再决定接 `Composition Sweep`、`Random Occupancy` 还是 `Random Doping`。

## 常见问题与排查
- 结果没有变化时，先检查目标元素、规则字符串或 `Comp tag` 来源是否真的命中了输入结构。
- 如果输出成分偏离预期，优先检查是“目标配比定义”问题，还是“离散落位/随机替换”步骤把分布拉偏。
- 这组卡片不会自动替你决定工作流分工；需要系统扫配比时先用 `Composition Sweep`，需要真实落位时再接 `Random Occupancy` 或 `Random Doping`。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Dop(n={...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
