<!-- card-schema: {"card_name": "Conditional Replace", "source_file": "src/NepTrainKit/ui/views/_card/conditional_replace_card.py", "serialized_keys": ["params"]} -->

# 条件替换（Conditional Replace）

`Group`: `Alloy`  
`Class`: `ConditionalReplaceCard`  
`Source`: `src/NepTrainKit/ui/views/_card/conditional_replace_card.py`

## 功能说明
按空间表达式对目标元素执行条件替换（conditional replacement），构建区域选择性化学改性样本。

它最适合的场景是：只替换满足局部规则的位点，而不是对所有候选原子做无差别随机替换。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：只替换满足局部规则的位点，而不是对所有候选原子做无差别随机替换

**输入：** 一个含目标元素的结构，以及明确的替换规则或条件表达式

**目标：** 把“只换表面 O”或“只换某一 group 内位点”这类条件约束写进替换流程

**参数设置：**
- `target` 写要被替换的原始元素或位点条件
- `replacements` 写候选替换元素及比例
- `condition` 用来限制哪些位点允许参与

**输出：** 只有命中条件的位点会被替换；未命中的位置保持原样

**怎么验证结果合理：**
- 抽查被替换的原子是否真的满足条件
- 若结果和原结构完全一样，先检查 `condition` 是否过严
- 确认替换后标签和元素统计与预期一致

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 需要只在表层/局域区域替换元素。
- 目标任务 (Target objective): 增强局域化学环境变化覆盖。
- 建议添加条件 (Add-it trigger): 可以用 `x/y/z` 明确写出作用区域。
- 不建议添加条件 (Avoid trigger): 仅需全局替换，Random Doping 更直接。
> 物理提示 (Physics caution): 重点检查目标配比、实际元素统计和标签是否一致，避免“标签写对了、占位落错了”。

## 输入前提
- 先验证 `replacements` 语法正确。
- 先用 `condition=all` 验证路径，再收紧条件。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 核心操作参数 `ConditionalReplaceParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"target": "", "replacements": "", "condition": "", "seed": 0, "mode": 0}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的目标元素、替换配方、空间条件和随机种子字段一致。
- 配置建议 (Practical note): 新版本优先读取 `params`，旧字段仍保留用于兼容已有 workflow。

### `target` (Target element)
- UI Label: `Target element`
- 字段映射 (Field mapping): 序列化键 `target` <-> 界面标签 `Target element`。
- 控件标签 (Caption): `Target element`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 目标元素 (target species)。
- 对输出规模/物理性的影响: 限定被替换或处理的原子种类。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): `Target element` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `replacements` (Replacements)
- UI Label: `Replacements`
- 字段映射 (Field mapping): 序列化键 `replacements` <-> 界面标签 `Replacements`。
- 控件标签 (Caption): `Replacements`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string (`elem:ratio` list or JSON dict)
- 默认值 (Default): `""`
- 含义 (Meaning): 替换配方 (replacement map)，支持 `A:0.7,B:0.3` 或 JSON dict。
- 对输出规模/物理性的影响: 决定替换后化学组成分布。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 按映射语法填写，建议先用简单映射验证后再扩展。

### `condition` (Condition)
- UI Label: `Condition`
- 字段映射 (Field mapping): 序列化键 `condition` <-> 界面标签 `Condition`。
- 控件标签 (Caption): `Condition`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): string boolean expression on `x,y,z`
- 默认值 (Default): `""`
- 含义 (Meaning): 空间条件表达式 (condition expression)，支持 x/y/z 与逻辑运算。
- 对输出规模/物理性的影响: 命中区域越宽，替换越全局；越窄，越局域。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note):
  - 开启：需要启用 `Condition` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

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

### `mode` (Mode)
- UI Label: `Mode`
- 字段映射 (Field mapping): 序列化键 `mode` <-> 界面标签 `Mode`。
- 控件标签 (Caption): `Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `0`
- 含义 (Meaning): 操作模式 (operation mode)。
- 对输出规模/物理性的影响: 改变执行逻辑路径，影响样本分布。
- 参数联动 / 生效条件: 它决定当前工作流走哪条主分支；先选模式，再填写与该模式对应的字段。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：默认模式先验证
  - 平衡：按任务切换
  - 探索：探索模式配审计

### 替换输入 Schema (Replacement input schema)
- `replacements` 优先写成 `Co:0.7,Ni:0.3` 这种逗号+冒号字符串；只有在你明确需要回填内部序列化结果时，才考虑 JSON dict 字符串。
- `condition` 支持 `x/y/z` 与 `and/or/not` 逻辑表达式。
- 建议先用 `all` 验证替换路径，再收紧局域条件。

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "ConditionalReplaceCard",
  "check_state": true,
  "target": "O",
  "replacements": "F:1.0",
  "condition": "z>=8 and z<=10",
  "seed": [
    101
  ],
  "mode": 1
}
```

### 平衡（Balanced）
```json
{
  "class": "ConditionalReplaceCard",
  "check_state": true,
  "target": "O",
  "replacements": "F:0.7,N:0.3",
  "condition": "z>=6 and z<=14",
  "seed": [
    101
  ],
  "mode": 0
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "ConditionalReplaceCard",
  "check_state": true,
  "target": "O",
  "replacements": "F:0.4,N:0.3,Cl:0.3",
  "condition": "all",
  "seed": [
    101
  ],
  "mode": 0
}
```

## 推荐组合
- Conditional Replace -> Random Doping: 先做几何门控替换，再做更广泛替位采样。
- 先明确“目标配比”还是“具体落位”，再决定接 `Composition Sweep`、`Random Occupancy` 还是 `Random Doping`。
- 涉及 group 或局部位点限制时，可先接 `Group Label` 或规则类卡片再执行替换。

## 常见问题与排查
- 结果没有变化时，先检查目标元素、规则字符串或 `Comp tag` 来源是否真的命中了输入结构。
- 如果输出成分偏离预期，优先检查是“目标配比定义”问题，还是“离散落位/随机替换”步骤把分布拉偏。
- 这组卡片不会自动替你决定工作流分工；需要系统扫配比时先用 `Composition Sweep`，需要真实落位时再接 `Random Occupancy` 或 `Random Doping`。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Repl({...}->{...})`

## 可复现性说明
- `seed=[0]` 一般表示随机路径；固定非零值用于可重复对比。
- 输入顺序变化会改变样本对应关系。
