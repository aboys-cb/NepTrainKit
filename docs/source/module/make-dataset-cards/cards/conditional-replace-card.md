<!-- card-schema: {"card_name": "Conditional Replace", "source_file": "src/NepTrainKit/ui/views/_card/conditional_replace_card.py", "serialized_keys": ["target", "replacements", "condition", "seed", "mode"]} -->

# 条件替换（Conditional Replace）

`Group`: `Alloy`  
`Class`: `ConditionalReplaceCard`  
`Source`: `src/NepTrainKit/ui/views/_card/conditional_replace_card.py`

## 功能说明
按空间表达式对目标元素执行条件替换（conditional replacement），构建区域选择性化学改性样本。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 需要只在表层/局域区域替换元素。
- 目标任务 (Target objective): 增强局域化学环境变化覆盖。
- 建议添加条件 (Add-it trigger): 可以用 `x/y/z` 明确写出作用区域。
- 不建议添加条件 (Avoid trigger): 仅需全局替换，Random Doping 更直接。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先验证 `replacements` 语法正确。
- 先用 `condition=all` 验证路径，再收紧条件。


## 参数说明（完整）
### `target` (Target element)
- UI Label: `Target element`
- 字段映射 (Field mapping): 序列化键 `target` <-> 界面标签 `Target element`。
- 控件标签 (Caption): `Target element`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 目标元素 (target species)。
- 对输出规模/物理性的影响: 限定被替换或处理的原子种类。
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
- 推荐范围 (Recommended range):
  - 保守：默认模式先验证
  - 平衡：按任务切换
  - 探索：探索模式配审计

### 替换输入 Schema (Replacement input schema)
- `replacements` 支持 `Co:0.7,Ni:0.3` 或 JSON dict 字符串。
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
 
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 替换数为 0：通常是 `condition` 未命中目标元素。
- 替换比例偏差大：检查 `mode`（Random vs Exact ratio）。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Repl({...}->{...})`


## 可复现性说明
- `seed=[0]` 一般表示随机路径；固定非零值用于可重复对比。
- 输入顺序变化会改变样本对应关系。
