<!-- card-schema: {"card_name": "Random Doping", "source_file": "src/NepTrainKit/ui/views/_card/random_doping_card.py", "serialized_keys": ["rules", "doping_type", "max_atoms_condition", "use_seed", "seed"]} -->

# 随机掺杂（Random Doping）

`Group`: `Alloy`  
`Class`: `RandomDopingCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_doping_card.py`

## 功能说明
依据规则表执行替位掺杂（substitutional doping），可选随机采样或比例精确分配。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 掺杂浓度和成分覆盖不足。
- 目标任务 (Target objective): 构建可控掺杂比例和位点分布样本。
- 建议添加条件 (Add-it trigger): 已明确 target 元素与 dopant 组合。
- 不建议添加条件 (Avoid trigger): 只需全局占位随机化。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 至少配置一条可解析规则。
- 先用窄浓度区间做正确性验证。


## 参数说明（完整）
### `rules` (Rules)
- UI Label: `Rules`
- 字段映射 (Field mapping): 序列化键 `rules` <-> 界面标签 `Rules`。
- 控件标签 (Caption): `Rules`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string (JSON list)
- 默认值 (Default): `"[]"`
- 含义 (Meaning): 掺杂规则表 (doping rules)，字段含 `target/dopants/use/concentration/count/group`。
- 对输出规模/物理性的影响: 决定替换对象、替换比例和局域范围，是化学分布主控参数。
- 配置建议 (Practical note): 按规则语法填写，建议先单规则单帧验证后再扩展。

### `doping_type` (Doping Type)
- UI Label: `Doping Type`
- 字段映射 (Field mapping): 序列化键 `doping_type` <-> 界面标签 `Doping Type`。
- 控件标签 (Caption): `Doping Type`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"Random"`
- 含义 (Meaning): 掺杂采样类型 (doping type)。
- 对输出规模/物理性的影响: Random 强随机性，Exact 更接近目标比例。
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
- 推荐范围 (Recommended range):
  - 保守：0（随机）
  - 平衡：1-99（可复现）
  - 探索：100-9999（多 seed 对比）

### 规则输入 Schema (Rule input schema)
`rules` 在配置中保存为 JSON 字符串，语义为 rule object 列表。
- `target` (string): 被替换元素。
- `dopants` (object): 掺杂元素及权重，例如 `{"Ge":0.7,"C":0.3}`。
- `use` (string): `concentration` 或 `count`。
- `concentration` (list[2]): 浓度区间。
- `count` (list[2]): 替换数量区间。
- `group` (list[string], optional): 仅作用于指定 group。


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


## 常见问题与排查
- 规则未生效：检查 `target/dopants/use` 字段。
- 比例偏差：切换 `doping_type=Exact` 并收窄区间。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Dop(n={...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
