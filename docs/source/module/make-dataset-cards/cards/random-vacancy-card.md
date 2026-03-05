<!-- card-schema: {"card_name": "Random Vacancy", "source_file": "src/NepTrainKit/ui/views/_card/random_vacancy_card.py", "serialized_keys": ["rules", "max_atoms_condition", "use_seed", "seed"]} -->

# 随机空位（Random Vacancy）

`Group`: `Defect`  
`Class`: `RandomVacancyCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_vacancy_card.py`

## 功能说明
根据规则删除指定元素原子（rule-based vacancy），控制空位元素类型、数量和区域。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 空位缺陷覆盖不足或分布不可控。
- 目标任务 (Target objective): 精确控制空位类型与局域分布。
- 建议添加条件 (Add-it trigger): 需要按元素和 group 定向删原子。
- 不建议添加条件 (Avoid trigger): 仅需无规则随机空位。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先单规则验证，再叠加多规则。
- 使用 group 时确认输入包含 group 数组。


## 参数说明（完整）
### `rules` (Rules)
- UI Label: `Rules`
- 字段映射 (Field mapping): 序列化键 `rules` <-> 界面标签 `Rules`。
- 控件标签 (Caption): `Rules`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string (JSON list)
- 默认值 (Default): `"[]"`
- 含义 (Meaning): 空位规则表 (vacancy rules)，字段含 `element/count/group`。
- 对输出规模/物理性的影响: 控制删原子元素与密度分布。
- 配置建议 (Practical note): 按规则语法填写，建议先单规则单帧验证后再扩展。

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
`rules` 在配置中保存为 JSON 字符串，语义为 vacancy rule 列表。
- `element` (string): 删除目标元素。
- `count` (list[2]): 删除数量区间。
- `group` (list[string], optional): 仅作用于指定 group。


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "RandomVacancyCard",
  "check_state": true,
  "rules": "[{\"element\":\"O\",\"count\":[1,1]}]",
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
  "class": "RandomVacancyCard",
  "check_state": true,
  "rules": "[{\"element\":\"O\",\"count\":[1,3]}]",
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
  "class": "RandomVacancyCard",
  "check_state": true,
  "rules": "[{\"element\":\"O\",\"count\":[2,6]},{\"element\":\"Li\",\"count\":[1,3],\"group\":[\"surface_top\"]}]",
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
- Random Slab -> Random Vacancy: 在显式元素/group 控制下构建表面空位数据集。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 删除总是 0：检查 `element` 和 `count`。
- 结构过度破坏：收窄规则计数范围。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Vac(n={...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
