<!-- card-schema: {"card_name": "Random Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/random_occupancy_card.py", "serialized_keys": ["source", "manual", "mode", "samples", "group_filter", "use_seed", "seed"]} -->

# 随机占位（Random Occupancy）

`Group`: `Alloy`  
`Class`: `RandomOccupancyCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_occupancy_card.py`

## 功能说明
在给定总成分约束下随机分配位点元素（site occupancy assignment），用于同成分多排布样本扩展。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 同成分下占位排列单一，迁移泛化差。
- 目标任务 (Target objective): 增加位点排列多样性而保持总体成分。
- 建议添加条件 (Add-it trigger): 高熵或多元固溶体占位采样任务。
- 不建议添加条件 (Avoid trigger): 不需要占位随机化。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 确认 `source` 与成分字符串格式。
- 若使用 group 过滤，结构需已有 group 数组。


## 参数说明（完整）
### `source` (Source)
- UI Label: `Source`
- 字段映射 (Field mapping): 序列化键 `source` <-> 界面标签 `Source`。
- 控件标签 (Caption): `Source`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"Auto (Comp tag)"`
- 含义 (Meaning): 成分来源 (composition source)。
- 对输出规模/物理性的影响: 决定自动读取还是手工输入。
- 推荐范围 (Recommended range):
  - 保守：自动优先
  - 平衡：手工兜底
  - 探索：双来源交叉核验

### `manual` (Manual)
- UI Label: `Manual`
- 字段映射 (Field mapping): 序列化键 `manual` <-> 界面标签 `Manual`。
- 控件标签 (Caption): `Manual`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 手动成分字符串 (manual composition)。
- 对输出规模/物理性的影响: 用于显式指定元素比例。
- 配置建议 (Practical note): `Manual` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `mode` (Mode)
- UI Label: `Mode`
- 字段映射 (Field mapping): 序列化键 `mode` <-> 界面标签 `Mode`。
- 控件标签 (Caption): `Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"Exact"`
- 含义 (Meaning): 操作模式 (operation mode)。
- 对输出规模/物理性的影响: 改变执行逻辑路径，影响样本分布。
- 推荐范围 (Recommended range):
  - 保守：默认模式先验证
  - 平衡：按任务切换
  - 探索：探索模式配审计

### `samples` (Samples)
- UI Label: `Samples`
- 字段映射 (Field mapping): 序列化键 `samples` <-> 界面标签 `Samples`。
- 控件标签 (Caption): `Samples`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 每帧样本数 (samples per frame)。
- 对输出规模/物理性的影响: 控制输出体量和统计稳定性。
- 推荐范围 (Recommended range):
  - 保守：1-3
  - 平衡：5-10
  - 探索：20+ 需去重

### `group_filter` (Group Filter)
- UI Label: `Group Filter`
- 字段映射 (Field mapping): 序列化键 `group_filter` <-> 界面标签 `Group Filter`。
- 控件标签 (Caption): `Group Filter`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 分组过滤条件 (group filter)。
- 对输出规模/物理性的影响: 限制操作仅作用于指定 group。
- 配置建议 (Practical note): `Group Filter` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

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


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "source": "Auto (Comp tag)",
  "manual": "",
  "mode": "Exact",
  "samples": [
    1
  ],
  "group_filter": "",
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "source": "Auto (Comp tag)",
  "manual": "",
  "mode": "Exact",
  "samples": [
    1
  ],
  "group_filter": "",
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "source": "Auto (Comp tag)",
  "manual": "",
  "mode": "Exact",
  "samples": [
    20
  ],
  "group_filter": "",
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Group Label -> Random Occupancy: 将占位变化限制在指定 group。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 缺少成分来源：检查 `source/manual`。
- 统计偏差大：提高 `samples` 或切换 `mode`。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Occ({...}{...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
