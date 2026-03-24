<!-- card-schema: {"card_name": "Random Vacancy", "source_file": "src/NepTrainKit/ui/views/_card/random_vacancy_card.py", "serialized_keys": ["rules", "max_atoms_condition", "use_seed", "seed"]} -->

# 随机空位（Random Vacancy）

`Group`: `Defect`  
`Class`: `RandomVacancyCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_vacancy_card.py`

## 功能说明
根据规则删除指定元素原子（rule-based vacancy），控制空位元素类型、数量和区域。

它最适合的场景是：按元素或 group 规则删除特定位点，生成可控的空位样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：按元素或 group 规则删除特定位点，生成可控的空位样本

**输入：** 一个足够大的超胞，以及明确的删位规则

**目标：** 做带约束的删位，而不是只按整体浓度随机删原子

**参数设置：**
- `rules` 中写清元素、count 和可选 group 过滤
- `max_atoms_condition` 控制每帧生成多少个删位版本
- `use_seed` 在做规则对比时建议开启

**输出：** 多份按规则删位后的结构，删除位置可追溯

**怎么验证结果合理：**
- 检查删位元素和数量是否命中规则
- 若结果为空或无变化，先检查规则是否写错元素名或 group
- 缺陷过强时先降低 count

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 空位缺陷覆盖不足或分布不可控。
- 目标任务 (Target objective): 精确控制空位类型与局域分布。
- 建议添加条件 (Add-it trigger): 需要按元素和 group 定向删原子。
- 不建议添加条件 (Avoid trigger): 仅需无规则随机空位。
> 物理提示 (Physics caution): 重点检查缺陷附近的局部配位和是否形成孤立原子或明显断裂。

## 输入前提
- 先单规则验证，再叠加多规则。
- 使用 group 时确认输入包含 `atoms.arrays['group']`；如果数据来自 `.xyz`，请使用 EXTXYZ 风格的 `group` 列，普通三列 XYZ 不会保留该数组。

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
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
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
`rules` 在 card JSON 中保存为 JSON 字符串，但界面里每条规则的 `group` 建议按简单字符串填写，不建议用户自己手写嵌套 JSON。
- `element` (string): 删除目标元素。
- `count` (list[2]): 删除数量区间。
- `group` (string or list[string], optional): 界面里推荐写成 `surface_top,surface_bottom`；程序会拆成名称列表，并匹配 `atoms.arrays['group']`。如果输入来自 `.xyz`，请使用 EXTXYZ 风格的 `group` 列。

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
- 缺陷强度上升前，通常先用 `Super Cell` 扩大母胞，避免小胞里缺陷相互作用过强。
- 缺陷生成后建议抽查最短键长、局部配位和是否出现明显断裂。

## 常见问题与排查
- 输出为空或结构数明显偏少时，先检查规则是否命中、浓度/数量是否过严，或输入超胞是否太小。
- 若输出结构不合理，优先检查最短键长、局部配位和是否出现整块骨架塌缩，再降低缺陷强度。
- 参数越界时通常受 UI 范围限制；但“过激而仍在范围内”的配置不会被自动裁剪，程序会继续按当前设置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Vac(n={...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
