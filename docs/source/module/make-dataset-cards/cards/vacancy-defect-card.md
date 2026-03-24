<!-- card-schema: {"card_name": "Vacancy Defect Generation", "source_file": "src/NepTrainKit/ui/views/_card/vacancy_defect_card.py", "serialized_keys": ["engine_type", "num_condition", "num_radio_button", "concentration_radio_button", "concentration_condition", "max_atoms_condition", "use_seed", "seed"]} -->

# 空位缺陷生成（Vacancy Defect Generation）

`Group`: `Defect`  
`Class`: `VacancyDefectCard`  
`Source`: `src/NepTrainKit/ui/views/_card/vacancy_defect_card.py`

## 功能说明
按数量或浓度随机生成空位缺陷（vacancy sampling），快速覆盖缺陷强度分布。

它最适合的场景是：快速生成低到中等强度的随机空位族，用于缺陷训练。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：快速生成低到中等强度的随机空位族，用于缺陷训练

**输入：** 一个足够大的超胞，最好已经先扩到能容纳目标缺陷浓度

**目标：** 在“按空位数”与“按空位浓度”两种模式中选一种，批量扩出随机空位结构

**参数设置：**
- `num_radio_button` 与 `concentration_radio_button` 二选一
- `concentration_condition` 先从 0.5%-5% 量级开始
- `max_atoms_condition` 决定每帧生成多少个随机版本

**输出：** 多份删位数量不同或浓度不同的空位结构

**怎么验证结果合理：**
- 统计删位数是否落在预期区间
- 若空位太多导致骨架崩坏，先降低浓度或回到更大的母胞
- 若结果没有按浓度变化，先检查当前到底启用了哪种模式

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 缺陷密度维度不足，模型对空位数敏感。
- 目标任务 (Target objective): 快速构建低-中-高缺陷强度样本。
- 建议添加条件 (Add-it trigger): 需要高通量空位数据且不需复杂规则。
- 不建议添加条件 (Avoid trigger): 需要按元素/group 精细控制空位位置。
> 物理提示 (Physics caution): 重点检查缺陷附近的局部配位和是否形成孤立原子或明显断裂。

## 输入前提
- 先选 count 或 concentration 单一主模式。
- 控制 `max_atoms_condition` 先小后大。

## 参数说明（完整）
### `engine_type` (Random Engine)
- UI Label: `Random Engine`
- 字段映射 (Field mapping): 序列化键 `engine_type` <-> 界面标签 `Random Engine`。
- 控件标签 (Caption): `Random Engine`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=Sobol`, `1=Uniform`
- 默认值 (Default): `1`
- 含义 (Meaning): 随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。
- 对输出规模/物理性的影响: Uniform 在大批量生成时更快；Sobol 在样本较少时对空位数与位置的覆盖更均衡。样本很多时两者统计差异通常不大。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：小样本覆盖优先 Sobol
  - 平衡：先 Uniform 提速，再用 Sobol 抽样复核分布
  - 探索：超大样本阶段优先速度，固定引擎避免混杂偏差

### `num_condition` (Vacancy Num)
- UI Label: `Vacancy Num`
- 字段映射 (Field mapping): 序列化键 `num_condition` <-> 界面标签 `Vacancy Num`。
- 控件标签 (Caption): `Vacancy Num`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 采样数量控制 (sample count control)。
- 对输出规模/物理性的影响: 主要影响输出规模与耗时，不是幅度主控参数。
- 参数联动 / 生效条件: 只有 count 模式时它才直接决定删位强度；浓度模式下它不会作为主控参数。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：1-1
  - 平衡：1-2
  - 探索：2-5

### `num_radio_button` (Vacancy Num Mode)
- UI Label: `Vacancy Num Mode`
- 字段映射 (Field mapping): 序列化键 `num_radio_button` <-> 界面标签 `Vacancy Num Mode`。
- 控件标签 (Caption): `Vacancy Num Mode`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 计数模式开关 (count mode switch)。
- 对输出规模/物理性的影响: 按绝对数量控制缺陷强度。
- 参数联动 / 生效条件: 与 `concentration_radio_button` 二选一；当前代码以是否勾选 concentration 模式作为优先分支。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Vacancy Num Mode` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `concentration_radio_button` (Vacancy Concentration Mode)
- UI Label: `Vacancy Concentration Mode`
- 字段映射 (Field mapping): 序列化键 `concentration_radio_button` <-> 界面标签 `Vacancy Concentration Mode`。
- 控件标签 (Caption): `Vacancy Concentration Mode`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 浓度模式开关 (concentration mode switch)。
- 对输出规模/物理性的影响: 按比例控制缺陷强度。
- 参数联动 / 生效条件: 开启后按 `concentration_condition` 计算最大删位数；若关闭则退回 `num_condition`。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Vacancy Concentration Mode` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `concentration_condition` (Vacancy Concentration)
- UI Label: `Vacancy Concentration`
- 字段映射 (Field mapping): 序列化键 `concentration_condition` <-> 界面标签 `Vacancy Concentration`。
- 控件标签 (Caption): `Vacancy Concentration`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): 空位浓度比例 (vacancy concentration ratio)。取值 `0-1`，可按百分比理解为 `0%-100%`。
- 对输出规模/物理性的影响: 在浓度模式下，最大空位数按 `int(concentration * n_atoms)` 计算；例如 `0.02` 表示约 `2%` 原子被删。
- 参数联动 / 生效条件: 只有浓度模式时才按 `int(concentration * n_atoms)` 估算删位上限。
- 物理直觉 / 典型值: 把它直接理解成比例更直观；先从几个百分点或较小分数开始，通常更容易保持结构稳定。
- 推荐范围 (Recommended range):
  - 保守：0.005-0.02（约 0.5%-2%）
  - 平衡：0.02-0.08（约 2%-8%）
  - 探索：0.08-0.20（约 8%-20%，需稳定性评估）

### `max_atoms_condition` (Max Num)
- UI Label: `Max Num`
- 字段映射 (Field mapping): 序列化键 `max_atoms_condition` <-> 界面标签 `Max Num`。
- 控件标签 (Caption): `Max Num`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 每帧最大生成数 (max generated structures per frame)。
- 对输出规模/物理性的影响: 主要控制数据量和运行时间。
- 参数联动 / 生效条件: 它控制“每帧额外生成多少个随机版本”，不是控制超胞原子上限。
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

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 1,
  "num_condition": [
    2
  ],
  "num_radio_button": false,
  "concentration_radio_button": true,
  "concentration_condition": [
    0.02
  ],
  "max_atoms_condition": [
    100
  ],
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 1,
  "num_condition": [
    4
  ],
  "num_radio_button": false,
  "concentration_radio_button": true,
  "concentration_condition": [
    0.05
  ],
  "max_atoms_condition": [
    100
  ],
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 0,
  "num_condition": [
    8
  ],
  "num_radio_button": false,
  "concentration_radio_button": true,
  "concentration_condition": [
    0.1
  ],
  "max_atoms_condition": [
    100
  ],
  "use_seed": true,
  "seed": [
    0
  ]
}
```

## 推荐组合
- Vacancy Defect Generation -> Insert Defect: 联合采样空位与插隙缺陷族。
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
