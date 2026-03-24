<!-- card-schema: {"card_name": "Composition Sweep", "source_file": "src/NepTrainKit/ui/views/_card/composition_sweep_card.py", "serialized_keys": ["elements", "order", "method", "step", "n_points", "min_fraction", "include_endpoints", "use_seed", "seed", "max_outputs", "budget_mode"]} -->

# 成分扫描（Composition Sweep）

`Group`: `Alloy`  
`Class`: `CompositionSweepCard`  
`Source`: `src/NepTrainKit/ui/views/_card/composition_sweep_card.py`

## 功能说明
在元素池和元数约束下生成成分设计空间（composition design space），用于合金候选前置展开。

它最适合的场景是：先定义一批目标成分点，再交给下游卡片把成分真正落到原子位点。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：先定义一批目标成分点，再交给下游卡片把成分真正落到原子位点

**输入：** 一个母相结构，以及一组候选元素，例如 `Co,Cr,Ni,Al,Fe`

**目标：** 系统扫过二元到五元目标配比，而不是只做几次随机替换

**参数设置：**
- `order` 决定扫二元、三元还是更高元组合
- `method="Grid"` 适合低维规则网格，`"Sobol"` 更适合高维覆盖
- `max_outputs` 先按你的计算预算限制每帧输出量

**输出：** 多份带 `Comp(...)` 标签的结构副本；此时是“目标配比计划”，还不是实际离散占位结果

**怎么验证结果合理：**
- 检查 `Config_type` 中是否出现了目标成分标签
- 确认输出数量与 `order`、`method` 和 `max_outputs` 一致
- 需要真实随机合金时，记得继续接 `Random Occupancy`

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 元素组合覆盖单一，跨成分迁移误差高。
- 目标任务 (Target objective): 先覆盖成分空间，再下游做占位/掺杂。
- 建议添加条件 (Add-it trigger): 需要二元到多元合金系统化采样。
- 不建议添加条件 (Avoid trigger): 任务只关注单一固定化学计量。
> 物理提示 (Physics caution): 重点检查目标配比、实际元素统计和标签是否一致，避免“标签写对了、占位落错了”。

## 输入前提
- 明确元素池、元数上限和预算上限。
- 先小规模验证 `method` 与 `budget_mode` 的分布行为。

## 参数说明（完整）
### `elements` (Elements)
- UI Label: `Elements`
- 字段映射 (Field mapping): 序列化键 `elements` <-> 界面标签 `Elements`。
- 控件标签 (Caption): `Elements`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"Co,Cr,Ni"`
- 含义 (Meaning): 元素集合输入 (element set)。
- 对输出规模/物理性的影响: 决定参与操作的元素子集。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): `Elements` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `order` (Order)
- UI Label: `Order`
- 字段映射 (Field mapping): 序列化键 `order` <-> 界面标签 `Order`。
- 控件标签 (Caption): `Order`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"2,3,4,5"`
- 含义 (Meaning): 组合阶范围 (order)。
- 对输出规模/物理性的影响: 控制元数组合阶数。
- 参数联动 / 生效条件: 它定义“扫几元成分空间”，并直接影响预算如何在不同元数组合之间分配。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): `Order` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `method` (Method)
- UI Label: `Method`
- 字段映射 (Field mapping): 序列化键 `method` <-> 界面标签 `Method`。
- 控件标签 (Caption): `Method`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string): `Grid`, `Sobol`
- 默认值 (Default): `"Grid"`
- 含义 (Meaning): 成分点生成方法 (composition sampling method)，可选 `Grid` 或 `Sobol`。
- 对输出规模/物理性的影响: Grid 便于解释且步长可控；Sobol 属于低差异序列，在少样本/高阶(order>=4)时覆盖通常更稳。样本足够多时两者差异会减小。
- 参数联动 / 生效条件: `Grid` 更适合低维规则扫描，`Sobol` 更适合高维覆盖；两者对应的主控参数不同。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：少样本或高阶优先 Sobol
  - 平衡：Grid 先试跑，再用 Sobol 补覆盖盲区
  - 探索：大样本阶段按可解释性与算力预算选择

### `step` (Step)
- UI Label: `Step`
- 字段映射 (Field mapping): 序列化键 `step` <-> 界面标签 `Step`。
- 控件标签 (Caption): `Step`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.1]`
- 含义 (Meaning): 步长区间 (step range)。
- 对输出规模/物理性的影响: 主控扫描位移幅度与分辨率。
- 参数联动 / 生效条件: 只有 `method="Grid"` 时它才决定网格分辨率。
- 物理直觉 / 典型值: 它通常是控制变化幅度的主旋钮；先从能看清趋势的小幅度起步，再决定是否扩到探索档。
- 推荐范围 (Recommended range):
  - 保守：0.07-0.1
  - 平衡：0.1-0.15
  - 探索：0.15-0.25

### `n_points` (N Points)
- UI Label: `N Points`
- 字段映射 (Field mapping): 序列化键 `n_points` <-> 界面标签 `N Points`。
- 控件标签 (Caption): `N Points`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[50]`
- 含义 (Meaning): 采样点数 (number of points)。
- 对输出规模/物理性的影响: 点数越大覆盖越密，但计算开销更高。
- 参数联动 / 生效条件: 只有 `method="Sobol"` 时它才决定目标采样点数。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：25-50
  - 平衡：50-100
  - 探索：100-250

### `min_fraction` (Min Fraction)
- UI Label: `Min Fraction`
- 字段映射 (Field mapping): 序列化键 `min_fraction` <-> 界面标签 `Min Fraction`。
- 控件标签 (Caption): `Min Fraction`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.0]`
- 含义 (Meaning): 最小组分比例下限 (minimum fraction ratio)。取值 `0-1`，可按百分比理解。
- 对输出规模/物理性的影响: 约束每个元素的最小占比；例如 `0.05` 表示每个组分至少 `5%`，会过滤极端稀释端点。
- 物理直觉 / 典型值: 把它直接理解成比例更直观；先从几个百分点或较小分数开始，通常更容易保持结构稳定。
- 推荐范围 (Recommended range):
  - 保守：0.05-0.10（更稳）
  - 平衡：0.01-0.05（平衡覆盖）
  - 探索：0.00-0.01（覆盖边角但组合更散）

### `include_endpoints` (Include Endpoints)
- UI Label: `Include Endpoints`
- 字段映射 (Field mapping): 序列化键 `include_endpoints` <-> 界面标签 `Include Endpoints`。
- 控件标签 (Caption): `Include Endpoints`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 是否包含端点 (include endpoints)。
- 对输出规模/物理性的影响: 控制是否保留纯端元和边界成分点。
- 参数联动 / 生效条件: 只在 Grid 模式下有意义，用于决定是否把端点成分也纳入扫描。
- 怎么判断该开还是该关: 只有当你明确知道这个开关会改变当前工作流目标时才开启；不确定时先保持默认并用小样本验证。
- 配置建议 (Practical note):
  - 开启：需要启用 `Include Endpoints` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

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

### `max_outputs` (Max Outputs)
- UI Label: `Max Outputs`
- 字段映射 (Field mapping): 序列化键 `max_outputs` <-> 界面标签 `Max Outputs`。
- 控件标签 (Caption): `Max Outputs`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[500]`
- 含义 (Meaning): 输出上限 (maximum outputs)。
- 对输出规模/物理性的影响: 限制样本规模，防止组合爆炸。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：250-500
  - 平衡：500-1000
  - 探索：1000-2500

### `budget_mode` (Budget Mode)
- UI Label: `Budget Mode`
- 字段映射 (Field mapping): 序列化键 `budget_mode` <-> 界面标签 `Budget Mode`。
- 控件标签 (Caption): `Budget Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"Equal+Reflow"`
- 含义 (Meaning): 预算分配策略 (budget mode)。
- 对输出规模/物理性的影响: 决定不同子空间获得样本名额的比例。
- 参数联动 / 生效条件: 当 `order` 同时包含多种元数时，它决定 `max_outputs` 如何分配到每种元数组合。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：均匀分配
  - 平衡：轻度偏置
  - 探索：强偏置探索

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "CompositionSweepCard",
  "check_state": true,
  "elements": "Co,Cr,Ni",
  "order": "2,3,4,5",
  "method": "Grid",
  "step": [
    0.1
  ],
  "n_points": [
    50
  ],
  "min_fraction": [
    0.0
  ],
  "include_endpoints": true,
  "use_seed": false,
  "seed": [
    0
  ],
  "max_outputs": [
    10
  ],
  "budget_mode": "Equal+Reflow"
}
```

### 平衡（Balanced）
```json
{
  "class": "CompositionSweepCard",
  "check_state": true,
  "elements": "Co,Cr,Ni",
  "order": "2,3,4,5",
  "method": "Grid",
  "step": [
    0.1
  ],
  "n_points": [
    50
  ],
  "min_fraction": [
    0.0
  ],
  "include_endpoints": true,
  "use_seed": false,
  "seed": [
    0
  ],
  "max_outputs": [
    500
  ],
  "budget_mode": "Equal+Reflow"
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "CompositionSweepCard",
  "check_state": true,
  "elements": "Co,Cr,Ni",
  "order": "2,3,4,5",
  "method": "Grid",
  "step": [
    0.1
  ],
  "n_points": [
    50
  ],
  "min_fraction": [
    0.0
  ],
  "include_endpoints": true,
  "use_seed": true,
  "seed": [
    0
  ],
  "max_outputs": [
    1500
  ],
  "budget_mode": "Capacity-weighted"
}
```

## 推荐组合
- Composition Sweep -> Random Occupancy: 将目标成分标签转换为显式原子占位。
- Composition Sweep -> Random Doping: 先做成分展开，再做位点定向替位。
- 先明确“目标配比”还是“具体落位”，再决定接 `Composition Sweep`、`Random Occupancy` 还是 `Random Doping`。

## 常见问题与排查
- 结果没有变化时，先检查目标元素、规则字符串或 `Comp tag` 来源是否真的命中了输入结构。
- 如果输出成分偏离预期，优先检查是“目标配比定义”问题，还是“离散落位/随机替换”步骤把分布拉偏。
- 这组卡片不会自动替你决定工作流分工；需要系统扫配比时先用 `Composition Sweep`，需要真实落位时再接 `Random Occupancy` 或 `Random Doping`。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Comp({...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
