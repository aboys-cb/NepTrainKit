<!-- card-schema: {"card_name": "Composition Sweep", "source_file": "src/NepTrainKit/ui/views/_card/composition_sweep_card.py", "serialized_keys": ["elements", "order", "method", "step", "n_points", "min_fraction", "include_endpoints", "use_seed", "seed", "max_outputs", "budget_mode"]} -->

# 成分扫描（Composition Sweep）

`Group`: `Alloy`  
`Class`: `CompositionSweepCard`  
`Source`: `src/NepTrainKit/ui/views/_card/composition_sweep_card.py`

## 功能说明
在元素池和元数约束下生成成分设计空间（composition design space），用于合金候选前置展开。

:::{important}
该卡片只在 `info` 中添加组分信息，并不实际生成合金结构，需要配合 `Random Occupancy` 使用。
:::

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 元素组合覆盖单一，跨成分迁移误差高。
- 目标任务 (Target objective): 先覆盖成分空间，再下游做占位/掺杂。
- 建议添加条件 (Add-it trigger): 需要二元到多元合金系统化采样。
- 不建议添加条件 (Avoid trigger): 任务只关注单一固定化学计量。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


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

### `max_outputs` (Max Outputs)
- UI Label: `Max Outputs`
- 字段映射 (Field mapping): 序列化键 `max_outputs` <-> 界面标签 `Max Outputs`。
- 控件标签 (Caption): `Max Outputs`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[500]`
- 含义 (Meaning): 输出上限 (maximum outputs)。
- 对输出规模/物理性的影响: 限制样本规模，防止组合爆炸。
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


## 常见问题与排查
- 组合数远超预算：收窄 `order` 并降低 `max_outputs`。
- 成分分布偏置明显：调整 `method` 与 `budget_mode`。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Comp({...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
