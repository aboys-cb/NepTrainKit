<!-- card-schema: {"card_name": "Composition Sweep", "source_file": "src/NepTrainKit/ui/views/_card/composition_sweep_card.py", "serialized_keys": ["params"]} -->

# 成分扫描（Composition Sweep）

`Group`: `Alloy` | `Class`: `CompositionSweepCard`

## 功能说明

在元素池和元数约束下生成成分设计空间。对选定元素集合按二元到五元组合展开，用网格或 Sobol 低差异序列在 simplex 上采样目标配比点，输出带 `Comp(...)` 标签的结构副本。本卡只生成目标配比计划，不真正落位——需要下游接 `Random Occupancy` 才能得到离散原子占位。

## 操作示例

### 场景：模型在偏离训练集化学计量比时预测崩了

你在 Co-Cr-Ni 三元系上训练了一个 NEP 模型，训练数据全部来自等摩尔 CoCrNi 单相。模型对等摩尔构型的能量和力预测很好，但一跑 Co30Cr20Ni50 这种偏离等摩尔的组分，能量误差翻了三倍。诊断结果是：训练集所有数据的成分落在 simplex 的一个点上，模型根本没有见过成分变化导致的局域化学环境差异。

**诊断思路：** 合金模型的基本要求是能在整个成分空间内插，而不是只记住一个成分点。如果训练集成分覆盖太窄，模型只能外推——成分漂得越远，误差越大。解决办法是先在目标成分空间内均匀撒一批配比点，再去下游逐个落位，让模型从头学起不同成分下的化学环境。

**输入：** 一个弛豫好的 CoCrNi 超胞结构（当前训练集里唯一的结构，等摩尔配比）

**目标：** 从 Co-Cr-Ni 三个元素出发，生成二元 + 三元所有配比的网格采样，约 500 个目标配比点，后续接 `Random Occupancy` 落地

**参数设置：**
- `elements` = `Co,Cr,Ni`
- `order` = `2,3` （二元 + 三元组合）
- `method` = `Grid`，`step` = `0.1`
- `max_outputs` = `500`

**输出：** 约 500 个带 `Comp(Co=...,Cr=...,Ni=...)` 标签的结构副本。此时结构内原子种类未变，标签只是目标配比计划

**怎么验证训练集质量改善：**
- 重训后跑几个偏移等摩尔的测试组分，力的 MAE 应该不再随成分漂移而显著恶化
- 检查 `Config_type` 中 Comp 标签的成分分布是否覆盖了 simplex 各区域——如果某些边角空白，减小 step 或换 Sobol 补点
- 如果全部集中等摩尔附近但没有边角覆盖，打开 `include_endpoints`，降低 `min_fraction` 到 0
- 如果输出量不够覆盖所有 order 组合，提高 `max_outputs`，或把 `budget_mode` 换成 `Capacity-weighted` 让高元数组合分到更多名额

### 什么时候加这张卡、什么时候不加

**加：**
- 模型在偏离训练集化学计量比时误差显著增大
- 需要系统覆盖二元到多元合金成分空间，而非少量随机掺杂
- 作为合金 pipeline 的第一步，先定义配比，再接占位卡片

**不加：**
- 只需单一固定化学计量比的结构
- 已经有明确掺杂规则且不需要扫配比空间 → 直接用 `Random Doping`
- 需要增补的是同一成分下的不同原子排布 → 直接用 `Random Occupancy`

## 参数说明


### Elements（elements）

类型：`str`。默认：`'Co,Cr,Ni'`。指定参与生成、替换或扰动的元素集合。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Order（order）

类型：`str`。默认：`'2,3,4,5'`。控制 `order` 对应的生成或过滤行为。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Method（method）

类型：`str`。默认：`'Grid'`。选择采样或构型枚举方法。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |

### Step（step）

类型：`float`。默认：`0.1`。设置连续路径或扫描的步长。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### N Points（n_points）

类型：`int`。默认：`50`。设置连续组成扫描的采样点数量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Min Fraction（min_fraction）

类型：`float`。默认：`0.0`。控制 `min_fraction` 对应的生成或过滤行为。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Include Endpoints（include_endpoints）

类型：`bool`。默认：`True`。控制 `include_endpoints` 对应的生成或过滤行为。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Use Seed（use_seed）

类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现的训练集生成或测试时打开；做最终大规模探索且希望保留随机多样性时可关闭。

### Seed（seed）

类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一 seed 应产生同一批候选；只有在 `use_seed` 打开时才改变结果。

生效条件：`use_seed=True`。

### Max Outputs（max_outputs）

类型：`int`。默认：`500`。限制这张卡最多输出多少个结构。

物理直觉：这是防止链式卡片数量爆炸的预算阀；上游结构很多时应先按计算预算设上限。

### Budget Mode（budget_mode）

类型：`str`。默认：`'Equal+Reflow'`。控制 `budget_mode` 对应的生成或过滤行为。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |

## 推荐预设

### 快速试探（二元+三元 Grid，~100 点）
```json
{
  "class": "CompositionSweepCard",
  "check_state": true,
  "elements": "Co,Cr,Ni",
  "order": "2,3",
  "method": "Grid",
  "step": [0.1],
  "n_points": [50],
  "min_fraction": [0.0],
  "include_endpoints": true,
  "use_seed": false,
  "seed": [0],
  "max_outputs": [100],
  "budget_mode": "Equal+Reflow"
}
```

### 常规覆盖（全组合 Grid，~500 点）
```json
{
  "class": "CompositionSweepCard",
  "check_state": true,
  "elements": "Co,Cr,Ni,Al",
  "order": "2,3,4,5",
  "method": "Grid",
  "step": [0.1],
  "n_points": [50],
  "min_fraction": [0.0],
  "include_endpoints": true,
  "use_seed": false,
  "seed": [0],
  "max_outputs": [500],
  "budget_mode": "Equal+Reflow"
}
```

### 高维探索（Sobol 全组合，~1500 点）
```json
{
  "class": "CompositionSweepCard",
  "check_state": true,
  "elements": "Co,Cr,Ni,Al,Fe",
  "order": "2,3,4,5",
  "method": "Sobol",
  "step": [0.1],
  "n_points": [150],
  "min_fraction": [0.0],
  "include_endpoints": true,
  "use_seed": true,
  "seed": [42],
  "max_outputs": [1500],
  "budget_mode": "Capacity-weighted"
}
```

## 推荐组合

- `Composition Sweep` → `Random Occupancy`：先定义目标配比，再落到离散原子占位。合金 pipeline 的标准起手式。
- `Composition Sweep` → `Random Occupancy` → `Random Doping`：配比覆盖 + 占位多样性 + 额外局部替位。
- `Composition Sweep` → `Random Occupancy` → `Atomic Perturb`：成分 + 占位 + 坐标噪声。

## 常见问题

**输出和输入完全一样。** `elements` 少于 2 个元素，或 `order` 不合法，或 `max_outputs` <= 0。

**标签成分和原子实际组成不一致。** 正常——Composition Sweep 只打目标配比标签，不改原子种类。需要接 `Random Occupancy` 才会真正替换原子。

**组合爆炸。** 5 元 + Grid + step=0.05 可产生上万个点。先用 `max_outputs` 卡住上限，小样本验证后再放量。

**某些 order 没有输出。** 检查 `budget_mode` 分配和 element 数量。只有 3 个元素但 order 含 4 和 5 时，程序自动跳过不可执行的组合。

## 输出标签

`Comp(Co=0.3333,Cr=0.3333,Ni=0.3333)` 格式。每个输出结构通过 `Config_type` 携带目标配比。

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入顺序可复现。Grid 模式的网格点本身是确定性的，`use_seed` 主要影响组合选取顺序和 Sobol 生成。
