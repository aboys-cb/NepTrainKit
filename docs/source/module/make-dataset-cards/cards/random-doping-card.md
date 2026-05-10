<!-- card-schema: {"card_name": "Random Doping", "source_file": "src/NepTrainKit/ui/views/_card/random_doping_card.py", "serialized_keys": ["params"]} -->

# 随机掺杂（Random Doping）

`Group`: `Alloy` | `Class`: `RandomDopingCard`

## 功能说明

按规则表对指定元素位点做替位掺杂。可以精确控制"替换谁、换成什么、换多少"，支持按原子百分比、质量百分比或固定数量采样。

**和 `Random Occupancy` / `Composition Sweep` 的区别：**
- `Composition Sweep`：定义"目标配比空间"，不改原子占位
- `Random Occupancy`：把目标配比落到离散占位上，需要输入有 `Comp(...)` 标签
- `Random Doping`：直接给定规则（target + dopants + 比例），做一次随机替换。更直接、更手工，但不扫配比空间

## 操作示例

### 场景：纯元素模型在掺杂体系上完全失效

你在纯 Si 上训练了一个 NEP 模型，能量和力都收敛得很好。然后尝试跑 Si:Ge 合金——模型直接崩溃：碰到 Ge 原子的局域环境完全没见过，力预测误差跳了一个数量级。

**诊断思路：** 训练集里只有 Si-Si 键环境，模型不知道 Ge 原子的存在，更不知道 Si-Ge 键和 Ge-Ge 键应该长什么样。需要往训练集中加入真实的 Si 位点被 Ge 替换的结构，让模型学习掺杂原子周围的局域化学环境。

**输入：** 一个纯 Si 超胞结构，已经弛豫

**目标：** 每帧替换 3~8% 的 Si 为 Ge，每帧生成 20 个版本，覆盖不同掺杂落点

**参数设置：**
- `Rules`：target=`Si`, dopants=`Ge`, use=`atomic_percent`, percent=`[3, 8]`
- `Doping Type`：`Exact` （对比实验需要稳定的掺杂数量）
- `Structures`：`[20]`

**输出：** 20 个掺杂结构，每帧中 3~8% 的 Si 变成 Ge，带 `Dop(n=...)` 标签

**怎么验证训练集质量改善：**
- 重训后跑 Si:Ge 测试集，力的 MAE 应该回到和纯 Si 训练集接近的水平
- 抽查几个掺杂输出，Ge 原子的最近邻距离是否合理（Si-Ge 键长应略大于 Si-Si）
- 如果模型对高浓度掺杂仍然不准，扩大 `percent` 上限到 15~20%
- 如果只掺一种元素，dopants 可以直接写 `Ge`，等价于 `Ge:1.0`
- 如果需要多元素掺杂（如同时掺 Ge 和 C），在 dopants 里写 `Ge:0.7,C:0.3`

### 什么时候加这张卡、什么时候不加

**加：**
- 模型在掺杂/合金体系上预测质量明显差于纯元素体系
- 需要覆盖特定掺杂元素周围的局域化学环境
- 有明确的掺杂规则（target + dopants + 比例），但不需系统扫描完整配比空间

**不加：**
- 需要系统扫配比空间 → 先用 `Composition Sweep`，再接 `Random Occupancy`
- 只需要全局随机占位 → 用 `Random Occupancy`

## 参数说明

### Rules（rules）

`list[dict[str, Any]]`，默认空列表。每条 rule 定义目标元素、替换元素、比例/计数和可选 group。复杂掺杂优先拆成多条明确 rule，不要把不同物理缺陷混成一个概率池。

rule 内的典型字段：`target`（被替换元素）、`dopants`（替换元素及权重 dict）、`use`（`atomic_percent` / `mass_percent` / `count`）、`percent`（百分比范围 `[min, max]` 或 `[fixed]`）、`count`（固定替换个数）、`group`（可选，限制只在此 group 内操作）。

### Doping Type（doping_type）

`str`，默认 `Random`。`Random` 按权重概率随机采样，每帧替换数量有浮动。`Exact` 尽量匹配精确计数，替换数量更稳定，适合对比实验。

### Max Structures（max_structures）

`int`，默认 1。每个输入结构最多输出的掺杂构型数。低浓度精确掺杂可用 10-50，高维随机合金应后接 FPS 控制预算。

### Use Seed（use_seed）

`bool`，默认 false。打开后固定种子可复现。对比实验时开，探索阶段可以关着。

### Seed（seed）

`int`，默认 0。不同取值产生不同的替换分布。

生效条件：`use_seed=True`。

## 推荐预设

### 低浓度单元素掺杂（Si:Ge, 1~2%）
```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "params": {
    "rules": [
      {
        "target": "Si",
        "dopants": {"Ge": 1.0},
        "use": "atomic_percent",
        "percent": [1, 2]
      }
    ],
    "doping_type": "Exact",
    "max_structures": 20,
    "use_seed": true,
    "seed": 101
  }
}
```

### 中浓度双元素掺杂（Si:Ge/C, 3~8%）
```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "params": {
    "rules": [
      {
        "target": "Si",
        "dopants": {"Ge": 0.7, "C": 0.3},
        "use": "atomic_percent",
        "percent": [3, 8]
      }
    ],
    "doping_type": "Exact",
    "max_structures": 20,
    "use_seed": true,
    "seed": 101
  }
}
```

### 高浓度多元素探索（Si:Ge/C/Sn, 8~20%）
```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "params": {
    "rules": [
      {
        "target": "Si",
        "dopants": {"Ge": 0.5, "C": 0.3, "Sn": 0.2},
        "use": "atomic_percent",
        "percent": [8, 20]
      }
    ],
    "doping_type": "Random",
    "max_structures": 20,
    "use_seed": true,
    "seed": 101
  }
}
```

## 推荐组合

- `Group Label` → `Random Doping`：只掺杂特定子晶格/层，不污染其他区域
- `Composition Sweep` → `Random Occupancy` → `Random Doping`：先扫配比 → 落位 → 再补局部随机替换
- `Random Doping` → `Atomic Perturb`：掺杂后加坐标噪声，松驰局部应力

## 常见问题

**输出和输入一样，没有替换。** 检查 rules 是否为空、target 元素是否真的存在于输入结构中、group 过滤是否把候选位点全滤掉了。

**掺杂比例偏离预期。** 如果用 `Random` 模式，小样本下统计浮动大。换成 `Exact` 可以减少波动。如果用的是 `mass_percent`，确认目标元素和掺杂元素的质量差异是否导致了原子比例偏移。

**掺杂后键长异常。** 这是纯化学替换，不做结构弛豫。替换后键长取决于新元素的原子半径和原晶格参数的匹配度。如果键长明显不合理，建议后接弛豫计算。

**多条规则之间的交互。** 规则按顺序执行。如果两条规则操作同一个 target 元素，第二条会在第一条的结果上继续替换。注意不要设计互相冲突的规则。

## 输出标签

- `Dop(n={替换原子数})`

## 可复现性

勾选 `use_seed` + 固定 `seed` 可复现。注意输入结构顺序变化也会影响结果——建议把 seed 与 pipeline 配置一起版本化。
