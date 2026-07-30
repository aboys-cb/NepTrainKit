<!-- card-schema: {"card_name": "Random Doping", "source_file": "src/NepTrainKit/ui/views/_card/random_doping_card.py", "serialized_keys": ["params"]} -->

# 随机掺杂（Random Doping）

**分类：** 合金与组分

## 功能说明

按规则表对指定元素位点做替位掺杂。可以精确控制"替换谁、换成什么、换多少"，支持按原子百分比、质量百分比或固定数量采样。规则填写了 group 时，输入必须真的带有对应标签；卡片不会把缺失的 group 当成“全部位点”。

**和 `Random Occupancy` / `Composition Space Sampling` 的区别：**
- `Composition Space Sampling`：定义"目标配比空间"，不改原子占位
- `Random Occupancy`：把目标配比落到离散占位上，需要输入有 `Comp(...)` 标签
- `Random Doping`：直接给定规则（target + dopants + 比例），做一次随机替换。更直接、更手工，但不扫配比空间

## 原理与公式

每条规则先按目标元素和可选 group 得到 $N_{\mathrm{eligible}}$ 个候选位点。`原子百分比`
模式使用

$$
N_{\mathrm{dope}}=\left\lfloor
N_{\mathrm{eligible}}\,p_{\mathrm{atom}}\right\rfloor.
$$

若输入的是掺杂元素质量权重 $w_e$，程序先换算成原子抽样权重

$$
p_e=\frac{w_e/M_e}{\sum_j w_j/M_j},
$$

其中 $M_e$ 是元素原子质量。`固定数量`直接使用指定整数；`随机数量`在给定上下限内抽取。
确定总数后，`精确`替换按最大余数法分配各掺杂元素计数，`随机`替换按类别概率逐位点抽样。
质量百分比模式按当前实现用目标原子的总质量和掺杂物平均原子质量估算个数，因此输出后仍应
重新统计实际最终质量分数。

## 操作示例

### 场景：纯元素模型在掺杂体系上完全失效

你在纯 Si 上训练了一个 NEP 模型，能量和力都收敛得很好。然后尝试跑 Si:Ge 合金——模型直接崩溃：碰到 Ge 原子的局域环境完全没见过，力预测误差跳了一个数量级。

**诊断思路：** 训练集里只有 Si-Si 键环境，模型不知道 Ge 原子的存在，更不知道 Si-Ge 键和 Ge-Ge 键应该长什么样。需要往训练集中加入真实的 Si 位点被 Ge 替换的结构，让模型学习掺杂原子周围的局域化学环境。

**输入：** 一个纯 Si 超胞结构，已经弛豫

**目标：** 每帧替换 3~8% 的 Si 为 Ge，每帧生成 20 个版本，覆盖不同掺杂落点

**参数设置：**
- `规则`（`rules`）：target=`Si`, dopants=`Ge`, use=`atomic_percent`, percent=`[3, 8]`
- `掺杂方式`（`doping_type`）：`Exact` （对比实验需要稳定的掺杂数量）
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
- 需要系统扫配比空间 → 先用 `Composition Space Sampling`，再接 `Random Occupancy`
- 只需要全局随机占位 → 用 `Random Occupancy`

## 参数说明

### 规则（rules）

`list[dict[str, Any]]`，默认空列表。每条 rule 定义目标元素、替换元素、比例/计数和可选 group。复杂掺杂优先拆成多条明确 rule，不要把不同物理缺陷混成一个概率池。

rule 内的典型字段：`target`（被替换元素）、`dopants`（替换元素及权重 dict）、`ratio_type`（dopants 权重采用 `atom` 原子比或 `mass` 质量比）、`use`（`atomic_percent` / `mass_percent` / `count`）、`percent`（百分比范围 `[min, max]` 或 `[fixed]`）、`count`（固定替换个数或随机范围）、`count_mode`（`fixed` / `random`）、`group`（可选，限制只在此 group 内操作）。

`percent` 必须落在 0~100；百分比换算成离散原子数时向下取整，因此小超胞的低百分比可能得到 0 个替换。明确设置 0% 会保留原结构，不会被强制改成 1 个掺杂原子。固定数量、随机数量或质量百分比范围会先按上界计算最大请求数；只要上界可能超过当前候选位点数，整条规则就会确定性报错，不会因 seed 不同而时而成功、时而失败，也不会静默截断。多种 dopant 的质量百分比换算会按 `ratio_type` 对应的原子分数计算平均质量；切换规则里的“原子比/质量比”会改变换算和元素分配。填写 group 后，如果输入没有 `atoms.arrays["group"]`、标签未命中或目标元素未命中，也会明确报错。

### 掺杂方式（doping_type）

`str`，默认 `Random`。这个选项控制多个 dopant 如何分配到已经选定的替换位点，不改变 `percent` 或 `count` 推导出的替换总数。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| `Random` | 每个替换位点按 dopant 权重独立抽样 | 需要统计涨落和更多随机组成时 |
| `Exact` | 用最大余数法把权重换成整数计数，再随机打乱落点 | 对比实验需要每帧 dopant 计数尽量贴近目标比例时 |

### 最大结构数（max_structures）

`int`，默认 1。每个输入结构最多输出的掺杂构型数。低浓度精确掺杂可用 10-50，高维随机合金应后接 FPS 控制预算。

### 使用随机种子（use_seed）

`bool`，默认 false。打开后固定种子可复现。对比实验时开，探索阶段可以关着。

### 随机种子（seed）

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
- `Composition Space Sampling` → `Random Occupancy` → `Random Doping`：先扫配比 → 落位 → 再补局部随机替换
- `Random Doping` → `Atomic Perturb`：掺杂后加坐标噪声，松驰局部应力

## 常见问题

**输出和输入一样，没有替换。** 空 rules 会保留输入；显式 0% 或小超胞中百分比向下取整为 0 时，也会得到未替换端点。target 不存在、group 缺失或标签未命中则会直接报错，不再生成假成功输出。

**掺杂比例偏离预期。** 如果用 `Random` 模式，小样本下统计浮动大。换成 `Exact` 可以减少波动。如果用的是 `mass_percent`，确认目标元素和掺杂元素的质量差异是否导致了原子比例偏移。

**掺杂后键长异常。** 这是纯化学替换，不做结构弛豫。替换后键长取决于新元素的原子半径和原晶格参数的匹配度。如果键长明显不合理，建议后接弛豫计算。

**提示请求数量超过候选位点。** 卡片不会替你把 10 个改成“最多能换的 4 个”。范围参数按上界校验；请降低 `count` 或 `percent` 上限、扩大超胞，或检查前一条规则是否已经消耗了相同 target/group 的候选位点。

**多条规则之间的交互。** 规则按顺序执行。如果多条规则操作重叠的 target/group 候选池，卡片会按各范围上界联合预检；只要前面的规则存在耗尽后续容量的可能，就会在随机采样前统一报错，而不是让同一工作流随 seed 时成时败。需要保留多条规则时，应改用互不重叠的 group；同一候选池上的多种 dopant 通常应合并到一条规则的 `dopants` 中。

## 输出标签

- `Dop(n={替换原子数})`

## 可复现性

勾选 `使用随机种子`（`use_seed`） + 固定 `随机种子`（`seed`）可复现。同一候选位点顺序会复用同一随机路径，因此批量结构若原子排序完全一致，掺杂落点编号也可能一致；需要结构间更丰富的落点时，应使用不同 seed 或在上游生成不同占位顺序。建议把 seed 与 pipeline 配置一起版本化。
