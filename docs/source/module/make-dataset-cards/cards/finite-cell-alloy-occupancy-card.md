<!-- card-schema: {"card_name": "Finite-Cell Alloy Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/finite_cell_alloy_occupancy_card.py", "serialized_keys": ["params"]} -->

# 有限晶胞合金占位（Finite-Cell Alloy Occupancy）

**分类：** 合金与组分

## 功能说明

按有限位点上真正可实现的整数计数生成合金占位。输入若含 `atoms.arrays["sublattice"]`，每个子晶格独立约束；普通结构则全部位点组成名为 `all` 的单一 site set。

组成计划先确定每种元素的整数个数，再生成不重复的随机排布。卡片不会先采任意连续比例再静默取整，也不会把不可整除的目标写成 Exact。metadata 同时保存请求、实际计数、实际分数、composition ID 和 arrangement ID。

卡片主界面是可视化规则编辑器。先选择“全部位点”或“按子晶格”，再为每个位点集合添加元素，并选择一种组成模式。模式切换后只显示当前需要的比例或计数字段：

- `fixed_fraction`：给出固定目标比例；可整除时记为 `exact`，否则采用确定性的最近整数计划并记为 `nearest_integer`。
- `fraction_range`：枚举实际整数分数落在各元素范围内的计划。
- `count_range`：枚举各元素整数计数落在范围内且总和等于位点数的计划。

## 原理与公式

有限晶胞不能实现任意连续比例。若一个子晶格有 $N$ 个位点，程序枚举满足

$$
n_e\in\mathbb Z_{\ge0},\qquad \sum_e n_e=N
$$

且落入用户比例/计数上下限的可行整数向量。比例范围会转换为
$\lceil f_{e,\min}N\rceil\le n_e\le\lfloor f_{e,\max}N\rfloor$；固定比例用最大余数法
得到和为 $N$ 的计数。给定计数向量时，不同占位排列数为

$$
\Omega=\frac{N!}{\prod_e n_e!}.
$$

多个独立子晶格的组合数是各自 $\Omega$ 的乘积。卡片在可行组合中去重采样，并受每组和
总输出上限约束；它不会声称已穷举超出预算的全部排列。

## 操作示例

### 场景：小晶胞 HEA 的“目标 20%”反复落到同一个实际组成

32 位点随机固溶体经过连续 Composition Space Sampling 后再取整，多个目标点可能都变成同一整数计数组合，训练集看似有很多组成，实际却重复。

**输入：** 32 原子 A1/fcc 周期超胞。
**目标：** 直接枚举 Fe/Co/Ni 的可实现整数计数，并为每个组成生成 4 个不同排布。
**参数设置：** `位点划分规则`使用“全部位点”（若输入含 A 单子晶格则用 `A`）和“计数范围”，
给三种元素明确整数范围；`每种成分的排列数`填 `4`，固定`随机种子`，并设置
`每个输入的最大输出数`。
**输出：** 每个 composition ID 对应唯一整数计数组合；同一 composition 下的 arrangement ID 不重复。
**怎么验证训练集质量改善：** 统计 metadata 中的 `counts`，确认每帧总和为 32、composition ID 无重复；重训时再按实际分数切片比较误差，而不是按原连续目标比例分组。

## 参数说明

### 位点划分规则（site_rules）

默认界面先显示 A/B 规则模板中的 `X` 占位符，但 `X` 不是可输出的元素。接入上游结构且规则尚未被手动编辑时，卡片会立即按首个输入结构的实际 site set、元素和当前比例生成安全规则。例如 Ni/Al 的 A/B 结构会自动得到真实的 Ni、Al 规则，而不会生成 `X` 原子。

```json
{"A":{"composition":{"X":1.0},"elements":["X"],"mode":"fixed_fraction"},"B":{"composition":{"X":1.0},"elements":["X"],"mode":"fixed_fraction"}}
```

“无子晶格标签（all）”模板会建立 `all` 规则；“A/B 子晶格”模板会建立彼此独立的 A、B 规则。JSON 顶层 key 必须与输入的 site set 完全一致。有 `sublattice` 时逐个编辑 A、B 等标签；没有时使用 `all`。卡片不会把 `all` 静默套到每个子晶格，也不会使用 `group` 代替 `sublattice`。

规则仍由卡片自动管理时，载入上游结构会按实际标签和元素自动匹配：普通无标签结构使用 `all`，A1/A2/A3 等单子晶格原型使用单独的 `A` 规则，L1₂/B2/L1₀ 保持 A/B。界面会明确提示自动匹配结果。只要用户编辑过规则、主动选择过模板、粘贴过 JSON 或载入过持久化参数，卡片就不再自动覆盖，标签不匹配或仍含 `X` 时改为显示错误。

切换位点划分或应用模板会整体替换当前规则。已有手动编辑时，界面会先确认，避免误删元素、范围和 site set。

输入结构可用后，界面会在对应规则旁显示实际位点数，并即时提示缺少标签、多余标签、非法/重复元素、范围错误和无整数解。底层持久化仍是 `位点划分规则`（`site_rules`）字符串；“高级：查看/粘贴 JSON”默认收起，可用于复制现有格式或粘贴旧配置。非法 JSON 不会覆盖当前有效的可视化规则。

固定组成示例：

```json
{"A":{"elements":["Fe","Co"],"mode":"fixed_fraction","composition":{"Fe":0.5,"Co":0.5}}}
```

可视化编辑器要求固定比例之和为 1，例如 `0.5 + 0.5`。旧项目或直接调用 operation 的 JSON 若使用 `1 + 1` 这类相对权重，底层仍会归一化以保持兼容；界面新建和修改规则时不再依赖这种隐式归一化。

分数范围示例：

```json
{"B":{"elements":["Al","Ni"],"mode":"fraction_range","fractions":{"Al":[0.25,0.75],"Ni":[0.25,0.75]}}}
```

整数范围示例：

```json
{"A":{"elements":["Fe","Co"],"mode":"count_range","counts":{"Fe":[8,16],"Co":[8,16]}}}
```

### 每种组成的排布数（arrangements_per_composition）

`int`，默认 1。每个实际整数计数组合请求多少个不同的原子排布。

如果某个组成的理论唯一排布数小于请求值，只返回理论上存在的数量。例如两个位点各放一个 Fe/Co 只有 2 种排布，请求 10 也不会复制结果凑数。

### 使用固定种子（use_seed）

`bool`，默认 true。开启后，组成预算抽样和每个组成的排布都由固定 seed 派生。

组成可行域本身不随 seed 改变；seed 只决定超预算时选哪些计划以及元素落到哪些具体位点。

### 随机种子（seed）

`int`，默认 0。固定随机路径的基础种子。

生效条件：`use_seed=true`。相同输入、规则和 seed 会得到相同 composition/arrangement 顺序与占位；更换 seed 时实际计数不变，但排布通常会改变。

### 每个输入的最大输出数（max_outputs）

`int`，默认 200。单个输入结构最多返回的总结构数。

预算不足以覆盖全部组成时，operation 先在整数计划索引空间中按固定步长确定性选择可覆盖的组成，再采用“组成优先”的轮转调度：先为每个选中组成生成第 1 个排布，再生成各组成的第 2 个排布，以此类推。因此，小预算不会先被单一组成的多个排布耗尽。

输入结构与规则都可用时，UI 用数据集中的**首个输入结构**计算位点数和可行整数组成数，并显示输出上限估计。这个数字是 `可行组成数 × 每组成请求排布数` 的上界；某些组成的唯一排布数不足时，实际输出会更少。没有输入时只提示先载入上游结构，不显示伪精确数字。

## 推荐预设

### 32 位点二元随机固溶体

```json
{
  "class": "FiniteCellAlloyOccupancyCard",
  "params": {
    "site_rules": "{\"A\":{\"elements\":[\"Fe\",\"Co\"],\"mode\":\"count_range\",\"counts\":{\"Fe\":[8,24],\"Co\":[8,24]}}}",
    "arrangements_per_composition": 4,
    "use_seed": true,
    "seed": 17,
    "max_outputs": 68
  }
}
```

### L1₂ 子晶格部分无序

```json
{
  "class": "FiniteCellAlloyOccupancyCard",
  "params": {
    "site_rules": "{\"A\":{\"elements\":[\"Fe\",\"Co\",\"Ni\"],\"mode\":\"fraction_range\",\"fractions\":{\"Fe\":[0.25,0.5],\"Co\":[0.25,0.5],\"Ni\":[0.0,0.5]}},\"B\":{\"elements\":[\"Al\",\"Ta\"],\"mode\":\"count_range\",\"counts\":{\"Al\":[4,8],\"Ta\":[0,4]}}}",
    "arrangements_per_composition": 3,
    "use_seed": true,
    "seed": 23,
    "max_outputs": 120
  }
}
```

## 推荐组合

- `Ordered Alloy Prototype → Finite-Cell Alloy Occupancy`：覆盖 L1₂、B2、L1₀ 的有序和分子晶格无序占位。
- `Crystal Prototype Builder → Super Cell → Finite-Cell Alloy Occupancy`：普通 FCC/BCC 结构没有 `sublattice` 时使用 `all` 单 site set 生成随机固溶体。
- `Finite-Cell Alloy Occupancy → Atomic Perturb → Lattice Strain`：占位确定后再补坐标和晶格扰动；磁矩仍交给 Magnetic Order 或 Spin Disorder。

## 常见问题

**提示 missing rules 或 unknown site sets。** 顶层 key 必须逐一匹配输入 `sublattice` 的实际标签；普通结构只能使用 `all`。operation 不会回退到 `group`。

**提示没有整数解。** 各元素计数下限之和可能大于位点数，或上限之和小于位点数。分数范围也会先换算成当前晶胞可实现的整数边界，再检查总和。

**输出少于 `每种组成的排布数`（`arrangements_per_composition`）。** 该组成的理论排布数已耗尽，或触发了 `每个输入的最大输出数`（`max_outputs`）。这不是随机失败，卡片不会用重复排布伪造样本数。触发总输出上限时，卡片优先覆盖不同组成，再为同一组成增加排布。

**提示仍含占位元素 X。** `X` 只用于没有输入时展示规则结构，operation 会拒绝生成含 `X` 的结果。接入真实结构后可让未编辑的规则自动匹配，也可以手动把 `X` 替换为实际元素。

## 输出标签

`Config_type` 追加 `FiniteAlloy(comp=<composition_id>,arr=<arrangement_id>)`。`finite_cell_alloy` metadata 是 JSON 字符串，包含每个 site set 的实际 `counts`、`fractions`、请求规则、实现状态、composition ID、arrangement ID 和派生 seed。

## 可复现性

固定 `use_seed=true` 与 `随机种子`（`seed`）后严格可复现。超预算组成抽样使用确定索引顺序；排布去重以逐位元素序列为准，不依赖集合迭代顺序。
