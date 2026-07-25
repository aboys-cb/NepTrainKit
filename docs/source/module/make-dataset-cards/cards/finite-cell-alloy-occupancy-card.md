<!-- card-schema: {"card_name": "Finite-Cell Alloy Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/finite_cell_alloy_occupancy_card.py", "serialized_keys": ["params"]} -->

# 有限晶胞合金占位（Finite-Cell Alloy Occupancy）

`Group`: `Alloy` | `Class`: `FiniteCellAlloyOccupancyCard`

## 功能说明

按有限位点上真正可实现的整数计数生成合金占位。输入若含 `atoms.arrays["sublattice"]`，每个子晶格独立约束；普通结构则全部位点组成名为 `all` 的单一 site set。

组成计划先确定每种元素的整数个数，再生成不重复的随机排布。卡片不会先采任意连续比例再静默取整，也不会把不可整除的目标写成 Exact。metadata 同时保存请求、实际计数、实际分数、composition ID 和 arrangement ID。

卡片主界面是可视化规则编辑器。先选择“全部位点”或“按子晶格”，再为每个位点集合添加元素，并选择一种组成模式。模式切换后只显示当前需要的比例或计数字段：

- `fixed_fraction`：给出固定目标比例；可整除时记为 `exact`，否则采用确定性的最近整数计划并记为 `nearest_integer`。
- `fraction_range`：枚举实际整数分数落在各元素范围内的计划。
- `count_range`：枚举各元素整数计数落在范围内且总和等于位点数的计划。

## 操作示例

### 场景：小晶胞 HEA 的“目标 20%”反复落到同一个实际组成

32 位点随机固溶体经过连续 Composition Sweep 后再取整，多个目标点可能都变成同一整数计数组合，训练集看似有很多组成，实际却重复。

**输入：** 32 原子 A1/fcc 周期超胞。
**目标：** 直接枚举 Fe/Co/Ni 的可实现整数计数，并为每个组成生成 4 个不同排布。
**参数设置：** `site_rules` 使用 `all`（若输入含 A 单子晶格则用 `A`）和 `count_range`，给三种元素明确整数范围；`arrangements_per_composition=4`，固定 seed，设置 `max_outputs`。
**输出：** 每个 composition ID 对应唯一整数计数组合；同一 composition 下的 arrangement ID 不重复。
**怎么验证训练集质量改善：** 统计 metadata 中的 `counts`，确认每帧总和为 32、composition ID 无重复；重训时再按实际分数切片比较误差，而不是按原连续目标比例分组。

## 参数说明

### 位点划分与 Site Rules（site_rules）

默认界面使用“A/B 有序合金”模板，与默认的 Ordered Alloy Prototype 可直接连接：

```json
{"A":{"composition":{"X":1.0},"elements":["X"],"mode":"fixed_fraction"},"B":{"composition":{"X":1.0},"elements":["X"],"mode":"fixed_fraction"}}
```

“普通单晶格合金”按钮会建立 `all` 规则；“A/B 有序合金”按钮会建立彼此独立的 A、B 规则。JSON 顶层 key 必须与输入的 site set 完全一致。有 `sublattice` 时逐个编辑 A、B 等标签；没有时使用 `all`。卡片不会把 `all` 静默套到每个子晶格。

规则仍处于初始占位状态时，载入上游结构会按实际标签自动匹配：普通无标签结构使用 `all`，A1/A2/A3 等单子晶格原型使用单独的 `A` 规则，L1₂/B2/L1₀ 保持 A/B。界面会明确提示自动匹配结果。只要用户编辑过规则、主动选择过模板、粘贴过 JSON 或载入过持久化参数，卡片就不再自动覆盖，标签不匹配时改为显示错误。

输入结构可用后，界面会在对应规则旁显示实际位点数，并即时提示缺少标签、多余标签、非法/重复元素、范围错误和无整数解。底层持久化仍是 `site_rules` 字符串；“高级：查看/粘贴 JSON”默认收起，可用于复制现有格式或粘贴旧配置。非法 JSON 不会覆盖当前有效的可视化规则。

固定组成示例：

```json
{"A":{"elements":["Fe","Co"],"mode":"fixed_fraction","composition":{"Fe":0.5,"Co":0.5}}}
```

分数范围示例：

```json
{"B":{"elements":["Al","Ni"],"mode":"fraction_range","fractions":{"Al":[0.25,0.75],"Ni":[0.25,0.75]}}}
```

整数范围示例：

```json
{"A":{"elements":["Fe","Co"],"mode":"count_range","counts":{"Fe":[8,16],"Co":[8,16]}}}
```

### Arrangements per Composition（arrangements_per_composition）

`int`，默认 1。每个实际整数计数组合请求多少个不同的原子排布。

如果某个组成的理论唯一排布数小于请求值，只返回理论上存在的数量。例如两个位点各放一个 Fe/Co 只有 2 种排布，请求 10 也不会复制结果凑数。

### Use Seed（use_seed）

`bool`，默认 true。开启后，组成预算抽样和每个组成的排布都由固定 seed 派生。

组成可行域本身不随 seed 改变；seed 只决定超预算时选哪些计划以及元素落到哪些具体位点。

### Seed（seed）

`int`，默认 0。固定随机路径的基础种子。

生效条件：`use_seed=true`。相同输入、规则和 seed 会得到相同 composition/arrangement 顺序与占位；更换 seed 时实际计数不变，但排布通常会改变。

### Max Outputs（max_outputs）

`int`，默认 200。单个输入结构最多返回的总结构数。

预算不足以覆盖全部组成时，operation 在整数计划索引空间中按固定步长确定性抽样，不依赖 Python 集合、目录或文件系统顺序。输入结构与规则都可用时，UI 直接调用 `operation.estimate(...)`，显示各标签位点数、可行整数组成数、每组成请求排布数、截断前理论输出、`max_outputs`、预计实际输出和是否截断。没有输入时只提示先运行或载入上游结构，不显示伪精确数字。

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

### L1₂ 分子晶格部分无序

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

**输出少于 `arrangements_per_composition`。** 该组成的理论排布数已耗尽，或触发了 `max_outputs`。这不是随机失败，卡片不会用重复排布伪造样本数。

## 输出标签

`Config_type` 追加 `FiniteAlloy(comp=<composition_id>,arr=<arrangement_id>)`。`finite_cell_alloy` metadata 是 JSON 字符串，包含每个 site set 的实际 `counts`、`fractions`、请求规则、实现状态、composition ID、arrangement ID 和派生 seed。

## 可复现性

固定 `use_seed=true` 与 `seed` 后严格可复现。超预算组成抽样使用确定索引顺序；排布去重以逐位元素序列为准，不依赖集合迭代顺序。
