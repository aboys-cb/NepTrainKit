<!-- card-schema: {"card_name": "Finite-Cell Alloy Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/finite_cell_alloy_occupancy_card.py", "serialized_keys": ["params"]} -->

# 有限晶胞合金占位（Finite-Cell Alloy Occupancy）

**分类：** 合金与组分

## 功能说明

这张卡把合金比例落实为有限晶胞中真正可实现的整数原子数，再为每种整数组成生成不重复的占位排布。输入含 `atoms.arrays["sublattice"]` 时，各子晶格分别约束；没有该数组时，全部位点属于名为 `all` 的单一位点集合。

组成模式有三种：

- `fixed_fraction`：指定固定目标比例；可整除时记为 `exact`，否则用最大余数法得到最近整数计划，并记为 `nearest_integer`。
- `fraction_range`：枚举实际整数占比落在各元素范围内的计划。
- `count_range`：枚举各元素整数计数落在范围内、且总和等于位点数的计划。

## 原理与公式

有限晶胞不能实现任意连续比例。一个位点集合含 $N$ 个位点时，程序寻找满足

$$
n_e\in\mathbb Z_{\ge0},\qquad \sum_e n_e=N
$$

的整数计数。无量纲占比范围 $f_e\in[0,1]$ 会转换为

$$
\lceil f_{e,\min}N\rceil\le n_e\le\lfloor f_{e,\max}N\rfloor.
$$

给定一组计数后，该位点集合的不同排布总数为

$$
\Omega=\frac{N!}{\prod_e n_e!}.
$$

多个子晶格的排布数是各自 $\Omega$ 的乘积。设选中的整数组成集合为 $P$，每种组成请求 $A$ 个排布，总输出上限为 $M$，则实际输出数为

$$
N_\mathrm{out}=\min\left(M,\sum_{i\in P}\min(A,\Omega_i)\right).
$$

当预算不足时，卡片先覆盖不同组成，再轮流增加每种组成的第 2、第 3 个排布。它不会复制排布凑数，也不会把超出预算的结果称为已穷举。

## 操作示例

### 小晶胞 HEA 的多个目标比例落到同一实际组成

32 位点随机固溶体若先采连续比例再取整，多个目标点可能变成同一整数计数组合。此时应直接使用“计数范围”枚举 Fe/Co/Ni 的可行计数，设置“每种组成的排布数”为 4，并用“每个输入的最大输出数”控制规模。

输出中，相同实际计数组合具有相同的 composition ID；同一 composition 下的 arrangement ID 不重复。验证时应检查每帧 `counts` 之和为 32、`(composition_id, arrangement_id)` 二元组唯一，并按 metadata 中的实际占比切片比较模型误差。

## 参数说明

### 位点划分规则（site_rules）

可视化编辑器用于选择“全部位点”或“按子晶格”，并为每个位点集合设置元素和组成模式。JSON 顶层 key 必须与输入的 `sublattice` 标签完全一致；无标签结构只使用 `all`。`group` 不会代替 `sublattice`。

未手动编辑规则时，接入上游结构会按**首个输入结构**的标签、元素和当前占比自动匹配规则。默认模板中的 `X` 只是占位符，不能输出；手动编辑、选择模板、粘贴 JSON 或恢复持久化参数后，规则不再被自动覆盖。

固定比例为无量纲数，范围是 $[0,1]$，且可视化编辑器要求同一位点集合内的比例和为 1。计数范围的单位是“个位点”，上下限必须是非负整数。旧 JSON 中 `1 + 1` 这类相对权重仍可由底层归一化，但新规则应直接填写 `0.5 + 0.5`。

规则示例：

```json
{"A":{"elements":["Fe","Co"],"mode":"count_range","counts":{"Fe":[8,16],"Co":[8,16]}}}
```

“高级：查看或粘贴 JSON”可复制或载入旧规则；非法 JSON 不会覆盖当前有效规则。切换位点划分或应用模板会替换整组规则，已有手动编辑时界面会先确认。

### 每种组成的排布数（arrangements_per_composition）

`int`，默认 1，单位为“个结构”。它表示每种实际整数组成请求多少个不同排布；若理论排布数 $\Omega$ 更少，则只返回实际存在的数量。

### 使用固定种子（use_seed）

`bool`，默认 true。开启后，组成预算抽样和具体位点排布均由固定 seed 派生。

### 随机种子（seed）

`int`，默认 0，无单位，仅在 `use_seed=true` 时生效。相同输入、规则和 seed 会得到相同结果。若所有可行组成都在预算内，改变 seed 不改变可行计数集合，只改变排布；若 `max_outputs` 截断了组成集合，seed 也会改变被选中的整数组成。

### 每个输入的最大输出数（max_outputs）

`int`，默认 200，单位为“个结构”。它限制每个输入结构返回的总结构数，并采用“先覆盖组成、再增加排布”的分配顺序。

## 输出预览与失败条件

UI 只用数据集的**首个输入结构**计算位点数、可行整数组成数和输出上界。若同一数据集中的结构具有不同原子数、`sublattice` 标签或位点数，后续结构会按各自真实位点重新计算，输出数可能不同，也可能因规则不匹配而失败；这类数据最好分组处理。

以下情况会显式失败，不会返回伪成功的空结果：规则与位点集合不匹配、仍含 `X`、比例或计数范围非法，以及没有可行整数解。对于排布空间很大的组成，程序采用有上限的去重随机尝试；若尝试耗尽而未达到本应存在的目标数量，也会显式报告已生成数量与请求数量。

## 推荐配置

32 位点二元合金可从以下规则开始，再按训练集缺口收紧计数范围：

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

## 推荐组合

- `Ordered Alloy Prototype → Finite-Cell Alloy Occupancy`：保留 L1₂、B2、L1₀ 等原型的子晶格划分，并在各子晶格内生成受约束的无序占位。
- `Crystal Prototype Builder → Super Cell → Finite-Cell Alloy Occupancy`：先得到足够位点，再生成可实现的随机固溶体组成。
- `Finite-Cell Alloy Occupancy → Atomic Perturb → Lattice Strain`：先确定占位，再补充坐标和晶格扰动。

## 常见问题

**为什么输出少于请求值？** 某种组成的理论排布数可能不足，或总输出已达到 `max_outputs`；卡片不会复制结构凑数。

**为什么提示规则不匹配？** 顶层 key 必须逐一匹配输入的 `sublattice` 标签；无标签结构只能使用 `all`。

**为什么没有整数解？** 各元素下限之和可能大于位点数，或上限之和小于位点数；占比范围也会先换算为当前晶胞可实现的整数边界。

## 输出标签与可复现性

`Config_type` 追加 `FiniteAlloy(comp=<composition_id>,arr=<arrangement_id>)`。`finite_cell_alloy` metadata 保存实际 `counts`、`fractions`、请求规则、实现状态、composition ID、arrangement ID 和派生 seed。

固定 `use_seed=true` 和 `seed` 后结果可复现；关闭固定种子后不保证重复运行得到相同组成抽样或排布。
