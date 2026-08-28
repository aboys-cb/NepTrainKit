<!-- card-schema: {"card_name": "Random Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/random_occupancy_card.py", "serialized_keys": ["params"]} -->

# 随机占位（Random Occupancy）

**分类：** 合金与组分

## 功能说明

这张卡把目标比例落实为实际原子占位。目标组成只能来自二者之一：`Auto (Comp tag)` 严格读取输入 `Config_type` 中最后一个 `Comp(...)` 标签；`Manual` 严格读取界面的“元素 / 目标比例”表格。所选来源缺少有效组成时会报错，不会退回另一来源。

它适合接在 `Composition Space Sampling` 后把组成计划变成真实结构，也可对手工组成生成多个随机占位样本。若只想替换某一种现有元素，应使用 `Random Doping`。

## 原理与公式

设目标位点数为 $N$，归一化目标比例为 $p_e$。`Exact` 使用最大余数法得到固定的最近可行整数计数：

$$
n_e^{(0)}=\lfloor Np_e\rfloor,\qquad
\sum_e n_e=N,
$$

不足的位点按余数 $Np_e-n_e^{(0)}$ 从大到小补齐。若多个元素的余数相同，连续比例本身不能唯一决定多出的位点给谁，因此应以界面预览和输出 metadata 中的实际整数计数为准。`Random` 则对每个样本独立抽取

$$
\mathbf n\sim\operatorname{Multinomial}(N,\mathbf p).
$$

两种模式都会随机打乱元素在目标位点上的位置。`Exact` 的每个样本具有相同整数计数，但该计数不一定与连续目标比例完全相等；`Random` 的计数会在样本间波动，只在统计平均上接近目标比例。

填写 `group_filter` 时，$N$ 只计算 `atoms.arrays["group"]` 中命中标签的位点，其他位点的元素保持不变。每个输入严格生成 `samples` 个输出，因此

$$
N_\mathrm{out}=N_\mathrm{input}\times \mathrm{samples}.
$$

不同样本可能得到相同计数甚至相同排布；卡片不去重，也不声称直接控制短程有序参数。

## 操作示例

### 同一组成缺少占位多样性

如果训练集中每种组成只有一种占位，模型可能没有见过同一整数计数下的其他局域化学环境。可在带 `Comp(Co=0.3333,Cr=0.3333,Ni=0.3334)` 标签的结构后添加本卡，选择 `Exact` 并把 `samples` 设为 5。

每个输入会得到 5 个随机占位样本，整数计数相同但排布允许重复。验证时先检查 `random_occupancy` metadata 中的 `actual_counts` 与 `eligible_sites`，再按实际组成比较不同排布上的留出误差；若唯一排布数很小，应减少 `samples`，而不是把重复输出当成新覆盖。

## 参数说明

### 来源（source）

`str`，默认 `Auto (Comp tag)`。可选值为 `Auto (Comp tag)` 和 `Manual`。Auto 严格读取 `Config_type` 中最后一个有效 `Comp(...)`；Manual 严格读取下面的表格，两者不会互相回退。

### 手动（manual）

`str`，默认空。仅在 `source=Manual` 时显示“元素 / 目标比例”表格；序列化为 `Co:0.333,Cr:0.333,Ni:0.334`。比例是无量纲非负权重，运行时会自动归一化，至少需要一个正值。

### 模式（mode）

`str`，默认 `Exact`。`Exact` 为每个样本使用同一组最近可行整数计数；`Random` 为每个样本从多项分布重新抽取计数。两种模式都会随机分配具体位置。

### 样本数（samples）

`int`，默认 1，单位为“个结构/输入”。输出数量严格等于输入帧数乘以该值，但输出之间不保证唯一。

### 分组筛选（group_filter）

`str`，默认空。逗号分隔的 group 标签，如 `A,B`。留空表示全部位点；非空时只改变命中标签的局部目标集合。输入必须带 `atoms.arrays['group']` 且至少有一个标签命中，否则显式失败。

### 使用随机种子（use_seed）

`bool`，默认 false。开启后，程序由基础 seed、输入 `Config_type` 的稳定 ID 和样本序号派生随机种子。

### 随机种子（seed）

`int`，默认 0，无单位，仅在 `use_seed=true` 时生效。相同输入参数、`Config_type` 和 seed 可复现；数据集中的帧顺序不参与派生。

## 推荐配置

```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "params": {
    "source": "Auto (Comp tag)",
    "manual": "",
    "mode": "Exact",
    "samples": 5,
    "group_filter": "",
    "use_seed": true,
    "seed": 42
  }
}
```

## 推荐组合

- `Composition Space Sampling` → `Random Occupancy`：标准合金 pipeline，配比 → 落位。
- `Layer Groups` → `Random Occupancy`：先按原子层写入 group 标签，再限制占位区域。

## 常见问题

**提示缺少目标成分。** Auto 模式下检查上游是否有有效 `Comp(...)`；Manual 模式下检查表格是否至少含一个正权重。卡片不会静默使用另一来源。

**实际比例与 Comp 标签不完全相等。** 有限位点需要整数计数。Exact 返回最近可行计数；Random 还包含多项抽样波动。应以输出 chemical symbols 和 metadata 的 `actual_counts` 为准。

**多个输出完全相同。** 样本是独立随机抽取，不做唯一性筛选；位点少或组成偏斜时重复概率更高。

**group_filter 报错。** 检查输入是否有与原子数等长的 `atoms.arrays['group']`，且至少一个标签命中。空结构、空标签项、缺少数组和零命中均会显式失败。

## 输出标签

`Config_type` 追加 `Occ(E)` / `Occ(R)`；使用固定种子时写为 `Occ(E,s=...)` / `Occ(R,s=...)`。E 和 R 只表示计数模式，不表示排布唯一。

`atoms.info["random_occupancy"]` 是 JSON 字符串，包含目标组成 `target`、实际 `actual_counts` / `actual_fractions`、目标位点数 `eligible_sites`、分组 `groups`、模式 `mode`、样本序号 `sample_index` 和实际派生 `seed`。实际化学组成以输出结构的 chemical symbols 与这些字段为准；原 `Comp(...)` 仍表示请求目标。

未知元素、负数或非有限权重、非法模式、非正 `samples`、负 seed、空结构及无有效目标位点都会显式失败，不会返回伪成功结果。
