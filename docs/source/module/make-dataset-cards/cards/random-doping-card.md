<!-- card-schema: {"card_name": "Random Doping", "source_file": "src/NepTrainKit/ui/views/_card/random_doping_card.py", "serialized_keys": ["params"]} -->

# 随机掺杂（Random Doping）

**分类：** 合金与组分

## 功能说明

按一条或多条规则，把指定元素的部分位点替换为掺杂元素。每条规则回答四个问题：替换哪种元素、是否只处理特定 `group`、替换成什么元素、替换多少个。它适合构造低浓度或局部分组掺杂；若要按一个整体目标组成重新分配全部位点，应使用 `Random Occupancy`。

规则从上到下作用于同一份当前结构。前一条规则改变的元素可能成为后一条规则的候选，因此具有依赖关系的规则必须按预期顺序排列。

## 原理与公式

设某条规则当前有 $N$ 个候选位点。

使用原子百分比 $p$ 时，替换数为

$$
n=\lfloor Np/100\rfloor.
$$

百分比填写为区间时，每个输出先在区间内均匀抽取一个百分比，再向下取整。小结构上的低百分比可能始终得到 $n=0$；如果整套规则的最大可能替换数仍为 0，卡片会在运行前报错并建议提高用量或扩大结构。

“质量预算 %”沿用已有 `mass_percent` 数据语义：先把百分比乘以候选目标原子的原始总质量，再除以掺杂元素的平均原子质量：

$$
n=\left\lfloor
\frac{(p/100)N M_\mathrm{target}}{\bar M_\mathrm{dopant}}
\right\rfloor.
$$

它表示相对于原目标原子质量的替换预算，不是输出结构的最终质量分数。实际结果应从输出元素和 metadata 重新统计。

替换总数确定后，“掺杂元素分配”只决定多种掺杂元素如何分享这 $n$ 个位点：

- `Random`：每个位点按权重独立抽样，实际元素计数允许波动；
- `Exact`：使用最大余数法得到最近可行的整数计数，再随机打乱落点。

若掺杂权重采用质量比 $w_e$，程序先换算为原子抽样概率

$$
q_e=\frac{w_e/M_e}{\sum_j w_j/M_j}.
$$

因此顶层 `Random / Exact` 不改变替换位点总数，只改变 Ge/C 等多种掺杂元素之间的分配。

## 操作示例

### 补充 Si–Ge 局域环境

若模型只在纯 Si 结构上训练，Si 位点被 Ge 替换后的局域化学环境没有覆盖。可输入含 64 个 Si 位点的超胞，添加规则：

- 替换元素：`Si`
- 替换为：`Ge`
- 替换用量：原子百分比 `3–8%`
- 每个输入的输出数：`20`

界面预览会显示该输入有 64 个候选位点，整数替换范围为 1–5。运行后检查 `random_doping` metadata 中的实际替换数和元素计数，再用独立的掺杂结构验证集比较加入这些样本前后的能量、力误差。卡片只改变元素，原子坐标、晶胞和 PBC 保持不变。

## 参数说明

### 规则（rules）

`list[dict[str, Any]]`，默认空列表，但运行至少需要一条完整规则。每条规则包含：

- `target`：被替换的元素；
- `dopants`：掺杂元素及相对权重，例如 `Ge:0.7,C:0.3`；
- `ratio_type`：`atom` 或 `mass`，表示 `dopants` 权重的基准；
- `use`：`atomic_percent`、`mass_percent` 或 `count`；
- `percent`：百分比的最小值和最大值；
- `count`：原子数的最小值和最大值；
- `count_mode`：`fixed` 或 `random`；
- `group`：可选的 group 标签列表。

固定数量要求 `count` 的两端相同；随机数量在闭区间内抽取整数。填写 group 后，输入必须包含 `atoms.arrays["group"]` 且至少有一个对应目标元素命中。范围上界超过可用位点、规则重叠可能耗尽后续候选时，会在随机采样前确定性报错。

### 掺杂元素分配（doping_type）

`str`，默认 `Random`。`Random` 独立抽样各掺杂元素；`Exact` 用最大余数法固定每种掺杂元素的整数计数。二者都随机选择具体被替换位点。

### 每个输入的输出数（max_structures）

`int`，默认 1。参数名为兼容已有工作流保留，实际合同是严格数量：每个合法输入生成恰好该数量的输出，不是“最多”。总输出数为

$$
N_\mathrm{out}=N_\mathrm{input}\times\mathrm{max\_structures}.
$$

各输出独立抽样，允许得到相同替换数或相同落点，不做去重。

### 使用固定随机种子（use_seed）

`bool`，默认 false。开启后使用可复现的样本级派生种子。

### 随机种子（seed）

`int`，默认 0，无单位，仅在 `use_seed=true` 时生效。派生种子结合基础 seed、输入 `Config_type` 的稳定 ID 和样本序号；相同输入、参数和 seed 可复现。不同输入若 `Config_type` 也完全相同，仍可能复用相同随机路径。

## 推荐配置

```json
{
  "class": "RandomDopingCard",
  "check_state": true,
  "params": {
    "rules": [
      {
        "target": "Si",
        "dopants": {"Ge": 0.7, "C": 0.3},
        "ratio_type": "atom",
        "use": "atomic_percent",
        "percent": [3, 8],
        "count": [1, 1],
        "count_mode": "fixed",
        "group": []
      }
    ],
    "doping_type": "Exact",
    "max_structures": 20,
    "use_seed": true,
    "seed": 101
  }
}
```

## 常见问题

**为什么提示无法替换任何原子？** 候选位点数乘以百分比后会向下取整。检查界面预览的整数范围；若最大值仍为 0，请提高用量或先扩大结构。

**为什么选择固定配比后，替换总数仍会变化？** 固定配比只固定多种掺杂元素之间的整数分配。替换总数仍由每条规则的百分比或随机数量范围决定。

**为什么 group 规则没有运行？** 输入必须包含 `atoms.arrays["group"]`，标签拼写必须匹配，而且对应分组中必须存在被替换元素。

## 输出与迁移说明

`Config_type` 追加 `Dop(n=...)`。`atoms.info["random_doping"]` 是 JSON 字符串，记录分配模式、样本序号、实际派生 seed、总替换数，以及每条规则的候选位点数、实际替换数和实际掺杂元素计数。

只要元素实际改变，原有 calculator、能量、力、应力、virial、磁矩、spin、charges 等物种相关参考标签会被清除；坐标、晶胞、PBC、group 和普通来源信息保留。

本次调整保留旧 JSON 字段和值，但有三项有意变化：空规则和确定无替换效果的规则不再返回未改变结构；固定 seed 的旧随机序列会因样本级派生方式而改变；随机范围合法抽到 0 时仍会显式写入 `Dop(n=0)` 和 metadata，便于识别该端点。
