<!-- card-schema: {"card_name": "Random Vacancy", "source_file": "src/NepTrainKit/ui/views/_card/random_vacancy_card.py", "serialized_keys": ["params"]} -->

# 随机空位（Random Vacancy）

`Group`: `Defect` | `Class`: `RandomVacancyCard`

## 功能说明

按规则表对指定元素原子做删除操作，生成可控的空位缺陷结构。每条规则精确控制"删除什么元素、删多少个、在哪个 region 删"，支持元素过滤和 group 区域约束。

**和 `Vacancy Defect Generation` 的区别：**
- `Random Vacancy`：规则驱动，按元素+group 精确删除位点，适合定向研究某类空位缺陷
- `Vacancy Defect Generation`：统计驱动，按整体浓度或数量随机删除，适合快速生成低/中/高缺陷强度的分布样本

## 操作示例

### 场景：模型对表面氧空位预测完全错误

你在 LiCoO2 上训练了一个 NEP 模型，体相性质预测很好，但一跑表面 slab 加氧空位的构型，能量误差是体相的 3 倍。检查发现训练集里没有缺氧的结构，模型根本不知道氧空位附近 Co 原子的局域环境长什么样。

**诊断思路：** 氧空位周围，Co 原子从正常的 Co-O 八面体配位变成五配位甚至四配位，键长、电荷分布都变了。训练集里只有完美晶体结构，模型完全靠外推处理这些配位变化。需要往训练集中加入精确控制的氧空位构型，让模型见过"氧被拿走"之后的配位环境。

**输入：** 一个 LiCoO2 的 slab 结构，已经用 `Group Label` 标记了表面层和体相的 group

**目标：** 只删表面层的氧原子，每次删 1~3 个，每帧生成 20 个不同落点的空位版本

**参数设置：**
- `Rules`：`[{"element":"O","count":[1,3],"group":["surface"]}]`
- `Structures`：`[20]`
- `Use Seed`：勾选，`Seed`：`[42]`

**输出：** 20 个空位结构，每帧中 1~3 个表面氧被删除，带 `Vac(n=...)` 标签

**怎么验证训练集质量改善：**
- 重训后用同样含氧空位的表面结构推理，能量误差应显著下降
- 抽查删除位置最近邻 Co 原子的键长分布是否在物理合理范围
- 如果只有表面需要空位，坚持用 group 约束；去掉 group 会生成体相空位，稀释训练数据
- 如果需要同时覆盖体相和表面空位，加第二条规则不带 group

### 什么时候加这张卡、什么时候不加

**加：**
- 需要按元素和 group 精确控制空位位置，而不是按整体浓度随机删
- 研究特定元素的空位缺陷对模型预测的影响
- 下游磁性卡需要特定子晶格有明确的 vacancy pattern

**不加：**
- 只需要整体浓度覆盖 → 用 `Vacancy Defect Generation`
- 体系本身原子数很少（<10），删任何一个都会剧烈改变化学计量比

## 参数说明

### Rules（rules）
类型：`list[dict[str, Any]]`。默认：`field(default_factory=list)`。定义每条随机替换或空位生成规则。

JSON 字符串（界面中可用简化语法输入，程序自动转换）。每条 rule 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `element` | string | 被删除的元素，如 `O` |
| `count_mode` | string | `fixed` 精确删除 `count[0]` 个原子；`random` 在 `count[0]..count[1]` 之间随机 |
| `count` | [min, max] | 固定数量写 `[n,n]`，随机范围写 `[min,max]` |
| `group` | string / list（可选） | 限制只删除特定 group 标签内的原子。需要输入有 `atoms.arrays['group']`。界面里写成 `surface_top,surface_bottom` |

规则为空时，卡片不产生任何删除，输出 = 输入。

多条规则按顺序执行。如果两条规则操作同一个元素，第二条会在第一条的结果上继续删除。注意不要设计互相冲突的规则。

### Max Structures（max_structures）
类型：`int`。默认：`1`。限制每个输入结构最多输出多少个候选结构。

每输入帧生成多少个空位版本。

- 10~30：轻量定向验证
- 30~50：常规覆盖
- 50~100：建议后接 `FPS Filter`

### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

勾选 `use_seed` + 固定 `seed` 可复现。PM 随机性受 seed + 输入顺序联合控制。

### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

勾选 `use_seed` + 固定 `seed` 可复现。PM 随机性受 seed + 输入顺序联合控制。

生效条件：`use_seed=True`。

## 推荐预设

### 单元素单空位（2 个输出，验证规则命中用）
```json
{
  "class": "RandomVacancyCard",
  "check_state": true,
  "params": {
    "rules": [
      {"element": "O", "count_mode": "fixed", "count": [1, 1]}
    ],
    "max_structures": 2,
    "use_seed": true,
    "seed": 42
  }
}
```

### 单元素低浓度空位（20 个输出，常规覆盖）
```json
{
  "class": "RandomVacancyCard",
  "check_state": true,
  "params": {
    "rules": [
      {"element": "O", "count_mode": "random", "count": [1, 3]}
    ],
    "max_structures": 20,
    "use_seed": true,
    "seed": 42
  }
}
```

### 多元素带 group 约束（20 个输出，表面定向空位）
```json
{
  "class": "RandomVacancyCard",
  "check_state": true,
  "params": {
    "rules": [
      {"element": "O", "count_mode": "random", "count": [1, 3], "group": ["surface"]},
      {"element": "Li", "count_mode": "fixed", "count": [1, 1], "group": ["surface"]}
    ],
    "max_structures": 20,
    "use_seed": true,
    "seed": 42
  }
}
```

## 推荐组合

- `Group Label` → `Random Vacancy`：先标记 surface/bulk group，再定向删位
- `Super Cell` → `Random Vacancy`：先扩胞到足够大，避免小胞里缺陷相互作用过强
- `Random Vacancy` → `FPS Filter`：大批量生成后做代表性筛选

## 常见问题

**输出和输入一样，没有删除。** 检查 rules 是否为空、element 是否真的存在于输入结构中、group 过滤是否把候选位点全滤掉了。

**删完骨架塌缩。** 检查 count 上限是否太长。在 20 原子小胞里删 10 个必然崩。先扩胞再删。

**多条规则交互异常。** 规则顺序执行。如果一条规则删了大量原子，下一条规则的候选池可能已经变了。

## 输出标签

`Vac(n={删除原子数})`

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。输入结构顺序变化也会影响结果，建议把 seed 与 pipeline 配置一起版本化。
