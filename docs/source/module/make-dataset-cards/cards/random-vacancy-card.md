<!-- card-schema: {"card_name": "Random Vacancy", "source_file": "src/NepTrainKit/ui/views/_card/random_vacancy_card.py", "serialized_keys": ["params"]} -->

# 按规则生成空位（Random Vacancy）

`Group`: `Defect` | `Class`: `RandomVacancyCard`

## 功能说明

按规则从指定元素的候选位点中均匀随机选择原子并删除。每条规则控制“删除哪种元素、删除多少个、是否限制在已有 `group` 标签内”；没有被删除的原子坐标、晶胞和 PBC 保持不变。

**和 `Vacancy Defect Generation` 的区别：**
- `Random Vacancy`：规则驱动，按元素和已有 group 限定候选池，再随机选择具体位点，适合定向研究某类空位缺陷
- `Vacancy Defect Generation`：统计驱动，按整体浓度或数量随机删除，适合快速生成低/中/高缺陷强度的分布样本

本卡不会识别表面、体相或化学子晶格。`group` 是输入结构中已经存在的 `atoms.arrays["group"]` 标签；如果规则填写了 group 而输入没有对应数组或标签，卡片会明确报错，不会退化成全元素删位。

## 操作示例

### 场景：模型对表面氧空位预测完全错误

你在 LiCoO2 上训练了一个 NEP 模型，体相性质预测很好，但一跑表面 slab 加氧空位的构型，能量误差是体相的 3 倍。检查发现训练集里没有缺氧的结构，模型根本不知道氧空位附近 Co 原子的局域环境长什么样。

**诊断思路：** 氧空位周围，Co 原子从正常的 Co-O 八面体配位变成五配位甚至四配位，键长、电荷分布都变了。训练集里只有完美晶体结构，模型完全靠外推处理这些配位变化。需要往训练集中加入精确控制的氧空位构型，让模型见过"氧被拿走"之后的配位环境。

**输入：** 一个 LiCoO2 的 slab 结构，并且上游建模或导入流程已经写入 `surface` / `bulk` group 标签

**目标：** 只删表面层的氧原子，每次删 1~3 个，每帧生成 20 个不同落点的空位版本

**参数设置：**
- `Rules`：`[{"element":"O","count_mode":"random","count":[1,3],"group":["surface"]}]`
- `Maximum Outputs per Input`：`20`
- `Use Seed`：勾选，`Seed`：`[42]`

**输出：** 最多 20 个不重复的空位结构，每个结构中 1~3 个表面氧被删除，带 `Vac(n=...)` 标签。若可用位点组合不足，实际输出数会少于 20。

**怎么验证训练集质量改善：**
- 重训后用同样含氧空位的表面结构推理，能量误差应显著下降
- 抽查删除位置最近邻 Co 原子的键长分布是否在物理合理范围
- 如果只有表面需要空位，坚持用 group 约束；去掉 group 会生成体相空位，稀释训练数据
- 如果需要同时覆盖体相和表面空位，加第二条规则不带 group
- 删除操作不会自动弛豫空位近邻；进入训练集前应做几何检查，并按任务需要进行 DFT 弛豫或单点计算

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

`list[dict[str, Any]]`，默认空列表。界面默认提供一条空规则；必须填写元素后才能运行。

每条 rule 控制一次删除——删什么元素、删几个、在哪个 region 删：

| 字段 | 类型 | 说明 |
|------|------|------|
| `element` | string | 被删除的元素，如 `O` |
| `count_mode` | string | `fixed` 精确删除 `count[0]` 个原子；`random` 在闭区间 `count[0]..count[1]` 中随机 |
| `count` | [min, max] | 固定数量写 `[n,n]`，随机范围写 `[min,max]` |
| `group` | string / list（可选） | 限制只删除输入结构已有 group 标签内的原子。界面里可写 `surface_top,surface_bottom` |

固定数量必须至少为 1；随机范围允许最小值为 0，以便把原始结构作为一个可能端点，但最大值必须至少为 1。请求上限不能超过候选原子数，也不能把结构全部删空。

多条规则按顺序执行。如果两条规则的候选池重叠，第二条只会看到第一条删除后剩余的原子。随机范围偶尔抽到不可行组合时，程序会丢弃这一次尝试并继续采样，不会因为一个随机分支终止整张卡片；如果所有尝试都无法留下至少一个原子，才会明确报错。

### Max Structures（max_structures）

`int`，默认 1。每个输入帧最多保留多少个不同的空位版本。卡片按删除位点集合去重，因此可用组合不足时输出会少于该值。

10~30 做定向验证，30~50 做常规覆盖，50~100 建议后接一张 `FPS Filter`，否则很多空位落点在描述符空间里高度重复。

### Use Seed（use_seed）

`bool`，默认 false。打开后同一结构内容 + 同一参数 + 同一 seed 会得到完全相同的删除结果。不同输入结构会派生不同随机序列，改变数据集顺序不会改变单帧结果。

### Seed（seed）

`int`，默认 0。随机种子值。

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

- `Group Label` → `Random Vacancy`：可按坐标规则生成 A/B 两组后定向删位；`Group Label` 不会自动识别 surface/bulk 或化学子晶格
- 已有 `surface` / `bulk` / `sublattice` 标签 → `Random Vacancy`：直接填写已有标签定向删位
- `Super Cell` → `Random Vacancy`：先扩胞到足够大，避免小胞里缺陷相互作用过强
- `Random Vacancy` → `FPS Filter`：大批量生成后做代表性筛选

## 常见问题

**提示“需要填写元素”。** 默认规则仍为空；填写要删除的元素符号，例如 `O`。

**提示输入没有 group 或没有匹配原子。** 本卡不会猜测或忽略 group。先确认输入结构确实带有对应 `atoms.arrays["group"]` 标签，或清空规则中的 group 限制。

**生成数少于设置值。** 输出会按删除位点集合去重。例如只有一个 O 位点且固定删除 1 个时，不论上限设置多大都只有一种结构。

**删位后局部结构不合理。** 本卡只删除原子，不移动剩余原子。在小胞中删除过多原子会产生不合理化学计量和强缺陷相互作用；应先扩胞，并在下游进行几何检查或弛豫。

**多条规则交互异常。** 规则顺序执行。如果一条规则删了大量原子，下一条规则的候选池会变小。随机模式会跳过偶发的不可行组合；如果持续提示无法生成非空结构，说明规则的固定数量或最小删除数已经互相冲突，应减少删除数、拆开 group，或先扩胞。

## 输出标签

`Vac(n={删除原子数})`

## 可复现性

勾选 `use_seed` 并固定 `seed` 后，随机序列由 seed 与单帧结构内容共同决定：同一结构可复现，不同结构不会机械复用同一组原子编号，数据集重新排序也不改变单帧结果。
