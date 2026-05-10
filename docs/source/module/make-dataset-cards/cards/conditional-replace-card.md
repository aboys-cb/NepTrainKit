<!-- card-schema: {"card_name": "Conditional Replace", "source_file": "src/NepTrainKit/ui/views/_card/conditional_replace_card.py", "serialized_keys": ["params"]} -->

# 条件替换（Conditional Replace）

`Group`: `Alloy` | `Class`: `ConditionalReplaceCard`

## 功能说明

按空间坐标条件对指定元素做区域选择性替换。用 x/y/z 坐标表达式（如 `z>=8 and z<=10`）筛选候选位点，只替换命中区域内的原子，未命中区域保持原样。适合表面钝化、界面修饰、层状材料选择性改性等场景。

与 `Random Doping` 的区别：`Random Doping` 对全体 target 原子做无差别随机替换；`Conditional Replace` 先用坐标条件筛选位点，再在命中区域内做替换。

## 操作示例

### 场景：模型在表面化学上完全失效，但体相预测很好

你在 MgO 表面上训练了一个 NEP 模型，训练数据覆盖了完美晶体和表面弛豫结构。体相弹性常数和声子谱都很好，但一跑有表面吸附或表面钝化的结构，模型就崩了——能量误差是体相的 10 倍。诊断结果是：训练集里所有 O 原子都处于体相八面体配位环境，模型从未见过低配位表面 O 的局域环境，更不知道表面 O 被 F 替换后会发生什么。

**诊断思路：** 表面原子的化学环境与体相完全不同——配位数低、悬挂键多、更活泼。如果训练集缺少表面区域的元素变化（如 F 钝化表面 O），模型会对表面化学完全盲猜。解决办法是用条件替换把表面几层的特定元素换成钝化元素，让模型学习表面化学环境。

**输入：** 一个 MgO 表面 slab 结构，z 方向 0-16A，表面 O 位于高 z 区域

**目标：** 把顶层 2A 区域内的 O 替换为 F，模拟表面氟化

**参数设置：**
- `target` = `O`
- `replacements` = `F:1.0`
- `condition` = `z>=8 and z<=10`
- `mode` = `1` （Exact 模式，确保全部命中位点被替换）

**输出：** 1 个结构，满足 z 在 8-10A 的所有 O 被替换为 F，其余区域不变。带 `Repl(O->F)` 标签

**怎么验证训练集质量改善：**
- 重训后跑表面吸附测试，力的 MAE 应该显著降低，不再在表面区域出现异常大力误差
- 抽查输出，确认只有表面区域被替换，体相原子保持不变——如果体相也被动了，condition 表达式有问题
- 如果需要多元素钝化，把 `replacements` 写成 `F:0.5,Cl:0.5`
- 如果表面层厚度不确定，先用 `all` 条件验证替换配方正确性，再收紧 z 范围

### 什么时候加这张卡、什么时候不加

**加：**
- 模型对表面/界面/晶界区域的预测质量显著差于体相
- 需要做区域选择性化学改性：表面钝化、界面掺杂、层状材料夹层替换
- 替换规则可以用 x/y/z 坐标表达式写清楚

**不加：**
- 不需要空间选择性 → 用 `Random Doping` 更直接
- 替换规则无法用 x/y/z 表达式描述（如按近邻配位筛选）
- 只需要全局组合配比变化 → `Composition Sweep` + `Random Occupancy`

## 参数说明

### Target（target）

`str`，默认空。被替换的元素符号，如 `O`、`Si`。只替换匹配该元素且满足下面 `condition` 的那部分位点。

### Replacements（replacements）

`str`，默认空。替换配方，支持逗号冒号 `F:0.7,N:0.3` 或 JSON dict `{"F":0.7,"N":0.3}` 两种写法。比例会被自动归一化，在命中区域内按比例随机分配替换元素。

### Condition（condition）

`str`，默认空。空间条件表达式，不填或填 `all` = 所有 target 原子都命中。支持变量 `x`/`y`/`z`（笛卡尔坐标，单位 A），比较运算符 `<`/`>`/`<=`/`>=`/`==`/`!=`，逻辑 `and`/`or`/`not`，以及四则运算。

典型例子：
- `z>=8 and z<=10`：替换 z 在 8-10A 范围内的 target 原子
- `x>5 or y<2`：替换特定 xy 象限内的原子
- `z<=4`：只替换底层

### Mode（mode）

`int`，默认 0。`0` = Random 模式（按比例概率随机采样），`1` = Exact 模式（尽量精确匹配比例计数）。

### Seed（seed）

`int`，默认 0。非 0 时固定随机路径，控制多元素替换的分配随机性。`0` 表示每次运行随机。

## 推荐预设

### 表面单元素钝化（表面 O→F，窄窗口）
```json
{
  "class": "ConditionalReplaceCard",
  "check_state": true,
  "target": "O",
  "replacements": "F:1.0",
  "condition": "z>=8 and z<=10",
  "seed": [101],
  "mode": 1
}
```

### 表面多元素钝化（表面 O→F/N，宽窗口）
```json
{
  "class": "ConditionalReplaceCard",
  "check_state": true,
  "target": "O",
  "replacements": "F:0.7,N:0.3",
  "condition": "z>=6 and z<=14",
  "seed": [101],
  "mode": 0
}
```

### 全区域多元素替换（所有 O→F/N/Cl，用于探索区）
```json
{
  "class": "ConditionalReplaceCard",
  "check_state": true,
  "target": "O",
  "replacements": "F:0.4,N:0.3,Cl:0.3",
  "condition": "all",
  "seed": [101],
  "mode": 0
}
```

## 推荐组合

- `Group Label` → `Conditional Replace`：先打 group 标签分割区域，再用坐标条件做精细筛选。
- `Conditional Replace` → `Atomic Perturb`：替换后加坐标噪声，松驰替换引入的局部应力。
- `Conditional Replace` → `Random Doping`：先做区域选择性替换覆盖表面化学，再做全局替位补样。

## 常见问题

**输出 = 输入，没有替换。** `target` 为空，或输入结构中不存在该元素，或 `condition` 把所有候选位点都过滤掉了。先用 `condition=all` 验证替换逻辑，再逐步收紧条件。

**替换了不该替换的位点。** condition 表达式区域范围写错了。检查结构坐标范围，确认上下限对应目标区域。Slab 的真空层可能导致 z 坐标超出预期。

**condition 解析失败。** 检查语法：变量只能用 x/y/z，运算符拼写正确（`and` 非 `AND`，`==` 非 `=`）。

**替换后键长不合理。** 纯化学替换不做结构弛豫。如果键长明显异常，建议后接弛豫计算。

## 输出标签

`Repl(O->F)` / `Repl(O->F,N,Cl)` 格式。仅在确实发生了替换时才写入标签。

## 可复现性

固定非零 `seed` → 相同输入可复现。多元素替换模式下 seed 控制哪个位点分配到哪个替换元素。`seed=0` 表示每次运行随机。
