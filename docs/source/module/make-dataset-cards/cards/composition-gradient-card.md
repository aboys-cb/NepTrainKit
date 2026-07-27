<!-- card-schema: {"card_name": "Composition Gradient", "source_file": "src/NepTrainKit/ui/views/_card/composition_gradient_card.py", "serialized_keys": ["params"]} -->

# 成分梯度（Composition Gradient）

`Group`: `Alloy` | `Class`: `CompositionGradientCard`

## 功能说明

`Composition Gradient` 沿晶格 `a`、`b` 或 `c` 方向排列可替换原子，再把它们分成原子数尽量相同的若干层。第一层使用起始成分，最后一层使用结束成分，中间层做线性插值。卡片只重新分配元素，不移动原子，也不改变晶胞。

它适合扩散偶、梯度合金和界面过渡层。如果你需要的是多个空间均匀、但全局成分不同的合金结构，应使用 `Composition Space Sampling → Random Occupancy`，而不是这张卡。

## 操作示例

### 场景：模型缺少扩散偶过渡层

Ni-Co 训练集包含纯 Ni、纯 Co 和均匀随机合金，但没有从 Ni-rich 到 Co-rich 的空间过渡结构。模型在界面扩散偶上出现系统误差。

**输入：** 沿晶格 `a` 方向足够长的 Ni 或 Ni-Co 超胞。
**目标：** 生成晶格 `a` 方向从 `Ni:1,Co:0` 到 `Ni:0,Co:1` 的配比梯度。
**参数设置：** `elements=Ni,Co`，`axis=a`，`bins=8`，`target_elements=Ni,Co`，`samples=3`，开启 `use_seed`。
**输出：** 每个输入结构生成 3 个具有相同层配比、不同层内随机排布的梯度结构。
**怎么验证训练集质量改善：** 按晶格 `a` 方向分层统计元素比例，应从 Ni-rich 单调过渡到 Co-rich；重训后扩散偶界面测试误差应下降。

## 参数说明

### Elements（elements）

`str`，默认 `Ni,Co`。梯度端点中参与插值的元素集合。二元扩散偶写两个元素即可，三元过渡层把你关心的所有成分变化元素都列进去。

### Start Composition（start_composition）

`str`，默认 `Ni:1,Co:0`。梯度起点的目标元素比例，对应 axis 低坐标端。比例不要求归一化，各元素相对比值决定第一层的目标配比。

### End Composition（end_composition）

`str`，默认 `Ni:0,Co:1`。梯度终点的目标元素比例，对应 axis 高坐标端。扩散偶通常把一个元素从 1 过渡到 0、另一个从 0 过渡到 1。

### Axis（axis）

`str`，默认 `a`，可选 `a`、`b` 或 `c`。它们表示晶格坐标方向，不是笛卡尔 X/Y/Z。程序按对应的分数坐标从低到高排列原子：起始成分放在低坐标端，结束成分放在高坐标端。

选真实扩散方向或界面法向。旧配置中的 `x/y/z` 仍可加载，并分别按 `a/b/c` 处理。

如果这个方向是周期的，晶胞高坐标端会和低坐标端相接。起止成分不同时，周期边界处会出现一次额外的成分突变；做周期扩散偶时要确认这正是你想要的边界条件。

### Bins（bins）

`int`，默认 8。程序按位置排序后，把候选原子分成原子数尽量相同的若干组；它不是按等距离切出若干个等厚空间层。

`bins` 越大，成分变化看起来越细，但每层原子越少、整数取整误差越大。某层有 `N` 个候选原子时，单个元素能够表达的最小成分步长约为 `1/N`。例如每层只有 4 个原子，只能稳定表达 0%、25%、50%、75% 和 100%。如果输入只有 32 个候选原子，设 8 层已经意味着每层约 4 个原子。

### Target Elements（target_elements）

`str`，默认空。这里填写输入结构中允许被替换的现有元素，不是输出元素。留空会让所有原子都参与重新分配；例如输入含 Ni、Co、Al，而 `elements=Ni,Co` 时，Al 也会被改成 Ni 或 Co。

多组元体系通常应显式填写，例如 `Ni,Co`，以保留 Al 子晶格。当前参数只能按元素筛选，不能按空间区域或 `group` 标签筛选。

### Samples（samples）

`int`，默认 1。每个输入结构生成几个随机排布。同一层的元素数量不变，改变的只是这些元素落在哪些候选位点上。1-3 个适合补少量扩散偶构型；如果要研究同一成分梯度下的局域无序差异，可以增加输出数量。

### Use Seed（use_seed）

`bool`，默认 false。需要可复现训练集或对比实验时打开；最终大规模随机探索可以关掉，但结果不能逐帧复现。

### Seed（seed）

`int`，默认 0。同一输入、同一参数、同一 seed 生成同一批候选。

生效条件：`use_seed=True`。

## 推荐预设

### 二元扩散偶

```json
{
  "class": "CompositionGradientCard",
  "params": {
    "elements": "Ni,Co",
    "start_composition": "Ni:1,Co:0",
    "end_composition": "Ni:0,Co:1",
    "axis": "a",
    "bins": 8,
    "target_elements": "Ni,Co",
    "samples": 3,
    "use_seed": true,
    "seed": 42
  }
}
```

用于 Ni-rich 到 Co-rich 的一维过渡层。

### 三元梯度层

```json
{
  "class": "CompositionGradientCard",
  "params": {
    "elements": "Co,Cr,Ni",
    "start_composition": "Co:0.8,Cr:0.1,Ni:0.1",
    "end_composition": "Co:0.1,Cr:0.4,Ni:0.5",
    "axis": "c",
    "bins": 10,
    "target_elements": "Co,Cr,Ni",
    "samples": 2,
    "use_seed": true,
    "seed": 9
  }
}
```

用于多元合金的层状成分过渡。

## 推荐组合

- `Super Cell → Composition Gradient → Geometry Filter`：先给梯度方向足够层数，再做配比梯度和几何检查。
- `Crystal Prototype Builder → Super Cell → Composition Gradient → Atomic Perturb`：先生成晶体模板，再加空间成分梯度和局部位移。
- `Composition Gradient → FPS Filter`：生成多组梯度结构后选代表帧。

## 常见问题

**层内成分不等于目标小数。** 原子数是整数，每层按最接近的整数计数分配。增大超胞或减少 `bins` 可以提高每层成分分辨率。

**运行报错或没有输出。** `elements` 少于 2 个、起止成分解析失败，或 `target_elements` 没有匹配到原子。

**倾斜晶胞中的方向怎么看。** `a/b/c` 始终表示晶格分数坐标方向。即使晶格向量倾斜，也不会切换成笛卡尔 X/Y/Z。

**为什么周期边界处还有一个突变界面。** 周期方向的高坐标端会与低坐标端相接，而线性梯度的结束成分通常不同于起始成分。需要连续周期成分场时，这张卡当前不适用。

## 输出标签

`CompGrad(ax={a|b|c},b={bins},s={seed})`。`s` 只在 `use_seed=True` 时出现。

## 可复现性

开启 `use_seed` 后，层内元素排布由 `seed`、输入结构标识和 sample 序号共同决定。
