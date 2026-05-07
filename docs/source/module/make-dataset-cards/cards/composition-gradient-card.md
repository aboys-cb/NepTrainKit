<!-- card-schema: {"card_name": "Composition Gradient", "source_file": "src/NepTrainKit/ui/views/_card/composition_gradient_card.py", "serialized_keys": ["params"]} -->

# 成分梯度（Composition Gradient）

`Group`: `Alloy` | `Class`: `CompositionGradientCard`

## 功能说明

`Composition Gradient` 沿 x/y/z 方向把结构分成若干坐标层，并在每层按起点成分到终点成分的线性插值替换元素。它用于扩散偶、梯度合金和界面过渡层，不等价于全局随机合金。

## 操作示例

### 场景：模型缺少扩散偶过渡层

Ni-Co 训练集包含纯 Ni、纯 Co 和均匀随机合金，但没有从 Ni-rich 到 Co-rich 的空间过渡结构。模型在界面扩散偶上出现系统误差。

**输入：** 沿 x 方向足够长的 Ni 或 Ni-Co 超胞。
**目标：** 生成 x 方向从 `Ni:1,Co:0` 到 `Ni:0,Co:1` 的配比梯度。
**参数设置：** `elements=Ni,Co`，`axis=x`，`bins=8`，`samples=3`，开启 `use_seed`。
**输出：** 每个输入结构生成 3 个具有相同层配比、不同层内随机排布的梯度结构。
**怎么验证训练集质量改善：** 按 x 分层统计元素比例，应从 Ni-rich 单调过渡到 Co-rich；重训后扩散偶界面测试误差应下降。

## 参数说明
### Elements（elements）
类型：`str`。默认：`'Ni,Co'`。指定沿空间方向做配比梯度的元素集合。

物理直觉：这里列的是梯度端点中参与插值的元素。二元扩散偶写两个元素；三元过渡层写所有会参与比例变化的元素。

### Start Composition（start_composition）
类型：`str`。默认：`'Ni:1,Co:0'`。指定梯度起点的目标元素比例。

物理直觉：起点组成对应 axis 低坐标端。比例不要求归一化，但各元素相对比例决定第一层目标配比。

### End Composition（end_composition）
类型：`str`。默认：`'Ni:0,Co:1'`。指定梯度终点的目标元素比例。

物理直觉：终点组成对应 axis 高坐标端。扩散偶通常把一个元素从 1 过渡到 0，另一个从 0 过渡到 1。

### Axis（axis）
类型：`str`。默认：`'x'`。选择配比梯度沿 `x/y/z` 哪个空间轴展开。

物理直觉：选择真实扩散或界面法向方向。cell 在这个方向上必须有足够层数，否则每层原子太少，整数取整会让目标比例失真。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

### Bins（bins）
类型：`int`。默认：`8`。设置沿梯度轴划分的组成层数。

物理直觉：bins 越多，梯度越细，但每层原子越少、组成取整误差越大。小超胞用 4-8 层，大超胞或界面模型可用 8-20 层。

### Target Elements（target_elements）
类型：`str`。默认：`''`。指定梯度替换时允许被替换的目标元素。

物理直觉：留空时所有匹配元素都可被重排；只想替换某个子晶格、层或初始元素集合时显式列出，避免把基底或不参与扩散的元素也改掉。

生效条件：只希望对部分元素做组成梯度替换时。

### Samples（samples）
类型：`int`。默认：`1`。设置每个输入结构生成的样本数量。

物理直觉：同一层目标比例下，samples 只改变层内随机排布。1-3 个适合扩散偶，5+ 才适合做层内无序统计。

### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现训练集、测试或对比实验时打开；最终大规模随机探索可以关闭，但结果不能逐帧复现。

### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一输入、同一参数和同一 seed 应生成同一批候选；只有 `use_seed=True` 时改变它才会改变随机输出。

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
    "axis": "x",
    "bins": 8,
    "target_elements": "",
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
    "axis": "z",
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

- `Super Cell -> Composition Gradient -> Geometry Filter`：先给梯度方向足够层数，再做配比梯度和几何检查。
- `Crystal Prototype Builder -> Super Cell -> Composition Gradient -> Atomic Perturb`：先生成晶体模板，再加空间成分梯度和局部位移。
- `Composition Gradient -> FPS Filter`：生成多组梯度结构后选代表帧。

## 常见问题

**层内成分不等于目标小数。** 原子数是整数，每层按最接近的整数计数分配。增大超胞或减少 `bins` 可以提高每层成分分辨率。

**运行报错或没有输出。** `elements` 少于 2 个、起止成分解析失败，或 `target_elements` 没有匹配到原子。

**周期结构的起点在哪里。** 周期方向使用 wrapped fractional coordinate 排序；非周期方向使用 Cartesian coordinate 排序。

## 输出标签

`CompGrad(ax={x|y|z},b={bins},s={seed})`。`s` 只在 `use_seed=True` 时出现。

## 可复现性

开启 `use_seed` 后，层内元素排布由 `seed`、输入结构标识和 sample 序号共同决定。
