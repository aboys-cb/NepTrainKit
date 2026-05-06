<!-- card-schema: {"card_name": "Layer Copy", "source_file": "src/NepTrainKit/ui/views/_card/layer_copy_card.py", "serialized_keys": ["params"]} -->

# 层复制（Layer Copy）

`Group`: `Structure` | `Class`: `LayerCopyCard`

## 功能说明

对选定原子的 z 坐标施加表达式调制 `dz = f(x, y, z)`，然后将变形后的层按指定层间距和层数沿 z 方向堆叠复制。可选扩胞 z 向盒子和增加真空。

## 操作示例

### 场景：模型在多层异质结的层间耦合上预测失败

你在单层 MoS2 上训练了一个 NEP 模型，单层性质很好。然后尝试跑双层 MoS2——层间距离预测偏了 0.5 A，层间滑移能垒完全不对。

**诊断思路：** 单层 MoS2 里所有原子都是表面原子（双面都有真空），但双层结构中层间区域的 S 原子面对的是另一层 S，局域环境完全不同。训练集里只有单层数据，模型没见过"上下都有近邻原子"的层间环境。需要从单层出发，复制出多层堆叠结构。

**输入：** 一个弛豫好的单层 MoS2 slab

**目标：** 用 sin 调制给层加起伏，然后复制 3 层，层间距 6.5 A

**参数设置：**
- `Preset Index` = `1`（自定义 dz 表达式）
- `Dz Expr` = `sin(x/pi) + sin(y/pi)`
- `Layers` = `[3]`
- `Distance` = `[6.5]`
- `Extend Cell Z` = 勾选

**输出：** 3 层 MoS2，每层原子 z 位置有起伏调制，盒子高度自动扩展，带 `SWC(L=3,dz=6.5)` 标签

**怎么验证训练集质量改善：**
- 重训后计算双层 MoS2 的层间距和层间滑移能垒，应接近 DFT 参考
- 检查层间是否重叠：最低层的顶部原子和上一层的底部原子距离应 > 2.5 A
- 如果模型只对平整层好而对起伏层差，增大 `dz_expr` 的振幅或者换更复杂的位移函数
- 如果需要构建多层异质结（如 MoS2/WS2），先用两张独立的 Layer Copy 卡分别处理，再拼接

### 什么时候加这张卡、什么时候不加

**加：**
- 研究二维材料、层状材料（MoS2、石墨、BN）的多层性质
- 需要从单层/少层出发构造多层堆叠
- 需要给平整层加波纹、起伏等层间调制

**不加：**
- 体系不是层状结构（体相晶体、分子团簇）
- 只需要简单扩胞 → 用 `Super Cell`

## 参数说明

### `Preset Index`（preset_index）

预设索引，选择内置的变换模板。`1` 为自定义模式，使用 `Dz Expr` 中的表达式。

### `Dz Expr`（dz_expr）

z 方向位移表达式，支持三角函数、指数等数学函数。可用变量：`x`, `y`, `z`, `pi`, `e`。

- `sin(x/pi) + sin(y/pi)`：双向波纹
- `0.5 * sin(x/3)`：单向正弦起伏
- `0`：纯堆叠不调制

通过 `Params`（expression_params）字段定义额外标量参数，格式 `name=value, name=value`。

### `Apply Mode`（apply_mode）

| 值 | 模式 | 行为 |
|----|------|------|
| `0` | 全局 | 所有原子都参与调制 |
| `1` | 按元素 | 仅 `Elements` 列出的元素参与 |
| `2` | 按 z 范围 | 仅 `Z Range` 内的原子参与 |

### `Elements`（elements）

仅 `apply_mode=1` 时生效。逗号分隔的元素列表，如 `S,Mo`。

### `Z Range`（z_range）

仅 `apply_mode=2` 时生效。`[最小值, 最大值]`。

### `Layers`（layers）

复制层数。`2` = 双层，`3` = 三层堆叠。

### `Distance`（distance）

层间距离，单位 A。应与层状材料的范德华间距匹配（如 MoS2 ~ 6.5 A，石墨 ~ 3.35 A）。

### `Extend Cell Z`（extend_cell_z）

勾选 → 自动扩展 z 向盒子高度以容纳新增层。建议保持开启。

### `Wrap`（wrap）

勾选 → 坐标回卷到胞内。默认关闭。

### `Extra Vacuum`（extra_vacuum）

在自动扩展盒子高度的基础上额外增加的真空长度。0~5 A 通常足够。

## 推荐预设

### 双层（平整堆叠，2 层，间距 6.5 A）
```json
{
  "class": "LayerCopyCard",
  "check_state": true,
  "preset_index": 1,
  "dz_expr": "0",
  "params": "",
  "apply_mode": 0,
  "elements": "",
  "z_range": [-5, 5],
  "wrap": false,
  "extend_cell_z": true,
  "extra_vacuum": [0.0],
  "layers": [2],
  "distance": [6.5]
}
```

### 三层起伏（sin 调制，3 层，间距 3.5 A）
```json
{
  "class": "LayerCopyCard",
  "check_state": true,
  "preset_index": 1,
  "dz_expr": "sin(x/pi) + sin(y/pi)",
  "params": "",
  "apply_mode": 0,
  "elements": "",
  "z_range": [-5, 5],
  "wrap": false,
  "extend_cell_z": true,
  "extra_vacuum": [5.0],
  "layers": [3],
  "distance": [3.5]
}
```

### 多层大起伏（cos 调制，6 层，间距 3.0 A）
```json
{
  "class": "LayerCopyCard",
  "check_state": true,
  "preset_index": 1,
  "dz_expr": "2 * cos(x/2) + sin(y/2)",
  "params": "",
  "apply_mode": 0,
  "elements": "",
  "z_range": [-10, 10],
  "wrap": false,
  "extend_cell_z": true,
  "extra_vacuum": [10.0],
  "layers": [6],
  "distance": [3.0]
}
```

## 推荐组合

- `Layer Copy` → `Insert Defect`：先构建多层结构，再采样层间插隙位点
- `Layer Copy` → `Atomic Perturb`：层堆叠 → 加局域热噪声
- `Super Cell` → `Layer Copy`：先扩胞增大面内尺寸，再沿 z 堆叠

## 常见问题

**输出只有一个结构（没有复制）。** `layers` = 1 时只产生单层。确认 `layers` >= 2。

**层间重叠。** `distance` 太小。检查输入层的 z 方向厚度，distance 应大于该厚度。

**表达式报错。** 检查 `dz_expr` 语法，支持的函数列表见参数说明。先用 `0` 测试流水线是否打通。

**Apply Mode 过滤后没有原子参与。** 检查 `elements` 拼写是否和结构中的元素名一致，或者 `z_range` 是否覆盖了原子所在的 z 区间。

## 输出标签

`SWC(L={层数},dz={层间距})`

## 可复现性

无随机性。同参数同输入 → 严格一致输出。
