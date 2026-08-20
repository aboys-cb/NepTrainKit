<!-- card-schema: {"card_name": "Interface Layer Mixing", "source_file": "src/NepTrainKit/ui/views/_card/interface_layer_mix_card.py", "serialized_keys": ["params"]} -->

# 界面随机互混（Interface Layer Mixing）

`Group`: `Alloy` | `Class`: `InterfaceLayerMixCard`

## 功能说明

`Interface Layer Mixing` 针对**双层 / 界面结构**：先自动识别界面所在（界面法向轴 + 界面位置），把结构分成界面下方（L）与界面上方（R）两侧，再按你指定的两侧原子层数，把靠近界面的薄层选中，在选中的 L/R 薄层之间随机配对并**交换原子元素类型**——格点位置、晶胞都不变，只有哪些位点放哪种元素变了。

浓度表示"发生了交换的原子数占选中薄层原子总数的比例"，例如 0.5 意味着选中层里一半的原子换成了对侧的元素。你可以固定目标浓度生成多帧，也可以用**浓度梯度**让 `num_structures` 帧从初始浓度线性过渡到终止浓度。

它和 `Composition Gradient` 的区别：`Composition Gradient` 沿某个方向对**整段结构**做配比渐变；这张卡只动**界面两侧的少数原子层**，模拟互扩散初期的界面薄层互混，其余原子层保持原样。

## 操作示例

### 场景：模型在界面互扩散构型上失效

Ni/Al 界面模型在纯 Ni、纯 Al 和均匀合金上误差正常，但给出一个两侧各带 2-3 层互混原子的界面构型时，界面附近原子力预测明显偏差。训练集里从来没有"界面两侧元素互换"的中间态，模型只能靠外推。

**输入：** 一个已弛豫的 Ni(上)/Al(下) 双层结构，界面沿晶格 `c` 方向，两侧各有 6 层以上原子。
**目标：** 生成界面两侧各 2 层的互混构型，让模型见过界面处 Fe/Al 元素互换的局域环境。
**参数设置：** `axis=auto`，`left_layers=2`，`right_layers=2`，`mode=fixed`，`concentration=0.5`，`num_structures=20`，开启 `use_seed`。
**输出：** 20 个结构，每个都有约一半选中层原子被换到对侧；界面仍清晰，只是薄层变混。
**怎么验证训练集质量改善：** 重训后看界面附近原子力 MAE 是否下降；如果互混浓度不够丰富，把 `concentration` 或 `num_structures` 调大，或改用 `gradient` 模式覆盖 0 → c_max 的完整区间。

### 场景：补一条浓度-能量曲线

你想研究界面互混程度对总能量的影响。固定浓度只能给出一个点，用梯度模式一次跑出从纯界面到完全互混的一串结构。

**输入：** 同上，界面可自动检测的双层结构。
**目标：** 从 `c=0`（未互混）到 `c=c_max`（可交换容量上限，完全互混）逐帧覆盖。
**参数设置：** `mode=gradient`，`gradient_start=0`，`gradient_end=1`，`num_structures=11`，`use_seed=True`。
**输出：** 11 个结构，浓度 0 → 1 线性分布，对应可交换原子对数逐步增加。
**怎么验证：** 导出后按 `c` 值统计交换原子数，应呈线性；把各帧能量画出，应得到一条平滑的互混能曲线。

## 参数说明

### Axis（axis）

`str`，默认 `auto`。界面法向所在的晶格轴。

选 `auto` 时，程序对 `a`/`b`/`c` 三个方向各算一次"切分为上下两半后成分直方图差异"，取差异最大的轴作为界面法向。对成分在界面两侧明显不同的双层，`auto` 通常一次就能找对。

如果你的体系两侧成分接近（例如 Cu/Ni 固溶体两边差不多），`auto` 可能分不清哪一侧是 L 哪一侧是 R，此时手动选轴更可靠。`a`/`b`/`c` 表示晶格分数坐标方向，不是笛卡尔 X/Y/Z。

### Auto-locate interface（auto_position）

`bool`，默认 true。是否自动定位界面在轴上的位置。

卡片固定为 true，界面位置始终自动检测：在法向轴上找"成分变化最剧烈"的间隙，取间隙中点作为界面分数坐标，再按它分 L/R。卡片不再提供关闭选项；该字段保留在参数与序列化 JSON 中，仅用于兼容，运行结构时始终走自动定位。

### Interface Position（interface_position）

`float`，默认 0.5，范围 [0, 1]。界面在法向上的分数坐标。

卡片固定为 0.5，实际界面位置由自动定位决定并覆盖该值，不提供手动输入。字段保留在参数与序列化 JSON 中，仅用于兼容。小于界面坐标的原子归 L 侧，大于等于的归 R 侧。

### L侧层数（left_layers）

`int`，默认 2，范围 1-100。从界面往下数、参与互混的 L 侧原子层数。

第 1 层就是紧贴界面的一层。程序按分数坐标四舍五入到 0.01 来区分原子层（与层错卡片的分层约定一致），对轻微弛豫的面内噪声是稳定的。层数越多，选中原子越多、`c_max` 越大，但离界面远的层也进入互混，物理上更接近"厚互混层"。

### R侧层数（right_layers）

`int`，默认 2，范围 1-100。从界面往上数、参与互混的 R 侧原子层数。语义与 `left_layers` 对称。

当 L/R 两侧层数不同时，`c_max` 由原子数较少的一侧决定（见下文 `c_max` 说明），多余的那侧原子不会被选中参与交换。

### 模式（mode）

`str`，默认 `fixed`，可选 `fixed` 或 `gradient`。控制浓度怎么在输出结构间取值。卡片中用「固定浓度」/「梯度浓度」两个互斥勾选框切换，对应 `mode=fixed` / `mode=gradient`；选中「梯度浓度」时下方显示一行 `初始 ~ 终止` 两个百分比输入（百分比符号只在终止框后出现一次）。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| `fixed` | 所有输出结构都用同一个目标浓度 | 补单一互混程度的多帧随机排布 |
| `gradient` | 从初始浓度到终止浓度线性插值 | 扫一条浓度-能量曲线或完整互混区间 |

### Concentration（concentration）

`float`，默认 0.5，范围 [0, c_max]，`fixed` 模式生效。目标浓度 = 交换原子数 / 选中薄层总原子数。UI 中按百分比输入（0–100%，默认 50%），内部换算为 0–1 分数。

它不是"选中的 L 原子里有多少换走"，而是两侧合计的交换占比：每交换一对（1 个 L ↔ 1 个 R）计 2 个原子。给定选中层原子数 n_L、n_R，可交换对数上限 `k_max = min(n_L, n_R)`，浓度上限 `c_max = 2·k_max / (n_L+n_R)`。n_L=n_R 时 c_max 恰好是 1.0；两侧不均衡时 c_max < 1，超出会直接报错而不是静默截断。

### Gradient Start（gradient_start）

`float`，默认 0.0，范围 [0, c_max]，`gradient` 模式生效。第一个生成结构的目标浓度。UI 中按百分比输入（默认 0%）。

`num_structures=1` 时，只生成这一个浓度的结构。

### Gradient End（gradient_end）

`float`，默认 1.0，范围 [0, c_max]，`gradient` 模式生效。最后一个生成结构的目标浓度。UI 中按百分比输入（默认 100%）。中间帧按 `start + (end-start)·i/(num-1)` 线性取值，`num_structures=1` 时忽略。

### 结构数（num_structures）

`int`，默认 1，范围 1-10000。每个输入结构生成多少个输出结构。

`fixed` 模式下这些输出浓度相同、随机排布不同；`gradient` 模式下它们按浓度线性分布。程序严格生成恰好这么多结构——参数不合法会在任何随机操作之前报错，不会静默少给或假装成功。

### 使用随机种子（use_seed）

`bool`，默认 false。开启后固定随机种子，同一输入 + 同一参数可复现同一批结构。

### Seed（seed）

`int`，默认 0。随机种子基准值。

生效条件：`use_seed=True`。实际随机源按 `seed + 输入结构标识 × 1000003 + 帧序号` 派生，所以同一输入下，种子相同则逐帧可复现。

## 推荐预设

### 界面两侧各 2 层、半互混多帧

```json
{
  "class": "InterfaceLayerMixCard",
  "params": {
    "axis": "auto",
    "auto_position": true,
    "interface_position": 0.5,
    "left_layers": 2,
    "right_layers": 2,
    "mode": "fixed",
    "concentration": 0.5,
    "gradient_start": 0.0,
    "gradient_end": 1.0,
    "num_structures": 20,
    "use_seed": true,
    "seed": 42
  }
}
```

补界面互混构型的主预设：薄层互混、多帧随机排布。适用于大多数界面体系。

### 0 → 1 浓度扫描

```json
{
  "class": "InterfaceLayerMixCard",
  "params": {
    "axis": "auto",
    "auto_position": true,
    "interface_position": 0.5,
    "left_layers": 3,
    "right_layers": 3,
    "mode": "gradient",
    "concentration": 0.5,
    "gradient_start": 0.0,
    "gradient_end": 1.0,
    "num_structures": 11,
    "use_seed": true,
    "seed": 7
  }
}
```

用于互混能曲线。层数加到 3 扩大 `c_max` 覆盖范围，浓度 0→1 线性扫描。

## 推荐组合

- `Super Cell → Interface Layer Mixing → Geometry Filter`：先把界面方向扩到足够层数，再做互混，最后检查是否有异常短键或异常体积。
- `Crystal Prototype Builder → Super Cell → Interface Layer Mixing → Atomic Perturb`：先构造双层原型，扩胞后互混，再加少量热扰动得到近平衡样本。
- `Interface Layer Mixing → FPS Filter`：批量生成互混构型后，用代表性子集压缩训练集规模。

## 常见问题

**报"Not enough atomic layers"。** 界面一侧（或两侧）的可用层数少于 `left_layers`/`right_layers`。通常是因为自动定位出的界面位置偏一侧、或那一侧层本身太少。减少对应层数。

**报"concentration exceeds swap capacity"。** `fixed` 模式目标浓度或 `gradient` 模式浓度上界超过了该界面的 `c_max`。`c_max = 2·min(n_L,n_R)/(n_L+n_R)`，两侧原子数越接近、层数越多，`c_max` 越接近 1。降低浓度或增加层数，并保证浓度 ≤ `c_max`，超出会直接报错而不是静默截断。

**两侧选中区域是同一种元素。** 例如自动定位的界面正好落在纯 Al 区域内，或结构本身只有一种元素，L/R 选中层全是 Al——交换不会改变任何东西，卡片直接报错而不是浪费时间生成无效结构。检查结构是否真的是异种元素双层。

**`auto` 检测不到界面。** 两侧成分相似（固溶体、成分接近的合金）时，所有方向的成分对比度都接近 0，`auto` 会报"no interface"。手动选一个方向轴。

## 输出标签

`IfaceMix(L={left_layers},R={right_layers},c={concentration})`，开启 `use_seed` 时追加 `,s={seed}`。示例：`IfaceMix(L=2,R=2,c=0.5,s=42)`。

## 可复现性

开启 `use_seed` 后，每一帧的随机交换由 `seed + 输入结构标识 × 1000003 + 帧序号` 派生；同一输入、同一参数、同一种子生成完全相同的结果。关闭 `use_seed` 时结果不可逐帧复现。
