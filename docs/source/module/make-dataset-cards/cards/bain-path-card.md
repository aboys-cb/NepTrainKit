<!-- card-schema: {"card_name": "Bain Path", "source_file": "src/NepTrainKit/ui/views/_card/bain_path_card.py", "serialized_keys": ["params"]} -->

# Bain 路径（Bain Path）

`Group`: `Lattice` | `Class`: `BainPathCard`

## 功能说明

沿一个指定晶轴做四方畸变，生成一组不同 `c/a` 比的晶胞。它保留原子数和元素组成，只改变晶格矢量，可用于给 bcc/fcc/fct 转变、四方相变路径、近似马氏体变形等场景补训练结构。

这张卡不是随机扰动卡。它扫描的是晶胞形状自由度：选定 `x`、`y` 或 `z` 作为 c 轴，按 `ca_range` 改变该方向长度；`constant_volume` 会同步缩放另外两条轴让体积保持不变，`scale_volume` 会在此基础上再扫体积，`free_c` 只改选中轴。

## 操作示例

### 场景：模型在 bcc 到 fct 变形路径上能量曲线不平滑

你的 NEP 模型在 bcc Fe 的平衡结构附近力误差很低，但沿 Bain 路径拉伸时能量曲线出现不合理的尖峰。训练集中有平衡 bcc 和少量随机应变，却缺少系统的四方畸变结构，模型只能把这条相变路径当外推。

**输入：** 一个弛豫后的 bcc 或近 bcc 超胞。

**目标：** 在保持体积近似不变的前提下扫描 `c/a = 0.8, 1.0, 1.2`，让模型见过从压扁到拉长的晶胞形状。

**参数设置：**
- `axis = "z"`
- `ca_range = [0.8, 1.2, 0.2]`
- `mode = "constant_volume"`
- `scale_atoms = true`

**输出：** 3 个结构，带 `Bain(ax=z,ca=...,V=...,mode=constant_volume)` 标签。

**怎么验证训练集质量改善：** 重训后重新扫同一 Bain 路径，能量曲线应更平滑，峰值位置不应随采样点轻微移动而大幅跳变。如果仍然不稳，缩小步长到 0.05 或在 `scale_volume` 模式下给每个 `c/a` 加入 0.95、1.00、1.05 三个体积点。

## 参数说明

### Axis（axis）

`str`，默认 `"z"`。指定哪条晶格矢量当作 Bain 路径里的 c 轴。

选 `z` 最常见，因为很多输入结构已经把相变或薄膜方向放在第三条晶格矢量。若你的候选畸变方向实际沿 x 或 y，直接切换即可；不要先手动换轴再跑卡片，那样更容易把后续标签和结构方向搞混。

| 选项 | 含义 | 什么时候选 |
|------|------|-------------|
| `x` | 缩放第一条晶格矢量 | 变形方向沿 a 轴 |
| `y` | 缩放第二条晶格矢量 | 变形方向沿 b 轴 |
| `z` | 缩放第三条晶格矢量 | 变形方向沿 c 轴，默认选择 |

### CA Range（ca_range）

`tuple[float, float, float]`，默认 `(1.0, 1.0, 1.0)`。格式是 `[起点, 终点, 步长]`，扫描选中轴的相对缩放因子。

`1.0` 表示原始晶胞；`0.9` 是选中轴压短 10%；`1.1` 是拉长 10%。如果只是补近平衡响应，`0.95-1.05` 通常够用；研究 bcc/fcc/fct 路径时可以扩到 `0.7-1.4`，但每批输出都要检查最近邻距离和体积是否仍在你愿意做 DFT 的范围内。

### Mode（mode）

`str`，默认 `"constant_volume"`。决定选中轴变化时，另外两条轴和总体积怎么处理。

| 选项 | 含义 | 什么时候选 |
|------|------|-------------|
| `constant_volume` | 选中轴乘以 `r`，另外两轴乘以 `1/sqrt(r)` | 想隔离形状效应，不想把体积变化混进去 |
| `scale_volume` | 先做 constant-volume Bain，再按 `volume_scale_range` 统一缩放体积 | 同时补四方畸变和体积响应 |
| `free_c` | 只缩放选中轴，另外两轴不动 | 模拟外延约束或单轴加载 |

### Volume Scale Range（volume_scale_range）

`tuple[float, float, float]`，默认 `(1.0, 1.0, 1.0)`。格式是 `[起点, 终点, 步长]`，表示输出体积相对原始体积的比例。

这个参数只在 `mode = "scale_volume"` 时生效。比如 `[0.95, 1.05, 0.05]` 会为每个 `c/a` 生成 0.95、1.00、1.05 三个体积点；如果 `ca_range` 有 5 个点，总输出就是 15 个结构。范围别一上来开太大，体积和形状同时扫很容易把训练集放大。

生效条件：`mode` 必须是 `scale_volume`。

### Scale Atoms（scale_atoms）

`bool`，默认 `true`。打开后原子随晶胞一起按分数坐标缩放；关闭后保留 Cartesian 坐标，只改 cell。

训练集构型生成通常应保持打开，因为这代表结构整体经历晶格畸变。关闭更像是在改周期边界而不是改内部结构，适合少数外延盒子或调试场景；关掉后务必检查原子是否贴近边界或产生异常近邻。

## 推荐预设

### 近平衡四方响应

适合先确认模型对小形变是否平滑，输出 3 个结构。

```json
{
  "class": "BainPathCard",
  "check_state": true,
  "params": {
    "axis": "z",
    "ca_range": [0.95, 1.05, 0.05],
    "mode": "constant_volume",
    "volume_scale_range": [1.0, 1.0, 1.0],
    "scale_atoms": true
  }
}
```

### 相变路径粗扫

适合 bcc/fct 路径先粗看能量曲线形状，输出 7 个结构。

```json
{
  "class": "BainPathCard",
  "check_state": true,
  "params": {
    "axis": "z",
    "ca_range": [0.7, 1.3, 0.1],
    "mode": "constant_volume",
    "volume_scale_range": [1.0, 1.0, 1.0],
    "scale_atoms": true
  }
}
```

### 形状加体积联合覆盖

适合怀疑模型同时缺体积响应和四方畸变响应时使用，输出数量是 `ca_range` 点数乘以体积点数。

```json
{
  "class": "BainPathCard",
  "check_state": true,
  "params": {
    "axis": "z",
    "ca_range": [0.9, 1.1, 0.1],
    "mode": "scale_volume",
    "volume_scale_range": [0.95, 1.05, 0.05],
    "scale_atoms": true
  }
}
```

## 推荐组合

- `Bain Path` -> `Atomic Perturb`：先补系统四方畸变，再给每个形变点加入小坐标扰动，覆盖非零温局域环境。
- `Super Cell` -> `Bain Path`：先扩胞再扫路径，避免小胞里单个缺陷或磁序周期过短。
- `Bain Path` -> `Lattice Strain`：先覆盖相变主路径，再补普通轴向应变，用来稳定弹性响应。

## 常见问题

**输出结构数比预期多。** `scale_volume` 会对每个 `c/a` 再乘上每个体积点；总数是两组扫描点数的乘积。

**体积没有保持不变。** 检查 `mode` 是否为 `constant_volume`。`free_c` 会改变体积，`scale_volume` 会按 `volume_scale_range` 继续改体积。

**结构看起来被拉得太夸张。** 缩小 `ca_range`，先用 `0.95-1.05` 验证模型响应，再决定是否覆盖更宽相变路径。

## 输出标签

`Bain(ax={axis},ca={r},V={V/V0},mode={mode})`

## 可复现性

无随机性。同一输入结构和同一参数会生成完全一致的结构列表。
