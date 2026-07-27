<!-- card-schema: {"card_name": "Group Label", "source_file": "src/NepTrainKit/ui/views/_card/group_label_card.py", "serialized_keys": ["params"]} -->

# 分组标记（Group Label）

`Group`: `Alloy` | `Class`: `GroupLabelCard`

## 功能说明

按当前晶胞的分数坐标规则，把原子分成两个 group，并写入 `atoms.arrays["group"]`。这些标签可供下游 `Magnetic Order`、`Random Doping` 和 `Random Vacancy` 选择对应原子。

**这是一张元数据卡：不改坐标、不改元素，也不写 `atoms.arrays["sublattice"]`。** `group` 是下游工作流使用的选择标签；`sublattice` 是晶体学位点身份，两者保持独立。

两种规则都相对于**当前晶胞和坐标原点**计算：

$$g_{\mathrm{layer}}=\left\lfloor 2(\mathbf{s}\cdot\mathbf{k})\right\rfloor\bmod 2$$

$$g_{\mathrm{parity}}=\left(\sum_{\alpha=x,y,z}\left\lfloor2s_\alpha+\frac{1}{2}\right\rfloor\right)\bmod 2$$

其中 $\mathbf{s}$ 是包裹到当前晶胞内的分数坐标。第二式采用明确的 half-up 舍入，避免 NumPy 偶数舍入在恰好 `.5` 时造成不对称分组。

## 操作示例

### 场景：训练集缺少沿晶格体对角线交替的 AFM 层

一个 2×2×2 BCC Fe 超胞的训练集主要包含 FM 构型，模型在分层 AFM 构型上的能量误差明显更大。需要先把当前超胞按 `111` 分数坐标相位分为 A/B 两组，再让 `Magnetic Order` 按 group A/B 赋予相反磁矩。

**输入：** 2×2×2 BCC Fe 周期超胞。

**参数设置：**

- `Grouping rule` = `k_vector`
- `Layer vector` = `111`
- `Group A label` = `A`
- `Group B label` = `B`
- `Overwrite existing group labels` = 关闭（输入没有 group 时仍会正常生成）

**运行前检查：** 卡片会以首个输入结构显示 `A=8 · B=8`。如果预览显示某一组为 0，应先扩胞或换一个 layer vector，而不是继续生成只有一种标签的伪 A/B 数据。

**输出：** 结构的坐标和元素不变，新增 A/B group 标签并追加 `Grp(k111,A/B)`。

**怎么验证训练集补充有效：**

- 检查输出 EXTXYZ 的 `group` 列确实同时包含 A/B。
- 接入 `Magnetic Order` 的 group A/B 模式，确认 Fe 磁矩同时存在正负号。
- 用新增 AFM 构型重训后，单独比较分层 AFM 切片的能量和磁力误差。

### 能力边界

- 本卡只生成两类**坐标规则分组**，不是自动识别化学子晶格、表面/体相或任意区域。
- `fractional_parity` 是当前晶胞的 half-grid 模式，不是通用 NaCl 子晶格识别器。
- 改变超胞、重新选择晶胞或平移坐标原点，可能改变分组结果；运行前应查看计数预览并检查输出。
- 如果结构已有可信的 `group`，保持 overwrite 关闭即可原样保留。

## 参数说明

### 分组规则（mode）

`str`，默认 `k_vector`。

| 模式 | 算法 | 适用场景 |
|------|------|---------|
| `k_vector` | $\lfloor2(\mathbf{s}\cdot\mathbf{k})\rfloor\bmod2$ | 沿所选分数坐标相位生成两组层 |
| `fractional_parity` | 对 $2\mathbf{s}$ 做 half-up 舍入后取坐标和奇偶 | 在当前晶胞的 half-grid 上生成棋盘式两组 |

旧项目中的 `k-vector layers (recommended)`、`k-vector layers`、`fractional parity (2x rounding)` 会继续按对应新模式读取；未知值会明确报错，不再静默回退。

### 层向量（kvec）

`str`，默认 `111`，仅 `k_vector` 模式使用。可选：

- `100`：沿晶格 a 的分数坐标相位。
- `010`：沿晶格 b。
- `001`：沿晶格 c。
- `110`：沿 a+b。
- `111`：沿 a+b+c。

这些数字描述当前晶胞的分数坐标相位，不等同于程序自动识别出的晶面族。

### A 组标签（group_a）

`str`，默认 `A`。写入偶相位原子的标签。必须非空，并与 `group_b` 不同；下游引用时区分大小写。

### B 组标签（group_b）

`str`，默认 `B`。写入奇相位原子的标签。必须非空，并与 `group_a` 不同。

### 覆盖已有分组（overwrite）

`bool`，默认 false。关闭时：

- 输入没有 `group`：按当前规则正常生成。
- 输入已有 `group`：原样保留现有标签，其他参数不再重写它。

开启时会覆盖已有的 `atoms.arrays["group"]`。卡片预览会明确显示“保留”或“覆盖”状态。

## 推荐预设

### 111 分数坐标层，标签 A/B

```json
{
  "class": "GroupLabelCard",
  "check_state": true,
  "params": {
    "mode": "k_vector",
    "kvec": "111",
    "group_a": "A",
    "group_b": "B",
    "overwrite": false
  }
}
```

### 110 分数坐标层，自定义标签

```json
{
  "class": "GroupLabelCard",
  "check_state": true,
  "params": {
    "mode": "k_vector",
    "kvec": "110",
    "group_a": "up",
    "group_b": "down",
    "overwrite": true
  }
}
```

### 当前晶胞 half-grid parity

```json
{
  "class": "GroupLabelCard",
  "check_state": true,
  "params": {
    "mode": "fractional_parity",
    "kvec": "111",
    "group_a": "A",
    "group_b": "B",
    "overwrite": false
  }
}
```

## 推荐组合

- `Super Cell` → `Group Label` → `Magnetic Order`：先保证当前晶胞有足够相位层，再按 group A/B 生成 AFM。
- `Group Label` → `Random Doping` / `Random Vacancy`：只在由本卡坐标规则生成的某一组中替位或删位。

## 常见问题

**预览只有 A 或只有 B。** 当前晶胞中的所有原子落在同一相位。先扩胞，或更换 layer vector / grouping rule。不要把单标签结果当成有效的两组分区。

**平移坐标或扩胞后分组变化。** 这是当前算法的定义：两种模式都锚定在当前晶胞的分数坐标和原点。需要跨不同晶胞保持固定晶体学身份时，应在结构生成阶段维护 `sublattice`，不要把本卡当作 sublattice 识别器。

**输入已有 group，但参数似乎没有生效。** overwrite 默认关闭，此时卡片会保留现有 group；预览会显示已有标签计数。确认确实需要覆盖后再开启。

**下游读不到 group。** 确认输出使用 EXTXYZ 等能保存 per-atom arrays 的格式，并确保下游填写的标签与 `group_a` / `group_b` 完全一致。

## 输出标签

新生成或覆盖分组时，`Config_type` 追加：

- `Grp(k111,A/B)`：k-vector 111。
- `Grp(par,A/B)`：fractional parity。

overwrite 关闭且输入已有 group 时，输出是原结构的副本，不追加新的 `Grp(...)` 标签。

## 可复现性

无随机性。同一晶胞、坐标原点、原子坐标和参数会得到严格一致的 group 标签。
