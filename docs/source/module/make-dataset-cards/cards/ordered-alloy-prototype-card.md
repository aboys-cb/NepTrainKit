<!-- card-schema: {"card_name": "Ordered Alloy Prototype", "source_file": "src/NepTrainKit/ui/views/_card/ordered_alloy_prototype_card.py", "serialized_keys": ["params"]} -->

# 有序合金原型（Ordered Alloy Prototype）

**分类：** 合金与组分

## 功能说明

生成带逐原子 `sublattice` 数组的 A1、A2、A3、L1₂、B2 或 L1₀ 基础晶胞。A/B 表示固定的晶体学位点身份，不占用磁性流程使用的 `group`。

输出是带 A/B 子晶格身份的基础晶胞。需要更大结构时，在后面连接 [扩胞（Super Cell）](super-cell-card.md)。

## 原理与公式

每个原型由晶胞矩阵 $\mathbf C_0$、分数坐标 $\{\mathbf s_j\}$ 和子晶格标签 $\{\lambda_j\}$ 定义：

$$
\mathbf r_j=\mathbf s_j\mathbf C_0,
\qquad
\lambda_j\in\{A,B\}.
$$

基础晶胞的位点数固定：

| 原型 | 位点数 | 子晶格计数 | 形状参数 |
|---|---:|---|---|
| A1 / FCC | 4 | A=4 | 立方 |
| A2 / BCC | 2 | A=2 | 立方 |
| A3 / HCP | 2 | A=2 | $c=a(c/a)$ |
| L1₂ / A₃B | 4 | A=3，B=1 | 立方 |
| B2 / AB | 2 | A=1，B=1 | 立方 |
| L1₀ / AB | 4 | A=2，B=2 | $c=a(c/a)$ |

连接“扩胞”卡后，坐标、元素和 `sublattice` 会一起复制。例如 L1₂ 基胞的 A:B=3:1 经 $2\times2\times2$ 扩胞后得到 24:8。

### A1/A2/A3 与普通晶体原型的区别

它们在几何上分别等同于 FCC、BCC、HCP。仅当后续流程需要统一的 `sublattice=A` 标签或 `X` 占位符时才使用本卡；只需要单元素晶体时，优先使用“晶体原型构建”。

### X 占位符

`X` 表示尚未确定的元素位点，原子序数为 0，不能直接用于训练或 DFT。含 `X` 的结构应先经过“有限晶胞合金占位”，将每个位点替换为真实元素。

## 操作示例

### 生成 L1₂ 的 32 位点占位模板

1. 选择 `L1₂ / A₃B`，晶格常数设为单点；
2. A、B 均保留 `X`，预览应显示基础晶胞 A=3、B=1；
3. 后接“扩胞”，设为 $2\times2\times2$，得到 A=24、B=8；
4. 后接“有限晶胞合金占位”，分别为 A/B 子晶格设置真实元素和整数计数。

若模型在 L1₂ 相的误差偏高，应在 DFT 标注和重训后按原型、子晶格占位程度分别比较能量和力误差，而不是只看混合总 RMSE。

### 直接生成 Cu₃Au

选择 `L1₂ / A₃B`，A 填 `Cu`、B 填 `Au`。输出即为固定 3:1 化学计量的 Cu₃Au 基胞；需要更大结构时，后接“扩胞”即可。

## 参数说明

### 有序原型（prototype）

`str`，默认 `L12/A3B`。可选 `A1/fcc`、`A2/bcc`、`A3/hcp`、`L12/A3B`、`B2/AB`、`L10/AB`，位点定义见上表。

### 晶格常数范围（a_range）

`tuple[float, float, float]`，默认 `(3.6, 3.6, 0.1)`，单位 Å，依次为起点、终点、正步长。反向端点会交换；每个采样点输出一个基础晶胞。

### 晶格轴比（covera）

`float`，默认 `1.0`。仅 A3/HCP 和 L1₀ 显示并生效，定义 $c/a$；立方原型固定为 1。

### 子晶格元素（sublattice_elements）

`str`，默认 `A:X,B:X`。界面将其拆成 A、B 两个独立输入框；单子晶格原型只显示 A。每项接受一个真实元素符号或 `X`。

### 最大输出数（max_outputs）

`int`，默认 `200`，至少为 1。若 `a` 扫描点更多，只输出从较小 $a$ 开始的前几个点；界面会提前显示截断。

## 常见问题

**在哪里设置原子数或重复倍率？** 在本卡后连接“扩胞”卡。扩胞会保留 `sublattice` 数组。

**A1/A2/A3 应该选本卡还是晶体原型卡？** 需要子晶格标签或 `X` 占位时选本卡；只生成普通单元素 FCC/BCC/HCP 时选晶体原型卡。

**旧配置为什么不再按内部倍率扩胞？** 旧版扩胞字段已经移除。加载时会显示迁移提示，请补上一张“扩胞”卡恢复尺寸。

## 输出合同

- 每个 $a$ 点输出一个具有理想原型坐标的三维周期基础晶胞。
- `atoms.arrays["sublattice"]` 保存逐原子 A/B 标签。
- `ordered_alloy_prototype` metadata 保存原型、晶格参数、原型阶段的元素映射和当前结构的子晶格计数；经过“扩胞”后计数会同步更新。
- `Config_type` 追加 `OrderedProto(<prototype>,a=<a>)`。

<details>
<summary>示例配置：L1₂ Cu₃Au 基础晶胞</summary>

```json
{
  "class": "OrderedAlloyPrototypeCard",
  "check_state": true,
  "params": {
    "prototype": "L12/A3B",
    "a_range": [3.75, 3.75, 0.1],
    "covera": 1.0,
    "sublattice_elements": "A:Cu,B:Au",
    "max_outputs": 1
  }
}
```

</details>
