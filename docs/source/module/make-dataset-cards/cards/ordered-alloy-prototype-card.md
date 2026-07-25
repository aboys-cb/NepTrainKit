<!-- card-schema: {"card_name": "Ordered Alloy Prototype", "source_file": "src/NepTrainKit/ui/views/_card/ordered_alloy_prototype_card.py", "serialized_keys": ["params"]} -->

# 有序合金原型（Ordered Alloy Prototype）

`Group`: `Alloy` | `Class`: `OrderedAlloyPrototypeCard`

## 功能说明

生成带晶体学子晶格身份的周期结构，首批支持 A1/fcc、A2/bcc、A3/hcp、L1₂/A3B、B2/AB 和 L1₀/AB。子晶格写入逐原子数组 `atoms.arrays["sublattice"]`；它不会占用 `group`，因此后续仍可独立添加磁序分层、奇偶层或区域标签。

这张卡只建立晶胞、周期边界、分数坐标和子晶格。元素占位可以先给真实元素，也可以用 `X` 占位符，再交给 `Finite-Cell Alloy Occupancy` 生成整数可实现的有序或部分无序合金。

## 操作示例

### 场景：L1₂ 相测试误差高，但训练集中没有保留角点/面心身份

模型在 A3B 型有序相上的力误差明显高于随机固溶体。原训练集把所有 fcc 位点混成同一个集合，无法系统覆盖 A、B 子晶格分别无序化的局域环境。

**输入：** 无需输入结构。
**目标：** 先生成 32 原子的 L1₂ 周期超胞，其中 A、B 子晶格严格为 24:8。
**参数设置：** `prototype="L12/A3B"`，`a_range=[3.6,3.6,0.1]`，`sublattice_elements="A:X,B:X"`，手动 `rep=[2,2,2]`，`max_atoms=32`。
**输出：** 一个 32 原子周期结构，`sublattice` 中有 24 个 `A` 和 8 个 `B`。
**怎么验证训练集质量改善：** 下游分别约束 A/B 占位并完成 DFT 标注后，按原型和子晶格无序程度切片比较能量/力误差；不能只看混合后的总 RMSE。

## 参数说明

### Prototype（prototype）

`str`，默认 `"L12/A3B"`。选择晶体原型及其固定的分数坐标和子晶格拓扑。

| 选项 | 基胞位点 | 子晶格计数 |
|---|---:|---|
| `A1/fcc` | 4 | A=4 |
| `A2/bcc` | 2 | A=2 |
| `A3/hcp` | 2 | A=2 |
| `L12/A3B` | 4 | A=3，B=1 |
| `B2/AB` | 2 | A=1，B=1 |
| `L10/AB` | 4 | A=2，B=2 |

### Lattice Range（a_range）

`tuple[float, float, float]`，默认 `[3.6, 3.6, 0.1]`，依次是晶格常数 `a` 的起点、终点和步长，单位 Å。

需要补体积响应时可扫描一段窄范围；只要一个结构时令起点等于终点。每个扫描点都受 `max_outputs` 限制。

### c/a（covera）

`float`，默认 1.0。A3/hcp 和 L1₀ 使用该值设置 `c/a`；立方 A1、A2、L1₂、B2 固定使用 1，并在界面中禁用该输入，避免误以为它会生效。

A3 常从理想密排值约 1.633 起步；L1₀ 则应使用目标体系的四方畸变范围。不要用这个参数替代后续的系统应变扫描。

### Sublattice Elements（sublattice_elements）

`str`，默认 `"A:X,B:X"`。按 `标签:元素` 指定基胞占位，例如 `A:Fe,B:Al`；`X` 表示尚未决定元素的占位符。

元素选择不带材料先验：卡片不会假设过渡金属属于 A，也不会限制主族元素只能进入 B。切换原型时，界面明确提示所需标签和基胞计数；单子晶格原型只显示 A，不再保留一个当前不会使用的 B 配置。未知标签会明确报错。

### Auto Supercell（auto_supercell）

`bool`，默认 true。开启后，在不超过 `max_atoms` 的前提下自动选择三轴重复数。

自动模式适合快速得到接近预算的周期胞；需要固定 24:8、8:8 等可审计计数时，手动重复通常更直观。

### Max Atoms（max_atoms）

`int`，默认 128。每个输出结构允许的原子数硬上限。

如果基胞本身或手动 `rep` 超过这个上限，operation 直接抛出 `ValueError`，不会偷偷缩小重复数或返回超预算结构。

### Rep（rep）

`tuple[int, int, int]`，默认 `[2, 2, 2]`。手动模式下沿三个晶格矢量的重复数。

生效条件：`auto_supercell=false`。L1₂ 的 `[2,2,2]` 正好得到 32 原子，其中 A:B=24:8。

### Max Outputs（max_outputs）

`int`，默认 200。晶格常数扫描最多返回的结构数。

扫描范围很密时，它是独立于原子数的第二道预算门；达到上限后按 `a_range` 的确定顺序停止。

## 推荐预设

### L1₂ 32 原子占位模板

```json
{
  "class": "OrderedAlloyPrototypeCard",
  "params": {
    "prototype": "L12/A3B",
    "a_range": [3.6, 3.6, 0.1],
    "covera": 1.0,
    "sublattice_elements": "A:X,B:X",
    "auto_supercell": false,
    "max_atoms": 32,
    "rep": [2, 2, 2],
    "max_outputs": 1
  }
}
```

### L1₀ 四方参数扫描

```json
{
  "class": "OrderedAlloyPrototypeCard",
  "params": {
    "prototype": "L10/AB",
    "a_range": [3.7, 3.9, 0.1],
    "covera": 0.95,
    "sublattice_elements": "A:Fe,B:Pt",
    "auto_supercell": true,
    "max_atoms": 64,
    "rep": [2, 2, 2],
    "max_outputs": 3
  }
}
```

## 推荐组合

- `Ordered Alloy Prototype → Finite-Cell Alloy Occupancy`：先锁定晶体学子晶格，再按每个子晶格的整数计数生成占位。
- `Ordered Alloy Prototype → Lattice Strain → Atomic Perturb`：补有序相的应变和局域振动环境。
- `Ordered Alloy Prototype → Magnetic Order`：晶体学 `sublattice` 与磁性 `group` 分开保存，再独立构造磁序。

## 常见问题

**提示未知子晶格标签。** 首批原型只使用 A/B；占位映射可同时保留 A、B 方便切换原型，但不能写任意新标签。

**手动超胞超过 `max_atoms`。** 卡片会明确失败，不会改变用户给出的重复数。调小 `rep` 或提高原子数预算。

**为什么没有弛豫。** 本卡只生成几何原型。弛豫、势函数调用、LAMMPS、chemical MC 和 DFT 提交不属于该卡职责。

## 输出标签

`Config_type` 追加 `OrderedProto(<prototype>,a=<a>,rep=<na>x<nb>x<nc>)`。`ordered_alloy_prototype` metadata 记录原型、晶格参数、重复数、子晶格元素和实际子晶格计数。

## 可复现性

本卡没有随机采样。同一组参数总是得到相同的 cell、PBC、分数坐标、元素和 `sublattice` 数组。
