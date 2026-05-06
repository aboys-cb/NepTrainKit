<!-- card-schema: {"card_name": "Random Slab", "source_file": "src/NepTrainKit/ui/views/_card/random_slab_card.py", "serialized_keys": ["params"]} -->

# 随机表面切片（Random Slab）

`Group`: `Surface` | `Class`: `RandomSlabCard`

## 功能说明

从体相结构按 Miller 指数、层数和真空厚度范围批量切出 slab，覆盖不同表面取向和几何参数组合。

## 操作示例

### 场景：模型在表面吸附能上系统性偏离 DFT

你在体相 Pt 上训练了一个 NEP 模型，体相弹性和声子都很好。然后跑 Pt(111) 表面上的 CO 吸附能——模型预测的吸附能比 DFT 弱 0.5 eV，误差远超可接受范围。

**诊断思路：** 训练集里全是体相 Pt 结构，模型从来没见过表面的低配位原子。Pt(111) 表面的台阶位、平台位的局域环境与体相完全不同——配位数从 12 降到 9 甚至更低。模型需要见过多种表面取向和厚度的 slab 才能泛化到表面化学。

**输入：** 一个体相 fcc Pt 单胞

**目标：** 切出 (111) 面，3~6 层厚，10 A 真空，覆盖多种 Miller 指数组合

**参数设置：**
- `H Range` = `[0, 1, 1]`，`K Range` = `[0, 1, 1]`，`L Range` = `[1, 3, 1]`
- `Layer Range` = `[3, 6, 1]`
- `Vacuum Range` = `[10, 10, 1]`

**输出：** 多个 slab 结构，覆盖不同 hkl 组合、3~6 层厚度、10 A 真空，带 `Slab(hkl=...,L=...,vac=...)` 标签

**怎么验证训练集质量改善：**
- 重训后计算 Pt(111) 上的 CO 吸附能，应该接近 DFT 参考
- 检查 slab 上下表面没有重叠，真空层足够隔离周期镜像
- 如果表面能预测仍不准，扩大 `l_range` 到 `[1, 5, 1]` 覆盖更多高指数面
- 如果 slab 太薄表面效应过强，增大 `layer_range` 的下限

### 什么时候加这张卡、什么时候不加

**加：**
- 模型在表面相关任务（吸附、表面能、表面反应）上预测差
- 需要覆盖多种表面取向和厚度
- 作为后续吸附/缺陷卡片的母结构

**不加：**
- 只做体相性质训练，不需要表面结构
- 已经手动切好了 slab，只需要做表面修饰

## 参数说明

### Miller 指数范围（h_range / k_range / l_range）

`[最小值, 最大值, 步长]`，整数，定义切片的 Miller 指数空间。

hkl 为 (0,0,0) 的组合自动跳过。输出总数 = (h 点数) x (k 点数) x (l 点数) x (层数) x (真空数)。

- 保守：h/k/l = [0,1,1]，适合低指数面
- 平衡：h/k = [0,2,1], l = [1,4,1]，覆盖更多取向
- 探索：h/k = [0,3,1], l = [1,5,1]，组合数快速增长

### `Layer Range`（layer_range）

`[最小值, 最大值, 步长]`，整数，定义 slab 厚度（层数）。

- 3~6 层：标准 slab 厚度，上下表面原子环境近似体相
- 1~2 层：超薄 slab，量子尺寸效应明显，谨慎使用
- 8+ 层：厚 slab，计算成本高

### `Vacuum Range`（vacuum_range）

`[最小值, 最大值, 步长]`，单位 A。

- 10 A：大多数场景的下限，足够隔绝镜像相互作用
- 15~20 A：吸附大分子时需要，确保吸附物不与镜像 slab 相交
- 5 A 以下：可能不可靠，镜像 slab 的电子密度会穿透真空层

## 推荐预设

### 低指数面（111，3~6 层，10 A，少量输出）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [0, 1, 1],
  "k_range": [0, 1, 1],
  "l_range": [1, 1, 1],
  "layer_range": [3, 6, 1],
  "vacuum_range": [10, 10, 1]
}
```

### 多取向覆盖（h/k 0-1, l 1-3，3~6 层，10-15 A）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [0, 1, 1],
  "k_range": [0, 1, 1],
  "l_range": [1, 3, 1],
  "layer_range": [3, 6, 1],
  "vacuum_range": [10, 15, 5]
}
```

### 全取向探索（h/k 0-2, l 1-5，3~9 层，10-20 A）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [0, 2, 1],
  "k_range": [0, 2, 1],
  "l_range": [1, 5, 1],
  "layer_range": [3, 9, 3],
  "vacuum_range": [10, 20, 5]
}
```

## 推荐组合

- `Random Slab` → `Insert Defect`：先切表面，再加吸附原子
- `Random Slab` → `Random Vacancy`：先切表面，再删表面原子构造空位
- `Super Cell` → `Random Slab`：先扩胞到足够大面内尺寸，再切片

## 常见问题

**输出为空。** 通常因为 (0,0,0) 被跳过后没有有效 hkl 组合，或者体相的 Miller 指数组合和晶格不兼容。

**输出数量爆炸。** h_range、k_range、l_range、layer_range、vacuum_range 五重循环。先估算：Nh x Nk x Nl x Nlayer x Nvacuum。如果数字太大，先收窄 l_range 和 layer_range。

**Slab 上下表面重叠。** layer_range 下限太小时，slab 厚度可能小于 2 层原子间距。增大下限。

## 输出标签

`Slab(hkl={h}{k}{l},L={层数},vac={真空})`

## 可复现性

无随机性。同参数同输入 → 严格一致输出。
