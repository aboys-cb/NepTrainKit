<!-- card-schema: {"card_name": "Crystal Prototype Builder", "source_file": "src/NepTrainKit/ui/views/_card/crystal_prototype_builder_card.py", "serialized_keys": ["params"]} -->

# 晶体原型构建（Crystal Prototype Builder）

`Group`: `Lattice` | `Class`: `CrystalPrototypeBuilderCard`

## 功能说明

从 fcc / bcc / hcp / diamond 等标准晶格原型直接生成晶体结构。选定元素和晶格常数后，程序按模板构造单胞并自动扩胞至目标原子数。这是一张生成器卡（Generator），不需要输入结构。

## 操作示例

### 场景：训练集缺少跨晶型的比较样本

你在 fcc Ni 上训练了一个 NEP 模型，fcc 相的弹性、声子都很好。但这个 Ni 体系理论上可能存在 hcp 亚稳相——模型在 hcp 相上预测的能量偏高，无法正确给出 fcc/hcp 的能量差。

**诊断思路：** 训练集里只有一种晶型，模型自然偏向它见过的唯一结构。需要把 fcc 和 hcp 两种原型的干净结构都加入训练集，让模型有内插依据来比较不同晶型的能量。这些原型结构不需要是真实实验构型——它们作为"锚点"帮助模型构造相空间。

**输入：** 无（生成器卡，不依赖上游结构）

**目标：** 生成 fcc + hcp 两种晶型的 Ni 结构，晶格常数在实验值附近略有扫描

**参数设置：**
- `Lattice` = `fcc`，`Element` = `Ni`，`A Range` = `[3.52, 3.52, 0.05]`
- `Auto Supercell` = 勾选，`Max Atoms` = `[256]`
- 再开一张同参数但 `Lattice` = `hcp` 的卡

**输出：** fcc 和 hcp 两种晶型的 Ni 超胞结构，带 `Proto(fcc,a=...,rep=...)` / `Proto(hcp,a=...,rep=...)` 标签

**怎么验证训练集质量改善：**
- 重训后计算 fcc/hcp 能量差，应接近 DFT 参考
- 检查输出结构原子数和对称性：fcc 应为立方对称，hcp 六角对称
- 如果输出原子数远小于 `max_atoms`，说明 auto_supercell 找不到合适的扩胞因子——关掉 `auto_supercell`，手动设 `Rep`

### 什么时候加这张卡、什么时候不加

**加：**
- 训练集缺少某种标准晶型的干净构型
- 需要对比不同晶型的能量（相稳定性）
- 作为后续缺陷/表面/磁性流程的结构工厂起点

**不加：**
- 已经有充足的真实结构（实验或 DFT 弛豫的），不需要人造原型
- 体系是复杂多相材料，简单晶格原型不能代表实际相

## 参数说明

### `Lattice`（lattice）

晶格原型类型。支持：`fcc`、`bcc`、`hcp`、`sc`、`diamond`、`zincblende`、`rocksalt`、`cesiumchloride`、`fluorite`、`wurtzite`。

- 主晶型（fcc/bcc/hcp）：常见金属
- 化合物晶型（rocksalt/wurtzite/fluorite）：二元化合物

### `Element`（element）

元素符号，如 `Cu`、`Ni`、`Si`。对于化合物原型（如 rocksalt），只用第一个元素，后续需要 `Composition Sweep` + `Random Occupancy` 组合来实现元素分布。

### `A Range`（a_range）

`[最小值, 最大值, 步长]`，晶格常数 a（单位 A）。

- 单点扫描：`[3.52, 3.52, 0.1]` → 1 个结构
- 区间扫描：`[3.50, 3.60, 0.02]` → 6 个结构
- 注意：步长越小 → 结构越多。配合 `max_outputs` 控制上限。

### `C/a`（covera）

仅 hcp 晶型使用。c/a 比例，理想值 ≈ 1.633。

### `Auto Supercell`（auto_supercell）

勾选 → 程序自动计算扩胞因子以逼近 `Max Atoms`。不勾选 → 手动用 `Rep` 指定扩胞倍数。

### `Max Atoms`（max_atoms）

目标最大原子数。仅 `auto_supercell` 勾选时生效。256~512 为常规。

### `Rep`（rep）

`[na, nb, nc]`，三个方向的扩胞倍数。仅 `auto_supercell` 关闭时生效。

### `Max Outputs`（max_outputs）

输出结构总数上限。防止 a_range 扫描+多晶型产生过多结构。

## 推荐预设

### fcc Cu 单点（1 个输出，作为流程起点）
```json
{
  "class": "CrystalPrototypeBuilderCard",
  "check_state": true,
  "lattice": "fcc",
  "element": "Cu",
  "a_range": [3.615, 3.615, 0.1],
  "covera": [1.633],
  "auto_supercell": true,
  "max_atoms": [256],
  "rep": [4, 4, 4],
  "max_outputs": [1]
}
```

### fcc Cu 晶格扫描（5 个点，覆盖平衡晶格附近）
```json
{
  "class": "CrystalPrototypeBuilderCard",
  "check_state": true,
  "lattice": "fcc",
  "element": "Cu",
  "a_range": [3.55, 3.65, 0.025],
  "covera": [1.633],
  "auto_supercell": true,
  "max_atoms": [256],
  "rep": [4, 4, 4],
  "max_outputs": [20]
}
```

### 多晶型探索（fcc + hcp，各 5 个点）
```json
{
  "class": "CrystalPrototypeBuilderCard",
  "check_state": true,
  "lattice": "hcp",
  "element": "Ni",
  "a_range": [3.50, 3.56, 0.015],
  "covera": [1.633],
  "auto_supercell": true,
  "max_atoms": [256],
  "rep": [4, 4, 4],
  "max_outputs": [20]
}
```

## 推荐组合

- `Crystal Prototype Builder` → `Atomic Perturb`：先建原型，再加坐标噪声
- `Crystal Prototype Builder` → `Composition Sweep` → `Random Occupancy`：先生成干净模板，再进行成分修饰
- `Crystal Prototype Builder` → `Lattice Strain`：先建原型，再做应变扫描

## 常见问题

**输出只有 1 个结构。** a_range 的起点=终点且步长>0 时只产生单点。如果需要扫描，设定不同起点和终点。

**原子数远小于 `max_atoms`。** auto_supercell 的扩胞因子是整数，可能因为单胞原子数多导致最近整数倍已超过 max_atoms。手动关掉 auto_supercell 改 `rep`。

**hcp 结构不是六角的。** hcp 的 ASE bulk 构造需要正确的 covera。确认 covera 在 1.6 左右。

## 输出标签

`Proto({晶格类型},a={晶格常数},rep={na}x{nb}x{nc})`

## 可复现性

无随机性。同参数 → 严格一致输出。
