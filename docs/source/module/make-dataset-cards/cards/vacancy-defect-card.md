<!-- card-schema: {"card_name": "Vacancy Defect Generation", "source_file": "src/NepTrainKit/ui/views/_card/vacancy_defect_card.py", "serialized_keys": ["params"]} -->

# 空位缺陷生成（Vacancy Defect Generation）

`Group`: `Defect` | `Class`: `VacancyDefectCard`

## 功能说明

按数量或比例删除原子，快速生成低到高缺陷强度的空位结构分布。支持 Sobol / Uniform 两种随机引擎，可选 count 或 fraction 两种主模式，并可选择固定空位数或随机空位数。

**和 `Random Vacancy` 的区别：**
- `Vacancy Defect Generation`：统计驱动，按整体比例/数量生成空位，适合快速覆盖空位密度分布
- `Random Vacancy`：规则驱动，按元素+group 精确删除位点，适合定向研究

## 操作示例

### 场景：模型把空位构型全部预测成接近完美晶体

你在 100 原子的 Fe 超胞上训练了一个 NEP，训练集里只有弛豫好的完美晶体。模型推理一个含 1 个空位的构型时，能量几乎和完美晶体一样——模型没学过"少了一个原子"是什么后果。

**诊断思路：** 模型在完美晶体上收敛是因为训练集只有完美晶体。空位附近原子的力场完全不同：键断了一半，局域电子密度变化大。需要往训练集里加入从 1 个空位到 ~5% 浓度的分布，让模型覆盖低/中/高缺陷密度段。

**输入：** 一个 100 原子的 Fe 超胞

**目标：** 用 fraction 模式生成 5% 空位浓度，每帧 50 个随机落点版本

**参数设置：**
- `Vacancy Fraction`：勾选
- `Vacancy Fraction`：`[0.05]`（100 原子里精确删除 5 个空位）
- `Count Mode`：`Fixed count`
- `Structures`：`[50]`
- `Random Engine`：`Uniform`

**输出：** 50 个空位结构，每个都删除 5 个原子，但空位落点不同，带 `Vac(n=5)` 标签

**怎么验证训练集质量改善：**
- 重训后用含空位的测试集推理，力 MAE 应对空位近邻原子也有合理精度
- 如果低浓度 OK 但高浓度崩，说明训练集高浓度样本不够——增加 `concentration_condition`
- 如果样本覆盖感觉不均匀，换 Sobol 引擎再跑一批对比

### 什么时候加这张卡、什么时候不加

**加：**
- 需要快速覆盖"少了一个原子"对力/能量的影响
- 训练集只有完美晶体，需要低/中/高缺陷密度梯度
- 不需要按元素类型区分空位来源

**不加：**
- 需要精确控制删哪种元素 → 用 `Random Vacancy`
- 缺陷浓度 >20%——此时结构通常已剧烈变形，需先确认是否物理上有意义

## 参数说明

### 主模式：Count vs Concentration

**`Vacancy Count`**（num_radio_button）：勾选 → count 模式，用 `Vacancy Count` 的整数值决定删几个原子。默认开启。

**`Vacancy Fraction`**（concentration_radio_button）：勾选 → fraction 模式，用 `Vacancy Fraction` 的比例计算目标删除数：`target_defects = int(fraction * n_atoms)`。与 count 模式二选一。

**`Vacancy Count`**（num_condition）：count 模式下的空位数量。范围为 1-8。

**`Vacancy Fraction`**（concentration_condition）：fraction 模式下的空位比例。0.02 表示约 2% 的原子被删。推荐 0.005-0.08。

**`Count Mode`**（count_mode）：`fixed` 精确删除上面指定的数量；`random` 在 1 到指定数量之间随机。新建卡片默认 `fixed`，只有显式选择 `Random up to value` 时才随机空位数量。

### `Random Engine`（engine_type）

| 值 | 引擎 | 特点 |
|----|------|------|
| `0` (Sobol) | 准随机序列 | 样本少时对空位数量和位置的覆盖更均衡 |
| `1` (Uniform) | 均匀随机 | 大批量时更快，统计差异不大 |

### `Structures`（max_structures）

每输入帧生成多少个空位版本。

- 10~50：轻量补样
- 50~100：常规覆盖
- 100+：建议后接 `FPS Filter`

### `Use Seed` / `Seed`（use_seed / seed）

勾选 `use_seed` + 固定 `seed` 可复现。

## 推荐预设

### 保守（Safe，Sobol 引擎，2% 浓度，50 个输出）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 0,
  "num_radio_button": false,
  "concentration_radio_button": true,
  "num_condition": [1],
  "concentration_condition": [0.02],
  "count_mode": "fixed",
  "max_atoms_condition": [50],
  "use_seed": true,
  "seed": [42]
}
```

### 平衡（Balanced，Uniform 引擎，5% 浓度，50 个输出）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 1,
  "num_radio_button": false,
  "concentration_radio_button": true,
  "num_condition": [1],
  "concentration_condition": [0.05],
  "count_mode": "fixed",
  "max_atoms_condition": [50],
  "use_seed": true,
  "seed": [42]
}
```

### 探索（Exploration，Sobol，10% 浓度，100 个输出）
```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "engine_type": 0,
  "num_radio_button": false,
  "concentration_radio_button": true,
  "num_condition": [1],
  "concentration_condition": [0.10],
  "count_mode": "fixed",
  "max_atoms_condition": [100],
  "use_seed": true,
  "seed": [42]
}
```

## 推荐组合

- `Super Cell` → `Vacancy Defect Generation`：先扩胞再删，避免小胞里空位相互作用太强
- `Vacancy Defect Generation` → `Insert Defect`：空位 + 插隙，互补缺陷族
- `Vacancy Defect Generation` → `Random Vacancy`：先覆盖密度分布，再补定向规则删除

## 常见问题

**输出结构数远少于预期。** 空位数为 0 时不会生成。检查浓度模式下 `concentration * n_atoms` 是否≥1。

**空位太多导致骨架崩坏。** 浓度模式 `0.10` 以上已到强缺陷区。先检查是否有孤立原子或明显断裂，再决定是否回调。

**count 和 fraction 怎么选。** 二者只选一个：想精确删几个原子用 `Vacancy Count`；想按体系大小等比例删用 `Vacancy Fraction`。

## 输出标签

`Vac(n={删除原子数})`

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。输入结构顺序变化也会影响结果。
