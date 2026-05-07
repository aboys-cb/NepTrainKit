<!-- card-schema: {"card_name": "Insert Defect", "source_file": "src/NepTrainKit/ui/views/_card/interstitial_adsorbate_card.py", "serialized_keys": ["params"]} -->

# 插隙/吸附缺陷（Insert Defect）

`Group`: `Defect` | `Class`: `InsertDefectCard`

## 功能说明

在体相或表面结构中随机插入额外原子。支持 Interstitial（体相间隙位插入）和 Adsorption（表面吸附位放置）两种模式，用最小距离约束避免原子碰撞。

## 操作示例

### 场景：模型对 Li 扩散路径预测完全错误

你在 LiCoO2 上训练了一个 NEP 模型跑 Li 扩散。模型预测的扩散势垒比 DFT 低了一半——中间态的能量被严重低估。检查训练集发现：所有结构里 Li 都待在八面体位点，从来没见过 Li 在四面体间隙位的结构。

**诊断思路：** 扩散路径上的过渡态对应 Li 从一个八面体位穿过四面体间隙位跃迁到相邻八面体位的过程。训练集里只有基态构型，模型自然会低估中间态能量。需要往训练集里加入 Li 在四面体间隙位附近的构型，让模型见过这些局域环境。

**输入：** 一个 LiCoO2 超胞

**目标：** 在体相中插入单个 Li 原子于随机间隙位，生成 50 个候选构型

**参数设置：**
- `Mode` = `Interstitial`
- `Species comma-separated` = `Li`
- `Atoms per structure` = `[1]`
- `Structures to generate` = `[50]`
- `Min distance (A)` = `[1.4]`
- `Max Attempts` = `[200]`

**输出：** 50 个插隙结构，每个含一个额外 Li 原子放置于不与宿主原子碰撞的位置，带 `Ins(int,n=1)` 标签

**怎么验证训练集质量改善：**
- 重训后用 NEB 算 Li 扩散势垒，应接近 DFT 参考值
- 抽查插入原子的最近邻距离：≥ min_distance 是底线，太近（<1.0 A）的结构应该不会出现
- 如果成功率很低（输出远少于 50 个），增大 `max_attempts` 或降低 `min_distance`
- 如果是表面吸附场景，切到 `Mode` = `Adsorption`，调整 `offset` 控制吸附高度

### 什么时候加这张卡、什么时候不加

**加：**
- 模型对扩散、吸附、插层位点预测不准
- 训练集只有完美晶格占据，缺少间隙原子环境
- 研究表面催化、电池材料离子输运

**不加：**
- 只需基态占据 → 不需要额外插入原子
- 插入物种与宿主剧烈反应（如碱金属+水）→ 先确认化学可行性

## 参数说明



### 插入模式

#### Mode（mode）
类型：`int`。默认：`0`。选择这张卡的主操作模式。

物理直觉：主模式会改变生成语义；不要只为了多出结构而混用物理含义不同的模式。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |


#### Species（species）
类型：`str`。默认：`''`。设置插入或吸附的元素列表。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Insert Count（insert_count）
类型：`int`。默认：`1`。设置每个结构插入的原子数。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


#### Structure Count（structure_count）
类型：`int`。默认：`10`。设置每个输入结构生成的候选结构数量。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


### 几何约束

#### Min Distance（min_distance）
类型：`float`。默认：`1.4`。设置新原子或随机坐标与现有原子的最小距离约束。

物理直觉：它是几何硬约束；设得过小会保留短键坏结构，设得过大可能导致随机放置失败。


#### Max Attempts（max_attempts）
类型：`int`。默认：`200`。限制随机放置失败前的最大尝试次数。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。


### 表面吸附

#### Axis（axis）
类型：`int`。默认：`2`。选择操作沿哪个空间轴或磁矩参考轴定义。

物理直觉：坐标轴参数会改变空间分层、吸附方向或磁矩方向；使用前先确认结构取向。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。


#### Offset（offset）
类型：`float`。默认：`1.5`。设置表面吸附相对表面的偏移距离。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：`mode` 选择 Adsorption 时。


### 随机性

#### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现的训练集生成或测试时打开；做最终大规模探索且希望保留随机多样性时可关闭。


#### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一 seed 应产生同一批候选；只有在 `use_seed` 打开时才改变结果。

生效条件：`use_seed=True`。

## 推荐预设

### 体相单原子间隙（Interstitial，50 个输出）
```json
{
  "class": "InsertDefectCard",
  "check_state": true,
  "mode": 0,
  "species": "Li",
  "insert_count": [1],
  "structure_count": [50],
  "min_distance": [1.4],
  "max_attempts": [200],
  "use_seed": true,
  "seed": [42],
  "axis": 2,
  "offset": [1.5]
}
```

### 表面单原子吸附（Adsorption，50 个输出）
```json
{
  "class": "InsertDefectCard",
  "check_state": true,
  "mode": 1,
  "species": "O,H",
  "insert_count": [1],
  "structure_count": [50],
  "min_distance": [1.6],
  "max_attempts": [300],
  "use_seed": true,
  "seed": [42],
  "axis": 2,
  "offset": [2.0]
}
```

### 多物种间隙探索（Interstitial，200 个输出）
```json
{
  "class": "InsertDefectCard",
  "check_state": true,
  "mode": 0,
  "species": "Li:0.5,Na:0.3,Mg:0.2",
  "insert_count": [2],
  "structure_count": [200],
  "min_distance": [1.2],
  "max_attempts": [400],
  "use_seed": true,
  "seed": [42],
  "axis": 2,
  "offset": [1.5]
}
```

## 推荐组合

- `Random Slab` → `Insert Defect`：先切表面 slab，再加吸附原子
- `Insert Defect` → `Random Vacancy`：互补缺陷族（间隙 + 空位）
- `Insert Defect` → `FPS Filter`：大批量生成后做代表性筛选

## 常见问题

**输出为空或数量远少于预期。** `min_distance` 太严、`max_attempts` 太小，或者输入超胞太密找不到合适位点。先放宽 min_distance 到 1.0 A 试跑。

**插入原子和宿主重叠。** 检查 min_distance 是否 ≤ 0（被跳过）。如果 min_distance 合理但仍重叠，增大 `max_attempts`。

**Adsorption 模式下原子放在 box 外面。** 检查 axis 是否指向了正确的表面法向。如果 slab 的 c 轴不是表面法向，需要调整 axis。

## 输出标签

`Ins(int,n={插入数})` / `Ins(ad,n={插入数})`

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。输入顺序变化也会影响随机路径。
