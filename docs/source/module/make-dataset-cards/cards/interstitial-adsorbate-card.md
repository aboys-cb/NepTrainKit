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

### 插入模式和数量

#### Mode（mode）

`int`，默认 0。`0` = Interstitial（体相插隙），在晶胞内随机采样位点放入额外原子；`1` = Adsorption（表面吸附），沿表面法向在表面上方放置原子。

Interstitial 模式下 `axis` 和 `offset` 不生效。Adsorption 模式需要 slab 输入，`axis` 决定了吸附原子沿哪条晶轴放。

#### Species（species）

`str`，默认空。逗号分隔的元素列表，指定要插入什么原子。支持权重写法：`Li:0.7,Na:0.3` 表示每次随机放置时有 70% 概率插 Li，30% 概率插 Na。权重不写则等概率。

#### Insert Count（insert_count）

`int`，默认 1。每个生成结构里插几个额外原子。1~3 是常见范围——单原子插隙最容易放进去，数量越大碰撞失败概率越高。

#### Structure Count（structure_count）

`int`，默认 10。每输入帧生成多少个插入版本。

10~50 做轻量验证，50~200 常规覆盖，200 以上建议接 `FPS Filter`。

### 几何约束

#### Min Distance（min_distance）

`float`，默认 1.4。候选插入点与已有原子的最小允许距离，单位 A。

对致密晶体，1.6~2.5 A 比较保守安全；1.2~1.6 A 是一个平衡点；压到 0.8~1.2 A 碰撞风险明显升高，只有在你确信结构能容纳时才用。

#### Max Attempts（max_attempts）

`int`，默认 200。每个待插入原子的最大随机尝试次数。增大它可以提高放置成功率，但耗时线性增长——如果你发现 200 次几乎每次都触顶失败，说明问题不是尝试次数不够，而是 `min_distance` 设大了或者胞太密。

### 表面吸附

#### Axis（axis）

`int`，默认 2。Adsorption 模式下它是表面法向。如果你的 slab 真空沿 z，默认 2（z 轴）就不用改；表面取向不同时一定要调，否则吸附高度方向会错。

生效条件：涉及方向、分层、表面或向量初始化的模式都会使用。

#### Offset（offset）

`float`，默认 1.5。表面吸附时原子放在表面上方多远，单位 A。

生效条件：`mode` 选择 Adsorption 时。

### 随机性

#### Use Seed（use_seed）

`bool`，默认 false。打开后固定 seed → 同样输入和参数每次跑出一样的插入结果。对比实验时开，纯探索可以关掉。

#### Seed（seed）

`int`，默认 0。随机种子值。

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
