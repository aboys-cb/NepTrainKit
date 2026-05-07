<!-- card-schema: {"card_name": "Random Packing", "source_file": "src/NepTrainKit/ui/views/_card/random_packing_card.py", "serialized_keys": ["params"]} -->

# Random Packing

`Group`: `Structure` | `Class`: `RandomPackingCard`

## 功能说明

`Random Packing` 是原子坐标卡，不是磁性卡。它保留输入结构的 `cell` 和 `pbc`，按输入组成或手动精确组成在 cell 内重新随机放置原子，并用全局和 pair-specific 最小距离约束挡掉明显不可能的短键初态。

这张卡解决的是“训练集缺少无序初始构型”，不是“缺少磁矩无序”。磁矩无序使用 `Spin Disorder` 或 `Correlated Random Spin`。

## 操作示例

### 场景：模型只见过有序晶格，遇到无序初态能量排序不稳定

输入是一个已有 cell 和组成的候选结构。你希望在相同 cell 和元素计数下生成 20 个随机原子 packing，用于 amorphous seed、液相初态或高温无序初态的 DFT 采样。

参数设置：`structures=20`，`composition=""`，`min_distance=1.5`，`pair_min_distances="Fe-O:1.8,O-O:1.2"`，`strict_mode=True`，打开 `use_seed`。

输出结构保留原 cell/pbc 和元素计数，坐标随机重排，`Config_type` 追加 `RandPack(...)`。检查输出时重点看最短距离分布和后续 DFT 是否仍有明显短键崩溃。

## 参数说明

### 组成和数量

#### Structures（structures）
类型：`int`。默认：`1`。设置要生成的随机构型数量。

物理直觉：每个输入结构会乘上这个数量。10-20 个适合无序 seed 试探；50+ 会迅速放大后续 DFT 预算。

#### Composition（composition）
类型：`str`。默认：`''`。指定目标组成；留空时使用输入结构的元素和计数。

物理直觉：留空表示保持输入结构元素计数；手动组成必须给整数计数，例如 `Fe:32,O:64`。这张卡不接受比例，因为最终要生成真实原子列表。

### 距离约束

#### Min Distance（min_distance）
类型：`float`。默认：`1.5`。设置新原子或随机坐标与现有原子的最小距离约束。

物理直觉：全局硬球距离。设得太小会保留短键坏结构；设得太大在高密度 cell 中会放不下。先按最短合理键长的 70-90% 试。

#### Pair Min Distances（pair_min_distances）
类型：`str`。默认：`''`。按元素对覆盖全局最小距离约束。

物理直觉：元素半径差异明显时使用，例如 `Fe-O:1.8,O-O:1.2`。pair-specific 会覆盖全局距离，未列出的元素对仍走 `min_distance`。

#### Max Attempts Per Atom（max_attempts_per_atom）
类型：`int`。默认：`500`。限制每个原子随机放置失败前的最大尝试次数。

物理直觉：高密度、大 min_distance 或复杂 pair 约束会增加失败概率。增大它只增加搜索时间，不改变物理可行性；一直失败说明约束或 cell/组成不可能。

#### Strict Mode（strict_mode）
类型：`bool`。默认：`True`。决定单个样本失败时是中断还是跳过该样本。

物理直觉：打开时任何样本放置失败都会报错，保证输出数量契约；关闭时跳过失败样本，适合探索高密度约束但输出数可能少于设定。

### 随机性

#### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

物理直觉：需要可复现训练集、测试或对比实验时打开；最终大规模随机探索可以关闭，但结果不能逐帧复现。

#### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

物理直觉：同一输入、同一参数和同一 seed 应生成同一批候选；只有 `use_seed=True` 时改变它才会改变随机输出。

生效条件：`use_seed=True`。

## 推荐预设

### 保持输入组成

```json
{
  "class": "RandomPackingCard",
  "params": {
    "structures": 10,
    "composition": "",
    "min_distance": 1.5,
    "pair_min_distances": "",
    "max_attempts_per_atom": 500,
    "strict_mode": true,
    "use_seed": true,
    "seed": 42
  }
}
```

### 手动精确组成

```json
{
  "class": "RandomPackingCard",
  "params": {
    "structures": 20,
    "composition": "Fe:32,O:64",
    "min_distance": 1.4,
    "pair_min_distances": "Fe-O:1.8,O-O:1.2,Fe-Fe:2.0",
    "max_attempts_per_atom": 1000,
    "strict_mode": true,
    "use_seed": true,
    "seed": 7
  }
}
```

## 推荐组合

- `Crystal Prototype Builder -> Super Cell -> Random Packing`：复用目标 cell 尺度生成无序 seed。
- `Random Packing -> Geometry Filter`：先生成，再过滤短键、体积和密度异常。
- `Random Packing -> Atomic Perturb`：在已经满足硬距离约束的无序 seed 上叠加小位移噪声。

## 常见问题

**为什么不接受比例组成？** 因为这张卡生成真实原子列表，必须知道整数原子数。比例到整数的舍入会改变组成语义，应该在上游先确定超胞和计数。

**为什么默认 strict？** 链式卡片里输出数量通常会继续相乘。默认跳过失败样本会让后续数量偏离用户设定，所以默认保持严格数量契约。

## 输出标签

`RandPack(n={atom_count},d={min_distance},s={seed})`。`s` 只在 `use_seed=True` 时出现。

## 可复现性

开启 `use_seed` 后，随机 packing 由 `seed`、输入结构稳定 ID 和 sample 序号共同决定。相同输入、相同参数、相同 seed 会生成相同坐标。
