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

`int`，默认 1。每个输入结构生成几个独立的随机 packing 版本。

10-20 个适合无序初始构型试探；50+ 时注意——每个输出结构都要靠随机试错来放置，高密度体系会显著变慢，而且 DFT 预算也成倍放大。

#### Composition（composition）

`str`，默认空。留空 = 沿用输入结构的元素种类和原子计数。手动指定时写整数计数，如 `Fe:32, O:64`。

注意这一卡不接受比例（不接受 `Fe:0.33` 这种写法）。因为随机 packing 需要确切知道每个种类有多少个原子才能放置。

### 距离约束

原子是一个一个随机扔进盒子里的。每扔一个，都要检查它跟已经放置的所有原子之间的距离够不够大。下面三个参数控制"多大算够大"以及"试多少次放弃"。

#### Min Distance（min_distance）

`float`，默认 1.5 Å。全局最小原子对距离——任何两个原子都不允许比这个值更近。

设太小会放行短键坏结构，设太大在高密度 cell 里可能根本放不下所有原子。一个安全的起点是：取体系最短合理键长的 70-90%。比如 Si-Si 键约 2.35A，设 1.8A 通常没问题。

#### Pair Min Distances（pair_min_distances）

`str`，默认空。按元素对覆盖全局距离。例如 `Fe-O:1.8, O-O:1.2`。

元素半径差异大时你会想用这个——O-O 通常比 Fe-O 可以更近一些，同时 Fe-O 可能需要比全局 `min_distance` 更宽松。写在 pair 里的距离覆盖全局值，没写的元素对仍然用 `min_distance`。

#### Max Attempts Per Atom（max_attempts_per_atom）

`int`，默认 500。每个原子最多随机尝试多少次放置位置，超过就认为放不下。

增大它只增加搜索时间，不改变物理可行性。如果你发现它一直失败（几乎每次都触达 500 次上限），不是该调大它——是你的 `min_distance` 设大了，或者 cell 相对原子数太小。先调松距离或扩胞。

#### Strict Mode（strict_mode）

`bool`，默认 true。打开：只要有一个原子放不下，整个样本就报错，保证输出数量 = 设定值。关闭：放不下的样本被跳过，最终输出可能比你设的 `structures` 少。

如果你在做参数扫描、不确定什么约束组合能放得下→ 先关掉 strict，看实际产出了几个样本，再决定要不要调参数。

### 随机性

#### Use Seed（use_seed）

`bool`，默认 false。打开后同参数同输入 → 相同的 packing 结果。对比实验时开，纯探索可以关。

#### Seed（seed）

`int`，默认 0。仅 `use_seed` 打开时生效。

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
