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

`structures`：int，默认 `1`。每个输入结构生成多少个随机 packing。

`composition`：string，默认 `""`。空字符串表示保持输入结构的元素计数。非空时必须写精确正整数计数，例如 `Fe:32,O:64`；不接受 `Fe:0.5,O:0.5` 这类比例，因为本卡需要确定的离散原子数。

`min_distance`：float，默认 `1.5`。所有未被 pair-specific 规则覆盖的元素对使用这个最小距离，单位 Angstrom。

`pair_min_distances`：string，默认 `""`。pair-specific 最小距离，格式如 `Fe-O:1.8, O-O:1.2, Fe-Fe:2.0`。元素顺序无关，未指定的 pair 回到 `min_distance`。

`max_attempts_per_atom`：int，默认 `500`。每个原子最多尝试多少次随机位置。

`strict_mode`：bool，默认 `True`。开启时，只要任一 requested sample 放置失败，整张卡失败且不返回半成品。关闭时失败 sample 被跳过；如果一个都没有成功才报错。

`use_seed`：bool，默认 `False`。开启后随机坐标可复现。

`seed`：int，默认 `0`。`use_seed=True` 时作为基础 seed，并与输入结构 ID 和 sample 序号派生每个输出的随机流。

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
