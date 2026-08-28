<!-- card-schema: {"card_name": "Random Packing", "source_file": "src/NepTrainKit/ui/views/_card/random_packing_card.py", "serialized_keys": ["params"]} -->

# 随机原子堆积（Random Packing）

**分类：** 结构

## 功能说明

当训练集只有有序晶体、缺少无序初态时，这张卡以每个输入结构的晶胞为盒子，重新随机放置全部原子。它保留输入的 `cell`、`pbc` 和结构级 `info`；元素计数来自输入组成或手动整数计数。

原位点已经不存在，因此 `spin`、`group`、`initial_magmoms` 等逐原子数组不会复制到输出。生成后应检查最短距离分布，再进入弛豫、分子动力学或单点计算流程。

## 原理与公式

每个候选位置先在晶胞的分数坐标中均匀采样：

$$
\mathbf{s}\sim U([0,1)^3),\qquad
\mathbf{r}=\mathbf{s}^{\mathsf T}\mathbf{C},
$$

其中 $\mathbf C$ 是输入的完整 $3\times3$ 晶胞矩阵。距离检查使用输入 `pbc` 对应的最小镜像距离，因此支持非正交晶胞和混合周期边界。

新原子 $i$ 只有满足

$$
d_{ij}^{\mathrm{MIC}}\ge d_{\min}(Z_i,Z_j),\qquad \forall j<i
$$

才被接受。元素对规则优先于全局阈值；没有专用规则的原子对使用全局最小距离。程序先尝试约束较大的元素，以减少后期无位置可放的概率。

若每个输出含 $N$ 个原子、每个输入请求 $M$ 个输出，则每个输入的生成预算为

$$
N_{\mathrm{generated}}=N\times M.
$$

它不得超过“生成原子预算/输入”。严格模式成功时输出数精确为 $M$；关闭严格模式后，失败样本会跳过，输出数最多为 $M$。若全部失败，卡片仍会明确报错。

## 参数说明

### 输出与组成

#### 每个输入的输出数（structures）

`int`，默认 `1`。严格模式下是精确数量；非严格模式下是最大数量。

#### 组成（composition）

`str`，默认空。手动模式下填写整数计数，例如 `Fe:32,O:64`。比例值不接受。

#### 组成模式（composition_mode）

`str`，默认 `input`。`input` 沿用输入的元素种类和计数；`manual` 使用 `composition` 中的整数计数，空表会明确报错。旧 JSON 没有此字段时，非空 `composition` 自动按 `manual` 恢复。

#### 生成原子预算/输入（max_generated_atoms）

`int`，默认 `10000`。限制 `structures × 每个输出原子数`，在随机采样和内存分配前检查。需要扩大规模时可以主动提高。

### 距离与失败合同

#### 全局最小距离（min_distance）

`float`，默认 `1.5 Å`。所有未单独指定的元素对都使用此下限。它只是初始约束，应根据体系中合理短键选择，而不是通用键长。

#### 分元素对最小距离（pair_min_distances）

`str`，默认空。高级表格中的规则覆盖对应元素对的全局阈值，例如 `Fe-O:1.8,O-O:1.2`。

#### 每原子最大尝试次数（max_attempts_per_atom）

`int`，默认 `500`。单个原子达到此尝试次数仍没有合法位置时，该输出失败。持续失败通常意味着晶胞、原子数和距离约束不相容；单纯增大次数只会增加搜索时间。

#### 严格模式（strict_mode）

`bool`，默认 `true`。开启时任一请求样本失败就停止并报错；关闭后跳过失败样本，因此实际输出可能少于请求值。

### 随机性

#### 使用随机种子（use_seed）

`bool`，默认 `false`。开启后，相同输入、参数和 seed 产生相同坐标。

#### 随机种子（seed）

`int`，默认 `0`，仅在 `use_seed=true` 时生效。实际样本 seed 还结合输入结构稳定标识和样本序号；完全相同的重复输入会得到相同结果。

## 操作示例

输入为含 96 个原子的 Fe–O 晶胞，希望生成 20 个可复现的无序初态：

```json
{
  "class": "RandomPackingCard",
  "params": {
    "structures": 20,
    "composition": "",
    "composition_mode": "input",
    "min_distance": 1.5,
    "pair_min_distances": "Fe-O:1.8,O-O:1.2",
    "max_attempts_per_atom": 500,
    "strict_mode": true,
    "use_seed": true,
    "seed": 42,
    "max_generated_atoms": 10000
  }
}
```

预览应显示每个输出 `96` 个原子、每个输入生成 `96×20=1920` 个原子。输出的 `cell` 和 `pbc` 与输入一致，所有原子对满足各自阈值，`Config_type` 追加 `RandPack(...)`。

## 检查结果

- 查看最短原子对距离是否符合全局和元素对阈值。
- 确认输出组成、晶胞、PBC 和请求数量。
- 对无序初态做后续计算时，检查弛豫或动力学是否出现异常重叠和数值崩溃。

## 常见问题

**为什么手动组成必须写整数？** 随机装填需要先建立确定的原子列表；比例到整数的换算应在上游确定晶胞规模后完成。

**为什么提高尝试次数仍可能失败？** 尝试次数只扩大随机搜索，不会增加晶胞可容纳的空间。持续失败时应优先检查原子数、晶胞体积和距离阈值。
