# 能量基线平移：去掉巨大的常数能量

能量基线平移最常见的用途，是在训练前去掉 DFT 总能量中的巨大常数项。某些 ABACUS、CP2K 数据按原子归一后仍可能达到数千甚至数万 eV，而模型真正需要学习的是小得多的结构间能量差。直接用 `float32` 训练时，大数会占用有效位数；先把基线移到零附近，可以给这些小能量差留下更多数值精度。

在 `energy` 图上，这类问题通常表现为 DFT 能量和 NEP 预测**整体错开**，但点云的相对形状仍然一致。平移只处理能量零点，不会让模型本身变准；如果散点很散、弯曲，或者只有少数结构偏离，应先检查模型和数据。

## 1. 先看能量图

在 `NEP Dataset Display` 中打开带 DFT 能量的数据和对应的 NEP 模型，然后看 `energy` 散点图。

```{image} ../_static/image/generated/tutorials/energy_baseline_shift_entry.png
:alt: 能量图和能量基线平移按钮
:class: docs-screenshot
```

确认是巨大的整体基线后，点击工具栏中的 `Energy baseline shift`。

> 平移会修改当前数据集里的 `energy`。程序会把第一次平移前的值记在 `energy_original`，但不会自动撤销后续平移。操作前仍建议先导出一份原始数据。

## 2. 按这个例子设置

```{image} ../_static/image/generated/show_nep_reference/g_shift_dialog.png
:alt: 能量基线平移设置窗口
:class: docs-screenshot
```

第一次使用时这样设置即可：

1. `Use existing preset` 保持空白。
2. 如果数据来自同一软件、同一套计算设置，把分组规则填写为 `.*`，让所有 `Config_type` 共用一套基线。
3. `Alignment mode` 选择 `DFT to NEP`。它根据当前 NEP 预测，拟合每种元素的能量基线，并平移数据集中的 DFT 能量。
4. `Max generations`、`Population size` 和 `Convergence tolerance` 先保留默认值。
5. 点击 `OK`，等待能量图重新绘制。

不要仅因为数据里有多个 `Config_type` 就逐组拟合。即使来自同一软件，每组单独拟合也可能把求解过程中的微小偏差吸收到基线里；同一计算来源使用 `.*` 更稳妥。

`Config_type` 分组主要用于区分不同能量基线来源。只有数据混合了不同软件时才拆组，例如 `abacus.*;cp2k.*`，多条规则用英文分号 `;` 分隔。不同泛函、赝势或计算精度带来的差异不只影响能量零点，不要靠平移强行混合。

## 3. 看结果是否正确

平移后回到 `energy` 图：

```{image} ../_static/image/generated/tutorials/energy_baseline_shift_result.png
:alt: 消除整体能量偏移后的散点图
:class: docs-screenshot
```

- 巨大的绝对能量基线被去掉，能量回到更适合训练的数值范围；
- 点云整体靠近对角线，说明主要差异确实来自能量零点；
- 点云的相对顺序和形状应该基本保留；
- 如果仍明显散开，不要继续反复平移，转去检查异常结构、分组规则和模型误差。

能量基线不是给所有结构减去同一个常数。程序会根据结构中的元素个数，减去对应的逐元素基线。因此，包含不同元素或不同原子数的结构也可以放在同一数据集中处理。

确认结果后，用 `Save` 导出新文件，不要覆盖原始训练集。以后需要处理同一类数据时，可以勾选 `Save baseline as preset` 保存本次基线；预设只适用于元素和 `Config_type` 分组方式相符的数据。
