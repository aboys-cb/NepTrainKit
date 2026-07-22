# DFT-D3：统一数据中的色散校正口径

这个功能直接修改当前数据集的 DFT 参考标签，用于加上或减去 DFT-D3 色散校正。只有明确知道原始数据是否包含 D3、使用了什么交换关联泛函时才运行；不确定时不要靠散点图位置猜。

| 当前数据 | 目标口径 | 选择 |
| --- | --- | --- |
| 不含 D3 | 改成包含 D3 | `Add DFT-D3` |
| 已含 D3 | 改成不含 D3 | `Subtract DFT-D3` |

## 1. 先保存原始数据

在 `NEP Dataset Display` 中打开数据和对应的 `nep.txt`，确认当前标签来源与泛函，然后先导出一份原始数据。DFT-D3 会修改 energy、force 以及已有的 virial/stress 标签，而且重复运行会继续累加，没有单独的撤销按钮。

```{image} ../_static/image/generated/tutorials/dft_d3_entry.png
:alt: 能量图和 DFT-D3 校正入口
:class: docs-screenshot
```

点击工具栏中的 `DFT D3`。该计算固定使用 CPU，与当前选择的 NEP 后端无关。

## 2. 泛函必须与原计算一致

```{image} ../_static/image/generated/show_nep_reference/g_dftd3_dialog.png
:alt: DFT-D3 校正设置窗口
:class: docs-screenshot
```

图中 `pbe` 只对应 PBE 数据：

1. 第一项填写原始 DFT 计算使用的泛函，例如 `pbe`、`scan` 或 `pbesol`。
2. `D3 cutoff` 和 `D3 cutoff_cn` 第一次先保留当前默认值；只有已有计算规范明确要求时才修改。
3. 按目标选择 `Add DFT-D3` 或 `Subtract DFT-D3`。
4. 点击 `OK`，等待全部结构处理完成。

不要因为模型使用了某种泛函就填写那个名称；这里应与当前 DFT 标签的原始计算口径一致。

## 3. 看标签是否按预期改变

```{image} ../_static/image/generated/tutorials/dft_d3_result.png
:alt: 加上 PBE DFT-D3 后的能量图
:class: docs-screenshot
```

本例真实加上了 PBE DFT-D3，因此参考能量、力和 virial 发生变化，NEP 预测本身没有改变。散点更靠近或更远离对角线都不能单独证明操作正确；关键是模式、泛函和原数据口径是否匹配。

检查完成后用新文件名导出，例如 `train_pbe_d3.xyz`。不要覆盖无法重新获得的原始标签。
