# 检查净力（Check Net Force）

## 作用与适用场景

计算每个结构所有原子力的矢量和 `|ΣF|`，并选中超过阈值的结构。它常用于发现未满足平移不变性、
标签映射错误或 DFT 输出异常的数据。

## 判定原理

对含 $N$ 个原子的结构，程序先求参考力的矢量和，再取欧氏模：

$$
\mathbf F_{\mathrm{net}}=\sum_{i=1}^{N}\mathbf F_i,\qquad
F_{\mathrm{net}}=\left\|\mathbf F_{\mathrm{net}}\right\|_2
=\sqrt{F_x^2+F_y^2+F_z^2}.
$$

当 $F_{\mathrm{net}}>F_{\mathrm{threshold}}$ 时选中该结构。力的单位为 eV/Å，所以净力和阈值
也都是 eV/Å。理想的孤立周期体系满足平移不变性时净力应接近零，但有限 SCF 精度、
约束计算或外场也可能带来真实残差；这个工具是质量复核门槛，不是自动纠错器。

## 参数与执行结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-net-force-start -->
:end-before: <!-- display-net-force-end -->
```

## 操作后检查

阈值单位为 `eV/Å`，必须大于零。选中后结合结构原子数、计算精度和标签来源判断；
工具只标出超阈值结构，不会修改力标签。
