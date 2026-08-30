# DFT-D3 修正

界面按钮：`DFT D3`。

## 作用与适用场景

计算 DFT-D3 色散修正，并按所选模式加到或从当前能量、力和 virial 中扣除。
适合统一训练数据是否包含 D3 的标签约定，不是用于随意改善散点图。

:::{note}
DFT-D3 计算仅支持 CPU。即使 NEP 后端选择了 GPU，执行这项操作时也会改用 CPU。
:::

## 修正原理

DFT-D3 用元素、原子间距和配位环境计算经验色散能 $E_{\mathrm{D3}}$，并由其对坐标和
应变的导数得到力 $\mathbf F_{\mathrm{D3}}$ 与 virial
$\mathbf W_{\mathrm{D3}}$。本工具不重新做 DFT，只对已有标签执行一致的加减：

$$
\begin{aligned}
E' &= E+sE_{\mathrm{D3}},\\
\mathbf F'_i &= \mathbf F_i+s\mathbf F_{\mathrm{D3},i},\\
\mathbf W' &= \mathbf W+s\mathbf W_{\mathrm{D3}},
\end{aligned}
\qquad
s=\begin{cases}+1,&\text{加上}\\-1,&\text{减去。}\end{cases}
$$

`交换-相关泛函`决定 D3 阻尼参数，必须与原始 DFT 泛函一致；`D3 截断半径`控制色散原子对，
`配位数截断半径`控制配位环境计算，两者单位均为 Å。截断半径不是“精度越大越好”的普通滑块：
增大会提高计算量，也可能改变与原始标签约定的一致性。

## 参数与执行结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-dft-d3-start -->
:end-before: <!-- display-dft-d3-end -->
```

## 操作后检查

`functional` 必须与原始 DFT 计算采用的泛函一致。完成后比较修正前后的能量和力分布，
并确认 `Add/Subtract` 与目标标签约定一致；方向选反会造成整批数据系统性偏移。
