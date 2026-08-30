# 能量基线平移（Energy Baseline Shift）

## 作用与适用场景

按元素计数拟合原子参考能，并从结构总能量中减去对应基线。适合统一不同
`Config_type` 或不同组成数据的能量零点，便于比较形成能附近的变化。

这个操作会修改当前结构的 energy。第一次平移前的数值会写入 `energy_original`；
同一结构再次平移时不会覆盖已经保存的原始能量。

## 拟合与平移原理

设结构 $i$ 的元素计数向量为 $\mathbf n_i$，待拟合的逐元素基线为
$\mathbf\varepsilon_g$，其中 $g$ 是由 `分组规则`匹配到的 `Config_type` 组。
程序用自然进化策略（NES）最小化

$$
\mathcal L_g=
\frac{1}{|g|}\sum_{i\in g}
\left[E_i^{\mathrm{DFT}}-\mathbf n_i^\mathsf T
\mathbf\varepsilon_g-T_i\right]^2,
$$

然后写回

$$
E_i^{\mathrm{new}}=E_i^{\mathrm{DFT}}
-\mathbf n_i^\mathsf T\mathbf\varepsilon_g.
$$

这里 $\varepsilon_{g,e}$ 的单位是 eV/atom，$\mathbf n_i^\mathsf T\mathbf\varepsilon_g$
是结构总能量基线，单位 eV。三种`对齐方式`只改变目标 $T_i$：

| 对齐方式 | 目标 $T_i$ | 适合用途 |
|---|---:|---|
| `参考组` | 当前选中参考结构的平均总能量 | 把不同组对齐到用户指定参考零点 |
| `零基线` | $0$ | 去掉逐元素线性基线，观察剩余相对能量 |
| `DFT 对齐 NEP` | $E_i^{\mathrm{NEP}}$ | 让平移后的 DFT 总能量尽量贴合 NEP 总能量 |

`DFT 对齐 NEP`会把界面中的 NEP 每原子能量乘以原子数后再作为总能量目标。
`最大迭代代数`、`种群大小`和`收敛容差`控制 NES 搜索，不改变上述目标函数。

## 参数与执行结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-energy-shift-start -->
:end-before: <!-- display-energy-shift-end -->
```

## 操作后检查

完成后检查实际平移结构数、未匹配的 `Config_type` 和重绘后的能量分布。导出 `extxyz` 后，
抽查 `energy` 与 `energy_original`，确认新旧能量都可追溯。预设来自其他数据集时尤其要检查分组是否匹配。
