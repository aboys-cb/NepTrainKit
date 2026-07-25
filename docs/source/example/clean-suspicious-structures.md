# 快速找出短键和净力异常结构

训练前先做一次结构体检，通常能提前发现原子重叠、周期边界错误或未收敛的力。这里用两个互补入口建立“待复核列表”，程序不会自动删除任何结构。

## 运行两项检查

在工具栏中依次使用 `Find non-physical structures` 和 `Check net force`：

```{image} ../_static/image/generated/tutorials/structure_quality_checks.png
:alt: 非物理距离和净力检查入口
:class: docs-screenshot
```

短键检查使用 `Settings` 中的半径系数。默认 `0.7` 表示：两原子距离小于其共价半径之和的 `0.7` 倍时，结构会被选中。它适合发现明显碰撞，不是材料通用的成键判据。

净力检查在弹窗中填写阈值，单位是 `eV/Å`：

```{image} ../_static/image/generated/show_nep_reference/g_force_dialog.png
:alt: 净力阈值设置窗口
:class: docs-screenshot
```

程序计算每个结构的 $|\sum_i \mathbf{F}_i|$。本例用 `0.1 eV/Å` 做演示；真实阈值应结合计算精度和体系大小决定。

## 把选中结果当作复核清单

```{image} ../_static/image/generated/tutorials/structure_quality_result.png
:alt: 两项检查选出的可疑结构
:class: docs-screenshot
```

演示数据中人为加入了一个原子重叠和一个净力异常，真实检查后共选中 2 个结构。请逐个查看几何、周期边界和对应 DFT 输出，确认是计算失败还是合理的高能构型。建议先导出选中结构留档，再删除确认无效的数据；如果误删，可以立即使用撤销删除。
