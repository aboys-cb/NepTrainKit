# 按图上范围选择（Select by Range）

## 作用与适用场景

按照当前散点图的 x/y 数值范围选中结构。适合截取误差长尾、能量区间或描述符图中的局部点群。
切换到另一张图后，x/y 的物理含义也会随当前坐标轴改变。

## 判定原理

设当前点的横、纵坐标为 $x_i,y_i$，输入范围为
$[x_{\min},x_{\max}]$ 和 $[y_{\min},y_{\max}]$。程序先自动交换写反的上下限，再计算

$$
M_x(i)=(x_{\min}\le x_i\le x_{\max}),\qquad
M_y(i)=(y_{\min}\le y_i\le y_{\max}).
$$

`同时满足（AND）`使用 $M(i)=M_x(i)\land M_y(i)$；`任一满足（OR）`使用
$M(i)=M_x(i)\lor M_y(i)$。边界点会被选中。这里比较的是**当前子图实际显示的数据**，
不是固定的能量或力字段，因此运行前先看清坐标轴名称。

## 参数与执行结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-select-range-start -->
:end-before: <!-- display-select-range-end -->
```

## 操作后检查

范围框默认填入当前图的数据极值。缩小范围后先核对高亮点位置和 `Sel` 数量；
`同时满足（AND）`要求 x、y 同时落入范围，`任一满足（OR）`只要求其中一项满足。
