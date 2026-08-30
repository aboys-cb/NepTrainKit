# 查找最大误差结构（Find Max Error Point）

## 作用与适用场景

从当前图对应的误差指标中选出最大的前 `N` 个结构，用于快速定位最需要复核的标签、
异常构型或训练集缺口。结果依赖当前子图，切换 energy、force 或 virial 后含义不同。

## 排名原理

当前图中 x 轴是 NEP 预测值，y 轴是参考值（通常为 DFT）。对图中第 $i$ 行数据，
程序按当前子图包含的分量计算绝对误差和：

$$
e_i=\sum_{k\in C}\left|y_{ik}-x_{ik}\right|,
$$

其中 $C$ 是当前图实际参与比较的列。例如能量图通常每个结构只有一行；力图中一个结构
会有多个原子行。程序先按 $e_i$ 从大到小排序，再把行映射回结构编号并去重，直到得到
前 $N$ 个不同结构。因此在力图中，它回答的是“至少有一个原子分量误差很大的结构”，
不是按结构平均 RMSE 排名。

## 参数与执行结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-max-error-start -->
:end-before: <!-- display-max-error-end -->
```

## 操作后检查

不要仅凭误差排名删除结构。点开选中的结构，结合 `Config_type`、几何和 DFT 来源判断：
它可能是坏标签，也可能是训练集中真正缺少的重要边界构型。
