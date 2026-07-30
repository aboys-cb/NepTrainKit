# 查看数据分布（Explore distributions）

## 作用与适用场景

在统一的训练集评估页面中查看能量、力、virial、预测值或误差分布，并从分箱反向选择结构。
适合先看整体长尾，再追到对应结构，而不是逐帧浏览。

## 可选字段与联动

```{include} nep-dataset-display-content.md
:start-after: <!-- display-distributions-start -->
:end-before: <!-- display-distributions-end -->
```

## 操作后检查

确认当前字段是否真的提供 Reference、Prediction 或 Error。点击分箱后检查回写的结构数量；
分布只描述当前活动数据，已经删除的结构不在统计范围内。
