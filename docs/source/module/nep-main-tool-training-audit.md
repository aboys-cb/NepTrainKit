# 进入训练集评估（Training Set Audit）

## 作用与适用场景

把当前活动结构送入训练集评估页面，检查阻塞项、组分、标签、局域环境、结构相和磁类型。
这个入口不复制数据，也不会自动删除或修改结构。

## 评估范围与结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-training-audit-start -->
:end-before: <!-- display-training-audit-end -->
```

## 操作后检查

先看概览中的训练阻塞项，再进入数据地图和复核队列。点击图表或复核项可以把真实结构索引
回选到数据集查看页面；数据变化后应重新检查，不能继续使用旧快照。
