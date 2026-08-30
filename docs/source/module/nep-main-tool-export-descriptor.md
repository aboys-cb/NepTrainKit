# 导出结构描述符

界面按钮：`Export structure descriptor`。

## 作用与适用场景

把当前数据对应的结构描述符导出为文本文件，供外部降维、聚类或覆盖分析使用。
这个操作不会改变当前数据、选择集或模型。

## 输出

```{include} nep-dataset-display-content.md
:start-after: <!-- display-export-descriptor-start -->
:end-before: <!-- display-export-descriptor-end -->
```

## 操作后检查

等待后台任务结束，再确认目标文件存在且行数与当前导出范围一致。描述符来自当前活动模型；
切换 `nep.txt` 后重新导出，不能把不同模型产生的描述符直接当作同一坐标空间。
