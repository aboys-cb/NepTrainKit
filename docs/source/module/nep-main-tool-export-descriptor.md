# 导出结构描述符（Export structure descriptor）

界面按钮：`导出结构描述符`。

## 作用与适用场景

把**当前选中结构**的描述符导出为文本文件，供外部降维、聚类或覆盖分析使用。先选中结构再点击；
没有选中任何结构时只会提示，不会生成文件。这个操作不会改变当前数据、选择集或模型。

完整的分类导出和绘图流程见[描述符分布绘图示例](../example/Plot-descriptor.md)。

## 输出

```{include} nep-dataset-display-content.md
:start-after: <!-- display-export-descriptor-start -->
:end-before: <!-- display-export-descriptor-end -->
```

## 操作后检查

等待后台任务结束，再确认目标文件存在且行数等于选中的结构数。描述符来自当前活动模型；
切换 `nep.txt` 后重新导出，不能把不同模型产生的描述符直接当作同一坐标空间。
