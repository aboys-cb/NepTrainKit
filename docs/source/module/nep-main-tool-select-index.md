# 按索引选择（Select by Index）

## 作用与适用场景

根据结构编号或切片表达式批量选择结构。适合复现实验记录中的结构编号、从外部异常清单回选结构，
或精确选择不连续的多个区间。这个工具只更新选择集，不会删除或修改结构。

## 参数与示例

```{include} nep-dataset-display-content.md
:start-after: <!-- display-select-index-start -->
:end-before: <!-- display-select-index-end -->
```

## 操作后检查

查看底部 `Sel` 数量是否与表达式预期一致。勾选“使用原始索引”时，编号对应刚导入数据时的索引；
关闭后则对应当前活动数据的行号，删除结构后两者可能不同。
