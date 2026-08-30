# 浏览与基础选择

对应 `Reset View`、`Pan View`、`Mouse Selection`、`Inverse Selection`、
`Delete Selected Items`、`Undo Selection` 和 `Undo Delete`。

## 这些工具会改变什么

- `Reset View` 和 `Pan View` 只改变当前图的观察范围，不改结构和选择。
- `Mouse Selection`、`Inverse Selection` 和 `Undo Selection` 改变选择集，适合在图上逐步圈出目标结构。
- `Delete Selected Items` 把选中结构移出活动数据集；`Undo Delete` 可以恢复最近一次删除。

选择、标记和删除是三种不同状态。操作后看底部 `Sel / Rej / Rm / Now`，确认变化落在预期状态。

## 按钮行为

```{include} nep-dataset-display-content.md
:start-after: <!-- display-basic-tools-start -->
:end-before: <!-- display-basic-tools-end -->
```

## 操作后检查

先确认选中点数量，再执行删除或导出。误选时优先用 `Undo Selection`，不要通过删除再恢复来代替选择撤销。
