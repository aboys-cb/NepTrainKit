# 编辑结构信息（Edit Info）

## 作用与适用场景

批量修改当前选中结构的 metadata，例如补充 `Config_type`、重命名旧字段或删除错误标签。
它只修改结构信息，不改变元素、坐标、晶胞和数值标签。

## 可执行操作

```{include} nep-dataset-display-content.md
:start-after: <!-- display-edit-info-start -->
:end-before: <!-- display-edit-info-end -->
```

## 操作后检查

应用前确认窗口会列出新增、删除和重命名摘要。执行后在结构信息区抽查一个选中结构，
再导出数据；未导出前的修改只存在于当前会话数据中。
