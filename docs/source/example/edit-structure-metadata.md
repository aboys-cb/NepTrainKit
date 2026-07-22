# 批量整理结构元数据

`Config_type` 和自定义标签常用于记录数据来源、计算阶段或复核状态。需要批量修正时，先选中目标结构，再使用 `Edit info`，不要为了改几个标签重新拆分和合并文件。

## 先限定修改范围

本例先选中前 3 个结构，然后点击工具栏中的 `Edit info`：

```{image} ../_static/image/generated/tutorials/edit_metadata_entry.png
:alt: 编辑选中结构元数据的入口
:class: docs-screenshot
```

弹窗顶部列出选中结构共有的标签：

```{image} ../_static/image/generated/show_nep_reference/g_editinfo_dialog.png
:alt: 编辑结构元数据窗口
:class: docs-screenshot
```

可以直接增加 `review_status=checked` 这类追踪标签。若要替换 `Config_type`，先删除旧的 `Config_type`，再增加新的值，例如 `Config_type=reviewed_bulk`。确认后，同一组修改会应用到所有选中结构。

## 立即核对结果

```{image} ../_static/image/generated/tutorials/edit_metadata_result.png
:alt: 三个结构的 Config_type 已批量更新
:class: docs-screenshot
```

右侧信息栏应显示新值，底部仍应是 `Sel: 3`。元数据编辑没有独立撤销操作，批量修改前最好保留原文件。

`Config_type` 应表示真正有意义的分组，例如不同计算软件或明确的数据阶段。不要给每个结构随意创建一个新组；在能量基线平移等操作中，过细分组反而会引入不稳定的独立基线。
