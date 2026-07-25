# 按索引精确选中一批结构

已经从日志、训练结果或外部脚本拿到结构编号时，不必在散点图里逐个点选。`Select by Index` 可以一次选中离散编号和连续切片，适合复核、导出或删除一批已知结构。

## 输入编号

打开数据后，点击工具栏中的 `Select by Index`。

```{image} ../_static/image/generated/tutorials/select_by_index_entry.png
:alt: 按索引选择结构的入口
:class: docs-screenshot
```

在弹窗中输入 `0, 5, 10:13`：

```{image} ../_static/image/generated/show_nep_reference/g_index_dialog.png
:alt: 按索引选择结构设置
:class: docs-screenshot
```

这里的编号从 `0` 开始，切片右端不包含在内，因此实际选中 `0、5、10、11、12`，共 5 个结构。逗号用于分隔多段输入，`10:13` 的写法与 Python 切片一致。

一般保持 `Use original indices` 勾选。这样即使之前筛选或删除过结构，编号仍对应最初载入文件时的顺序；取消勾选后，编号指当前活动结构列表中的位置。

## 核对再处理

```{image} ../_static/image/generated/tutorials/select_by_index_result.png
:alt: 按索引选中五个结构后的界面
:class: docs-screenshot
```

确认底部状态栏显示 `Sel: 5`，再用右下角的结构编号逐个查看。后续可以只导出选中结构，也可以删除或继续分析。若编号来自另一个文件，先确认两个文件的结构顺序一致；仅凭编号相同不能说明结构相同。
