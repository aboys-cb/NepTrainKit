# 最远点采样：从候选池挑代表结构

最远点采样（FPS）适合从大量相似候选结构中挑出一个更有代表性的子集。它只判断描述符距离，不判断结构是否合理；如果候选池里有原子重叠等坏结构，应先清洗，再做 FPS。

## 1. 打开描述符图

在 `NEP Dataset Display` 中打开清洗后的候选结构和可用于计算描述符的 `nep.txt`。双击 `descriptor` 小图，把它切换到主图区，然后点击 `Sparse samples`。

```{image} ../_static/image/generated/tutorials/fps_sampling_entry.png
:alt: 描述符图和最远点采样入口
:class: docs-screenshot
```

## 2. 第一次这样设置

下面的示例从 25 个同类结构中挑 8 个：

```{image} ../_static/image/generated/show_nep_reference/g_sparse_dialog.png
:alt: 最远点采样设置窗口
:class: docs-screenshot
```

1. `Selection strategy` 选择 `Global FPS`。它适合同一元素集合的候选池。
2. `Sampling mode` 选择 `Fixed count (FPS)`，`Sample limit` 填计划保留的结构数，例如 `8`。
3. `Min distance` 先用当前默认值 `0.01`。描述符尺度取决于模型；如果最终不足 8 个，说明距离条件过严，应调小后重试。
4. `Descriptor source` 选择 `Reduced (PCA)`，其余高级选项先留空。
5. 点击 `OK`。

如果候选池混合了不同元素集合，改用 `Element-set balanced FPS`。它会先给每个元素集合分配名额，因此 `Sample limit` 不能小于元素集合数量。`Training dataset` 只在希望避开已有训练集覆盖区域时填写；第一次从单一候选池采样可以留空。

## 3. 导出选中的代表结构

```{image} ../_static/image/generated/tutorials/fps_sampling_result.png
:alt: 最远点采样选中的八个代表结构
:class: docs-screenshot
```

运行后检查底部状态栏：本例应显示 `Sel: 8`、`Unsel: 17`。图中红色点是选中的代表结构。

确认数量后，从顶部 `Save` 菜单选择 `Export Selected` 导出。不要点击 `Delete selected items`，否则删除的正是刚选出的代表结构。

`Use current selection as region` 会改变选择含义：程序只在当前选区中采样，并把保留下来的代表点从选区中取消。这个选项方便随后删除选区内的冗余结构，但第一次使用时建议不要勾选。
