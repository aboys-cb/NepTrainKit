# Make Dataset

Make Dataset 是一个基于 cards 的结构生成与过滤流水线编辑器。

## 数据流

- 线性链：上游输出传递给下游
- Card Group：组内共享同一输入并汇合输出
- Filter：可全局或组内筛选

## 推荐流程

1. 导入结构（XYZ/POSCAR/CIF）
2. 构建 pipeline
3. 保存/加载 card JSON
4. 导出 `make_dataset.xyz`

:::{tip}
如果后续需要做最远点采样（FPS），建议先导出 `xyz`，在第一个模块用 `nep89` 做预测清洗并删除不合理结构，再执行 FPS。
:::

## 文档入口

- [Make Dataset 卡片手册](make-dataset-cards/index.md)
- [Make Dataset 配方示例](make-dataset-cards/recipes.md)
