# 筛选

这两张卡片处理整批结构，而不是逐个生成新构型：

1. 先用 `Geometry Filter` 删除违反明确几何阈值的结构。
2. 必要时在 `NEP Dataset Display` 中继续复核异常结构。
3. 候选池基本干净后，再用 `FPS Filter` 选择代表结构。

```{toctree}
:maxdepth: 1

../cards/geometry-filter-card
../cards/fps-filter-card
```
