# NEP Dataset Display

`NEP Dataset Display` 是用来查看、筛选和导出结构数据的页面。它有两类常见用途：

- **DFT 之前**：把 `Make Dataset` 生成的候选结构导入进来，先删掉明显异常结构，再采样去 DFT。
- **训练之后**：导入训练结构和模型输出，找误差大的结构，判断下一轮数据应该补哪里。

它不是训练器，也不替代 DFT。这里的重点是把结构和图上的异常点连起来，让你能快速决定：
保留、删除、导出，还是回到上游重新生成。

```{image} ../_static/image/generated/show_nep_overview.png
:alt: NEP Dataset Display overview
:class: docs-screenshot
```

## 数据从哪里来

可以导入三类数据：

| 输入 | 什么时候用 | 你会看到什么 |
| --- | --- | --- |
| 候选结构 `xyz` / `extxyz` | `Make Dataset` 刚生成结构后 | 结构视图、组成、标签、预测或几何异常 |
| 训练结构 + `nep.txt` | 想用模型快速预筛或回看预测 | 能量、力、应力等预测结果 |
| 训练输出目录 | GPUMD 训练结束后 | 预测-标签散点、误差最大结构、可导出的子集 |

入口是顶部 `Open`。如果只是看一批候选结构，直接打开 `xyz` 文件；如果要回看完整训练结果，
打开包含训练输出的目录或对应文件。

## 第一次应该看哪里

先看三个区域：

1. **误差图或分布图**：先找离群点，不要一上来逐个翻结构。
2. **结构视图**：点中图上的结构后，看右侧是否有过近原子、不合理配位、奇怪表面或异常磁矩。
3. **状态行**：关注 `Orig / Now / Rm / Sel / Unsel / Rej`，确认当前删掉、选中和保留的数量。

如果你正在处理 `Make Dataset` 的候选结构，建议先看完整流程：
[候选结构清洗后再进入 DFT](../workflows/clean-candidate-structures.md)。

## 按目的选工具

| 目的 | 推荐入口 | 结果 |
| --- | --- | --- |
| 找最极端的异常点 | `Find Max Error Point` | 选中误差最大的前 N 个结构 |
| 按图上范围选一批点 | `Select by Range` | 选中落在指定坐标范围内的结构 |
| 找过近原子或疑似坏构型 | `Finding non-physical structures` | 选中疑似非物理结构 |
| 检查净力是否异常 | `Check Net Force` | 选中净力超过阈值的结构 |
| 按来源筛结构 | 搜索框 `tag` 模式 | 按 `Config_type` 找某类生成结构 |
| 按元素筛结构 | 搜索框 `elements` 模式 | 找含有或不含某些元素的结构 |
| 删除当前选中结构 | `Delete Selected Items` | 从当前数据集中移除选中结构 |
| 后悔删除 | `Undo` | 恢复最近一次删除 |
| 导出干净子集 | 顶部 `Save` / 导出菜单 | 保存后进入 DFT、FPS 或训练流程 |

如果你不知道按钮图标对应哪个功能，查
[Show NEP 详细参考](show-nep-reference.md)。参考页按按钮列出了弹窗参数和执行结果。

## 候选结构清洗时怎么用

候选结构清洗的重点不是追求预测精度，而是先剔除明显坏样本：

1. 打开 `Make Dataset` 导出的候选结构。
2. 用结构视图和 `Finding non-physical structures` 找几何异常。
3. 如果有可用 NEP 模型，再看预测力、能量或分布长尾。
4. 选中异常结构后删除，或单独导出做记录。
5. 导出清洗后的候选池，再做 FPS 或其他代表性采样。

这里要避免一个顺序错误：不要把未清洗的候选池直接交给 FPS。FPS 只看描述符距离，
坏结构可能因为“离得远”被选中。

## 训练结果回看时怎么用

训练结束后，这个页面更像诊断台：

1. 打开训练输出。
2. 先看 energy / force / stress / virial 的散点图。
3. 用 `Find Max Error Point` 找误差最大的结构。
4. 在右侧查看结构和 `Config_type`，判断误差来源。
5. 导出这些结构，作为下一轮补数据或排查 DFT 标签的入口。

判断时不要只看一个总 RMSE。更有用的是确认误差集中在哪类结构：
缺陷、表面、应变、高能构型、某个元素组合，还是某个生成卡片的输出。

## 搜索模式

- `tag`：基于 `Config_type` 正则匹配。
- `formula`：基于化学式正则匹配。
- `elements`：元素集合语法，例如 `+Fe -O`。
- `expression`：基于结构级表达式筛选，支持 `natoms`、元素统计、能量、力、应力、virial 和 `atomic.<name>`。

`expression` 的完整语法、字段规则和示例见
[Show NEP 详细参考](show-nep-reference.md)。
