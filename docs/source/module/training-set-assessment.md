# 训练集评估（Training Set Audit）

训练集评估用于检查当前训练数据：先找会阻止训练的技术问题，再查看组分、标签、
局域环境、结构相和磁类型如何分布。每一项结果都能回到
`NEP Dataset Display` 中的原始结构。

```{toctree}
:maxdepth: 1
:hidden:

training-audit-overview
training-audit-data-map
training-audit-advanced
training-audit-review-queue
training-audit-target-model
training-audit-phase
training-audit-magnetism
training-audit-report
```

## 从数据集查看进入

先在 `NEP Dataset Display` 中打开训练数据，再使用以下任一入口：

| 入口 | 打开位置 | 适合做什么 |
| --- | --- | --- |
| 顶部 `Save` 分裂菜单 → `评估当前数据集` | 「概览」 | 完整检查当前活动结构 |
| 左侧图表工具栏 → `查看数据分布` | 「数据地图」→「数据分布」 | 查看能量、力、virial、预测值或误差分布 |

评估默认只读取当前活动结构；已删除结构不在本次范围内。删除结构、修改标签或切换
`nep.txt` 后，应点击「重新检查」，旧结果不能继续代表变化后的数据。

## 按页面标签阅读

1. [概览与快速体检](training-audit-overview.md)：先处理阻塞项，再看建议复核项。
2. [数据地图](training-audit-data-map.md)：查看精确组分、结构相和磁类型分布。
3. [进阶证据](training-audit-advanced.md)：结合活动模型的 cutoff 检查局域环境。
4. [复核队列](training-audit-review-queue.md)：记录每类问题的人工判断。
5. [目标与模型](training-audit-target-model.md)：有明确目标空间时再比较覆盖规则。
6. [结构相识别](training-audit-phase.md)和[磁类型识别](training-audit-magnetism.md)：核对判据与能力边界。
7. [导出报告与当前限制](training-audit-report.md)：留档、协作以及判断本页面不能回答的问题。

这个页面给出的是**当前数据快照的证据**，不会仅凭一份训练集宣称物理空间已经完整覆盖，
也不会在缺少独立测试数据时判断势函数是否可靠。
