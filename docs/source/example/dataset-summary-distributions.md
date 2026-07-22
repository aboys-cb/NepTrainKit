# 一分钟看懂数据集组成和分布

拿到陌生训练集时，先回答两个问题：数据由哪些组和元素组成，关键物理量是否有异常长尾。`Dataset summary` 给出总体组成，`Explore distributions` 再把具体字段展开。

## 先看总体组成

点击 `Dataset summary`。分组方式跟随顶部当前的搜索模式：选择 `Config_type` 时按配置类型统计，选择化学式时按化学式统计。

```{image} ../_static/image/generated/tutorials/dataset_summary.png
:alt: 数据集摘要窗口
:class: docs-screenshot
```

本例共有 25 个活动结构、6250 个原子，Pb 和 Te 各占 50%，所有结构属于同一个 `Config_type`。先在这里发现结构数、元素比例或分组数量不符合预期，再回头检查合并与导入步骤。

## 再看力等关键字段

打开 `Explore distributions`，选择数据字段、统计范围和分组方式。本例查看 `force`，按元素分组，并用力的模长作横坐标：

```{image} ../_static/image/generated/tutorials/force_distribution.png
:alt: 按元素统计的原子力分布
:class: docs-screenshot
```

结果使用了全部 6250 个原子样本，可以直接比较 Pb、Te 的覆盖范围。还可以切换参考值、预测值或误差，用同样方法检查能量、应力和磁力等字段。

分布图只负责暴露覆盖和异常，不替你判断数据好坏。发现孤立尖峰、异常长尾或某个组明显偏移后，再回到主界面选中对应结构，结合几何和原始计算结果复核。
