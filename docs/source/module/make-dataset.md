# Make Dataset

`Make Dataset` 是候选训练结构的生产台。它把“扩胞、应变、扰动、缺陷、表面、掺杂、磁性设置”
这些操作拆成卡片，让你把一批可信初始结构扩展成候选池。

它的输出通常还不是最终训练集。更稳的路线是：

```text
导入初始结构
-> 用卡片生成候选结构
-> 导出候选池
-> 在 NEP Dataset Display 里清洗异常结构
-> 采样后送去 DFT
```

```{image} ../_static/image/generated/make_data_empty.png
:alt: Make Dataset workspace
:class: docs-screenshot
```

## 基本操作顺序

1. 用顶部 `Open` 导入初始结构。
2. 点击 `Add new card` 添加生成或筛选卡片。
3. 展开卡片，只设置与当前物理目标直接相关的参数。
4. 勾选要运行的卡片。
5. 点击 `Run`。
6. 在卡片上检查输出数量并导出结果。

不要跳过第一步。多数卡片需要已有结构作为输入；如果你想从晶体原型从零生成结构，
应该使用对应的生成型卡片，并单独检查输出是否符合材料体系。

## 卡片怎么串

最常见的是线性链：

```text
Super Cell -> Lattice Strain -> Atomic Perturb
```

每张卡会处理上一张卡的输出。一般原则是：

- 先改晶胞，再改原子坐标。
- 先做确定性结构变换，再做随机扰动。
- 先生成候选池，再做清洗和采样。

如果同一个输入结构要走多条分支，例如一支做表面、一支做空位，可以用 `Card Group`。
组内卡片共享同一输入，输出再汇总。

## 第一次选哪张卡

| 目标 | 先看这些卡 |
| --- | --- |
| 扩大晶胞 | `Super Cell` |
| 补弹性响应 | `Lattice Strain` / `Shear Matrix Strain` / `Shear Angle Strain` |
| 补近平衡扰动 | `Atomic Perturb` / `Vibration Perturb` |
| 做缺陷 | `Random Vacancy` / `Vacancy Defect` / `Insert Defect` |
| 做表面或层错 | `Random Slab` / `Stacking Fault` |
| 做合金或占位变化 | `Random Doping` / `Random Occupancy` / `Composition Sweep` |
| 做磁性构型 | `Magnetic Order` / `Set Magnetic Moments` / `Spin Spiral` |

完整选择表见 [Make Dataset 卡片手册](make-dataset-cards/index.md)。

## 什么时候导出

每张卡运行后都应该先看输出数量。数量符合预期，再导出为中间文件，例如：

```text
candidate_pool.xyz
candidate_pool_clean.xyz
candidate_pool_fps.xyz
```

这样后续发现问题时，可以追溯到底是生成阶段、清洗阶段还是采样阶段引入的。

## FPS Filter 放在哪里

`FPS Filter` 适合在候选结构已经基本干净后做代表性采样。它不负责判断结构是否物理合理。

如果候选池来自强扰动、随机缺陷、表面切片或随机占位，建议先导出候选结构，
到 `NEP Dataset Display` 里清洗，再回来或继续使用 `FPS Filter` 采样。

完整流程见 [候选结构清洗后再进入 DFT](../workflows/clean-candidate-structures.md)。

## 保存和恢复工作区

顶部 `Save` / `Load` 会把当前工作区的卡片顺序、参数和启停状态保存成 JSON。
这适合保存一条可复用的生成方案，例如“某个合金体系的缺陷候选池生成流程”。

保存工作区不等于保存生成结构。结构结果仍需要从卡片导出。

## 文档入口

- [快速开始](../quickstart.md)：从安装到生成第一批候选结构。
- [卡片手册](make-dataset-cards/index.md)：按目标选卡、易混卡片对比、每张卡的完整参数说明。
- [配方示例](make-dataset-cards/recipes.md)：多卡组合示例。
- [自定义卡片开发](custom-card-development.md)：把已有脚本封装成 Make Dataset 卡片。

## 从文档 JSON 直接创建卡片

如果文档里给出的是单张卡片 JSON、卡片数组，或者完整的 `card_config.json`，可以先复制代码块，再回到 `Make Dataset` 页面执行 `Load -> Paste Card JSON`。

这个入口不会清空当前工作区，只会把剪贴板里的卡片追加到末尾。需要完整替换工作区时，仍然使用 `Load -> Import Card Config` 导入保存好的 JSON 文件。
