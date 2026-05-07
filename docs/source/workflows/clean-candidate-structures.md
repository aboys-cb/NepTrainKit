# 候选结构清洗后再进入 DFT

这条流程适合处理 `Make Dataset` 生成的大批候选结构。目标不是在本地得到最终训练标签，
而是在 DFT 之前删掉明显不值得计算的结构，避免把计算资源花在坏构型和重复构型上。

一条稳妥路线是：

```text
弛豫好的初始结构
-> Make Dataset 生成候选结构
-> NEP Dataset Display 查看和清洗异常结构
-> FPS Filter 或其他采样方法选择代表结构
-> DFT 标注
-> GPUMD 训练
```

## 1. 从初始结构开始

先准备一批可信的初始结构。它们通常来自已有训练集、弛豫后的晶体、表面模型、缺陷模型或手工构建的原型。

在 `Make Dataset` 页面，先用顶部 `Open` 导入这些初始结构，再添加生成卡片。

```{image} ../_static/image/generated/make_data_empty.png
:alt: Make Dataset input and workspace
:class: docs-screenshot
```

这一步不要跳过。没有初始结构时，大多数构型生成卡没有明确的处理对象；即使某些生成卡能从零创建结构，
也应该单独确认它的输出是否符合你要研究的材料体系。

## 2. 生成候选结构，不要急着 DFT

`Make Dataset` 负责把初始结构扩展成候选池。例如：

- `Lattice Strain` / `Shear Strain`：补应变附近结构。
- `Atomic Perturb` / `Vibration Perturb`：补近平衡扰动。
- `Random Vacancy` / `Vacancy Defect` / `Insert Defect`：补缺陷环境。
- `Random Slab` / `Stacking Fault`：补表面和层错。
- `Random Doping` / `Random Occupancy`：补合金或占位变化。

这些操作会扩大覆盖，但也会引入坏结构。比如扰动过大导致原子距离过近，随机占位产生极端局部环境，
表面或缺陷生成后出现不合理配位。候选结构应该先导出为一个中间文件，例如
`candidate_pool.xyz`，再进入下一步清洗。

## 3. 在 NEP Dataset Display 里先看结构

把 `candidate_pool.xyz` 导入 `NEP Dataset Display`。这一步先做肉眼和几何层面的检查：

- 结构是否明显破碎或重叠。
- 元素组成是否符合预期。
- `Config_type` 是否能追溯到生成来源。
- 是否存在明显过短键长或非物理局部环境。

```{image} ../_static/image/generated/show_nep_overview.png
:alt: NEP Dataset Display candidate inspection
:class: docs-screenshot
```

建议先用这些工具：

- `Finding non-physical structures`：按近邻距离快速找疑似坏构型。
- `Select by Range`：按图上范围选择异常点。
- `Find Max Error Point`：在有参考或预测结果时定位最极端样本。
- 搜索框：按 `Config_type`、元素或表达式筛某一类结构。
- `Delete Selected Items` / `Undo`：删除和撤销。

## 4. 用 NEP 预测做快速预筛

如果你有可用的预筛模型，可以让 `NEP Dataset Display` 给候选结构做快速预测。
这里的预测值不是 DFT 标签，也不能当作训练真值；它只用于发现明显离谱的候选样本。

常见判断包括：

- 某些结构预测力远高于同批样本，大概率有过近原子或极端局部环境。
- 能量分布出现孤立长尾，需要回到结构视图确认。
- 某一类 `Config_type` 系统性异常，说明上游卡片参数可能过激。

如果没有可信预筛模型，也不要强行依赖预测值。先做几何检查和人工抽查，再保守地缩小生成参数。

## 5. 先清洗，再 FPS

`FPS Filter` 的作用是从候选池里选代表结构，不是判断结构是否物理合理。

如果候选池里混入很多坏结构，直接 FPS 会有一个风险：坏结构在描述符空间里往往很“远”，
因此可能被优先选中。结果看起来覆盖更广，实际却把 DFT 预算花在不该算的结构上。

更稳的顺序是：

```text
候选池
-> 删除明显坏结构
-> 删除或隔离预测力极大的结构
-> 检查每类 Config_type 是否还保留足够样本
-> 再用 FPS Filter 做代表性采样
```

清洗后的输出可以命名为 `candidate_pool_clean.xyz`。如果之后做 FPS，输出可以命名为
`candidate_pool_fps.xyz`，这样每一步都能追溯。

## 6. 送去 DFT 后再回看

DFT 标注完成后，把带有能量、力、应力的结构再导入 `NEP Dataset Display`。
这时检查重点会变化：

- 是否有 DFT 未收敛或异常标签。
- 力和能量分布是否出现不合理长尾。
- 某些构型族是否过多或过少。
- 训练后模型在哪些结构上误差最大。

这一步的输出会决定下一轮 `Make Dataset` 应该补什么：是补某类缺陷、某个应变范围，
还是缩小某张生成卡的参数范围。

## 常见误区

**误区 1：生成后直接 DFT。**
候选结构不是最终训练集。先清洗通常比事后处理失败 DFT 更省时间。

**误区 2：先 FPS 再清洗。**
FPS 不知道物理合理性，只知道描述符距离。坏结构可能因为“特别不一样”被选中。

**误区 3：把 NEP 预测当作 DFT 标签。**
NEP 预测只适合预筛和排序，最终训练标签仍然来自 DFT。

**误区 4：只看总数量。**
候选池有 5000 个结构不等于覆盖好。更重要的是每类 `Config_type` 的质量和多样性。
