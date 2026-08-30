# 用一条流程学会 生成数据集卡片系统

这不是四张卡片的功能介绍，而是一个完整的工作流示例。目标是从 2 原子的 Si 晶胞出发，同时生成掺杂和空位两类候选，再做小幅扰动和几何筛选。走完一次后，你应该能看懂卡片顺序、`Branch Merge` 分支、启停状态、输入输出数量以及流程复用方式。

## 先认清卡片和右侧检查器

卡片在中间画布中显示顺序、摘要和运行状态；选中卡片后，参数及文档入口出现在右侧检查器：

```{image} ../_static/image/generated/tutorials/card_system_controls.png
:alt: 生成数据集卡片与右侧检查器中的通用操作
:class: docs-screenshot
```

| 编号 | 功能 | 什么时候用 |
| --- | --- | --- |
| 1 | 启用或跳过卡片 | 取消勾选后，运行流程时会跳过这张卡；下游接收上一张已启用卡片的输出 |
| 2 | 展开或收起卡片结构 | 查看分组、分叉或紧凑摘要时使用；普通参数在右侧编辑 |
| 3 | 在线手册 | 打开与软件语言一致的完整参数、限制和示例页 |
| 4 | 卡片信息 | 查看版本、贡献者和来源信息 |
| 5 | 复制单张卡片 JSON | 复制当前卡片和参数，之后可从左侧当前工作流区域粘贴 |
| 6 | 移除卡片 | 从工作区移除；不会删除已经导出的文件 |

运行成功后，卡片上才会出现“查看本卡输出”和“导出本卡输出”。拖动卡片标题栏可以调整顺序。改完顺序必须重新运行，因为顶层卡片的输出会依次传给下一张已启用卡片。

## 组装这个例子

1. 用左上角 `Load` 导入 `Si2.vasp`。
2. 用 `Add new card` 打开卡片选择器，依次添加 `Super Cell`、`Branch Merge`、`Random Doping`、`Vacancy Defect`、`Atomic Perturb` 和 `Geometry Filter`。没有选中容器时，新卡片进入顶层。
3. 按住 `Random Doping` 的标题栏，把它拖进 `Branch Merge` 的空白区域；再用同样方法拖入 `Vacancy Defect`。拖错时，把子卡片拖回工作区空白处即可回到顶层。
4. 拖动标题栏，把剩余顶层卡片排成下面的顺序，再填写参数。

```text
Super Cell
→ Branch Merge [Random Doping | Vacancy Defect]
→ Atomic Perturb
→ Geometry Filter
```

```{image} ../_static/image/generated/tutorials/card_system_workflow.png
:alt: 包含线性步骤和并行分支的完整卡片流程
:class: docs-screenshot
```

中间工具栏的 `Run` 运行所有已启用卡片，`Stop` 中止正在运行的卡片。复制或粘贴整条工作流从左侧“当前工作流”区域操作；右侧检查器的复制按钮只复制当前卡片，两者不要混用。

本例使用的参数只用于把数据流讲清楚：

| 步骤 | 关键输入 | 预期输出 |
| --- | --- | --- |
| `Super Cell` | `2 × 2 × 2` | 1 个 16 原子 Si 超胞 |
| `Random Doping` | 目标 `Si`；掺杂元素直接填 `Ge`；固定替换 1 个；生成 2 个；seed `42` | 2 个 `Si15Ge` |
| `Vacancy Defect` | 固定移除 1 个原子；生成 2 个；seed `42` | 2 个 15 原子 Si 空位结构 |
| `Atomic Perturb` | 最大位移 `0.05 Å`；每个输入生成 2 个；seed `42` | $4\times2=8$ 个扰动结构 |
| `Geometry Filter` | 最短距离 `1.5 Å`；要求有限晶胞 | 筛掉明显碰撞或无效晶胞 |

掺杂元素输入框不是 JSON。单一元素直接写 `Ge`；多元素比例写 `Ge:0.7,C:0.3`。界面恢复已保存参数时也会保持这种写法，便于直接照着修改。

## 线性链和 Branch Merge 不要混淆

顶层卡片是**串行**的：`Super Cell` 的输出进入 `Branch Merge`，分支汇总结果再进入 `Atomic Perturb`，最后交给 `Geometry Filter`。所以扩胞应放在缺陷生成前，筛选应放在随机生成后。

`Branch Merge` 内部是**独立分支**。本例的掺杂卡和空位卡都收到同一个 16 原子 Si 超胞，它们不会前后相接：

```text
                 ┌→ Random Doping ─┐
Si16 → Branch Merge│                ├→ 汇总为 4 个结构
                 └→ Vacancy Defect ┘
```

只有“同一个输入要走几条不同路线”时才使用 `Branch Merge`。如果你想先掺杂、再从掺杂结构中造空位，就应把两张卡放在顶层，按 `Random Doping → Vacancy Defect` 排列，而不是放进同一个组。

## 运行后沿着数量检查

勾选需要参与流程的卡片，点击顶部 `Run`：

```{image} ../_static/image/generated/tutorials/card_system_result.png
:alt: 卡片流程各步骤的真实输入输出数量
:class: docs-screenshot
```

本例真实运行结果为：

```text
1 个 Si2
→ 1 个 Si16
→ 2 个 Si15Ge + 2 个 Si15，共 4 个分支结果
→ 8 个扰动结构
→ 几何筛选保留 8 个
```

每张卡底部都显示 `Input → Output / Time`。数量从哪一步开始不符合预期，就先检查那一步，不要只看最终文件。如果某张卡输出为空，流程会停在这里，下游卡片不会继续运行。

运行完成后有三种查看或导出范围：

- 卡片上的查看、导出按钮：只处理这一张卡片的输出。
- 中间工具栏的 `View output`：合并查看所有已勾选且已经产生结果的卡片，包含中间步骤。
- 页面级导出动作中的 `Export final workflow output`：只保存最后完成的已启用卡片输出；`Export all available card outputs` 用于保留全部中间结果。

## 保存的是结构，还是流程

这两个概念必须分开：

- 左侧 `Save as workflow` 保存卡片顺序、参数和启停状态，**不包含生成结构**。
- 当前工作流区域的粘贴动作会把剪贴板中的单张卡片、卡片列表或完整流程追加到当前工作区，不会自动清空原卡片。
- 更多菜单中的导入、导出动作可用于交换工作流 JSON 文件。
- 导出卡片输出或最终流程输出，保存的才是结构数据。

第一次把流程调通后，建议在工作流库中保存并命名。需要在其他机器使用时再导出 JSON。更换输入结构时先检查元素、原子数和晶胞是否仍满足这些参数，再运行；不要把同一份流程配置当作所有材料都适用的物理方案。
