# 生成数据集

`生成数据集` 是候选训练结构的生产台。它把“扩胞、应变、扰动、缺陷、表面、掺杂、磁性设置”
这些操作拆成卡片，让你把一批可信初始结构扩展成候选池。

```{toctree}
:maxdepth: 4
:hidden:

如何选择卡片 <make-dataset-cards/index>
配方示例 <make-dataset-cards/recipes>
晶格 <make-dataset-cards/categories/structure-cell>
结构 <make-dataset-cards/categories/structure>
表面 <make-dataset-cards/categories/surface>
缺陷 <make-dataset-cards/categories/defect-surface>
扰动 <make-dataset-cards/categories/deformation-perturbation>
合金与组分 <make-dataset-cards/categories/composition-alloy>
磁性 <make-dataset-cards/categories/magnetism>
分子与溶剂 <make-dataset-cards/categories/molecule-solvation>
筛选 <make-dataset-cards/categories/filter-sampling>
容器 <make-dataset-cards/categories/workflow-metadata>
```

它的输出通常还不是最终训练集。更稳的路线是：

```text
导入初始结构
-> 用卡片生成候选结构
-> 导出候选池
-> 在 NEP Dataset Display 里清洗异常结构
-> 采样后送去 DFT
```

```{image} ../_static/image/generated/make_data_empty.png
:alt: 生成数据集 workspace
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
Super Cell → Lattice Strain → Atomic Perturb
```

每张卡会处理上一张卡的输出。一般原则是：

- 先改晶胞，再改原子坐标。
- 先做确定性结构变换，再做随机扰动。
- 先生成候选池，再做清洗和采样。

如果同一个输入结构要走多条分支，例如一支做表面、一支做空位，可以用 `Branch Merge`（分支合并）。
组内卡片共享同一输入，输出再汇总。

## 什么时候导出

每张卡运行后都应该先看输出数量。数量符合预期，再导出为中间文件，例如：

```text
candidate_pool.xyz
candidate_pool_clean.xyz
candidate_pool_fps.xyz
```

这样后续发现问题时，可以追溯到底是生成阶段、清洗阶段还是采样阶段引入的。

## 使用工作流库和模板

左侧工作流库分为三类：

- `My workflows`：已经命名并保存的工作流。
- `Built-in`：随软件提供的起始配方，只读。
- `My templates`：你保存或导入的模板。

双击模板只会打开**未修改的预览**，不会立刻算作新建工作流，也不会触发离开前的未保存提示。首次修改参数、卡片或顺序后，状态才变为“基于该模板”的未保存工作流；需要保留时，点击 `Save as workflow` 另存自己的副本。模板始终不会被改写。

第一批内置模板包括晶体应变、超胞原子扰动、合金成分与占位、超胞空位候选和已有磁矩扰动。鼠标停在模板上会显示输入要求。例如，“已有磁矩扰动”只读取结构中已有的 `spin` 或 ASE 初始磁矩，不会为缺失磁矩的元素猜测数值。

工作流和用户模板以独立 JSON 文件保存在用户配置目录的 `workflows/saved` 和 `workflows/templates` 下；它们不存入 `config.sqlite`。内置模板则打包在软件中，不写入用户目录。两类 JSON 都只保存卡片顺序、参数和启停状态，不保存输入结构、运行结果或运行状态。

保存工作流不等于保存生成结构。结构结果仍需从卡片导出。

## 从文档 JSON 直接创建卡片

如果文档里给出的是单张卡片 JSON、卡片数组，或者完整的 `card_config.json`，可以先复制代码块，再回到 `生成数据集` 页面执行 `Load → Paste Card JSON`。

这个入口不会清空当前工作区，只会把剪贴板里的卡片追加到末尾。需要完整替换工作区时，仍然使用 `Load → Import Card Config` 导入保存好的 JSON 文件。
