功能指南
========

这里按 NepTrainKit 左侧页面顺序回答“我正在这个页面，具体该查哪一项”。首页负责选择
大方向；本页直接给出每个页面最常用的子入口。

NEP Dataset Display
-------------------

用于打开结构和训练输出、查看散点图、定位结构、清洗数据和导出子集。

* 不会打开结构、``nep.txt`` 或训练输出：:doc:`nep-display-open-data`
* 不认识图上工具或不清楚 x/y 含义：:doc:`nep-display-main-plot-tools`
* 想按标签、元素或表达式筛选：:doc:`nep-display-filter`
* 想标记、删除、恢复或导出结构：:doc:`nep-display-structure-tools`
* 页面没图、数量不对或工具不可用：:doc:`nep-display-status-and-errors`
* 先看完整页面逻辑：:doc:`NEP-dataset-display`

训练集评估（Training Set Audit）
----------------------------------

用于检查当前训练数据快照的标签质量、组成、结构相、磁类型和需要人工复核的问题。

* 第一次进入，先处理阻塞项：:doc:`training-audit-overview`
* 查看组分、标签范围和数据分布：:doc:`training-audit-data-map`
* 检查局域环境证据：:doc:`training-audit-advanced`
* 核对结构相或磁类型判据：:doc:`training-audit-phase` /
  :doc:`training-audit-magnetism`
* 记录人工判断并导出结果：:doc:`training-audit-review-queue` /
  :doc:`training-audit-report`
* 先理解整页阅读顺序：:doc:`training-set-assessment`

生成数据集
----------

用于从基础结构生成应变、扰动、合金、缺陷、表面、分子或磁性候选构型。

* 不认识顶部按钮和卡片运行顺序：:doc:`make-dataset`
* 不知道该选哪张卡：:doc:`make-dataset-cards/index`
* 已经选定卡片，要查参数和原理：从 :doc:`make-dataset-cards/index`
  进入对应分类和具体卡片
* 想直接参考完整多卡流程：:doc:`make-dataset-cards/recipes`

数据管理（Data Management）
---------------------------

用于记录项目、模型、数据和输出路径之间的关系。先看
:doc:`data-management` 中的推荐记录方式、常用操作和存储位置。它不替代结构文件本身，
也不会自动判断哪一版模型更可靠。

设置（Settings）
----------------

用于调整 NEP 后端、绘图、结构显示、NEP89 和更新行为。进入 :doc:`settings` 后，
先确认设置影响的是计算、显示还是更新；遇到 CUDA 不可用时先看“NEP 后端”，不要只反复切换
下拉框。

还不知道从哪里开始时，看 :doc:`../quickstart`；已经有候选结构准备送 DFT 时，看
:doc:`../workflows/clean-candidate-structures`。术语不清楚或遇到报错时，直接进入
:doc:`../reference/glossary` 或 :doc:`../reference/troubleshooting`。
