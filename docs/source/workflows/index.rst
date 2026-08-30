工作流
======

这里的页面按真实任务组织，而不是按按钮组织。按钮细节可以查模块参考；工作流页只回答：
数据从哪里来，在哪一步该判断什么，最后输出给谁。

三条工作流分别对应“训练前、训练后和跨轮次管理”：

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: 训练前：清洗候选结构
      :link: clean-candidate-structures
      :link-type: doc

      已经生成候选池，但还没有送 DFT。先排除明显坏结构，再选择代表样本。
      输出通常是 ``candidate_pool_clean.xyz`` 或待标注子集。

   .. grid-item-card:: 训练后：从误差找缺口
      :link: review-training-results
      :link-type: doc

      已经有训练／测试结果。把最大误差和异常点映射回结构，判断下一轮该补什么数据。
      输出通常是复核结构集和下一轮生成目标。

   .. grid-item-card:: 多轮迭代：管理版本
      :link: manage-iterations
      :link-type: doc

      已经开始第二轮及以后。固定数据、模型和报告的对应关系，避免覆盖上一轮结果。
      输出是可追溯的轮次目录和记录。

它们在完整流程中的关系是：

.. code-block:: text

   生成候选结构
        │
        ▼
   清洗候选结构 ──→ DFT 标注 ──→ GPUMD 训练
                                      │
                                      ▼
                              从训练误差定位缺口
                                      │
                                      └──→ 下一轮候选结构

   “组织多轮数据和模型版本”贯穿每一轮，记录输入、输出和模型对应关系。

如果你只是看不懂某个按钮或参数，不必从工作流开始，直接进入 :doc:`../module/index`。
如果你要完成一次具体的单步操作，例如能量平移或按索引选结构，进入
:doc:`../example/index`。

.. toctree::
   :maxdepth: 1
   :hidden:

   候选结构清洗后再进入 DFT <clean-candidate-structures>
   从训练误差定位下一轮数据缺口 <review-training-results>
   组织多轮数据和模型版本 <manage-iterations>
