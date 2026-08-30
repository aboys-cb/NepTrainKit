NepTrainKit 文档
================

NepTrainKit 用于准备、检查和可视化 NEP 训练数据。它负责生成候选结构、检查异常样本、
筛选代表结构和回看训练结果；DFT 标注与 GPUMD 训练仍在对应软件中完成。

从哪里开始
----------

先按你现在遇到的问题选择入口。第一次使用才需要从“快速开始”按顺序阅读；已经打开软件、
只是看不懂某个页面或参数时，直接进入“功能指南”。

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: 第一次安装和使用
      :link: quickstart
      :link-type: doc

      从安装、启动、生成第一批候选结构，一直走到清洗前的检查。

   .. grid-item-card:: 软件里的某个功能不会用
      :link: module/index
      :link-type: doc

      按桌面软件页面顺序查按钮、参数、默认行为和输出。

   .. grid-item-card:: 我已经有候选结构
      :link: workflows/clean-candidate-structures
      :link-type: doc

      在 ``NEP Dataset Display`` 中检查异常，再做代表性采样并送去 DFT。

   .. grid-item-card:: 我想复盘训练结果
      :link: workflows/review-training-results
      :link-type: doc

      从误差和异常结构追到数据缺口，组织下一轮补数据。

   .. grid-item-card:: 软件报错或没有输出
      :link: reference/troubleshooting
      :link-type: doc

      按“导入失败、卡片无输出、CUDA 不可用”等症状直接排查。

   .. grid-item-card:: 看不懂术语或字段
      :link: reference/glossary
      :link-type: doc

      集中查询 ``Config_type``、FPS、GSFE、virial 和 ``spin:R:3``。

按软件页面查功能
----------------

下面的顺序与 NepTrainKit 左侧导航一致。进入功能页后，左侧目录会继续按该页面的界面顺序展开。

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - 软件页面
     - 什么时候查
     - 直接进入
   * - ``NEP Dataset Display``
     - 不会打开数据、看图、筛结构、改数据或导出
     - :doc:`module/NEP-dataset-display`
   * - ``Training Set Audit``
     - 不清楚训练数据覆盖、质量、结构相或磁类型
     - :doc:`module/training-set-assessment`
   * - ``生成数据集``
     - 不知道选哪张卡、参数怎么填或卡片如何串联
     - :doc:`module/make-dataset`
   * - ``Data Management``
     - 不清楚项目、模型版本和结果路径如何管理
     - :doc:`module/data-management`
   * - ``Settings``
     - 需要调整 NEP 后端、绘图、结构显示或更新行为
     - :doc:`module/settings`

理解完整数据流
--------------

NepTrainKit 在训练流程中的位置是：

.. code-block:: text

   生成数据集 生成候选结构
   → NEP Dataset Display 检查并清洗异常结构
   → FPS Filter 或其他方法选择代表结构
   → DFT 标注能量、力、应力
   → GPUMD 训练 NEP
   → NEP Dataset Display 回看误差并开始下一轮

如果你是在完成一整段工作，而不是查询单个按钮，进入 :doc:`workflows/index`。
如果你已经知道具体目标，例如能量平移、最大误差筛选或 DFT-D3 修正，进入
:doc:`example/index`。

安装与引用
----------

安装、运行时后端检查和第一次操作见 :doc:`quickstart`。

如果你的研究使用了 NepTrainKit，请引用：

.. code-block:: bibtex

   @article{CHEN2025109859,
   title = {NepTrain and NepTrainKit: Automated active learning and visualization toolkit for neuroevolution potentials},
   journal = {Computer Physics Communications},
   volume = {317},
   pages = {109859},
   year = {2025},
   issn = {0010-4655},
   doi = {https://doi.org/10.1016/j.cpc.2025.109859},
   url = {https://www.sciencedirect.com/science/article/pii/S0010465525003613},
   author = {Chengbing Chen and Yutong Li and Rui Zhao and Zhoulin Liu and Zheyong Fan and Gang Tang and Zhiyong Wang},
   }

.. toctree::
   :maxdepth: 1
   :caption: 开始使用
   :hidden:

   快速开始 <quickstart>

.. toctree::
   :maxdepth: 2
   :caption: 端到端工作流
   :hidden:

   工作流总览 <workflows/index>

.. toctree::
   :maxdepth: 5
   :caption: 功能指南
   :hidden:

   功能总览 <module/index>
   数据集查看（NEP Dataset Display） <module/NEP-dataset-display>
   训练集评估（Training Set Audit） <module/training-set-assessment>
   生成数据集 <module/make-dataset>
   数据管理（Data Management） <module/data-management>
   设置（Settings） <module/settings>

.. toctree::
   :maxdepth: 3
   :caption: 操作指南
   :hidden:

   按具体任务操作 <example/index>

.. toctree::
   :maxdepth: 2
   :caption: 参考资料
   :hidden:

   参考资料总览 <reference/index>
   支持格式 <formats>

.. toctree::
   :maxdepth: 2
   :caption: 开发者
   :hidden:

   开发者文档总览 <developer/index>
   自定义卡片开发 <module/custom-card-development>
   卡片文档编写规范 <module/make-dataset-cards/writing-guide>
   Python API <api/index>
