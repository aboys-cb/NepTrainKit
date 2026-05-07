功能模块
========

NepTrainKit 的页面可以按数据流理解，而不是按菜单名称死记：

.. list-table::
   :header-rows: 1

   * - 你现在要做什么
     - 进入哪里
     - 典型输出
   * - 制作一批候选训练结构
     - :doc:`make-dataset`
     - ``candidate_pool.xyz``
   * - 清洗候选结构、删除异常样本
     - :doc:`NEP-dataset-display`
     - ``candidate_pool_clean.xyz``
   * - 查每张生成卡片的用途和参数
     - :doc:`make-dataset-cards/index`
     - 可复用的卡片配置
   * - 管理多个项目、模型版本和结果路径
     - :doc:`data-management`
     - 可检索的实验记录
   * - 调整显示、后端和性能行为
     - :doc:`settings`
     - 更适合本机环境的默认行为

如果你还不确定该从哪开始，先看 :doc:`../quickstart`；如果你已经有一批候选结构，
直接看 :doc:`../workflows/clean-candidate-structures`。

.. toctree::
   :maxdepth: 2

   NEP 数据展示 <NEP-dataset-display>
   Show NEP 详细参考 <show-nep-reference>
   数据生成（Make dataset） <make-dataset>
   数据生成卡片手册 <make-dataset-cards/index>
   自定义卡片开发 <custom-card-development>
   数据管理 <data-management>
   设置 <settings>
