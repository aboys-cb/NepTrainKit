.. NepTrainKit documentation master file.

NepTrainKit 文档
==================

NepTrainKit 是一个面向 NEP（neuroevolution potential）训练数据管理、可视化与数据生成的工具箱。

.. note::

   使用 ``pip`` 安装时会自动检测 CUDA。若检测到可用 CUDA，将构建 GPU backend；否则构建 CPU backend。
   如需手动指定 CUDA，请在安装前设置 ``CUDA_HOME`` 或 ``CUDA_PATH``。

   Linux/WSL2 示例::

      export CUDA_HOME=/usr/local/cuda-12.4
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
      pip install NepTrainKit

   Windows (PowerShell) 示例::

      $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
      $env:Path = "$env:CUDA_PATH\bin;" + $env:Path
      pip install NepTrainKit

引用 NepTrainKit
-----------------

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
   :maxdepth: 2
   :caption: 文档目录

   快速开始 <quickstart>
   支持格式 <formats>
   功能模块 <module/index>
   示例 <example/index>
   API 参考 <api/index>
