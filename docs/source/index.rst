.. NepTrainKit documentation master file, created by
   sphinx-quickstart on Wed Dec  4 19:50:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NepTrainKit documentation
=========================
NepTrainKit is a toolkit focused on the operation and visualization of neuroevolution potential (NEP) training datasets. It is mainly used to simplify and optimize the NEP model training process, providing an intuitive graphical interface and analysis tools to help users adjust train dataset.

.. note::

   Installing via ``pip`` auto-detects CUDA. If a compatible CUDA toolkit
   is available, the NEP backend is compiled with GPU acceleration; otherwise,
   a CPU-only backend is built. If CUDA is not detected automatically, set
   one of ``CUDA_HOME`` or ``CUDA_PATH`` and update your loader path before running
   ``pip install``.

   Linux/WSL2 example::

      export CUDA_HOME=/usr/local/cuda-12.4
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
      pip install NepTrainKit

   Windows (PowerShell) example::

      $env:CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
      $env:Path = "$env:CUDA_PATH\\bin;" + $env:Path
      pip install NepTrainKit

Citing NepTrainKit
-------------------
If you rely on NepTrainKit for published research, please cite the following article and acknowledge the upstream NEP projects where appropriate:

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
   :caption: Documentation:

   Quickstart <quickstart>
   Supported Formats <formats>
   Module    <module/index>
   Example    <example/index>
   API Reference <api/index>
   Changelog <changelog>
