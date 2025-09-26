.. NepTrainKit documentation master file, created by
   sphinx-quickstart on Wed Dec  4 19:50:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NepTrainKit documentation
=========================
NepTrainKit is a toolkit focused on the operation and visualization of neuroevolution potential (NEP) training datasets. It is mainly used to simplify and optimize the NEP model training process, providing an intuitive graphical interface and analysis tools to help users adjust train dataset.

.. note::

   When installing NepTrainKit via pip on Linux, the build auto-detects CUDA. If a compatible CUDA toolkit is available, the NEP backend is compiled with GPU acceleration; otherwise, a CPU-only backend is built.

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
   module    <module/index>
   example    <example/index>
   Changelog <changelog>
