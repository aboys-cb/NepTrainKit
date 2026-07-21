<div align="center">
<a href="https://github.com/aboys-cb/NepTrainKit">
  <img src="./src/NepTrainKit/src/images/logo.png" width="25%" alt="NepTrainKit logo">
</a><br>
<a href="https://pypi.org/project/NepTrainKit"><img src="https://img.shields.io/pypi/dm/NepTrainKit?logo=pypi&logoColor=white&color=blue&label=PyPI" alt="PyPI downloads"></a>
<a href="https://python.org/downloads"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python version"></a>
<a href="https://codecov.io/github/aboys-cb/NepTrainKit"><img src="https://codecov.io/github/aboys-cb/NepTrainKit/graph/badge.svg?token=HQ5FMLD91F" alt="Codecov"></a>
<a href="https://github.com/aboys-cb/NepTrainKit/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue" alt="License"></a>
<br><br>
<strong>English</strong> | <a href="./README.zh-CN.md">简体中文</a>
</div>

# NepTrainKit

NepTrainKit is a desktop application for preparing, auditing, and visualizing training datasets for neuroevolution potentials (NEPs). It complements, rather than replaces, long-running GPUMD training or DFT calculations. Its focus is the repetitive work around those calculations: generating candidate structures, cleaning problematic samples, selecting representative configurations, and passing a well-prepared dataset back into the DFT and GPUMD workflow.

## What you can do

- **Make Dataset**: generate strained, perturbed, defective, surface, doped, magnetic, and solvated candidate structures with composable cards.
- **NEP Dataset Display**: inspect structures, errors, and distributions; remove problematic samples; and export a clean subset.
- **Representative selection**: reduce a candidate pool to a smaller, more representative set with methods such as farthest point sampling (FPS).
- **Training-result inspection**: load NEP- and DeepMD-related outputs, locate high-error structures, and identify data gaps for the next iteration.
- **Project tracking**: use Data Management to record models, dataset paths, and notes across multiple iterations.

## Installation

We recommend installing NepTrainKit in a dedicated Python environment. Python 3.10 or later is required.

```bash
conda create -n nepkit python=3.10
conda activate nepkit
pip install NepTrainKit
```

After installation, launch the application with either command:

```bash
nepkit
# or
NepTrainKit
```

### NEP compute backend

NepTrainKit does not compile or bundle an NEP compute backend. `pip` installs the separate `nep-adapters` dependency:

| Platform | Installed backend |
| --- | --- |
| macOS / Windows | CPU |
| Linux x86_64 | CPU and CUDA in one wheel |

The Linux CUDA path requires a compatible NVIDIA driver, but installing the wheel does not require a local CUDA toolkit or NVCC. Source builds and supported CUDA architectures are documented in the [NEPAdapters repository](https://github.com/MagTheoryLab/NEPAdapters).

After launching NepTrainKit, select `Auto`, `CPU`, or `CUDA` under `Settings → NEP Backend`. `Auto` uses CUDA when the installed wheel, driver, and model support it; otherwise NepTrainKit explains why it is continuing on CPU. Explicit `CUDA` requests fail instead of silently changing backend.

Confirm the installed runtime with:

```bash
python -c "import nep_adapters as n; print(n.backend_status('cpu')); print(n.backend_status('cuda'))"
```

### Windows package

If you prefer not to compile NepTrainKit locally, download `NepTrainKit.win32.zip` from [GitHub Releases](https://github.com/aboys-cb/NepTrainKit/releases). This package is for Windows only.

## Documentation and support

- User documentation: [neptrainkit.readthedocs.io](https://neptrainkit.readthedocs.io/en/latest/)
- Release notes: [GitHub Releases](https://github.com/aboys-cb/NepTrainKit/releases)
- Bug reports and feature requests: [GitHub Issues](https://github.com/aboys-cb/NepTrainKit/issues)
- Community: [QQ group invitation](https://qm.qq.com/q/wPDQYHMhyg)

If this is your first time using NepTrainKit, begin with **Quickstart** and **Cleaning candidate structures before DFT** in the user documentation. If you already know which class of configurations you need, go directly to the Make Dataset card reference.

## Citation

If NepTrainKit contributes to your research, please cite:

```bibtex
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
```

## License and third-party code

NepTrainKit is licensed under the GNU General Public License v3.0 or later. See [LICENSE](./LICENSE) for details.

NEP computation is provided by the separate [nep-adapters](https://github.com/MagTheoryLab/NEPAdapters) dependency. NepTrainKit no longer vendors the NEP_CPU or GPUMD backend source trees.

See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for the remaining attribution in this repository. The `nep-adapters` distribution carries its own backend source notices and licenses.
