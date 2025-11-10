<div align="center">
<a href="https://github.com/aboys-cb/NepTrainKit">
  <img src="./src/NepTrainKit/src/images/logo.svg" width="25%" alt="NepTrainKit Logo">
</a><br>    
<a href="https://pypi.org/project/NepTrainKit"><img src="https://img.shields.io/pypi/dm/NepTrainKit?logo=pypi&logoColor=white&color=blue&label=PyPI" alt="PyPI Downloads"></a>   
<a href="https://python.org/downloads"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python Version"></a>  
<a href="https://codecov.io/github/aboys-cb/NepTrainKit"><img src="https://codecov.io/github/aboys-cb/NepTrainKit/graph/badge.svg?token=HQ5FMLD91F" alt="Codecov"></a>
<a href="https://github.com/aboys-cb/NepTrainKit/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue" alt="License"></a>
</div>


---

# NepTrainKit

**NepTrainKit** is a toolkit focused on the operation and visualization of **neuroevolution potential** (NEP) training datasets. It is mainly used to simplify and optimize the NEP model training process, providing an intuitive graphical interface and analysis tools to help users adjust  train dataset.

---

## Community Support

- Join the community chat: [https://qm.qq.com/q/wPDQYHMhyg](https://qm.qq.com/q/wPDQYHMhyg)
- Report issues or contribute via [GitHub Issues](https://github.com/aboys-cb/NepTrainKit/issues)

---

## Installation

> **It is strongly recommended to use pip for installation.**

### Method 1: Install via pip

If you are using Python 3.10 or a later version, you can install `NepTrainKit` using an environment manager like `conda`:

1. Create a new environment:

   ```bash
   conda create -n nepkit python=3.10
   ```

2. Activate the environment:

   ```bash
   conda activate nepkit
   ```

3. For CentOS users, install PySide6 (required for GUI functionality):

   ```bash
   conda install -c conda-forge pyside6
   ```

- Install directly using the `pip install` command:

  ```bash
  pip install NepTrainKit
  ```

  > **GPU build note (Linux/WSL2):** The build auto‑detects CUDA. If a compatible CUDA
  > toolkit is present, the NEP backend is compiled with GPU acceleration; otherwise,
  > it falls back to a CPU‑only build. If CUDA is not detected automatically, export
  > one of `CUDA_HOME` or `CUDA_PATH` and ensure the `lib64` directory is on your loader path
  > before running `pip install`:

  ```bash
  # choose your installed CUDA version/path
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
  # now install
  pip install NepTrainKit
  ```

  > **GPU build note (Windows PowerShell):** Set `CUDA_PATH` (or `CUDA_HOME`) and add
  > `bin` to `Path` before `pip install`:

  ```powershell
  $env:CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
  $env:Path = "$env:CUDA_PATH\\bin;" + $env:Path
  pip install NepTrainKit
  ```

  After installation, you can call the program using either `NepTrainKit` or `nepkit`.

- For the **latest version** (from GitHub):

  ```bash
  pip install git+https://github.com/aboys-cb/NepTrainKit.git
  ```

---

### Method 2: Windows Executable

A standalone executable is available for Windows users.

- Visit the [Releases](https://github.com/aboys-cb/NepTrainKit/releases) page
- Download `NepTrainKit.win32.zip`

 > Note: Only supported on Windows platforms.

### GPU Acceleration (optional)

- NepTrainKit includes an optional GPU‑accelerated NEP backend.
- Requirements: NVIDIA GPU/driver compatible with CUDA 12.4 runtime.
- Selection: In the app, go to Settings → NEP Backend and choose Auto/CPU/GPU.
  - Auto tries GPU first and falls back to CPU if unavailable.
  - Adjust GPU Batch Size to balance speed and memory.
  - If you see “CUDA driver version is insufficient for CUDA runtime version”, switch to CPU.

---

## Documentation

For detailed usage documentation and examples, please refer to the official documentation:  
[https://neptrainkit.readthedocs.io/en/latest/index.html](https://neptrainkit.readthedocs.io/en/latest/index.html)

- What's new: see `docs/source/changelog.md` or the Documentation "Changelog" page.

---

## Citation

If you use NepTrainKit in academic work, please cite the following publication and acknowledge the upstream projects where appropriate:

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

## Licensing and Attribution


- License: This repository is licensed under the GNU General Public License v3.0
  (or, at your option, any later version). See `LICENSE` at the repository root.
- Third‑party code: NepTrainKit incorporates source files and adapted logic from:
  - NEP_CPU (by Zheyong Fan, Junjie Wang, Eric Lindgren, and contributors):
    https://github.com/brucefan1983/NEP_CPU (GPL‑3.0‑or‑later)
  - GPUMD (by Zheyong Fan and the GPUMD development team):
    https://github.com/brucefan1983/GPUMD (GPL‑3.0‑or‑later)
- Directory‑level notes: See `src/nep_cpu/README.md` and `src/nep_gpu/README.md` for
  file‑level provenance, what was modified or added, and links to the upstream projects.
  A consolidated overview is also available in `THIRD_PARTY_NOTICES.md`.
- Redistribution: Any modifications and redistributions must remain under the GPL and
  preserve copyright and license notices, per the GPL requirements.

For academic use, cite NepTrainKit as shown above and acknowledge
NEP_CPU and/or GPUMD as appropriate.
