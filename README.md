<div align="center">
<a href="https://github.com/aboys-cb/NepTrainKit">
  <img src="./src/NepTrainKit/src/images/logo.png" width="25%" alt="NepTrainKit Logo">
</a><br>
<a href="https://pypi.org/project/NepTrainKit"><img src="https://img.shields.io/pypi/dm/NepTrainKit?logo=pypi&logoColor=white&color=blue&label=PyPI" alt="PyPI Downloads"></a>
<a href="https://python.org/downloads"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python Version"></a>
<a href="https://codecov.io/github/aboys-cb/NepTrainKit"><img src="https://codecov.io/github/aboys-cb/NepTrainKit/graph/badge.svg?token=HQ5FMLD91F" alt="Codecov"></a>
<a href="https://github.com/aboys-cb/NepTrainKit/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue" alt="License"></a>
</div>

# NepTrainKit

NepTrainKit 是面向 NEP 训练集准备、检查和可视化的桌面工具。它不替代 GPUMD 的长时间训练，也不替代 DFT 计算；它负责训练前后最容易反复手工处理的部分：生成候选结构、清洗异常样本、筛选代表结构，并把干净的数据接回 DFT 和 GPUMD 训练流程。

## 主要功能

- **Make Dataset**：用卡片生成应变、扰动、缺陷、表面、掺杂、磁性和溶剂化候选结构。
- **NEP Dataset Display**：查看结构、误差和分布，删除异常样本，导出干净子集。
- **代表性筛选**：用 FPS 等方法从候选池中挑选更少、更有代表性的结构。
- **训练结果回看**：加载 NEP / DeepMD 相关输出，定位高误差结构和下一轮数据缺口。
- **项目记录**：用 Data Management 记录多轮模型、数据路径和备注。

## 安装

建议使用独立 Python 环境。当前包要求 Python 3.10 或更新版本。

```bash
conda create -n nepkit python=3.10
conda activate nepkit
pip install NepTrainKit
```

安装完成后，可以用任一命令启动：

```bash
nepkit
# 或
NepTrainKit
```

### GPU 后端

安装过程会优先构建 CPU 后端；如果能找到可用的 CUDA / NVCC，也会尝试构建 GPU 后端。CUDA 没准备好时，安装会跳过 GPU 扩展，软件仍可用 CPU 后端运行。

Linux / WSL2：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
pip install NepTrainKit
```

Windows PowerShell：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:Path = "$env:CUDA_PATH\bin;" + $env:Path
pip install NepTrainKit
```

如果想明确跳过 GPU 构建，可以设置：

```bash
NEPKIT_BUILD_GPU=0 pip install NepTrainKit
```

需要手动指定 GPU 架构时，在安装前设置 `NEP_GPU_GENCODE`，例如：

```bash
export NEP_GPU_GENCODE="arch=compute_89,code=sm_89"
pip install NepTrainKit
```

运行后可在 `Settings → NEP Backend` 中选择 `Auto`、`CPU` 或 `GPU`。如果遇到 `CUDA driver version is insufficient for CUDA runtime version`，先切到 `CPU` 后端，再检查驱动和 CUDA 版本。

### Windows 可执行包

不想本地编译时，可以到 [Releases](https://github.com/aboys-cb/NepTrainKit/releases) 下载 `NepTrainKit.win32.zip`。该包只面向 Windows。

## 文档

- 在线文档：[neptrainkit.readthedocs.io](https://neptrainkit.readthedocs.io/en/latest/index.html)
- 更新说明：[GitHub Releases](https://github.com/aboys-cb/NepTrainKit/releases)
- 问题反馈：[GitHub Issues](https://github.com/aboys-cb/NepTrainKit/issues)
- 社区交流：[QQ 群链接](https://qm.qq.com/q/wPDQYHMhyg)

第一次使用建议先看在线文档里的“快速开始”和“候选结构清洗后再进入 DFT”。如果你已经知道自己要补哪类数据，直接查 Make Dataset 卡片手册。

## 引用

如果你的研究使用了 NepTrainKit，请引用：

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

## 许可证和第三方代码

本仓库使用 GNU General Public License v3.0 or later。详见 [LICENSE](./LICENSE)。

NepTrainKit 包含并改写了部分第三方代码：

- [NEP_CPU](https://github.com/brucefan1983/NEP_CPU)：Zheyong Fan、Junjie Wang、Eric Lindgren 及贡献者，GPL-3.0-or-later。
- [GPUMD](https://github.com/brucefan1983/GPUMD)：Zheyong Fan 和 GPUMD 开发团队，GPL-3.0-or-later。

目录级来源说明见 [src/nep_cpu/README.md](./src/nep_cpu/README.md)、[src/nep_gpu/README.md](./src/nep_gpu/README.md) 和 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。再分发时请保留版权和许可证说明。
