# 快速开始

本页覆盖从安装到首次使用的最短路径。

## 1. 安装

- Python 3.10–3.12
- 建议使用独立环境

```bash
conda create -n nepkit python=3.10
conda activate nepkit
pip install NepTrainKit
```

Linux/WSL2 可按需设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
pip install NepTrainKit
```

Windows PowerShell 可按需设置：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:Path = "$env:CUDA_PATH\bin;" + $env:Path
pip install NepTrainKit
```

## 2. 启动

```bash
nepkit
# 或
NepTrainKit
```

## 3. 主要模块

- `NEP Dataset Display`：导入、可视化、筛选、导出。
- `Make Dataset`：用 cards 组装生成/过滤 pipeline。
- `Data Management`：按 Project/Model(version) 管理数据。
- `Settings`：配置 backend、绘图引擎和性能参数。

## 4. 可复现性建议

- 对支持的卡片开启 `Use seed` 并固定 `seed`
- 固定卡片顺序与输入顺序
- 结合 `Config_type` 检查导出结果
