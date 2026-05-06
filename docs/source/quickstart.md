# 快速开始

本页带你从安装到生成第一组训练数据，约 10 分钟。

## 1. 安装

Python 3.10–3.12，建议独立环境：

```bash
conda create -n nepkit python=3.10
conda activate nepkit
pip install NepTrainKit
```

Linux/WSL2 如需 GPU：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
pip install NepTrainKit
```

Windows PowerShell 如需 GPU：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:Path = "$env:CUDA_PATH\bin;" + $env:Path
pip install NepTrainKit
```

## 2. 启动

```bash
nepkit
```

## 3. 搭建第一个训练集（5 分钟走通）

假设你有一个弛豫好的 Si 单胞，想做一组应变结构来训练弹性响应。

### 3.1 导入结构

切换到 `Make Dataset` 页签，点击 `Open` 导入你的结构文件（支持 `.xyz` / `POSCAR` / `CIF` / `extxyz`）。

### 3.2 添加一张卡

从左侧卡片列表找到 `Lattice Strain`，点击或拖入工作区。这张卡会在输入结构上施加受控的晶格应变。

### 3.3 设置参数

点击卡片展开设置面板：
- `Axes` = `uniaxial`
- `X` = `[-2, 2, 1]`（从 -2% 压缩到 +2% 拉伸，每 1% 一步）

### 3.4 运行

勾选卡片的复选框（表示这张卡参与运行），点击顶部 `Run`。卡片状态变为绿色表示完成。

### 3.5 导出

点击卡片的导出按钮，选择保存路径。默认文件名 `export_Lattice_Strain_structure.xyz`，内含 5 个不同应变的 Si 结构。

### 3.6 下一步

- 加 `Atomic Perturb` 在应变后补坐标噪声
- 加 `FPS Filter` 从大批量输出中选代表结构
- 参考 [配方示例](module/make-dataset-cards/recipes.md) 学习多卡组合

## 4. 主要模块

| 模块 | 做什么 | 什么时候用 |
|------|--------|-----------|
| `NEP Dataset Display` | 导入、可视化、筛选、导出训练结果 | 想看图分析模型在哪些结构上误差大 |
| `Make Dataset` | 卡片流水线生成/过滤训练结构 | 想系统扩充训练集覆盖 |
| `Data Management` | 按项目/模型版本管理数据 | 多组实验需要版本化追踪 |
| `Settings` | 配置后端、绘图引擎、性能参数 | 需要切 GPU/CPU 或换绘图引擎 |

## 5. 可复现性

- 勾选卡片的 `Use Seed` 并固定 `Seed` 值
- 固定卡片顺序和输入顺序
- 导出后用 `Config_type` 标签追溯每个结构的生成来源
