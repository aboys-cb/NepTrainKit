# 快速开始

这页只做一件事：从安装到生成第一批候选结构。跑通以后，再去看完整清洗流程和每张卡片的参数。

## 1. 安装

建议使用独立环境，Python 版本用 3.10 到 3.13。

```bash
conda create -n nepkit python=3.10
conda activate nepkit
pip install NepTrainKit
```

NepTrainKit 会安装独立的 `nep-adapters` 运行时。macOS 和 Windows wheel 提供 CPU；Linux x86_64 wheel 同时提供 CPU 和 CUDA。安装 wheel 不要求本机安装 CUDA Toolkit 或 NVCC，但 CUDA 计算需要兼容的 NVIDIA 驱动。

安装后先确认运行时状态：

```bash
python -c "import nep_adapters as n; print(n.backend_status('cpu')); print(n.backend_status('cuda'))"
```

如果 `cuda` 显示不可用，NepTrainKit 的 `Auto` 模式会说明原因并使用 CPU。只有在必须使用 CUDA 时才明确选择 `CUDA`；不可用时程序会直接报错。

启动软件：

```bash
nepkit
```

## 2. 打开 Make Dataset

第一次建议从 `Make Dataset` 开始。它的职责是制作训练集候选结构：输入一批基础结构，
选择一张或多张卡片，生成应变、扰动、缺陷、表面、掺杂或磁性构型。

切到 `Make Dataset` 后，先点击窗口顶部的 `Open` 按钮导入初始结构。这里的初始结构
通常是已经弛豫好的 `xyz`、`extxyz`、`POSCAR` 或 `CIF` 文件；后面的卡片都会以这批结构
作为输入继续生成候选构型。

```{image} _static/image/generated/make_data_empty.png
:alt: Make Dataset empty workspace
:class: docs-screenshot
```

图中需要记住四个位置：

- `Open input structures`：导入初始结构。没有输入结构时，多数生成卡片没有可处理对象。
- `Add new card`：添加构型生成或筛选卡片。
- `Run selected cards`：只运行已勾选的卡片。
- `Workflow workspace`：卡片会按顺序放在这里，后续也在这里检查结果和导出。

## 3. 生成一组候选结构

这里用 `Lattice Strain` 作为第一张卡。它的输入是已有结构，输出是一组晶格被拉伸或压缩后的结构。
这一步得到的是候选结构，不是最终训练集。

1. 先用 `Open` 导入初始结构。
2. 点击 `Add new card`。
3. 选择 `Lattice Strain`。
4. 设置一个小范围，例如 X/Y/Z 都从 `-2%` 到 `2%`，步长 `1%`。
5. 勾选这张卡，点击顶部 `Run`。
6. 运行完成后，从卡片上的导出按钮保存结果。

```{image} _static/image/generated/make_data_lattice_strain.png
:alt: Lattice Strain quickstart
:class: docs-screenshot
```

这张图对应的使用逻辑是：

- 先添加卡片，不要一开始就改很多高级参数。
- 只改与物理目标直接相关的范围，例如应变轴、最小值、最大值、步长。
- 运行后先看输出数量是否符合预期，再导出到下一步检查。

## 4. 检查候选结构

`Make Dataset` 生成的是候选结构，不建议默认直接送 DFT。最短检查路径是：

```{image} _static/image/generated/show_nep_overview.png
:alt: NEP Dataset Display overview
:class: docs-screenshot
```

1. 在 `NEP Dataset Display` 打开候选结构。
2. 先看结构和明显异常点。
3. 删除或单独导出异常样本。
4. 清洗后再决定是否做 FPS 代表性采样。

## 5. 进入 DFT 和训练

清洗后的结构再送去做第一性原理计算。DFT 标注完成后，用 GPUMD 训练 NEP；
训练结束后，再回到 `NEP Dataset Display` 查看训练误差。

完整的清洗顺序、为什么不要先 FPS、以及如何用 NEP 预测做预筛，见
[候选结构清洗后再进入 DFT](workflows/clean-candidate-structures.md)。

如果你在做缺陷、表面、掺杂或磁性结构，直接去看 [Make Dataset 卡片手册](module/make-dataset-cards/index.md)。手册按卡片用途组织，比逐个试菜单更快。

如果你要从标准晶体开始，而不是导入已有结构，可以先用 `Crystal Prototype Builder` 生成 `fcc`、`bcc`、`hcp` 或 `fcc111` 原型。需要扫 fcc (111) 的 GSFE 路径时，先生成 `fcc111` 原型，再接 `Stacking Fault / GSFE Path`；不要把普通 cubic `fcc` cell 直接拿去扫 `(111)`。
