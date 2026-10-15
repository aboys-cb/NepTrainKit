# 快速开始

这页只做一件事：从安装到生成第一批候选结构。跑通以后，再去看完整清洗流程和每张卡片的参数。

如果软件已经能正常启动，你只是想查某个页面的按钮或参数，不必从头阅读本页，直接进入
[功能指南](module/index.rst)。功能指南按桌面软件页面顺序组织。

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

如果 `cuda` 显示不可用，NepTrainKit 的 `Auto` 模式（`设置` → `NEP 后端`）会说明原因并使用 CPU。只有在必须使用 CUDA 时才明确选择 `CUDA`；不可用时程序会直接报错。

启动软件：

```bash
nepkit
```

不想配置 Python 环境的 Windows 用户，也可以从
[GitHub Releases](https://github.com/aboys-cb/NepTrainKit/releases) 下载
`NepTrainKit.windows-x86_64.zip`，解压后直接运行 `NepTrainKit.exe`。

界面语言默认跟随系统，可以在 `设置` → `语言` 中切换，重启后生效。本文档按中文界面的按钮名称编写；
使用英文界面时，请切换到英文文档对照。

## 2. 打开「构建数据集」页面

第一次建议从 `构建数据集` 开始。它的职责是制作训练集候选结构：输入一批基础结构，
选择一张或多张卡片，生成应变、扰动、缺陷、表面、掺杂或磁性构型。

切到 `构建数据集` 后，先点击窗口顶部的 `打开` 按钮导入初始结构。这里的初始结构
通常是已经弛豫好的 `xyz`、`extxyz`、`POSCAR` 或 `CIF` 文件；后面的卡片都会以这批结构
作为输入继续生成候选构型。

```{image} _static/image/generated/make_data_empty.png
:alt: 构建数据集空白工作区
:class: docs-screenshot
```

图中标出了四个位置：

| 编号 | 图中标注 | 作用 |
| --- | --- | --- |
| ① | 工作流库 | 左侧区域，保存、打开和复用整条工作流。 |
| ② | 添加卡片 | `添加新卡片` 按钮，添加构型生成或筛选卡片。 |
| ③ | 运行选中卡片 | 顶部 `运行` 按钮，只运行已勾选的卡片。 |
| ④ | 卡片检查器 | 右侧区域，查看和修改当前选中卡片的参数与说明。 |

导入初始结构用的是窗口顶部的 `打开` 按钮（图中未标出）。没有输入结构时，多数生成卡片没有可处理的对象。

## 3. 生成一组候选结构

这里用“晶格应变（`Lattice Strain`）”作为第一张卡。它的输入是已有结构，输出是一组晶格被拉伸或压缩后的结构。
这一步得到的是候选结构，不是最终训练集。

1. 先用 `打开` 导入初始结构。
2. 点击 `添加新卡片`。
3. 选择“晶格应变（`Lattice Strain`）”。
4. 设置一个小范围，例如 X/Y/Z 都从 `-2%` 到 `2%`，步长 `1%`。
5. 勾选这张卡，点击顶部 `运行`。
6. 运行完成后，从卡片上的导出按钮保存结果。

```{image} _static/image/generated/make_data_lattice_strain.png
:alt: 晶格应变卡片快速上手
:class: docs-screenshot
```

这张图对应的使用逻辑是：

- 先添加卡片，不要一开始就改很多高级参数。
- 只改与物理目标直接相关的范围，例如应变轴、最小值、最大值、步长。
- 运行后先看输出数量是否符合预期，再导出到下一步检查。

## 4. 检查候选结构

`构建数据集` 生成的是候选结构，不建议默认直接送 DFT。最短检查路径是：

```{image} _static/image/generated/show_nep_overview.png
:alt: NEP 数据集查看总览
:class: docs-screenshot
```

1. 点击左侧导航中的 `NEP 数据集查看`，再点击顶部 `打开` → `打开文件…`，选择上一步导出的
   `xyz` 文件。
2. 先看左侧散点图里的离群点，点中后在右侧结构区确认是否有原子重叠或明显破碎。
3. 选中异常结构后点击工具栏 `删除选中项`；需要留档时先用 `保存` → `导出已选择结构` 单独保存。
4. 清洗完成后，用 `保存` → `导出当前保留结构（N）…` 保存干净的候选池。若随后做 FPS 代表性采样，用 `导出已选择结构（N）…` 保存选中的子集。

## 5. 进入 DFT 和训练

清洗后的结构再送去做第一性原理计算。DFT 标注完成后，用 GPUMD 训练 NEP；
训练结束后，再回到 `NEP 数据集查看` 查看训练误差。

完整的清洗顺序、为什么不要先 FPS、以及如何用 NEP 预测做预筛，见
[候选结构清洗后再进入 DFT](workflows/clean-candidate-structures.md)。

如果你在做缺陷、表面、掺杂或磁性结构，直接去看 [构建数据集卡片手册](module/make-dataset-cards/index.md)。手册按卡片用途组织，比逐个试菜单更快。

手头没有初始结构时，可以先用 [晶体原型构建](module/make-dataset-cards/cards/crystal-prototype-builder-card.md)
卡片从 `fcc`、`bcc`、`hcp` 等标准原型开始。
