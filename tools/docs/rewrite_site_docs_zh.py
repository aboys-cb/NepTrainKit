"""Rewrite hand-written docs pages into Chinese-first content."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]


FILES: dict[str, str] = {
    "docs/source/index.rst": dedent(
        """
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

              $env:CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
              $env:Path = "$env:CUDA_PATH\\bin;" + $env:Path
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
           变更日志 <changelog>
        """
    ).strip()
    + "\n",
    "docs/source/quickstart.md": dedent(
        """
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
        $env:CUDA_PATH = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
        $env:Path = "$env:CUDA_PATH\\bin;" + $env:Path
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
        """
    ).strip()
    + "\n",
    "docs/source/formats.md": dedent(
        """
        # 支持的结构/结果格式

        NepTrainKit 可读取并转换常见材料模拟输出为内部 `Structure` 表示。

        ## NEP 数据

        - `.xyz` / `.extxyz`
        - `nep.txt`
        - `energy_*.out` / `force_*.out` / `virial_*.out` / `stress_*.out` / `descriptor*.out`

        ## VASP

        - `OUTCAR`：读取 lattice/positions/force/stress/virial
        - `XDATCAR`：读取逐帧晶胞与坐标

        ## LAMMPS

        - `dump` / `lammpstrj`：支持正交/三斜晶胞与多种坐标列

        ## ASE

        - `.traj`：通过 `ase.io.iread()` 逐帧导入

        ## 说明

        - 导入器支持 `cancel_event` 协作中断
        - 大数据场景启用渲染抽样，不影响底层选择精度
        """
    ).strip()
    + "\n",
    "docs/source/changelog.md": dedent(
        """
        # 变更日志

        记录 v2.5.4 到 v2.6.3 的关键变更。

        ## v2.6.1（2025-09-12）

        - 新增 GPU NEP backend（Auto/CPU/GPU）与 GPU Batch Size
        - 新增 Data Management 模块（项目、版本、标签、备注、检索）
        - 新增 OrganicMolConfig 相关卡片能力
        - 新增能量基线对齐、DFT-D3、Edit Info、descriptor 导出
        - 优化 Vispy 性能与 native 计算链路
        - 兼容旧版 DeepMD/NPY

        ## v2.6.3（2025-09-14）

        - 新增/增强导入器：VASP OUTCAR/XDATCAR、LAMMPS dump、ASE traj
        - ResultData 加载链路重构为 importer pipeline
        - 修复 VisPy picking 精度与循环依赖问题
        - 优化距离与键长计算性能
        - GPU backend 增强自检并在异常时自动回退 CPU
        """
    ).strip()
    + "\n",
    "docs/source/api/index.rst": dedent(
        """
        API 参考
        =========

        下列页面由 ``automodule`` 自动生成；源码 docstring 保持英文原文。

        .. automodule:: NepTrainKit.core.structure
           :members:
           :undoc-members:
           :show-inheritance:

        .. automodule:: NepTrainKit.core.calculator
           :members:
           :undoc-members:
           :show-inheritance:

        .. automodule:: NepTrainKit.core.io
           :members:
           :undoc-members:
           :show-inheritance:
        """
    ).strip()
    + "\n",
    "docs/source/example/index.rst": dedent(
        """
        示例
        ======

        .. toctree::
           :maxdepth: 2

           NEP 数据展示示例 <NEP-display>
           Descriptor 绘图示例 <Plot-descriptor>
        """
    ).strip()
    + "\n",
    "docs/source/example/NEP-display.md": dedent(
        """
        # NEP Dataset Display 示例

        ## 数据导入

        使用 `nep.txt` 与 `train.xyz` 作为最小输入；支持拖拽导入。

        ![GIF Image](../_static/image/example/display/import.gif)

        ## 轨迹播放

        点击播放按钮可逐帧预览结构：

        <img src="../_static/image/play.svg" alt="play" width='30' height='30' />

        ![GIF Image](../_static/image/example/display/play.gif)

        ## 数据筛选

        - 最大误差点筛选：![GIF Image](../_static/image/example/display/maxerror.gif)
        - FPS 稀疏采样：![GIF Image](../_static/image/example/display/fps.gif)
        - 鼠标选择/反选：![GIF Image](../_static/image/example/display/mouse.gif)
        - `Config_type` 组合筛选：![GIF Image](../_static/image/example/display/config1.gif)

        ## 导出

        过滤完成后导出数据：

        ![GIF Image](../_static/image/example/display/save.gif)
        """
    ).strip()
    + "\n",
    "docs/source/example/Plot-descriptor.md": dedent(
        """
        # Descriptor 绘图示例

        以 CsPbI3 为例，导出 descriptor 后绘制结构分布。

        ![Image](../_static/image/example/plot_descriptor/main_ui.png)

        ## 导出步骤

        1. 选择结构：<img src="../_static/image/pen.svg" alt="pen" width='30' height='30' />
        2. 点击导出：<img src="../_static/image/export.svg" alt="export" width='30' height='30' />
        3. 选择文件路径

        ## 绘图

        使用 [plot_descriptor.py](https://github.com/aboys-cb/NepTrainKit/blob/master/tools/plot_descriptor.py)，
        配置 `config` 和 `method` 后执行：

        `python plot_descriptor.py`

        ![Image](../_static/image/example/plot_descriptor/descriptor_scatter_plot.png)
        """
    ).strip()
    + "\n",
    "docs/source/module/index.rst": dedent(
        """
        功能模块
        ============

        .. toctree::
           :maxdepth: 2

           NEP 数据展示 <NEP-dataset-display>
           数据生成（Make dataset） <make-dataset>
           数据生成卡片手册 <make-dataset-cards/index>
           自定义卡片开发 <custom-card-development>
           数据管理 <data-management>
           设置 <settings>
        """
    ).strip()
    + "\n",
    "docs/source/module/NEP-dataset-display.md": dedent(
        """
        # NEP Dataset Display

        ## 界面概览

        包括工具栏、结果可视化区、结构视图、信息区和工作路径区。

        ![interface](../_static/image/interface.png)

        ## 数据导入与导出

        - 导入：Open 按钮或拖拽
        - 导出：保存结果或导出当前选择

        ## 常用工具

        - 筛选：Index/Range/Max Error/FPS/Config_type
        - 编辑：Delete/Undo/Edit Info
        - 分析：Export descriptor / Energy Baseline / DFT-D3
        - 结构视图：Show Bonds / Show Arrows / Export current structure

        ## 搜索模式

        - `tag`：基于 `Config_type` 正则匹配
        - `formula`：基于化学式正则匹配
        - `elements`：元素集合语法（`E` / `+E` / `-E`）

        ## 可视化说明

        结果区包含 descriptor、energy、force、pressure、potential energy 子图。
        点击主图点可同步查看右侧结构信息。
        """
    ).strip()
    + "\n",
    "docs/source/module/make-dataset.md": dedent(
        """
        # Make Dataset

        Make Dataset 是一个基于 cards 的结构生成与过滤流水线编辑器。

        ## 数据流

        - 线性链：上游输出传递给下游
        - Card Group：组内共享同一输入并汇合输出
        - Filter：可全局或组内筛选

        ## 推荐流程

        1. 导入结构（XYZ/POSCAR/CIF）
        2. 构建 pipeline
        3. 保存/加载 card JSON
        4. 导出 `make_dataset.xyz`

        ## 文档入口

        - [Make Dataset 卡片手册](make-dataset-cards/index.md)
        - [Make Dataset 配方示例](make-dataset-cards/recipes.md)
        """
    ).strip()
    + "\n",
    "docs/source/module/custom-card-development.md": dedent(
        """
        # 自定义卡片开发

        ## 放置目录

        将自定义卡片放在用户配置目录下的 `cards/` 目录。

        ## 最小模板

        - 继承 `MakeDataCard`
        - 实现 `init_ui` / `process_structure` / `to_dict` / `from_dict`
        - 使用 `@CardManager.register_card` 注册

        ## 约定

        - `process_structure` 返回 `list[ase.Atoms]`
        - `to_dict` 持久化所有行为参数
        - `from_dict` 完整恢复状态
        - 必要时写入 `Config_type` 标签
        """
    ).strip()
    + "\n",
    "docs/source/module/data-management.md": dedent(
        """
        # 数据管理（Data Management）

        用于按 Project/Model(version) 管理训练数据与实验版本。

        ## 核心对象

        - Project：树形项目节点
        - Model(version)：数据条目（路径、指标、备注、标签）
        - Tags：颜色标签体系

        ## 常用操作

        - 新建/修改/删除 Project
        - 新建/修改/删除 Model(version)
        - 打开本地目录或 URL
        - `Ctrl+F` 高级检索

        ## 存储位置

        - Windows：`C:\\Users\\<You>\\AppData\\Local\\NepTrainKit\\mlpman.db`
        - Linux：`~/.config/NepTrainKit/mlpman.db`
        """
    ).strip()
    + "\n",
    "docs/source/module/settings.md": dedent(
        """
        # 设置（Settings）

        用于配置显示行为、后端策略与性能参数，修改后即时生效。

        ## 个性化

        - Force data format：Raw / Norm
        - Canvas Engine：PyQtGraph / Vispy
        - Auto loading
        - Cache output files
        - Covalent radius coefficient
        - Sort atoms
        - Use card group menu

        ## NEP 设置

        - NEP Backend：CPU / GPU / Auto
        - GPU Batch Size

        ## 关于

        - NEP89 下载检查
        - 文档与反馈入口
        - 版本与更新检查
        """
    ).strip()
    + "\n",
    "docs/source/module/make-dataset-cards/index.md": dedent(
        """
        # Make Dataset 卡片手册

        本手册提供所有卡片的参数级说明与推荐实践。

        维护规范： [卡片文档编写规范](writing-guide.md)

        ```{toctree}
        :maxdepth: 1
        :hidden:

        writing-guide
        recipes
        cards/super-cell-card
        cards/crystal-prototype-builder-card
        cards/perturb-card
        cards/vibration-perturb-card
        cards/magmom-rotation-card
        cards/cell-strain-card
        cards/cell-scaling-card
        cards/shear-matrix-card
        cards/shear-angle-card
        cards/random-slab-card
        cards/random-doping-card
        cards/composition-sweep-card
        cards/random-occupancy-card
        cards/conditional-replace-card
        cards/group-label-card
        cards/magnetic-order-card
        cards/random-vacancy-card
        cards/vacancy-defect-card
        cards/stacking-fault-card
        cards/interstitial-adsorbate-card
        cards/organic-mol-config-pbc-card
        cards/layer-copy-card
        cards/fps-filter-card
        cards/card-group
        ```

        ## 快速入口

        - 晶格：Super Cell / Lattice Strain / Lattice Perturb
        - 缺陷：Random Vacancy / Insert Defect / Stacking Fault
        - 合金：Random Doping / Composition Sweep / Conditional Replace
        - 磁性：Magnetic Order / Magmom Rotation
        - 过滤：FPS Filter
        - 容器：Card Group
        """
    ).strip()
    + "\n",
    "docs/source/module/make-dataset-cards/recipes.md": dedent(
        """
        # Make Dataset 配方示例（Recipes）

        ## 高熵合金

        `Composition Sweep -> Random Occupancy -> Random Doping -> FPS Filter`

        ## 富缺陷表面

        `Random Slab -> Insert Defect -> Random Vacancy -> FPS Filter`

        ## 磁性数据

        `Group Label -> Magnetic Order -> Magmom Rotation`

        ## 有机构象

        `Organic Mol Config -> Atomic Perturb or Lattice Perturb -> FPS Filter`
        """
    ).strip()
    + "\n",
    "docs/source/module/make-dataset-cards/writing-guide.md": dedent(
        """
        # 卡片文档编写规范

        目标：文档可决策、可执行、可复现。

        ## 语言规范

        - 正文中文
        - 关键术语可保留英文（如 `seed`、`Config_type`、`sampling mode`）
        - 字段名与 serialized key 保持英文

        ## 必备章节

        - 功能说明
        - 适用场景与不适用场景
        - 输入前提
        - 参数说明（完整）
        - 推荐预设（可直接复制 JSON）
        - 推荐组合
        - 常见问题与排查
        - 输出标签 / 元数据变更
        - 可复现性说明

        ## 关键校验

        - `Default` 必须与运行时代码默认值一致
        - 空值必须写 `""` 或 `null`
        - 禁止 `(empty)` 占位写法
        - 字符串枚举与数字严格区分（如 `"111"` vs `111`）

        ## 维护命令

        - `python tools/docs/audit_card_docs.py`
        - `python -m sphinx -W -b html docs/source docs/build/html`
        """
    ).strip()
    + "\n",
}


def main() -> int:
    for rel, content in FILES.items():
        path = ROOT / rel
        path.write_text(content, encoding="utf-8")
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

