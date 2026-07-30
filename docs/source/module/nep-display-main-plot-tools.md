# 主图区：看图、选择与处理数据

这一页对应软件左侧主图区及其竖向工具栏。目录顺序与界面按钮从上到下保持一致。
点击左侧目录中的具体工具，可以直接查看参数、默认行为和执行结果。

```{toctree}
:maxdepth: 1
:hidden:

nep-main-tool-basic
nep-main-tool-select-index
nep-main-tool-select-range
nep-main-tool-select-lattice
nep-main-tool-max-error
nep-main-tool-sparse-samples
nep-main-tool-nonphysical
nep-main-tool-net-force
nep-main-tool-edit-info
nep-main-tool-export-descriptor
nep-main-tool-energy-shift
nep-main-tool-dft-d3
nep-main-tool-training-audit
nep-main-tool-distributions
```

## 按界面顺序选择工具

| 界面按钮 | 什么时候用 | 参数入口 |
| --- | --- | --- |
| `Reset View` / `Pan View` / `Mouse Selection` 等 | 调整主图、圈选、反选、撤销或删除 | [浏览与基础选择](nep-main-tool-basic.md) |
| `Select by Index` | 已知结构编号或切片 | [按索引选择](nep-main-tool-select-index.md) |
| `Select by Range` | 按当前图的 x/y 范围选点 | [按图上范围选择](nep-main-tool-select-range.md) |
| `Select by Lattice` | 按晶格长度和角度筛选 | [按晶格参数选择](nep-main-tool-select-lattice.md) |
| `Find Max Error Point` | 找当前指标误差最大的前 N 个结构 | [查找最大误差结构](nep-main-tool-max-error.md) |
| `Sparse samples` | 用 FPS 选择代表结构 | [代表性采样](nep-main-tool-sparse-samples.md) |
| `Finding non-physical structures` | 查找近邻距离异常结构 | [查找疑似非物理结构](nep-main-tool-nonphysical.md) |
| `Check Net Force` | 按净力阈值检查标签 | [检查净力](nep-main-tool-net-force.md) |
| `Edit Info` | 批量增加、删除或重命名 metadata | [编辑结构信息](nep-main-tool-edit-info.md) |
| `Export structure descriptor` | 导出当前数据的结构描述符 | [导出结构描述符](nep-main-tool-export-descriptor.md) |
| `Energy Baseline Shift` | 拟合并应用能量基线平移 | [能量基线平移](nep-main-tool-energy-shift.md) |
| `DFT D3` | 增加或移除 DFT-D3 修正 | [DFT-D3 修正](nep-main-tool-dft-d3.md) |
| `Training Set Audit` | 打开训练集整体质量评估 | [进入训练集评估](nep-main-tool-training-audit.md) |
| `Explore distributions` | 查看数值字段分布并反向选结构 | [查看数据分布](nep-main-tool-distributions.md) |

英文名称来自软件按钮，目录名称统一采用“中文用途（界面按钮）”，便于中文阅读和界面对照。
