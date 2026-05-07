# 支持格式

这页按“你手里有什么文件”来选入口。NepTrainKit 会把常见结构和训练结果转换成内部
`Structure` 表示，用于显示、筛选、生成候选结构或导出。

## 结构文件

| 文件 | 推荐入口 | 常见用途 |
| --- | --- | --- |
| `.xyz` / `.extxyz` | `Make Dataset` 或 `NEP Dataset Display` | 作为初始结构、候选池或训练集 |
| `POSCAR` / `CONTCAR` | `Make Dataset` | 作为初始晶体结构生成候选池 |
| `CIF` | `Make Dataset` | 从晶体结构开始构建候选集 |
| ASE `.traj` | `NEP Dataset Display` | 查看已有轨迹或转换结构 |

如果你要继续生成新结构，优先导入 `Make Dataset`。如果你要检查、删除、筛选或导出子集，
优先导入 `NEP Dataset Display`。

## NEP 训练相关文件

| 文件 | 作用 |
| --- | --- |
| `nep.txt` | NEP 模型文件，可用于预测、回看或预筛 |
| `energy_*.out` | 能量预测/标签结果 |
| `force_*.out` | 力预测/标签结果 |
| `virial_*.out` | virial 预测/标签结果 |
| `stress_*.out` | stress 预测/标签结果 |
| `descriptor*.out` | 描述符输出，用于可视化或采样分析 |

训练结束后，把训练结构和这些输出一起导入 `NEP Dataset Display`，可以查看散点图、
定位误差最大的结构，并导出下一轮要处理的子集。

## VASP

- `OUTCAR`：读取晶胞、坐标、力、应力和 virial。
- `XDATCAR`：读取逐帧晶胞和坐标。

VASP 结果通常用于把 DFT 标注转成训练结构，或在 `NEP Dataset Display` 中检查标签质量。

## LAMMPS

- `dump`
- `lammpstrj`

支持正交和三斜晶胞，以及常见坐标列。导入后建议先抽查晶胞、元素类型和坐标单位是否符合预期。

## 大数据和导入说明

- 大数据集导入时，界面显示可能做抽样或延迟渲染；底层选择和导出仍以完整数据为准。
- 导入过程支持中断。
- 不同来源的文件最好保留清晰命名，例如 `candidate_pool.xyz`、`candidate_pool_clean.xyz`、`dft_labeled.xyz`。
