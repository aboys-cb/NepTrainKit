# 支持格式

这页按“你手里有什么文件”来选入口。NepTrainKit 会把常见结构和训练结果转换成内部
`Structure` 表示，用于显示、筛选、生成候选结构或导出。

## 30 秒快速判断

| 你手里的内容 | 能否直接打开 | 从哪里打开 | 打开后通常做什么 |
| --- | --- | --- | --- |
| `.xyz` / `.extxyz` | 可以 | `生成数据集` 或 `NEP Dataset Display` | 继续生成，或检查、筛选和导出 |
| `POSCAR` / `CONTCAR` / `.cif` | 可以 | `生成数据集` | 作为基础晶体生成候选结构 |
| `OUTCAR` / `XDATCAR` | 可以 | `NEP Dataset Display` | 检查或转换 VASP 结果 |
| `dump` / `lammpstrj` | 通常可以 | `NEP Dataset Display` | 查看 LAMMPS 轨迹；数值 `type` 需要元素映射 |
| DeepMD `deepmd/npy` 目录 | 可以，但必须是完整目录 | `NEP Dataset Display` | 查看或转换 DeepMD 数据 |
| `energy_*.out` / `force_*.out` / `virial_*.out` | 不能单独使用 | 与对应结构文件一起打开 | 回看预测值与参考值 |
| CP2K 优化或 MD 的多步主输出 | 不建议直接打开 | 先转换成 EXTXYZ | 保留逐步晶胞、坐标和标签 |
| 不在表中的结构或轨迹 | 不要只改扩展名 | 先转换成 EXTXYZ | 用 `Lattice`、`Properties` 和 `pbc` 明确数据契约 |

只想确认“我的 `.xyz` 能不能打开”时，先看文件首帧：原子数要与原子行一致；周期结构应有
有效 `Lattice`；逐原子列必须与 `Properties` 声明一致。满足这些条件的普通 XYZ/EXTXYZ
可以直接尝试导入。失败时根据错误信息检查具体字段，不要用补零或伪元素绕过。

:::{tip}
要继续生成新结构，进入 `生成数据集`；要看图、删结构、筛选或导出子集，进入
`NEP Dataset Display`。同一种文件在两个入口中的用途不同，但底层结构数据不会因为入口不同
自动增加 DFT 标签。
:::

对应操作入口：

- [在“生成数据集”中打开基础结构](module/make-dataset.md)
- [在“NEP Dataset Display”中打开结构、模型和训练输出](module/nep-display-open-data.md)
- [导入失败时按症状排查](reference/troubleshooting.md)

## 结构文件

| 文件 | 推荐入口 | 常见用途 |
| --- | --- | --- |
| `.xyz` / `.extxyz` | `生成数据集` 或 `NEP Dataset Display` | 作为初始结构、候选池或训练集 |
| `POSCAR` / `CONTCAR` | `生成数据集` | 作为初始晶体结构生成候选池 |
| `CIF` | `生成数据集` | 从晶体结构开始构建候选集 |
| ASE `.traj` | `NEP Dataset Display` | 查看已有轨迹或转换结构 |
| DeepMD `deepmd/npy` 目录 | `NEP Dataset Display` | 查看或导出 DeepMD 风格结构数据 |

如果你要继续生成新结构，优先导入 `生成数据集`。如果你要检查、删除、筛选或导出子集，
优先导入 `NEP Dataset Display`。

DeepMD 目录导入要求存在 `type.raw`、`type_map.raw`、`set.* / box.npy` 和
`set.* / coord.npy`，并且帧数、原子数、元素映射、晶胞和坐标必须一致。缺少这些契约时
会明确失败，不会用伪元素、单位晶胞或不完整帧继续加载。

### EXTXYZ 数据契约

`NEP Dataset Display` 会按扩展 XYZ 契约校验原子数、`Lattice`、`Properties`、元素符号、
逐原子列数和逻辑字段。文件声明的原子数与实际行数不一致、列缺失、非法元素、非法 PBC
或周期晶胞退化时会明确失败，不会用零值补齐。与 GPUMD/NEP 的 EXTXYZ 约定一致，
没有写 `pbc` 的帧默认按全周期 `T T T` 处理；非周期或部分周期数据必须在文件头明确写出。

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

## 高级结构格式

`NEP Dataset Display → Open File…` 的 `Advanced / experimental structure files` 筛选器
还会显示 CP2K 输出与 n2p2 `input.data`。这两个入口按文件内容识别格式，不会只因为扩展名
匹配就把损坏文件当成有效数据。

### CP2K 单点输出

- 支持带 CP2K 签名的 `.out` / `.log` 主输出。
- 当前入口只支持包含一组坐标和一组力的三维周期单点输出；优化或 MD 的多步历史应先转换为
  EXTXYZ，不能直接把多个坐标块拼成一个结构。
- 读取 `CELL| Vector a/b/c [angstrom]`、`MODULE QUICKSTEP: ATOMIC COORDINATES IN
  ANGSTROM`、Hartree 能量、原子单位力和 GPa stress，并转换到 eV、Å 和
  eV/Å³。
- 缺少完整 cell、没有坐标、力数量不一致、cell 退化或含多个坐标/力块时会明确失败。
- 当前 CP2K 主输出入口记录 `pbc="T T T"`，因此不要用于 `PERIODIC NONE` 或部分周期计算。
  这类计算应从 CP2K 导出带明确 `Lattice` 和 `pbc` 的 EXTXYZ 后再导入。

CP2K 的几何优化文档也建议使用轨迹或最终结构文件承载多步几何；主输出并不是完整轨迹容器。

### n2p2 `input.data`

- 支持名为 `input.data` 的文件以及 `.data` / `.cfg` 文件。
- 结构块必须由 `begin` / `end` 完整包围；每个 atom 行严格包含位置、元素、电荷、逐原子能量
  和三维力。
- n2p2 CFG 数值按 Bohr / Hartree 读取，并转换为 Å、eV 和 eV/Å。
- 0 或 3 行 lattice 都是合法契约；只有 1–2 行 lattice、损坏数值、非法元素或缺少 `end`
  会使整次导入失败。

## LAMMPS

- `dump`
- `lammpstrj`

支持正交和三斜晶胞，以及常见坐标列。导入后建议先抽查晶胞、元素类型和坐标单位是否符合预期。
每一帧必须包含完整的 `BOX BOUNDS` 边界标志、一个完整坐标三元组（`xs/ys/zs`、
`x/y/z` 或 `xu/yu/zu`）以及 `element` 或 `type` 列。仅有数值 `type` 时，需要由导入
界面或调用方提供完整的“类型编号 → 元素”映射；NepTrainKit 不会把未知类型替换成伪元素。
原子行截断、非法元素、缺失映射或退化晶胞都会使整次导入明确失败，不会返回不完整轨迹。

### 带自旋的推荐 dump

使用 spin LAMMPS 时，推荐按下面的列名输出磁矩：

```text
compute spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz
dump dpgen_dump all custom 100 traj.dump id type x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]
```

导入器把 `c_spin[1]` 解释为磁矩模长，把 `c_spin[2]`、`c_spin[3]`、`c_spin[4]` 解释为三个方向分量，并重建逐原子自旋向量：

$$
\mathbf s_i=c_{\mathrm{spin}[1]}
\left(c_{\mathrm{spin}[2]},c_{\mathrm{spin}[3]},c_{\mathrm{spin}[4]}\right).
$$

`c_spin[5]`、`c_spin[6]`、`c_spin[7]` 是磁力分量，不会写入 `spin:R:3`。如果缺少 `c_spin[1:4]` 中任意一列，导入器不会为该帧生成自旋字段。

## 导出

`NEP Dataset Display` 可以把当前数据导出为两类结果：

- `xyz` / `extxyz`：适合继续回到 GPUMD、ASE 或 生成数据集 流程。
- `DeepMD NPY`：标准 `deepmd/npy`。在导出弹窗中选择按 `Config_type` 或按化学式建立子目录。
- `DeepMD NPY (Mixed)`：把化学式可以不同的帧写入逐帧 `real_atom_types.npy`。弹窗中的“虚拟原子填充”与 dpdata 的 `atom_numb_pad` 一致：`0` 按精确原子数分目录；正整数会把原子数向上补到该数的倍数，使不同原子数的体系能够合并。补位类型为 `-1`，重新读取时会自动移除。

标准 NPY 选择按 `Config_type` 分组时，同一 `Config_type` 下的结构必须具有一致的元素与
原子数、PBC、逐原子字段和能量/virial 覆盖；一个 `Config_type` 包含多种化学式时应改选
按化学式分组，或改用 Mixed。Mixed 按填充后的原子数聚合，同一目录内仍必须具有一致的 PBC、
逐原子字段和标签覆盖。

两种 NPY 都保留全周期或全非周期 PBC；全非周期目录写入 `nopbc`。部分周期边界无法表示，
导出会在写文件前失败。填充值越大，能够合并的原子数越多，但坐标和逐原子数组也会更大；例如最大原子数为 64 时，填充值 64 可合为一个目录。标准 NPY 会按 `type_map` 重排原子及其逐原子字段；Mixed 保持每帧
原子顺序，并用 `real_atom_types.npy` 保存对应元素类型。

## 大数据和导入说明

- 大数据集导入时，界面显示可能做抽样或延迟渲染；底层选择和导出仍以完整数据为准。
- 导入过程支持中断。
- 不同来源的文件最好保留清晰命名，例如 `candidate_pool.xyz`、`candidate_pool_clean.xyz`、`dft_labeled.xyz`。
