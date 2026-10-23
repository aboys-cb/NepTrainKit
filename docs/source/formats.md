# 支持格式

这页按“你手里有什么文件”来选入口。NepTrainKit 会把常见结构和训练结果转换成内部
`Structure` 表示，用于显示、筛选、生成候选结构或导出。

## 30 秒快速判断

| 你手里的内容 | 能否直接打开 | 从哪里打开 | 打开后通常做什么 |
| --- | --- | --- | --- |
| `.xyz` / `.extxyz` | 可以 | `构建数据集` 或 `NEP 数据集查看` | 继续生成，或检查、筛选和导出 |
| `POSCAR` / `CONTCAR` / `.cif` | 可以 | `构建数据集` | 作为基础晶体生成候选结构 |
| `OUTCAR` / `XDATCAR` | 可以 | `NEP 数据集查看` | 检查或转换 VASP 结果 |
| `dump` / `lammpstrj` / `.gz` | 可以 | `NEP 数据集查看` | 查看 LAMMPS 轨迹；数值 `type` 会先从同目录 data/in 文件解析，仍缺时再询问 |
| DeepMD `deepmd/npy` 目录 | 可以，但必须是完整目录 | `NEP 数据集查看` | 查看或转换 DeepMD 数据 |
| `energy_*.out` / `force_*.out` / `virial_*.out` | 不能单独使用 | 与对应结构文件一起打开 | 回看预测值与参考值 |
| CP2K 优化或 MD 的多步主输出 | 不建议直接打开 | 先转换成 EXTXYZ | 保留逐步晶胞、坐标和标签 |
| 不在表中的结构或轨迹 | 不要只改扩展名 | 先转换成 EXTXYZ | 用 `Lattice`、`Properties` 和 `pbc` 明确的数据格式 |

只想确认“我的 `.xyz` 能不能打开”时，先看文件首帧：原子数要与原子行一致；周期结构应有
有效 `Lattice`；逐原子列必须与 `Properties` 声明一致。满足这些条件的普通 XYZ/EXTXYZ
可以直接尝试导入。失败时根据错误信息检查具体字段，不要用补零或伪元素绕过。

:::{tip}
要继续生成新结构，进入 `构建数据集`；要看图、删结构、筛选或导出子集，进入
`NEP 数据集查看`。同一种文件在两个入口中的用途不同，但底层结构数据不会因为入口不同
自动增加 DFT 标签。
:::

对应操作入口：

- [在“构建数据集”中打开基础结构](module/make-dataset.md)
- [在“NEP 数据集查看”中打开结构、模型和训练输出](module/nep-display-open-data.md)
- [导入失败时按症状排查](reference/troubleshooting.md)

## 结构文件

| 文件 | 推荐入口 | 常见用途 |
| --- | --- | --- |
| `.xyz` / `.extxyz` | `构建数据集` 或 `NEP 数据集查看` | 作为初始结构、候选池或训练集 |
| `POSCAR` / `CONTCAR` | `构建数据集` | 作为初始晶体结构生成候选池 |
| `CIF` | `构建数据集` | 从晶体结构开始构建候选集 |
| ASE `.traj` | `NEP 数据集查看` | 查看已有轨迹或转换结构 |
| DeepMD `deepmd/npy` 目录 | `NEP 数据集查看` | 查看或导出 DeepMD 风格结构数据 |

如果你要继续生成新结构，优先导入 `构建数据集`。如果你要检查、删除、筛选或导出子集，
优先导入 `NEP 数据集查看`。

DeepMD 目录导入要求存在 `type.raw`、`type_map.raw`、`set.* / box.npy` 和
`set.* / coord.npy`，并且帧数、原子数、元素映射、晶胞和坐标必须一致。缺少这些信息时
会明确失败，不会用伪元素、单位晶胞或不完整帧继续加载。

### EXTXYZ 格式要求

`NEP 数据集查看` 会按扩展 XYZ 格式要求校验原子数、`Lattice`、`Properties`、元素符号、
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

训练结束后，把训练结构和这些输出一起导入 `NEP 数据集查看`，可以查看散点图、
定位误差最大的结构，并导出下一轮要处理的子集。

## VASP

- `OUTCAR`：读取晶胞、坐标、力、应力和 virial。
- `XDATCAR`：读取逐帧晶胞和坐标。

VASP 结果通常用于把 DFT 标注转成训练结构，或在 `NEP 数据集查看` 中检查标签质量。

## 高级结构格式

`NEP 数据集查看 → 打开文件…` 的 `Advanced / experimental structure files` 筛选器
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
- 0 或 3 行 lattice 都是合法格式；只有 1–2 行 lattice、损坏数值、非法元素或缺少 `end`
  会使整次导入失败。

## LAMMPS

- `dump`
- `lammpstrj`
- `dump.gz` / `lammpstrj.gz`（`custom/gz` 输出）

支持正交、斜方（restricted triclinic）以及 `dump_modify triclinic/general` 的通用三斜晶胞，
也支持 gzip 压缩的 dump。导入后建议先抽查晶胞、元素类型和坐标单位是否符合预期。

每一帧需要一个完整坐标三元组（`xs/ys/zs`、`x/y/z`、`xu/yu/zu`、`xsu/ysu/zsu` 之一）
以及 `element`、`typelabel` 或 `type` 列。用 `dump_modify element` 或
`dump_modify types labels` 可以直接把元素符号写进 dump；只有数值 `type` 时，导入器按以下
顺序解析：

1. 调用方显式提供的“类型编号 → 元素”映射；
2. 同目录 `*.data` / `data.*` 文件里的 `Atom Type Labels` 段；
3. 同目录 `*.data` 的 `Masses` 段，以及 `in.*` 脚本里的 `mass` 命令（仅当质量在 0.1%
   以内唯一匹配标准原子量时采用，避免把粗粒化质量猜成元素）；
4. 同目录 `in.*` 脚本里的 `labelmap atom` 命令；
5. 导入界面的“类型 → 元素”输入框。

自动线索只采信与 dump 明显相关的文件：data 文件必须声明**相同的原子数**（或与 dump 同名），
`in.*` 脚本必须提到该 dump 文件名（或与 dump 同名）。共享目录（例如桌面）里其他体系的
data/输入文件不会参与猜测；因此对“只 dump 了某个 group”的轨迹，即使真实 data 文件在旁
也会改为询问。

无法自动确定映射时会打开导入对话框：显示帧数、每帧原子数、晶胞、列名，并按类型列出元素
输入框（可直接输入符号/原子序数、自动补全，也可从元素周期表选择，已识别的会预填）。
“调整字段映射”里可以修改坐标三元组、是否读取力、自旋定义（模长 + 方向 / 三列向量 /
忽略）以及额外列；打开时填的就是自动识别的结果，只有改动过的部分才会作为覆盖保存。
LAMMPS 的四列自旋（模长、方向 X/Y/Z）会重建成结构里的 3 分量自旋向量；额外列可以勾选
保留/忽略，也可以直接重命名输出字段（名称需唯一、非空，且不能与内置字段重名）。

“自定义字段”可以把多列合成一个字段：填一个名称，再用逗号分隔列名（例如 `foo` 和
`c_foo[1], c_foo[2], c_foo[3]`），导入后就是 `foo:R:3`。被占用的列不会重复出现在额外列里；
自定义字段名同样需要唯一，且不能与坐标/力/自旋占用的列重叠。同一处还可以选择是否按镜像标志展开坐标、保留额外列、损坏帧直接报错，以及“记住
此文件夹的映射”；勾选记住后，同一文件夹、相同原子数与类型集合的 dump 再次导入时不再
询问，列覆盖也会一起记住。

多个文件给出互相矛盾的类型映射时，冲突类型不会被猜测，而是留给界面确认。仍然无法解析的
类型会让该次导入明确失败，并列出具体类型编号；NepTrainKit 不会把未知类型替换成伪元素。

其余数值列不会丢弃：`vx/vy/vz` 合并为 `vel:R:3`，其他列（`q`、`mass`、`c_pe`、`c_ID`、
`v_name` 等）保留为 `dump.<列名>` 逐原子字段，导出 extxyz 时一并写出。`id`、`mol`、
`proc`、`procp1` 与镜像标志 `ix/iy/iz` 属于簿记信息，不写入结构；`ix/iy/iz` 可在
`unwrap=True` 时用于展开坐标。

最后帧截断（例如 LAMMPS 被中断，只写出半帧）会丢弃该帧并给出警告，保留前面的完整帧；
帧头损坏或缺少必要列仍然让导入失败。需要“任何异常都报错”的调用方可以使用 `strict=True`。

### 带自旋的推荐 dump

使用 spin LAMMPS 时，推荐按下面的列名输出磁矩：

```text
compute spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz
dump dpgen_dump all custom 100 traj.dump id type x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]
```

对 compute 名为 `spin` 或 `sp` 的 `c_<名称>[1..4]`，导入器把 `[1]` 解释为磁矩模长，把
`[2]`、`[3]`、`[4]` 解释为三个方向分量，并重建逐原子自旋向量：

$$
\mathbf s_i=c_{\mathrm{spin}[1]}
\left(c_{\mathrm{spin}[2]},c_{\mathrm{spin}[3]},c_{\mathrm{spin}[4]}\right).
$$

只有 3 列（`c_<名称>[1..3]`，或 `c_<名称>[2..4]`）时按向量直接读取；`spx/spy/spz` 也按向量
读取。`c_spin[5]`、`c_spin[6]`、`c_spin[7]` 是磁力分量，不会写入 `spin:R:3`，但会作为
`dump.c_spin[5]` 等字段保留。缺少完整的三分量时，该帧不会生成自旋字段。

原子表解析优先使用随包分发的原生扩展 `NepTrainKit._native._dump`，并按帧并行；线程数默认
使用可用核心数，可用 `NEPKIT_FASTDUMP_THREADS` 限制。扩展缺失或不兼容时自动回退到等价的
Python 参考实现。排查两条路径的差异时，可以设置环境变量 `NEPKIT_DISABLE_FASTDUMP=1` 强制
使用 Python 路径。

## 导出

`NEP 数据集查看` 可以把当前数据导出为两类结果：

- `xyz` / `extxyz`：适合继续回到 GPUMD、ASE 或构建数据集流程。
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
