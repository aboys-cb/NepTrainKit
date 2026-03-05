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
