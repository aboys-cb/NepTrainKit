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
