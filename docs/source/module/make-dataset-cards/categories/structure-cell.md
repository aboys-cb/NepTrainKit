# 晶格

与卡片 UI 的“晶格”分组一致。这些卡片负责生成晶体原型、扩胞，或者系统改变晶胞的
长度、角度与形状。它们与只移动原子坐标的“扰动”卡片分开。

| 目标 | 使用卡片 |
| --- | --- |
| 从标准晶体原型开始 | `Crystal Prototype Builder` |
| 扩大已有晶胞 | `Super Cell` |
| 扫描轴向应变或随机晶格变化 | `Lattice Strain` / `Lattice Perturb` |
| 扫描 Bain 四方畸变 | `Bain Path` |
| 施加矩阵或角度剪切 | `Shear Matrix Strain` / `Shear Angle Strain` |

```{toctree}
:maxdepth: 1

../cards/crystal-prototype-builder-card
../cards/super-cell-card
../cards/cell-strain-card
../cards/cell-scaling-card
../cards/bain-path-card
../cards/shear-matrix-card
../cards/shear-angle-card
```
