# 晶格

与卡片 UI 的“晶格”分组一致。这些卡片负责生成晶体原型、扩胞，或者系统改变晶胞的
长度、角度与形状。它们与只移动原子坐标的“扰动”卡片分开。

| 目标 | 使用卡片 |
| --- | --- |
| 从标准晶体原型开始 | `晶体原型构建` |
| 扩大已有晶胞 | `扩胞` |
| 扫描轴向应变或随机晶格变化 | `晶格应变` / `晶格随机扰动` |
| 扫描 Bain 四方畸变 | `Bain 路径` |
| 施加矩阵或角度剪切 | `剪切矩阵应变` / `剪切角应变` |

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
