# 应变、路径与扰动

这组卡片补充晶格响应、相变路径和近平衡位移。先判断需要改变的是晶胞、原子坐标，
还是已有振动模态；三者不要混用。

| 目标 | 使用卡片 |
| --- | --- |
| 扫描轴向晶格应变 | `Lattice Strain` |
| 随机缩放晶胞 | `Lattice Perturb` |
| 扫描 Bain 四方畸变路径 | `Bain Path` |
| 施加矩阵或角度剪切 | `Shear Matrix Strain` / `Shear Angle Strain` |
| 添加随机原子位移 | `Atomic Perturb` |
| 沿已有振动模态生成位移 | `Vib Mode Perturb` |

```{toctree}
:maxdepth: 1

../cards/cell-strain-card
../cards/cell-scaling-card
../cards/bain-path-card
../cards/shear-matrix-card
../cards/shear-angle-card
../cards/perturb-card
../cards/vibration-perturb-card
```
