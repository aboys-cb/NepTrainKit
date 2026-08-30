# 扰动

与卡片 UI 的“扰动”分组一致。这两张卡片只改变原子坐标：一种加入无模型随机位移，
另一种沿输入中已有的振动模态位移；它们不会扫描晶胞应变。

| 目标 | 使用卡片 |
| --- | --- |
| 添加随机原子位移 | `Atomic Perturb` |
| 沿已有振动模态生成位移 | `Vib Mode Perturb` |

```{toctree}
:maxdepth: 1

../cards/perturb-card
../cards/vibration-perturb-card
```
