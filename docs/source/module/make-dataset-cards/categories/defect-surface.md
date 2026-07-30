# 缺陷与表面

这组卡片改变表面边界或原子数，也包含显式层错路径。表面生成、删位、插入原子和层间滑移
是不同操作；需要组合时，按“先构造边界，再引入局部缺陷”的顺序执行。

```{toctree}
:maxdepth: 1

../cards/random-slab-card
../cards/random-vacancy-card
../cards/vacancy-defect-card
../cards/interstitial-adsorbate-card
../cards/strict-gsfe-path-card
```

旧版 `Stacking Fault` 已从添加菜单隐藏，只用于历史 JSON 兼容。新任务统一使用
`Stacking Fault / GSFE Path`。
