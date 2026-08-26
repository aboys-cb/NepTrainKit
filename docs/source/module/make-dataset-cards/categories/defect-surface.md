# 缺陷

与卡片 UI 的“缺陷”分组一致。这些卡片负责删位、插入原子、加入吸附原子或生成层错路径。
自由表面的构造单独放在“表面”分组；表面缺陷通常先建表面，再接这里的局部操作。

```{toctree}
:maxdepth: 1

../cards/random-vacancy-card
../cards/vacancy-defect-card
../cards/interstitial-adsorbate-card
../cards/strict-gsfe-path-card
```

旧版 `Stacking Fault` 已从添加菜单隐藏，只用于历史 JSON 兼容。新任务统一使用
`Stacking Fault / GSFE Path`。
