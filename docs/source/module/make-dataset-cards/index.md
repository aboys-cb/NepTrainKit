# Make Dataset 卡片手册

本手册提供所有卡片的参数级说明与推荐实践。

维护规范： [卡片文档编写规范](writing-guide.md)

```{toctree}
:maxdepth: 1
:hidden:

writing-guide
recipes
cards/super-cell-card
cards/crystal-prototype-builder-card
cards/perturb-card
cards/vibration-perturb-card
cards/magmom-rotation-card
cards/spin-spiral-card
cards/cell-strain-card
cards/cell-scaling-card
cards/shear-matrix-card
cards/shear-angle-card
cards/random-slab-card
cards/random-doping-card
cards/composition-sweep-card
cards/random-occupancy-card
cards/conditional-replace-card
cards/group-label-card
cards/magnetic-order-card
cards/random-vacancy-card
cards/vacancy-defect-card
cards/stacking-fault-card
cards/interstitial-adsorbate-card
cards/organic-mol-config-pbc-card
cards/layer-copy-card
cards/fps-filter-card
cards/card-group
```

## 快速入口

:::{tip}
涉及 `FPS Filter` 的流程，建议先导出 `xyz`，在第一个模块用 `nep89` 做预测清洗，再进行 FPS。
:::

- 晶格：Super Cell / Lattice Strain / Lattice Perturb
- 缺陷：Random Vacancy / Insert Defect / Stacking Fault
- 合金：Random Doping / Composition Sweep / Conditional Replace
- 磁性：Magnetic Order / Magmom Rotation
- 过滤：FPS Filter
- 容器：Card Group
