# Make Dataset 配方示例（Recipes）

:::{tip}
推荐流程：生成结构后先导出 `xyz`，导入第一个模块使用 `nep89` 预测并删除不合理结构，再做最远点采样（FPS）。
:::

## 高熵合金

`Composition Sweep -> Random Occupancy -> Random Doping -> 导出 xyz -> 模块1(NEP89 预测清洗) -> FPS`

## 富缺陷表面

`Random Slab -> Insert Defect -> Random Vacancy -> 导出 xyz -> 模块1(NEP89 预测清洗) -> FPS`

## 磁性数据

`Group Label -> Magnetic Order -> Magmom Rotation`

## 有机构象

`Organic Mol Config -> Atomic Perturb or Lattice Perturb -> 导出 xyz -> 模块1(NEP89 预测清洗) -> FPS`
