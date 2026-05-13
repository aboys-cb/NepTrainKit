# Make Dataset 卡片手册

这份手册不只解释“每个参数是什么意思”，更强调三件事：

- 我应该先选哪张卡
- 这张卡在什么场景下值得加
- 输出后我该怎么看结果是否合理

维护规范见 [卡片文档编写规范](writing-guide.md)。

```{toctree}
:maxdepth: 1
:hidden:

writing-guide
recipes
cards/super-cell-card
cards/crystal-prototype-builder-card
cards/perturb-card
cards/vibration-perturb-card
cards/set-magnetic-moments-card
cards/magmom-rotation-card
cards/small-angle-spin-tilt-card
cards/spin-disorder-card
cards/correlated-random-spin-card
cards/spin-spiral-card
cards/folded-helix-card
cards/cell-strain-card
cards/cell-scaling-card
cards/shear-matrix-card
cards/shear-angle-card
cards/random-slab-card
cards/random-doping-card
cards/composition-sweep-card
cards/composition-gradient-card
cards/random-occupancy-card
cards/conditional-replace-card
cards/random-packing-card
cards/group-label-card
cards/magnetic-order-card
cards/random-vacancy-card
cards/vacancy-defect-card
cards/stacking-fault-card
cards/interstitial-adsorbate-card
cards/organic-mol-config-pbc-card
cards/local-solvation-card
cards/solvent-box-fill-card
cards/layer-copy-card
cards/fps-filter-card
cards/geometry-filter-card
cards/card-group
```

## 快速上手路径

如果你是第一次使用 Make Dataset，推荐先按下面顺序理解：

1. 先看“按目标选卡”，确定主卡片。
2. 再看对应卡片页里的“操作示例”，确认参数量级。
3. 最后参考 [配方示例（Recipes）](recipes.md) 组织多卡流程。

:::{tip}
涉及 `FPS Filter` 的高通量流程，通常先导出 `xyz`，在 `NEP Dataset Display`
里清洗明显异常结构，再做代表性采样。清洗时可以用内置 NEP89 或当前体系已有模型做预筛，
但不要把它们当作 DFT 标签。`FPS Filter` 本身适合做末端代表性筛选，不适合替代结构生成卡片。
:::

## 按目标选卡

| 我的需求 | 推荐卡片 | 常见前置卡片 | 不要误用 |
| --- | --- | --- | --- |
| 扩大晶胞尺寸，为缺陷或表面操作留空间 | `Super Cell` | `Crystal Prototype Builder` | 把 `Random Slab` 当成扩胞工具 |
| 从晶体原型直接生成一批标准结构 | `Crystal Prototype Builder` | 无 | 用 `Super Cell` 手工拼基础晶型 |
| 在固定 cell 和组成下生成无序原子坐标初态 | `Random Packing` | `Geometry Filter` | 把它当成磁矩无序卡 |
| 过滤短键、异常体积或异常密度结构 | `Geometry Filter` | 强扰动、随机占位、表面或缺陷生成卡 | 把 `FPS Filter` 当成几何质量检查 |
| 给近平衡结构加轻微坐标噪声 | `Atomic Perturb` | `Super Cell` / 已弛豫输入 | 用大幅 `Lattice Perturb` 代替原子热扰动 |
| 给晶胞参数做体积或轴向缩放 | `Lattice Perturb` / `Lattice Strain` | `Super Cell` | 用 `Atomic Perturb` 改晶格 |
| 做剪切应变或角度应变 | `Shear Matrix Strain` / `Shear Angle Strain` | 已知目标应变方向 | 用 `Lattice Strain` 强行模拟纯剪切 |
| 生成表面切片 | `Random Slab` | `Super Cell` | 用 `Vacancy Defect Generation` 做表面 |
| 做单点随机合金 | `Random Doping` | `Composition Sweep` 可选 | 用 `Composition Sweep` 代替具体占位落点 |
| 扫描多种目标配比 | `Composition Sweep` | 无 | 用 `Random Doping` 手工凑配比网格 |
| 沿空间方向做配比梯度 | `Composition Gradient` | 已扩胞且有足够层数的结构 | 用全局随机占位假装扩散偶或梯度层 |
| 把目标配比真正落到原子占位上 | `Random Occupancy` | `Composition Sweep` | 只做 `Composition Sweep` 就当已经生成随机合金 |
| 按条件替换某类位点 | `Conditional Replace` | `Group Label` 可选 | 用 `Random Doping` 做规则替换 |
| 做随机空位族 | `Vacancy Defect Generation` | `Super Cell` | 用 `Random Slab` 生成空位表面 |
| 按元素或 group 精细删位 | `Random Vacancy` | `Group Label` 可选 | 用 `Vacancy Defect Generation` 写复杂规则 |
| 做插隙或吸附缺陷 | `Insert Defect` | `Random Slab` / `Super Cell` | 用 `Random Doping` 代替插入 |
| 做层错样本 | `Stacking Fault` | `Super Cell` | 用 `Random Slab` 代替层错位移 |
| 给结构打分组标签，供后续分组操作使用 | `Group Label` | `Super Cell` | 直接在磁卡里假设已有 group |
| 初始化 FM / AFM / PM 磁序 | `Magnetic Order` | `Group Label` 可选 | 用 `Set Magnetic Moments` 代替多磁态生成 |
| 生成 FM/AFM 到 PM 之间的无序度梯度 | `Spin Disorder` | `Set Magnetic Moments` / `Magnetic Order` | 把离散翻转塞进 `Magmom Rotation` |
| 生成有空间相关长度的非共线随机磁矩 | `Correlated Random Spin` | `Set Magnetic Moments` / `Magnetic Order` | 把它叫成 `Spin Glass` |
| 只想把磁矩写到结构里，不想生成多磁态分支 | `Set Magnetic Moments` | 无 | 用 `Magnetic Order` 做静态赋值 |
| 基于已有磁矩做旋转、多步 canting、全局偏转或螺旋 | `Magmom Rotation` / `Small-Angle Spin Tilt` / `Spin Spiral` / `Folded Helix` | `Set Magnetic Moments` / `Magnetic Order` | 直接拿空白结构做旋转 |
| 从振动模式生成位移样本 | `Vib Mode Perturb` | 已包含模态数组的结构 | 用 `Atomic Perturb` 代替模态扰动 |
| 从有机分子构象空间采样 | `Organic Mol Config` | 已识别分子结构 | 用无机卡片强行扰动有机体系 |
| 在局部离子、极性中心或溶质周围补溶剂壳 | `Local Solvation` | 已有溶质或离子结构 | 把它当成平衡溶剂化或量化优化 |
| 在周期 cell 中生成整盒溶剂初态 | `Solvent Box Fill` | 已有非奇异周期 cell | 用局部溶剂壳卡替代整盒填充 |
| 做容器化分支流程 | `Card Group` | 任意共享输入 | 把 `Card Group` 当成筛选器 |
| 从干净候选池中选代表性结构 | `FPS Filter` | `NEP Dataset Display` 清洗后的候选池 | 把 `FPS Filter` 当成第一道质量检查 |

## 易混卡片对比

### `Random Slab` vs `Vacancy Defect Generation`

- `Random Slab` 改的是边界条件和表面取向，结果会引入真空层与自由表面。
- `Vacancy Defect Generation` 改的是体相或表面内部的删位强度，不会自动生成表面。
- 想研究“表面缺陷”，通常先 `Random Slab`，再 `Insert Defect` 或 `Vacancy Defect Generation`。

### `Random Doping` vs `Composition Sweep` vs `Random Occupancy`

- `Composition Sweep` 负责定义“目标配比空间”，输出仍是带目标配比标签的结构副本。
- `Random Occupancy` 负责把目标配比真正落到离散原子位点上。
- `Random Doping` 更适合“给定规则后直接做一次随机替换”，而不是系统地扫完整配比空间。

### `Atomic Perturb` vs `Vib Mode Perturb`

- `Atomic Perturb` 是无模型的随机位移，适合快速补近平衡噪声。
- `Vib Mode Perturb` 基于已有振动模态，适合更接近特定频率空间的位移采样。
- 输入里没有模态数组时，不能直接用 `Vib Mode Perturb`。

### `Set Magnetic Moments` vs `Magnetic Order` vs `Magmom Rotation`

- `Set Magnetic Moments` 只负责把磁矩写进去，适合静态初始化。
- `Magnetic Order` 会生成 FM / AFM / PM 等多磁态分支。
- `Magmom Rotation` 基于已有磁矩做角度扰动，适合补充非共线或局部旋转样本。

## 按分组浏览

- `Lattice`: `Super Cell`、`Crystal Prototype Builder`、`Random Packing`、`Lattice Strain`、`Lattice Perturb`、`Shear Matrix Strain`、`Shear Angle Strain`
- `Perturbation`: `Atomic Perturb`、`Vib Mode Perturb`
- `Alloy`: `Composition Sweep`、`Composition Gradient`、`Random Occupancy`、`Random Doping`、`Conditional Replace`
- `Defect / Surface`: `Random Slab`、`Random Vacancy`、`Vacancy Defect Generation`、`Insert Defect`、`Stacking Fault`、`Layer Copy`
- `Magnetism`: `Set Magnetic Moments`、`Magnetic Order`、`Spin Disorder`、`Correlated Random Spin`、`Magmom Rotation`、`Small-Angle Spin Tilt`、`Spin Spiral`、`Folded Helix`
- `Filter / Container`: `Geometry Filter`、`FPS Filter`、`Card Group`
- `Organic`: `Organic Mol Config`、`Local Solvation`、`Solvent Box Fill`
