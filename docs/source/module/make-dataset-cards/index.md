# 选择生成数据集卡片

这页先帮你选对卡片。选定后，再进入对应分类查看输入前提、参数、输出和检查方法：

- 我应该先选哪张卡
- 这张卡在什么场景下值得加
- 输出后我该怎么看结果是否合理

```{toctree}
:maxdepth: 1
:hidden:

recipes
categories/structure-cell
categories/deformation-perturbation
categories/composition-alloy
categories/defect-surface
categories/magnetism
categories/molecule-solvation
categories/filter-sampling
categories/workflow-metadata
```

```{contents} 本页目录
:local:
:depth: 1
```

## 快速上手路径

如果你是第一次使用“生成数据集”，推荐先按下面顺序理解：

1. 先看“按目标选卡”，确定主卡片。
2. 再看对应卡片页里的“操作示例”，确认参数量级。
3. 最后参考 [配方示例（Recipes）](recipes.md) 组织多卡流程。

:::{tip}
涉及 `FPS Filter` 的高通量流程，通常先导出 `xyz`，在 `NEP Dataset Display`
里清洗明显异常结构，再做代表性采样。清洗时可以用内置 NEP89 或当前体系已有模型做预筛，
但不要把它们当作 DFT 标签。`FPS Filter` 本身适合做末端代表性筛选，不适合替代结构生成卡片。
:::

## 30 秒确定方向

先判断你要改变什么，再进入对应小表。不要从四十张卡片中逐个试：

```text
没有初始结构
├─ 从标准晶型开始 → 晶体原型构建
└─ 从随机坐标开始 → 随机原子堆积

已经有初始结构
├─ 改晶胞或坐标 → 晶胞、应变与扰动
├─ 改元素和组成 → 合金与组成
├─ 改原子数或表面 → 缺陷与表面
├─ 改磁矩 → 磁性构型
├─ 做分子或溶剂 → 分子与溶剂
└─ 不再生成，只想清洗或选代表结构 → 过滤与采样
```

如果一条任务跨越多步，例如“生成表面并加入吸附原子”，先找改变边界条件的主卡，
再接局部操作卡。完整串联方式见[配方示例](recipes.md)。

## 按场景选卡

### 从哪里得到初始结构

| 我的需求 | 推荐卡片 | 使用前先确认 |
| --- | --- | --- |
| 从标准晶型直接生成结构 | [晶体原型构建](cards/crystal-prototype-builder-card.md) | 原型、晶格常数和元素映射符合目标体系 |
| 扩大现有晶胞，为缺陷或表面操作留空间 | [扩胞](cards/super-cell-card.md) | 这是重复晶胞，不会生成自由表面 |
| 在固定晶胞和组成下生成随机坐标初态 | [随机原子堆积](cards/random-packing-card.md) | 后接几何健全性过滤，不要把初态直接送 DFT |
| 复制层状结构并控制层间重复距离 | [分层堆叠](cards/layer-copy-card.md) | 输入已经是可重复的层状结构 |

### 改晶胞、应变或原子坐标

| 我的需求 | 推荐卡片 | 不要误用 |
| --- | --- | --- |
| 扫描指定轴向的拉伸和压缩 | [晶格应变](cards/cell-strain-card.md) | 用原子扰动改变晶格 |
| 给晶格长度和角度加随机变化 | [晶格随机扰动](cards/cell-scaling-card.md) | 把随机扰动当成严格应变路径 |
| 沿四方相变或外延方向扫描 `c/a` | [Bain 路径](cards/bain-path-card.md) | 用普通轴向应变替代系统路径 |
| 做矩阵剪切或晶格角剪切 | [剪切矩阵应变](cards/shear-matrix-card.md) / [剪切角应变](cards/shear-angle-card.md) | 用轴向应变模拟纯剪切 |
| 给近平衡结构加坐标噪声 | [原子扰动](cards/perturb-card.md) | 用大幅晶格扰动代替热扰动 |
| 沿已有振动模态生成位移 | [振动模式扰动](cards/vibration-perturb-card.md) | 输入没有模态数组时改用原子扰动 |

### 改元素、配比或占位

| 我的需求 | 推荐卡片 | 使用边界 |
| --- | --- | --- |
| 建立带晶体学子晶格的有序合金 | [有序合金原型](cards/ordered-alloy-prototype-card.md) | `sublattice` 是位点身份，不等同于普通 `group` |
| 枚举有限晶胞中整数可实现的组成和排布 | [有限晶胞合金占位](cards/finite-cell-alloy-occupancy-card.md) | 连续比例必须落到整数原子数 |
| 直接做一次随机替换或掺杂 | [随机掺杂](cards/random-doping-card.md) | 不适合系统扫描完整配比空间 |
| 先定义一组目标配比 | [成分空间采样](cards/composition-sweep-card.md) | 输出是目标配比标签，不代表占位已经改变 |
| 把目标配比真正落到离散位点 | [随机占位](cards/random-occupancy-card.md) | 可接在成分空间采样之后 |
| 沿空间方向形成配比梯度 | [成分梯度](cards/composition-gradient-card.md) | 输入需要足够多的层或位点 |
| 互混界面两侧的少数原子层 | [界面随机互混](cards/interface-layer-mix-card.md) | 输入需要界面法向明确的异种元素双层 |
| 按坐标、元素或分组规则替换位点 | [条件替换](cards/conditional-replace-card.md) | 复杂规则不要退化成无约束随机掺杂 |

### 生成缺陷、表面或层错

| 我的需求 | 推荐卡片 | 常见前置步骤 |
| --- | --- | --- |
| 生成不同晶面的表面板层 | [随机表面板层](cards/random-slab-card.md) | 先确认晶胞和 Miller 指数 |
| 在全部原子中按数量或比例随机删位 | [全局随机空位](cards/vacancy-defect-card.md) | 通常先扩胞 |
| 只删除指定元素或 `group` 中的位点 | [随机空位](cards/random-vacancy-card.md) | 需要时先做分组标记 |
| 插入间隙原子或表面吸附原子 | [插隙与表面吸附](cards/interstitial-adsorbate-card.md) | 体相通常先扩胞，吸附通常先建表面 |
| 扫描指定晶面和滑移方向的层错路径 | [层错 / GSFE 路径](cards/strict-gsfe-path-card.md) | 输入必须已经按目标晶面定向 |

### 生成磁性构型

| 我的需求 | 推荐卡片 | 使用前先确认 |
| --- | --- | --- |
| 只把磁矩写入结构 | [设置磁矩](cards/set-magnetic-moments-card.md) | 这不会自动生成多种磁序 |
| 生成 FM、AFM 或 PM 分支 | [磁序](cards/magnetic-order-card.md) | AFM 可能需要先做分组标记 |
| 从有序磁态逐步增加无序度 | [自旋无序](cards/spin-disorder-card.md) | 输入已有有效磁矩 |
| 生成具有空间相关长度的非共线磁矩 | [关联随机自旋](cards/correlated-random-spin-card.md) | 这是相关随机场，不应直接称为自旋玻璃 |
| 旋转、canting 或生成螺旋磁序 | [磁矩旋转](cards/magmom-rotation-card.md) / [小角度自旋倾斜](cards/small-angle-spin-tilt-card.md) / [自旋螺旋](cards/spin-spiral-card.md) / [折返螺旋磁序](cards/folded-helix-card.md) | 不要对没有磁矩的空白结构直接旋转 |

### 处理分子和溶剂

| 我的需求 | 推荐卡片 | 使用边界 |
| --- | --- | --- |
| 在周期边界中采样有机分子构象 | [有机构象采样](cards/organic-mol-config-pbc-card.md) | 输入应能识别分子和可旋转键 |
| 在溶质或离子周围补局部溶剂壳 | [局部溶剂壳](cards/local-solvation-card.md) | 只生成初始构型，不代表已经平衡 |
| 在周期晶胞中填充整盒溶剂 | [周期溶剂盒](cards/solvent-box-fill-card.md) | 晶胞必须有效且体积非零 |

### 清洗、采样或组织流程

| 我的需求 | 推荐卡片 | 放在流程哪里 |
| --- | --- | --- |
| 过滤短键、异常体积或异常密度 | [几何健全性过滤](cards/geometry-filter-card.md) | 放在强扰动、随机占位和缺陷生成之后 |
| 从干净候选池中选代表结构 | [代表性采样（FPS）](cards/fps-filter-card.md) | 先清洗异常结构，再采样 |
| 为后续规则或 AFM 标记 A/B 组 | [分组标记](cards/group-label-card.md) | 放在读取 `group` 的卡片之前 |
| 合并共享同一输入的多个分支 | [分支合并组](cards/card-group.md) | 子卡并行接收同一输入，不是顺序流水线 |
| 建立持续独立的多条卡片链 | [永久分叉](cards/workflow-fork.md) | 分支保持独立，只有显式 Merge 后才形成共同下游 |

## 易混卡片对比

### 随机表面板层 vs 全局随机空位

- `Random Slab` 改的是边界条件和表面取向，结果会引入真空层与自由表面。
- `Vacancy Defect Generation` 改的是体相或表面内部的删位强度，不会自动生成表面。
- 想研究“表面缺陷”，通常先 `Random Slab`，再 `Insert Defect` 或 `Vacancy Defect Generation`。

### 随机掺杂 vs 成分空间采样 vs 随机占位

- `Composition Space Sampling` 负责定义“目标配比空间”，输出仍是带目标配比标签的结构副本。
- `Random Occupancy` 负责把目标配比真正落到离散原子位点上。
- `Random Doping` 更适合“给定规则后直接做一次随机替换”，而不是系统地扫完整配比空间。

### 有序合金原型 vs 有限晶胞合金占位

- `Ordered Alloy Prototype` 建立晶胞、周期边界、分数坐标和 A/B 晶体学子晶格，不负责枚举组成。
- `Finite-Cell Alloy Occupancy` 接收已有位点或子晶格，先确定可实现的整数计数，再生成不重复排布。
- 要覆盖 L1₂、B2、L1₀ 的有序到部分无序路径，通常先生成原型，再做有限晶胞占位。

### 原子扰动 vs 振动模式扰动

- `Atomic Perturb` 是无模型的随机位移，适合快速补近平衡噪声。
- `Vib Mode Perturb` 基于已有振动模态，适合更接近特定频率空间的位移采样。
- 输入里没有模态数组时，不能直接用 `Vib Mode Perturb`。

### 层错只保留一个新建入口

- 新任务统一使用 `Stacking Fault / GSFE Path`，显式填写 `plane_hkl`、位于面内的 `slip_uvw`、切面位置和位移路径。
- 它不限材料和晶面，但要求输入已经定向：第三晶胞方向必须垂直于 `plane_hkl`。普通 cubic fcc cell 若要扫原始 `(111)` 面，应先用 `Crystal Prototype Builder` 的 `fcc111` 原型或自行构造已定向晶胞。
- 旧 `StackingFaultCard` 仍保留在序列化注册表中，用于载入历史 JSON；因为它自动从全局笛卡尔轴推导滑移方向，所以不再显示在“添加新卡片”和“查找卡片”中。

### 设置磁矩 vs 磁序 vs 磁矩旋转

- `Set Magnetic Moments` 只负责把磁矩写进去，适合静态初始化。
- `Magnetic Order` 会生成 FM / AFM / PM 等多磁态分支。
- `Magmom Rotation` 基于已有磁矩做角度扰动，适合补充非共线或局部旋转样本。

磁性卡片对外统一使用 NEP / Show NEP 的 `spin:R:3` EXTXYZ 字段。卡片内部仍同步维护 ASE 的 `initial_magmoms`，用于兼容 ASE 操作和旧工作流；导出文件只保留 `spin`。读取已有磁矩时优先使用 `spin`，只有输入没有 `spin` 时才回退到旧的 `initial_magmoms`。标量模式会按卡片中明确设置的 `Axis` 转成三分量 `spin`；零向量 Axis 会明确失败，不会静默假设方向。

## 按功能浏览

| 我正在处理什么 | 进入这里 |
| --- | --- |
| 初始结构、晶胞大小或层状结构 | [结构来源与晶胞](categories/structure-cell.md) |
| 应变、相变路径、原子或模态扰动 | [应变、路径与扰动](categories/deformation-perturbation.md) |
| 合金原型、配比、占位或替换 | [合金与组成](categories/composition-alloy.md) |
| 表面、空位、插隙、吸附或层错 | [缺陷与表面](categories/defect-surface.md) |
| 共线、非共线和空间相关磁构型 | [磁性构型](categories/magnetism.md) |
| 有机构象、局部溶剂壳和周期溶剂盒 | [分子与溶剂](categories/molecule-solvation.md) |
| 几何清洗或代表性采样 | [过滤与代表性采样](categories/filter-sampling.md) |
| 分组标签、分支和输出合并 | [工作流与元数据](categories/workflow-metadata.md) |

旧版 `Stacking Fault` 只用于载入历史 JSON，不再属于新任务的卡片目录。迁移方法见
[旧版层错位移](cards/stacking-fault-card.md)。
