# NepTrainKit 原生加速模块架构

## 目标

原生代码只负责稳定、可复用的计算原语，Python 层继续负责产品规则、审查结论和 UI。新增加速功能不再直接堆进 `NepTrainKit/core`，也不再使用含义模糊的 `_fast*` 名称。

## 目录与职责

```text
src/native/
├── include/neptrainkit/native/
│   └── periodic_neighbors.hpp  # 共享周期性邻居深模块
├── io/
│   ├── module.cpp              # EXTXYZ 解析边界
│   └── fast_float.h
├── audit/
│   └── module.cpp              # Audit 批处理与聚合原语
├── magnetism/
│   └── module.cpp              # 磁矩布局与统计原语
├── phase/
│   └── module.cpp              # PhaseSketch 与候选相细化原语
└── sampling/
    └── module.cpp              # FPS 距离场更新原语

src/NepTrainKit/_native/
├── _io.*
├── _audit.*
├── _magnetism.*
├── _phase.*
└── _sampling.*

src/NepTrainKit/core/
└── geometry_cache.py            # 跟随 StructureData 生命周期的只读几何快照
```

NEP 模型计算由独立的 `nep-adapters` 运行时提供；它不属于应用层 `_native` 模块，也不在 NepTrainKit 内复制后端实现。

## 共享邻居契约

`PeriodicNeighborSearch<Scalar>` 是 Audit 和 Phase 唯一的周期性邻居实现：

- 晶格按行向量解释，分数坐标满足 `fractional @ cell`。
- 支持正交、非正交、全周期、部分周期和非周期结构。
- 周期晶格必须可逆；奇异输入直接报错，不静默近似。
- 排除中心原子的零平移自映像；允许同一原子的非零周期映像。
- KNN 结果采用确定性排序；半径查询保持 Audit 的严格 `< cutoff` 语义。
- Phase 与碰撞识别使用 `float`；需要保留归一化距离细节的 Local Chemistry 使用 `double`。
- 小结构使用直接扫描，较大结构使用 cell-list；这个选择封装在模块内部，调用方不感知。

## Dataset 几何快照

`StructureData.geometry_snapshot()` 是重复结构分析的共享 seam：

- 首次请求时把 positions、cell、PBC、原子序数和 atom offsets 整理成连续只读数组；
- 删除和撤销只改变激活投影，不重建不变的底层几何；
- 最近一次激活/选定范围的投影会复用；
- 数据质量、局域化学、非物理结构识别和后续相分析读取同一份快照；
- 新载入数据会创建新的 `StructureData`，因此缓存自然随 dataset 一起销毁，不使用全局缓存。

接口假设 `StructureData` 的底层 Structure 对象在生命周期内不被原地修改；当前产品的删除/撤销只修改掩码，符合这一契约。

## Python/C++ 边界

- `_io`：只解析和索引 EXTXYZ，不理解训练集审查规则。
- `_audit`：只做批量几何搜索、标签数组聚合和数值统计，不生成产品结论。
- `_magnetism`：只做磁矩布局识别和批量统计，不决定磁性分类文案。
- `_phase`：只做局域特征与候选相数值指标；`phase_partition_primitives` 一次生成可复用邻居场和 a-CNA 标签，`common_prototype_mapping_metrics` 批量计算单个原型映射的几何/化学门槛指标，均不决定物理分层、UI 文案或下一步建议。
- `_sampling`：只做 FPS 距离场更新，不决定物理分层、配额或产品策略。
- Python 层负责输入适配、审查阈值、finding、排序、报告和交互。
- 原生模块是可选加速层；模块未编译、加载失败或输入不受支持时，保留等价的 Python/NumPy 路径，避免功能整体失效。

一个能力只应进入一个领域模块。只有出现第二个真实调用方的计算原语，才下沉到 `include/neptrainkit/native` 共享层。

## 验证门禁

修改共享邻居实现时至少需要同时通过：

1. 正交与三斜晶格的 reference/oracle 对比；
2. Audit 半径查询、短距离扫描和局域化学聚合测试；
3. Phase KNN、PhaseSketch 和候选相细化测试；
4. OpenMP 与无 OpenMP 两种构建；
5. 真实小帧数据与中大型结构的 A/B 性能检查。

正确性失败或真实吞吐退化超过门禁的实现不保留。
