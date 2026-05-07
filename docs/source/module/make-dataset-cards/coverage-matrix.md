# Make Dataset 覆盖矩阵

这页不按“还能做多少张卡”整理，而按训练集还缺哪些物理状态来判断：现有卡能明确覆盖的，不新建；缺的是串联方法的，写 recipe；确实改变生成语义的，才进入新卡或扩展旧卡。

## 决策规则

| 判断问题 | 结论 |
| --- | --- |
| 现有卡能否用明确参数直接生成目标状态？ | 已覆盖，只补文档入口或 recipe。 |
| 目标只是多张现有卡的固定顺序？ | 写 recipe，不新增卡。 |
| 目标改变了单张卡的核心物理语义？ | 新卡优先，避免把离散无序、连续扰动、筛选后处理混在一个控件里。 |
| 目标是同一物理语义下的确定性变体？ | 扩展旧卡。 |
| 目标需要额外模型假设且不同体系差异很大？ | 暂缓，先写边界说明。 |

## Lattice / 晶格

### 现有覆盖

| 方向 | 现有卡 | 当前状态 |
| --- | --- | --- |
| 构建基础晶体 | `Crystal Prototype Builder` | 已覆盖简单原型 |
| 扩胞 | `Super Cell` | 已覆盖 |
| 轴向/体积应变 | `Lattice Strain` | 已覆盖，可用于 EOS |
| 随机晶格扰动 | `Lattice Perturb` | 已覆盖近平衡随机体积/角度扰动 |
| 剪切应变 | `Shear Matrix Strain`, `Shear Angle Strain` | 已覆盖 |
| 原子热扰动 | `Atomic Perturb`, `Vib Mode Perturb` | 半覆盖 |

### 缺口判断

| 优先级 | 建议 | 解决什么 | 决策 |
| --- | --- | --- | --- |
| P1 | `Geometry Filter` | 自动过滤短键、密度异常、体积异常结构，应放在 `FPS Filter` 前 | 新卡 |
| P1 | EOS / 声子 / 弹性常数串卡流程 | 用户不知道怎么组合 `Lattice Strain` / `Vib Mode Perturb` / 剪切卡 | recipe |
| P2 | `Disordered Seed / Random Pack` | 保持 cell 和组成，随机重排坐标，并施加最小距离约束 | 新卡 |
| P2 | `Vib Mode Perturb` thermal sampling | 有模态和频率时按温度采样，比无模型随机位移更接近热振动 | 扩展旧卡 |
| P3 | `Interface Builder` | 异质结、电极-电解质界面、界面间距和失配处理 | 新卡，后置 |

`Atomic Perturb` 不应直接包装成 temperature mode。质量缩放只能近似速度分布，不等价于真实热位移采样。

## Spin / 磁性

### 现有覆盖

| 方向 | 现有卡 | 当前状态 |
| --- | --- | --- |
| 设置磁矩 | `Set Magnetic Moments` | 已覆盖 |
| FM / AFM / PM | `Magnetic Order` | 已覆盖静态磁序和完全随机 PM |
| 小角随机扰动 | `Magmom Rotation` | 覆盖低温附近扰动 |
| 局部 canting | `Small-Angle Spin Tilt` | 覆盖单自旋、pair、group pair |
| 长周期纹理 | `Spin Spiral`, `Folded Helix` | 已覆盖螺旋/折返 |

### 缺口判断

| 优先级 | 建议 | 解决什么 | 决策 |
| --- | --- | --- | --- |
| P1 | `Spin Disorder` | 从 FM/AFM 到 PM 的无序度梯度，如 10%、30%、50%、70% 翻转或随机化 | 新卡 |
| P1 | `Small-Angle Spin Tilt` 增加 `Global tilt` | 外场下集体偏转角扫描，近似 spin-flop / metamagnetic 路径 | 扩展旧卡 |
| P2 | `Spin Disorder` 支持 cone disorder | 非共线有限温度扰动：围绕参考轴在 cone 内随机 | 新卡能力 |
| P3 | `Correlated Random Spin` | 有空间相关长度的随机非共线态 | 新卡，后置 |
| 暂缓 | `Spin Glass / Frustrated` 专卡 | 物理定义依赖晶格、交换模型和约束条件 | 暂缓 |

`flip_fraction` 不应塞进 `Magmom Rotation`。旋转是连续方向扰动，翻转是离散无序，混在一起会让用户误判生成数据的物理含义。

## Alloy / Composition

| 优先级 | 建议 | 解决什么 | 决策 |
| --- | --- | --- | --- |
| P1 | `Composition Gradient` | 沿 x/y/z 做配比梯度，覆盖扩散偶、梯度合金、界面过渡层 | 新卡 |
| P2 | `Layer Composition Replace` 或扩展 `Conditional Replace` | 按 layer/group 设置不同替换规则 | 先评估旧卡扩展 |
| P3 | `Composition Constraint Filter` | 过滤不满足目标配比或元素比例的结构 | 新卡或并入过滤体系 |

## Filter / Dataset 后处理

| 优先级 | 建议 | 解决什么 | 决策 |
| --- | --- | --- | --- |
| P1 | `Geometry Filter` | 自动挡掉明显坏结构，直接改善“生成 -> 清洗 -> FPS -> DFT”主流程 | 新卡 |
| P2 | `Duplicate / Similarity Filter` | 用结构 fingerprint 或几何 hash 去重 | 新卡 |
| P2 | `Stratified Sampler` | 按 `Config_type`、元素组成、能量/力区间分层抽样 | 新卡 |
| P3 | `Train/Test Split` | 固定 seed 按结构类型拆分训练/测试集 | 新卡或数据管理功能 |

`Geometry Filter` 优先级最高，因为它处理的是进入 DFT 前的硬质量门槛，不是末端代表性采样。

## Surface / Interface

| 优先级 | 建议 | 解决什么 | 决策 |
| --- | --- | --- | --- |
| P2 | `Surface Adsorption Sites` | 当前插入缺陷偏随机，缺少 atop / bridge / hollow 这类位点意识 | 新卡 |
| P3 | `Interface Builder` | A/B 材料拼接、间距、真空、失配处理 | 新卡，后置 |
| P3 | `Grain Boundary Builder` | 晶界训练集 | 新卡，后置 |

## 建议路线

| 顺序 | 项目 | 类型 | 原因 |
| --- | --- | --- | --- |
| 1 | `Geometry Filter` | 新卡 | 覆盖面最大，能直接减少坏结构进入 FPS 和 DFT |
| 2 | `Spin Disorder` | 新卡 | 补齐 FM/AFM 到 PM 之间的无序度梯度 |
| 3 | `Small-Angle Spin Tilt: Global tilt` | 扩展旧卡 | 同属确定性 canting 语义，改动边界清楚 |
| 4 | `Composition Gradient` | 新卡 | 覆盖扩散偶、梯度合金和界面过渡层 |
| 5 | `Disordered Seed / Random Pack` | 新卡 | 补无序初始构型，但要先定最小距离约束语义 |
| 6 | `Vib Mode Perturb: thermal sampling` | 扩展旧卡 | 需要明确模态归一化、频率单位和温度采样公式 |
| 7 | `Surface Adsorption Sites` | 新卡 | 表面位点价值明确，但依赖表面识别 |
| 8 | `Interface Builder` | 新卡 | 价值高，但需要多输入结构架构 |
| 9 | `Correlated Random Spin` | 新卡 | 需要空间相关长度模型，后置 |

第一批只做 1-4。它们覆盖面大、和现有架构匹配，而且不会引入过多物理承诺。
