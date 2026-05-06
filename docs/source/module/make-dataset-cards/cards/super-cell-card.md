<!-- card-schema: {"card_name": "Super Cell", "source_file": "src/NepTrainKit/ui/views/_card/super_cell_card.py", "serialized_keys": ["params"]} -->

# 超胞生成（Super Cell）

`Group`: `Lattice`  
`Class`: `SuperCellCard`  
`Source`: `src/NepTrainKit/ui/views/_card/super_cell_card.py`

## 功能说明
按倍率、目标胞长或原子数上限扩胞（supercell expansion），为缺陷/表面/磁操作提供空间。

它最适合的场景是：为后续做表面、空位或磁性操作，先把母胞扩到合适尺寸。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\mathbf{T}=\mathrm{diag}(n_a,n_b,n_c),\quad \mathbf{C}'=\mathbf{C}\mathbf{T}$$
$$N'=N\cdot n_a n_b n_c$$
$$n_a^{(\max)}=\max\left(\left\lfloor\frac{L_a^*}{\lVert\mathbf{a}\rVert}\right\rfloor,1\right),\quad n_a^{(\min)}=\left\lfloor\frac{L_a^*}{\lVert\mathbf{a}\rVert}\right\rfloor+1$$

## 操作示例
### 场景：为后续做表面、空位或磁性操作，先把母胞扩到合适尺寸

**输入：** 一个已弛豫的小胞结构，例如 Si、Fe 或氧化物原胞

**目标：** 在“固定倍率”“目标胞长”“最大原子数”三种策略中选一条主路径，把结构扩到下游可用规模

**参数设置：**
- 想固定复制倍数时用 `super_scale_*`
- 想把胞长扩到某个阈值时用 `super_cell_*`
- 想受预算约束时用 `max_atoms_*`，并用 `fixed_axis_*` 锁定某些方向

**输出：** 1 个或多组超胞结构；尺寸变化主要体现在晶格长度和原子总数上

**怎么验证结果合理：**
- 检查扩胞方向与下游任务一致，例如做 slab 时常锁住法向
- 确认原子数没有超出计算预算
- 导入 JSON 后若模式标志混乱，先保证只保留一条主路径

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 原胞太小，周期镜像效应干扰明显。
- 目标任务 (Target objective): 降低边界伪相互作用并支持复杂操作。
- 建议添加条件 (Add-it trigger): 下游需要 vacancy/interstitial/slab/magnetic 采样。
- 不建议添加条件 (Avoid trigger): 算力受限且小胞已满足任务需求。
> 物理提示 (Physics caution): 重点检查体积变化、晶胞条件数和最近邻距离，避免把几何畸变直接放大到非物理区间。

## 输入前提
- 先选定一种扩胞模式作为主路径。
- 设置原子数上限避免超预算。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由 supercell 行为、主模式、倍数/目标晶胞/最大原子数和固定轴控件组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"behavior_type": 0, "mode": "scale", "super_scale": [3, 3, 3], "target_cell": [20.0, 20.0, 20.0], "max_atoms": 100, "fixed_axis_flags": [false, false, false], "fixed_axis_scale": [1, 1, 1]}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组扩胞参数。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

### `super_cell_type` (Super Cell Type)
- UI Label: `Super Cell Type`
- 字段映射 (Field mapping): 序列化键 `super_cell_type` <-> 界面标签 `Super Cell Type`。
- 控件标签 (Caption): `Super Cell Type`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `0`
- 含义 (Meaning): 超胞模式类型 (supercell mode type)。
- 对输出规模/物理性的影响: 决定采用倍率、目标胞长或原子上限策略。
- 参数联动 / 生效条件: 三种扩胞思路本质上是“固定倍率 / 目标胞长 / 原子数预算”三选一；先明确主目标，再决定配套的 radio 按钮。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：单模式先跑通
  - 平衡：按任务切换
  - 探索：多模式并行需对照

### `super_scale_radio_button` (Super Scale Radio Button)
- UI Label: `Super Scale Radio Button`
- 字段映射 (Field mapping): 序列化键 `super_scale_radio_button` <-> 界面标签 `Super Scale Radio Button`。
- 控件标签 (Caption): `Super Scale Radio Button`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 倍率模式开关 (scale mode switch)。
- 对输出规模/物理性的影响: 控制是否按固定倍率扩胞。
- 参数联动 / 生效条件: 开启后主控参数是 `super_scale_condition`；这一路更适合你已经知道复制倍数时使用。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Super Scale Radio Button` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `super_scale_condition` (Super Scale Condition)
- UI Label: `Super Scale Condition`
- 字段映射 (Field mapping): 序列化键 `super_scale_condition` <-> 界面标签 `Super Scale Condition`。
- 控件标签 (Caption): `Super Scale Condition`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[3, 3, 3]`
- 含义 (Meaning): 倍率参数 (scale factors)。
- 对输出规模/物理性的影响: 定义各方向复制倍数。
- 参数联动 / 生效条件: 只有 `super_scale_radio_button=true` 时它才是主控参数；若同时锁定某个轴，该轴会被 `fixed_axis_scale` 覆盖。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：2x 左右
  - 平衡：2-4x
  - 探索：5x+ 高成本

### `super_cell_radio_button` (Super Cell Radio Button)
- UI Label: `Super Cell Radio Button`
- 字段映射 (Field mapping): 序列化键 `super_cell_radio_button` <-> 界面标签 `Super Cell Radio Button`。
- 控件标签 (Caption): `Super Cell Radio Button`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 目标胞长模式开关 (target-cell mode switch)。
- 对输出规模/物理性的影响: 控制是否按目标胞长扩胞。
- 参数联动 / 生效条件: 开启后主控参数切换为 `super_cell_condition`；适合按目标胞长而不是按固定倍数扩胞。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Super Cell Radio Button` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `super_cell_condition` (Super Cell Condition)
- UI Label: `Super Cell Condition`
- 字段映射 (Field mapping): 序列化键 `super_cell_condition` <-> 界面标签 `Super Cell Condition`。
- 控件标签 (Caption): `Super Cell Condition`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[20, 20, 20]`
- 含义 (Meaning): 目标胞长参数 (target cell condition)。
- 对输出规模/物理性的影响: 定义扩胞后的最小胞长目标。
- 参数联动 / 生效条件: 只有 `super_cell_radio_button=true` 时按目标胞长枚举；被锁定的轴不再按这个条件自由变化。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：20 到 20，step 20
  - 平衡：20 到 20，step 10
  - 探索：20 到 20，step 40

### `max_atoms_radio_button` (Max Atoms Radio Button)
- UI Label: `Max Atoms Radio Button`
- 字段映射 (Field mapping): 序列化键 `max_atoms_radio_button` <-> 界面标签 `Max Atoms Radio Button`。
- 控件标签 (Caption): `Max Atoms Radio Button`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 原子上限模式开关 (max-atoms mode switch)。
- 对输出规模/物理性的影响: 用于限制扩胞后结构规模。
- 参数联动 / 生效条件: 开启后由 `max_atoms_condition` 控制结构规模上限，适合算力预算明确的场景。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Max Atoms Radio Button` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `max_atoms_condition` (Max Atoms Condition)
- UI Label: `Max Atoms Condition`
- 字段映射 (Field mapping): 序列化键 `max_atoms_condition` <-> 界面标签 `Max Atoms Condition`。
- 控件标签 (Caption): `Max Atoms Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[100]`
- 含义 (Meaning): 每帧最大生成数 (max generated structures per frame)。
- 对输出规模/物理性的影响: 主要控制数据量和运行时间。
- 参数联动 / 生效条件: 只有 `max_atoms_radio_button=true` 时它才成为主控约束；输入太小时仍可能先得到少量更小的超胞。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：10-50
  - 平衡：50-200
  - 探索：200+ 需 FPS

### `fixed_axis_flags` (Fixed Axis Flags)
- UI Label: `Fixed Axis Flags`
- 字段映射 (Field mapping): 序列化键 `fixed_axis_flags` <-> 界面标签 `Fixed Axis Flags`。
- 控件标签 (Caption): `Fixed Axis Flags`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool list[3]
- 默认值 (Default): `[false, false, false]`
- 含义 (Meaning): 控制 a/b/c 三个方向是否锁定为固定扩包倍数。
- 对输出规模/物理性的影响: 适合 slab 等场景，将某一轴保持不扩包，同时让其余方向继续按当前模式变化。
- 参数联动 / 生效条件: 被锁定的轴不再跟随主模式自由枚举，而是直接读取 `fixed_axis_scale` 中对应的倍数。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 配置建议 (Practical note):
  - 开启：对应轴会强制使用 `fixed_axis_scale` 中的倍数。
  - 关闭：对应轴按当前 mode 的规则自由生成。

### `fixed_axis_scale` (Fixed Axis Scale)
- UI Label: `Fixed Axis Scale`
- 字段映射 (Field mapping): 序列化键 `fixed_axis_scale` <-> 界面标签 `Fixed Axis Scale`。
- 控件标签 (Caption): `Fixed Axis Scale`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[1, 1, 1]`
- 含义 (Meaning): 为已锁定的 a/b/c 轴提供固定扩包倍数。
- 对输出规模/物理性的影响: 一旦对应轴被锁定，就直接采用这里的倍数，不再受目标长度或最大原子数枚举影响。
- 参数联动 / 生效条件: 只对 `fixed_axis_flags=true` 的方向生效；未锁定的方向仍按当前主模式生成。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 推荐范围 (Recommended range):
  - 保守：1, 1, 1
  - 平衡：1-2 倍
  - 探索：2-4 倍，需同步关注原子数

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "super_cell_type": 0,
  "super_scale_radio_button": false,
  "super_scale_condition": [
    2,
    2,
    2
  ],
  "super_cell_radio_button": true,
  "super_cell_condition": [
    20,
    20,
    20
  ],
  "max_atoms_radio_button": false,
  "fixed_axis_flags": [
    false,
    false,
    false
  ],
  "fixed_axis_scale": [
    1,
    1,
    1
  ],
  "max_atoms_condition": [
    200
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "super_cell_type": 0,
  "super_scale_radio_button": false,
  "super_scale_condition": [
    2,
    2,
    2
  ],
  "super_cell_radio_button": true,
  "super_cell_condition": [
    20,
    20,
    20
  ],
  "max_atoms_radio_button": false,
  "fixed_axis_flags": [
    false,
    false,
    true
  ],
  "fixed_axis_scale": [
    1,
    1,
    1
  ],
  "max_atoms_condition": [
    200
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "super_cell_type": 0,
  "super_scale_radio_button": false,
  "super_scale_condition": [
    2,
    2,
    2
  ],
  "super_cell_radio_button": true,
  "super_cell_condition": [
    20,
    20,
    20
  ],
  "max_atoms_radio_button": false,
  "fixed_axis_flags": [
    false,
    false,
    true
  ],
  "fixed_axis_scale": [
    1,
    1,
    1
  ],
  "max_atoms_condition": [
    200
  ]
}
```

## 推荐组合
- Super Cell -> Vacancy Defect Generation: 保证删缺陷后仍有足够原子数。
- 作为后续缺陷、表面或磁性卡片的母胞准备步骤。
- 若扩胞后结构规模明显上升，建议在流程末端再接 `FPS Filter` 控制代表性样本数。

## 常见问题与排查
- 输出为空或远少于预期时，先检查各范围参数是否真的形成了有效扫描组合；很多这类卡片在参数只给定单点时只会输出很少的结构。
- 如果结构明显不合理，先看体积、晶胞角和最近邻距离，再把主控幅度或步长回调到更小的量级。
- 模式冲突时以当前 UI 状态和代码分支为准；导入旧 JSON 后如果发现多个主模式字段同时存在，建议手工确认只保留一条主路径。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `SC({...}x{...}x{...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
