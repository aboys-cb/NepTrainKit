<!-- card-schema: {"card_name": "Super Cell", "source_file": "src/NepTrainKit/ui/views/_card/super_cell_card.py", "serialized_keys": ["super_cell_type", "super_scale_radio_button", "super_scale_condition", "super_cell_radio_button", "super_cell_condition", "max_atoms_radio_button", "max_atoms_condition"]} -->

# 超胞生成（Super Cell）

`Group`: `Lattice`  
`Class`: `SuperCellCard`  
`Source`: `src/NepTrainKit/ui/views/_card/super_cell_card.py`

## 功能说明
按倍率、目标胞长或原子数上限扩胞（supercell expansion），为缺陷/表面/磁操作提供空间。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\mathbf{T}=\mathrm{diag}(n_a,n_b,n_c),\quad \mathbf{C}'=\mathbf{C}\mathbf{T}$$
$$N'=N\cdot n_a n_b n_c$$
$$n_a^{(\max)}=\max\left(\left\lfloor\frac{L_a^*}{\lVert\mathbf{a}\rVert}\right\rfloor,1\right),\quad n_a^{(\min)}=\left\lfloor\frac{L_a^*}{\lVert\mathbf{a}\rVert}\right\rfloor+1$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 原胞太小，周期镜像效应干扰明显。
- 目标任务 (Target objective): 降低边界伪相互作用并支持复杂操作。
- 建议添加条件 (Add-it trigger): 下游需要 vacancy/interstitial/slab/magnetic 采样。
- 不建议添加条件 (Avoid trigger): 算力受限且小胞已满足任务需求。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先选定一种扩胞模式作为主路径。
- 设置原子数上限避免超预算。


## 参数说明（完整）
### `super_cell_type` (Super Cell Type)
- UI Label: `Super Cell Type`
- 字段映射 (Field mapping): 序列化键 `super_cell_type` <-> 界面标签 `Super Cell Type`。
- 控件标签 (Caption): `Super Cell Type`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int)
- 默认值 (Default): `0`
- 含义 (Meaning): 超胞模式类型 (supercell mode type)。
- 对输出规模/物理性的影响: 决定采用倍率、目标胞长或原子上限策略。
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
- 推荐范围 (Recommended range):
  - 保守：10-50
  - 平衡：50-200
  - 探索：200+ 需 FPS


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
  "max_atoms_condition": [
    200
  ]
}
```


## 推荐组合
- Super Cell -> Vacancy Defect Generation: 保证删缺陷后仍有足够原子数。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 结构规模过大：启用 max-atoms 限制。
- 目标尺寸不达标：检查模式开关与参数对应关系。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `SC({...}x{...}x{...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
