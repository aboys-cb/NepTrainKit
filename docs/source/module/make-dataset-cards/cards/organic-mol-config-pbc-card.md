<!-- card-schema: {"card_name": "Organic Mol Config", "source_file": "src/NepTrainKit/ui/views/_card/organic_mol_config_pbc_card.py", "serialized_keys": ["perturb_per_frame", "torsion_range_deg", "max_torsions_per_conf", "gaussian_sigma", "pbc_mode", "local_cutoff", "local_subtree", "bond_detect_factor", "bond_keep_min_factor", "bond_keep_max_factor", "bond_keep_max_enable", "nonbond_min_factor", "max_retries", "mult_bond_factor", "nonpbc_box_size", "bo_c_const", "bo_threshold", "use_seed", "seed"]} -->

# 有机构象采样（Organic Mol Config）

`Group`: `Organic`  
`Class`: `OrganicMolConfigPBCCard`  
`Source`: `src/NepTrainKit/ui/views/_card/organic_mol_config_pbc_card.py`

## 功能说明
对有机体系进行扭转+局域扰动采样，并用键/非键约束控制构象可用性。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 分子构象覆盖不足，模型对构象变化敏感。
- 目标任务 (Target objective): 在保持化学拓扑合理的前提下扩展构象空间。
- 建议添加条件 (Add-it trigger): 有机晶体、分子体系、多构象任务。
- 不建议添加条件 (Avoid trigger): 纯无机体系。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 确认拓扑可识别，约束参数先用保守值。
- 先小批量验证有效构象率。


## 参数说明（完整）
### `perturb_per_frame` (Perturb Per Frame)
- UI Label: `Perturb Per Frame`
- 字段映射 (Field mapping): 序列化键 `perturb_per_frame` <-> 界面标签 `Perturb Per Frame`。
- 控件标签 (Caption): `Perturb Per Frame`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[100]`
- 含义 (Meaning): 每帧扰动数 (perturbations per frame)。
- 对输出规模/物理性的影响: 主要影响样本规模和运行时间。
- 推荐范围 (Recommended range):
  - 保守：20-50
  - 平衡：50-150
  - 探索：200+ 配过滤

### `torsion_range_deg` (Torsion Range Deg)
- UI Label: `Torsion Range Deg`
- 字段映射 (Field mapping): 序列化键 `torsion_range_deg` <-> 界面标签 `Torsion Range Deg`。
- 控件标签 (Caption): `Torsion Range Deg`。
- 控件解释 (Widget): 按字段类型解析。
- 类型/范围 (Type/Range): list[2]
- 默认值 (Default): `[-180.0, 180.0]`
- 含义 (Meaning): 扭转角范围 (torsion range)。
- 对输出规模/物理性的影响: 主控构象变化幅度。
- 推荐范围 (Recommended range):
  - 保守：±20~30°
  - 平衡：±45~60°
  - 探索：±90°+ 仅探索

### `max_torsions_per_conf` (Max Torsions Per Conf)
- UI Label: `Max Torsions Per Conf`
- 字段映射 (Field mapping): 序列化键 `max_torsions_per_conf` <-> 界面标签 `Max Torsions Per Conf`。
- 控件标签 (Caption): `Max Torsions Per Conf`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[50]`
- 含义 (Meaning): 每构象扭转数上限 (max torsions per config)。
- 对输出规模/物理性的影响: 越大构象变化越复杂。
- 推荐范围 (Recommended range):
  - 保守：1-3
  - 平衡：4-6
  - 探索：7+

### `gaussian_sigma` (Gaussian Sigma)
- UI Label: `Gaussian Sigma`
- 字段映射 (Field mapping): 序列化键 `gaussian_sigma` <-> 界面标签 `Gaussian Sigma`。
- 控件标签 (Caption): `Gaussian Sigma`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.03]`
- 含义 (Meaning): 高斯扰动强度 (gaussian sigma)。
- 对输出规模/物理性的影响: 主控局域随机位移幅度。
- 推荐范围 (Recommended range):
  - 保守：0.01-0.02
  - 平衡：0.03-0.05
  - 探索：0.08+ 需后筛

### `pbc_mode` (PBC Mode)
- UI Label: `PBC Mode`
- 字段映射 (Field mapping): 序列化键 `pbc_mode` <-> 界面标签 `PBC Mode`。
- 控件标签 (Caption): `PBC Mode`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"auto"`
- 含义 (Meaning): 周期处理模式 (PBC mode)。
- 对输出规模/物理性的影响: 决定周期/非周期下的约束处理路径。
- 配置建议 (Practical note): `PBC Mode` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `local_cutoff` (Local Cutoff)
- UI Label: `Local Cutoff`
- 字段映射 (Field mapping): 序列化键 `local_cutoff` <-> 界面标签 `Local Cutoff`。
- 控件标签 (Caption): `Local Cutoff`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[200]`
- 含义 (Meaning): 局域截断半径 (local cutoff)。
- 对输出规模/物理性的影响: 影响邻域构造范围与性能。
- 推荐范围 (Recommended range):
  - 保守：100-200
  - 平衡：200-400
  - 探索：400-1000

### `local_subtree` (Local Subtree)
- UI Label: `Local Subtree`
- 字段映射 (Field mapping): 序列化键 `local_subtree` <-> 界面标签 `Local Subtree`。
- 控件标签 (Caption): `Local Subtree`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[100]`
- 含义 (Meaning): 局域子图规模 (local subtree size)。
- 对输出规模/物理性的影响: 控制拓扑搜索深度。
- 推荐范围 (Recommended range):
  - 保守：50-100
  - 平衡：100-200
  - 探索：200-500

### `bond_detect_factor` (Bond Detect Factor)
- UI Label: `Bond Detect Factor`
- 字段映射 (Field mapping): 序列化键 `bond_detect_factor` <-> 界面标签 `Bond Detect Factor`。
- 控件标签 (Caption): `Bond Detect Factor`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[1.15]`
- 含义 (Meaning): 成键检测因子 (bond detect factor)。
- 对输出规模/物理性的影响: 越大越容易判定成键。
- 推荐范围 (Recommended range):
  - 保守：0.805-1.15
  - 平衡：1.15-1.72
  - 探索：1.72-2.88

### `bond_keep_min_factor` (Bond Keep Min Factor)
- UI Label: `Bond Keep Min Factor`
- 字段映射 (Field mapping): 序列化键 `bond_keep_min_factor` <-> 界面标签 `Bond Keep Min Factor`。
- 控件标签 (Caption): `Bond Keep Min Factor`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.6]`
- 含义 (Meaning): 最小保键因子 (bond keep min factor)。
- 对输出规模/物理性的影响: 限制最短可接受键长比例。
- 推荐范围 (Recommended range):
  - 保守：0.42-0.6
  - 平衡：0.6-0.9
  - 探索：0.9-1.5

### `bond_keep_max_factor` (Bond Keep Max Factor)
- UI Label: `Bond Keep Max Factor`
- 字段映射 (Field mapping): 序列化键 `bond_keep_max_factor` <-> 界面标签 `Bond Keep Max Factor`。
- 控件标签 (Caption): `Bond Keep Max Factor`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[1.15]`
- 含义 (Meaning): 最大保键因子 (bond keep max factor)。
- 对输出规模/物理性的影响: 限制最长可接受键长比例。
- 推荐范围 (Recommended range):
  - 保守：0.805-1.15
  - 平衡：1.15-1.72
  - 探索：1.72-2.88

### `bond_keep_max_enable` (Bond Keep Max Enable)
- UI Label: `Bond Keep Max Enable`
- 字段映射 (Field mapping): 序列化键 `bond_keep_max_enable` <-> 界面标签 `Bond Keep Max Enable`。
- 控件标签 (Caption): `Bond Keep Max Enable`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 启用最大保键约束 (enable max bond keep)。
- 对输出规模/物理性的影响: 决定是否执行键长上限约束。
- 配置建议 (Practical note):
  - 开启：需要启用 `Bond Keep Max Enable` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `nonbond_min_factor` (Nonbond Min Factor)
- UI Label: `Nonbond Min Factor`
- 字段映射 (Field mapping): 序列化键 `nonbond_min_factor` <-> 界面标签 `Nonbond Min Factor`。
- 控件标签 (Caption): `Nonbond Min Factor`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.8]`
- 含义 (Meaning): 非键最小距离因子 (nonbond min factor)。
- 对输出规模/物理性的影响: 过小会增加非键碰撞风险。
- 推荐范围 (Recommended range):
  - 保守：0.56-0.8
  - 平衡：0.8-1.2
  - 探索：1.2-2

### `max_retries` (Max Retries)
- UI Label: `Max Retries`
- 字段映射 (Field mapping): 序列化键 `max_retries` <-> 界面标签 `Max Retries`。
- 控件标签 (Caption): `Max Retries`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[12]`
- 含义 (Meaning): 最大重试次数 (max retries)。
- 对输出规模/物理性的影响: 提高有效样本率但增加耗时。
- 推荐范围 (Recommended range):
  - 保守：5-10
  - 平衡：10-30
  - 探索：30+

### `mult_bond_factor` (Mult Bond Factor)
- UI Label: `Mult Bond Factor`
- 字段映射 (Field mapping): 序列化键 `mult_bond_factor` <-> 界面标签 `Mult Bond Factor`。
- 控件标签 (Caption): `Mult Bond Factor`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.87]`
- 含义 (Meaning): 多键修正因子 (multiple-bond factor)。
- 对输出规模/物理性的影响: 调节多重键约束强度。
- 推荐范围 (Recommended range):
  - 保守：0.609-0.87
  - 平衡：0.87-1.3
  - 探索：1.3-2.17

### `nonpbc_box_size` (Nonpbc Box Size)
- UI Label: `Nonpbc Box Size`
- 字段映射 (Field mapping): 序列化键 `nonpbc_box_size` <-> 界面标签 `Nonpbc Box Size`。
- 控件标签 (Caption): `Nonpbc Box Size`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[100.0]`
- 含义 (Meaning): 非周期盒尺寸 (non-PBC box size)。
- 对输出规模/物理性的影响: 定义非周期模式的可用空间尺度。
- 推荐范围 (Recommended range):
  - 保守：50-100
  - 平衡：100-200
  - 探索：200-500

### `bo_c_const` (Bo C Const)
- UI Label: `Bo C Const`
- 字段映射 (Field mapping): 序列化键 `bo_c_const` <-> 界面标签 `Bo C Const`。
- 控件标签 (Caption): `Bo C Const`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.3]`
- 含义 (Meaning): 键级常数 C (bond-order constant C)。
- 对输出规模/物理性的影响: 影响键级衰减曲线形状。
- 推荐范围 (Recommended range):
  - 保守：0.21-0.3
  - 平衡：0.3-0.45
  - 探索：0.45-0.75

### `bo_threshold` (Bo Threshold)
- UI Label: `Bo Threshold`
- 字段映射 (Field mapping): 序列化键 `bo_threshold` <-> 界面标签 `Bo Threshold`。
- 控件标签 (Caption): `Bo Threshold`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.2]`
- 含义 (Meaning): 键级阈值 (bond-order threshold)。
- 对输出规模/物理性的影响: 控制成键/断键判定边界。
- 推荐范围 (Recommended range):
  - 保守：0.14-0.2
  - 平衡：0.2-0.3
  - 探索：0.3-0.5

### `use_seed` (Use Seed)
- UI Label: `Use Seed`
- 字段映射 (Field mapping): 序列化键 `use_seed` <-> 界面标签 `Use Seed`。
- 控件标签 (Caption): `Use Seed`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 是否启用固定随机种子 (deterministic seed switch)。
- 对输出规模/物理性的影响: 开启后可复现实验；关闭后每次采样分布会变化。
- 配置建议 (Practical note):
  - 开启：需要可复现对比时开启。
  - 关闭：探索阶段可关闭以增加随机覆盖。

### `seed` (Seed)
- UI Label: `Seed`
- 字段映射 (Field mapping): 序列化键 `seed` <-> 界面标签 `Seed`。
- 控件标签 (Caption): `Seed`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[0]`
- 含义 (Meaning): 随机种子值 (random seed value)。
- 对输出规模/物理性的影响: 只影响随机路径，不改变物理模型本身。
- 推荐范围 (Recommended range):
  - 保守：0（随机）
  - 平衡：1-99（可复现）
  - 探索：100-9999（多 seed 对比）


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "perturb_per_frame": [
    100
  ],
  "torsion_range_deg": [
    -30,
    30
  ],
  "max_torsions_per_conf": [
    5
  ],
  "gaussian_sigma": [
    0.01
  ],
  "pbc_mode": "auto",
  "local_cutoff": [
    150
  ],
  "local_subtree": [
    40
  ],
  "bond_detect_factor": [
    1.15
  ],
  "bond_keep_min_factor": [
    0.6
  ],
  "bond_keep_max_factor": [
    1.15
  ],
  "bond_keep_max_enable": false,
  "nonbond_min_factor": [
    0.8
  ],
  "max_retries": [
    12
  ],
  "mult_bond_factor": [
    0.87
  ],
  "nonpbc_box_size": [
    100.0
  ],
  "bo_c_const": [
    0.3
  ],
  "bo_threshold": [
    0.2
  ],
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "perturb_per_frame": [
    100
  ],
  "torsion_range_deg": [
    -60,
    60
  ],
  "max_torsions_per_conf": [
    5
  ],
  "gaussian_sigma": [
    0.03
  ],
  "pbc_mode": "auto",
  "local_cutoff": [
    150
  ],
  "local_subtree": [
    40
  ],
  "bond_detect_factor": [
    1.15
  ],
  "bond_keep_min_factor": [
    0.6
  ],
  "bond_keep_max_factor": [
    1.15
  ],
  "bond_keep_max_enable": false,
  "nonbond_min_factor": [
    0.8
  ],
  "max_retries": [
    12
  ],
  "mult_bond_factor": [
    0.87
  ],
  "nonpbc_box_size": [
    100.0
  ],
  "bo_c_const": [
    0.3
  ],
  "bo_threshold": [
    0.2
  ],
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "perturb_per_frame": [
    100
  ],
  "torsion_range_deg": [
    -120,
    120
  ],
  "max_torsions_per_conf": [
    8
  ],
  "gaussian_sigma": [
    0.08
  ],
  "pbc_mode": "auto",
  "local_cutoff": [
    150
  ],
  "local_subtree": [
    40
  ],
  "bond_detect_factor": [
    1.15
  ],
  "bond_keep_min_factor": [
    0.6
  ],
  "bond_keep_max_factor": [
    1.15
  ],
  "bond_keep_max_enable": false,
  "nonbond_min_factor": [
    0.8
  ],
  "max_retries": [
    12
  ],
  "mult_bond_factor": [
    0.87
  ],
  "nonpbc_box_size": [
    100.0
  ],
  "bo_c_const": [
    0.3
  ],
  "bo_threshold": [
    0.2
  ],
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Organic Mol Config -> Lattice Perturb: 将扭转多样性与轻量晶胞变化结合。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 有效样本率低：逐步放宽 `bond_keep` 和 `nonbond` 约束。
- 结构失真：降低 `torsion_range_deg` 与 `gaussian_sigma`。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `TG(n={...},sig={...},pbc={...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
