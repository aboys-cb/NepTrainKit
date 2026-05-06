<!-- card-schema: {"card_name": "Organic Mol Config", "source_file": "src/NepTrainKit/ui/views/_card/organic_mol_config_pbc_card.py", "serialized_keys": ["params"]} -->

# 有机构象采样（Organic Mol Config）

`Group`: `Organic`  
`Class`: `OrganicMolConfigPBCCard`  
`Source`: `src/NepTrainKit/ui/views/_card/organic_mol_config_pbc_card.py`

## 功能说明
对有机体系进行扭转+局域扰动采样，并用键/非键约束控制构象可用性。

它最适合的场景是：为有机分子或分子晶体生成构象样本，再交给下游做轻微扰动或筛选。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：为有机分子或分子晶体生成构象样本，再交给下游做轻微扰动或筛选

**输入：** 一个包含有机分子拓扑的结构

**目标：** 系统覆盖扭转角、局部高斯扰动和盒子模式差异，而不是只做刚性平移

**参数设置：**
- `torsion_range_deg` 先用 10-30 度量级试跑
- `perturb_per_frame` 控制每个输入扩出的构象数
- `bond_keep_*` 和 `nonbond_min_factor` 先保持保守，避免非物理穿插

**输出：** 每个输入结构会扩出多份构象候选，主要差异来自扭转与局部位移

**怎么验证结果合理：**
- 检查分子内部键长没有被大幅拉坏
- 确认不同构象不是简单重复平移
- 若失败率高，先放宽 `max_retries` 并收窄扭转范围

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 分子构象覆盖不足，模型对构象变化敏感。
- 目标任务 (Target objective): 在保持化学拓扑合理的前提下扩展构象空间。
- 建议添加条件 (Add-it trigger): 有机晶体、分子体系、多构象任务。
- 不建议添加条件 (Avoid trigger): 纯无机体系。
> 物理提示 (Physics caution): 重点检查分子内部键拓扑和非键碰撞，避免把构象采样做成断键采样。

## 输入前提
- 确认拓扑可识别，约束参数先用保守值。
- 先小批量验证有效构象率。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 核心操作参数 `OrganicMolConfigPBCParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"perturb_per_frame": 100, "torsion_range_deg": [-180.0, 180.0], "max_torsions_per_conf": 50, "gaussian_sigma": 0.03, "pbc_mode": "auto", "local_cutoff": 200, "local_subtree": 100, "bond_detect_factor": 1.15, "bond_keep_min_factor": 0.6, "bond_keep_max_factor": 1.15, "bond_keep_max_enable": false, "nonbond_min_factor": 0.8, "max_retries": 12, "mult_bond_factor": 0.87, "nonpbc_box_size": 100.0, "bo_c_const": 0.3, "bo_threshold": 0.2, "use_seed": false, "seed": 0}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的扭转采样、PBC、键长守卫、重试和随机种子字段一致。
- 配置建议 (Practical note): 新版本优先读取 `params`，旧字段仍保留用于兼容已有 workflow。

### `perturb_per_frame` (Perturb Per Frame)
- UI Label: `Perturb Per Frame`
- 字段映射 (Field mapping): 序列化键 `perturb_per_frame` <-> 界面标签 `Perturb Per Frame`。
- 控件标签 (Caption): `Perturb Per Frame`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[100]`
- 含义 (Meaning): 每帧扰动数 (perturbations per frame)。
- 对输出规模/物理性的影响: 主要影响样本规模和运行时间。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 怎么判断该开还是该关: 做可复现实验或要对比不同参数时开启；纯探索阶段可以先关闭以增加随机覆盖。
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
- 参数联动 / 生效条件: `seed` 只有在 `use_seed=true` 时才真正固定随机路径。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 构象卡片通常先做主构象变化，再用轻度 `Atomic Perturb` 补局部热噪声。
- 对有机体系做后筛时，优先先看键长、非键距离和拓扑保持情况。

## 常见问题与排查
- 输出为空或成功率很低时，常见原因是构象空间限制过严、重试次数不足，或者输入分子拓扑本身不完整。
- 若构象明显不合理，先检查键长、非键距离和扭转角窗口，再减小随机扰动幅度。
- 程序会尽量按当前保守阈值避免明显碰撞，但不会替代真正的后验结构检查。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `TG(n={...},sig={...},pbc={...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
