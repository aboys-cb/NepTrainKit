<!-- card-schema: {"card_name": "Atomic Perturb", "source_file": "src/NepTrainKit/ui/views/_card/perturb_card.py", "serialized_keys": ["engine_type", "organic", "scaling_condition", "num_condition", "use_element_scaling", "element_scalings", "use_seed", "seed"]} -->

# 原子扰动（Atomic Perturb）

`Group`: `Perturbation`  
`Class`: `PerturbCard`  
`Source`: `src/NepTrainKit/ui/views/_card/perturb_card.py`

## 功能说明
对原子坐标施加随机扰动（atomic perturbation），补充近平衡态局部位移样本。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\Delta\mathbf{r}_i=\boldsymbol\xi_i\odot d_i,\quad \boldsymbol\xi_i\in[-1,1]^3,\quad \mathbf{r}_i'=\mathbf{r}_i+\Delta\mathbf{r}_i$$
$$\text{organic cluster 模式: }\forall j\in\mathcal{C},\ \mathbf{r}_j'=\mathbf{r}_j+\Delta\mathbf{r}_{\text{anchor}(\mathcal{C})}$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 力预测在小位移区间不稳定。
- 目标任务 (Target objective): 覆盖局域势能面邻域。
- 建议添加条件 (Add-it trigger): 缺少热噪声近似样本。
- 不建议添加条件 (Avoid trigger): 已有大量高质量 MD 热扰动数据。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 结构应先弛豫到合理状态。
- 按元素扰动前确认 `element_scalings` 完整。


## 参数说明（完整）
### `engine_type` (Engine Type)
- UI Label: `Engine Type`
- 字段映射 (Field mapping): 序列化键 `engine_type` <-> 界面标签 `Engine Type`。
- 控件标签 (Caption): `Engine Type`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=Sobol`, `1=Uniform`
- 默认值 (Default): `1`
- 含义 (Meaning): 随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。
- 对输出规模/物理性的影响: Uniform 更快，适合高吞吐批量扰动；Sobol 在少量样本时能更均匀覆盖位移方向。样本数量增大后，两者差距通常减小。
- 推荐范围 (Recommended range):
  - 保守：少量样本与基线验证用 Sobol
  - 平衡：中等规模用 Uniform 预跑并抽查 Sobol
  - 探索：大规模以吞吐优先，可固定一种引擎保持一致性

### `organic` (Organic)
- UI Label: `Organic`
- 字段映射 (Field mapping): 序列化键 `organic` <-> 界面标签 `Organic`。
- 控件标签 (Caption): `Organic`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 有机团簇识别与刚性移动开关 (organic cluster rigid mode)。
- 对输出规模/物理性的影响: 开启后先识别有机团簇，扰动时对有机分子做刚性整体移动，减少分子内键长/拓扑被破坏；输入含有机分子时应开启。
- 配置建议 (Practical note):
  - 开启：输入包含有机分子时必须开启；会先识别团簇并按分子刚性整体移动。
  - 关闭：仅在确认为纯无机体系时关闭。

### `scaling_condition` (Scaling Condition)
- UI Label: `Scaling Condition`
- 字段映射 (Field mapping): 序列化键 `scaling_condition` <-> 界面标签 `Scaling Condition`。
- 控件标签 (Caption): `Scaling Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.3]`
- 含义 (Meaning): 最大位移距离 (max displacement distance)，单位 `Å`。
- 对输出规模/物理性的影响: 这是绝对长度而非百分比；每个原子的位移向量按 `[-1,1]` 随机方向乘以该距离上限（或元素专属上限）。
- 推荐范围 (Recommended range):
  - 保守：0.05-0.15 Å
  - 平衡：0.15-0.30 Å
  - 探索：0.30-0.50 Å（建议配后筛）

### `num_condition` (Num Condition)
- UI Label: `Num Condition`
- 字段映射 (Field mapping): 序列化键 `num_condition` <-> 界面标签 `Num Condition`。
- 控件标签 (Caption): `Num Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[50]`
- 含义 (Meaning): 采样数量控制 (sample count control)。
- 对输出规模/物理性的影响: 主要影响输出规模与耗时，不是幅度主控参数。
- 推荐范围 (Recommended range):
  - 保守：25-50
  - 平衡：50-100
  - 探索：100-250

### `use_element_scaling` (Use Element Scaling)
- UI Label: `Use Element Scaling`
- 字段映射 (Field mapping): 序列化键 `use_element_scaling` <-> 界面标签 `Use Element Scaling`。
- 控件标签 (Caption): `Use Element Scaling`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `false`
- 含义 (Meaning): 按元素扰动缩放 (use element scaling)。
- 对输出规模/物理性的影响: 允许不同元素使用不同扰动幅度。
- 配置建议 (Practical note):
  - 开启：需要启用 `Use Element Scaling` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

### `element_scalings` (Element Scalings)
- UI Label: `Element Scalings`
- 字段映射 (Field mapping): 序列化键 `element_scalings` <-> 界面标签 `Element Scalings`。
- 控件标签 (Caption): `Element Scalings`。
- 控件解释 (Widget): 按字段类型解析。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{}`
- 含义 (Meaning): 元素缩放字典 (element scaling dict)。
- 对输出规模/物理性的影响: 定义元素到扰动系数映射。
- 配置建议 (Practical note): 建议围绕任务目标小步调整，并先做单帧验证。

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
  "class": "PerturbCard",
  "check_state": true,
  "engine_type": 0,
  "organic": false,
  "scaling_condition": [
    0.01
  ],
  "num_condition": [
    20
  ],
  "use_element_scaling": false,
  "element_scalings": {},
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "PerturbCard",
  "check_state": true,
  "engine_type": 0,
  "organic": false,
  "scaling_condition": [
    0.03
  ],
  "num_condition": [
    20
  ],
  "use_element_scaling": false,
  "element_scalings": {},
  "use_seed": false,
  "seed": [
    0
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "PerturbCard",
  "check_state": true,
  "engine_type": 0,
  "organic": false,
  "scaling_condition": [
    0.08
  ],
  "num_condition": [
    20
  ],
  "use_element_scaling": true,
  "element_scalings": {},
  "use_seed": true,
  "seed": [
    0
  ]
}
```


## 推荐组合
- Lattice Strain -> Atomic Perturb: 结合全局形变与局部位移噪声。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 出现短键：降低 `scaling_condition` 或启用元素缩放。
- 重复样本多：下调 `num_condition` 并加过滤。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Pert(d={...},{...})`


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
