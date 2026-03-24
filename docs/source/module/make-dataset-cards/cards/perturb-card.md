<!-- card-schema: {"card_name": "Atomic Perturb", "source_file": "src/NepTrainKit/ui/views/_card/perturb_card.py", "serialized_keys": ["engine_type", "organic", "scaling_condition", "num_condition", "use_element_scaling", "element_scalings", "use_seed", "seed"]} -->

# 原子扰动（Atomic Perturb）

`Group`: `Perturbation`  
`Class`: `PerturbCard`  
`Source`: `src/NepTrainKit/ui/views/_card/perturb_card.py`

## 功能说明
对原子坐标施加随机扰动（atomic perturbation），补充近平衡态局部位移样本。

它最适合的场景是：为 Si 或金属小胞补充室温附近的轻微原子热扰动样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\Delta\mathbf{r}_i=\boldsymbol\xi_i\odot d_i,\quad \boldsymbol\xi_i\in[-1,1]^3,\quad \mathbf{r}_i'=\mathbf{r}_i+\Delta\mathbf{r}_i$$
$$\text{organic cluster 模式: }\forall j\in\mathcal{C},\ \mathbf{r}_j'=\mathbf{r}_j+\Delta\mathbf{r}_{\text{anchor}(\mathcal{C})}$$

## 操作示例
### 场景：为 Si 或金属小胞补充室温附近的轻微原子热扰动样本

**输入：** 一个已弛豫的体相或分子晶体结构

**目标：** 围绕平衡位置生成多份局部位移样本，补近平衡势能面覆盖

**参数设置：**
- `scaling_condition=[0.03]-[0.08] Å` 常作为第一轮试跑区间
- `num_condition` 先从 20-50 起步
- `organic=true` 只在输入确实包含有机团簇时开启

**输出：** 每个输入结构扩出多份坐标略有差异的扰动构型，主晶格不变

**怎么验证结果合理：**
- 抽查最短键长变化不要明显超出目标体系容忍区间
- 若重复度高，可换 `engine_type` 或增大样本数
- 若出现断键，优先回调 `scaling_condition`

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 力预测在小位移区间不稳定。
- 目标任务 (Target objective): 覆盖局域势能面邻域。
- 建议添加条件 (Add-it trigger): 缺少热噪声近似样本。
- 不建议添加条件 (Avoid trigger): 已有大量高质量 MD 热扰动数据。
> 物理提示 (Physics caution): 重点检查位移后最短键长和局部角度；幅度先小后大，比一次性追求大覆盖更稳妥。

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
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
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
- 怎么判断该开还是该关: 只有当你明确知道这个开关会改变当前工作流目标时才开启；不确定时先保持默认并用小样本验证。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 与晶格类卡片串联时，先做晶格变化，再补局部位移噪声。
- 大批量生成后可在流程末端接 `FPS Filter` 去掉重复样本。

## 常见问题与排查
- 输出为空时，优先检查输入是否满足这张卡的前提，例如是否已有振动模态、是否启用了正确的模式。
- 如果出现短键、断键或明显高能构型，先降低主控位移幅度，再缩小每帧样本数做抽样检查。
- 随机种子只控制采样路径，不会自动修正非物理参数；参数过激时程序仍会按当前配置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Pert(d={...},{...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
