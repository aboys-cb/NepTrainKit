<!-- card-schema: {"card_name": "Lattice Perturb", "source_file": "src/NepTrainKit/ui/views/_card/cell_scaling_card.py", "serialized_keys": ["params"]} -->

# 晶格扰动（Lattice Perturb）

`Group`: `Lattice`  
`Class`: `CellScalingCard`  
`Source`: `src/NepTrainKit/ui/views/_card/cell_scaling_card.py`

## 功能说明
对晶胞尺度与坐标做轻量随机扰动（lattice perturbation），用于补充近平衡态的几何变化样本。

它最适合的场景是：为已弛豫晶体补充轻度体积缩放与少量晶格噪声。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$f_k\in[1-m,1+m],\quad a_i'=f_i a_i$$
$$\theta_j'=g_j\theta_j\quad(\text{when }perturb\_angle=true)$$
$$\mathbf{b}'=[b'\cos\gamma',\ b'\sin\gamma',\ 0]$$
$$c_x'=c'\cos\beta',\quad c_y'=\frac{c'(\cos\alpha'-\cos\beta'\cos\gamma')}{\sin\gamma'},\quad c_z'=\sqrt{c'^2-c_x'^2-c_y'^2}$$
$$\mathbf{c}'=[c_x',\ c_y',\ c_z']$$

## 操作示例
### 场景：为已弛豫晶体补充轻度体积缩放与少量晶格噪声

**输入：** 一个体相晶体或分子晶体，基态晶胞参数已经可信

**目标：** 生成围绕平衡体积附近的晶格尺度变化样本，用于体积响应或热膨胀附近训练

**参数设置：**
- `scaling_condition` 从 0.01-0.05 这一类小幅尺度变化开始
- `num_condition` 先用 10-30 检查分布，再放大
- `perturb_angle=false` 时先只看长度变化，避免角度扰动混入

**输出：** 一批带不同晶格尺度标签的结构，原子内部拓扑应基本保持稳定

**怎么验证结果合理：**
- 检查体积变化是否落在目标窗口内
- 最近邻距离不要被压缩到明显短键区间
- 若体积变化过大，先回调 `scaling_condition`

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 模型对小体积变化或轻微几何噪声敏感。
- 目标任务 (Target objective): 扩展近似热涨落区域的结构覆盖。
- 建议添加条件 (Add-it trigger): 需要比静态结构更密集的微扰样本。
- 不建议添加条件 (Avoid trigger): 结构已经明显不稳定或接近相变边界。
> 物理提示 (Physics caution): 重点检查体积变化、晶胞条件数和最近邻距离，避免把几何畸变直接放大到非物理区间。

## 输入前提
- 输入结构应先完成基本几何清洗（无明显重叠）。
- 若输入含有机分子，必须开启 `organic` 以启用团簇识别和刚性移动。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由随机引擎、最大缩放、输出数量、角度扰动、有机识别和 seed 控件组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"engine_type": 1, "max_scaling": 0.04, "max_num": 50, "perturb_angle": true, "identify_organic": false, "use_seed": false, "seed": 0}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组晶格扰动参数。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

### `engine_type` (Engine Type)
- UI Label: `Engine Type`
- 字段映射 (Field mapping): 序列化键 `engine_type` <-> 界面标签 `Engine Type`。
- 控件标签 (Caption): `Engine Type`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=Sobol`, `1=Uniform`
- 默认值 (Default): `1`
- 含义 (Meaning): 随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。
- 对输出规模/物理性的影响: Uniform 生成更快；Sobol 在样本较少时覆盖更均匀。样本规模足够大时，两者分布差异通常会缩小。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：小样本优先 Sobol
  - 平衡：先用 Uniform 快速试跑再抽样对比 Sobol
  - 探索：大样本阶段按速度与复现需求择优

### `perturb_angle` (Perturb Angle)
- UI Label: `Perturb Angle`
- 字段映射 (Field mapping): 序列化键 `perturb_angle` <-> 界面标签 `Perturb Angle`。
- 控件标签 (Caption): `Perturb Angle`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 角度扰动开关 (perturb angle switch)。
- 对输出规模/物理性的影响: 开启后除尺度变化外还会引入晶格角度变化，覆盖更广但失稳风险更高。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要启用 `Perturb Angle` 对应行为时开启。
  - 关闭：希望保持默认/更保守行为时关闭。

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
- 默认值 (Default): `[0.04]`
- 含义 (Meaning): 最大缩放幅度系数 `m` (max scaling ratio)。取值为 `0-1` 的比例值；例如 `0.04` 表示 `4%`。
- 对输出规模/物理性的影响: 长度/角度因子按 `1±m` 采样；例如 `m=0.04` 时，晶格长度与角度扰动幅度约为 `±4%`。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：0.01-0.03（约 1%-3%）
  - 平衡：0.03-0.06（约 3%-6%）
  - 探索：0.06-0.1（约 6%-10%，需严格质检）

### `num_condition` (Num Condition)
- UI Label: `Num Condition`
- 字段映射 (Field mapping): 序列化键 `num_condition` <-> 界面标签 `Num Condition`。
- 控件标签 (Caption): `Num Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[50]`
- 含义 (Meaning): 每个输入结构的晶格扰动采样数 (samples per input structure)。
- 对输出规模/物理性的影响: 该卡片主要改变晶格尺度/角度，原子坐标会随晶格同步缩放；单纯增大该值会快速放大数据量，但微观局域多样性提升有限。细粒度多样性建议主要由 `Atomic Perturb` 等原子级扰动补充。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：8-20（先验证分布与质检流程）
  - 平衡：20-50（常规训练覆盖）
  - 探索：50-100（仅在明确需要更多晶格态时，建议联用 Atomic Perturb）

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
  "class": "CellScalingCard",
  "check_state": true,
  "engine_type": 0,
  "perturb_angle": false,
  "organic": false,
  "scaling_condition": [
    0.01
  ],
  "num_condition": [
    20
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
  "class": "CellScalingCard",
  "check_state": true,
  "engine_type": 0,
  "perturb_angle": false,
  "organic": false,
  "scaling_condition": [
    0.03
  ],
  "num_condition": [
    20
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
  "class": "CellScalingCard",
  "check_state": true,
  "engine_type": 0,
  "perturb_angle": true,
  "organic": false,
  "scaling_condition": [
    0.08
  ],
  "num_condition": [
    20
  ],
  "use_seed": true,
  "seed": [
    0
  ]
}
```

## 推荐组合
- Lattice Perturb -> Atomic Perturb: 将晶格扰动与原子扰动联用。
- 作为后续缺陷、表面或磁性卡片的母胞准备步骤。
- 若扩胞后结构规模明显上升，建议在流程末端再接 `FPS Filter` 控制代表性样本数。

## 常见问题与排查
- 输出为空或远少于预期时，先检查各范围参数是否真的形成了有效扫描组合；很多这类卡片在参数只给定单点时只会输出很少的结构。
- 如果结构明显不合理，先看体积、晶胞角和最近邻距离，再把主控幅度或步长回调到更小的量级。
- 模式冲突时以当前 UI 状态和代码分支为准；导入旧 JSON 后如果发现多个主模式字段同时存在，建议手工确认只保留一条主路径。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `LSc(max={...},{...})`

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
