<!-- card-schema: {"card_name": "Insert Defect", "source_file": "src/NepTrainKit/ui/views/_card/interstitial_adsorbate_card.py", "serialized_keys": ["params", "mode", "species", "insert_count", "structure_count", "min_distance", "max_attempts", "use_seed", "seed", "axis", "offset"]} -->

# 插隙/吸附缺陷（Insert Defect）

`Group`: `Defect`  
`Class`: `InsertDefectCard`  
`Source`: `src/NepTrainKit/ui/views/_card/interstitial_adsorbate_card.py`

## 功能说明
在体相或表面插入额外原子/片段（interstitial or adsorbate insertion），采样缺陷与吸附构型。

它最适合的场景是：在 slab 或体相中插入吸附原子/插隙原子，补充表面或间隙位缺陷样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：在 slab 或体相中插入吸附原子/插隙原子，补充表面或间隙位缺陷样本

**输入：** 一个 slab 或足够大的体相超胞，以及想插入的物种，如 H、O 或 Li

**目标：** 生成一批不同插入位点和高度的候选结构，用于表面吸附或间隙缺陷训练

**参数设置：**
- `mode` 先分清是做吸附还是插隙
- `species` 写清候选插入物种
- `min_distance` 和 `offset` 先用保守值避免直接碰撞

**输出：** 每个输入结构会扩出若干带额外原子的候选构型

**怎么验证结果合理：**
- 检查插入原子没有与宿主原子明显重叠
- 表面吸附时确认插入方向与 `axis` 一致
- 若输出为空，先检查最小距离和尝试次数是否过严

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 模型对插层/吸附位点预测误差高。
- 目标任务 (Target objective): 补充间隙位和吸附态样本。
- 建议添加条件 (Add-it trigger): 研究扩散、吸附、非本征缺陷。
- 不建议添加条件 (Avoid trigger): 仅关注完美晶体基态。
> 物理提示 (Physics caution): 重点检查缺陷附近的局部配位和是否形成孤立原子或明显断裂。

## 输入前提
- `species` 必须是合法输入并与体系匹配。
- 根据密度先设保守 `min_distance`。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 核心操作参数 `InsertDefectParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"mode": 0, "species": "", "insert_count": 1, "structure_count": 10, "min_distance": 1.4, "max_attempts": 200, "use_seed": false, "seed": 0, "axis": 2, "offset": 1.5}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的插入数量、结构数量、最小距离、吸附轴和随机种子字段一致。
- 配置建议 (Practical note): 新版本优先读取 `params`，旧字段仍保留用于兼容已有 workflow。

### `mode` (Mode)
- UI Label: `Mode`
- 字段映射 (Field mapping): 序列化键 `mode` <-> 界面标签 `Mode`。
- 控件标签 (Caption): `Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=Interstitial`, `1=Adsorption`（UI 下拉显示字符串）
- 默认值 (Default): `0`
- 含义 (Meaning): 插入模式枚举 (insertion mode enum)。UI 下拉显示字符串选项，但配置序列化保存为整数索引：`0=Interstitial`，`1=Adsorption`。
- 对输出规模/物理性的影响: `Interstitial` 在晶胞内部随机采样候选位点；`Adsorption` 在选定表面法向上方按 `offset` 放置，并启用 `axis/offset` 参数。
- 参数联动 / 生效条件: 先分清你是在做表面吸附还是体相插隙；两种模式对 `axis` 和 `offset` 的依赖不同。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
- 推荐范围 (Recommended range):
  - 保守：先用 Interstitial 跑通最小距离与成功率
  - 平衡：按任务切换 Interstitial/Adsorption
  - 探索：Adsorption 批量生成前固定 axis 并抽样检查表面位点

### `species` (Species comma-separated)
- UI Label: `Species comma-separated`
- 字段映射 (Field mapping): 序列化键 `species` <-> 界面标签 `Species comma-separated`。
- 控件标签 (Caption): `Species (comma separated)`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `""`
- 含义 (Meaning): 插入物种 (inserted species)。
- 对输出规模/物理性的影响: 定义插入元素或片段类型。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): `Species comma-separated` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `insert_count` (Atoms per structure)
- UI Label: `Atoms per structure`
- 字段映射 (Field mapping): 序列化键 `insert_count` <-> 界面标签 `Atoms per structure`。
- 控件标签 (Caption): `Atoms per structure`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 每个生成结构插入的原子数 (atoms per generated structure)。
- 对输出规模/物理性的影响: 该值越大，缺陷密度越高且碰撞失败概率上升；需与 `min_distance`、`max_attempts` 联合调节。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 控件范围 (setRange): `1-20`。
- 推荐范围 (Recommended range):
  - 保守：1-2
  - 平衡：2-6
  - 探索：6-20

### `structure_count` (Structures to generate)
- UI Label: `Structures to generate`
- 字段映射 (Field mapping): 序列化键 `structure_count` <-> 界面标签 `Structures to generate`。
- 控件标签 (Caption): `Structures to generate`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[10]`
- 含义 (Meaning): 每个输入结构生成的输出数量 (structures to generate per input)。
- 对输出规模/物理性的影响: 直接决定该卡片输出规模与运行耗时。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 控件范围 (setRange): `1-1000`。
- 推荐范围 (Recommended range):
  - 保守：10-100
  - 平衡：100-400
  - 探索：400-1000

### `min_distance` (Min distance Å)
- UI Label: `Min distance Å`
- 字段映射 (Field mapping): 序列化键 `min_distance` <-> 界面标签 `Min distance Å`。
- 控件标签 (Caption): `Min distance (Å)`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[1.4]`
- 含义 (Meaning): 候选插入位点与已有原子的最小距离阈值 (minimum allowed distance, `Å`)。
- 对输出规模/物理性的影响: 阈值越大越保守、物理性更稳，但可行位点更少、成功率会下降。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 控件范围 (setRange): `0.0-10.0`。
- 推荐范围 (Recommended range):
  - 保守：1.6-2.5 Å
  - 平衡：1.2-1.6 Å
  - 探索：0.8-1.2 Å（仅探索）

### `max_attempts` (Max Attempts)
- UI Label: `Max Attempts`
- 字段映射 (Field mapping): 序列化键 `max_attempts` <-> 界面标签 `Max Attempts`。
- 控件标签 (Caption): `Max attempts`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[200]`
- 含义 (Meaning): 每个待插入原子的最大随机尝试次数 (maximum random attempts per atom)。
- 对输出规模/物理性的影响: 提高该值可提升成功率，但会线性增加采样耗时。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 控件范围 (setRange): `1-1000`。
- 推荐范围 (Recommended range):
  - 保守：50-200
  - 平衡：200-600
  - 探索：600-1000

### `use_seed` (Use seed)
- UI Label: `Use seed`
- 字段映射 (Field mapping): 序列化键 `use_seed` <-> 界面标签 `Use seed`。
- 控件标签 (Caption): `Use seed`。
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
- 对输出规模/物理性的影响: 只影响随机插入路径，不改变物理判据。
- 参数联动 / 生效条件: `seed` 只有在 `use_seed=true` 时才真正固定随机路径。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 控件范围 (setRange): `0-2147483647`。
- 推荐范围 (Recommended range):
  - 保守：0（随机）
  - 平衡：1-99（可复现）
  - 探索：100-9999（多 seed 对比）

### `axis` (Surface axis)
- UI Label: `Surface axis`
- 字段映射 (Field mapping): 序列化键 `axis` <-> 界面标签 `Surface axis`。
- 控件标签 (Caption): `Surface axis`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=a(x)`, `1=b(y)`, `2=c(z)`（仅 Adsorption 模式使用）
- 默认值 (Default): `2`
- 含义 (Meaning): 表面法向轴枚举 (surface normal axis enum)。仅在 `mode=Adsorption` 时生效：`0=a(x)`，`1=b(y)`，`2=c(z)`。
- 对输出规模/物理性的影响: 决定沿哪一条晶轴法向定义“表面上方”放置方向，会直接改变吸附位点分布。
- 参数联动 / 生效条件: 表面吸附时它通常定义“往哪一侧表面外法向放置”；体相插隙场景里方向性通常弱一些。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
- 推荐范围 (Recommended range):
  - 保守：先用 `c(z)` 与常见 slab 约定对齐
  - 平衡：按实际 slab 法向选择对应轴
  - 探索：多轴探索前先可视化核查法向是否正确

### `offset` (Offset distance Å)
- UI Label: `Offset distance Å`
- 字段映射 (Field mapping): 序列化键 `offset` <-> 界面标签 `Offset distance Å`。
- 控件标签 (Caption): `Offset distance (Å)`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[1.5]`
- 含义 (Meaning): 吸附放置高度偏移 (offset distance, `Å`)。
- 对输出规模/物理性的影响: 仅在 `Adsorption` 模式生效；偏移越大越远离表面，局域相互作用通常减弱。
- 参数联动 / 生效条件: 吸附模式下它更像“离表面多高”，插隙模式下则更像“沿参考方向偏移多少”。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 控件范围 (setRange): `0.0-10.0`。
- 推荐范围 (Recommended range):
  - 保守：1.0-2.0 Å
  - 平衡：2.0-4.0 Å
  - 探索：4.0-8.0 Å

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "InsertDefectCard",
  "check_state": true,
  "mode": 0,
  "species": "",
  "insert_count": [
    1
  ],
  "structure_count": [
    10
  ],
  "min_distance": [
    1.4
  ],
  "max_attempts": [
    10
  ],
  "use_seed": false,
  "seed": [
    0
  ],
  "axis": 2,
  "offset": [
    1.5
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "InsertDefectCard",
  "check_state": true,
  "mode": 0,
  "species": "",
  "insert_count": [
    1
  ],
  "structure_count": [
    10
  ],
  "min_distance": [
    1.4
  ],
  "max_attempts": [
    200
  ],
  "use_seed": false,
  "seed": [
    0
  ],
  "axis": 2,
  "offset": [
    1.5
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "InsertDefectCard",
  "check_state": true,
  "mode": 0,
  "species": "",
  "insert_count": [
    1
  ],
  "structure_count": [
    30
  ],
  "min_distance": [
    1.4
  ],
  "max_attempts": [
    600
  ],
  "use_seed": true,
  "seed": [
    0
  ],
  "axis": 2,
  "offset": [
    1.5
  ]
}
```

## 推荐组合
- Insert Defect -> Random Vacancy: 构建互补缺陷族样本。
- 缺陷强度上升前，通常先用 `Super Cell` 扩大母胞，避免小胞里缺陷相互作用过强。
- 缺陷生成后建议抽查最短键长、局部配位和是否出现明显断裂。

## 常见问题与排查
- 输出为空或结构数明显偏少时，先检查规则是否命中、浓度/数量是否过严，或输入超胞是否太小。
- 若输出结构不合理，优先检查最短键长、局部配位和是否出现整块骨架塌缩，再降低缺陷强度。
- 参数越界时通常受 UI 范围限制；但“过激而仍在范围内”的配置不会被自动裁剪，程序会继续按当前设置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Ins({...},n={...})`
- Inserted atoms initialize new per-atom arrays at default/zero values where applicable.

## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
