<!-- card-schema: {"card_name": "Insert Defect", "source_file": "src/NepTrainKit/ui/views/_card/interstitial_adsorbate_card.py", "serialized_keys": ["mode", "species", "insert_count", "structure_count", "min_distance", "max_attempts", "use_seed", "seed", "axis", "offset"]} -->

# 插隙/吸附缺陷（Insert Defect）

`Group`: `Defect`  
`Class`: `InsertDefectCard`  
`Source`: `src/NepTrainKit/ui/views/_card/interstitial_adsorbate_card.py`

## 功能说明
在体相或表面插入额外原子/片段（interstitial or adsorbate insertion），采样缺陷与吸附构型。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 模型对插层/吸附位点预测误差高。
- 目标任务 (Target objective): 补充间隙位和吸附态样本。
- 建议添加条件 (Add-it trigger): 研究扩散、吸附、非本征缺陷。
- 不建议添加条件 (Avoid trigger): 仅关注完美晶体基态。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- `species` 必须是合法输入并与体系匹配。
- 根据密度先设保守 `min_distance`。


## 参数说明（完整）
### `mode` (Mode)
- UI Label: `Mode`
- 字段映射 (Field mapping): 序列化键 `mode` <-> 界面标签 `Mode`。
- 控件标签 (Caption): `Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(int): `0=Interstitial`, `1=Adsorption`（UI 下拉显示字符串）
- 默认值 (Default): `0`
- 含义 (Meaning): 插入模式枚举 (insertion mode enum)。UI 下拉显示字符串选项，但配置序列化保存为整数索引：`0=Interstitial`，`1=Adsorption`。
- 对输出规模/物理性的影响: `Interstitial` 在晶胞内部随机采样候选位点；`Adsorption` 在选定表面法向上方按 `offset` 放置，并启用 `axis/offset` 参数。
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
- 默认值 (Default): `0`
- 含义 (Meaning): 表面法向轴枚举 (surface normal axis enum)。仅在 `mode=Adsorption` 时生效：`0=a(x)`，`1=b(y)`，`2=c(z)`。
- 对输出规模/物理性的影响: 决定沿哪一条晶轴法向定义“表面上方”放置方向，会直接改变吸附位点分布。
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
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 插入成功率低：减小 `min_distance` 或增加 `max_attempts`。
- 速度过慢：先降 `structure_count` 再逐步放宽。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Ins({...},n={...})`
- Inserted atoms initialize new per-atom arrays at default/zero values where applicable.


## 可复现性说明
- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。
- 上游随机卡片或输入顺序变化仍会改变最终样本集合。
- 建议把 seed 与 pipeline 配置一起版本化记录。
