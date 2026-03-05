<!-- card-schema: {"card_name": "Random Slab", "source_file": "src/NepTrainKit/ui/views/_card/random_slab_card.py", "serialized_keys": ["h_range", "k_range", "l_range", "layer_range", "vacuum_range"]} -->

# 随机表面切片（Random Slab）

`Group`: `Surface`  
`Class`: `RandomSlabCard`  
`Source`: `src/NepTrainKit/ui/views/_card/random_slab_card.py`

## 功能说明
按 Miller 指数、层数和真空范围随机生成 slab，构建表面取向与厚度覆盖样本。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 表面相关任务误差显著高于体相。
- 目标任务 (Target objective): 扩展表面几何和边界条件分布。
- 建议添加条件 (Add-it trigger): 吸附、表面反应、界面任务。
- 不建议添加条件 (Avoid trigger): 只做体相性质训练。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先用窄 h/k/l 范围试跑。
- 保证真空层下限足够避免镜像相互作用。


## 参数说明（完整）
### `h_range` (H Range)
- UI Label: `H Range`
- 字段映射 (Field mapping): 序列化键 `h_range` <-> 界面标签 `H Range`。
- 控件标签 (Caption): `H Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0, 1, 1]`
- 含义 (Meaning): h 指数范围 (h range)。
- 对输出规模/物理性的影响: 控制表面取向扫描维度之一。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `k_range` (K Range)
- UI Label: `K Range`
- 字段映射 (Field mapping): 序列化键 `k_range` <-> 界面标签 `K Range`。
- 控件标签 (Caption): `K Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[0, 1, 1]`
- 含义 (Meaning): k 指数范围 (k range)。
- 对输出规模/物理性的影响: 控制表面取向扫描维度之一。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `l_range` (L Range)
- UI Label: `L Range`
- 字段映射 (Field mapping): 序列化键 `l_range` <-> 界面标签 `L Range`。
- 控件标签 (Caption): `L Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[1, 3, 1]`
- 含义 (Meaning): l 指数范围 (l range)。
- 对输出规模/物理性的影响: 控制表面取向扫描维度之一。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `layer_range` (Layer Range)
- UI Label: `Layer Range`
- 字段映射 (Field mapping): 序列化键 `layer_range` <-> 界面标签 `Layer Range`。
- 控件标签 (Caption): `Layer Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[3, 6, 1]`
- 含义 (Meaning): 层数范围 (layer range)。
- 对输出规模/物理性的影响: 主控 slab 厚度分布。
- 推荐范围 (Recommended range):
  - 保守：3 到 6，step 1
  - 平衡：3 到 6，step 0.5
  - 探索：3 到 6，step 2

### `vacuum_range` (Vacuum Range)
- UI Label: `Vacuum Range`
- 字段映射 (Field mapping): 序列化键 `vacuum_range` <-> 界面标签 `Vacuum Range`。
- 控件标签 (Caption): `Vacuum Range`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3]
- 默认值 (Default): `[10, 10, 1]`
- 含义 (Meaning): 真空层范围 (vacuum range)。
- 对输出规模/物理性的影响: 影响表面镜像相互作用强度。
- 推荐范围 (Recommended range):
  - 保守：10Å 左右
  - 平衡：12-20Å
  - 探索：20Å+


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [
    0,
    1,
    1
  ],
  "k_range": [
    0,
    1,
    1
  ],
  "l_range": [
    1,
    3,
    1
  ],
  "layer_range": [
    3,
    6,
    1
  ],
  "vacuum_range": [
    10,
    10,
    1
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [
    0,
    1,
    1
  ],
  "k_range": [
    0,
    1,
    1
  ],
  "l_range": [
    1,
    3,
    1
  ],
  "layer_range": [
    3,
    6,
    1
  ],
  "vacuum_range": [
    10,
    10,
    1
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "RandomSlabCard",
  "check_state": true,
  "h_range": [
    0,
    1,
    1
  ],
  "k_range": [
    0,
    1,
    1
  ],
  "l_range": [
    1,
    3,
    1
  ],
  "layer_range": [
    3,
    6,
    1
  ],
  "vacuum_range": [
    10,
    10,
    1
  ]
}
```


## 推荐组合
- Random Slab -> Insert Defect -> Random Vacancy: 生成更接近实际的缺陷表面样本。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- slab 过薄：提升 `layer_range` 下限。
- 表面伪相互作用：提高 `vacuum_range`。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Slab(hkl={...}{...}{...},L={...},vac={...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
