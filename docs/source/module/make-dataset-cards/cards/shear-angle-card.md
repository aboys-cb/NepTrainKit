<!-- card-schema: {"card_name": "Shear Angle Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_angle_card.py", "serialized_keys": ["organic", "alpha_range", "beta_range", "gamma_range"]} -->

# 剪切角应变（Shear Angle Strain）

`Group`: `Lattice`  
`Class`: `ShearAngleCard`  
`Source`: `src/NepTrainKit/ui/views/_card/shear_angle_card.py`

## 功能说明
在保持晶格长度下扰动 alpha/beta/gamma 角，采样角度剪切自由度。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\alpha'=\alpha+\Delta\alpha,\quad \beta'=\beta+\Delta\beta,\quad \gamma'=\gamma+\Delta\gamma$$
$$\mathbf{C}'=\mathrm{cellpar\_to\_cell}(a,b,c,\alpha',\beta',\gamma')$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 角度相关响应误差高，低对称体系泛化差。
- 目标任务 (Target objective): 独立覆盖角度畸变通道。
- 建议添加条件 (Add-it trigger): 研究角度剪切或低对称晶胞变化。
- 不建议添加条件 (Avoid trigger): 仅关心体积和轴向拉伸。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先小角度范围验证稳定性。
- 若输入含有机分子，必须开启 `organic`；仅纯无机体系可关闭。


## 参数说明（完整）
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

### `alpha_range` (Alpha)
- UI Label: `Alpha`
- 字段映射 (Field mapping): 序列化键 `alpha_range` <-> 界面标签 `Alpha`。
- 控件标签 (Caption): `Alpha`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], degrees `[min,max,step]`
- 默认值 (Default): `[-2.0, 2.0, 1.0]`
- 含义 (Meaning): Alpha 角扫描区间（单位 `°`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 表示相对原始晶格角的增量 `Δalpha`；例如 `[-2,2,1]` 表示在 `-2°` 到 `+2°` 内按 `1°` 扫描。
- 推荐范围 (Recommended range):
  - 保守：±1°
  - 平衡：±3°
  - 探索：±6°

### `beta_range` (Beta)
- UI Label: `Beta`
- 字段映射 (Field mapping): 序列化键 `beta_range` <-> 界面标签 `Beta`。
- 控件标签 (Caption): `Beta`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], degrees `[min,max,step]`
- 默认值 (Default): `[-2.0, 2.0, 1.0]`
- 含义 (Meaning): Beta 角扫描区间（单位 `°`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 表示相对原始晶格角的增量 `Δbeta`；例如 `[-2,2,1]` 表示在 `-2°` 到 `+2°` 内按 `1°` 扫描。
- 推荐范围 (Recommended range):
  - 保守：±1°
  - 平衡：±3°
  - 探索：±6°

### `gamma_range` (Gamma)
- UI Label: `Gamma`
- 字段映射 (Field mapping): 序列化键 `gamma_range` <-> 界面标签 `Gamma`。
- 控件标签 (Caption): `Gamma`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], degrees `[min,max,step]`
- 默认值 (Default): `[-2.0, 2.0, 1.0]`
- 含义 (Meaning): Gamma 角扫描区间（单位 `°`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 表示相对原始晶格角的增量 `Δgamma`；例如 `[-2,2,1]` 表示在 `-2°` 到 `+2°` 内按 `1°` 扫描。
- 推荐范围 (Recommended range):
  - 保守：±1°
  - 平衡：±3°
  - 探索：±6°


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "organic": false,
  "alpha_range": [
    -1,
    1,
    1
  ],
  "beta_range": [
    -1,
    1,
    1
  ],
  "gamma_range": [
    -1,
    1,
    1
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "organic": false,
  "alpha_range": [
    -3,
    3,
    1
  ],
  "beta_range": [
    -3,
    3,
    1
  ],
  "gamma_range": [
    -3,
    3,
    1
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "organic": false,
  "alpha_range": [
    -6,
    6,
    2
  ],
  "beta_range": [
    -6,
    6,
    2
  ],
  "gamma_range": [
    -6,
    6,
    2
  ]
}
```


## 推荐组合
- Shear Angle Strain -> Lattice Perturb: 先扫描角度，再轻量扰动晶格长度。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 晶胞近奇异：降低角度极值。
- 样本过多：减少三角同时扫描范围。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Ang({...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
