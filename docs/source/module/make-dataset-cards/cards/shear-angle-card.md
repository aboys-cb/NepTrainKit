<!-- card-schema: {"card_name": "Shear Angle Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_angle_card.py", "serialized_keys": ["organic", "alpha_range", "beta_range", "gamma_range"]} -->

# 剪切角应变（Shear Angle Strain）

`Group`: `Lattice`  
`Class`: `ShearAngleCard`  
`Source`: `src/NepTrainKit/ui/views/_card/shear_angle_card.py`

## 功能说明
在保持晶格长度下扰动 alpha/beta/gamma 角，采样角度剪切自由度。

它最适合的场景是：通过直接修改晶胞角度生成剪切角应变样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\alpha'=\alpha+\Delta\alpha,\quad \beta'=\beta+\Delta\beta,\quad \gamma'=\gamma+\Delta\gamma$$
$$\mathbf{C}'=\mathrm{cellpar\_to\_cell}(a,b,c,\alpha',\beta',\gamma')$$

## 操作示例
### 场景：通过直接修改晶胞角度生成剪切角应变样本

**输入：** 一个晶体结构，且你关心的是角度剪切而不是长度拉伸

**目标：** 在控制 alpha/beta/gamma 的前提下生成角度畸变结构

**参数设置：**
- `alpha_range/beta_range/gamma_range` 先从小角度变化开始
- `organic=true` 仅在分子晶体中需要保护内部拓扑时开启
- 先只放开一个角度，避免三轴同时变化难以判断

**输出：** 晶胞角发生可控变化的结构，适合研究剪切响应

**怎么验证结果合理：**
- 确认主要变化确实发生在角度而不是长度
- 若条件数恶化明显，先减小角度步长
- 输出若很少，先检查三个角度范围是否真的形成组合

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 角度相关响应误差高，低对称体系泛化差。
- 目标任务 (Target objective): 独立覆盖角度畸变通道。
- 建议添加条件 (Add-it trigger): 研究角度剪切或低对称晶胞变化。
- 不建议添加条件 (Avoid trigger): 仅关心体积和轴向拉伸。
> 物理提示 (Physics caution): 重点检查体积变化、晶胞条件数和最近邻距离，避免把几何畸变直接放大到非物理区间。

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
- 怎么判断该开还是该关: 只有当你明确知道这个开关会改变当前工作流目标时才开启；不确定时先保持默认并用小样本验证。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 作为后续缺陷、表面或磁性卡片的母胞准备步骤。
- 若扩胞后结构规模明显上升，建议在流程末端再接 `FPS Filter` 控制代表性样本数。

## 常见问题与排查
- 输出为空或远少于预期时，先检查各范围参数是否真的形成了有效扫描组合；很多这类卡片在参数只给定单点时只会输出很少的结构。
- 如果结构明显不合理，先看体积、晶胞角和最近邻距离，再把主控幅度或步长回调到更小的量级。
- 模式冲突时以当前 UI 状态和代码分支为准；导入旧 JSON 后如果发现多个主模式字段同时存在，建议手工确认只保留一条主路径。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Ang({...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
