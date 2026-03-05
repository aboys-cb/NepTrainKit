<!-- card-schema: {"card_name": "Lattice Strain", "source_file": "src/NepTrainKit/ui/views/_card/cell_strain_card.py", "serialized_keys": ["organic", "engine_type", "x_range", "y_range", "z_range"]} -->

# 晶格应变（Lattice Strain）

`Group`: `Lattice`  
`Class`: `CellStrainCard`  
`Source`: `src/NepTrainKit/ui/views/_card/cell_strain_card.py`

## 功能说明
按轴向组合扫描应变（uniaxial/biaxial/triaxial/isotropic），系统构建应力-应变覆盖数据。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\epsilon_i=\frac{s_i}{100},\quad \mathbf{C}'=\mathbf{D}\mathbf{C},\quad \mathbf{D}=\mathrm{diag}(1+\epsilon_x,1+\epsilon_y,1+\epsilon_z)$$
$$\text{isotropic: }\mathbf{C}'=(1+\epsilon)\mathbf{C}$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 弹性相关预测不稳，轴向响应泛化差。
- 目标任务 (Target objective): 构建可解释的应变网格数据。
- 建议添加条件 (Add-it trigger): 需要比较不同应变路径的模型表现。
- 不建议添加条件 (Avoid trigger): 仅需要局部坐标噪声而非系统应变。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先确定 `engine_type` 与研究问题匹配。
- 控制步长与范围，防止组合数失控。


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

### `engine_type` (Axes)
- UI Label: `Axes`
- 字段映射 (Field mapping): 序列化键 `engine_type` <-> 界面标签 `Axes`。
- 控件标签 (Caption): `Axes`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string): `uniaxial`, `biaxial`, `triaxial`, `isotropic`
- 默认值 (Default): `"uniaxial"`
- 含义 (Meaning): 应变轴模式 (strain axes mode)，可选 `uniaxial / biaxial / triaxial / isotropic`。
- 对输出规模/物理性的影响: 决定同时施加应变的轴向组合与样本规模：轴数越多，组合空间越大、计算成本越高。
- 推荐范围 (Recommended range):
  - 保守：isotropic 或 uniaxial 基线
  - 平衡：biaxial 覆盖主要耦合响应
  - 探索：triaxial 仅在预算充足且有明确需求时启用

### `x_range` (X)
- UI Label: `X`
- 字段映射 (Field mapping): 序列化键 `x_range` <-> 界面标签 `X`。
- 控件标签 (Caption): `X`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], percent `[min,max,step]`
- 默认值 (Default): `[-5.0, 5.0, 1.0]`
- 含义 (Meaning): X 方向应变扫描区间（单位 `%`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 按百分比应变扫描 x 轴（例如 `-5` 到 `5` 表示 `-5%` 到 `+5%`）。范围越宽或步长越小，组合数增长越快。
- 推荐范围 (Recommended range):
  - 保守：±1-2%
  - 平衡：±3-5%
  - 探索：±6%+

### `y_range` (Y)
- UI Label: `Y`
- 字段映射 (Field mapping): 序列化键 `y_range` <-> 界面标签 `Y`。
- 控件标签 (Caption): `Y`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], percent `[min,max,step]`
- 默认值 (Default): `[-5.0, 5.0, 1.0]`
- 含义 (Meaning): Y 方向应变扫描区间（单位 `%`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 按百分比应变扫描 y 轴（例如 `-5` 到 `5` 表示 `-5%` 到 `+5%`）。范围越宽或步长越小，组合数增长越快。
- 推荐范围 (Recommended range):
  - 保守：±1-2%
  - 平衡：±3-5%
  - 探索：±6%+

### `z_range` (Z)
- UI Label: `Z`
- 字段映射 (Field mapping): 序列化键 `z_range` <-> 界面标签 `Z`。
- 控件标签 (Caption): `Z`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], percent `[min,max,step]`
- 默认值 (Default): `[-5.0, 5.0, 1.0]`
- 含义 (Meaning): Z 方向应变扫描区间（单位 `%`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 按百分比应变扫描 z 轴（例如 `-5` 到 `5` 表示 `-5%` 到 `+5%`）。范围越宽或步长越小，组合数增长越快。
- 推荐范围 (Recommended range):
  - 保守：±1-2%
  - 平衡：±3-5%
  - 探索：±6%+


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "CellStrainCard",
  "check_state": true,
  "organic": false,
  "engine_type": "isotropic",
  "x_range": [
    -1,
    1,
    1
  ],
  "y_range": [
    -1,
    1,
    1
  ],
  "z_range": [
    -1,
    1,
    1
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "CellStrainCard",
  "check_state": true,
  "organic": false,
  "engine_type": "biaxial",
  "x_range": [
    -3,
    3,
    1
  ],
  "y_range": [
    -3,
    3,
    1
  ],
  "z_range": [
    -3,
    3,
    1
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "CellStrainCard",
  "check_state": true,
  "organic": false,
  "engine_type": "triaxial",
  "x_range": [
    -6,
    6,
    2
  ],
  "y_range": [
    -6,
    6,
    2
  ],
  "z_range": [
    -6,
    6,
    2
  ]
}
```


## 推荐组合
- Lattice Strain -> Atomic Perturb: 对每个应变帧增加局部差异。
- Lattice Strain -> Shear Matrix Strain: 在轴向应变后补充非对角剪切分量覆盖。


## 常见问题与排查
- 输出数量过大：减少轴组合或增大步长。
- 高应变导致异常结构：收窄 `x/y/z_range` 极值。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Str({...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
