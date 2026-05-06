<!-- card-schema: {"card_name": "Lattice Strain", "source_file": "src/NepTrainKit/ui/views/_card/cell_strain_card.py", "serialized_keys": ["params", "organic", "engine_type", "x_range", "y_range", "z_range"]} -->

# 晶格应变（Lattice Strain）

`Group`: `Lattice`  
`Class`: `CellStrainCard`  
`Source`: `src/NepTrainKit/ui/views/_card/cell_strain_card.py`

## 功能说明
按轴向组合扫描应变（uniaxial/biaxial/triaxial/isotropic），系统构建应力-应变覆盖数据。

它最适合的场景是：为弹性响应或应力训练生成单轴/双轴应变样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\epsilon_i=\frac{s_i}{100},\quad \mathbf{C}'=\mathbf{D}\mathbf{C},\quad \mathbf{D}=\mathrm{diag}(1+\epsilon_x,1+\epsilon_y,1+\epsilon_z)$$
$$\text{isotropic: }\mathbf{C}'=(1+\epsilon)\mathbf{C}$$

## 操作示例
### 场景：为弹性响应或应力训练生成单轴/双轴应变样本

**输入：** 一个已弛豫晶体，且你知道主要想拉伸或压缩哪个方向

**目标：** 在不引入剪切的前提下，对 x/y/z 三个方向做受控应变扫描

**参数设置：**
- `engine_type` 先选清楚是单轴、双轴还是全方向策略
- `x_range` / `y_range` / `z_range` 先从百分之几级别的小应变开始
- `organic=true` 仅在分子晶体中需要额外保护分子内部结构时开启

**输出：** 一批带轴向应变差异的结构，晶格长度变化比角度变化更明显

**怎么验证结果合理：**
- 检查目标方向的晶格参数是否按预期变化
- 非目标方向不要出现意外大幅畸变
- 输出若为空，先检查各方向范围是否真的形成了有效扫描

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 弹性相关预测不稳，轴向响应泛化差。
- 目标任务 (Target objective): 构建可解释的应变网格数据。
- 建议添加条件 (Add-it trigger): 需要比较不同应变路径的模型表现。
- 不建议添加条件 (Avoid trigger): 仅需要局部坐标噪声而非系统应变。
> 物理提示 (Physics caution): 重点检查体积变化、晶胞条件数和最近邻距离，避免把几何畸变直接放大到非物理区间。

## 输入前提
- 先确定 `engine_type` 与研究问题匹配。
- 控制步长与范围，防止组合数失控。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由 `Axes`、`X`、`Y`、`Z`、`Organic` 控件组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"axes": "uniaxial", "x_range": [-5.0, 5.0, 1.0], "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0], "identify_organic": false}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组应变参数。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

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

### `engine_type` (Axes)
- UI Label: `Axes`
- 字段映射 (Field mapping): 序列化键 `engine_type` <-> 界面标签 `Axes`。
- 控件标签 (Caption): `Axes`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string): `uniaxial`, `biaxial`, `triaxial`, `isotropic`
- 默认值 (Default): `"uniaxial"`
- 含义 (Meaning): 应变轴模式 (strain axes mode)，可选 `uniaxial / biaxial / triaxial / isotropic`。
- 对输出规模/物理性的影响: 决定同时施加应变的轴向组合与样本规模：轴数越多，组合空间越大、计算成本越高。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 作为后续缺陷、表面或磁性卡片的母胞准备步骤。

## 常见问题与排查
- 输出为空或远少于预期时，先检查各范围参数是否真的形成了有效扫描组合；很多这类卡片在参数只给定单点时只会输出很少的结构。
- 如果结构明显不合理，先看体积、晶胞角和最近邻距离，再把主控幅度或步长回调到更小的量级。
- 模式冲突时以当前 UI 状态和代码分支为准；导入旧 JSON 后如果发现多个主模式字段同时存在，建议手工确认只保留一条主路径。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Str({...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
