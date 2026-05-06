<!-- card-schema: {"card_name": "Shear Matrix Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_matrix_card.py", "serialized_keys": ["params"]} -->

# 剪切矩阵应变（Shear Matrix Strain）

`Group`: `Lattice`  
`Class`: `ShearMatrixCard`  
`Source`: `src/NepTrainKit/ui/views/_card/shear_matrix_card.py`

## 功能说明
通过 xy/yz/xz 剪切矩阵生成非对角形变样本，覆盖剪切应变相关结构变化。

它最适合的场景是：通过非对角剪切矩阵项生成受控剪切样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\gamma_{xy}=\frac{s_{xy}}{100},\ \gamma_{yz}=\frac{s_{yz}}{100},\ \gamma_{xz}=\frac{s_{xz}}{100}$$
$$\mathbf{S}=\begin{bmatrix}1&\gamma_{xy}&\gamma_{xz}\\0&1&\gamma_{yz}\\0&0&1\end{bmatrix},\quad \mathbf{C}'=\mathbf{C}\mathbf{S}$$
$$\text{symmetric=true 时再加 }S_{21}=\gamma_{xy},\ S_{32}=\gamma_{yz},\ S_{31}=\gamma_{xz}$$

## 操作示例
### 场景：通过非对角剪切矩阵项生成受控剪切样本

**输入：** 一个晶体结构，并已明确想改动 `xy`、`yz` 或 `xz` 哪个剪切分量

**目标：** 比角度法更直接地控制剪切矩阵元素，补充弹性或畸变数据

**参数设置：**
- `xy_range/yz_range/xz_range` 先从小百分比开始
- `symmetric=true` 时适合更规则的剪切扫描
- `organic=true` 只在分子晶体场景下考虑

**输出：** 一批剪切矩阵不同的结构，晶格基矢之间的夹角和投影关系会变化

**怎么验证结果合理：**
- 检查非对角项变化是否符合设定
- 若畸变后结构明显失稳，先减小剪切幅度
- 一次只放开一到两个分量更容易调参

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 对剪切应力相关性质预测不稳。
- 目标任务 (Target objective): 系统覆盖剪切分量及对称性差异。
- 建议添加条件 (Add-it trigger): 需要非对角应变采样。
- 不建议添加条件 (Avoid trigger): 仅做体积或单轴应变。
> 物理提示 (Physics caution): 重点检查体积变化、晶胞条件数和最近邻距离，避免把几何畸变直接放大到非物理区间。

## 输入前提
- 先确认 `symmetric` 策略。
- 单分量试跑后再三分量联扫。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由 XY/YZ/XZ 剪切范围、对称剪切和有机识别控件组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"xy_range": [-5.0, 5.0, 1.0], "yz_range": [-5.0, 5.0, 1.0], "xz_range": [-5.0, 5.0, 1.0], "symmetric": true, "identify_organic": false}`
- 含义 (Meaning): UI-independent 参数快照，供 core operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组剪切矩阵参数。
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

### `symmetric` (Symmetric Shear)
- UI Label: `Symmetric Shear`
- 字段映射 (Field mapping): 序列化键 `symmetric` <-> 界面标签 `Symmetric Shear`。
- 控件标签 (Caption): `Symmetric Shear`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 对称剪切开关 (symmetric shear)。
- 对输出规模/物理性的影响: 开启后更接近对称形变路径，通常更稳定。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note):
  - 开启：需要对称剪切路径时开启。
  - 关闭：仅测试非对称分量时关闭。

### `xy_range` (XY)
- UI Label: `XY`
- 字段映射 (Field mapping): 序列化键 `xy_range` <-> 界面标签 `XY`。
- 控件标签 (Caption): `XY`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], percent `[min,max,step]`
- 默认值 (Default): `[-5.0, 5.0, 1.0]`
- 含义 (Meaning): XY 剪切分量扫描区间（单位 `%`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 剪切矩阵分量按 `sxy/100` 写入，`sxy=5` 即 `0.05` 剪切分量。范围越宽或步长越小，生成组合越多。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：±1-2%
  - 平衡：±3-5%
  - 探索：±6%+

### `yz_range` (YZ)
- UI Label: `YZ`
- 字段映射 (Field mapping): 序列化键 `yz_range` <-> 界面标签 `YZ`。
- 控件标签 (Caption): `YZ`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], percent `[min,max,step]`
- 默认值 (Default): `[-5.0, 5.0, 1.0]`
- 含义 (Meaning): YZ 剪切分量扫描区间（单位 `%`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 剪切矩阵分量按 `syz/100` 写入，`syz=5` 即 `0.05` 剪切分量。范围越宽或步长越小，生成组合越多。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：±1-2%
  - 平衡：±3-5%
  - 探索：±6%+

### `xz_range` (XZ)
- UI Label: `XZ`
- 字段映射 (Field mapping): 序列化键 `xz_range` <-> 界面标签 `XZ`。
- 控件标签 (Caption): `XZ`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], percent `[min,max,step]`
- 默认值 (Default): `[-5.0, 5.0, 1.0]`
- 含义 (Meaning): XZ 剪切分量扫描区间（单位 `%`），格式为 `[min,max,step]`。
- 对输出规模/物理性的影响: 剪切矩阵分量按 `sxz/100` 写入，`sxz=5` 即 `0.05` 剪切分量。范围越宽或步长越小，生成组合越多。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
- 推荐范围 (Recommended range):
  - 保守：±1-2%
  - 平衡：±3-5%
  - 探索：±6%+

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "organic": false,
  "symmetric": true,
  "xy_range": [
    -1,
    1,
    1
  ],
  "yz_range": [
    -1,
    1,
    1
  ],
  "xz_range": [
    -1,
    1,
    1
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "organic": false,
  "symmetric": true,
  "xy_range": [
    -3,
    3,
    1
  ],
  "yz_range": [
    -3,
    3,
    1
  ],
  "xz_range": [
    -3,
    3,
    1
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "organic": false,
  "symmetric": false,
  "xy_range": [
    -6,
    6,
    2
  ],
  "yz_range": [
    -6,
    6,
    2
  ],
  "xz_range": [
    -6,
    6,
    2
  ]
}
```

## 推荐组合
- Shear Matrix Strain -> Atomic Perturb: 用局部位移进一步细化剪切结构。
- 作为后续缺陷、表面或磁性卡片的母胞准备步骤。
- 若扩胞后结构规模明显上升，建议在流程末端再接 `FPS Filter` 控制代表性样本数。

## 常见问题与排查
- 输出为空或远少于预期时，先检查各范围参数是否真的形成了有效扫描组合；很多这类卡片在参数只给定单点时只会输出很少的结构。
- 如果结构明显不合理，先看体积、晶胞角和最近邻距离，再把主控幅度或步长回调到更小的量级。
- 模式冲突时以当前 UI 状态和代码分支为准；导入旧 JSON 后如果发现多个主模式字段同时存在，建议手工确认只保留一条主路径。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Shr({...},sym={...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
