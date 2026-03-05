<!-- card-schema: {"card_name": "Shear Matrix Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_matrix_card.py", "serialized_keys": ["organic", "symmetric", "xy_range", "yz_range", "xz_range"]} -->

# 剪切矩阵应变（Shear Matrix Strain）

`Group`: `Lattice`  
`Class`: `ShearMatrixCard`  
`Source`: `src/NepTrainKit/ui/views/_card/shear_matrix_card.py`

## 功能说明
通过 xy/yz/xz 剪切矩阵生成非对角形变样本，覆盖剪切应变相关结构变化。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\gamma_{xy}=\frac{s_{xy}}{100},\ \gamma_{yz}=\frac{s_{yz}}{100},\ \gamma_{xz}=\frac{s_{xz}}{100}$$
$$\mathbf{S}=\begin{bmatrix}1&\gamma_{xy}&\gamma_{xz}\\0&1&\gamma_{yz}\\0&0&1\end{bmatrix},\quad \mathbf{C}'=\mathbf{C}\mathbf{S}$$
$$\text{symmetric=true 时再加 }S_{21}=\gamma_{xy},\ S_{32}=\gamma_{yz},\ S_{31}=\gamma_{xz}$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 对剪切应力相关性质预测不稳。
- 目标任务 (Target objective): 系统覆盖剪切分量及对称性差异。
- 建议添加条件 (Add-it trigger): 需要非对角应变采样。
- 不建议添加条件 (Avoid trigger): 仅做体积或单轴应变。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先确认 `symmetric` 策略。
- 单分量试跑后再三分量联扫。


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

### `symmetric` (Symmetric Shear)
- UI Label: `Symmetric Shear`
- 字段映射 (Field mapping): 序列化键 `symmetric` <-> 界面标签 `Symmetric Shear`。
- 控件标签 (Caption): `Symmetric Shear`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 对称剪切开关 (symmetric shear)。
- 对输出规模/物理性的影响: 开启后更接近对称形变路径，通常更稳定。
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
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 形变后结构失真：收窄剪切幅度并优先对称剪切。
- 组合数过大：增大步长。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Shr({...},sym={...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
