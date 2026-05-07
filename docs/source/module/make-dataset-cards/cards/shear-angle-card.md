<!-- card-schema: {"card_name": "Shear Angle Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_angle_card.py", "serialized_keys": ["params"]} -->

# 剪切角应变（Shear Angle Strain）

`Group`: `Lattice` | `Class`: `ShearAngleCard`

## 功能说明

在保持晶格长度不变的前提下扰动 alpha/beta/gamma 角，采样角度剪切自由度。每个角度通道独立控制 min/max/step（单位：度），扫描的是相对原始角度的增量。

$$\alpha'=\alpha+\Delta\alpha,\quad \beta'=\beta+\Delta\beta,\quad \gamma'=\gamma+\Delta\gamma$$

$$\mathbf{C}'=\mathrm{cellpar\_to\_cell}(a,b,c,\alpha',\beta',\gamma')$$

## 操作示例

### 场景：模型在单斜晶系上预测崩了，但正交晶系没问题

你在 HfO2 上训练了一个 NEP 模型。训练集里只有一个高温四方相（正交晶格），模型对四方相的能量和力预测很好。但一跑常温单斜相，能量误差跳了 3 倍——因为单斜相的 beta 角偏离 90 度约 10 度，模型没见过非正交的晶胞角度。

**诊断思路：** 不同晶系的本质区别是晶胞角度。模型只在正交晶格上训练，隐式地学会了"角度总是 90 度"的先验。只要 beta 偏离 90 度，晶格矢量在笛卡尔空间的投影关系就变了，原子间距分布也跟着变——模型全靠外推。解决办法是沿角度方向生成一组结构，让模型见过非 90 度的晶胞。

**输入：** 一个弛豫好的 HfO2 四方相结构（alpha=beta=gamma=90°）

**目标：** 沿 beta 方向生成 -6° 到 +6° 的角度扫描，步长 2°，覆盖单斜相可能出现的 beta 偏离

**参数设置：**
- `alpha_range` = `[0, 0, 1]` （不扫 alpha）
- `beta_range` = `[-6, 6, 2]`
- `gamma_range` = `[0, 0, 1]` （不扫 gamma）

**输出：** 7 个结构（-6°, -4°, -2°, 0°, +2°, +4°, +6°），晶格长度不变，只有 beta 角偏离 90 度

**怎么验证训练集质量改善：**
- 重训后用单斜相参考数据做推理，能量误差应显著下降
- 检查输出结构的体积：角度变化会轻微改变体积（晶格长度不变但矢量投影关系变了），确认体积变化 < 2%
- 如果模型对更多角度方向仍不准，同时放开 alpha 和 gamma 做多角度联合扫描
- 如果角度扫描范围 > ±10° 后结构明显非物理，可能是该体系不支持如此大的角度偏离

### 什么时候加这张卡、什么时候不加

**加：**
- 训练集仅覆盖高对称晶系（立方、四方），但目标体系存在低对称相（单斜、三斜）
- 模型在角度相关性质（如压电系数、角度依赖的弹性常数）上偏差大
- 需要区分"角度变化"和"长度变化"对模型的影响

**不加：**
- 体系始终维持高对称、角度不可能偏离 90 度 —— 加了产生无物理意义结构
- 需要同时控制长度和角度 → 用 `Lattice Perturb` 同时扰动两者更方便
- 想用矩阵分量而非角度控制剪切 → `Shear Matrix Strain` 更适合

## 参数说明


### Alpha Range（alpha_range）

类型：`tuple[float, float, float]`。默认：`(-2.0, 2.0, 1.0)`。设置 alpha 晶格角扫描范围。

物理直觉：角度扫描用于补晶胞角自由度；极端角度需要配合几何过滤。

### Beta Range（beta_range）

类型：`tuple[float, float, float]`。默认：`(-2.0, 2.0, 1.0)`。设置 beta 晶格角扫描范围。

物理直觉：角度扫描用于补晶胞角自由度；极端角度需要配合几何过滤。

### Gamma Range（gamma_range）

类型：`tuple[float, float, float]`。默认：`(-2.0, 2.0, 1.0)`。设置 gamma 晶格角扫描范围。

物理直觉：角度扫描用于补晶胞角自由度；极端角度需要配合几何过滤。

### Identify Organic（identify_organic）

类型：`bool`。默认：`False`。决定是否识别有机分子并保护内部几何。

物理直觉：有分子或有机片段时打开，防止晶格扰动破坏分子内部键长；纯无机晶体通常关闭。

## 推荐预设

### 单角度探索（仅 beta，±4°，步长 1°）
```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "alpha_range": [0, 0, 1],
  "beta_range": [-4, 4, 1],
  "gamma_range": [0, 0, 1],
  "identify_organic": false
}
```

### 双角度覆盖（beta+gamma，±3°，步长 1°，~49 个输出）
```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "alpha_range": [0, 0, 1],
  "beta_range": [-3, 3, 1],
  "gamma_range": [-3, 3, 1],
  "identify_organic": false
}
```

### 三角度全扫描（alpha+beta+gamma，±5°，步长 2°，~216 个输出）
```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "alpha_range": [-5, 5, 2],
  "beta_range": [-5, 5, 2],
  "gamma_range": [-5, 5, 2],
  "identify_organic": false
}
```

## 推荐组合

- `Lattice Strain` -> `Shear Angle Strain`：先做长度应变，再补角度剪切
- `Shear Angle Strain` -> `Atomic Perturb`：角度畸变后加原子坐标噪声
- `Shear Angle Strain` + `Shear Matrix Strain`：两者走不同形变路径，互补覆盖

## 常见问题

**输出为空。** 检查每个 range 的步长是否 > 0。不扫的通道不能设 `[0, 0, 0]`，必须设 `[0, 0, 1]` 让 range 至少产生一个 0 点。

**输出数量爆炸。** 三通道联扫 = Na * Nb * Ng。默认 `[-2, 2, 1]` 三通道 = 5^3 = 125 个结构。步长减半会 8 倍增长。建议从单通道开始逐次加。

**角度变化后体积也变了。** 晶格长度保持不变但角度变化会改变笛卡尔空间的矢量投影，体积会轻微变化。这是正常的几何效应，不是 bug。如果体积变化超过预期，可能是角度步长太大——减小 step。

**输出晶格接近奇异。** 如果角度偏离过大导致 cellpar_to_cell 产生的晶格矩阵接近退化，结构会不可用。典型症状是晶格行列式接近 0。把角度 range 回调到更保守的范围。

## 输出标签

`Ang(a={da}°,b={db}°,g={dg}°)` —— 只显示有非零变化的通道。

## 可复现性

无随机性。同参数同输入 → 严格一致输出。所有角度按固定网格扫描。
