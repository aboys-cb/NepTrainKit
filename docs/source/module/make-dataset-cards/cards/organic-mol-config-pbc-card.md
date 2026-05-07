<!-- card-schema: {"card_name": "Organic Mol Config", "source_file": "src/NepTrainKit/ui/views/_card/organic_mol_config_pbc_card.py", "serialized_keys": ["params"]} -->

# 有机构象采样（Organic Mol Config）

`Group`: `Organic` | `Class`: `OrganicMolConfigPBCCard`

## 功能说明

对有机分子体系进行扭转角采样 + 高斯局域扰动，生成构象候选用作训练数据。通过键长约束和非键最小距离约束保证输出的化学合理性——不会把构象采样做成断键采样。

$$P(\text{torsion})\sim U(\theta_{\min},\theta_{\max}),\quad \Delta\mathbf{r}_i\sim\mathcal{N}(0,\sigma^2),\quad \text{s.t. } d_{\text{bond}}\in[f_{\min}r_0,f_{\max}r_0],\ d_{\text{nonbond}}\ge f_{\text{nb}} r_{\text{vdw}}$$

## 操作示例

### 场景：模型在小分子构象变化上的力预测完全不准

你在一个有机分子晶体上训练了一个 NEP 模型，分子内的平衡键长和键角都学得不错。但你把同一个分子旋转一个二面角后，力的 MAE 跳了 4 倍。诊断发现训练集里每个分子只有一种构象——模型不知道扭转角改变时力场该怎么变。

**诊断思路：** 有机分子的势能面在二面角方向上是多极值的（如 gauche 和 anti 构象）。如果训练集只有一种构象，模型在这个柔性自由度上只能外推。需要系统地对可旋转二面角做采样，并保留那些键长不崩、非键不穿插的构象作为训练样本。

**输入：** 一个有机分子晶体结构，分子拓扑可由程序自动识别

**目标：** 每帧生成 100 个构象候选，扭转角在 ±60 度内采样，sigma 0.03 的高斯位移

**参数设置：**
- `Perturb Per Frame` = `[100]`
- `Torsion Range Deg` = `[-60, 60]`
- `Max Torsions Per Conf` = `[5]`
- `Gaussian Sigma` = `[0.03]`
- `Bond Keep Min Factor` = `[0.6]`，`Bond Keep Max Enable` = 勾选

**输出：** 最多 100 个构象，带 `TG(n=100,sig=0.03,pbc=auto)` 标签

**怎么验证训练集质量改善：**
- 重训后对手动旋转二面角后的测试构型推理，力 MAE 应显著缩小
- 抽查键长分布：输出分子中没有明显短于 `0.6 * r0` 或长于 `1.15 * r0` 的键
- 抽查非键距离：没有不同片段原子重叠（距离 > `0.8 * r_vdw`）
- 如果成功率低（< 10%），放宽 `bond_keep_max_factor` 或增大 `max_retries`

### 什么时候加这张卡、什么时候不加

**加：**
- 研究有机分子晶体、聚合物、分子体系的构象空间
- 模型对柔性自由度的预测差（扭转角变化引起力/能量跳变）
- 需要覆盖分子内多种构象（gauche, anti, eclipsed）

**不加：**
- 纯无机体系（没有可旋转二面角）
- 分子本身是刚性的（如苯环），扭转采样没有意义
- 只需要坐标扰动 → 用 `Atomic Perturb`

## 参数说明

### 构象采样

#### Perturb Per Frame（perturb_per_frame）
类型：`int`。默认：`100`。设置每个输入构型尝试生成的分子构象候选数。

每输入帧生成多少个构象候选。20~100 为常规范围。注意：实际有效输出 = 候选数 x 成功率，受约束条件影响。

#### Torsion Range Deg（torsion_range_deg）
类型：`tuple[float, float]`。默认：`(-180.0, 180.0)`。设置有机分子扭转角范围。

`[最小值, 最大值]`，二面角旋转范围，单位度。±30° 保守，±180° 全覆盖。

#### Max Torsions Per Conf（max_torsions_per_conf）
类型：`int`。默认：`50`。限制每个构象最多扰动的扭转键数量。

每个构象最多旋转多少个二面角。1~5 为常规。越多构象越复杂，成功率越低。

#### Gaussian Sigma（gaussian_sigma）
类型：`float`。默认：`0.03`。设置局部随机位移的高斯标准差。

局域高斯扰动强度，单位 A。0.01~0.03 为常规。

#### Max Retries（max_retries）
类型：`int`。默认：`12`。设置构象重试次数。

每个构象候选的最大重试次数。约束失败后重试。增大可提高成功率但线性增加耗时。

### PBC 和局部环境

#### PBC Mode（pbc_mode）
类型：`str`。默认：`'auto'`。选择构象扰动时如何处理周期边界。

`auto` / `pbc` / `nonpbc`。周期/非周期模式。

#### Local Cutoff（local_cutoff）
类型：`int`。默认：`200`。设置局部构象扰动的截断距离。

物理直觉：控制局部构象扰动影响范围。小 cutoff 更局域，大 cutoff 会带动更大分子片段，适合柔性链或配体。

#### Local Subtree（local_subtree）
类型：`int`。默认：`100`。决定是否按分子局部子树传播扭转。

物理直觉：打开后扭转会沿分子拓扑传播到子树，适合真实单键旋转；关闭时更像局部坐标噪声。

#### Nonpbc Box Size（nonpbc_box_size）
类型：`float`。默认：`100.0`。设置非周期分子的包围盒尺寸。

物理直觉：非周期分子需要足够大的盒子避免镜像相互作用。盒子太小会制造假接触，太大只增加体积相关过滤风险。

### 键识别和键长约束

#### Bond Detect Factor（bond_detect_factor）
类型：`float`。默认：`1.15`。设置键识别的距离系数。

成键检测因子，越大越容易判定成键。默认 1.15。

#### Bond Keep Min Factor（bond_keep_min_factor）
类型：`float`。默认：`0.6`。设置保留键长下限系数。

最小保键因子。允许的最短键长 = `factor * r0`。0.6 为默认。

#### Bond Keep Max Factor（bond_keep_max_factor）
类型：`float`。默认：`1.15`。设置保留键长上限系数。

物理直觉：限制扰动后键长不能过度拉伸。值太小会拒绝合理柔性构象，太大可能放过断键结构。

#### Bond Keep Max Enable（bond_keep_max_enable）
类型：`bool`。默认：`False`。决定是否启用键长上限检查。

物理直觉：需要严格保护分子内键时打开；如果体系本身含软键或反应路径，打开会误删高能但有意义的构型。

#### Mult Bond Factor（mult_bond_factor）
类型：`float`。默认：`0.87`。设置多重键识别系数。

物理直觉：用于识别多重键并限制其扭转自由度。含 C=C、C=O、芳香环时很重要；纯单键柔性分子影响较小。

#### BO C Const（bo_c_const）
类型：`float`。默认：`0.3`。设置键级估计公式中的衰减常数。

物理直觉：键级判断的衰减常数，影响哪些键被当成强键保护。除非在调试键识别，通常保持默认。

#### BO Threshold（bo_threshold）
类型：`float`。默认：`0.2`。设置判断化学键是否保留的键级阈值。

物理直觉：键级阈值越高，只有更强的键会被识别和保护；阈值越低，更多弱相互作用会被当成键，可能过度限制构象。

### 非键约束

#### Nonbond Min Factor（nonbond_min_factor）
类型：`float`。默认：`0.8`。设置非键原子最小距离系数。

非键最小距离因子。允许的非键最近距离 = `factor * r_vdw`。0.8 为默认。

### 随机性

#### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

勾选 + 固定 seed 可复现。种子与结构的稳定 ID 联合影响采样路径。

#### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

勾选 + 固定 seed 可复现。种子与结构的稳定 ID 联合影响采样路径。

生效条件：`use_seed=True`。

## 推荐预设

### 保守（小角度扭转，±30°，50 候选）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "perturb_per_frame": [50],
  "torsion_range_deg": [-30, 30],
  "max_torsions_per_conf": [3],
  "gaussian_sigma": [0.01],
  "pbc_mode": "auto",
  "local_cutoff": [150],
  "local_subtree": [40],
  "bond_detect_factor": [1.15],
  "bond_keep_min_factor": [0.6],
  "bond_keep_max_factor": [1.15],
  "bond_keep_max_enable": true,
  "nonbond_min_factor": [0.8],
  "max_retries": [12],
  "mult_bond_factor": [0.87],
  "nonpbc_box_size": [100.0],
  "bo_c_const": [0.3],
  "bo_threshold": [0.2],
  "use_seed": true,
  "seed": [42]
}
```

### 平衡（中等扭转，±60°，100 候选）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "perturb_per_frame": [100],
  "torsion_range_deg": [-60, 60],
  "max_torsions_per_conf": [5],
  "gaussian_sigma": [0.03],
  "pbc_mode": "auto",
  "local_cutoff": [150],
  "local_subtree": [40],
  "bond_detect_factor": [1.15],
  "bond_keep_min_factor": [0.6],
  "bond_keep_max_factor": [1.15],
  "bond_keep_max_enable": true,
  "nonbond_min_factor": [0.8],
  "max_retries": [12],
  "mult_bond_factor": [0.87],
  "nonpbc_box_size": [100.0],
  "bo_c_const": [0.3],
  "bo_threshold": [0.2],
  "use_seed": true,
  "seed": [42]
}
```

### 探索（大角度扭转，±120°，100 候选）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "perturb_per_frame": [100],
  "torsion_range_deg": [-120, 120],
  "max_torsions_per_conf": [8],
  "gaussian_sigma": [0.05],
  "pbc_mode": "auto",
  "local_cutoff": [150],
  "local_subtree": [40],
  "bond_detect_factor": [1.15],
  "bond_keep_min_factor": [0.6],
  "bond_keep_max_factor": [1.15],
  "bond_keep_max_enable": true,
  "nonbond_min_factor": [0.8],
  "max_retries": [20],
  "mult_bond_factor": [0.87],
  "nonpbc_box_size": [100.0],
  "bo_c_const": [0.3],
  "bo_threshold": [0.2],
  "use_seed": true,
  "seed": [42]
}
```

## 推荐组合

- `Organic Mol Config` → `Atomic Perturb`：先做构象主采样，再加轻度热噪声
- `Organic Mol Config` → `FPS Filter`：大批量生成后做代表性筛选
- 对有机体系后筛时，优先看键长、非键距离和拓扑保持情况

## 常见问题

**输出成功率很低（<10%）。** 约束太严或者扭转范围太大导致大部分候选被拒绝。先收窄 `torsion_range_deg`，增大 `max_retries`，放宽 `bond_keep_max_factor`。

**键被拉断了。** `bond_keep_max_enable` 没开，或者 `bond_keep_max_factor` 太大。开启该约束并设定合理上限。

**非键原子穿插。** `nonbond_min_factor` 太小。增大到 0.9~1.0。

**扭转角采样没有效果。** 确认分子中存在可旋转的单键。刚性环分子没有可旋转二面角。

## 输出标签

`TG(n={候选数},sig={sigma},pbc={模式})`

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。构象生成受 seed + 结构稳定 ID 联合控制，不同 seed 产生不同的扭转角和扰动方向。
