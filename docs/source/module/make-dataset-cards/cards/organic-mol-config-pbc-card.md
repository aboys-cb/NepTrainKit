<!-- card-schema: {"card_name": "Organic Mol Config", "source_file": "src/NepTrainKit/ui/views/_card/organic_mol_config_pbc_card.py", "serialized_keys": ["params"]} -->

# 有机构象采样（Organic Mol Config）

`Group`: `Organic` | `Class`: `OrganicMolConfigPBCCard`

## 功能说明

对有机分子体系做单键子树旋转，并可叠加逐原子的高斯坐标噪声。程序先按共价半径、距离上限和 Pauling 键级阈值建立启发式拓扑；只有从拓扑中删除后能分开两侧子树的内部键才会参与旋转，因此环内键不会被当成普通可旋转单键。

每个候选还要经过原始键对的长度检查和非键原子的碰撞检查；失败时会减半旋转角与噪声后重试，全部重试仍失败的候选会被跳过，不会用未变化的输入结构冒充成功输出。

$$\Delta\theta\sim U(\theta_{\min},\theta_{\max}),\quad \Delta\mathbf{r}_i\sim\mathcal{N}(0,\sigma^2),\quad d_{ij}\text{ 的阈值均按 }r_i^{\mathrm{cov}}+r_j^{\mathrm{cov}}\text{ 缩放}$$

:::{warning}
这里的“键”和“可旋转键”来自几何启发式，不是 RDKit 等化学感知工具给出的正式键型。金属配位、离子对、非常规价态或已经严重畸变的输入可能识别错误；先看卡片首帧预览中的“检测键数 / 可旋转键数”，再抽查输出。
:::

## 操作示例

### 场景：模型在小分子构象变化上的力预测完全不准

你在一个有机分子晶体上训练了一个 NEP 模型，分子内的平衡键长和键角都学得不错。但你把同一个分子旋转一个二面角后，力的 MAE 跳了 4 倍。诊断发现训练集里每个分子只有一种构象——模型不知道扭转角改变时力场该怎么变。

**诊断思路：** 有机分子的势能面在二面角方向上是多极值的（如 gauche 和 anti 构象）。如果训练集只有一种构象，模型在这个柔性自由度上只能外推。需要系统地对可旋转二面角做采样，并保留那些键长不崩、非键不穿插的构象作为训练样本。

**输入：** 一个有机分子或全三维周期的分子晶体；程序按当前几何估计分子拓扑

**目标：** 每帧生成 100 个构象候选，扭转角在 ±60 度内采样，sigma 0.03 的高斯位移

**参数设置：**
- `Requested outputs per input` = `100`
- `Torsion angle increment` = `[-60, 60]`
- `Rotatable bonds per output` = `5`
- `Gaussian coordinate noise` = `0.03 Å`
- 展开“拓扑和几何保护设置”，勾选 `Enable upper bound`

**输出：** 最多 100 个通过保护条件的构象，带 `TG(req=100,ok={实际成功数},sig=0.03,pbc=auto)` 标签

**怎么验证训练集质量改善：**
- 重训后对手动旋转二面角后的测试构型推理，力 MAE 应显著缩小
- 抽查键长分布：原始检测键没有短于 `0.6 * (r_i^cov+r_j^cov)`；打开上限后也没有长于设定上限
- 抽查非键距离：没有低于 `0.8 * (r_i^cov+r_j^cov)` 的非键原子对
- 如果成功率低（< 10%），放宽 `bond_keep_max_factor` 或增大 `max_retries`

### 什么时候加这张卡、什么时候不加

**加：**
- 研究有机小分子、分子晶体或非周期分子的构象空间
- 模型对柔性自由度的预测差（扭转角变化引起力/能量跳变）
- 需要覆盖分子内多种构象（gauche, anti, eclipsed）

**不加：**
- 纯无机体系（没有可旋转二面角）
- 分子本身是刚性的（如苯环），且你也不需要高斯坐标噪声
- 需要化学价态、芳香性或反应键的严格识别；这张卡只有几何启发式拓扑
- 只需要坐标扰动 → 用 `Atomic Perturb`

## 参数说明

### 构象采样

#### Requested outputs per input（perturb_per_frame）
`int`，默认 100。每个输入结构尝试生成多少个候选，20–100 为常规。保护条件失败的候选会被跳过，所以实际输出数可能小于该值；若一个都没有通过，卡片直接报错。

#### Torsion angle increment（torsion_range_deg）
`tuple[float, float]`，默认 `(-180.0, 180.0)`。这是加到当前构象上的**旋转增量**范围，不是输出二面角的绝对目标值。±30° 较保守，±180° 覆盖更广；范围越大，几何保护失败概率通常越高。

#### Rotatable bonds per output（max_torsions_per_conf）
`int`，默认 5。每个输出最多选择多少条不同的可旋转键。实际数量不会超过首帧预览显示的可旋转键数；设为 0 时只保留高斯坐标噪声。

#### Gaussian coordinate noise（gaussian_sigma）
`float`，默认 0.03 Å。扭转之后，对**每个原子**独立添加笛卡尔高斯噪声；它不是只作用于局部子树。常规可从 0.01–0.03 Å 开始。

#### Max Retries（max_retries）
`int`，默认 12。每个候选在保护条件失败后的额外重试次数。每次重试都会把旋转增量和高斯噪声减半；增大该值会增加耗时，也可能让最终通过的结构越来越接近输入。

### PBC 和局部环境

#### PBC Mode（pbc_mode）
`str`，默认 `'auto'`。可选值：

- `auto`：只在输入三个 PBC 方向都打开时使用周期模式；普通无周期分子不会因为带一个显示 cell 就被误判为周期体系。
- `yes`：强制全三维周期，输入必须有有限且非奇异的 3×3 cell。
- `no`：按非周期分子处理，并在输出中写入显示盒。

二维或一维混合 PBC 暂不支持；`auto` 会明确报错，不会把部分周期静默改成全三维周期。

#### Local Cutoff（local_cutoff）
`int`，默认 150。这不是距离 cutoff，而是切换局部模式的**总原子数阈值**。当输入原子数大于该值时，每次旋转的子树会受 `local_subtree` 限制；小体系则旋转完整一侧子树。

#### Local Subtree（local_subtree）
`int`，默认 40。局部模式中，一条键一侧最多一起旋转的原子数。它只有在输入原子数超过 `local_cutoff` 时生效。

#### Nonpbc Box Size（nonpbc_box_size）
`float`，默认 100.0 Å。非周期输出被居中后写入的立方显示 cell 边长。输出 PBC 仍为 false，因此该盒子不会参与碰撞检查，也不会制造周期镜像。

### 键识别和键长约束

#### Bond Detect Factor（bond_detect_factor）
`float`，默认 1.15。候选键距离必须不超过 `factor * (r_i^cov+r_j^cov)`，同时还要通过 `bo_threshold`。数值越大越容易把相邻分子或配位接触误识别为键。

#### Bond Keep Min Factor（bond_keep_min_factor）
`float`，默认 0.6。原始检测键允许的最短距离为 `factor * (r_i^cov+r_j^cov)`。它防的是过度压缩，不是断键；设太大会拒绝合理的短键。

#### Bond Keep Max Factor（bond_keep_max_factor）
`float`，默认 1.15。勾选上限后，原始检测键允许的最长距离系数。

#### Bond Keep Max Enable（bond_keep_max_enable）
`bool`，默认 false。打开后才启用键长上限检查；关闭时没有隐藏的默认上限。研究非反应构象时建议打开；反应路径或软配位键可能需要关闭。

#### Mult Bond Factor（mult_bond_factor）
`float`，默认 0.87。短于 `factor * (r_i^cov+r_j^cov)` 的键不参与旋转；估计键级大于等于 2 的键也不参与。环内键由于不是拓扑桥边，同样不会旋转。

#### BO C Const（bo_c_const）
`float`，默认 0.3 Å。用于 `exp((r_i^cov+r_j^cov-d)/c)` 的衰减常数。它影响拓扑边和估计键级，除非预览中的检测键数明显不合理，否则保持默认。

#### BO Threshold（bo_threshold）
`float`，默认 0.2。估计 Pauling 键级必须高于该值才会形成拓扑边；阈值越低，越容易把弱接触当成键。

### 非键约束

#### Nonbond Min Factor（nonbond_min_factor）
`float`，默认 0.8。非键原子允许的最近距离为 `factor * (r_i^cov+r_j^cov)`；这里使用的是共价半径，不是范德华半径。发现穿插时可适当调大，但过大会大量拒绝紧密堆积的分子晶体候选。

### 随机性

#### Use Seed（use_seed）
`bool`，默认 false。打开 + 固定 seed 后同参数同输入可复现。种子与结构的稳定 ID 联合影响采样路径。

#### Seed（seed）
`int`，默认 0。固定随机种子值。仅 `use_seed` 打开时生效，两个不同 seed 产生两组不同的扭转角和扰动方向。

生效条件：`use_seed=True`。

## 推荐预设

### 保守（小角度扭转，±30°，50 候选）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "params": {
    "perturb_per_frame": 50,
    "torsion_range_deg": [-30, 30],
    "max_torsions_per_conf": 3,
    "gaussian_sigma": 0.01,
    "pbc_mode": "auto",
    "local_cutoff": 150,
    "local_subtree": 40,
    "bond_detect_factor": 1.15,
    "bond_keep_min_factor": 0.6,
    "bond_keep_max_factor": 1.15,
    "bond_keep_max_enable": true,
    "nonbond_min_factor": 0.8,
    "max_retries": 12,
    "mult_bond_factor": 0.87,
    "nonpbc_box_size": 100.0,
    "bo_c_const": 0.3,
    "bo_threshold": 0.2,
    "use_seed": true,
    "seed": 42
  }
}
```

### 平衡（中等扭转，±60°，100 候选）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "params": {
    "perturb_per_frame": 100,
    "torsion_range_deg": [-60, 60],
    "max_torsions_per_conf": 5,
    "gaussian_sigma": 0.03,
    "pbc_mode": "auto",
    "local_cutoff": 150,
    "local_subtree": 40,
    "bond_detect_factor": 1.15,
    "bond_keep_min_factor": 0.6,
    "bond_keep_max_factor": 1.15,
    "bond_keep_max_enable": true,
    "nonbond_min_factor": 0.8,
    "max_retries": 12,
    "mult_bond_factor": 0.87,
    "nonpbc_box_size": 100.0,
    "bo_c_const": 0.3,
    "bo_threshold": 0.2,
    "use_seed": true,
    "seed": 42
  }
}
```

### 探索（大角度扭转，±120°，100 候选）
```json
{
  "class": "OrganicMolConfigPBCCard",
  "check_state": true,
  "params": {
    "perturb_per_frame": 100,
    "torsion_range_deg": [-120, 120],
    "max_torsions_per_conf": 8,
    "gaussian_sigma": 0.05,
    "pbc_mode": "auto",
    "local_cutoff": 150,
    "local_subtree": 40,
    "bond_detect_factor": 1.15,
    "bond_keep_min_factor": 0.6,
    "bond_keep_max_factor": 1.15,
    "bond_keep_max_enable": true,
    "nonbond_min_factor": 0.8,
    "max_retries": 20,
    "mult_bond_factor": 0.87,
    "nonpbc_box_size": 100.0,
    "bo_c_const": 0.3,
    "bo_threshold": 0.2,
    "use_seed": true,
    "seed": 42
  }
}
```

## 推荐组合

- `Organic Mol Config` → `Atomic Perturb`：先做构象主采样，再加轻度热噪声
- `Organic Mol Config` → `FPS Filter`：大批量生成后做代表性筛选
- 对有机体系后筛时，优先看键长、非键距离和拓扑保持情况

## 常见问题

**输出数量明显小于请求值。** 其余候选没有通过保护条件。先检查预览中的检测键和可旋转键是否合理，再收窄 `torsion_range_deg`、减小高斯噪声，或调整真正导致失败的保护阈值。增大 `max_retries` 只会用更小扰动继续尝试。

**键被拉断了。** `bond_keep_max_enable` 没开，或者 `bond_keep_max_factor` 太大。开启该约束并设定合理上限。

**非键原子穿插。** `nonbond_min_factor` 太小。增大到 0.9~1.0。

**预览显示 0 条可旋转键。** 环内键、多重键、端基键以及一侧没有邻居的键不会参与子树旋转。若高斯噪声大于 0，卡片仍能生成坐标扰动；若只想做这种扰动，`Atomic Perturb` 更直接。

**普通无周期分子在 Auto 下报奇异 cell。** 当前版本会按输入的 PBC 标志判断，不会再把零 cell 当成周期晶胞；如果仍看到该错误，请确认是否手动选择了“强制全三维 PBC”。

## 输出标签

`TG(req={请求数},ok={实际成功数},sig={sigma},pbc={模式})`

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。构象生成受 seed + 结构稳定 ID 联合控制，不同 seed 产生不同的扭转角和扰动方向。
