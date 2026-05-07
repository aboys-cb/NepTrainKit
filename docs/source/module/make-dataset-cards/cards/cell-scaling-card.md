<!-- card-schema: {"card_name": "Lattice Perturb", "source_file": "src/NepTrainKit/ui/views/_card/cell_scaling_card.py", "serialized_keys": ["params"]} -->

# 晶格扰动（Lattice Perturb）

`Group`: `Lattice` | `Class`: `CellScalingCard`

## 功能说明

对晶胞尺度和角度做轻量随机扰动，为近平衡态补充几何变化样本。每个输出结构在原始晶格基础上引入独立随机的长度缩放和（可选的）角度缩放。

$$f_k\in[1-m,1+m],\quad a_i'=f_i a_i,\quad \theta_j'=g_j\theta_j$$

## 操作示例

### 场景：模型预测热膨胀系数偏差超过 30%

你在 Si 上训练了一个 NEP 模型，弛豫态的能量和力精度都过关，但用准谐近似算出来的热膨胀系数比实验值大了 30%。诊断发现：训练集里所有结构都处在同一平衡体积，模型对体积轻微变化时的能量梯度完全是外推出来的——外推不准。

**诊断思路：** 热膨胀系数取决于声子频率对体积的导数，归根结底需要模型在平衡体积附近有足够的体积-能量采样点。纯静态弛豫结构只有一个体积，不够。解决办法是围绕平衡晶格随机生成一批稍微膨胀或收缩的结构，用少量白噪声覆盖近邻体积空间。

**输入：** 一个弛豫好的 Si 单胞

**目标：** 围绕平衡体积生成 30 个随机晶格扰动的结构，缩放幅度 4%，覆盖 ±4% 的体积范围

**参数设置：**
- `max_scaling` = `0.04` （4% 缩放幅度）
- `max_num` = `30`
- `perturb_angle` = `false` （先只看长度变化，避免角度扰动混入）

**输出：** 30 个结构，每个的晶格长度在原始值的 96%~104% 之间随机分布，原子分数坐标随晶格同步缩放

**怎么验证训练集质量改善：**
- 重训后重新跑准谐近似计算热膨胀系数，应更接近实验值
- 检查输出结构的体积分布直方图：应均匀覆盖 [V*(1-0.04), V*(1+0.04)]，不要在两端堆积
- 如果改善不够，增大 `max_scaling` 到 0.06~0.08，同时换 Sobol 引擎覆盖更均匀
- 抽查最短键长：如果 4% 缩放后最短键比 DFT 参考键长短了 > 0.2A，说明缩放过大，回调 `max_scaling`

### 什么时候加这张卡、什么时候不加

**加：**
- 模型对体积变化敏感，体积-能量曲线或热膨胀相关性质偏差大
- 训练集中所有结构体积过于集中，缺少近平衡体积散布
- 需要模型在 ±5% 体积范围内有内插能力

**不加：**
- 只需要特定方向、特定幅度的应变 → 用 `Lattice Strain` 更精确控制
- 体系本身刚度极大、体积变化无物理意义（如高压相），扰动只产生无意义结构
- 输入结构含有机分子但没开 `identify_organic` —— 分子内键会被错误拉伸

## 参数说明

### Engine Type（engine_type）
类型：`int`。默认：`1`。选择随机或准随机采样引擎。

随机引擎。`0` = Sobol 序列，少量样本时覆盖更均匀；`1` = 均匀随机，生成更快。样本数 ≥ 50 时两者差异不大；≤ 20 时优先 Sobol。

### Max Scaling（max_scaling）
类型：`float`。默认：`0.04`。设置晶格长度或角度扰动的最大幅度。

最大缩放幅度系数 `m`。晶格长度在 `[1-m, 1+m]` 内均匀采样。若 `perturb_angle=true`，角度也在同样相对范围内采样。0.01~0.03 适合补近平衡声子区，0.04~0.06 适合热膨胀覆盖，0.08+ 需抽查最近邻距离防止非物理键长。

### Max Num（max_num）
类型：`int`。默认：`50`。设置扰动或采样结构的输出数量。

每输入帧生成的扰动样本数。20~30 足够覆盖 ±4% 体积空间的 3D 长度自由度；50~100 适用于同时扰动角度的 6D 空间。注意这是随机采样不是系统网格——增大此值提高的是覆盖密度而非覆盖范围。

### Perturb Angle（perturb_angle）
类型：`bool`。默认：`True`。决定是否同时扰动晶格角度。

是否同时扰动晶格角度（alpha/beta/gamma）。开启后角度在原始值的 `[1-m, 1+m]` 倍范围内随机缩放——注意这是角度值的乘性缩放，不是加性偏移。非正交晶格体系（单斜、三斜）建议开启；正交或更高对称晶系可以先关掉，长度覆盖够用再加。

### Identify Organic（identify_organic）
类型：`bool`。默认：`False`。决定是否识别有机分子并保护内部几何。

开启后识别有机分子团簇，扰动时有机分子做刚性整体移动。分子晶体（MOF、有机半导体）必须开，否则分子内键长会被晶格缩放破坏。纯无机体系关掉以节省计算。

### Use Seed（use_seed）
类型：`bool`。默认：`False`。决定是否使用固定随机种子保证可复现。

固定随机种子。需要对比实验或可复现输出时勾选。纯探索阶段可关闭以增加随机覆盖。

### Seed（seed）
类型：`int`。默认：`0`。设置固定随机种子的整数值。

随机种子值。仅 `use_seed=true` 时生效。不同 seed 产生不同扰动分布。

生效条件：`use_seed=True`。

## 推荐预设

### 近平衡声子覆盖（max_scaling=2%，仅长度，Sobol）
```json
{
  "class": "CellScalingCard",
  "check_state": true,
  "params": {
    "engine_type": 0,
    "max_scaling": 0.02,
    "max_num": 30,
    "perturb_angle": false,
    "identify_organic": false,
    "use_seed": true,
    "seed": 42
  }
}
```

### 热膨胀 + 角度覆盖（max_scaling=5%，长度+角度，50 样本）
```json
{
  "class": "CellScalingCard",
  "check_state": true,
  "params": {
    "engine_type": 1,
    "max_scaling": 0.05,
    "max_num": 50,
    "perturb_angle": true,
    "identify_organic": false,
    "use_seed": true,
    "seed": 42
  }
}
```

### 大范围探索（max_scaling=8%，100 样本，含有机保护）
```json
{
  "class": "CellScalingCard",
  "check_state": true,
  "params": {
    "engine_type": 0,
    "max_scaling": 0.08,
    "max_num": 100,
    "perturb_angle": true,
    "identify_organic": true,
    "use_seed": true,
    "seed": 42
  }
}
```

## 推荐组合

- `Lattice Strain` -> `Lattice Perturb`：先系统扫特定方向应变，再补充随机近邻散布
- `Lattice Perturb` -> `Atomic Perturb`：晶格扰动后再加原子级坐标噪声
- `Super Cell` -> `Lattice Perturb`：扩胞后在更大尺度上做晶格扰动

## 常见问题

**输出全是相同结构或分布极窄。** `max_scaling` 是否太小（如 0.001）？或者 `max_num` 太小导致偶然均匀。先增大 `max_scaling` 到 0.02 以上，增大 `max_num` 到 20 以上。

**分子晶体输出中键被拉断。** `identify_organic` 没开。对于 MOF、有机晶体等，必须开启此开关，否则晶格缩放会直接拉伸分子内键。

**角度扰动后结构角度变化不符合预期。** 注意此卡的角度扰动是乘性缩放（原始角度乘以随机因子），不是加性偏移。如果原始角度是 90 度、`max_scaling=0.04`，新角度在 86.4~93.6 度之间。

**Sobol 和 Uniform 输出差异大。** `max_num` 较小时（< 20）Sobol 覆盖明显更均匀。样本数 ≥ 50 后两者差异缩小。如果在意复现性，固定 `engine_type` + `use_seed`。

## 输出标签

`LSc(max={max_scaling},{U|S})` —— `U` 表示 Uniform，`S` 表示 Sobol。

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。注意 Sobol 引擎的 scramble 也受 seed 控制，给定 seed 后序列完全确定。
