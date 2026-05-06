<!-- card-schema: {"card_name": "Atomic Perturb", "source_file": "src/NepTrainKit/ui/views/_card/perturb_card.py", "serialized_keys": ["params"]} -->

# 原子扰动（Atomic Perturb）

`Group`: `Perturbation` | `Class`: `PerturbCard`

## 功能说明

对原子坐标施加随机位移扰动，补充近平衡势能面的局部位移样本。每个原子在 `[-max_distance, max_distance]` 范围内独立随机偏移，晶格保持不变。

$$\Delta\mathbf{r}_i=\boldsymbol\xi_i\odot d_i,\quad \boldsymbol\xi_i\in[-1,1]^3,\quad \mathbf{r}_i'=\mathbf{r}_i+\Delta\mathbf{r}_i$$

开启 `identify_organic` 后有机分子团簇做刚性整体移动，避免分子内键被拉伸。

## 操作示例

### 场景：静力学测试集上力 MAE 只有 50 meV/A，但 MD 轨迹上跳到 200 meV/A

你在 Si 上训练了一个 NEP 模型，用 DFT 静态计算的数据做验证，力的 MAE 只有 50 meV/A。但用这个模型跑 300K NVT MD 时，力的误差跳到了 200 meV/A——因为训练集里所有原子都在平衡位置，模型没见过原子偏离平衡态的构型。

**诊断思路：** 静态弛豫结构代表势能面极小点。MD 轨迹在极小点周围的势能面上采样，模型需要对平衡位置附近的位移方向有力学响应。如果训练集全是极小点，模型对偏离平衡的力梯度完全是外推的。解决方法是对每个结构的原子做随机小位移，让模型学习近平衡势能面的曲率。

**输入：** 一个弛豫好的 Si 超胞（64 原子）

**目标：** 生成 50 个扰动版本，每个原子的位移幅度 ≤ 0.15A（约室温 Si 振动幅度的 2~3 倍）

**参数设置：**
- `max_distance` = `0.15` （单位 A）
- `max_num` = `50`
- `engine_type` = `0` （Sobol，少量样本覆盖更均匀）

**输出：** 50 个结构，每个原子的坐标在原始位置附近随机偏移 0~0.15A

**怎么验证训练集质量改善：**
- 重训后用一段 MD 轨迹做推理，力的 MAE 应该显著下降，接近静力学测试集水平
- 抽查最短键长：0.15A 位移下 Si-Si 键长变化应在 ±0.3A 以内（两个相邻原子可能向不同方向偏移）。如果出现 < 1.8A 的短键，降低 `max_distance`
- 如果改善不够，增大 `max_num` 到 100、也增大 `max_distance` 到 0.25A
- 如果体系含多种元素且质量差异大（如含 H + 重元素），考虑开启 `use_element_scaling` 给轻元素更大位移

### 什么时候加这张卡、什么时候不加

**加：**
- 训练集主要由静态弛豫结构组成，缺少力方向信息
- 模型在 MD 模拟或非平衡构型上力预测偏差大
- 需要补近平衡势能面的曲率（力常数）信息

**不加：**
- 训练集已有大量 MD 轨迹数据——MD 本身已覆盖热振动位移，再加随机扰动是冗余的
- 输入结构未弛豫——扰动一个本身就不在极小点的结构没有意义，先弛豫再扰动
- 位移幅度设置的过大（> 0.5A），产生大量非物理构型——这种大位移应该用 MD 而不是随机扰动

## 参数说明

**`engine_type`**（int，默认 1）：随机引擎。`0` = Sobol，少量样本时位移方向覆盖更均匀；`1` = Uniform，更快。样本数 ≥ 100 后差异缩小。

**`max_distance`**（float，默认 0.3 A）：每个原子的最大位移距离。这是绝对长度（单位 A），不是百分比。室温下典型固体原子振动幅度约 0.05~0.1A，设 0.15~0.3A 通常合适。含 H 等轻元素时考虑开启 `use_element_scaling` 单独调大 H 的幅度。0.01~0.05A 适合微扰验证，0.3~0.5A 需要后筛检查非物理键长。

**`max_num`**（int，默认 50）：每输入帧生成的扰动样本数。20~50 适合轻量补样；50~100 适合常规覆盖；100+ 建议后接 `FPS Filter` 去重。

**`identify_organic`**（bool，默认 false）：有机团簇识别。分子晶体必须开启，纯无机关闭。

**`use_element_scaling`**（bool，默认 false）：按元素设置不同扰动幅度。开启后 `max_distance` 被 `element_scalings` 中对应元素的系数覆盖。多元素体系（如含 H+重元素）建议开启。

**`element_scalings`**（dict，默认 {}）：元素 → 最大位移的映射，如 `{"H":0.5,"O":0.15,"Si":0.1}`。仅 `use_element_scaling=true` 时生效。未列出的元素使用 `max_distance`。

**`use_seed`**（bool，默认 false）：固定随机种子。

**`seed`**（int，默认 0）：随机种子值。仅 `use_seed=true` 时生效。

## 推荐预设

### 室温微扰（0.1A，Sobol，30 样本）
```json
{
  "class": "PerturbCard",
  "check_state": true,
  "engine_type": 0,
  "max_distance": 0.1,
  "max_num": 30,
  "identify_organic": false,
  "use_element_scaling": false,
  "element_scalings": {},
  "use_seed": true,
  "seed": 42
}
```

### 常规补力场（0.2A，Uniform，50 样本）
```json
{
  "class": "PerturbCard",
  "check_state": true,
  "engine_type": 1,
  "max_distance": 0.2,
  "max_num": 50,
  "identify_organic": false,
  "use_element_scaling": false,
  "element_scalings": {},
  "use_seed": true,
  "seed": 42
}
```

### 多元素差异化扰动（H 0.5A / 重元素 0.15A，100 样本）
```json
{
  "class": "PerturbCard",
  "check_state": true,
  "engine_type": 0,
  "max_distance": 0.15,
  "max_num": 100,
  "identify_organic": false,
  "use_element_scaling": true,
  "element_scalings": {"H": 0.5, "C": 0.2, "O": 0.15},
  "use_seed": true,
  "seed": 42
}
```

## 推荐组合

- `Lattice Strain` -> `Atomic Perturb`：先做全局形变，再补局部位移
- `Shear Matrix Strain` -> `Atomic Perturb`：剪切变形后加坐标噪声
- `Atomic Perturb` -> `FPS Filter`：大批量扰动后去重

## 常见问题

**扰动后出现异常短键（< 1.0A）。** 两个相邻原子随机位移方向恰好相反，距离骤减。降低 `max_distance` 或增大 `max_num` 后通过 `FPS Filter` 筛掉短键结构。这是随机扰动的固有局限——它不做碰撞检测。

**分子晶体中分子碎掉了。** `identify_organic` 没开。开启后整个分子团簇做刚性平移，分子内键长完全保留。

**扰动后力 MAE 没改善。** 可能 `max_distance` 太小，位移落在模型已经学好的线性区内。增大到 0.3~0.5A 试试。也可能体系本身的势能面非谐性很强，随机扰动方向不一定是最需要的——此时用 `Vib Mode Perturb` 替代，沿真实振动模方向做位移。

**Sobol 引擎 output 和预期不一致。** Sobol 序列的 scramble 受 seed 控制。如果 `use_seed=false`，不同运行的 Sobol 序列也不同。需要可复现时必须 `use_seed=true`。

## 输出标签

`Pert(d={max_distance},{U|S})` —— `U` 表示 Uniform，`S` 表示 Sobol。

## 可复现性

勾选 `use_seed` + 固定 `seed` → 相同输入可复现。
