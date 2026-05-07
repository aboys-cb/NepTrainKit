# Make Dataset 配方示例（Recipes）

这份页面回答两个问题：

- 多张卡片应该按什么顺序排
- 每一步大概会生成什么样的结果

:::{tip}
下面的 JSON 采用“卡片配置片段”的写法，字段名与当前 `to_dict()` 保持一致，可直接对照卡片页里的推荐预设填写。第一版 recipes 不依赖截图，也不绑定固定示例文件。
:::

每条配方的卡片链只负责生成候选池。链条末尾写的 `Show NEP 清洗 -> FPS`
表示后处理方向：先在 `NEP Dataset Display` 里删掉明显异常结构，再做代表性采样。
如果你有当前体系的 NEP 模型，可以用它做预筛；没有时可以先做几何检查和人工抽查。

## 高熵合金

### 目标说明

从单一晶体原型出发，系统扫描多组目标配比，并把目标配比真正落到原子占位上，最后补一轮随机掺杂扩大局部化学环境。

### 输入假设

- 已有一个体相母相结构，或者先用 `Crystal Prototype Builder` 生成基础结构。
- 目标元素集合为 `Co,Cr,Ni,Al,Fe`。
- 希望最终拿到“多配比 + 多随机占位”的 HEA 训练集。

### 卡片顺序

`Crystal Prototype Builder -> Composition Sweep -> Random Occupancy -> Random Doping -> 导出 xyz -> Show NEP 清洗 -> FPS`

### 每步 JSON 配置

#### Step 1. `Crystal Prototype Builder`

```json
{
  "class": "CrystalPrototypeBuilderCard",
  "params": {
    "lattice": "fcc",
    "element": "Ni",
    "a_range": [3.52, 3.52, 0.1],
    "covera": 1.633,
    "auto_supercell": true,
    "max_atoms": 64,
    "rep": [2, 2, 2],
    "max_outputs": 1
  }
}
```

**每步预期输出：** 1 个基础 fcc 结构；原子数受 `max_atoms` 控制，适合作为后续配比扫描母体。

#### Step 2. `Composition Sweep`

```json
{
  "class": "CompositionSweepCard",
  "params": {
    "elements": "Co,Cr,Ni,Al,Fe",
    "order": "4,5",
    "method": "Sobol",
    "step": 0.1,
    "n_points": 24,
    "min_fraction": 0.05,
    "include_endpoints": true,
    "use_seed": true,
    "seed": 42,
    "max_outputs": 24,
    "budget_mode": "Equal+Reflow"
  }
}
```

**每步预期输出：** 24 个带 `Comp(...)` 标签的目标配比结构副本；此时还没有真正替换原子位点。

#### Step 3. `Random Occupancy`

```json
{
  "class": "RandomOccupancyCard",
  "params": {
    "source": "Auto (Comp tag)",
    "manual": "",
    "mode": "Exact",
    "samples": 2,
    "group_filter": "",
    "use_seed": true,
    "seed": 42
  }
}
```

**每步预期输出：** 每个目标配比落点成 2 个离散占位结构；如果 Step 2 输出 24 个配比点，这一步通常得到约 48 个随机合金候选。

#### Step 4. `Random Doping`

```json
{
  "class": "RandomDopingCard",
  "params": {
    "rules": [
      {
        "target": "Ni",
        "dopants": {"Co": 1.0, "Cr": 1.0},
        "use": "count",
        "count_mode": "random",
        "count": [1, 2],
        "percent": [0.0, 1.0]
      }
    ],
    "doping_type": "Exact",
    "max_structures": 2,
    "use_seed": true,
    "seed": 123
  }
}
```

**每步预期输出：** 对每个离散占位结构再扩出 2 个局部替换版本，进一步增加局部环境多样性。

### 最终数据集特征

- 同时覆盖四元和五元目标配比
- 每个目标配比都落到了实际原子位点
- 局部化学环境不只来自占位，还来自额外随机替换

### 常见失败点

- 只做 `Composition Sweep` 就停止：这只定义目标配比，不代表已经得到真实随机合金。
- `Random Occupancy` 的 `source` 没有读取 `Comp(...)`：会导致输出与目标配比不一致。
- `Random Doping` 规则写得过重：容易把原本的目标配比拉偏。

## 富缺陷表面

### 目标说明

从体相结构构造多取向表面，再在表面上加入插隙/吸附和空位，得到既有表面效应又有缺陷效应的数据集。

### 输入假设

- 输入是已弛豫的体相晶体。
- 目标任务是表面反应、表面稳定性或富缺陷 slab 训练。

### 卡片顺序

`Super Cell -> Random Slab -> Insert Defect -> Vacancy Defect Generation -> 导出 xyz -> Show NEP 清洗 -> FPS`

### 每步 JSON 配置

#### Step 1. `Super Cell`

```json
{
  "class": "SuperCellCard",
  "params": {
    "behavior_type": 0,
    "mode": "scale",
    "super_scale": [2, 2, 1],
    "target_cell": [20.0, 20.0, 20.0],
    "max_atoms": 200,
    "fixed_axis_flags": [false, false, true],
    "fixed_axis_scale": [1, 1, 1]
  }
}
```

**每步预期输出：** 1 个横向扩展、法向保持 1 倍的母相结构；适合作为表面切片前的体相输入。

#### Step 2. `Random Slab`

```json
{
  "class": "RandomSlabCard",
  "params": {
    "h_range": [0, 1, 1],
    "k_range": [0, 1, 1],
    "l_range": [1, 3, 1],
    "layer_range": [4, 8, 1],
    "vacuum_range": [12, 18, 2]
  }
}
```

**每步预期输出：** 多个低指数到中低指数 slab；真空层足够隔离上下表面，输出通常已经明显多于 Step 1。

#### Step 3. `Insert Defect`

```json
{
  "class": "InterstitialAdsorbateCard",
  "params": {
    "mode": 1,
    "species": "H,O",
    "insert_count": 1,
    "structure_count": 2,
    "min_distance": 1.2,
    "max_attempts": 50,
    "use_seed": true,
    "seed": 7,
    "axis": 2,
    "offset": 1.5
  }
}
```

**每步预期输出：** 每个 slab 表面再分出 2 个带表面吸附或插隙的候选结构。

#### Step 4. `Vacancy Defect Generation`

```json
{
  "class": "VacancyDefectCard",
  "params": {
    "engine_type": 1,
    "num_condition": 1,
    "use_num": false,
    "concentration_condition": 0.03,
    "count_mode": "fixed",
    "max_structures": 3,
    "use_seed": true,
    "seed": 19
  }
}
```

**每步预期输出：** 每个已有表面缺陷结构再扩出约 3 个轻度空位版本；空位强度受浓度控制，而不是固定删位数。

### 最终数据集特征

- 同时覆盖不同 slab 取向、厚度和真空层
- 既有表面，也有插隙/吸附和空位
- 缺陷强度可控，适合做表面缺陷多样性训练

### 常见失败点

- 先做空位再做 slab：会把“体相删位”误当成“表面缺陷”。
- `vacuum_range` 过小：上下表面镜像相互作用会污染结果。
- `concentration_condition` 设得过大：slab 太薄时容易直接破坏表面骨架。

## 磁性数据

### 目标说明

从一个磁性母相结构出发，先写入分组标签，再生成 FM / AFM / PM 分支，最后对已有磁矩做旋转补样。

### 输入假设

- 输入结构是磁性材料。
- 你希望同时覆盖多磁序，而不是只保留单一 MAGMOM 初始化。

### 卡片顺序

`Group Label -> Magnetic Order -> Magmom Rotation`

### 每步 JSON 配置

#### Step 1. `Group Label`

```json
{
  "class": "GroupLabelCard",
  "params": {
    "mode": "k-vector layers (recommended)",
    "kvec": "111",
    "group_a": "A",
    "group_b": "B",
    "overwrite": true
  }
}
```

**每步预期输出：** 1 个带 `group` 数组的结构；后续 `Magnetic Order` 若走 `group A/B` 模式会直接使用这些标签。

#### Step 2. `Magnetic Order`

```json
{
  "class": "MagneticOrderCard",
  "params": {
    "format": "Collinear (scalar)",
    "axis": [0.0, 0.0, 1.0],
    "magmom_map": "Fe:2.2",
    "use_element_dirs": false,
    "default_moment": 0.0,
    "apply_elements": "Fe",
    "gen_fm": true,
    "gen_afm": true,
    "afm_mode": "group A/B",
    "afm_kvec": "111",
    "afm_group_a": "A",
    "afm_group_b": "B",
    "afm_zero_unknown": true,
    "gen_pm": true,
    "pm_count": 8,
    "pm_direction": "sphere",
    "pm_cone_angle": 30.0,
    "pm_balanced": true,
    "use_seed": true,
    "seed": 88
  }
}
```

**每步预期输出：** 至少得到 1 个 FM、1 个 AFM 和 8 个 PM 候选；如果 `afm_mode` 设为 `group A/B`，真正起作用的是 `afm_group_a/b`，不是 `afm_kvec`。

#### Step 3. `Magmom Rotation`

```json
{
  "class": "MagneticMomentRotationCard",
  "params": {
    "elements": "Fe",
    "max_angle": 15.0,
    "num_structures": 3,
    "lift_scalar": true,
    "axis": [0.0, 0.0, 1.0],
    "disturb_magnitude": false,
    "magnitude_factor": [1.0, 1.0],
    "use_seed": true,
    "seed": 99
  }
}
```

**每步预期输出：** 对每个已有磁序结构再扩出 3 个小角度旋转版本，适合补充非共线附近样本。

### 最终数据集特征

- 同时包含 FM、AFM、PM 和小角度旋转分支
- AFM 的分组来源清晰，可追溯
- 随机性由 seed 控制，可做重复实验

### 常见失败点

- 没有先提供 `magmom_map` 或合理 `default_moment`：生成的磁矩幅值会退化。
- `afm_mode` 设成 `group A/B` 却没有 `group` 数组：AFM 分支会失去预期分组信息。
- 直接用 `Magmom Rotation` 处理没有初始磁矩的结构：旋转对象本身就不存在。

## 有机构象

### 目标说明

为有机分子或分子晶体生成构象样本，再叠加轻微原子扰动，得到更适合 NEP 训练的局部结构分布。

### 输入假设

- 输入里包含有机分子，或者是分子晶体。
- 希望同时覆盖扭转构象和局部热噪声，而不是只做刚性平移。

### 卡片顺序

`Organic Mol Config -> Atomic Perturb -> 导出 xyz -> Show NEP 清洗 -> FPS`

### 每步 JSON 配置

#### Step 1. `Organic Mol Config`

```json
{
  "class": "OrganicMolConfigPBCCard",
  "params": {
    "perturb_per_frame": 8,
    "torsion_range_deg": [-25.0, 25.0],
    "max_torsions_per_conf": 2,
    "gaussian_sigma": 0.01,
    "pbc_mode": "molecule in box",
    "local_cutoff": 5,
    "local_subtree": 100,
    "bond_detect_factor": 1.15,
    "bond_keep_min_factor": 0.75,
    "bond_keep_max_factor": 1.30,
    "bond_keep_max_enable": true,
    "nonbond_min_factor": 0.80,
    "max_retries": 50,
    "mult_bond_factor": 1.10,
    "nonpbc_box_size": 20.0,
    "bo_c_const": 0.35,
    "bo_threshold": 0.25,
    "use_seed": true,
    "seed": 5
  }
}
```

**每步预期输出：** 每帧输入扩出 8 个构象版本；主变化来自扭转角和局部高斯扰动，而不是晶格变形。

#### Step 2. `Atomic Perturb`

```json
{
  "class": "PerturbCard",
  "params": {
    "engine_type": 0,
    "max_distance": 0.03,
    "max_num": 3,
    "identify_organic": true,
    "use_element_scaling": false,
    "element_scalings": {},
    "use_seed": true,
    "seed": 6
  }
}
```

**每步预期输出：** 每个有机构象再扩出 3 个轻微原子热扰动版本；`organic=true` 会尽量保持分子内部拓扑不被原子级随机位移破坏。

### 最终数据集特征

- 同时覆盖扭转构象和局部热噪声
- 分子内部键拓扑更容易保持稳定
- 适合作为有机体系初始训练集或补样集

### 常见失败点

- `organic` 没开：原子级扰动更容易拉坏分子内部键。
- `torsion_range_deg` 设得太宽：容易生成明显高能或自相交构象。
- 先做大幅原子扰动再做构象采样：会把构象变化和非物理断键混在一起。
