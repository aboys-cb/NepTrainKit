<!-- card-schema: {"card_name": "Random Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/random_occupancy_card.py", "serialized_keys": ["params"]} -->

# 随机占位（Random Occupancy）

**分类：** 合金与组分

## 功能说明

在给定总成分约束下，将目标配比真正落到原子占位上。读取 `Comp(...)` 标签或手工输入的成分字符串，用精确或随机模式把各元素分配到离散晶格位点，输出带 `Occ(...)` 标签的真实化学占位结构。

典型用法：接在 `Composition Space Sampling` 之后，把目标配比计划转化为可以跑 DFT/NEP 的实际结构。

## 原理与公式

设可占位点数为 $N$，归一化目标比例为 $p_e$。`精确`模式使用最大余数法：

$$
n_e^{(0)}=\lfloor Np_e\rfloor,\qquad
\sum_e n_e=N,
$$

不足的位点按 $Np_e-n_e^{(0)}$ 从大到小补齐，然后把含 $n_e$ 个元素 $e$ 的占位池
随机打乱。`随机`模式则一次性从多项分布
$\mathbf n\sim\operatorname{Multinomial}(N,\mathbf p)$ 抽取计数，再打乱占位池。
因此精确模式每个样本的元素计数相同；随机模式只在大量样本的统计平均上接近目标比例。
`分组筛选`会把 $N$ 限制为命中 group 标签的位点数。

## 操作示例

### 场景：同成分不同排布下的能量预测，模型偏差从 5 跳到 50 meV/atom

你在 CoCrNi 训练集上跑了 `Composition Space Sampling`，覆盖了从纯元素到等摩尔的各种配比。但每个配比只生成了一个占位结构——对 Co0.33Cr0.33Ni0.33，训练集里 Cr 永远在角落、Co 永远在面心。模型学到的不是"这个成分"，而是"这个成分 + 这个特定排布"。拿到另一个同样成分但 Cr/Co 位置互换的结构，能量预测偏差从 5 meV/atom 跳到了 50 meV/atom。

**诊断思路：** 对给定成分，短程化学有序度——哪种原子偏好和哪种原子做邻居——显著影响总能量和局域力。训练集里每种成分只有一个排布样本，模型就把成分和特定排布绑死了。解决：给每个目标配比生成多个不同占位版本。

**输入：** 一批带 `Comp(Co=0.3333,Cr=0.3333,Ni=0.3333)` 标签的结构（来自上游 `Composition Space Sampling`）

**目标：** 每个目标配比生成 5 个不同原子排布版本，覆盖排布多样性

**参数设置：**
- `来源`（`source`） = `Auto (Comp tag)` （自动读取上游 Comp 标签）
- `模式`（`mode`） = `Exact` （精确匹配目标配比，每次排布不同但元素计数一致）
- `样本数`（`samples`） = `5`

**输出：** 每个输入结构产生 5 个带 `Occ(E)` 标签的结构，元素组成与 Comp 标签一致但原子排布各不相同

**怎么验证训练集质量改善：**
- 重训后对同一成分的不同排布跑推理，能量方差应该合理——不应全坍缩到一个值，也不应异常发散
- 抽查几个占位输出，确认元素计数与目标配比一致（Exact 模式）或统计上接近（Random 模式）
- 如果不同排布之间的能量差异非常小且体系对排布不敏感，可以减少 `样本数`（`samples`）；差异大则增大
- 如果上游没有 `Comp(...)` 标签导致输出=输入，切换到手工模式并填写 `手动`（`manual`）成分字符串

### 什么时候加这张卡、什么时候不加

**加：**
- 上游有 `Composition Space Sampling` 或手工定义了目标配比，需要落到具体原子占位
- 同一成分下需要多个不同原子排布来覆盖短程化学有序度
- 高熵合金、固溶体等需要成分-排布联合采样的体系

**不加：**
- 不需要改变原子占位
- 需要直接指定替换规则而非从配比出发 → 用 `Random Doping`
- 输入本身已经是真实的离散占位结构且不再需要多样性

## 参数说明

### 来源（source）

`str`，默认 `Auto (Comp tag)`。`Auto` 从输入结构的 `Config_type` 中读取 `Comp(...)` 标签作为目标配比，适合接在 `Composition Space Sampling` 之后。`手动`（`manual`）从下面的 `手动`（`manual`）字段读取手工配比字符串。

### 手动（manual）

`str`，默认空。仅在 `source=Manual` 时显示“元素 / 目标比例”表格；序列化仍保存为 `Co:0.333,Cr:0.333,Ni:0.334`。比例会被自动归一化，旧字符串工作流仍可直接恢复。

### 模式（mode）

`str`，默认 `Exact`。`Exact` 精确匹配目标配比，每个原子的元素分配满足目标计数（向下取整后按余数补足），适合对比实验。`Random` 按目标比例概率采样，整体统计接近但不严格匹配，适合探索性跑样。

### 样本数（samples）

`int`，默认 1。每个输入结构为每个目标配比生成多少个不同占位版本。典型值 1-20。注意总输出 = 上游配比点数 x 这个值，先估算规模再跑。

### 分组筛选（group_filter）

`str`，默认空。逗号分隔的 group 标签，如 `A,B`。限制只在这些 group 内的位点上做占位分配。填写后，输入必须带 `atoms.arrays['group']` 且至少有一个标签命中；否则卡片会明确报错，不会退化成全结构占位。

### 使用随机种子（use_seed）

`bool`，默认 false。打开后固定随机路径，`随机种子`（`seed`）不同取值产生不同占位分布。程序会结合输入结构的 stable config ID 为每个样本派生独立种子。

### 随机种子（seed）

`int`，默认 0。不同取值产生不同的占位分布。

生效条件：`use_seed=True`。

## 推荐预设

### 单样本落地（每配比 1 排布，快速验证占位路径）
```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "source": "Auto (Comp tag)",
  "manual": "",
  "mode": "Exact",
  "samples": [1],
  "group_filter": "",
  "use_seed": false,
  "seed": [0]
}
```

### 多样性占位（每配比 5 排布，常规训练用）
```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "source": "Auto (Comp tag)",
  "manual": "",
  "mode": "Exact",
  "samples": [5],
  "group_filter": "",
  "use_seed": true,
  "seed": [42]
}
```

### 高多样性子晶格（每配比 20 排布，仅限 group A）
```json
{
  "class": "RandomOccupancyCard",
  "check_state": true,
  "source": "Auto (Comp tag)",
  "manual": "",
  "mode": "Random",
  "samples": [20],
  "group_filter": "A",
  "use_seed": true,
  "seed": [42]
}
```

## 推荐组合

- `Composition Space Sampling` → `Random Occupancy`：标准合金 pipeline，配比 → 落位。
- `Group Label` → `Random Occupancy`：先打 group 标签，再限制占位区域。
- `Random Occupancy` → `Atomic Perturb`：占位后加坐标噪声松驰局部应力。

## 常见问题

**提示缺少目标成分。** 上游没有 `Comp(...)` 标签且 `手动`（`manual`）为空时，卡片会停止并报错。检查 `来源`（`source`）设置，或切换到 Manual 模式并填写成分；它不会再把原结构当成成功输出。

**占位后元素数量与标签不一致。** `Random` 模式下统计浮动是正常的。换 `Exact` 模式可精确匹配。

**输出数量远超预期。** 输出 = 输入帧数 x `样本数`（`samples`）。上游 500 个配比点 x `samples=5` = 2500 个结构。先估算总规模再跑。

**group_filter 报错。** 检查输入结构是否有 `atoms.arrays['group']` 且标签拼写完全匹配。缺少数组和零命中都会报错，避免本来只想改 A 组却意外改完整个结构。

## 输出标签

`Occ(E)` / `Occ(R)` / `Occ(E,s=...)` / `Occ(R,s=...)`。E = Exact，R = Random。使用 seed 时附加种子值便于追踪。

## 可复现性

勾选 `使用随机种子`（`use_seed`） + 固定 `随机种子`（`seed`）可复现。种子与输入结构的 stable config ID 联合派生样本级种子，相同配置 + 相同 seed → 相同占位序列。注意输入结构顺序变化会影响结果。
