<!-- card-schema: {"card_name": "Random Occupancy", "source_file": "src/NepTrainKit/ui/views/_card/random_occupancy_card.py", "serialized_keys": ["params"]} -->

# 随机占位（Random Occupancy）

`Group`: `Alloy` | `Class`: `RandomOccupancyCard`

## 功能说明

在给定总成分约束下，将目标配比真正落到原子占位上。读取 `Comp(...)` 标签或手工输入的成分字符串，用精确或随机模式把各元素分配到离散晶格位点，输出带 `Occ(...)` 标签的真实化学占位结构。

典型用法：接在 `Composition Sweep` 之后，把目标配比计划转化为可以跑 DFT/NEP 的实际结构。

## 操作示例

### 场景：同成分不同排布下的能量预测，模型偏差从 5 跳到 50 meV/atom

你在 CoCrNi 训练集上跑了 `Composition Sweep`，覆盖了从纯元素到等摩尔的各种配比。但每个配比只生成了一个占位结构——对 Co0.33Cr0.33Ni0.33，训练集里 Cr 永远在角落、Co 永远在面心。模型学到的不是"这个成分"，而是"这个成分 + 这个特定排布"。拿到另一个同样成分但 Cr/Co 位置互换的结构，能量预测偏差从 5 meV/atom 跳到了 50 meV/atom。

**诊断思路：** 对给定成分，短程化学有序度——哪种原子偏好和哪种原子做邻居——显著影响总能量和局域力。训练集里每种成分只有一个排布样本，模型就把成分和特定排布绑死了。解决：给每个目标配比生成多个不同占位版本。

**输入：** 一批带 `Comp(Co=0.3333,Cr=0.3333,Ni=0.3333)` 标签的结构（来自上游 `Composition Sweep`）

**目标：** 每个目标配比生成 5 个不同原子排布版本，覆盖排布多样性

**参数设置：**
- `source` = `Auto (Comp tag)` （自动读取上游 Comp 标签）
- `mode` = `Exact` （精确匹配目标配比，每次排布不同但元素计数一致）
- `samples` = `5`

**输出：** 每个输入结构产生 5 个带 `Occ(E)` 标签的结构，元素组成与 Comp 标签一致但原子排布各不相同

**怎么验证训练集质量改善：**
- 重训后对同一成分的不同排布跑推理，能量方差应该合理——不应全坍缩到一个值，也不应异常发散
- 抽查几个占位输出，确认元素计数与目标配比一致（Exact 模式）或统计上接近（Random 模式）
- 如果不同排布之间的能量差异非常小且体系对排布不敏感，可以减少 `samples`；差异大则增大
- 如果上游没有 `Comp(...)` 标签导致输出=输入，切换到手工模式并填写 `manual` 成分字符串

### 什么时候加这张卡、什么时候不加

**加：**
- 上游有 `Composition Sweep` 或手工定义了目标配比，需要落到具体原子占位
- 同一成分下需要多个不同原子排布来覆盖短程化学有序度
- 高熵合金、固溶体等需要成分-排布联合采样的体系

**不加：**
- 不需要改变原子占位
- 需要直接指定替换规则而非从配比出发 → 用 `Random Doping`
- 输入本身已经是真实的离散占位结构且不再需要多样性

## 参数说明

### `source`（source）

`Auto (Comp tag)` 或 `Manual`。Auto 从输入结构的 `Config_type` 中读取 `Comp(...)` 标签作为目标配比，适合接在 `Composition Sweep` 之后。Manual 从 `manual` 字段读取手工配比字符串。

### `manual`（manual）

成分字符串，如 `Co:0.333,Cr:0.333,Ni:0.334`。仅在 `source=Manual` 时生效。比例会被归一化。
如果只输入单个元素，如 `Ge`，会按 `Ge:1.0` 处理；输入 `Ge,C` 则两个元素默认权重都是 `1.0`。

### `mode`（mode）

`Exact` 或 `Random`。Exact 精确匹配目标配比，每个原子的元素分配满足目标计数（向下取整后按余数补足），适合对比实验。Random 按目标比例概率采样，整体统计接近但不严格匹配，适合探索性跑样。

### `samples`（samples）

整数，每个目标配比生成多少个不同占位版本。典型值 1-20。乘以上游配比点数 = 总输出量。

### `group_filter`（group_filter）

逗号分隔的 group 标签，如 `A,B`。限制只在这些 group 内的位点上做占位分配。需要输入结构已带 `atoms.arrays['group']`。

### `use_seed` / `seed`

勾选 `use_seed` 后固定随机路径，`seed` 不同取值产生不同占位分布。结合输入结构的 stable config ID 为每个样本派生独立种子。

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

- `Composition Sweep` → `Random Occupancy`：标准合金 pipeline，配比 → 落位。
- `Group Label` → `Random Occupancy`：先打 group 标签，再限制占位区域。
- `Random Occupancy` → `Atomic Perturb`：占位后加坐标噪声松驰局部应力。

## 常见问题

**输出 = 输入，没有变化。** 上游没有 `Comp(...)` 标签且 `manual` 为空。检查 `source` 设置，或切换到 Manual 模式并填写成分。

**占位后元素数量与标签不一致。** `Random` 模式下统计浮动是正常的。换 `Exact` 模式可精确匹配。

**输出数量远超预期。** 输出 = 输入帧数 x `samples`。上游 500 个配比点 x `samples=5` = 2500 个结构。先估算总规模再跑。

**group_filter 不生效。** 检查输入结构是否有 `atoms.arrays['group']` 且标签拼写完全匹配。

## 输出标签

`Occ(E)` / `Occ(R)` / `Occ(E,s=...)` / `Occ(R,s=...)`。E = Exact，R = Random。使用 seed 时附加种子值便于追踪。

## 可复现性

勾选 `use_seed` + 固定 `seed` 可复现。种子与输入结构的 stable config ID 联合派生样本级种子，相同配置 + 相同 seed → 相同占位序列。注意输入结构顺序变化会影响结果。
