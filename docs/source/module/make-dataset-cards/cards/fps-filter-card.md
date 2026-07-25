<!-- card-schema: {"card_name": "FPS Filter", "source_file": "src/NepTrainKit/ui/views/_card/fps_filter_card.py", "serialized_keys": ["params"]} -->

# FPS 过滤（FPS Filter）

`Group`: `Filter` | `Class`: `FPSFilterDataCard`

## 功能说明

基于结构级 NEP 均值描述符执行最远点采样（FPS），从大批量结构中挑选代表性子集。需要提供一个 NEP 模型文件（如内置的 `nep89.txt` 或你自己训练的模型）用于生成描述符。

卡片提供两种采样方式：

- `global`：兼容原有行为，对全部结构执行一次全局 FPS；
- `element_set`：先按元素集合分组，每组至少获得一个名额，剩余名额按组大小的平方根分配，再从组内中心开始 FPS。该模式仍使用原始结构均值描述符，不启用实验性的分元素 mean/std 或 robust scaling。

$$\mathbf{d}_i=\mathrm{NEP}(\text{structure}_i)$$

$$i_t=\arg\max_j\ \min_{i\in S_{t-1}}\lVert\mathbf{d}_j-\mathbf{d}_i\rVert_2,\quad S_t=S_{t-1}\cup\{i_t\}$$

## 操作示例

### 场景：2000 个候选结构训练出来效果不如 200 个精选的

你用多张生成卡产出了 2000 个候选结构，全部丢进训练集跑了 DFT 计算。重训后模型精度比只用前 200 个结构手动挑的版本还差——2000 个结构里大量重复构型，有效多样性反而被稀释了。

**诊断思路：** 卡片生成的候选集有大量统计冗余。比如 `Random Vacancy` 生成了 200 个空位结构，但从描述符角度看可能只有 15 种真正不同的局域环境。FPS 可以在描述符空间里按"最远优先"的原则挑出最不相似的子集，保留多样性去掉冗余。

**输入：** 2000 个候选结构，以及一个可用的 `nep89.txt` 模型文件

**目标：** 从 2000 个中挑出 200 个最具代表性的结构

**参数设置：** `strategy="global"`，`nep_path` 指向 NEP 模型，`n_samples=200`，`min_distance=0.01`。

**输出：** 200 个结构，在描述符空间中两两距离尽可能大，覆盖输入集的多样性

**怎么验证挑选质量：**
- 重训后用同一组独立测试集比较 FPS 精选 200 个与随机抽取 200 个；不要预设 FPS 一定更好
- 如果 200 个不够，增大 `num_condition` 到 500
- 如果挑出的结构仍有重复感，增大 `min_distance_condition` 到 0.05~0.1
- 注意：FPS 只挑不造——如果输入池本身没有某类构型，FPS 不会凭空变出来

### 场景：多个化学体系共用一个候选池

候选池同时包含单质、二元和三元结构时，全局 FPS 会让所有结构竞争同一预算。如果要求每种元素集合至少保留一个候选，可选择 `element_set`。例如 V、Co、Ni、V-Co、V-Ni、Co-Ni 和 V-Co-Ni 会形成 7 个组；当 `n_samples < 7` 时卡片明确报错，不静默漏组。

已有训练集可作为 warm start。只有元素集合相同的已有结构才参与候选距离初始化，卡片会优先选择当前训练集尚未覆盖的区域。已有训练集不计入 `n_samples` 输出数量。

### 什么时候加这张卡、什么时候不加

**加：**
- 生成链条产生了大量结构（>500），需要去冗余
- 怀疑训练集有大量重复/高度相似的构型
- 需要在保持覆盖的前提下压缩训练集体积
- 混合了多个元素集合，并要求每组都有明确预算

**不加：**
- 还在生成阶段，结构数量很少（<50）
- 输入池本身还没经过物理清洗（如已用 NEP 预筛、剔除异常结构）
- 模型还没训练出来——FPS 需要一个可用的 NEP 模型来生成描述符

## 参数说明
### Nep Path（nep_path）
`str`，默认 `MISSING`。指向用于生成 NEP 描述符的模型文件，必须是一个可用的 NEP 模型文件（如 `nep.txt` 或内置 `nep89.txt`）。路径为空或文件不存在时卡片直接报错。

### N Samples（n_samples）
`int`，默认 100。FPS 过滤后保留的代表结构数。这是输出数量，不是上游生成数量——DFT 预算固定时直接设你计划算的上限；但设太小会把稀有构型也筛掉。

### Min Distance（min_distance）
`float`，默认 0.01。描述符空间中的最小距离阈值。值越大去重越激进，但设太大 FPS 会过早停、选不够数；设太小又可能混入近重复结构。

### Backend（backend）
`str`，默认 `'auto'`。描述符计算后端。正常保持 `auto` 让程序自己选可用的 NEP 后端；只有在你需要调试性能或对比后端差异时才手动指定。

### Chunk Max Atoms（chunk_max_atoms）
`int`，默认 100000。每个描述符计算分块允许的总原子数，CPU 和 CUDA 使用同一语义。显存或内存不足时调小；它不是结构数量，同一分块中所有结构的原子数之和不能超过该值。

### Sampling Strategy（strategy）
`str`，默认 `"global"`。可选值：

| 值 | 行为 |
|---|---|
| `global` | 兼容旧流程；全部结构执行一次全局 FPS，从输入索引 0 开始 |
| `element_set` | 按元素集合分组、平方根分配名额、组内从描述符中心开始 FPS |

旧版卡片 JSON 没有该字段时按 `global` 恢复，避免加载旧工作流后静默改变输出。

### Existing Dataset Path（existing_dataset_path）
`str`，默认空字符串。仅 `strategy="element_set"` 使用。指向已有 `.xyz` 或 `.extxyz` 训练集，作为同元素集合内的 warm start。路径不存在、无法读取或不含结构时明确报错；`global` 模式忽略界面中隐藏的旧路径并序列化为空字符串。

## 推荐预设

### 兼容旧流程
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "params": {
    "nep_path": "path/to/nep.txt",
    "n_samples": 100,
    "min_distance": 0.01,
    "backend": "auto",
    "chunk_max_atoms": 100000,
    "strategy": "global",
    "existing_dataset_path": ""
  }
}
```

### 多元素集合平衡采样
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "params": {
    "nep_path": "path/to/nep.txt",
    "n_samples": 330,
    "min_distance": 0.01,
    "backend": "auto",
    "chunk_max_atoms": 100000,
    "strategy": "element_set",
    "existing_dataset_path": "path/to/train.xyz"
  }
}
```

## 推荐组合

- 任意生成链 → `NEP Dataset Display` 清洗 → `FPS Filter`：先去掉明显坏结构，再做代表性筛选
- `FPS Filter` → 导出 DFT 计算：控制进入 DFT 计算的结构数量，节省计算资源
- 多分支汇总后 → `FPS Filter`：先汇总各分支输出，再统一挑选

## 常见问题

**卡片报错"NEP file does not exist"。** `nep_path` 指向的文件不存在。确认路径正确。

**输出数量远小于 `num_condition`。** `min_distance_condition` 太严，无法选出足够多满足距离约束的结构。降低该阈值。

**平衡模式提示目标数量不足。** `n_samples` 小于元素集合组数。提高目标数量，或者在上游移除本轮不需要的化学体系。

**已有训练集没有改变结果。** warm start 只比较元素集合相同的结构。检查候选池与已有训练集的元素集合是否一致，并确认所用 NEP 模型支持这些元素。

**挑出的结构覆盖不全。** FPS 只能从现有池里选。如果某些局域环境在输入池里本来就没有，FPS 也无法补充。回到上游增加生成多样性。

**描述符计算很慢。** 大批量（>5000）时描述符计算可能耗时。先用 `FPS Filter` 之前控制好上游输出量。

## 输出标签

不新增专用 Config_type 标签。输出结构保留原标签。

## 可复现性

无随机性。同参数同输入会得到一致输出。`global` 模式保留原有索引 0 起点，因此改变输入顺序可能改变选择；`element_set` 从各组描述符中心开始，对输入顺序更稳定。特征完全相同的近重复结构仍可能互换原始索引，但覆盖等价。
