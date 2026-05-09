<!-- card-schema: {"card_name": "FPS Filter", "source_file": "src/NepTrainKit/ui/views/_card/fps_filter_card.py", "serialized_keys": ["params"]} -->

# FPS 过滤（FPS Filter）

`Group`: `Filter` | `Class`: `FPSFilterDataCard`

## 功能说明

基于 NEP 描述符空间的最远点采样（FPS），从大批量结构中挑选代表性子集，去除冗余同时保留多样性。需要提供一个 NEP 模型文件（如内置的 `nep89.txt` 或你自己训练的模型）用于生成描述符。

$$\mathbf{d}_i=\mathrm{NEP}(\text{structure}_i)$$

$$i_t=\arg\max_j\ \min_{i\in S_{t-1}}\lVert\mathbf{d}_j-\mathbf{d}_i\rVert_2,\quad S_t=S_{t-1}\cup\{i_t\}$$

## 操作示例

### 场景：2000 个候选结构训练出来效果不如 200 个精选的

你用多张生成卡产出了 2000 个候选结构，全部丢进训练集跑了 DFT 计算。重训后模型精度比只用前 200 个结构手动挑的版本还差——2000 个结构里大量重复构型，有效多样性反而被稀释了。

**诊断思路：** 卡片生成的候选集有大量统计冗余。比如 `Random Vacancy` 生成了 200 个空位结构，但从描述符角度看可能只有 15 种真正不同的局域环境。FPS 可以在描述符空间里按"最远优先"的原则挑出最不相似的子集，保留多样性去掉冗余。

**输入：** 2000 个候选结构，以及一个可用的 `nep89.txt` 模型文件

**目标：** 从 2000 个中挑出 200 个最具代表性的结构

**参数设置：**
- `Nep Path` = 指向你的 NEP 模型文件路径
- `Num Condition` = `[200]`
- `Min Distance Condition` = `[0.01]`

**输出：** 200 个结构，在描述符空间中两两距离尽可能大，覆盖输入集的多样性

**怎么验证挑选质量：**
- 重训后用同一组测试集对比：FPS 精选 200 个 vs 随机抽 200 个，前者精度应更高
- 如果 200 个不够，增大 `num_condition` 到 500
- 如果挑出的结构仍有重复感，增大 `min_distance_condition` 到 0.05~0.1
- 注意：FPS 只挑不造——如果输入池本身没有某类构型，FPS 不会凭空变出来

### 什么时候加这张卡、什么时候不加

**加：**
- 生成链条产生了大量结构（>500），需要去冗余
- 怀疑训练集有大量重复/高度相似的构型
- 需要在保持覆盖的前提下压缩训练集体积

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

### Batch Size（batch_size）
`int`，默认 1000。描述符批处理大小，只影响吞吐和内存。GPU 或大体系可以调大加速，内存紧张时适当调小。

## 推荐预设

### 轻度去重（保留 100 个，min_dist 0.01）
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "params": {
    "nep_path": "path/to/nep.txt",
    "n_samples": 100,
    "min_distance": 0.01,
    "backend": "auto",
    "batch_size": 1000
  }
}
```

### 常规去重（保留 200 个，min_dist 0.03）
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "params": {
    "nep_path": "path/to/nep.txt",
    "n_samples": 200,
    "min_distance": 0.03,
    "backend": "auto",
    "batch_size": 1000
  }
}
```

### 强去重（保留 50 个，min_dist 0.06）
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "params": {
    "nep_path": "path/to/nep.txt",
    "n_samples": 50,
    "min_distance": 0.06,
    "backend": "auto",
    "batch_size": 1000
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

**挑出的结构覆盖不全。** FPS 只能从现有池里选。如果某些局域环境在输入池里本来就没有，FPS 也无法补充。回到上游增加生成多样性。

**描述符计算很慢。** 大批量（>5000）时描述符计算可能耗时。先用 `FPS Filter` 之前控制好上游输出量。

## 输出标签

不新增专用 Config_type 标签。输出结构保留原标签。

## 可复现性

无随机性。同参数同输入 → 严格一致输出。FPS 是确定性贪心算法。
