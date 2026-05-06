<!-- card-schema: {"card_name": "FPS Filter", "source_file": "src/NepTrainKit/ui/views/_card/fps_filter_card.py", "serialized_keys": ["params"]} -->

# FPS 过滤（FPS Filter）

`Group`: `Filter`  
`Class`: `FPSFilterDataCard`  
`Source`: `src/NepTrainKit/ui/views/_card/fps_filter_card.py`

## 功能说明
基于特征距离执行最远点采样（FPS），用于在完成物理清洗后压缩冗余并保留多样性。

它最适合的场景是：从已经生成好的大批量结构中挑出代表性子集。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$\mathbf{d}_i=\mathrm{NEP89}(x_i)$$
$$i_t=\arg\max_j\ \min_{i\in S_{t-1}}\lVert\mathbf{d}_j-\mathbf{d}_i\rVert_2,\quad S_t=S_{t-1}\cup\{i_t\}$$
$$\min_{i\in S_t,j\in S_t,i\ne j}\lVert\mathbf{d}_i-\mathbf{d}_j\rVert_2\ge d_{\min}\ (\text{if feasible})$$

## 操作示例
### 场景：从已经生成好的大批量结构中挑出代表性子集

**输入：** 一个规模已经较大的候选数据集，以及可用的 `nep89` 模型路径

**目标：** 在不完全手工筛选的前提下，去掉明显重复的结构，只保留覆盖面更好的代表帧

**参数设置：**
- `nep_path` 指向可用的模型文件
- `num_condition` 先限定想保留的目标数量
- `min_distance_condition` 用来控制样本间最小描述符距离

**输出：** 数量更少但分布更均匀的代表性结构子集

**怎么验证结果合理：**
- 确认输出数量接近 `num_condition`
- 抽查保留下来的结构是否确实覆盖了不同局部环境
- 若输出过少，先放宽 `min_distance_condition` 或检查 `nep_path` 是否可用

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 数据量大但冗余高，训练收益下降。
- 目标任务 (Target objective): 在删除非物理结构后保留代表性结构分布。
- 建议添加条件 (Add-it trigger): 已完成 `nep89` 预测筛查并剔除不合理结构。
- 不建议添加条件 (Avoid trigger): 仍处于样本生成早期或尚未完成物理清洗。
> 物理提示 (Physics caution): 过滤阈值只是选样规则，不是物理约束；先确认输入池本身已经过基本清洗。

## 输入前提
- 先导出 xyz 并在第一个模块用 `nep89` 预测，删除不合理结构后再执行 FPS。
- 确认描述符模型路径 `nep_path` 有效。
- 先在小集试 `min_distance_condition` 对保留率影响。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> UI 控件读取后的纯参数对象。
- 控件标签 (Caption): `Operation Params`
- 控件解释 (Widget): 由 NEP 路径、目标数量、最小 descriptor 距离和当前 NEP 后端配置组合生成的内部参数字典。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `{"nep_path": "src/NepTrainKit/Config/nep89.txt", "n_samples": 100, "min_distance": 0.01, "backend": "auto", "batch_size": 500}`
- 含义 (Meaning): UI-independent 参数快照，供 dataset-level operation、测试和未来批处理入口复用。
- 对输出规模/物理性的影响: 本字段本身不新增物理行为；其内容与下面的 legacy 字段保持同一组 FPS 参数。
- 怎么判断该开还是该关: 这是序列化结构字段，不是用户开关；导入旧 JSON 时仍可由 legacy 字段恢复。

### `nep_path` (Nep Path)
- UI Label: `Nep Path`
- 字段映射 (Field mapping): 序列化键 `nep_path` <-> 界面标签 `Nep Path`。
- 控件标签 (Caption): `Nep Path`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"src/NepTrainKit/Config/nep89.txt"`
- 含义 (Meaning): 特征模型路径 (NEP model path)。
- 对输出规模/物理性的影响: 用于距离特征计算，路径失效会导致过滤退化。
- 怎么判断该开还是该关: 只在你明确知道该字段会命中输入结构时填写；不确定时先用最小样本验证命中情况。
- 配置建议 (Practical note): 用于生成描述符，默认使用 `src/NepTrainKit/Config/nep89.txt`，可替换为你自己的模型路径。

### `num_condition` (Num Condition)
- UI Label: `Num Condition`
- 字段映射 (Field mapping): 序列化键 `num_condition` <-> 界面标签 `Num Condition`。
- 控件标签 (Caption): `Num Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[100]`
- 含义 (Meaning): 采样数量控制 (sample count control)。
- 对输出规模/物理性的影响: 主要影响输出规模与耗时，不是幅度主控参数。
- 物理直觉 / 典型值: 它主要决定每帧会扩出多少个结构，直接影响后续计算预算与重复率。
- 推荐范围 (Recommended range):
  - 保守：50-100
  - 平衡：100-200
  - 探索：200-500

### `min_distance_condition` (Min Distance Condition)
- UI Label: `Min Distance Condition`
- 字段映射 (Field mapping): 序列化键 `min_distance_condition` <-> 界面标签 `Min Distance Condition`。
- 控件标签 (Caption): `Min Distance Condition`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): float（单值输入）
- 默认值 (Default): `[0.01]`
- 含义 (Meaning): 最小特征距离阈值 (minimum descriptor distance)。
- 对输出规模/物理性的影响: 阈值越大去冗余越强，保留样本越少。
- 物理直觉 / 典型值: 阈值越保守，越能避免重叠或错配，但输出数量也往往更快下降。
- 推荐范围 (Recommended range):
  - 保守：0.05-0.06
  - 平衡：0.05-0.06
  - 探索：0.05-0.06（仅探索）

## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "nep_path": "",
  "num_condition": [
    10
  ],
  "min_distance_condition": [
    0.001
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "nep_path": "",
  "num_condition": [
    20
  ],
  "min_distance_condition": [
    0.001
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "FPSFilterDataCard",
  "check_state": true,
  "nep_path": "",
  "num_condition": [
    60
  ],
  "min_distance_condition": [
    0.001
  ]
}
```

## 推荐组合
- 任意生成分支 -> 本卡: 建议把下采样放在分支末端，统一控制导出冗余。
- Card Group -> 本卡（组外）: 先汇总分支结果，再执行距离约束采样，避免组内依赖混乱。
- 过滤卡尽量放在生成链末端，不要放在结构还没展开之前。

## 常见问题与排查
- 输出过少时，通常是距离阈值太严、目标数量太小，或模型路径/输入数据本身存在问题。
- 如果筛选后的代表性不好，先回头检查前一阶段的数据分布，再调整筛选阈值，而不是只继续压缩数量。
- 过滤卡只负责“选”，不负责“造”；上游数据本身不合理时，过滤不会自动把它变成高质量数据。

## 输出标签 / 元数据变更
- 该卡片本身不新增专用 Config_type 标签。

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
