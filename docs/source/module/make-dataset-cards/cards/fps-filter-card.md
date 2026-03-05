<!-- card-schema: {"card_name": "FPS Filter", "source_file": "src/NepTrainKit/ui/views/_card/fps_filter_card.py", "serialized_keys": ["nep_path", "num_condition", "min_distance_condition"]} -->

# FPS 过滤（FPS Filter）

`Group`: `Filter`  
`Class`: `FPSFilterDataCard`  
`Source`: `src/NepTrainKit/ui/views/_card/fps_filter_card.py`

## 功能说明
基于特征距离执行最远点采样（FPS），用于在完成物理清洗后压缩冗余并保留多样性。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::

### 关键公式 (Core equations)
$$\mathbf{d}_i=\mathrm{NEP89}(x_i)$$
$$i_t=\arg\max_j\ \min_{i\in S_{t-1}}\lVert\mathbf{d}_j-\mathbf{d}_i\rVert_2,\quad S_t=S_{t-1}\cup\{i_t\}$$
$$\min_{i\in S_t,j\in S_t,i\ne j}\lVert\mathbf{d}_i-\mathbf{d}_j\rVert_2\ge d_{\min}\ (\text{if feasible})$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 数据量大但冗余高，训练收益下降。
- 目标任务 (Target objective): 在删除非物理结构后保留代表性结构分布。
- 建议添加条件 (Add-it trigger): 已完成 `nep89` 预测筛查并剔除不合理结构。
- 不建议添加条件 (Avoid trigger): 仍处于样本生成早期或尚未完成物理清洗。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先导出 xyz 并在第一个模块用 `nep89` 预测，删除不合理结构后再执行 FPS。
- 确认描述符模型路径 `nep_path` 有效。
- 先在小集试 `min_distance_condition` 对保留率影响。


## 参数说明（完整）
### `nep_path` (Nep Path)
- UI Label: `Nep Path`
- 字段映射 (Field mapping): 序列化键 `nep_path` <-> 界面标签 `Nep Path`。
- 控件标签 (Caption): `Nep Path`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"src/NepTrainKit/Config/nep89.txt"`
- 含义 (Meaning): 特征模型路径 (NEP model path)。
- 对输出规模/物理性的影响: 用于距离特征计算，路径失效会导致过滤退化。
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


## 常见问题与排查
- 先选中非物理结构：检查是否跳过了 `nep89` 清洗步骤。
- 样本保留过少：降低最小距离阈值。
- 去重不明显：适度提高阈值并确认特征有效。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片本身不新增专用 Config_type 标签。


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
