<!-- card-schema: {"card_name": "Stacking Fault", "source_file": "src/NepTrainKit/ui/views/_card/stacking_fault_card.py", "serialized_keys": ["hkl", "step", "layers"]} -->

# 层错构型（Stacking Fault）

`Group`: `Defect`  
`Class`: `StackingFaultCard`  
`Source`: `src/NepTrainKit/ui/views/_card/stacking_fault_card.py`

## 功能说明
沿指定晶面与滑移步长生成层错样本（stacking fault），补充位错相关局域构型。

### 快速上手
最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。

:::{tip}
高通量示例：建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，再执行最远点采样（FPS）。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 层错能或滑移路径预测误差大。
- 目标任务 (Target objective): 覆盖层错与滑移位移通道。
- 建议添加条件 (Add-it trigger): 缺陷力学/塑性相关任务。
- 不建议添加条件 (Avoid trigger): 与位错缺陷无关任务。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 先选低指数 `hkl` 平面验证。
- 先小 `step` 试跑再扩展。


## 参数说明（完整）
### `hkl` (h k l)
- UI Label: `h k l`
- 字段映射 (Field mapping): 序列化键 `hkl` <-> 界面标签 `h k l`。
- 控件标签 (Caption): `h k l`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], integers
- 默认值 (Default): `[1, 1, 1]`
- 含义 (Meaning): Miller 指数 (hkl plane)。
- 对输出规模/物理性的影响: 定义层错/滑移操作的晶面。
- 推荐范围 (Recommended range):
  - 保守：低指数（1-2）
  - 平衡：中指数（2-4）
  - 探索：高指数（4-6）

### `step` (Step)
- UI Label: `Step`
- 字段映射 (Field mapping): 序列化键 `step` <-> 界面标签 `Step`。
- 控件标签 (Caption): `Step`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], displacement `[start,end,step]`
- 默认值 (Default): `[0.0, 1.0, 0.5]`
- 含义 (Meaning): 步长区间 (step range)。
- 对输出规模/物理性的影响: 主控扫描位移幅度与分辨率。
- 推荐范围 (Recommended range):
  - 保守：0 到 1，step 0.5
  - 平衡：0 到 1，step 0.25
  - 探索：0 到 1，step 1

### `layers` (Layers)
- UI Label: `Layers`
- 字段映射 (Field mapping): 序列化键 `layers` <-> 界面标签 `Layers`。
- 控件标签 (Caption): `Layers`。
- 控件解释 (Widget): 数值输入 `SpinBoxUnitInputFrame`。
- 类型/范围 (Type/Range): int（单值输入）
- 默认值 (Default): `[1]`
- 含义 (Meaning): 层参数 (layer index/count)。
- 对输出规模/物理性的影响: 控制操作层位或层数覆盖。
- 推荐范围 (Recommended range):
  - 保守：1-1
  - 平衡：1-2
  - 探索：2-5


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "StackingFaultCard",
  "check_state": true,
  "hkl": [
    1,
    1,
    1
  ],
  "step": [
    0.0,
    0.3,
    0.1
  ],
  "layers": [
    1
  ]
}
```

### 平衡（Balanced）
```json
{
  "class": "StackingFaultCard",
  "check_state": true,
  "hkl": [
    1,
    1,
    1
  ],
  "step": [
    0.0,
    0.6,
    0.2
  ],
  "layers": [
    1
  ]
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "StackingFaultCard",
  "check_state": true,
  "hkl": [
    1,
    1,
    1
  ],
  "step": [
    0.0,
    1.0,
    0.25
  ],
  "layers": [
    2
  ]
}
```


## 推荐组合
- Stacking Fault -> Atomic Perturb: 在层错构型周围补充局部扰动。
- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。


## 常见问题与排查
- 滑移方向异常：检查 `hkl` 与晶胞取向。
- 结构突变过大：降低 `step` 上限。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `SF(hkl={...}{...}{...},d={...})`


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
