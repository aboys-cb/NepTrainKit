<!-- card-schema: {"card_name": "Group Label", "source_file": "src/NepTrainKit/ui/views/_card/group_label_card.py", "serialized_keys": ["mode", "kvec", "group_a", "group_b", "overwrite"]} -->

# 分组标记（Group Label）

`Group`: `Alloy`  
`Class`: `GroupLabelCard`  
`Source`: `src/NepTrainKit/ui/views/_card/group_label_card.py`

## 功能说明
为结构生成 `group` 标签数组，支撑后续 AFM 分组、局域替换和规则型筛选。

### 快速上手
最小可运行示例：对单帧执行后检查 `atoms.arrays['group']` 是否写入且标签与预期一致。

:::{tip}
高通量示例：先统一 `group_a/group_b` 命名规范，再批量生成并抽样核对下游卡片读取是否一致。
:::

### 关键公式 (Core equations)
$$g_{k\text{-vec}}=\left\lfloor 2(\mathbf{s}\cdot\mathbf{k})\right\rfloor\bmod 2$$
$$g_{parity}=\left(\mathrm{round}(2s_x)+\mathrm{round}(2s_y)+\mathrm{round}(2s_z)\right)\bmod 2$$


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 下游操作需要分组，但当前无 group 标签。
- 目标任务 (Target objective): 建立可复用的子晶格/区域标签语义。
- 建议添加条件 (Add-it trigger): 下游存在 Magnetic Order 或 rules+group 操作。
- 不建议添加条件 (Avoid trigger): 全流程不依赖 group 过滤。
> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。


## 输入前提
- 统一 `group_a/group_b` 命名规范。
- 确认是否允许覆盖已有 group（`overwrite`）。


## 参数说明（完整）
### `mode` (Mode)
- UI Label: `Mode`
- 字段映射 (Field mapping): 序列化键 `mode` <-> 界面标签 `Mode`。
- 控件标签 (Caption): `Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"k-vector layers (recommended)"`
- 含义 (Meaning): 操作模式 (operation mode)。
- 对输出规模/物理性的影响: 改变执行逻辑路径，影响样本分布。
- 配置建议 (Practical note): `Mode` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `kvec` (Kvec)
- UI Label: `Kvec`
- 字段映射 (Field mapping): 序列化键 `kvec` <-> 界面标签 `Kvec`。
- 控件标签 (Caption): `Kvec`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"111"`
- 含义 (Meaning): k 向量规则 (k-vector rule)。
- 对输出规模/物理性的影响: 影响子晶格分组结果和后续磁序构造。
- 配置建议 (Practical note): `Kvec` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。

### `group_a` (Group A)
- UI Label: `Group A`
- 字段映射 (Field mapping): 序列化键 `group_a` <-> 界面标签 `Group A`。
- 控件标签 (Caption): `Group A`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"A"`
- 含义 (Meaning): A 组标签 (group A label)。
- 对输出规模/物理性的影响: 定义分组命名供下游引用。
- 配置建议 (Practical note): 按项目命名规范填写，需与下游引用保持一致。

### `group_b` (Group B)
- UI Label: `Group B`
- 字段映射 (Field mapping): 序列化键 `group_b` <-> 界面标签 `Group B`。
- 控件标签 (Caption): `Group B`。
- 控件解释 (Widget): 文本输入 `LineEdit`（或可编辑下拉）。
- 类型/范围 (Type/Range): string
- 默认值 (Default): `"B"`
- 含义 (Meaning): B 组标签 (group B label)。
- 对输出规模/物理性的影响: 定义分组命名供下游引用。
- 配置建议 (Practical note): 按项目命名规范填写，需与下游引用保持一致。

### `overwrite` (Overwrite)
- UI Label: `Overwrite`
- 字段映射 (Field mapping): 序列化键 `overwrite` <-> 界面标签 `Overwrite`。
- 控件标签 (Caption): `Overwrite`。
- 控件解释 (Widget): 勾选开关 `CheckBox`。
- 类型/范围 (Type/Range): bool
- 默认值 (Default): `true`
- 含义 (Meaning): 覆盖已有标签 (overwrite existing labels)。
- 对输出规模/物理性的影响: 开启会重写旧 group 信息。
- 配置建议 (Practical note):
  - 开启：确认允许覆盖已有标签时开启。
  - 关闭：保留已有标签时关闭。


## 推荐预设（可直接复制 JSON）
### 基础模板（Baseline）
```json
{
  "class": "GroupLabelCard",
  "check_state": true,
  "mode": "k-vector layers (recommended)",
  "kvec": "111",
  "group_a": "A",
  "group_b": "B",
  "overwrite": true
}
```

### 兼容模板（Compatible）
```json
{
  "class": "GroupLabelCard",
  "check_state": true,
  "mode": "fractional parity (2x rounding)",
  "kvec": "111",
  "group_a": "A",
  "group_b": "B",
  "overwrite": false
}
```

### 自定义模板（Custom）
```json
{
  "class": "GroupLabelCard",
  "check_state": true,
  "mode": "k-vector layers (recommended)",
  "kvec": "110",
  "group_a": "S1",
  "group_b": "S2",
  "overwrite": true
}
```


## 推荐组合
- Group Label -> Magnetic Order: AFM 分组模式需要稳定的 group 标签。
- Group Label -> Random Doping / Random Vacancy: 在选定区域施加规则约束。


## 常见问题与排查
- 分组不符合预期：检查 `kvec` 与晶向定义。
- 下游读不到分组：核对是否被覆盖或未写入数组。
- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。
- 输出分布不符合目标：抽样检查后再回调关键参数。


## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Grp({...},{...}/{...})`
- 创建/覆盖 `atoms.arrays['group']` 标签数组。


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
