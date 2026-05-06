<!-- card-schema: {"card_name": "Group Label", "source_file": "src/NepTrainKit/ui/views/_card/group_label_card.py", "serialized_keys": ["params", "mode", "kvec", "group_a", "group_b", "overwrite"]} -->

# 分组标记（Group Label）

`Group`: `Alloy`  
`Class`: `GroupLabelCard`  
`Source`: `src/NepTrainKit/ui/views/_card/group_label_card.py`

## 功能说明
为结构生成 `group` 标签数组，支撑后续 AFM 分组、局域替换和规则型筛选。

它最适合的场景是：先把结构划分成 A/B 或层状 group，供后续 AFM、成对 canting 或定向替换使用。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

### 关键公式 (Core equations)
$$g_{k\text{-vec}}=\left\lfloor 2(\mathbf{s}\cdot\mathbf{k})\right\rfloor\bmod 2$$
$$g_{parity}=\left(\mathrm{round}(2s_x)+\mathrm{round}(2s_y)+\mathrm{round}(2s_z)\right)\bmod 2$$

## 操作示例
### 场景：先把结构划分成 A/B 或层状 group，供后续 AFM、成对 canting 或定向替换使用

**输入：** 一个具有明显层状、子晶格或几何分区特征的结构

**目标：** 生成稳定可复用的 `group` 数组，而不是在每张下游卡片里重复推断分组

**参数设置：**
- `mode` 先决定按 k-vector、空间位置还是其他方式分组
- `group_a` / `group_b` 保持简单稳定，便于下游直接引用
- `overwrite=true` 仅在你确定要覆盖旧分组时开启

**输出：** 原结构会新增 `group` 数组；下游磁性或缺陷卡可直接读取这些标签

**怎么验证结果合理：**
- 检查 `group` 数组是否真的写入输出结构
- 抽查 A/B 分组是否符合几何直觉
- 如果分组异常，先回调 `mode` 或 `kvec`

**EXTXYZ 示例：** 如果你想在导入的 `.xyz`/`.extxyz` 里直接带入 `group`，请使用 EXTXYZ 风格的属性列；普通三列 XYZ 不能保存 `group` 这类自定义数组。

```text
4
Lattice="5.43 0 0 0 5.43 0 0 0 5.43" Properties=species:S:1:pos:R:3:group:S:1 pbc="T T T"
Si 0.000 0.000 0.000 A
Si 1.357 1.357 1.357 B
Si 2.715 2.715 0.000 A
Si 4.072 4.072 1.357 B
```

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 下游操作需要分组，但当前无 group 标签。
- 目标任务 (Target objective): 建立可复用的子晶格/区域标签语义。
- 建议添加条件 (Add-it trigger): 下游存在 Magnetic Order 或 rules+group 操作。
- 不建议添加条件 (Avoid trigger): 全流程不依赖 group 过滤。
> 物理提示 (Physics caution): 重点检查目标配比、实际元素统计和标签是否一致，避免“标签写对了、占位落错了”。

## 输入前提
- 统一 `group_a/group_b` 命名规范。
- 确认是否允许覆盖已有 group（`overwrite`）。
- 如果输入来自 `.xyz` 文件并且你想保留现成分组，请使用 EXTXYZ 风格的 `Properties=...:group:S:1` 列；普通三列 XYZ 只会保留元素和坐标。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 核心操作参数 `GroupLabelParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"mode": "k-vector layers (recommended)", "kvec": "111", "group_a": "A", "group_b": "B", "overwrite": true}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的分组模式、k-vector 和标签名字段一致。
- 配置建议 (Practical note): 新版本优先读取 `params`，旧字段仍保留用于兼容已有 workflow。

### `mode` (Mode)
- UI Label: `Mode`
- 字段映射 (Field mapping): 序列化键 `mode` <-> 界面标签 `Mode`。
- 控件标签 (Caption): `Mode`。
- 控件解释 (Widget): 下拉选择 `ComboBox`（显示文本与序列化值可能不同）。
- 类型/范围 (Type/Range): enum(string)
- 默认值 (Default): `"k-vector layers (recommended)"`
- 含义 (Meaning): 操作模式 (operation mode)。
- 对输出规模/物理性的影响: 改变执行逻辑路径，影响样本分布。
- 参数联动 / 生效条件: 它决定当前工作流走哪条主分支；先选模式，再填写与该模式对应的字段。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 它决定程序走哪种离散策略；先选对模式，再去调该模式下真正起作用的数值参数。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
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
- 怎么判断该开还是该关: 只有当你明确知道这个开关会改变当前工作流目标时才开启；不确定时先保持默认并用小样本验证。
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
- 先明确“目标配比”还是“具体落位”，再决定接 `Composition Sweep`、`Random Occupancy` 还是 `Random Doping`。

## 常见问题与排查
- 结果没有变化时，先检查是否原本就带有 `group`，并且 `overwrite=false` 让这张卡按原样返回了输入结构。
- 如果分组看起来几乎全落在同一类，优先检查 `mode` / `kvec` 是否适合当前晶格；没有明显层状或子晶格特征时，某些模式本来就不会给出“好看”的 A/B 分组。
- 如果下游 `Random Doping`、`Random Vacancy` 或 `Magnetic Order` 没有命中预期分组，先确认导入后的结构里真的存在 `atoms.arrays['group']`，并且组名和规则字符串完全一致。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `Grp({...},{...}/{...})`
- 创建/覆盖 `atoms.arrays['group']` 标签数组。
- 下游 `Random Doping`、`Random Vacancy`、`Magnetic Order` 等卡片会直接读取该数组中的组名。

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
