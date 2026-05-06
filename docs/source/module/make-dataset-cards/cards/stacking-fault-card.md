<!-- card-schema: {"card_name": "Stacking Fault", "source_file": "src/NepTrainKit/ui/views/_card/stacking_fault_card.py", "serialized_keys": ["params", "hkl", "step", "layers"]} -->

# 层错构型（Stacking Fault）

`Group`: `Defect`  
`Class`: `StackingFaultCard`  
`Source`: `src/NepTrainKit/ui/views/_card/stacking_fault_card.py`

## 功能说明
沿指定晶面与滑移步长生成层错样本（stacking fault），补充位错相关局域构型。

它最适合的场景是：沿特定滑移面构造层错，补充位移错配附近的结构样本。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

## 操作示例
### 场景：沿特定滑移面构造层错，补充位移错配附近的结构样本

**输入：** 一个适合沿某个 hkl 面滑移的超胞结构

**目标：** 系统扫描层错位移，而不是靠随机扰动碰运气得到错层构型

**参数设置：**
- `hkl` 先选滑移面
- `layers` 控制从哪一层开始做相对位移
- `step` 先用细步长扫过位移路径

**输出：** 一批层间发生相对滑移的结构，适合层错能面或缺陷训练

**怎么验证结果合理：**
- 检查位移方向与目标滑移面一致
- 确认层错不是整体刚性平移
- 若结构重叠严重，先减小位移步长

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 层错能或滑移路径预测误差大。
- 目标任务 (Target objective): 覆盖层错与滑移位移通道。
- 建议添加条件 (Add-it trigger): 缺陷力学/塑性相关任务。
- 不建议添加条件 (Avoid trigger): 与位错缺陷无关任务。
> 物理提示 (Physics caution): 重点检查缺陷附近的局部配位和是否形成孤立原子或明显断裂。

## 输入前提
- 先选低指数 `hkl` 平面验证。
- 先小 `step` 试跑再扩展。

## 参数说明（完整）
### `params` (Operation Params)
- UI Label: `Operation Params`
- 字段映射 (Field mapping): 序列化键 `params` <-> 核心操作参数 `StackingFaultParams`。
- 控件标签 (Caption): `Operation Params`。
- 控件解释 (Widget): 由界面控件自动汇总，不需要手动编辑。
- 类型/范围 (Type/Range): object
- 默认值 (Default): `{"hkl": [1, 1, 1], "step": [0.0, 1.0, 0.5], "layers": 1}`
- 含义 (Meaning): UI 解耦后的核心参数快照，用于 CLI/批处理复用。
- 对输出规模/物理性的影响: 与展开后的 `hkl/step/layers` 字段一致。
- 配置建议 (Practical note): 新版本优先读取 `params`，旧字段仍保留用于兼容已有 workflow。

### `hkl` (h k l)
- UI Label: `h k l`
- 字段映射 (Field mapping): 序列化键 `hkl` <-> 界面标签 `h k l`。
- 控件标签 (Caption): `h k l`。
- 控件解释 (Widget): 区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。
- 类型/范围 (Type/Range): list[3], integers
- 默认值 (Default): `[1, 1, 1]`
- 含义 (Meaning): Miller 指数 (hkl plane)。
- 对输出规模/物理性的影响: 定义层错/滑移操作的晶面。
- 物理直觉 / 典型值: 先从小范围试跑并抽查输出，再决定是否扩大范围；范围越宽，覆盖越广，但极端构型风险也越高。
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
- 物理直觉 / 典型值: 它通常是控制变化幅度的主旋钮；先从能看清趋势的小幅度起步，再决定是否扩到探索档。
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
- 物理直觉 / 典型值: 这类参数主要控制方向、分层或周期；先用最容易人工检查的简单方向和短范围做验证。
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
- 缺陷强度上升前，通常先用 `Super Cell` 扩大母胞，避免小胞里缺陷相互作用过强。
- 缺陷生成后建议抽查最短键长、局部配位和是否出现明显断裂。

## 常见问题与排查
- 输出为空或结构数明显偏少时，先检查规则是否命中、浓度/数量是否过严，或输入超胞是否太小。
- 若输出结构不合理，优先检查最短键长、局部配位和是否出现整块骨架塌缩，再降低缺陷强度。
- 参数越界时通常受 UI 范围限制；但“过激而仍在范围内”的配置不会被自动裁剪，程序会继续按当前设置生成结果。

## 输出标签 / 元数据变更
- 该卡片输出的 Config_type 标签模式：
  - `SF(hkl={...}{...}{...},d={...})`

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
