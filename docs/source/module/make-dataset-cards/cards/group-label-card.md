<!-- card-schema: {"card_name": "Group Label", "source_file": "src/NepTrainKit/ui/views/_card/group_label_card.py", "serialized_keys": ["params"]} -->

# 分组标记（Group Label）

`Group`: `Alloy` | `Class`: `GroupLabelCard`

## 功能说明

为结构中每个原子写入 `atoms.arrays['group']` 标签（A 或 B），按 k-vector 层分组或分数坐标奇偶分组。这些标签供下游 `Magnetic Order`（AFM group A/B 模式）、`Random Vacancy`、`Random Doping` 的 group 约束使用。

**这是一张元数据卡，不改坐标也不改元素，只写 group 标签。**

$$g_{k\text{-vec}}=\left\lfloor 2(\mathbf{s}\cdot\mathbf{k})\right\rfloor\bmod 2,\quad g_{parity}=\left(\mathrm{round}(2s_x)+\mathrm{round}(2s_y)+\mathrm{round}(2s_z)\right)\bmod 2$$

## 操作示例

### 场景：k-vector AFM 无法匹配你的晶格，模型 AFM 能量系统性偏高

你在 NiO 上用 `Magnetic Order` 的 k-vector 111 模式生成了 AFM 结构。训练后模型对 AFM 构型的能量预测仍比 DFT 高 0.3 eV/atom——k-vector 的波节面穿过了两种不等价原子位，导致 Ni 和 O 都被混合分配正负号，没有形成"一层 Ni↑、一层 Ni↓"的清晰子晶格翻转。

**诊断思路：** k-vector 是晶体学均匀的翻转模式，适合简单 Bravais 晶格。但实际磁性材料常有不等价子晶格——你需要手动定义哪些原子属于 A 子晶格（正磁矩）、哪些属于 B（负磁矩）。`Group Label` 就是做这件事的：先在结构里写好 group 标签，下游 `Magnetic Order` 再用 group A/B 模式按标签翻转磁矩。

**输入：** 一个 NiO 超胞，Ni 和 O 交替排列在 fcc 子晶格上

**目标：** 按 k-vec 111 分组，奇数层 = A，偶数层 = B

**参数设置：**
- `Mode` = `k-vector layers (recommended)`
- `Kvec` = `111`
- `Group A` = `A`，`Group B` = `B`
- `Overwrite` = 勾选

**输出：** 结构不变，但每个原子现在有 `group` = `A` 或 `B`，带 `Grp(k111,A/B)` 标签

**怎么验证分组正确：**
- 打开输出的 extxyz 文件，检查 `group` 列是否出现 A/B 交替
- 对于 (111) 方向，相邻原子层应交替 A/B
- 用 `Magnetic Order` 的 group A/B 模式接在后面，确认 AFM 输出磁矩有正有负
- 如果分组几乎全是 A，换 kvec = `110` 或 `100` 验证

### 什么时候加这张卡、什么时候不加

**加：**
- 下游有 `Magnetic Order` 的 AFM group A/B 模式
- 下游 `Random Doping` / `Random Vacancy` 需要对特定区域操作（group 约束）
- 需要在结构中定义可复用的子晶格/区域标识

**不加：**
- 全流程不依赖 group 过滤
- 下游磁序只用 k-vector 模式（不需要 group 标签）

## 参数说明


### Mode（mode）

类型：`str`。默认：`'k-vector layers (recommended)'`。选择这张卡的主操作模式。

物理直觉：主模式会改变生成语义；不要只为了多出结构而混用物理含义不同的模式。

| 选项 | 含义 | 什么时候选 |
|------|------|-----------|
| 以 UI 下拉项为准 | 不同选项对应不同物理生成语义 | 选择前先看本页操作示例和推荐预设 |

### Kvec（kvec）

类型：`str`。默认：`'111'`。控制 `kvec` 对应的生成或过滤行为。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

### Group A（group_a）

类型：`str`。默认：`'A'`。指定 A 组原子或 group 标签。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：需要 group pair、手动 group 或 AFM group 模式时。

### Group B（group_b）

类型：`str`。默认：`'B'`。指定 B 组原子或 group 标签。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

生效条件：需要 group pair、手动 group 或 AFM group 模式时。

### Overwrite（overwrite）

类型：`bool`。默认：`True`。决定是否覆盖已有 group 标签。

物理直觉：根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。

## 推荐预设

### k-vector 111，标签 A/B（最常用）
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

### k-vector 110，自定义标签
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

### Fractional parity，保留已有标签
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

## 推荐组合

- `Group Label` → `Magnetic Order`：AFM 的 group A/B 模式需要 group 标签
- `Group Label` → `Random Doping` / `Random Vacancy`：用 group 约束定向掺杂/删除
- `Group Label` → `Random Doping` → `Magnetic Order`：先标记子晶格 → 掺杂特定子晶格 → 初始化对应磁序

## 常见问题

**输出结构没有变化。** 这是正常的——Group Label 只改标签不改坐标。检查输出 extxyz 文件中是否有 `group` 列。

**group 标签全是同一个值。** 结构可能根本不分层。检查结构是否有清晰的层状特征。如果晶胞只有 2 个原子（如 NaCl 单胞），k-vec 分组可能全落在同一类——先扩胞再到有足够多层的时候再跑。

**下游卡片读不到 group。** 确认 `overwrite=true`（如果之前没有 group），确认标签名和下游规则字符串完全一致（区分大小写），确认输入是 extxyz 格式（普通 xyz 不保留 group 数组）。

**EXTXYZ 文件中的 group 列。** 如果从 .xyz 导入并需要保留分组，使用 EXTXYZ 格式：在第二行加 `Properties=species:S:1:pos:R:3:group:S:1`，然后在每行坐标后加 group 值。

## 输出标签

`Grp({分组模式},{A标签}/{B标签})`

写入 `atoms.arrays['group']`，每个原子值为 group_a 或 group_b 字符串。

## 可复现性

无随机性。同参数同输入 → 严格一致输出。k-vec 和 fractional parity 分组均为确定性算法。
