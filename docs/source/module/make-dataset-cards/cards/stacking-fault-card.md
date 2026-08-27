:orphan:

<!-- card-schema: {"card_name": "Stacking Fault", "source_file": "src/NepTrainKit/ui/views/_card/stacking_fault_card.py", "serialized_keys": ["params"]} -->

# 旧版层错位移（Legacy Stacking Fault）

**分类：** 缺陷

## 兼容性说明

这张卡保留早期 NepTrainKit 的固定晶胞半区平移算法，用于载入和复现已有工作流。它的“切面一侧整体滑移”思路可以生成层错，也与常见 Atomsk 脚本采用同一种几何操作；问题在于卡片没有让用户显式指定滑移方向，切面参数也容易误解，因此不再出现在“添加新卡片”和“查找卡片”中。

旧 JSON 中的 `StackingFaultCard` 仍可正常载入和运行。新任务请使用 `GSFE Path`（序列化类名仍为 `StrictGSFEPathCard`）。后者显式要求：

- 当前晶胞的 `ab` 层错面；
- 当前晶胞基矢下的面内方向；
- 位移单位；
- 切面位置；
- 输入晶胞与目标层错面的定向一致性。

旧版卡没有显式滑移方向，因此仅凭 `近似晶面 h k l`（`hkl`）无法定义一个物理滑移系。

## 旧算法实际做什么

给定输入晶胞 $A$ 和 `hkl=(h,k,l)`，旧算法先用倒易晶格构造单位法向：

$$
\hat{\mathbf n}
=
\frac{h\mathbf b_1+k\mathbf b_2+l\mathbf b_3}
{\left|h\mathbf b_1+k\mathbf b_2+l\mathbf b_3\right|}
$$

随后从全局笛卡尔 x、y、z 轴中选择与 $\hat{\mathbf n}$ 最不平行的一根轴 $\mathbf e$，并定义：

$$
\hat{\mathbf s}
=
\frac{\hat{\mathbf n}\times\mathbf e}
{\left|\hat{\mathbf n}\times\mathbf e\right|}
$$

这个 $\hat{\mathbf s}$ 是确定性的面内方向，但不是用户指定的晶向。只要它恰好等于目标滑移方向，旧算法生成的结构可以是有效层错；否则结果只是另一个方向上的层间错排。把同一物理晶体整体旋转后，所选方向还可能改变。

原子按投影 $q_i=\mathbf r_i\cdot\hat{\mathbf n}$ 排序。`旧版投影层序号`（`layers`）选择第几个投影坐标作为阈值，并移动满足 $q_i\ge q_\mathrm{cut}$ 的原子。若 `旧版投影层序号`（`layers`）超出投影层数，算法静默改用中间投影层。最后沿 $\hat{\mathbf s}$ 施加绝对 Å 位移并把坐标折回周期晶胞。

因此旧卡的核心位移并非错误，但参数面不足以可靠表达任意滑移系：

- 默认 `layers=1` 可能选中全部原子，只产生整体平移，没有形成相对层移；
- “层”由投影坐标四舍五入到 8 位后去重，不是带容差的晶体学层识别；
- 超范围层号不会报错，而会回退到中间层；
- `hkl=(0,0,0)` 会原样返回输入结构；
- 输出是否对应目标层错取决于自动方向是否恰好等于目标晶向；
- 卡片不检查输入晶胞是否已按目标晶面定向。

## 参数说明

### 近似晶面 h k l（hkl）

长度为 3 的整数序列，默认 `(1,1,1)`。只用于构造投影法向，不足以定义物理滑移系。

### 旧版投影层序号（layers）

`int`，默认 1，必须至少为 1。选择排序后的第 `旧版投影层序号`（`layers`）个投影坐标作为移动阈值；并不是“参与层错的层数”。

### 位移（step）

三个浮点数 `[start, stop, step]`，默认 `[0.0, 1.0, 0.5]`，单位 Å。每个值都是沿旧算法自动选择的 $\hat{\mathbf s}$ 的绝对位移。

## 兼容配置示例

```json
{
  "class": "StackingFaultCard",
  "check_state": true,
  "params": {
    "hkl": [1, 1, 1],
    "step": [0.0, 1.0, 0.5],
    "layers": 2
  }
}
```

该示例仅用于重现旧工作流。新任务应使用显式卡片，把晶面、滑移方向和切面分别写清楚。

## 如何迁移到显式 GSFE

旧配置不能自动无损转换，因为其中没有记录滑移方向。迁移时需要根据材料和目标滑移系人工确定 `slip_uvw`，并重新检查切面。

示意配置：

```json
{
  "class": "StrictGSFEPathCard",
  "check_state": true,
  "params": {
    "plane_hkl": [0, 0, 1],
    "slip_uvw": [1, 0, 0],
    "displacement_range": [0.0, 1.0, 0.1],
    "displacement_unit": "fraction_of_vector",
    "cut_mode": "middle",
    "cut_fraction": 0.5,
    "layer_index": 0,
    "wrap": true
  }
}
```

这里的指数只是格式示例，不能直接照搬到具体材料。

## 输出标签

```text
SF(hkl={h}{k}{l},d={位移量})
```

标签表示旧算法参数，不证明输出是物理有效的层错或孪晶。

## 可复现性

无随机性。同一坐标系中的同一输入和参数会得到一致输出；把结构旋转到另一笛卡尔取向后，结果不保证等价。
