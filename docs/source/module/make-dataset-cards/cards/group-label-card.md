<!-- card-schema: {"card_name": "Layer Groups", "source_file": "src/NepTrainKit/ui/views/_card/group_label_card.py", "serialized_keys": ["params"]} -->

# 原子层分组（Layer Groups）

**分类：** 结构

## 功能说明

沿所选 `(hkl)` 晶面法向识别原子层，再把相邻层依次标为 `A、B、A、B…`。
每个输入固定得到一个输出；元素、坐标、晶胞和周期边界不变，只新增或更新逐原子数组
`atoms.arrays["group"]`。

这些标签可供磁序、定向空位或其他读取 `group` 的下游卡片选择原子。它不是结构筛选，
也不会改变元素。`Config_type` 中的 `Grp(...)` 只记录操作来源，不能替代逐原子 `group` 数组。

## 原理与公式

设 ASE 晶胞的三条晶格矢量组成矩阵 $\mathbf A$，Miller 指数为
$\mathbf h=(h,k,l)^\mathsf T$。程序使用倒空间法向

$$
\mathbf g=\mathbf A^{-1}\mathbf h,
\qquad
\hat{\mathbf n}=\frac{\mathbf g}{\lVert\mathbf g\rVert},
$$

并按原子在该法向上的投影距离

$$
u_i=\mathbf r_i\cdot\hat{\mathbf n}
   =\frac{\mathbf s_i\cdot\mathbf h}{\lVert\mathbf g\rVert}
$$

排序。这里 $\mathbf r_i$ 和 $\mathbf s_i$ 分别是笛卡尔坐标和分数坐标。因此 `(100)`
表示晶面指数，不等同于笛卡尔 X 方向；非正交晶胞仍使用真实倒空间法向。

投影区间宽度不超过 `layer_tolerance` 的原子归入同一层。若法向包含周期方向，程序会
按周期相位处理，并把晶胞两端在容差内的同一层合并。第 $\ell_i$ 层的标签为

$$
\operatorname{group}(i)=
\begin{cases}
\text{group\_a}, & \ell_i\bmod 2=0,\\
\text{group\_b}, & \ell_i\bmod 2=1.
\end{cases}
$$

输出数量始终为

$$
N_{\mathrm{out}}=N_{\mathrm{in}}.
$$

## 操作示例

例如 `(100)` 方向检测到四层、各有两个原子时，预览显示
`A(2) → B(2) → A(2) → B(2)`。运行后八个原子的 `group` 数组依次为
`A, A, B, B, A, A, B, B`，结构本身保持不变。

## 参数说明

### 晶面指数（miller_index）

`str`，默认 `111`。可选 `100`、`010`、`001`、`110`、`111`。选择的是晶面法向，
不是笛卡尔坐标轴。

### 层容差（layer_tolerance）

`float`，默认 `0.05 Å`。同层原子因热扰动产生轻微起伏时可适当增大；相邻层被误合并时
应减小。容差必须为正数。

### A 组标签（group_a）

`str`，默认 `A`。写入第 0、2、4…层，不能为空。

### B 组标签（group_b）

`str`，默认 `B`。写入第 1、3、5…层，不能为空且不能与 `group_a` 相同。

### 覆盖已有分组（overwrite）

`bool`，默认关闭。输入已经包含 `group` 时，关闭会原样保留现有标签；开启才按当前晶面
和容差重新分组。界面会显示首个输入的现有计数或新的层序预览。

## 使用与检查

1. 选择晶面并查看首个输入的层序与各组原子数。
2. 若只检测到一层，扩胞、换晶面或减小容差；卡片不会把单层结果伪装成有效 A/B 分组。
3. 若周期方向上是奇数层，普通原子选择仍可使用；周期 A/B 磁序会在边界出现同组相接。
4. 运行后确认输出文件仍包含逐原子 `group` 数组。EXTXYZ 可以保存字符串分组，但转换到
   其他格式时需确认该格式不会丢弃自定义逐原子数组。

典型磁序流程：

```text
Super Cell → Layer Groups → Magnetic Order
```

新建或覆盖标签时，`Config_type` 会追加类似：

```text
Grp(hkl111,tol=0.05,A/B)
```

旧版卡片曾使用晶胞相位二分或半网格奇偶规则。载入含 `mode` 或 `kvec` 的旧配置时，
程序会映射可保留的参数并显示迁移提示；重新运行前应核对新的真实层序预览。

## 常见问题

**为什么运行前提示只有一层？** 当前晶面方向上没有足够的独立原子层，或容差过大。
按界面建议扩胞、换晶面或减小容差。

**为什么参数改变后仍保留旧标签？** 输入已有 `group`，且“覆盖已有分组”处于关闭状态。
