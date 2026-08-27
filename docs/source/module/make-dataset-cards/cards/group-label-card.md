<!-- card-schema: {"card_name": "Layer Groups", "source_file": "src/NepTrainKit/ui/views/_card/group_label_card.py", "serialized_keys": ["params"]} -->

# 原子层分组（Layer Groups）

**分类：** 结构处理

## 功能说明

沿所选 `(hkl)` 晶面法向识别原子层，再按层序写入 `A、B、A、B…` 两组标签。输出保存在 `atoms.arrays["group"]`，可交给磁序、随机掺杂或空位卡选择对应原子。

卡片输出一个结构，坐标、元素、晶胞和周期边界保持不变。

## 原理与公式

设晶胞矩阵为 $\mathbf A$，原子的分数坐标为 $\mathbf s_i$，Miller 指数为
$\mathbf h=(h,k,l)^\mathsf T$。对应的倒空间法向为

$$
\mathbf g=\mathbf A^{-1}\mathbf h,
\qquad
\hat{\mathbf n}=\frac{\mathbf g}{\lVert\mathbf g\rVert}.
$$

原子沿法向的投影距离为

$$
u_i=\mathbf r_i\cdot\hat{\mathbf n}
   =\frac{\mathbf s_i\cdot\mathbf h}{\lVert\mathbf g\rVert}.
$$

若该法向包含周期方向，程序先对相位 $\mathbf s_i\cdot\mathbf h$ 取模 1，并用周期最小距离合并跨晶胞边界的同一层。例如相位 0.005 和 0.995 在容差允许时属于同一层。

程序再按 $u_i$ 从小到大排列原子；投影区间宽度不超过 `layer_tolerance` 的原子归入同一层。检测到的层依次编号为 $0,1,2,\ldots$，最终标签为

$$
\mathrm{group}(i)=
\begin{cases}
\text{group\_a}, & \ell_i\bmod 2=0,\\
\text{group\_b}, & \ell_i\bmod 2=1.
\end{cases}
$$

因此非正交晶胞也使用真实的倒空间法向，而不是把 `(100)` 简化理解为笛卡尔 X 方向。

## 操作示例

某结构沿 `(100)` 法向检测到四层：

| 层序号 | 原子数 | 标签 |
|---:|---:|---|
| 0 | 2 | A |
| 1 | 3 | B |
| 2 | 2 | A |
| 3 | 3 | B |

扩胞后如果得到八层，标签继续为 `A → B → A → B → A → B → A → B`，不会变成前半全 A、后半全 B。卡片预览会直接显示这种“标签（层内原子数）”序列。

## 参数说明

### 晶面指数（miller_index）

`str`，默认 `111`。可选 `100`、`010`、`001`、`110`、`111`，分别表示对应的 `(hkl)` 晶面族。

### 层容差（layer_tolerance）

`float`，默认 `0.05 Å`。两个原子沿晶面法向的投影距离相差不超过该值时，视为同一层。

- 热弛豫后同层原子略有起伏：可适当增大。
- 相邻层被误合并：减小该值。

### A 组标签（group_a）

`str`，默认 `A`。写入第 0、2、4…层，不能为空。

### B 组标签（group_b）

`str`，默认 `B`。写入第 1、3、5…层，不能为空且不能与 `group_a` 相同。

### 覆盖已有分组（overwrite）

`bool`，默认 `false`。输入已有 `group` 时，关闭会原样保留已有标签；开启才按当前原子层重新分组。

## 使用步骤

1. 选择希望分层的 `(hkl)` 晶面。
2. 查看首个输入的层序预览，例如 `A(2) → B(3) → A(2) → B(3)`。
3. 如果只检测到一层，先扩胞、换晶面或减小层容差。
4. 周期 AFM 使用时优先保证一个周期内的层数为偶数；奇数层会在周期边界出现同组相接，界面会给出提示。
5. 运行后再把结果接入 `Magnetic Order`、`Random Doping` 或 `Targeted Vacancy`。

典型工作流：

`Super Cell → Layer Groups → Magnetic Order`

## 输出

新生成或覆盖分组时，`Config_type` 追加类似：

```text
Grp(hkl111,tol=0.05,A/B)
```

旧版卡片的“晶胞相位二分”和“半网格奇偶”算法已经移除。载入含 `mode` 或 `kvec` 的旧配置时，程序会保留可映射的晶面和标签，同时显示迁移提示；重新运行前应核对新的逐层预览。

## 常见问题

**只检测到一层。** 当前晶面方向上没有足够的独立原子层，或层容差过大。扩胞、更换晶面或减小容差后再看预览。

**层数是奇数。** 分组仍可用于选择原子；若用于周期 AFM，边界两侧会出现同组相接，宜先构造偶数层超胞。

**修改参数后输出仍是旧标签。** 输入已有 `group` 且“覆盖已有分组”处于关闭状态。确认需要重写后再开启。
