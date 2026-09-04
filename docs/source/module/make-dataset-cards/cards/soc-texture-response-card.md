<!-- card-schema: {"card_name": "SOC / Texture Response", "source_file": "src/NepTrainKit/ui/views/_card/soc_texture_response_card.py", "serialized_keys": ["params"]} -->

# SOC / 纹理响应（SOC / Texture Response）

**分类：** 磁性

## 这张卡做什么

这张卡为含 SOC 的能量或磁力计算准备两类有序路径：

- **全局各向异性：** 整体旋转输入自旋纹理，保持所有相对自旋夹角不变。
- **有限 q 纹理：** 保留每个原子的磁矩模长，按 Bloch、cycloidal 或一般旋转平面重新生成全部方向。

有限 q 模式中的 `q=0` 是生成的共线参考帧，不是输入自旋方向的副本。

## 原理与公式

### 全局刚性旋转

所有磁矩绕同一笛卡尔轴旋转：

$$\mathbf S_i(\theta)=R_{\hat{\mathbf n}}(\theta)\mathbf S_i.$$

因此 $\mathbf S_i\cdot\mathbf S_j$、磁矩模长、原子坐标和晶胞都保持不变。启用时间反演后，再生成一组 $-\mathbf S_i(\theta)$。

### 有限 q 纹理

对原子 $i$，令

$$
\phi_i=\mathbf q\cdot\mathbf r_i+\phi_0,
$$

$$
\mathbf S_i=M_i\left[\sqrt{1-c^2}\left(
\cos\phi_i\,\mathbf e_1+\sin\phi_i\,\mathbf e_2\right)
+c\,\hat{\mathbf n}\right].
$$

$M_i$ 来自输入磁矩模长，$c=m_\parallel/|m|$ 是沿旋转平面法向 $\hat{\mathbf n}$ 的归一化分量。$c=0$ 为平面螺旋，$|c|=1$ 时旋转平面分量消失。

三种几何的区别：

- **Bulk / Bloch：** 自动令旋转平面法向平行于 q。
- **Interfacial / Cycloidal：** 自旋在 q 与表面法向张成的平面内旋转。
- **General spiral：** 用户直接指定旋转平面法向。

有符号扫描使用 $\mathbf q(s)=s\mathbf q_0$；正负分支对应相反手性。

## q 与周期晶胞

新建卡片默认使用晶胞倒空间整数索引 $(h,k,l)$：

$$
\mathbf q_0=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3,
\qquad \mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}.
$$

因此 q 会随输入晶胞自动换算，并在周期晶格矢量上严格闭合。默认 `(1,0,0)` 表示当前晶胞的第一根倒格矢，不等于笛卡尔 x 或 Miller 晶面法向；它也与默认界面法向 `(0,0,1)` 正交，所以三种有限 q 预设都能直接运行。

自定义 Cartesian q 时，闭合条件为

$$\frac{\mathbf q\cdot\mathbf a_i}{2\pi}\in\mathbb Z$$

对每个周期晶格矢量都成立。若不成立，应同时调整 q 和超胞，而不是关闭检查后生成带周期接缝的纹理。

## 输出数量怎么算

设扫描含 $N$ 个坐标：

- 全局各向异性：生成 $N$ 帧；启用时间反演后生成两个完整组，共 $2N$ 帧。
- 任一有限 q 模式：生成一个完整有符号 q 组，共 $N$ 帧。

“最大结构数”必须容纳全部请求组，不会静默省略时间反演组或截断路径。

## 操作示例

若模型在 SOC 计算中不能稳定区分相反手性的能量响应，可先构造包含目标传播方向的超胞，选择 `Bulk / Bloch`，使用默认晶胞倒空间 `(1,0,0)` 和 `-2,-1,0,1,2` 扫描。输出五帧分别对应 $-2q_0,-q_0,0,q_0,2q_0$；回填 DFT 标签后按 `response_group` 比较正负 q 分支。

## 参数说明

### 响应类型（response_kind）

默认 `Global anisotropy`。可选 `Global anisotropy`、`Bulk / Bloch`、`Interfacial / Cycloidal`、`General spiral`。

### 扫描坐标（coordinate_scan）

默认 `-2,-1,0,1,2`。全局模式中单位为度；有限 q 模式中表示基准 q 的有符号倍数。至少三个不同值并包含 0。

### 全局旋转轴（rotation_axis）

默认 `[0,1,0]`，为笛卡尔向量，仅用于全局各向异性。

### q 定义方式（q_definition）

新建卡片默认 `Cell reciprocal vector`；也可选 `Cartesian vector`。旧工作流缺少该字段时按原 Cartesian q 恢复。

### 晶胞倒空间索引（q_reciprocal_index）

默认 `(1,0,0)`。三个整数共同定义基准 q，不能全为 0；仅在晶胞倒空间模式下生效。

### Cartesian q（q_vector_cart）

默认旧值 `[0,0,0.1]` Å⁻¹，仅在 Cartesian 模式下生效。界面将其拆成笛卡尔方向和模长。

### 旋转平面法向（plane_normal）

默认 `[0,1,0]`，为笛卡尔向量，仅用于 `General spiral`。

### 表面法向（surface_normal）

默认 `[0,0,1]`，为笛卡尔向量，仅用于 `Interfacial / Cycloidal`；不能与 q 平行。

### 法向磁矩分量（cone_component）

默认 `0`，范围 `[-1,1]`，表示 $m_\parallel/|m|$。它改变锥形开口，但保持每个原子的磁矩模长。

### 初始相位（phase_deg）

默认 `0` 度。作为 $\phi_0$ 加到当前笛卡尔位置计算得到的相位上。

### 时间反演对照（include_time_reversal）

默认关闭，仅用于全局各向异性。开启后增加一组逐帧取负的自旋纹理。

### 要求周期闭合（require_commensurate）

默认开启，仅用于有限 q 模式。Cartesian q 不满足闭合条件时明确失败；晶胞倒空间整数索引天然满足该条件。

### 最大结构数（max_outputs）

默认 `100`。必须足以容纳当前完整路径；不足时运行前明确报错。

## 常见问题

- **Cartesian q 无法闭合：** 切换到晶胞倒空间模式，或同时调整 q 和超胞。
- **Cycloidal 模式提示法向平行：** 表面法向必须与 q 张成一个平面。
- **q=0 与输入自旋不同：** 有限 q 模式重新生成方向，q=0 是同一生成路径中的共线参考帧。

## 输出字段

输出磁矩写入 `spin:R:3`，并记录可辨认响应类型的 `response_group`、有符号 `response_coordinate`、`response_coordinate_unit`、`response_branch` 和任务来源。有限 q 的坐标单位明确记录为 `1/angstrom`；manifest 还保存 Cartesian q、晶胞倒空间坐标、周期、手性、平面法向、相位和法向分量。有限 q 使用 `response_probe=chirality`，目前用于响应分析或普通 E/MF 训练，不会激活只面向旋转路径的分组响应损失。
