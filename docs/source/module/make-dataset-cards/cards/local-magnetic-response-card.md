<!-- card-schema: {"card_name": "Local Magnetic Response", "source_file": "src/NepTrainKit/ui/views/_card/local_magnetic_response_card.py", "serialized_keys": ["params"]} -->

# 局域磁响应（Local Magnetic Response）

**分类：** 磁性

## 这张卡做什么

输入一帧带矢量磁矩的结构，围绕指定原子、原子对或 group 生成一条受控响应路径。每个目标对应一个完整 group，group 内含零扰动参考帧，适合比较局域磁矩变化前后的能量和磁力响应。

它与“自旋扰动”的区别是：这里的坐标有顺序、正负分支成对且每组自带参考帧；自旋扰动用于随机扩充构型。

## 原理与公式

设输入磁矩为 $\mathbf S_i$，笛卡尔旋转轴的单位向量为 $\hat{\mathbf n}$。旋转使用 Rodrigues 公式：

$$
R_{\hat{\mathbf n}}(\theta)\mathbf S
=\mathbf S\cos\theta
+(\hat{\mathbf n}\times\mathbf S)\sin\theta
+\hat{\mathbf n}(\hat{\mathbf n}\cdot\mathbf S)(1-\cos\theta).
$$

- **单原子倾斜：** 选中磁矩变为 $R_{\hat{\mathbf n}}(\theta)\mathbf S_i$。
- **原子对倾斜：** 左原子旋转 $+\theta/2$，右原子旋转 $-\theta/2$；$\theta$ 是两侧的总相对转角。
- **分组对倾斜：** 对两个 `group` 中的全部非零磁矩执行相同的 $+\theta/2$ 与 $-\theta/2$ 旋转。
- **磁矩模长：** 方向不变，选中磁矩变为 $\mathbf S_i(s)=s\mathbf S_i$。元数据中的响应坐标记录为 $s-1$，因此参考帧仍位于 0。

旋转模式的界面角度单位为度，输出 `response_coordinate` 使用弧度。

## 输出数量怎么算

每个目标、原子对或分组对各生成一个完整 group：

$$N_{\mathrm{out}}=N_{\mathrm{group}}\times N_{\mathrm{coordinate}}.$$

例如五点扫描 `-2,-1,0,1,2`：

- 首个合格原子：$1\times5=5$ 帧；
- 显式选择原子 `1,3`：$2\times5=10$ 帧；
- 自动找到 4 对近邻：$4\times5=20$ 帧；
- group A/B：$1\times5=5$ 帧。

“最大结构数”只在 group 之间截断，不会留下不完整的扫描。如果上限小于一个 group 的坐标数，运行会报错。

## 操作示例

1. 确认输入含 `spin:R:3` 或可转换为三分量的初始磁矩。
2. 选择响应类型，再选择目标原子、原子对或 group。
3. 旋转响应使用含 0 的对称角度扫描；模长响应使用含 1.0 的比例扫描。
4. 运行后按 `response_group` 检查每组是否同时含 minus、reference 和 plus 分支。

## 参数说明

### 响应类型（response_kind）

默认 `Atom pair canting`。可选 `Single-spin tilt`、`Atom pair canting`、`Group pair canting`、`Moment magnitude`。

### 旋转角扫描（coordinate_scan_deg）

默认 `-2,-1,0,1,2` 度，仅用于三种旋转响应。至少三个不同坐标并包含 0；输出元数据换算为弧度。

### 目标选择（target_mode）

默认 `First eligible atom`。单原子倾斜和磁矩模长可选择首个合格原子、全部合格原子或显式索引。

### 原子索引（target_indices）

默认空，仅在 `Explicit indices` 下显示。使用 1-based 索引，例如 `1,3-5`。

### 原子对来源（pair_source）

默认 `Manual indices`。可手动配对，也可按近邻壳层自动选取无重复原子对。

### 左侧原子索引（pair_left_indices）

默认 `1`。手动配对时使用 1-based 索引；数量必须与右侧一致。

### 右侧原子索引（pair_right_indices）

默认 `2`。按顺序与左侧索引一一配对。

### 近邻壳层（pair_shell）

默认 `1`，仅用于自动配对。`1` 表示最近邻壳层，`2` 表示第二近邻壳层。

### 分壳容差（pair_shell_tolerance）

默认 `0.05` Å。距离差不超过此值的原子对归入同一近邻壳层。

### 元素对筛选（pair_element_filter）

默认空，表示不筛选。可写 `Fe-Co` 或 `Fe-Fe,Fe-Co`。

### 标签对筛选（pair_group_filter）

默认空，表示不筛选。可写 `A-B`；输入必须含 `atoms.arrays['group']`。

### 键方向筛选（bond_filter_mode）

默认 `Any`。可选 `Any`、`Near axis`、`Near plane`，分别表示不限方向、靠近参考轴、靠近以参考向量为法向的平面。

### 键方向参考（bond_filter_axis）

默认 `[0,0,1]`，为笛卡尔向量；在 `Near axis` 中表示轴，在 `Near plane` 中表示平面法向。

### 键方向容差（bond_filter_tolerance）

默认 `20` 度。只在启用方向筛选时生效。

### 左分组（group_a）

默认 `A`。group A 的非零磁矩旋转 $+\theta/2$。

### 右分组（group_b）

默认 `B`。group B 的非零磁矩旋转 $-\theta/2$。

### 旋转轴（rotation_axis）

默认 `[0,1,0]`，为笛卡尔向量，仅用于旋转响应。若磁矩平行于旋转轴，该磁矩不会改变。

### 合格元素（apply_elements）

默认空，表示所有元素。只允许所列元素中模长非零的磁矩成为单原子目标或自动原子对候选，例如 `Fe,Co`。

### 模长比例扫描（moment_scale_scan）

默认 `0.8,0.9,1.0,1.1,1.2`，仅用于磁矩模长响应。比例必须非负、互不重复并包含 1.0。

### 最大结构数（max_outputs）

默认 `100`。只保留能完整放入预算的 group。

## 常见问题

- **没有可用目标：** 检查输入磁矩是否为非零三分量向量，以及原子索引和合格元素是否匹配。
- **自动配对为空：** 先关闭可选筛选并确认近邻壳层，再逐项加入元素对、标签对或键方向限制。
- **分组对为空：** 先用“原子层分组”卡写入 `group`，并确认两个标签中都存在非零磁矩。

## 输出字段

输出保留结构的原子、坐标、晶胞和 PBC，磁矩统一写入 `spin:R:3`。每帧带有 `response_group`、`response_coordinate`、`response_branch`、`response_task_id` 和来源结构标识；相同输入与参数会得到确定性的响应内容和任务标识。
