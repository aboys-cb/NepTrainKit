<!-- card-schema: {"card_name": "Local Magnetic Response", "source_file": "src/NepTrainKit/ui/views/_card/local_magnetic_response_card.py", "serialized_keys": ["params"]} -->

# 局域磁响应（Local Magnetic Response）

**分类：** 磁性

## 功能说明

把一个已有 `spin:R:3` 或矢量磁矩的结构展开成完整、对称且可追踪的局域响应组。单自旋、原子对和 group-pair 使用一致的小角旋转定义；每个目标拥有自己的零角参考帧，输出预算只按完整组截断。Moment magnitude preset 固定方向，只扫描模长。

## 原理与公式

原子对模式把总角度平分：

$$\theta_L=+\theta/2,\qquad\theta_R=-\theta/2.$$

训练预处理从同组 $\mathbf S(\theta)$ 数值反推 $d\mathbf S/d\theta$，再使用 $g(\theta)=\sum_i\mathbf m_i^{\mathrm{force}}\cdot d\mathbf S_i/d\theta$。卡片不生成 J/D 标签，也不持久化 `spin_tangent`。

## 操作示例

### 场景：总 RMSE 不高，但相邻自旋的小角响应符号不稳定

普通随机非共线帧没有提供可排序的局部扫描曲线。输入带自旋的目标结构，选择 `Atom pair canting`，手动给出 `1` 和 `2`，扫描 `-2,-1,0,1,2` 度。输出五帧构成一个 group；DFT 回填后检查正负角分支和 $g(\theta)$ 的奇偶部分。结果只证明这条受控路径可辨识，不能单独证明所有 J/D 已学准。

## 参数说明

### Response Kind（response_kind）

`str`，默认 `Atom pair canting`。可选 `Single-spin tilt`、`Atom pair canting`、`Group pair canting` 和 `Moment magnitude`；它决定目标是一颗自旋、一对自旋、两组自旋还是模长缩放。

### Coordinate Scan Degrees（coordinate_scan_deg）

`str`，默认 `-2,-1,0,1,2`。界面使用“最小值 / 最大值 / 步长”的角度范围控件，单位度；非等距旧扫描可打开“自定义坐标列表”。输出元数据换成弧度。必须至少三个不同点并包含零点，正负分支必须配对。

### Target Mode（target_mode）

`str`，默认 `First eligible atom`。单自旋或模长扫描的目标选择；填写 Target atoms 后界面使用显式索引。

### Target Indices（target_indices）

`str`，默认空。1-based 原子索引，例如 `1,3-5`。

### Pair Source（pair_source）

`str`，默认 `Manual indices`。原子对可手动指定，也可按近邻壳层自动选择。

### Pair Left Indices（pair_left_indices）

`str`，默认 `1`。手动 pair 的左侧 1-based 索引，与右侧逐项配对。

### Pair Right Indices（pair_right_indices）

`str`，默认 `2`。手动 pair 的右侧 1-based 索引。

### Pair Shell（pair_shell）

`int`，默认 `1`。自动 pair 使用的近邻壳层。

### Pair Shell Tolerance（pair_shell_tolerance）

`float`，默认 `0.05` Å。距离分壳容差。

### Pair Element Filter（pair_element_filter）

`str`，默认空。自动 pair 的元素对筛选，例如 `Fe-Co`。

### Pair Group Filter（pair_group_filter）

`str`，默认空。按 `atoms.arrays['group']` 筛选自动 pair。

### Bond Filter Mode（bond_filter_mode）

`str`，默认 `Any`。可限制为靠近指定轴或平面的键。

### Bond Filter Axis（bond_filter_axis）

三维 Cartesian 向量，默认 `[0,0,1]`。键方向筛选的参考轴。

### Bond Filter Tolerance（bond_filter_tolerance）

`float`，默认 `20` 度。键方向允许偏离参考方向的角度。

### Group A（group_a）

`str`，默认 `A`。group-pair 左组标签。

### Group B（group_b）

`str`，默认 `B`。group-pair 右组标签。

### Rotation Axis（rotation_axis）

三维 Cartesian 向量，默认 `[0,1,0]`。定义确定性的旋转平面法向。

### Apply Elements（apply_elements）

`str`，默认空。限制可参与目标选择的元素。

### Moment Scale Scan（moment_scale_scan）

`str`，默认 `0.8,0.9,1.0,1.1,1.2`。只在 Moment magnitude preset 生效；保持方向并按比例缩放模长。

### Max Outputs（max_outputs）

`int`，默认 `100`。预算不足一个完整 group 会报错；能容纳多个 group 时只保留完整 group。

## 推荐预设

```json
{"class":"LocalMagneticResponseCard","params":{"response_kind":"Atom pair canting","coordinate_scan_deg":"-2,-1,0,1,2","pair_left_indices":"1","pair_right_indices":"2","rotation_axis":[0,1,0],"max_outputs":100}}
```

## 推荐组合

- `Set Magnetic Moments → Local Magnetic Response`：先建立明确的参考自旋，再做局域扫描。
- `Group Label → Local Magnetic Response`：为 AFM 子晶格建立 group-pair 扫描。

## 常见问题

- 找不到 pair：检查 1-based 索引、近邻壳层、元素/group 和键方向筛选。
- 输出少于预期：上限只按完整 group 截断；提高 Max outputs。
- group 被判 invalid：检查扫描是否含零点、正负是否成对，以及 DFT 后同组磁矩是否坍缩或翻转。

## 输出标签

`Config_type` 追加 `MagResponse(<kind>,<branch>)`；最小 header 同时含 `response_schema/group/probe/coordinate/branch`。

## 可复现性

操作不使用随机数；相同结构和参数得到相同 group/task id 与相同自旋。
