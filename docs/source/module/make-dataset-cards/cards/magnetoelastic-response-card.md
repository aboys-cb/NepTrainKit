<!-- card-schema: {"card_name": "Magnetoelastic Response", "source_file": "src/NepTrainKit/ui/views/_card/magnetoelastic_response_card.py", "serialized_keys": ["params"]} -->

# 磁弹响应（Magnetoelastic Response）

**分类：** 磁性

## 功能说明

把结构坐标和同一套局域自旋 probe 做笛卡尔组合。每个 strain/volume 点都有独立且完整的 rotation group，共享同一个 response parent，因此 DFT 回填后可做跨应变配对和 difference-of-differences 验收。

## 原理与公式

晶胞通过变形梯度 $\mathbf F$ 更新，有限小应变记录为 $\boldsymbol\epsilon=(\mathbf F+\mathbf F^T)/2-\mathbf I$。位置采用“分数坐标固定、随晶胞变换”的约定；spin scan 仍以弧度写入 coordinate。

## 操作示例

### 场景：模型能拟合平衡结构，却不能复现应变下的磁响应变化

输入带自旋的弛豫结构，选择 `Uniaxial strain`，结构网格设 `-0.02,-0.01,0,0.01,0.02`，自旋角设 `-2,0,2`。输出 5 个结构点 × 3 个完整 probe。回填 DFT 后按 parent 和 structural coordinate 配对，比较各应变点的 $g(\theta)$；本卡只生成和审计数据，不在 NepTrainKit 内新增训练 loss。

## 参数说明

### Structural Mode（structural_mode）

`str`，默认 `Isotropic volume`。可选各向同性体积、单轴、双轴、对称剪切和 Bain/tetragonal 路径；volume 与 anisotropic strain 在 manifest 中使用不同 structural probe。

### Structural Scan（structural_scan）

`str`，默认 `-0.02,-0.01,0,0.01,0.02`。序列化时是无量纲结构坐标；界面按晶格路径动态显示“体积变化 / 轴向应变 / 面内应变 / 剪切应变 / 四方应变”，并使用与 Lattice Strain 相同的“最小值 / 最大值 / 步长（%）”控件。界面会自动把百分比转换成无量纲小数。扫描应含零点和对称正负点。

### Spin Scan Degrees（spin_scan_deg）

`str`，默认 `-2,0,2` 度。每个结构点重复的完整自旋 probe；输出 coordinate 为弧度。

### Rotation Axis（rotation_axis）

Cartesian 三维向量，默认 `[0,1,0]`。局域自旋旋转轴。

### Target Indices（target_indices）

`str`，默认 `1`。参与自旋 probe 的 1-based 原子索引。

### Strain Axis（strain_axis）

Cartesian 三维向量，默认 `[0,0,1]`。单轴/双轴/剪切模式的参考方向。

### Max Outputs（max_outputs）

`int`，默认 `100`。按每个结构点的一整套 spin probes 截断。

## 推荐预设

```json
{"class":"MagnetoelasticResponseCard","params":{"structural_mode":"Uniaxial strain","structural_scan":"-0.02,-0.01,0,0.01,0.02","spin_scan_deg":"-2,0,2","target_indices":"1","strain_axis":[0,0,1],"max_outputs":100}}
```

## 推荐组合

- `Set Magnetic Moments → Magnetoelastic Response`：确保各结构点从同一参考自旋出发。
- `Group Label → Magnetoelastic Response`：后续扩展 group-pair probe 时保留子晶格语义。

## 常见问题

- 输出数不是两个扫描长度的乘积：Max outputs 只保留能完整容纳的结构点。
- strain tensor 方向不对：Strain axis 是 Cartesian 方向，不是 Miller 指数。
- DFT 后 group invalid：同组 SCF 磁矩可能翻转或坍缩，应排除整组而不是强行进入 grouped loss。

## 输出标签

`Config_type` 追加 `MagResponse(magnetoelastic_spin_probe,branch)`；完整 $\mathbf F$、strain tensor 和 structural coordinate 位于 manifest。

## 可复现性

所有扫描确定性执行；parent/group/task id 由输入结构与参数语义稳定派生。
