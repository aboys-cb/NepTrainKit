<!-- card-schema: {"card_name": "Shear Matrix Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_matrix_card.py", "serialized_keys": ["params"]} -->

# 剪切矩阵应变（Shear Matrix Strain）

`Group`: `Lattice` | `Class`: `ShearMatrixCard`

## 功能说明

通过 xy/yz/xz 剪切矩阵分量生成非对角形变样本。对晶格矢量的三个剪切通道做系统扫描，每个通道独立控制 min/max/step。

$$\gamma_{xy}=\frac{s_{xy}}{100},\quad \mathbf{S}=\begin{bmatrix}1&\gamma_{xy}&\gamma_{xz}\\0&1&\gamma_{yz}\\0&0&1\end{bmatrix},\quad \mathbf{C}'=\mathbf{C}\mathbf{S}$$

`symmetric=true` 时填充 S 的下三角对应项，使剪切路径更接近对称形变。

## 操作示例

### 场景：模型预测剪切模量 C44 比 DFT 低 40%

你在 fcc Al 上训练了一个 NEP 模型，C11/C12 都对，但 C44 只有 DFT 值的 60%。诊断发现：C44 是对应于 xy/yz/xz 方向剪切的分量，而训练集里所有结构都是轴向拉伸/压缩（通过 `Lattice Strain` 生成的），模型没见过非对角剪切变形。

**诊断思路：** 剪切弹性常数来源于非对角应变。如果训练集只覆盖对角应变（axial strain），模型对剪切方向的刚度完全靠外推。需要往训练集里加入已知剪切分量的结构。

**输入：** 一个弛豫好的 fcc Al 单胞

**目标：** 沿 xy 方向做 -3% 到 +3% 剪切，步长 1%，对称模式，共 7 个结构

**参数设置：**
- `xy_range` = `[-3, 3, 1]`
- `yz_range` = `[0, 0, 1]` （不扫 yz）—— 注意要把步长非零的值写进去，否则 range 为空不生成
- `xz_range` = `[0, 0, 1]` （不扫 xz）
- `symmetric` = `true`

**输出：** 7 个结构，剪切矩阵的 xy 分量从 -0.03 到 +0.03，yz 和 xz 分量为 0

**怎么验证训练集质量改善：**
- 重训后重新计算弹性常数矩阵，C44 应该更接近 DFT 参考值
- 检查输出结构的晶格角变化：单斜/triclinic 体系剪切后角度会偏离正交，确认变化在物理合理范围
- 如果 C44 改善但 C55/C66 仍有偏差，再单独扫 yz_range 和 xz_range
- 如果三个分量都需要同时补，再同时放开三个 range —— 但注意输出数 = Nxy * Nyz * Nxz，组合数很快爆炸

### 什么时候加这张卡、什么时候不加

**加：**
- 模型剪切模量、非对角弹性常数系统性偏差
- 训练集缺少非正交晶格构型、所有结构都在高对称格点
- 需要覆盖剪切方向的应力-应变响应

**不加：**
- 只需要体积和单轴应变 → `Lattice Strain` 更直接
- 想通过角度而非矩阵分量控制剪切 → `Shear Angle Strain` 更适合
- 体系本身是刚性分子晶体，剪切只会破坏分子内拓扑 → 考虑 `identify_organic`

## 参数说明

**`xy_range`**（list[3]，默认 `[-5, 5, 1]` (%)）：XY 剪切分量扫描区间 `[min, max, step]`，单位 %。`sxy=5` 对应剪切矩阵分量 `0.05`。只扫一个分量时把另两个设为一个点（如 `[0, 0, 1]`）。

**`yz_range`**（list[3]，默认 `[-5, 5, 1]` (%)）：YZ 剪切分量，同上。

**`xz_range`**（list[3]，默认 `[-5, 5, 1]` (%)）：XZ 剪切分量，同上。

**`symmetric`**（bool，默认 true）：对称剪切。开启后剪切矩阵下三角同步填充（`S[1,0] = S[0,1]` 等），剪切路径更接近物理对称形变。研究非对称畸变时关闭。

**`identify_organic`**（bool，默认 false）：有机团簇识别。分子晶体必须开启，纯无机体系关闭。

## 推荐预设

### 单分量弹性补样（仅 xy，±3%，对称）
```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "xy_range": [-3, 3, 1],
  "yz_range": [0, 0, 1],
  "xz_range": [0, 0, 1],
  "symmetric": true,
  "identify_organic": false
}
```

### 双分量剪切覆盖（xy+yz，±5%，对称，~121 个输出）
```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "xy_range": [-5, 5, 1],
  "yz_range": [-5, 5, 1],
  "xz_range": [0, 0, 1],
  "symmetric": true,
  "identify_organic": false
}
```

### 三通道研究级（xy+yz+xz，±6%，步长 2%，非对称，~343 个输出）
```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "xy_range": [-6, 6, 2],
  "yz_range": [-6, 6, 2],
  "xz_range": [-6, 6, 2],
  "symmetric": false,
  "identify_organic": false
}
```

## 推荐组合

- `Lattice Strain` -> `Shear Matrix Strain`：先补轴向应变，再补剪切分量，覆盖完整弹性张量
- `Shear Matrix Strain` -> `Atomic Perturb`：剪切变形后加坐标噪声
- `Super Cell` -> `Shear Matrix Strain`：先扩胞再剪切，适合研究大尺度剪切响应

## 常见问题

**输出为空。** 检查 range 的步长是否 > 0。如果不扫某个分量，不能设置 `[0, 0, 0]` —— 步长为 0 会导致 range 为空。应该设置 `[0, 0, 1]` 确保至少生成一个点（0）。

**输出数量爆炸。** 三通道联扫的输出数 = Nxy * Nyz * Nxz。默认 `[-5, 5, 1]` 三个通道同时开 = 11^3 = 1331 个结构。建议先单通道试跑验证参数合理性后再加通道。

**剪切后结构物理不合理。** 剪切幅度过大导致晶格条件数恶化（接近奇异）。抽查最近邻距离和晶格行列式。±5% 以上建议逐步增加并每步检查。

**vs Shear Angle Strain 如何选择。** Shear Matrix 在笛卡尔坐标下直接修改剪切矩阵分量，适合和弹性常数计算对齐；Shear Angle 修改晶胞角度 alpha/beta/gamma，适合和 XRD/晶体学角度对齐。两者产生不同的形变路径，可以互补使用。

## 输出标签

`Shr(xy={sxy}%,yz={syz}%,xz={sxz}%,sym={0|1})`

## 可复现性

无随机性。同参数同输入 → 严格一致输出。所有剪切分量按固定网格扫描。
