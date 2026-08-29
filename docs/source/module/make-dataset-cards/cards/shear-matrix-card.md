<!-- card-schema: {"card_name": "Shear Matrix Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_matrix_card.py", "serialized_keys": ["params"]} -->

# 剪切矩阵应变（Shear Matrix Strain）

**分类：** 晶格

## 功能说明

在固定的全局笛卡尔 $x/y/z$ 坐标系中扫描剪切矩阵分量。它适合生成简单剪切路径，或补充非对角应变张量分量。

这里的 `xy`、`yz`、`xz` 指笛卡尔矩阵分量，不是晶格矢量 $a/b/c$ 的组合。对旋转或三斜晶胞仍使用同一全局笛卡尔定义。

## 原理与公式

程序按 ASE 的行矢量晶胞约定计算：

$$
\mathbf C'=\mathbf C\mathbf S.
$$

### 简单剪切 γij

$$
\mathbf S_{\mathrm{simple}}=
\begin{bmatrix}
1 & \gamma_{xy} & \gamma_{xz}\\
0 & 1 & \gamma_{yz}\\
0 & 0 & 1
\end{bmatrix}.
$$

界面输入单位为百分数，例如 `xy = 5%` 表示 $\gamma_{xy}=0.05$。坐标变换可写成：

$$
x'=x,\qquad
y'=y+\gamma_{xy}x,\qquad
z'=z+\gamma_{xz}x+\gamma_{yz}y.
$$

### 对称应变 εij

$$
\mathbf S_{\mathrm{symmetric}}=
\begin{bmatrix}
1 & \varepsilon_{xy} & \varepsilon_{xz}\\
\varepsilon_{xy} & 1 & \varepsilon_{yz}\\
\varepsilon_{xz} & \varepsilon_{yz} & 1
\end{bmatrix}.
$$

此模式的输入是应变张量非对角分量。在线性小应变约定下，工程剪切量满足：

$$
\gamma_{ij}=2\varepsilon_{ij}.
$$

因此，对称模式输入 `5%` 表示 $\varepsilon_{xy}=0.05$，对应工程剪切量 $\gamma_{xy}=0.10$；它与简单剪切模式的 `5%` 不是同一个物理幅度。

原子分数坐标随晶胞保持不变。`spin:R:3` 与 ASE `initial_magmoms` 保留输入全局坐标系中的笛卡尔分量，不随晶胞形变改写。

若三条范围分别产生 $N_{xy}$、$N_{yz}$、$N_{xz}$ 个值，则：

$$
N_{\mathrm{out}}=N_{\mathrm{in}}N_{xy}N_{yz}N_{xz}.
$$

默认每条范围为 `[-2, 2, 2]`%，对应小应变弹性响应中的负向、参考和正向三点，
因此每个输入生成 $3^3=27$ 个结构。界面会显示组合数和预计总输出数。

每个输出都必须保持有限、非奇异且为右手晶胞；否则程序会提示减小剪切分量。

## 操作示例

生成 $xy$ 简单剪切路径，范围为 $-3\%$ 到 $3\%$，步长 $1\%$：

```text
Deformation mode: Simple shear γij
γxy: -3, 3, 1
γyz: 0, 0, 1
γxz: 0, 0, 1
```

每个输入得到 7 个结构。`xy = 3%` 对应 $y'=y+0.03x$。不扫描的通道填写 `[0, 0, 1]`，因为步长必须为正数。

## 参数说明

### xy 分量范围（xy_range）

默认 `[-2, 2, 2]`，格式为 `[min, max, step]`，单位为百分数。其含义由形变模式决定：简单剪切中为 $\gamma_{xy}$，对称应变中为 $\varepsilon_{xy}$。

### yz 分量范围（yz_range）

默认 `[-2, 2, 2]`。简单剪切中为 $\gamma_{yz}$，对称应变中为 $\varepsilon_{yz}$。

### xz 分量范围（xz_range）

默认 `[-2, 2, 2]`。简单剪切中为 $\gamma_{xz}$，对称应变中为 $\varepsilon_{xz}$。

### 形变模式（symmetric）

默认 `true`，表示对称应变张量模式；`false` 表示简单剪切模式。该字段沿用布尔序列化值，界面中以两个明确的模式名称呈现。

### 保持识别出的分子刚性（identify_organic）

默认关闭。开启后，程序在晶胞仿射形变后恢复检测到的分子团内部几何。对希望保持分子内键长和键角的体系可开启。

## 配置示例

```json
{
  "class": "ShearMatrixCard",
  "check_state": true,
  "params": {
    "xy_range": [-3, 3, 1],
    "yz_range": [0, 0, 1],
    "xz_range": [0, 0, 1],
    "symmetric": false,
    "identify_organic": false
  }
}
```

## 常见问题

**为什么输出数会很快增加？** 三个范围按笛卡尔积组合，默认是 $3\times3\times3=27$ 个结构/输入。单通道扫描时，把另外两条范围设为 `[0, 0, 1]`。

**为什么对称模式和简单剪切的相同百分数效果不同？** 两者输入的物理量不同：简单剪切使用工程剪切量 $\gamma$，对称模式使用张量分量 $\varepsilon$；在线性小应变下 $\gamma=2\varepsilon$。

**为什么提示晶胞无效？** 当前分量组合使矩阵奇异或翻转成左手晶胞。减小范围，或避免同时使用过大的多个分量。

## 输出标签

`Shr(xy={sxy}%,yz={syz}%,xz={sxz}%,mode={simple|symmetric})`

零分量也会明确记录。全零组合是严格 no-op；该操作无随机性。
