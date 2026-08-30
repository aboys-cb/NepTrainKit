<!-- card-schema: {"card_name": "Shear Angle Strain", "source_file": "src/NepTrainKit/ui/views/_card/shear_angle_card.py", "serialized_keys": ["params"]} -->

# 晶格角应变（Shear Angle Strain）

**分类：** 晶格

## 功能说明

保持三条晶格矢量的长度不变，扫描晶格角的增量。适合单独补充角度形变数据，或检查模型对低对称晶胞的外推误差。

三个角的定义是：

- $\alpha=\angle(\mathbf b,\mathbf c)$
- $\beta=\angle(\mathbf a,\mathbf c)$
- $\gamma=\angle(\mathbf a,\mathbf b)$

输入框填写的是相对原始角度的**增量**，不是最终角度，单位均为度。

## 原理与公式

对每组增量 $(\Delta\alpha,\Delta\beta,\Delta\gamma)$：

$$
\alpha'=\alpha+\Delta\alpha,\qquad
\beta'=\beta+\Delta\beta,\qquad
\gamma'=\gamma+\Delta\gamma.
$$

程序用 $(a,b,c,\alpha',\beta',\gamma')$ 重建晶胞，其中 $a,b,c$ 保持原值。原子分数坐标随晶胞保持不变，因此笛卡尔位置会随形变移动。

重建过程保留输入晶胞的全局取向，不额外施加刚体旋转。因此项目使用的 `spin:R:3` 和 ASE 的 `initial_magmoms` 均保留原笛卡尔分量。若输入同时含有这两类磁矩数据，两者都不会被角度应变改写。

每个角范围按 `[min, max, step]` 取点。若三条范围分别产生 $N_\alpha$、$N_\beta$、$N_\gamma$ 个值，则：

$$
N_{\mathrm{out}}=N_{\mathrm{in}}N_\alpha N_\beta N_\gamma.
$$

默认每条范围为 `[-2, 2, 2]`，对应小角度弹性响应中的负向、参考和正向三点，
即每个输入生成 $3^3=27$ 个结构。界面会直接显示每个输入和全部输入的预计输出数。

最终三个角必须都位于 $(0^\circ,180^\circ)$，并能组成非奇异、右手晶胞；不满足时会提示减小角度增量。

## 操作示例

只扫描 $\beta$ 角的 $-4^\circ$ 到 $4^\circ$，步长 $2^\circ$：

```text
Alpha increment: 0, 0, 1
Beta increment: -4, 4, 2
Gamma increment: 0, 0, 1
```

每个输入得到 5 个结构，对应 $\Delta\beta=-4,-2,0,2,4^\circ$。三条晶格矢量长度不变；$\alpha$ 和 $\gamma$ 不变。

不扫描某一通道时应填写 `[0, 0, 1]`。步长仍需为正数。

## 参数说明

### α 增量（alpha_range）

默认 `[-2, 2, 2]`。$\alpha=\angle(\mathbf b,\mathbf c)$ 的增量范围，单位为度。

### β 增量（beta_range）

默认 `[-2, 2, 2]`。$\beta=\angle(\mathbf a,\mathbf c)$ 的增量范围，单位为度。

### γ 增量（gamma_range）

默认 `[-2, 2, 2]`。$\gamma=\angle(\mathbf a,\mathbf b)$ 的增量范围，单位为度。

### 保持识别出的分子刚性（identify_organic）

默认关闭。开启后，程序在晶胞仿射形变后恢复检测到的分子团内部几何，避免分子内键随晶胞一起拉伸。纯无机体系通常无需开启。

## 配置示例

```json
{
  "class": "ShearAngleCard",
  "check_state": true,
  "params": {
    "alpha_range": [0, 0, 1],
    "beta_range": [-4, 4, 2],
    "gamma_range": [0, 0, 1],
    "identify_organic": false
  }
}
```

## 常见问题

**为什么角度改变后体积也变了？** 体积同时取决于晶格长度和夹角。长度固定不代表体积固定，这是正常的几何结果。

**为什么输出数很快变大？** 三个角采用笛卡尔积组合。默认是 $3\times3\times3=27$ 个结构/输入；增加点数前应先确认确实需要联合扫描，而不是单角有限差分。

**为什么提示晶胞无效？** 某组最终角度无法组成非奇异右手晶胞。缩小增量范围，尤其要检查原始角度已经接近 $0^\circ$ 或 $180^\circ$ 的情况。

**磁矩会跟着晶胞转动吗？** 不会。这张卡只做角度应变并保留输入全局坐标系；`spin:R:3` 与 ASE `initial_magmoms` 的笛卡尔分量均保持不变。

## 输出标签

`Ang(alpha={Δα},beta={Δβ},gamma={Δγ})`

标签记录三个角的增量，单位为度。该操作无随机性，相同输入和参数会生成相同结果。
