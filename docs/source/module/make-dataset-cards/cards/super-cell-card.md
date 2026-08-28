<!-- card-schema: {"card_name": "Super Cell", "source_file": "src/NepTrainKit/ui/views/_card/super_cell_card.py", "serialized_keys": ["params"]} -->

# 扩胞（Super Cell）

**分类：** 晶格

## 功能说明

这张卡沿晶格矢量 $\mathbf a$、$\mathbf b$、$\mathbf c$ 整数次复制完整输入晶胞。输出保留输入结构的局部几何，只扩大周期重复范围。

| 你的目标 | 扩胞依据 | 关键设置 |
| --- | --- | --- |
| 已知要生成 4×4×4、6×6×1 等尺寸 | **指定重复倍率** | 直接填写 a、b、c 三个整数 |
| 希望输出晶格矢量达到某个长度 | **指定目标长度** | 一般选“至少达到”；希望整数扩胞不越过目标时选“不超过” |
| 只知道最多能计算多少原子 | **指定原子预算** | 填严格上限；程序自动选预算利用最多且形状较均衡的倍率 |

通常保持“单个超胞”。“枚举尺寸”会真实生成多组结构，只适合尺寸收敛或多尺度采样。对于 slab，应锁定含真空的晶格矢量并把固定倍率设为 1。

## 原理与公式

### 整数扩胞

设输入晶胞矩阵为 $\mathbf C$，输入原子数为 $N$。卡片使用对角超胞矩阵：

$$
\mathbf P=\operatorname{diag}(n_a,n_b,n_c),\qquad
\mathbf C'=\mathbf P\mathbf C,
$$

$$
N'=Nn_an_bn_c.
$$

$n_a,n_b,n_c$ 都是正整数。非正交晶胞同样按整根晶格矢量复制；这里的 a、b、c 不是 Cartesian x、y、z。

### 三种扩胞依据

#### 指定重复倍率

单输出直接使用填写的 $(n_a,n_b,n_c)$。例如输入长度为 $(5,6,7)$ Å，倍率为 $(3,2,1)$，输出长度就是 $(15,12,7)$ Å，原子数变为 6 倍。

#### 指定目标长度

设输入晶格矢量长度为 $L_i$，目标长度为 $T_i$：

$$
n_i=\max\left(1,\left\lceil T_i/L_i\right\rceil\right)
\quad\text{（至少达到）},
$$

$$
n_i=\max\left(1,\left\lfloor T_i/L_i\right\rfloor\right)
\quad\text{（不超过）}.
$$

例如输入长度 $(5,6,7)$ Å、目标 $(22,20,15)$ Å：“至少达到”得到 $(5,4,3)$，“不超过”得到 $(4,3,2)$。恰好整除时两者得到相同倍率。

这张卡不能缩胞。若输入长度已经超过“不超过”目标，倍率最低仍为 1，输出会保留原长度。

#### 指定原子预算

原子上限 $M$ 是严格预算，可行倍率满足：

$$
Nn_an_bn_c\le M.
$$

单输出严格按以下顺序选择：

1. 最大化输出原子数 $N'$；
2. 原子数并列时，最小化输出晶格矢量的最长/最短比。

例如输入有 2 个原子、长度为 $(5,6,7)$ Å，上限为 100。程序选择 $(5,5,2)$：输出恰好 100 个原子，长度为 $(25,30,14)$ Å，且比同原子数的细长方案更均衡。

若上限小于输入原子数，或固定轴倍率已经超过预算，卡片会直接报错。

### 单输出、枚举与固定轴

- **单个超胞：** 每个输入只生成最终选中的一组倍率。
- **枚举尺寸：** 重复倍率和目标长度模式枚举从 1 到目标倍率的全部整数组合；原子预算模式枚举全部预算内组合。每个输入最多允许 1000 个输出。
- **固定轴：** 被锁定轴直接使用固定倍率，不再参与当前模式的自动计算。该规则在三种依据和两种输出方式下都生效。

## 操作示例

### 生成 4×4×4 缺陷母结构

输入是 2 原子的 bcc Fe 常规胞。选择“指定重复倍率”，填写 `[4,4,4]`，保持“单个超胞”且不锁轴。输出 1 个 128 原子的结构，可继续连接空位或间隙卡片。

### Slab 只扩大面内尺寸

输入长度为 $(5.6,5.6,25)$ Å，c 为含真空的法向。选择“指定目标长度”和“至少达到”，目标填 `[20,20,25]` Å；锁定 c，固定倍率设为 1。最终倍率为 $(4,4,1)$，面内扩大而真空方向不复制。

## 参数说明

### 扩胞依据

#### 模式（mode）

`str`，默认 `"scale"`。`scale` 读取手动重复倍率，`cell` 按目标长度取整，`max_atoms` 在严格原子预算内搜索倍率。

#### 输出方式（output_mode）

`str`，默认 `"single"`。`single` 每个输入输出 1 个超胞；`enumerate` 输出全部中间或可行倍率，最多 1000 个。

#### 长度约束（target_policy）

`str`，默认 `"at_least"`。仅 `mode="cell"` 时生效；`at_least` 使用 ceil，`at_most` 使用 floor。

#### 重复倍数（super_scale）

`tuple[int, int, int]`，默认 `(3,3,3)`，范围 1–999。仅 `mode="scale"` 时生效，依次对应晶格矢量 a、b、c。

#### 目标长度（target_cell）

`tuple[float, float, float]`，默认 `(20,20,20)` Å，范围 `0.001–9999.000` Å。仅 `mode="cell"` 时生效；它是整数扩胞目标，不会连续缩放晶格。

#### 原子数上限（max_atoms）

`int`，默认 100，范围 1–10000。仅 `mode="max_atoms"` 时生效，是输出原子数的严格上限。

### 轴向覆盖

#### 固定轴（fixed_axis_flags）

`tuple[bool, bool, bool]`，默认 `(false,false,false)`。依次决定是否锁定晶格矢量 a、b、c 的重复次数。

#### 固定倍率（fixed_axis_scale）

`tuple[int, int, int]`，默认 `(1,1,1)`，范围 1–999。只对已锁定轴生效；未锁轴的输入框会禁用。

## 推荐预设

:::{dropdown} 4×4×4 母结构

```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "metadata": {},
  "params": {
    "mode": "scale",
    "output_mode": "single",
    "target_policy": "at_least",
    "super_scale": [4, 4, 4],
    "target_cell": [20.0, 20.0, 20.0],
    "max_atoms": 100,
    "fixed_axis_flags": [false, false, false],
    "fixed_axis_scale": [1, 1, 1]
  }
}
```
:::

:::{dropdown} Slab 面内至少 20 Å

```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "metadata": {},
  "params": {
    "mode": "cell",
    "output_mode": "single",
    "target_policy": "at_least",
    "super_scale": [3, 3, 3],
    "target_cell": [20.0, 20.0, 20.0],
    "max_atoms": 100,
    "fixed_axis_flags": [false, false, true],
    "fixed_axis_scale": [1, 1, 1]
  }
}
```
:::

:::{dropdown} 300 原子预算

```json
{
  "class": "SuperCellCard",
  "check_state": true,
  "metadata": {},
  "params": {
    "mode": "max_atoms",
    "output_mode": "single",
    "target_policy": "at_least",
    "super_scale": [3, 3, 3],
    "target_cell": [20.0, 20.0, 20.0],
    "max_atoms": 300,
    "fixed_axis_flags": [false, false, false],
    "fixed_axis_scale": [1, 1, 1]
  }
}
```
:::

## 常见问题

**为什么目标长度与实际长度不完全相同？** 这张卡只能选择整数倍率；“至少达到”可能略高于目标，“不超过”可能略低于目标。

**为什么原子预算没有恰好用满？** 输出原子数必须是输入原子数与三个整数倍率的乘积；没有合适的倍率组合时，会选择预算内最接近上限的结果。

**为什么枚举被拒绝？** 预计每个输入会产生超过 1000 个结构。请缩小目标或改用“单个超胞”。

**二维材料应该锁哪个轴？** 锁包含真空的晶格矢量，未必永远是 c；先检查输入 cell 和 PBC。

:::{dropdown} 旧工作流迁移

旧 `behavior_type` 的迁移规则如下：

| 旧值 | 新输出方式 | 新长度约束 |
| --- | --- | --- |
| `0`（Maximum） | `single` | `at_most` |
| `1`（Iteration） | `enumerate` | `at_most` |
| `2`（Minimum） | `single` | `at_least` |

旧版原子预算算法可能生成极细长超胞或返回原胞；新版统一使用“先最大化原子数，再选择形状较均衡的超胞”。
:::

## 输出合同

真正扩胞时，`Config_type` 追加 `SC({na}x{nb}x{nc})`，例如 `SC(4x4x1)`；倍率为 `(1,1,1)` 时只返回副本，不追加标签。卡片没有随机过程，相同输入和参数会得到相同结果。
