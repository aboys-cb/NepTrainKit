<!-- card-schema: {"card_name": "Spin Perturb", "source_file": "src/NepTrainKit/ui/views/_card/magmom_rotation_card.py", "serialized_keys": ["params"]} -->

# 磁矩扰动（Spin Perturb）

**分类：** 磁性

## 功能说明

围绕输入已有的磁态生成随机局部扰动样本。每个命中的非零磁矩在当前方向周围的球冠内独立采样，并可同时改变模长；每个输入严格生成指定数量的输出。

输入优先读取标准 `spin:R:3`，没有时兼容 ASE 的 `initial_magmoms`。标量磁矩必须先沿指定笛卡尔方向转成三维矢量。

## 原理与公式

令 $\hat{\mathbf m}_i$ 为目标原子 $i$ 的原始磁矩方向。程序在以它为中心、半角为 $\alpha_{\max}$ 的球冠内按立体角均匀采样：

$$
\cos\theta_i\sim U(\cos\alpha_{\max},1),
\qquad
\phi_i\sim U(0,2\pi),
\qquad
0\leq\theta_i\leq\alpha_{\max}.
$$

$\theta_i$ 就是新旧磁矩之间的真实夹角，$\phi_i$ 是绕原磁矩方向的方位角。这样不会把内部“随机旋转轴”暴露给用户，也不会让采样参数和实际偏转角不一致。界面中的“标量抬升方向”只在标量输入转为矢量时使用。

开启模长采样后，再独立采样：

$$
\lambda_i\sim U(f_{\min},f_{\max}),
\qquad
\mathbf m_i'=\lambda_i\widetilde{\mathbf m}_i.
$$

关闭模长采样时 $\lambda_i=1$。最大角度设为 0 时，可以只生成模长扰动样本。

## 操作示例

### 补充参考磁态附近的局部涨落

若模型只在严格 FM/AFM 方向上误差较低，而小角度偏转测试集上的能量或力误差明显升高，可从已有参考磁态出发：

1. 最大偏转角设为 `10°`；
2. 模长范围设为 `0.95–1.05`；
3. 每个输入生成 `5` 个结构并固定 seed；
4. 重训后在独立的小角度偏转测试集上比较误差，并检查输出磁矩的实际最大偏转和模长范围。

只想扰动 Fe 时，将目标元素填为 `Fe`；留空则选择所有非零磁矩原子。

## 参数说明

### 目标元素（elements）

`str`，默认空。使用逗号或空格分隔元素符号，例如 `Fe,Ni`。留空选择所有非零磁矩原子；填写后，所有命中的非零磁矩都会参与每个输出的独立随机扰动。

### 最大偏转角（max_angle）

`float`，默认 `10.0`，单位度，范围 `0–180`。它是新旧磁矩真实夹角的上界；球冠内按立体角均匀采样，不是每个原子都固定偏转该角度。

### 每个输入生成（num_structures）

`int`，默认 `5`，至少为 1。总输出数为：

$$
N_\mathrm{out}=N_\mathrm{input}\times\texttt{num\_structures}.
$$

### 标量磁矩转为矢量（lift_scalar）

`bool`，默认 `true`。输入为标量 `initial_magmoms` 时必须开启；三列 `spin:R:3` 或向量 `initial_magmoms` 不需要抬升。

### 标量抬升方向（axis）

`tuple[float, float, float]`，默认 `(0,0,1)`，使用笛卡尔坐标。仅用于将标量磁矩变成三维矢量，必须是有限非零向量；对已有向量磁矩不改变初始方向。

### 模长采样（disturb_magnitude）

`bool`，默认 `true`。开启后，旋转完成的每个目标磁矩再乘以独立随机缩放因子。

### 模长缩放范围（magnitude_factor）

`tuple[float, float]`，默认 `(0.95,1.05)`。依次为缩放因子下限和上限；两者必须有限、非负且下限不大于上限。

### 使用随机种子（use_seed）

`bool`，默认 `false`。开启后，相同结构、参数和 seed 得到相同结果。

### 随机种子（seed）

`int`，默认 `0`。随机流由该值和结构内容共同派生，因此数据集重新排序不会改变某一结构自己的结果。

## 常见问题

**为什么提示没有匹配的磁矩？** 目标元素没有出现在结构中，或命中原子的磁矩全为零。检查 `spin`/`initial_magmoms` 和目标元素。

**为什么标量磁矩无法运行？** 开启“标量磁矩转为矢量”，并给出有限、非零的笛卡尔抬升方向；也可以先用“设置磁矩”统一为 `spin:R:3`。

**与“可控自旋倾斜”有什么区别？** 本卡生成参考磁态周围的随机样本云；“可控自旋倾斜”按指定原子、原子对或分组生成确定性角度扫描。

## 输出合同

- 每个输入严格生成 `num_structures` 个输出；输入不满足要求时明确失败；
- 原子、坐标、晶胞和 PBC 与输入一致，输出磁矩写入 `spin:R:3` 并同步 ASE `initial_magmoms`；
- `Config_type` 追加 `SpinPert(...)`；最大角度为 0、仅采样模长时追加 `SpinScale(...)`；
- `spin_perturbation` metadata 记录球冠分布、目标元素、参数、seed、样本序号、目标数量、实际角度统计和实际缩放范围。

<details>
<summary>示例配置</summary>

```json
{
  "class": "MagneticMomentRotationCard",
  "check_state": true,
  "params": {
    "elements": "Fe",
    "max_angle": 10.0,
    "num_structures": 5,
    "lift_scalar": true,
    "axis": [0.0, 0.0, 1.0],
    "disturb_magnitude": true,
    "magnitude_factor": [0.95, 1.05],
    "use_seed": true,
    "seed": 42
  }
}
```

</details>
