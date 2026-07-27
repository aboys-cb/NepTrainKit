<!-- card-schema: {"card_name": "Vacancy Defect Generation", "source_file": "src/NepTrainKit/ui/views/_card/vacancy_defect_card.py", "serialized_keys": ["params"]} -->

# 全局随机空位（Vacancy Defect Generation）

`Group`: `Defect` | `Class`: `VacancyDefectCard`

## 功能说明

在每个输入结构的**全部原子位点**中随机删除若干原子，生成一组空位落点不同的结构。可以按绝对数量或整体比例决定删除量；不区分元素，也不读取 `group`。

这张卡只删除原子。保留下来的原子坐标、晶胞和周期性边界不变，因此输出是**未弛豫空位构型**。删除不同元素的概率由输入结构中的元素数量自然决定，多元素结构的实际组分可能随样本变化。

**和 `Random Vacancy` 的区别：**

- `Global Random Vacancy`：所有元素都可被删除，适合快速覆盖整体空位数量或比例。
- `Random Vacancy`：按元素和可选 `group` 规则选择候选位点，适合子晶格或特定元素空位。

## 快速使用

以一个 100 原子的超胞为例，要为每个输入生成最多 50 个、每个含 5 个空位的构型：

1. 选择 `Vacancy fraction (0–1)`，设为 `0.05`。
2. `Vacancies per output` 选择 `Fixed at the set value`。
3. `Maximum outputs per input` 设为 `50`。
4. `Site sampling` 保持 `Uniform random (recommended)`。
5. 做对照实验时勾选 `Use seed` 并填写固定种子。

首帧预览会显示实际删除的原子数和最多可生成的唯一输出数。输出带有 `Vac(n=5)` 标签。

## 参数说明

### 删除量

#### Vacancy count（num_condition）

`int`，默认 `1`。每个输出最多删除的原子数。必须至少为 1，且必须小于输入结构的原子数，程序不会再静默缩小过大的数值。

#### Vacancy fraction (0–1)（concentration_condition）

`float`，默认 `0.01`。按

```text
空位数 = floor(输入原子数 × 空位比例)
```

换算删除量。例如 100 个原子、比例 `0.05` 会删除 5 个原子。比例必须在 `(0, 1)` 内；若向下取整后为 0，卡片会提示当前结构至少需要的比例。

#### Use Num（use_num）

`bool`，默认 `true`。`true` 使用绝对数量；`false` 使用整体比例。界面中未选中的输入框会被禁用，避免误以为两个值会同时生效。

#### Vacancies per output（count_mode）

`str`，默认 `fixed`。

- `fixed`：每个输出都按上面解析出的数量删除。
- `random`：每个输出在 1 到解析值之间随机选择删除量。

`random` 改变的是**每个输出的空位数**，不是仅改变空位落点。

### 采样和输出

#### Maximum outputs per input（max_structures）

`int`，默认 `1`。每个输入结构请求的最大输出数。相同的删除位点组合会去重；当所有可能组合少于请求值时，实际输出会更少。

#### Site sampling（engine_type）

`int`，默认 `1`。

- `1`：Uniform，均匀随机，推荐作为通用默认。
- `0`：Sobol，准随机序列，适合希望更均匀扫描落点的情况。

Sobol 受 SciPy 维数上限约束，最多支持 21,200 个原子的输入；更大的结构请使用 Uniform。两种方式都只控制如何选择被删除的位点，不会移动原子。

#### Use Seed / Seed（use_seed / seed）

默认 `false` / `0`。启用后，相同参数和相同输入结构可复现。种子会与结构内容共同派生，因此数据集中几何不同的帧不会机械地删除相同原子序号。

## 推荐预设

### 单空位对照

```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "params": {
    "engine_type": 1,
    "num_condition": 1,
    "use_num": true,
    "concentration_condition": 0.01,
    "count_mode": "fixed",
    "max_structures": 20,
    "use_seed": true,
    "seed": 42
  }
}
```

### 5% 整体空位覆盖

```json
{
  "class": "VacancyDefectCard",
  "check_state": true,
  "params": {
    "engine_type": 1,
    "num_condition": 1,
    "use_num": false,
    "concentration_condition": 0.05,
    "count_mode": "fixed",
    "max_structures": 50,
    "use_seed": true,
    "seed": 42
  }
}
```

## 推荐组合

- `Super Cell` → `Global Random Vacancy`：先扩胞，再生成空位，降低周期镜像间的缺陷相互作用。
- `Global Random Vacancy` → 几何优化或 DFT：本卡输出未弛豫，需要后续计算得到真实缺陷响应。
- 需要按元素或子晶格删位时，改用 `Random Vacancy`，不要把两张卡串联来模拟一条规则。

## 常见问题

**为什么多元素结构的组分会变化？**

所有原子都是候选位点。若需要只删除 O、某种金属或某个 `group`，使用 `Random Vacancy`。

**为什么实际输出少于设置值？**

卡片会删除重复的空位组合。当结构很小或空位数接近原子数时，唯一组合数可能不足。

**为什么比例设置后提示删不到一个原子？**

比例采用向下取整。对 `N` 个原子的结构，至少需要 `1/N` 才能删除一个原子。

**Sobol 和 Uniform 控制什么？**

它们只控制空位落点的抽样顺序。通常使用 Uniform；需要准随机覆盖且原子数不超过 21,200 时再选 Sobol。

## 输出标签

`Vac(n={删除原子数})`

## 兼容性

旧工作流中的 `card_name`、`class` 和参数键保持不变；旧版顶层参数与当前 `params` 对象都可读取。
