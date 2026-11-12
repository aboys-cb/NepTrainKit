# 命令行与脚本运行工作流

在「生成数据集」页把工作流导出为 JSON 后，可以不开图形界面直接复现整条流程。适合批量重跑、
写进流水线、或者在服务器上生成训练集。

## 前提

工作流必须是当前版本导出的 JSON，也就是每张卡片带有 `params` 对象。旧版本导出的扁平格式
（参数直接写在卡片顶层）不支持命令行读取，需要先在界面里打开并重新导出一次。遇到这种文件时
命令行会明确报错，不会静默按默认值运行。

## 先看会做什么

执行前建议先 `preview`。它只解析 JSON，不产生任何结构：

```bash
nepkit workflow preview flow.json
```

输出示例：

```
workflow: flow.json
saved by: 3.0.1 (schema 2)
cards: 5

   1. SuperCellCard    structure
   2. CardGroup        group
   3. CellStrainCard   structure
   4. CellScalingCard  structure
   5. PerturbCard      structure

input: required
no structures were produced; use 'run' to execute
```

加上 `-i` 可以看到输入结构数量：

```bash
nepkit workflow preview flow.json -i input.xyz
```

如果工作流是生成型的（例如原型构建卡），`input` 会显示 `not required`，此时 `run` 不需要 `-i`。

## 执行并导出

```bash
nepkit workflow run flow.json -i input.xyz -o output.xyz
```

默认导出**全部卡片**的输出，也就是每一级的结果都会写进同一个文件。这和在界面里用
「查看全部输出」导出的范围一致。

按需只导出某张卡片的结果，用 `--card` 指定卡片类名，可以重复多次：

```bash
nepkit workflow run flow.json -i input.xyz -o only_supercell.xyz --card SuperCellCard
nepkit workflow run flow.json -i input.xyz -o two_stages.xyz \
    --card SuperCellCard --card PerturbCard
```

卡片类名就是 JSON 里的 `class` 字段，和界面中卡片对应的类一致。

输出格式为 extxyz，与界面导出保持一致。`-q` 只打印错误，适合放进脚本。

## 输出分两路

结果摘要走标准输出，错误走标准错误，脚本可以直接分开读：

```
$ nepkit workflow run flow.json -i input.xyz -o out.xyz
SuperCellCard        1 in       1 out
CellStrainCard       1 in      23 out
total: 1 in, 24 out
written: out.xyz
```

每张卡一行，按执行顺序排列，`in` 是这张卡收到多少结构，`out` 是它产出多少。这样能直接看出
哪一步在扩增、哪一步在过滤：

```
SuperCellCard         1 in       1 out
CellStrainCard        1 in    1331 out
CardGroup             1 in    1381 out
PerturbCard        1381 in   69050 out
```

最后一行的 `total` 里，输入数是整条流程的输入，输出数是全部卡片输出之和，通常大于最后一级；
只想看某一级时用 `--card`。

输出只用 ASCII 字符，所以在 Windows 中文终端（cp936）里不会乱码，也可以直接按空白列拆开：

```bash
nepkit workflow run flow.json -i input.xyz -o out.xyz | awk '$3=="in" {print $1, $5}'
```

默认情况下标准错误只会出现错误信息，不会混入库内部的计时和调试日志——即使界面里把日志级别
设成了「调试」也一样，因为命令行不以界面设置为准。

需要看库内部的诊断信息（例如某张卡的耗时）时加 `-v`：

```bash
nepkit workflow run flow.json -i input.xyz -o out.xyz -v
```

此时标准错误会出现类似这样的内容：

```
2026-11-05 16:05:01 | DEBUG | NepTrainKit.utils:wrapper:35 - Function 'descriptors' executed in 0.0017 seconds
```

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 |
| `2` | JSON 无法读取、卡片类未知、输入文件不存在，或某张卡片执行失败 |

任何一张卡片失败都会立即中止整条流程，并在错误信息里指明是哪张卡，不会输出半成品结果。

## 在 Python 脚本里调用

界面和命令行的能力，脚本里都能用。三个入口按需要选一个：

| 用途 | 入口 |
| --- | --- |
| 一次跑完，拿到结果 | `run_workflow(payload, structures)` |
| 先看计划、再决定是否执行 | `load_workflow(payload)` 返回的 `WorkflowRun` |
| 同一份流程反复跑 | 复用一个 `WorkflowRun`，省去重复解析 |

`payload` 可以是 JSON 文件路径、JSON 文本，或已经解析好的 dict。

### 最小例子

```python
from ase.io import read
from NepTrainKit.core.workflow import run_workflow
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export
from ase.io import write

# 读入结构，执行整条流程，取最后一级结果
structures = read("input.xyz", index=":")
result = run_workflow("flow.json", structures)
final = result.final()

print(len(final), "structures")
write("out.xyz", [prepare_magnetic_extxyz_export(a) for a in final], format="extxyz")
```

`run_workflow` 只负责计算，**不写文件**。要保存时自己写，这样结果可以直接交给后续处理，不必
先落盘再读回来。写 extxyz 时用 `prepare_magnetic_extxyz_export` 包装一下，磁矩信息才会正确
写成 extxyz 的标准字段，和界面导出的文件一致。

### 先看计划再执行

```python
from NepTrainKit.core.workflow import load_workflow

run = load_workflow("flow.json")

# 每一步：序号、卡片类名、显示名、类型（structure / dataset / generator / group / fork）
for stage in run.preview():
    print(stage.index, stage.card_class, stage.kind)

if run.requires_input():
    print("这条流程需要输入结构")
```

`preview()` 只解析 JSON，不执行任何卡片。生成型流程（例如原型构建卡）的 `requires_input()`
是 `False`，`run()` 时可以不传结构。

### 按卡片取结果

`WorkflowResult` 保留每一级的输出，不是只有最后一级：

未显式合并的永久分叉必须是流程的最后一个节点，其 `result.final()` 为空，因为它没有共同输出。各分支结果仍可按执行序号通过 `result.outputs(index=...)` 读取；需要连接共享下游时，先在分叉中启用 Merge。分叉内各分支接收同一输入，分支内部按卡片次序传递结果。

```python
result = run.run(read("input.xyz", index=":"))

result.all()                      # 全部卡片输出拼接在一起（等同命令行默认导出）
result.final()                     # 最后一级

result.outputs("SuperCellCard")    # 按卡片类名
result.outputs(index=2)            # 按执行序号，从 1 开始

# 遍历每一级，看清楚数据怎么扩增的
for output in result.card_outputs:
    print(output.index, output.card_class, output.input_count, "->", len(output))
```

指定了不存在的卡片或序号时抛 `WorkflowError`，错误信息里会列出所有可用的级别。

### 只导出某张卡

想复现 `--card` 的效果，取那一级的结果自己写：

```python
from ase.io import write

structures = read("input.xyz", index=":")
result = run_workflow("flow.json", structures)

write("only_supercell.xyz", result.outputs("SuperCellCard"), format="extxyz")
```

### 批量重跑

复用同一个 `run` 对象，避免每次都重新解析 JSON：

```python
run = load_workflow("flow.json")

for path in ("a.xyz", "b.xyz", "c.xyz"):
    structures = read(path, index=":")
    print(path, len(run.run(structures).final()))
```

`run.run()` 每次调用都是独立的，不会累积上一次的结果。

### 在自己的脚本里处理错误

`WorkflowError` 覆盖 JSON 格式、未知卡片、取不到结果这几类问题；某张卡执行失败时抛的是
`CardOperationError` 子类（例如模型文件不存在是 `FileNotFoundError`）。要一次抓住所有情况：

```python
from NepTrainKit.core.workflow import WorkflowError, load_workflow
from NepTrainKit.core.cards.errors import CardOperationError

try:
    result = load_workflow("flow.json").run(read("input.xyz", index=":"))
except (WorkflowError, CardOperationError) as exc:
    print("workflow failed:", exc)
```

## 与界面执行的差别

命令行复现的是同一套卡片逻辑，但以下几点不同：

- **不读取软件设置**。界面里的「排序原子」等全局开关不会影响命令行结果。
- **不写运行状态**。工作流库只保存配置，命令行同样不产生 `run_outcome` 之类的运行记录。
- **失败即中止**，与界面一致；不会跳过错过的卡片继续执行。

如果命令行结果和界面不一致，先确认两边用的是同一个导出的 JSON，并检查是否依赖了界面里的
全局设置。

## 换机器运行时的模型路径

模型路径分两种情况，行为不同。

**软件自带的模型（`nep89`）是可移植的。** 导出时不会写死某台机器上的绝对路径，而是写成标记
`bundled:nep89`：

```text
"params": { "nep_path": "bundled:nep89", ... }
```

运行时会解析成当前机器上真实的模型文件，所以在别的机器、别的安装位置或换了 Python 环境都能
直接跑。界面上显示的仍然是完整路径，标记只存在于数据里。

**你自己选的模型文件保留绝对路径。** 这类文件只存在于你指定的位置，软件无法替你在别的机器上
找到它，所以原样记录：

```text
"params": { "nep_path": "/opt/models/my_custom.nep", ... }
```

换机器运行这类工作流时，需要把 `nep_path` 改成目标机器上的真实路径。路径不存在时命令行会明确
报出找不到的文件：

```
error: NEP file does not exist: /opt/models/my_custom.nep
```

旧版本导出的 JSON 里 `nep_path` 是绝对路径，这类文件仍然可以正常加载；如果你希望它在其他机器
上也能用，把它改成 `bundled:nep89` 即可。

需要知道当前机器上自带模型的真实路径时：

```bash
python -c "from NepTrainKit.paths import get_bundled_nep89_path; print(get_bundled_nep89_path())"
```
