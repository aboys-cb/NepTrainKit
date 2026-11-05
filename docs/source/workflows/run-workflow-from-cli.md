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
input: 1 structure(s) from input.xyz          ← stdout
  SuperCellCard: 1
  CellStrainCard: 23
total: 24 structure(s)
written: out.xyz
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

```python
from NepTrainKit.core.workflow import load_workflow
from ase.io import read

run = load_workflow("flow.json")

# 只看执行计划，不运行
for stage in run.preview():
    print(stage.index, stage.card_class, stage.kind)

result = run.run(read("input.xyz", index=":"))

# 全部卡片输出
result.all()

# 只取最后一级
result.final()

# 只取某张卡片（类名），或按执行序号（从 1 开始）
result.outputs("PerturbCard")
result.outputs(index=1)

# 每张卡各自的输出，键为卡片类名
for output in result.card_outputs:
    print(output.index, output.card_class, len(output.structures))
```

`run()` 接受任意可迭代的结构对象，返回结果对象本身不写文件；写文件由调用方决定，这样便于
先把结果交给后续处理，而不是先写入磁盘再读取。

批量重跑时复用同一个 `load_workflow` 结果可以省去重复解析：

```python
run = load_workflow("flow.json")
for path in ("a.xyz", "b.xyz", "c.xyz"):
    structures = read(path, index=":")
    print(path, len(run.run(structures).final()))
```

## 与界面执行的差别

命令行复现的是同一套卡片逻辑，但以下几点不同：

- **不读取软件设置**。界面里的「排序原子」等全局开关不会影响命令行结果。
- **不写运行状态**。工作流库只保存配置，命令行同样不产生 `run_outcome` 之类的运行记录。
- **失败即中止**，与界面一致；不会跳过错过的卡片继续执行。

如果命令行结果和界面不一致，先确认两边用的是同一个导出的 JSON，并检查是否依赖了界面里的
全局设置。

## 换机器运行时注意模型路径

导出的 JSON 里，卡片引用的模型文件写的是**导出时那台机器上的绝对路径**。默认值是软件自带的
`Config/nep89.txt`，在界面里改成自定义路径时也一样，都是具体位置。

同一台机器上直接运行没有问题。把 JSON 拿到别的机器、别的位置的安装，或者换了 Python 环境时，
这个路径通常已经失效，命令行会明确报出找不到的文件：

```
error: NEP file does not exist: /Users/xxx/miniconda3/lib/python3.10/site-packages/NepTrainKit/Config/nep89.txt
```

此时把 JSON 里的 `nep_path` 改成当前机器上的真实路径即可。自带模型的路径可以用：

```bash
python -c "from NepTrainKit.paths import get_bundled_nep89_path; print(get_bundled_nep89_path())"
```

涉及模型路径的卡片（例如全局多样性筛选）在流水线里复用时，建议导出后直接把 `nep_path`
替换成目标环境的路径，而不要依赖导出时写入的值。
