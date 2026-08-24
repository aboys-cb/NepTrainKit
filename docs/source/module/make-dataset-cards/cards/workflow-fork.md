<!-- card-schema: {"card_name": "Permanent Fork", "source_file": "src/NepTrainKit/ui/views/_card/workflow_fork.py", "serialized_keys": ["branches", "merge"]} -->

# 永久分叉（Permanent Fork）

**分类：** 工作流

## 功能说明

把同一份上游数据复制给两条或更多独立的线性分支。每条分支可以继续串联多张卡，并保留自己的运行状态和最终输出。只有启用显式 `Merge` 后，各分支结果才会按分支顺序拼接为一个共同输出。

永久分叉表达的是**数据流并行**，不是同时启动多个计算线程。当前实现依次调度各分支，避免重型卡片争抢 CPU、GPU 和内存。

它不同于 [分支合并组](card-group.md)：`Card Group` 的子卡共享一次输入并立即合并；永久分叉中的每条路径可以继续执行完整的后续卡片链。

## 工作原理

设 Fork 输入为 $D$，启用分支为 $B_1,\ldots,B_n$，每条分支内部的卡片按顺序复合为 $F_i$。永久分叉遵循以下可检查规则：

$$
O_i = F_i(D),\qquad
O_{\mathrm{merge}} = O_1 \mathbin{\Vert} O_2 \mathbin{\Vert} \cdots \mathbin{\Vert} O_n.
$$

其中 $O_i$ 是第 $i$ 条分支的独立输出，$\Vert$ 表示按界面分支顺序稳定拼接。关闭 `Merge` 时只保留各 $O_i$，不会生成共同输出；启用 `Merge` 时，只有全部启用分支成功才生成 $O_{\mathrm{merge}}$。

## 操作示例

同一个体相母结构需要分别补充表面缺陷和体相响应数据：

```text
Crystal Prototype
        │
Permanent Fork
  ├─ A: Random Slab → Card Group(Vacancy, Adsorbate) → Atomic Perturb
  └─ B: Lattice Strain → Atomic Perturb → FPS Filter
```

不启用 `Merge` 时，A、B 分别产生独立输出；适合分别送往不同的 DFT 任务或训练集审查流程。需要共同下游时，先启用显式 `Merge`，再在 Fork 后增加过滤或导出卡片。

## 参数说明

### 分支（branches）

分支列表。每条分支保存稳定的 `id`、显示名称、启用状态和内部卡片序列。分支内部按界面顺序串行传递数据；不同分支始终从同一份 Fork 输入开始。

第一版不允许在分支内部再次嵌套 `Permanent Fork`，但允许使用 `Card Group`。

### 显式合并（merge）

布尔值，默认 `false`。

- `false`：每条分支独立终止和导出；Fork 后不能再连接共同下游卡片。
- `true`：所有启用分支成功后，按分支顺序拼接输出；拼接结果成为后续卡片输入。

`Merge` 只做稳定拼接，不自动去重或筛选。需要去重、几何过滤或 FPS 时，在 Merge 后添加相应过滤卡。

## 失败与停止语义

- 未合并模式下，一条分支失败不会删除其他成功分支的输出，其他分支仍会继续运行。
- 合并模式下，任一启用分支失败都会阻止 Merge 和共同下游。
- 停止工作流会停止所有仍在运行的分支卡片；部分结果不会冒充成功输出。

## 输出标签

永久分叉本身不修改 `Config_type`。来源标签由各分支中的实际操作卡写入。

## 可复现性

永久分叉本身没有随机性。结果可复现性取决于各分支内部卡片的 seed 和输入数据。

## 常见问题

### 为什么 Fork 后不能直接添加共同下游卡片？

未启用 `Merge` 时不存在唯一的共同输出，下游卡片无法判断该接收哪条分支。请先启用显式 `Merge`，或者分别导出各分支结果。

### 它会同时占用多张 GPU 或多个 CPU 任务吗？

不会。分支在数据语义上并行，但当前版本依次调度，以避免多个重型卡片争抢计算资源。

### 什么时候应该使用 Card Group？

如果多张子卡只需共享同一份输入，并在这一层立即合并结果，使用 `Card Group`。如果每条路径还要继续经过不同的卡片链，使用永久分叉。
