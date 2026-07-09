# 设置（Settings）

`Settings` 用来调整显示、导入导出、NEP 后端和绘图样式。刚开始不需要全部改；只有当你遇到加载慢、显示卡顿、预测慢、导出格式不符合下游要求，或想切换 CPU/GPU 后端时，再来这里。

## 先改哪些

| 你遇到的问题 | 可以看哪个设置 | 影响 |
| --- | --- | --- |
| 大结构显示卡顿 | `Canvas Engine` | 在 `Auto` / `PyQtGraph` / `VisPy` 间切换结构和散点渲染后端 |
| 力图数值不符合习惯 | `Force data format` | 在原始力和归一化显示之间切换 |
| 打开目录时不想自动加载 | `Auto loading` | 控制是否自动读取启动路径里的相关结果文件 |
| 卡片处理时原子顺序影响观察 | `Sort atoms` | 处理结构时按元素顺序重排原子 |
| 卡片菜单太拥挤 | `Use card group menu` | 按卡片 `group` 分组显示 Make Dataset 菜单 |
| DeepMD 导出不想丢目录层级 | `Keep DeepMD subfolders` | 导出 `deepmd/npy` 时保留导入时的子目录结构 |
| 反复打开同一批输出较慢 | `Cache output files` | 缓存 `*.out` 和 `descriptor.out`，减少重复解析 |
| 导出小数位不合适 | `Export significant digits` | 控制 `xyz` / `extxyz` 中逐原子数值的有效数字 |
| 源文件没有 `Config_type` | `Default Config_type` | 给缺少标签的结构补默认来源标签 |
| 非物理结构检测太松或太严 | `Covalent radius coefficient` | 调整近邻距离判据 |

## NEP 后端

`NEP Backend` 控制预测时优先使用 CPU、GPU 还是自动选择。

- `Auto`：默认路径，程序优先尝试可用后端。
- `CPU`：适合没有可用 GPU、驱动不匹配，或只是少量结构快速查看。
- `GPU`：适合批量预测或较大候选池。

`GPU Batch Size` 决定一次送入 GPU 的结构数量。显存不足时调小；显存充足但吞吐低时可以调大。

`Data Precision` 控制导入 DFT/结构数据后的存储精度。常规可保持默认；只有在内存压力明显或需要保留更高精度数值时再改。

## 绘图和结构显示

Plot Settings 只影响显示，不改变底层数据。

- `Scatter edge color` / `Scatter face color` / `Face alpha`：散点默认颜色和透明度。
- `PyQtGraph scatter size` / `VisPy scatter size` / `VisPy antialias`：不同画布后端的点大小和抗锯齿。
- `Structure background` / `Lattice line color`：结构视图背景和晶格线颜色。
- `Selected color` / `Show color` / `Current marker color` / `Current marker size`：选中点、高亮点和当前结构标记的显示样式。

如果只是想判断结构是否异常，优先调整点大小和当前结构标记；不要为了配色反复改变数据筛选流程。

## NEP89 和更新

`About NEP89` 会检查并更新内置 `nep89.txt`。NEP89 可以用于某些候选结构的快速预筛，但它不是所有材料体系的可靠模型。使用时把它当作异常结构筛查工具，不要当作 DFT 标签来源。

如果你有当前体系自己的 NEP 模型，优先使用自己的模型做预筛。

`About` 里的 `Check for Updates` 用于检查 NepTrainKit 新版本。这个检查只影响软件版本提示，不会自动修改你的训练数据。

## 修改设置的原则

- 先确认问题属于显示、读取、预测、导出还是卡片流程，再改对应设置。
- 一次只改一个关键选项，方便判断影响。
- 涉及数据清洗阈值时，改完要抽查被选中的结构，而不是只看数量变化。
