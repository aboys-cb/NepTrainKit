# 训练集评估最终设计

## 状态与权威性

本文是 NepTrainKit 训练集检查功能的权威产品与实现基线。

- 状态：已确认方向，按阶段实施。
- 生效日期：2026-07-12。
- 取代下列文档中的产品定位、页面结构和实施优先级：
  - `2026-07-09-training-set-audit-design.md`
  - `2026-07-10-training-set-audit-dashboard-redesign.md`
- 旧文档仍可用于追溯历史实现，不再作为新增功能的验收依据。
- 后续每完成一个阶段，必须在本文的“实施进度”中更新状态、验证命令和剩余风险。

## 一句话产品定义

训练集评估帮助用户区分三件事：

1. 当前数据有没有技术性阻塞或需要复核的问题。
2. 当前数据对用户声明的目标场景覆盖到了哪里、证据有多强。
3. 当前势函数在独立参考数据和目标任务上哪里可靠、哪里仍有风险。

没有目标定义时，软件只能给出数据内部证据，不能声称“覆盖充分”或“缺少某类物理”。
没有模型预测与独立参考时，软件不能声称势函数可靠，也不能把标签尾部冒充模型误差。

## 为什么要重新定位

当前实现已经具备有价值的证据能力：

- 组成分布。
- 标签范围和尾部复核集合。
- 基于 NEP cutoff 的局域化学分布。
- 元素对接触统计。
- 分布图与结构反向选择。
- `Observed / Interpretation / Limit` 证据表达。

但当前能力仍是“当前活动数据子集上的相对统计”，不是完整的 readiness audit：

- 审计范围没有成为一等信息。
- 低频不等于风险，高频不等于冗余。
- 固定分位尾部必然存在，不是异常检测。
- `nep.txt` 目前只提供 cutoff，没有参与误差或不确定性验证。
- GUI 在页面内重新组织底层 slices，HTML 仍直接显示底层 severity，存在两套语义。
- 独立的 Distribution Inspector 和 Audit 详情页职责重叠，用户需要在两个工具间来回切换。

因此，不继续堆图和指标。先收敛产品承诺、核心结果和用户动作。

## 用户能看到的最终效果

### 总入口

`NEP Dataset Display` 只保留一个主入口：

```text
检查当前数据集
```

打开独立的“训练集评估”页面。页面顶层有三个模式：

```text
┌ 快速体检 ───────┬ 目标覆盖 ───────┬ 模型验收 ───────┐
│ 无需目标和模型   │ 需要目标定义     │ 需要模型与参考   │
└─────────────────┴─────────────────┴─────────────────┘
```

未满足输入条件的模式显示“尚未配置”，而不是输出空图、假分数或红色告警。

### 页面首屏

首屏先回答问题，再给指标：

```text
训练集评估
当前活动数据 1,000 / 原始数据 19,313
数据指纹 · 范围 · 模型状态 · 规则版本

当前结论
未发现阻止训练的数据格式或标签问题；有 2 组结构建议人工复核。
这不代表目标物理已覆盖，也不代表势函数可靠。

需要处理        2
数据阻塞        0
待复核结构      137
模型误差        未评估

下一步
1. 检查 4 个疑似重复标签冲突结构
2. 查看最短原子距离异常的 12 个结构
3. 如需判断目标覆盖，请添加目标场景
```

首屏不显示 `High / Medium / Low` 严重度墙，不把低频 bin 数量当成问题数量。

## 三种模式

### 1. 快速体检

输入：当前数据集。目标与模型均可缺省。

回答：

- 数据能否被正常读取和用于训练流程。
- 是否存在明确的数据质量阻塞。
- 哪些结构值得人工复核。
- 当前数据内部有哪些分布线索。

#### 数据阻塞

第一阶段支持：

- 非有限数值：位置、晶胞、能量、力、virial/stress 中的 `NaN / Inf`。
- 标签形状错误：原子数与力/磁力数组不一致，结构级标签形状异常。
- 非法或退化晶胞：非周期数据除外；周期方向和晶胞不一致必须明确报告。
- 元素或原子序号异常。
- 明显重叠或异常短原子距离。
- 完全重复结构。
- 相同几何但标签冲突。
- train/validation/test split 泄漏：仅在 split 元数据存在时检查。

阻塞项必须给出：事实、受影响结构、判定规则、建议动作和限制。

#### 复核集合

以下内容只能作为 review set，不得自动判错或建议删除：

- 高力尾部。
- 高能量尾部。
- 极端体积或密度。
- 稀有 `config_type`。
- 低频组成或局域环境。
- 元素对接触不足。

固定分位选择必须显示阈值和选择比例，并明确说明“该集合按排名产生，不是异常检测结果”。

#### 数据分布探索

快速体检包含统一的“分布探索”工作区，吸收现有 Distribution Inspector 的能力：

- 任意可发现的数值字段。
- Reference / Prediction / Error 视图按可用性出现。
- Active / Selected 范围。
- Overall / Formula / Element 分组。
- 标量、向量分量和向量范数。
- 直方图及按 bin 反向选择结构。
- 从某个 Finding 一键打开对应字段和范围。

默认界面只显示 `字段 / 视图 / 分组 / 范围`。bin 数量、曲线类型和范数放入“高级选项”，避免让用户先理解绘图参数。

### 2. 目标覆盖

输入：目标定义必需；候选池或目标轨迹可选。

回答：

- 用户希望势函数用于什么场景。
- 当前训练数据在可观测物理空间中覆盖了哪些目标区域。
- 哪些目标区域只有弱证据或没有样本支持。
- 如果提供目标轨迹，哪些局域环境相对训练集新颖。

目标定义采用小表单，不使用长向导。最小字段：

- 任务模板：体相/声子、缺陷/扩散、塑性/断裂、表面/界面、液体/高温、碰撞、磁性、自定义。
- 元素与组成范围。
- 温度范围。
- 压力或应变范围。
- 关键结构类型或 `config_type`。
- 关注性质。

证据等级：

- 仅目标描述：只能检查显式 metadata 和可观测量范围。
- 目标结构或候选池：可以做目标与训练集的直接比较。
- 目标轨迹：可以做时间序列中的 descriptor novelty 和暴露频率分析。

候选池不是使用快速体检的前提，也不是所有用户必须准备的输入。

### 3. 模型验收

输入：模型预测与独立参考数据必需。目标任务可选，但给出“面向目标的可靠性结论”时必需。

回答：

- 能量、力、virial/stress、磁力等误差。
- 按 `config_type`、组成、元素、标签范围和目标切片的误差。
- 误差是否集中在某些结构家族或物理区域。
- 可选的模型集成分歧、descriptor novelty 或校准不确定性。
- 经明确协议定义的性质验证与 MD smoke test。

模型验收遵守：

- 训练误差不冒充独立验证误差。
- 一个总 RMSE 不冒充目标可靠性。
- 排名指标不冒充校准不确定性。
- 物性验证按独立协议实现，不建立一个含糊的“通用物性分数”。

## Finding：用户真正消费的核心结果

GUI、HTML 和未来 CLI 必须消费同一份 canonical Finding。Qt 页面不得再创建另一套产品结论。

每个 Finding 包含：

```text
Finding
  id
  kind: blocker | review | evidence | unavailable
  title
  conclusion
  observed
  rule
  limit
  structure_indices
  target_relevance: unknown | low | medium | high
  confidence: direct | derived | heuristic
  actions[]
  evidence_ids[]
  state: open | reviewed | accepted | resolved
```

规则：

- `blocker` 只用于违反明确数据契约的事实。
- `review` 表示需要人判断，不默认删除。
- `evidence` 是分布或比较事实，不带风险结论。
- `unavailable` 说明缺少输入，不能伪造结果。
- `target_relevance=unknown` 是合法且常见的状态。
- 第一阶段不计算一个汇总“风险分数”。

## 核心模块与接口

外部 seam 保持一个深模块接口：

```python
build_audit(context: AuditContext) -> AuditRun
```

调用者只需要构造 context，然后消费不可变结果。

```text
AuditContext
  dataset
  scope
  dataset_identity
  optional target_profile
  optional model_evidence
  ruleset_version

AuditRun
  generated_at
  scope
  fingerprints
  modes
  summary
  findings[]
  evidence[]
  plots[]
```

内部检查器可以按文件拆分，但第一阶段不公开 `CheckPlugin` 接口。只有出现第二个真实实现或外部提供者时，才建立 adapter seam。

快速体检中的晶胞有效性与短距离检查通过一个窄的批处理几何接口完成：Python 负责数据契约、规则和 Finding，原生 `NepTrainKit._native._audit` 扩展只接收连续的坐标、结构偏移、晶胞、PBC 与 cutoff，返回晶胞状态和每个结构是否存在短原子对。正交与非正交晶格使用相同的周期最短镜像语义；扩展不可用、晶胞接近数值秩阈值，或遇到合法的奇异部分周期晶胞时，使用同一接口下的 NumPy/SciPy 参考实现。两条路径必须由同一组行为测试约束。不得把审计结论、阈值政策或 UI 文案下沉到 C++。

### 审计范围

`AuditScope` 必须明确：

- `all`：原始全部结构。
- `active`：当前未删除结构。
- `selected`：当前选中结构。
- `custom`：调用者明确给出的 indices。
- 原始结构数、范围内结构数和原始 indices。

默认从 Dataset Display 进入时使用 `active`，首屏必须显示 `范围内数量 / 原始数量`。

### 指纹

每次 AuditRun 记录：

- 数据来源路径（如有）。
- 数据快照指纹。
- 活动范围指纹。
- 模型文件 SHA256（使用模型时）。
- 目标定义指纹（使用目标时）。
- ruleset version。

指纹用于识别结果是否过期，不用于给数据质量打分。

## Distribution Inspector 替代方案

### 决策

替代旧 UI，保留并深化其核心计算能力。

现有 `DistributionRequest`、字段发现、分组统计和 bin 到结构映射是有价值的实现，不应重写。旧的 `DistributionInspectorMessageBox` 是浅层独立入口，最终由训练集评估页面中的“分布探索”工作区替代。

### 迁移门

旧入口只有在下列能力全部通过测试后才删除：

- [x] 可选择任意数值字段。
- [x] Reference / Prediction / Error 可用性与旧工具一致。
- [x] Active / Selected scope 一致。
- [x] Formula / Element 分组一致。
- [x] 向量分量和 norm 一致。
- [x] 点击 bin 能得到相同的原始结构 indices。
- [x] Replace / Add / Intersect 选择行为一致。
- [x] 大数据分析继续复用原有 worker callback，不阻塞 UI。
- [ ] 文档和截图已切换到新入口。

迁移顺序：

1. Audit 页面接入现有分布核心。
2. Audit Finding 可深链到分布探索预设。
3. 工具栏 Dist 图标改为打开 Audit 的分布探索工作区。
4. 观察一个版本并保留兼容入口。
5. 删除 `DistributionInspectorMessageBox` 及只为它存在的 UI glue；保留核心分布模块和测试。

## 动作与工作流边界

Finding 的动作只允许调用已有工作流：

- 在 Dataset Display 中查看结构。
- 选择、追加选择或求交选择。
- 导出结构子集。
- 标记为已复核、接受或解决。
- 保存 HTML 报告。

第一阶段不自动删除数据，不自动生成 DFT 作业，不自动重训练。

未来可以把明确的缺口发送给 Make Dataset、DFT 或 Agent，但 NepTrainKit 的评估页面不负责长时间任务编排。

## 实施阶段

### Phase 0：锁定产品和 seam

- [x] 完成最终产品定义。
- [x] 明确三种模式和声明边界。
- [x] 明确 Distribution Inspector 采用能力迁移而非直接删除。
- [x] 在代码中建立 canonical `AuditRun / Finding / Evidence / AuditScope`。
- [x] GUI 和 HTML 只消费 canonical Finding。

验收：同一运行在 GUI 和 HTML 中具有相同 Finding 数量、分类、标题、结构 indices 和限制文本。

### Phase 1：快速体检可信化

- [x] 显示明确 scope、原始数量、活动数量和规则版本。
- [x] 加入数据与范围指纹。
- [x] 实现非有限数值、标签形状、晶胞、元素、短距离检查。
- [x] 实现完全重复和标签冲突检查。
- [x] 将固定分位尾部降级为 review set。
- [x] `config_type` 成为 evidence；模型误差切片留到 Phase 4。
- [x] 更新 HTML 报告和测试。

验收：无模型、无目标时可完成；所有 blocker 可由小 fixture 确定性复现；没有“覆盖充分”或“势函数可靠”表述。

### Phase 2：统一分布探索并替代 Dist UI

- [x] 在 Audit 详情页嵌入分布探索工作区。
- [x] 复用现有 `DistributionRequest` 计算，不复制算法。
- [ ] 实现 Finding 到字段/视图/范围的深链。
- [ ] 通过全部迁移门。
- [x] 切换工具栏入口。
- [ ] 删除旧 Dialog 和只为它存在的 glue。
- [x] 更新用户文档并移除旧窗口截图入口；新页面截图待视觉验收后补充。

验收：核心分布结果与迁移前 fixture 完全一致；用户不再需要在两个工具间切换。

### Phase 3：目标覆盖 MVP

- [ ] 实现最小目标表单和目标指纹。
- [ ] 用 `config_type`、组成、标签范围和结构 metadata 做首批目标比较。
- [ ] 可选接入目标结构或候选轨迹。
- [ ] 先以独立脚本验证 descriptor novelty，再进入 GUI。

验收：没有目标时该模式不做 coverage 声明；仅目标描述和有目标结构时显示不同证据等级。

### Phase 4：模型验收 MVP

- [ ] 复用现有 reference/prediction/error 数据。
- [ ] 实现结构级和原子级误差汇总。
- [ ] 按 `config_type`、组成、元素和目标切片。
- [ ] 模型文件与结果数据进入过期检测。
- [ ] 可选不确定性只在真实输入存在时出现。

验收：训练集标签尾部与模型误差严格分开；没有预测时明确显示“未评估”。

### Phase 5：性质协议与闭环入口

- [ ] 逐个实现有明确 oracle 和容差的性质协议。
- [ ] 加入最小 MD smoke test。
- [ ] 将缺口作为结构化任务发送给外部工作流。

该阶段不作为替代 Distribution Inspector 或完成快速体检的前置条件。

## 测试策略

核心测试穿过 `build_audit(context) -> AuditRun` seam，断言可观察结果，不依赖内部 checker 文件划分。

必须覆盖：

- `all / active / selected / custom` scope 映射。
- 数据变化、范围变化、模型变化导致指纹变化。
- 每类 blocker 的阳性和阴性 fixture。
- review set 不被标记为 blocker。
- GUI 与 HTML 使用相同 Finding。
- 分布迁移前后结果和反向结构 indices 一致。
- 所有输出使用临时目录，不污染 `tests/data`。

验证顺序：

1. 最小核心测试。
2. Audit 页面与 HTML 测试。
3. Distribution 迁移契约测试。
4. 邻近 Show NEP / selection / canvas 测试。
5. `git diff --check`。
6. 风险足够高时运行完整 `pytest -q`。

## 明确不做

- 不输出没有分母定义的 coverage 百分比。
- 不把低频、极端值、模型分歧直接等价为错误。
- 不用一个聚合分数替代证据。
- 不把候选池设为快速体检的必需输入。
- 不增加 SOAP/ACE 等重量级必需依赖。
- 不在 Qt 页面复制审计算法。
- 不在评估页面复制 3D 结构查看器。
- 不把长时间 DFT、训练和 HPC 编排塞进本模块。

## 实施进度

### 当前状态

- 2026-07-12：完成最终产品定位和 Dist 替代决策。
- 2026-07-12：完成 Phase 0；`AuditContext → AuditRun` 已记录 scope、指纹、规则版本和 canonical Findings，GUI 与 HTML 共用结果。
- 2026-07-12：完成 Phase 1 快速体检；加入明确的数据 blocker、重复/标签冲突、`Config_type` evidence 和保守 review set。
- 2026-07-12：Distribution Explorer 已嵌入统一页面，工具栏入口已切换；旧 Dialog 只作为一个版本的兼容壳保留。
- 2026-07-12：局域化学改用 `_native._audit` 分块返回通用 cutoff 邻居，并在原生层计算无政策含义的 typed neighbor/contact aggregates；Local Chemistry 和 Pair Contacts 的阈值、Finding 与解释仍留在 Python。FeNi 55,985 帧局域化学阶段先从 47.802 秒降至 6.366 秒；对已经达到“多结构支持”的元素对停止回传不再用于 Finding 的逐边距离明细后，3 次中位数进一步降至 3.310 秒。文件型 NepTrainKit 数据集的结构指纹改为源文件内容哈希与单调结构版本组合，通用调用仍保留逐结构内容哈希回退；指纹中位数从 1.801 秒降至 0.082 秒，完整首次 Audit 的 3 次中位数从最初约 52.6 秒降至 5.750 秒。正交、非正交和部分周期路径均与 Python/SciPy 参考对齐。未变化的数据范围、源文件和模型再次进入时复用上一次 AuditRun，不再重新计算；页面“重新检查”仍强制刷新。
- FeNi 真实数据验证：55,985 帧的 Python/SciPy 数据质量检查 3 次中位数为 11.435 秒，其中短距离检查占 10.875 秒。批处理 `_native._audit` C++/OpenMP 路径与 Python 参考结果完全一致；进一步把逐结构 NumPy `matrix_rank/SVD` 晶胞检查合并到原生批次后，7 次端到端中位数为 1.341 秒。标签有限值检查优先复用 `dataset.energy / dataset.virial / _force_vector_dataset` 的 reference 数组后，同轮逐结构/批量中位数为 1.389/1.245 秒，结果一致；相对初始版本整体约提升 9.19 倍。原生晶胞状态扫描为 0.000163 秒，Python/SVD 参考为 0.601429 秒；10,000 个随机正交/非正交、全周期/部分周期和退化晶胞为 0 mismatch。OpenMP 邻居 kernel 在 1/2/4/8 线程下的最终中位数分别为 0.153/0.101/0.086/0.058 秒。`float32` kernel（0.05276 秒）未胜过同轮 double（0.05243 秒），固定镜像数组/复用 scratch 版本退化至 0.0561 秒，两项实验均已撤回。最终结果无 blocker，42 个重复几何仅作为 review set。
- 2026-07-14：产品信息架构按 `training-set-audit-product-reset-v2.md` 落成四屏 Qt 页面。`DatasetInventory` 在 core 中按精确归一化组分聚合，保留超胞大小、`config_type` 和结构索引；概览、数据地图、复核队列、目标与模型均复用同一 `AuditRun`。目标比较当前只支持组成范围、离散点和显式最低结构数，状态措辞限定为“数量规则已满足/数量偏少/没有精确样本”，不声称物理覆盖。真实 FeNi 复验得到 55,985 帧、33 个精确组分点、纯 Fe 596、纯 Ni 943、Top 3 组分占 81.9%，42 个结构属于 14 组重复几何。

### 下一执行点

下一步只推进一个最高收益闭环：把复核状态和简单目标写入可追溯快照，并让 HTML 报告复用同一状态。模型误差、目标轨迹和局部偏聚在该闭环完成前不继续扩展。
