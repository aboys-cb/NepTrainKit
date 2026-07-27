---
name: make-dataset-card-dev
description: 设计、审查、实现和维护 NepTrainKit 的 Make Dataset 卡片。适用于逐卡审核功能边界、控件、参数、默认值、轴/参考系、文案、翻译和功能重复，也适用于把需求或脚本新增/迁移为卡片、修复卡片缺陷、优化预览性能，并按 Operation/Params 架构完成序列化、文档和严格测试。
---

# Make Dataset Card Dev

## 目标

把需求文本、已有脚本或现有卡片重构需求稳定转换为可交付卡片：

1. 业务逻辑在 `src/NepTrainKit/core/cards/`，不依赖 PySide、qfluentwidgets 或 MessageManager。
2. UI 只负责控件、参数读写、序列化和调用 operation。
3. `Params dataclass`、operation、UI、文档、测试保持同一套参数契约。
4. 注册、在线文档路径、测试和文档审计闭环。
5. 卡片先通过用户场景、功能边界和参数设计审查，再进入实现。

## 协作规则

先从需求本质确认卡片类型，再动代码。以下情况先和用户确认：

1. 需求表述存在多种合理解读。
2. 方案会改变已有行为、默认值或旧 JSON 兼容性。
3. 需要引入新依赖、跨页改运行时框架，或影响已有工作流。
4. 存在明显更短或更稳的实现路径。

当存在更优路径时，先给出“当前方案 vs 替代方案”的简短对比（收益、代价、风险），由用户拍板后再实施。

## 先判定卡片语义

不要强行把所有卡片塞进同一个接口。先选 operation 类型：

- `StructureOperation`: 单结构变换，签名为 `run_structure(structure, params) -> list[Atoms]`。
- `DatasetOperation`: 全数据集过滤或排序，签名为 `run_dataset(dataset, params) -> list[Atoms]`。
- `GeneratorOperation`: 无输入生成结构，签名为 `generate(params) -> list[Atoms]`。

对应参数必须用 frozen dataclass 表达，例如 `FooParams`。UI 通过 `get_params()` 构造 dataclass，通过 `set_params(params)` 恢复控件。

## 工作流

### 0. 先做卡片设计审查

新增卡片和逐卡重构都先读 `references/card-design-review.md`，从用户使用和材料设计角度检查：

- 使用场景、输入前提、输出语义和功能边界是否一句话说清；
- 是否与相邻卡片功能重复，是否把生成、过滤、标签等不相关职责塞在一起；
- 每个参数是否必要，名称、单位、参考系、默认值、范围、精度和禁用语义是否合理；
- 方向参数是否区分笛卡尔轴、晶格轴、Miller 面、滑移方向和局部法向，是否不必要地限制任意取向；
- 控件联动、渐进披露、输出数量预览是否顺手且不会留下“可编辑但不生效”；
- 查找卡片弹窗是否有一句简述，新增卡片下拉框、分组、枚举和提示是否完整翻译；
- 文档是否准确说明原理、限制和适用场景，而不是替未实现的自动识别或物理能力背书。

先输出“发现 / 用户影响 / 建议 / 是否改变旧行为”的短表。纯文案、翻译、明显死参数和无歧义控件联动可直接修；会改变物理语义、默认值或旧 JSON 时再询问用户。

### 1. 需求/脚本解析为卡片规格

先产出一个“卡片规格草案”，至少包含：

- `card_name`、`group`、`menu_icon`、`requires_input_dataset`
- 面向查找弹窗的一句简述
- operation 类型：`StructureOperation` / `DatasetOperation` / `GeneratorOperation`
- `Params dataclass` 字段清单：名称、类型、默认值、范围、是否必填
- 处理逻辑摘要：输入对象 -> 输出结构列表
- 随机性与 seed 策略
- `Config_type` 标签策略
- 与相邻卡片的边界和不重复理由
- 方向/晶面/层操作使用的参考系和任意取向支持边界
- 输出数量合同：精确数量、最多数量、过滤子集或允许失败

如果是脚本输入，额外提取：

- 脚本里的硬编码常量，改为 Params 字段和 UI 参数。
- 文件 I/O 副作用，改为内存数据流；不要覆盖用户文件。
- 随机数入口，明确是否暴露 `use_seed` 和 `seed`。

### 2. 实现 core operation

优先在已有模块里放置逻辑：

- 晶格/结构变换：`core/cards/lattice.py` 或 `core/cards/structure.py`
- 随机/组成/替换：`core/cards/alloy.py`
- 缺陷/表面：`core/cards/defect.py`
- 磁性：`core/cards/magnetism.py`
- 数据集过滤：`core/cards/filter.py`

实现规则：

- operation 不导入 `PySide6`、`qfluentwidgets`、`MessageManager`。
- 参数校验失败时抛出明确异常；UI 层负责展示错误。
- 结构变换后使用 `append_config_tag(...)` 写入可追溯标签。
- 不做静默物理替换、降级模型或伪成功返回；unsupported 就明确失败。
- 若需要序列化 dataclass，用 `params_to_dict(params)`；tuple 字段在 UI `to_dict()` 中按文档契约转成 list。

### 3. 实现 UI 卡片

UI 类放在 `src/NepTrainKit/ui/views/_card/*.py`，遵循现有风格：

- 新卡片默认继承 `MakeDataCard`。不要因为 operation 是 dataset 级别就新建 `FilterDataCard` 子类；当前 `FilterDataCard` 只保留给既有过滤卡的显示差异。
- `init_ui()` 构建控件。
- 数值参数用 `SpinBoxUnitInputFrame`，枚举用 `ComboBox`，开关用 `CheckBox` / `RadioButton`，字符串用 `LineEdit`。
- 提供 `create_operation()`、`get_params()`、`set_params(params)`。
- 所有卡片禁止覆盖 `run()`：基类 `MakeDataCard.run()` 已根据 `create_operation()` 返回的 operation 类型自动分发到正确线程。
- `process_structure()` 若保留，只能作为兼容委托层。结构卡调用 `run_structure(...)`；dataset/generator 卡片不要新增伪 `process_structure()` 通路。
- 生成型卡片使用 `GeneratorOperation` 和 `requires_input_dataset = False`。

### 4. 完成序列化

绑定必须完整：

- `to_dict()` 写入 `"params": params_to_dict(self.get_params())`。
- 保留必要的旧 key，保证旧 JSON 可加载。旧 key 双写是过渡态；新增持久化格式时再引入版本字段清理。
- `from_dict()` 优先读 `params`，没有时按旧 key 构造 Params，再调用 `set_params(params)`。
- 文档里的 `serialized_keys`、默认值和运行时 `to_dict()` 一致。

### 5. 注册、文档和在线路径

- 类上加 `@CardManager.register_card`。
- 在 `src/NepTrainKit/ui/views/_card/__init__.py` 导入并加入 `__all__`。
- 每张内置卡片必须有对应文档：
  - 源码：`src/NepTrainKit/ui/views/_card/foo_bar_card.py`
  - 文档：`docs/source/module/make-dataset-cards/cards/foo-bar-card.md`
  - 在线链接：`https://neptrainkit.readthedocs.io/en/latest/module/make-dataset-cards/cards/foo-bar-card.html`

**文档写作核心原则：从训练集诊断出发。**

操作示例必须回答"模型哪里不行 → 训练集缺什么 → 这张卡怎么补 → 怎么验证改善"，而不是只写"设参数→得结果"。禁止以下内容：

- 模板填充句："先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时再偏离"
- 同义反复开关建议："需要启用 XXX 时开启 / 希望保持默认时关闭"
- 把 `params`（序列化实现细节）作为用户参数列出
- 三档预设 JSON 几乎一样（Safe/Balanced/Aggressive 应有实质性参数差异）

详细规范见 `docs/source/module/make-dataset-cards/writing-guide.md`。

参数文档必须按 `Params dataclass` 字段逐项落标题：无功能组时使用 `### 参数名（key）`，有功能组时使用 `### 功能组` + `#### 参数名（key）`。不要把多个 key 合在一个标题里；枚举表必须写真实选项，不能写“以 UI 下拉项为准”；物理直觉不能用可套在任何参数上的模板句。`tools/docs/audit_card_docs.py` 会按这一契约检查。

### 6. 测试

最少覆盖：

- operation 可脱离 UI 直接运行，不需要 `QApplication` 或 Qt 控件。
- UI `get_params()` / `set_params()` / `to_dict()` / `from_dict()` 往返一致。
- 关键参数生效。
- 文档审计通过。

新建或实质修改卡片时，先按 `references/test-rigor-checklist.md` 选择适用风险并列行为矩阵。测试不能只验证“不报错”：

- 覆盖每个公开模式、关键分支，以及共享元素池、原子集合、输出预算或晶胞自由度的参数组合。
- 直接运行 operation，断言输出数量、组成、完整 cell/PBC、坐标/位移、距离/角度/体积/应变、magmom/group、过滤集合和 `Config_type` 等真实承诺。
- 涉及方向、层、表面、层错、复制或应变时，按支持边界加入非正交 cell、混合 PBC 和非默认取向，不能只测对角晶胞或常用预设。
- 对 `None`、`0`、空值、自动模式等哨兵建立真值表；运行默认参数并证明默认输出语义，而不只比较 dataclass。
- 随机卡先声明“严格数量 / 尽力生成 / 允许随机耗尽失败”合同，再测固定 seed、不变量和交互风险。多规则、共享候选或拒绝采样默认做至少 20 个低成本种子扫描；重型 operation 用缩小的同类约束或可控 RNG 构造最坏路径。
- 范围参数会换算为原子数、缺陷数或其他离散资源时，按“固定/范围 × 上界可行/不可行”四格真值表测试；若上界是否可行能由输入确定，必须在 RNG 前校验，不能让 seed 决定成功或失败。
- 分别断言成功有输出、合法空输出和失败；错误必须可见，partial output 按卡片合同处理，不能用空列表伪装成功。
- `preview/summary` 先声明精确还是估算，再测试与正式运行的对应关系；高成本预览必须覆盖后台执行、防抖/合并、只采用最新结果和关闭时的线程生命周期。
- 新旧 JSON 都要走 UI 往返；每个下拉值还必须从控件读成 Params 后直接运行 core 的对应分支，不能用“UI 往返一套字符串、operation 单测另一套字符串”冒充覆盖。动态检查注册、文档 URL、查找简述和翻译，不硬编码卡片总数。
- 改动共享 arrays/info 或导出路径时，对比卡片导出、内存 handoff 和再保存产物。
- 测试使用 tmp 目录，不写仓库 fixture 或运行产物；轮询终态，不用脆弱固定 sleep。

dataset/generator 卡片要直接测 `run_dataset(...)` 或 `generate(...)`；UI 往返只验证参数绑定，不能替代 operation 行为测试。

### 7. 性能与响应时间

Make Dataset 是 UI 工作流的一部分。生成类、采样类、过滤类和迁移自脚本的卡片，不能只证明结果正确；在不改变核心物理/几何语义和随机分布契约的前提下，默认要保证交互响应时间足够好。

新增或实质修改卡片时，按风险做性能检查：

- 如果 operation 存在随机尝试、两两距离、邻居搜索、PBC minimum-image、文件解析、矩阵分解、聚类/筛选、批量结构循环等潜在热点，必须跑一个代表性 `operation-only` 性能 smoke 或 profile。
- `set_dataset`、参数变化或展开面板触发的 preview/summary 也属于性能面。代表性输入约 100 ms 以上或明显随数据规模增长时，放到后台并合并连续请求；UI 只能应用最新结果，卡片关闭时不能销毁仍在运行的线程。
- 从旧脚本迁移卡片时，必须用同一输入文件和同一关键参数做 CLI / core A/B；报告比较轴是 `CLI end-to-end` 还是 `card core operation`，不要混淆进程启动和 I/O 成本。
- 如果 profile 显示热点来自可消除的重复计算、对象访问、全量两两循环、重复 cell 求逆、重复解析或可安全缓存的静态数据，交付前应直接优化，不等用户再次要求。
- 优化必须保持输出契约：数量、元素组成、cell/PBC、失败条件、seed 可复现性、`Config_type` 标签和文档参数语义不变。
- 不为了速度改采样分布、放宽碰撞约束、静默降低目标数量或引入启发式后处理；这类改变必须先和用户确认。
- 最终回复里简要给出代表性耗时、优化前后或脚本对比，以及仍然保留的热点/风险。

## 验证

从仓库根目录运行 `python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick`。更多模式见 `references/validation-playbook.md`。

## 交付格式

执行此 skill 时，输出按以下结构组织：

1. 卡片规格摘要
2. 实现清单
3. 验证结果
4. 性能与响应时间结果
5. 未覆盖风险

## 质量门槛

- core operation 与 UI 解耦。
- UI 参数、Params dataclass、文档默认值一一对应。
- 卡片功能边界清楚，与相邻卡片无无法解释的重复。
- 参数名称、单位、参考系、默认值、范围、精度、禁用语义和控件联动已经过设计审查。
- 没有“UI 有参数但逻辑没用”的死参数。
- 没有“逻辑硬编码但 UI 没暴露”的隐含参数。
- `to_dict` / `from_dict` 无丢字段和错类型。
- 非正交 cell、混合 PBC、哨兵/自动模式、合法空输出和旧 JSON 按适用风险覆盖。
- 随机 operation 的输出合同明确；交互风险有多种子或可控最坏路径测试。
- preview/summary 的精确或估算语义明确，高成本路径不阻塞 UI 且不回写过期结果。
- 对潜在长耗时卡片有 operation-only 性能检查；迁移脚本时有同输入同参数 A/B。
- 性能优化不改变核心输出语义、失败语义或 seed 可复现性。
- 查找简述、新增下拉项、枚举和状态文案完成翻译；测试不污染工作树。
- `tools/docs/audit_card_docs.py` 和相关 pytest 通过。

## Resources

### references/

- `references/card-touchpoints.md`：代码/文档/测试触点。
- `references/card-design-review.md`：逐卡审核功能边界、参数、控件、文案、翻译和重复功能。
- `references/validation-playbook.md`：验证与排错路径。
- `references/requirements-to-card-spec-template.md`：把需求/脚本先落成实现规格。
- `references/test-rigor-checklist.md`：按卡片风险选择测试矩阵，覆盖真实缺陷容易逃逸的交互与边界。

### scripts/

- `scripts/run_card_checks.py`：统一执行交付检查，包含 operation 架构审计和测试前后工作树漂移检查。
