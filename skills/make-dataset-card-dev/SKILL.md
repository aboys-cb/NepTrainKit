---
name: make-dataset-card-dev
description: 将“需求描述”或“已有小脚本”转化为 NepTrainKit 的 Make Dataset 卡片实现或维护现有卡片。适用于新增、迁移、重构 Make Dataset 卡片时，按 Operation/Params 架构拆分 core 业务逻辑与 PySide UI，完成卡片注册、参数序列化、文档、测试和 `tools/docs/audit_card_docs.py` 检查。
---

# Make Dataset Card Dev

## 目标

把需求文本、已有脚本或现有卡片重构需求稳定转换为可交付卡片：

1. 业务逻辑在 `src/NepTrainKit/core/cards/`，不依赖 PySide、qfluentwidgets 或 MessageManager。
2. UI 只负责控件、参数读写、序列化和调用 operation。
3. `Params dataclass`、operation、UI、文档、测试保持同一套参数契约。
4. 注册、在线文档路径、测试和文档审计闭环。

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

### 1. 需求/脚本解析为卡片规格

先产出一个“卡片规格草案”，至少包含：

- `card_name`、`group`、`menu_icon`、`requires_input_dataset`
- operation 类型：`StructureOperation` / `DatasetOperation` / `GeneratorOperation`
- `Params dataclass` 字段清单：名称、类型、默认值、范围、是否必填
- 处理逻辑摘要：输入对象 -> 输出结构列表
- 随机性与 seed 策略
- `Config_type` 标签策略

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

新建或实质修改卡片时，测试不能只验证“不报错”。必须在 `tests/cards/` 对应领域文件中覆盖该卡片的核心输出语义：

- 每个用户可选模式、关键参数分支和随机 seed 策略都要有代表性用例。
- 边界条件和非法参数要验证明确失败原因，不做静默降级或伪成功。
- operation 测试必须判定输出效果是否符合预期，而不是只验证能运行。断言应覆盖该卡片真正承诺改变或保持的对象，例如输出结构数量、原子/元素组成、cell/PBC、坐标或位移范围、距离/角度/体积/应变约束、magmom 变化、过滤保留/剔除集合、`Config_type` 追踪标签等。
- 对生成类和随机类卡片，至少要有一个固定 seed 的 operation-only 用例，断言可复现性和关键几何/组成约束；如果某个分支会改变 cell、选择中心、按密度/比例推导数量或保留部分输出，也要直接检查这些结果。
- 对失败路径，测试要断言失败原因和不产生伪成功输出；如果允许 partial output，必须检查 partial 的数量、标签和约束，而不是只检查返回了对象。
- 随机型卡片要验证固定 seed 的可复现性；没有 seed 的随机分布只检查物理/几何约束，不写脆弱的逐坐标快照。
- dataset/generator 卡片要直接测 `run_dataset(...)` 或 `generate(...)` 的输入输出契约；UI 往返测试只验证参数绑定，不能替代 operation 行为测试。

### 7. 性能与响应时间

Make Dataset 是 UI 工作流的一部分。生成类、采样类、过滤类和迁移自脚本的卡片，不能只证明结果正确；在不改变核心物理/几何语义和随机分布契约的前提下，默认要保证交互响应时间足够好。

新增或实质修改卡片时，按风险做性能检查：

- 如果 operation 存在随机尝试、两两距离、邻居搜索、PBC minimum-image、文件解析、矩阵分解、聚类/筛选、批量结构循环等潜在热点，必须跑一个代表性 `operation-only` 性能 smoke 或 profile。
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
- 没有“UI 有参数但逻辑没用”的死参数。
- 没有“逻辑硬编码但 UI 没暴露”的隐含参数。
- `to_dict` / `from_dict` 无丢字段和错类型。
- 对潜在长耗时卡片有 operation-only 性能检查；迁移脚本时有同输入同参数 A/B。
- 性能优化不改变核心输出语义、失败语义或 seed 可复现性。
- `tools/docs/audit_card_docs.py` 和相关 pytest 通过。

## Resources

### references/

- `references/card-touchpoints.md`：代码/文档/测试触点。
- `references/validation-playbook.md`：验证与排错路径。
- `references/requirements-to-card-spec-template.md`：把需求/脚本先落成实现规格。

### scripts/

- `scripts/run_card_checks.py`：统一执行交付检查，并包含 operation 架构审计。
