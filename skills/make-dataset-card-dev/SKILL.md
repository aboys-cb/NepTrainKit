---
name: make-dataset-card-dev
description: 将“需求描述”或“已有小脚本”转化为 NepTrainKit 的 Make Dataset 卡片实现。适用于按现有卡片风格设计 UI、实现数据生成/变换业务逻辑、完成 UI 与逻辑绑定（含 `to_dict`/`from_dict`）、接入卡片注册、补齐文档与测试，并通过 `tools/docs/audit_card_docs.py` 与相关 pytest 检查。
---

# Make Dataset Card Dev

## 目标

把用户输入（需求文本或脚本）稳定转换为可交付卡片：

1. UI 设计符合项目现有卡片风格。
2. 业务逻辑可运行且与 UI 参数一一对应。
3. UI-逻辑-序列化绑定完整。
4. 注册、文档、测试链路闭环。

## 协作规则（先确认再改）

以下情况先和用户确认，再进入代码修改：

1. 需求表述存在多种合理解读。
2. 方案会改变已有行为或兼容性。
3. 改动会跨多个模块，影响面较大。
4. 存在明显更优实现路径。

当存在更优路径时，先给出“当前方案 vs 替代方案”的简短对比（收益、代价、风险），由用户拍板后再实施。

## 输入类型

### A. 需求描述输入

- 典型输入：`“我想做一个按元素比例随机替换并可控 seed 的数据增强卡片”`
- 处理重点：先抽象“参数、约束、输出结构变化”。

### B. 小脚本输入

- 典型输入：用户提供一段 Python 数据生成脚本。
- 处理重点：从脚本反推出“可配置参数 + 默认值 + 算法主流程 + 随机性控制”。

## 工作流

### 1. 需求/脚本解析为卡片规格

先产出一个“卡片规格草案”，至少包含：

- `card_name`（UI 显示名）
- `group`（菜单分组）
- 是否依赖输入数据集（`requires_input_dataset`）
- 参数清单（名称、类型、默认值、范围、是否必填）
- 处理逻辑摘要（输入结构 -> 输出结构）
- 元数据标签策略（`Config_type` 如何追加）

如果是脚本输入，再额外提取：

- 脚本里的硬编码常量，改为 UI 参数。
- 随机数行为，是否暴露 seed。
- I/O 副作用，改为卡片内存处理流程（避免直接文件覆盖）。

### 2. 按项目风格设计 UI

遵循现有实现模式（见 `references/card-touchpoints.md`）：

- 统一用 `init_ui()` 构建控件。
- 常用控件优先：
  - 数值参数：`SpinBoxUnitInputFrame`
  - 枚举参数：`ComboBox`
  - 开关参数：`CheckBox` / `RadioButton`
  - 字符串参数：`LineEdit`
- 每个参数都提供清晰 label + tooltip。
- 布局使用 `settingLayout.addWidget(...)`，保持卡片风格一致。

### 3. 实现业务逻辑

- 在 `process_structure(self, structure)` 实现核心算法。
- 返回 `list[ase.Atoms]`，不要在此层做 UI 交互。
- 结构变换后使用 `append_config_tag(...)` 写入可追溯标签。
- 对输入异常做温和降级（返回原结构或空结果，并给出消息）。

### 4. 完成 UI 与逻辑绑定

绑定必须完整：

- UI 参数读取 -> 逻辑参数传递。
- `to_dict()` 写入全部关键状态。
- `from_dict()` 用稳定默认值恢复状态，兼容旧 JSON。
- 特殊卡片若需“无输入数据集运行”，重写 `run()` 并处理 `requires_input_dataset=False` 路径。

### 5. 注册与接入

- 类上加 `@CardManager.register_card`。
- 在 `src/NepTrainKit/ui/views/_card/__init__.py` 导入并加入 `__all__`。
- 确认 MakeData 页面可添加并执行该卡片。

### 6. 文档与测试同步

- 更新 `docs/source/module/make-dataset-cards/cards/*.md` 对应文档。
- 文档参数名、默认值、`serialized_keys` 与代码一致。
- 测试最少覆盖：
  - 基本运行不崩溃
  - 参数生效
  - `to_dict`/`from_dict` 往返一致

## 交付格式

执行此 skill 时，输出按以下结构组织：

1. 卡片规格摘要（从需求/脚本提取）
2. 实现清单（改了哪些文件、每个文件做了什么）
3. 验证结果（跑了哪些命令、通过/失败）
4. 未覆盖风险（如果有）

## 质量门槛

- UI 参数与业务逻辑参数一一对应，无“死参数”。
- 业务逻辑不依赖 UI 状态副作用，代码可测试。
- `to_dict`/`from_dict` 无丢字段和错类型。
- 文档审计与相关测试通过。

## 快速验证命令

- 快速检查：
  - `python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick`
- 含文档构建：
  - `python skills/make-dataset-card-dev/scripts/run_card_checks.py --with-docs`
- 全量回归：
  - `python skills/make-dataset-card-dev/scripts/run_card_checks.py --full`

## Resources

### references/

- `references/card-touchpoints.md`：代码/文档/测试触点。
- `references/validation-playbook.md`：验证与排错路径。
- `references/requirements-to-card-spec-template.md`：把需求/脚本先落成实现规格。

### scripts/

- `scripts/run_card_checks.py`：统一执行交付检查。
## Online Doc Mapping Rule

- Every built-in Make Dataset card must preserve a stable mapping between the card source filename and the docs page filename.
- For `src/NepTrainKit/ui/views/_card/foo_bar_card.py`, the docs page must be `docs/source/module/make-dataset-cards/cards/foo-bar-card.md`.
- The UI links cards directly to `https://neptrainkit.readthedocs.io/en/latest/module/make-dataset-cards/cards/foo-bar-card.html`.
- If a new card does not match this mapping, rename or update the docs page before finishing the task. Do not ship a built-in card without a valid online docs target.
