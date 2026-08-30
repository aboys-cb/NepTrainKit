# Card Touchpoints

Use this map to edit only what is needed.

## Core code paths

- `src/NepTrainKit/core/cards/operation.py`: `StructureOperation`, `DatasetOperation`, `GeneratorOperation`, and `params_to_dict`.
- `src/NepTrainKit/core/cards/*.py`: UI-independent Params dataclasses and operation implementations.
- `src/NepTrainKit/ui/views/_card/*.py`: PySide card UI, parameter binding, serialization, and operation delegation.
- `src/NepTrainKit/ui/views/_card/__init__.py`: card imports and exports.
- `src/NepTrainKit/core/card_manager.py`: registration mechanics.
- `src/NepTrainKit/ui/widgets/card_metadata.py`: Fluent 分类卡片目录、卡片简述、详情和本地化元数据。
- `src/NepTrainKit/ui/views/cards.py`: 全局控制台、单一新建卡片入口、运行/停止和查看输出。
- `src/NepTrainKit/ui/widgets/card_widget.py`: base `MakeDataCard` operation dispatch、卡片头、状态和顶层结果动作。
- `src/NepTrainKit/ui/widgets/compact_form.py`: 右侧检查器使用的紧凑字段、分组、响应式网格和分段选择器。
- `src/NepTrainKit/ui/widgets/docker.py`: 工作流画布、右侧卡片检查器、选择和响应式三栏工作台。
- `src/NepTrainKit/ui/widgets/workflow_library.py`: 左侧工作流/模板管理区及工作流级动作。
- `src/NepTrainKit/core/workflow_library.py`: 仅配置的工作流持久化与运行态字段清理。
- `src/NepTrainKit/ui/views/_card/card_group.py`: 共享输入、独立处理后立即合并的分流合并容器。
- `src/NepTrainKit/ui/views/_card/workflow_fork.py`: 保持独立线性路径并可显式合并的永久分叉容器。
- `src/NepTrainKit/ui/threads.py`: operation execution threads.
- `src/NepTrainKit/ui/pages/makedata.py`: workflow runtime integration.
- `src/NepTrainKit/translations/neptrainkit_zh_CN.ts`: 卡片名称、简述、控件、枚举和状态文案翻译源。

## Docs paths

- `docs/source/module/make-dataset-cards/cards/*.md`: per-card docs.
- `docs/source/module/make-dataset-cards/writing-guide.md`: **authoritative doc style reference.** Read before writing any card doc.
- `tools/docs/audit_card_docs.py`: minimal integrity check (key consistency, code-doc defaults match). Does NOT enforce style — style is enforced by the writing guide.
- `skills/make-dataset-card-dev/references/requirements-to-card-spec-template.md`: pre-coding card spec template.

## Tests to touch first

- `tests/test_makedata_source_card.py`: source-card execution in MakeData page.
- `tests/cards/`: operation, card transformation, and serialization tests grouped by card domain.
- `tests/test_card_library_dialog.py`: Fluent 卡片目录的分类、搜索、简述、文档链接、翻译和新增卡片入口。
- `tests/test_compact_form_widgets.py`: 紧凑字段、状态和分段选择器。
- `tests/cards/test_compact_numeric_inputs.py`: 默认检查器宽度下数值、单位和步进按钮可见性。
- `tests/test_workflow_branching.py`: 分流合并、永久分叉、拖拽目标、展开收起、运行语义、递归 JSON 和真实尺寸。
- `tests/test_workflow_library.py`: 工作流/模板 CRUD、运行态数据排除和工作台接入。
- `tests/test_i18n.py`: TS/QM 完整性和运行时翻译。
- `tests/test_makedata_failure_semantics.py`: 成功、合法空输出、失败和工作流链终态。
- `tests/test_threads.py`: 通用后台任务、回调线程和线程清理。
- Operation tests should avoid Qt setup: call `Operation().run_structure(atoms, Params(...))`, `run_dataset(...)`, or `generate(...)` directly.

## CI references

- `.github/workflows/pytest.yml`: local checks expected by CI.

## Architecture audit

- Built-in `MakeDataCard` and `FilterDataCard` subclasses should provide `create_operation()`.
- Structure cards use `StructureOperation`; dataset filters use `DatasetOperation`; no-input generators use `GeneratorOperation`.
- `src/NepTrainKit/core/cards/*.py` must not import `PySide6`, `qfluentwidgets`, or `MessageManager`.
- Built-in cards should not override `run()`; dispatch belongs in `MakeDataCard.run()`.
- `CardGroup` and `WorkflowFork` are orchestration containers and are exempt from the ordinary `create_operation()` rule; preserve their distinct merge semantics.
- Dataset-scaled preview/summary should use the shared background-thread helpers, coalesce repeated requests, discard stale results, and stop safely when the card closes.
- New parameter UIs should use the shared compact inspector widgets; `adapt_legacy_inspector_form()` is a compatibility path, not the target for new cards.
