# Card Touchpoints

Use this map to edit only what is needed.

## Core code paths

- `src/NepTrainKit/core/cards/operation.py`: `StructureOperation`, `DatasetOperation`, `GeneratorOperation`, and `params_to_dict`.
- `src/NepTrainKit/core/cards/*.py`: UI-independent Params dataclasses and operation implementations.
- `src/NepTrainKit/ui/views/_card/*.py`: PySide card UI, parameter binding, serialization, and operation delegation.
- `src/NepTrainKit/ui/views/_card/__init__.py`: card imports and exports.
- `src/NepTrainKit/core/card_manager.py`: registration mechanics.
- `src/NepTrainKit/ui/widgets/card_metadata.py`: Find card 列表、卡片简述、详情和本地化元数据。
- `src/NepTrainKit/ui/views/cards.py`: Add new card 下拉框和 Find card 入口。
- `src/NepTrainKit/ui/widgets/card_widget.py`: base `MakeDataCard` operation dispatch.
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
- `tests/test_card_library_dialog.py`: Find card 简述、文档链接、翻译和新增卡片入口。
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
- Dataset-scaled preview/summary should use the shared background-thread helpers, coalesce repeated requests, discard stale results, and stop safely when the card closes.
