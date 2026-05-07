# Card Touchpoints

Use this map to edit only what is needed.

## Core code paths

- `src/NepTrainKit/core/cards/operation.py`: `StructureOperation`, `DatasetOperation`, `GeneratorOperation`, and `params_to_dict`.
- `src/NepTrainKit/core/cards/*.py`: UI-independent Params dataclasses and operation implementations.
- `src/NepTrainKit/ui/views/_card/*.py`: PySide card UI, parameter binding, serialization, and operation delegation.
- `src/NepTrainKit/ui/views/_card/__init__.py`: card imports and exports.
- `src/NepTrainKit/core/card_manager.py`: registration mechanics.
- `src/NepTrainKit/ui/widgets/card_widget.py`: base `MakeDataCard` operation dispatch.
- `src/NepTrainKit/ui/threads.py`: operation execution threads.
- `src/NepTrainKit/ui/pages/makedata.py`: workflow runtime integration.

## Docs paths

- `docs/source/module/make-dataset-cards/cards/*.md`: per-card docs.
- `docs/source/module/make-dataset-cards/writing-guide.md`: **authoritative doc style reference.** Read before writing any card doc.
- `tools/docs/audit_card_docs.py`: minimal integrity check (key consistency, code-doc defaults match). Does NOT enforce style — style is enforced by the writing guide.
- `skills/make-dataset-card-dev/references/requirements-to-card-spec-template.md`: pre-coding card spec template.

## Tests to touch first

- `tests/test_makedata_source_card.py`: source-card execution in MakeData page.
- `tests/cards/`: operation, card transformation, and serialization tests grouped by card domain.
- Operation tests should avoid Qt setup: call `Operation().run_structure(atoms, Params(...))`, `run_dataset(...)`, or `generate(...)` directly.

## CI references

- `.github/workflows/pytest.yml`: local checks expected by CI.

## Architecture audit

- Built-in `MakeDataCard` and `FilterDataCard` subclasses should provide `create_operation()`.
- Structure cards use `StructureOperation`; dataset filters use `DatasetOperation`; no-input generators use `GeneratorOperation`.
- `src/NepTrainKit/core/cards/*.py` must not import `PySide6`, `qfluentwidgets`, or `MessageManager`.
- Built-in cards should not override `run()`; dispatch belongs in `MakeDataCard.run()`.
