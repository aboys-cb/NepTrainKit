# Card Touchpoints

Use this map to edit only what is needed.

## Core code paths

- `src/NepTrainKit/ui/views/_card/*.py`: card implementations.
- `src/NepTrainKit/ui/views/_card/__init__.py`: card imports and exports.
- `src/NepTrainKit/core/card_manager.py`: registration mechanics.
- `src/NepTrainKit/ui/pages/makedata.py`: workflow runtime integration.

## Docs paths

- `docs/source/module/make-dataset-cards/cards/*.md`: per-card docs.
- `docs/source/module/make-dataset-cards/writing-guide.md`: doc structure requirements.
- `tools/docs/audit_card_docs.py`: audit rules enforced in CI.
- `skills/make-dataset-card-dev/references/requirements-to-card-spec-template.md`: pre-coding card spec template.

## Tests to touch first

- `tests/test_makedata_source_card.py`: source-card execution in MakeData page.
- `tests/test_card.py`: behavior tests for card transformations.

## CI references

- `.github/workflows/pytest.yml`: local checks expected by CI.
