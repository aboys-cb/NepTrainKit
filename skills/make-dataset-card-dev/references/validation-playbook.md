# Validation Playbook

Run from repo root.

## Quick path (default)

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick
```

Runs:

- operation architecture audit
- `pytest tests/test_makedata_source_card.py tests/test_card.py -q`
- `python tools/docs/audit_card_docs.py`

## Include docs build

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --with-docs
```

Adds:

- `python -m sphinx -W -b html docs/source docs/build/html`

## Full regression path

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --full
```

Runs:

- operation architecture audit
- `pytest tests/`
- `python tools/docs/audit_card_docs.py`
- `python -m sphinx -W -b html docs/source docs/build/html`

## Triage order

1. Fix operation architecture violations first; do not leave core logic in UI cards.
2. Fix code/runtime errors from pytest.
3. Fix schema/section mismatches reported by `audit_card_docs.py`.
4. Fix docs warnings or broken references from Sphinx.

## Operation architecture audit

The validation script checks:

- every built-in `MakeDataCard` / `FilterDataCard` subclass has `create_operation()`;
- no built-in card has a custom `run()` unless the architecture is explicitly changed;
- `src/NepTrainKit/core/cards/*.py` does not import `PySide6`, `qfluentwidgets`, or `MessageManager`.
