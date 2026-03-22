# Validation Playbook

Run from repo root.

## Quick path (default)

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick
```

Runs:

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

- `pytest tests/`
- `python tools/docs/audit_card_docs.py`
- `python -m sphinx -W -b html docs/source docs/build/html`

## Triage order

1. Fix code/runtime errors from pytest.
2. Fix schema/section mismatches reported by `audit_card_docs.py`.
3. Fix docs warnings or broken references from Sphinx.
