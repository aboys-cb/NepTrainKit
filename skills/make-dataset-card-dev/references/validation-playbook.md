# Validation Playbook

Run from repo root.

## Quick path (default)

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick
```

Runs:

- operation architecture audit
- `pytest tests/test_makedata_source_card.py tests/cards -q`
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

## Card generation benchmark

Use this when a new or changed card may be slow:

```bash
python tools/benchmark_card_operations.py
```

This benchmarks structure-generating operations directly, without Qt UI startup. It also runs lightweight semantic checks on every scenario, so a faster path must still generate the expected output shape, tags, composition, geometry, or magnetic moments. Useful filters:

```bash
python tools/benchmark_card_operations.py --only spin --repeat 5
python tools/benchmark_card_operations.py --profile count50
python tools/benchmark_card_operations.py --only slab --json card_bench.json
```

## Triage order

1. Fix operation architecture violations first; do not leave core logic in UI cards.
2. Fix code/runtime errors from pytest.
3. Fix schema/section mismatches reported by `audit_card_docs.py`.
4. Fix docs warnings or broken references from Sphinx.

## New card test bar

When adding a card, update `tests/cards/` in the domain file that matches the card. The test must prove the card behavior, not only that execution finishes:

- cover each public mode and key parameter branch;
- cover invalid or boundary params with explicit errors;
- assert output semantics such as structure count, composition, cell/position/magmom changes, dataset filter decisions, generated tags, and `Config_type`;
- cover deterministic seed behavior for random operations;
- keep UI round-trip tests focused on parameter binding and serialization; operation behavior belongs in direct operation tests.

## Operation architecture audit

The validation script checks:

- every built-in `MakeDataCard` / `FilterDataCard` subclass has `create_operation()`;
- no built-in card has a custom `run()` unless the architecture is explicitly changed;
- `src/NepTrainKit/core/cards/*.py` does not import `PySide6`, `qfluentwidgets`, or `MessageManager`.
