# Validation Playbook

Run from repo root.

## Quick path (default)

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick
```

Runs:

- operation architecture audit
- `pytest tests/test_makedata_source_card.py tests/test_card_skill_checks.py tests/cards -q`
- `python tools/docs/audit_card_docs.py`
- worktree drift check: tests must not create or rewrite tracked/untracked repository artifacts

## Include docs build

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --with-docs
```

Adds:

- `python -m sphinx -W -b html docs/source docs/build/html`

## Include workflow UI checks

Use this whenever the change touches the right inspector, shared compact controls, card header, Fluent card catalog, grouping/forking, workflow library, responsive sizing, or translations:

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick --ui
```

Adds:

- `tests/test_compact_form_widgets.py`
- `tests/test_card_library_dialog.py`
- `tests/test_workflow_branching.py`
- `tests/test_workflow_library.py`
- `tests/test_i18n.py`

These tests exercise the real `MakeWorkflowArea` / `MakeDataWidget`, including narrow inspector width and wide-to-default resize cycles. A standalone card widget screenshot is not sufficient UI validation.

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
3. For UI changes, fix parent hierarchy, insertion/drag target, resize and horizontal-overflow failures before visual polish.
4. Fix schema/section mismatches reported by `audit_card_docs.py`.
5. Fix docs warnings or broken references from Sphinx.

## New card test bar

Before implementation, use `references/card-design-review.md` to review user scenario, feature boundary, parameter/control design, reference frames, discoverability, and translation. Then update `tests/cards/` in the matching domain file and select applicable risks from `references/test-rigor-checklist.md`.

- cover each public mode and key parameter branch, plus combinations that share an element pool, atom set, output budget, or cell degree of freedom;
- cover defaults, sentinels, invalid/boundary params, legacy JSON, and explicit errors;
- assert output semantics such as structure count, composition, cell/position/magmom changes, dataset filter decisions, generated tags, and `Config_type`;
- include non-orthogonal cells, mixed PBC, and non-default orientations when the public contract supports them;
- define the random output contract before testing fixed seeds, multi-seed interaction risks, and retry exhaustion;
- distinguish success with output, legitimate empty output, and failure, including user-visible UI state where applicable;
- classify preview/summary as exact or estimated; for expensive previews test worker-thread execution, latest-request wins, and safe close behavior;
- dynamically check registration, docs URL, card-search summary, and translation instead of hard-coding the number of cards;
- keep UI round-trip tests focused on parameter binding and serialization; operation behavior belongs in direct operation tests.

Full rationale and test patterns: `references/test-rigor-checklist.md`.

## Operation architecture audit

The validation script checks:

- every built-in `MakeDataCard` / `FilterDataCard` subclass has `create_operation()`;
- no built-in card has a custom `run()` unless the architecture is explicitly changed;
- `src/NepTrainKit/core/cards/*.py` does not import `PySide6`, `qfluentwidgets`, or `MessageManager`.
