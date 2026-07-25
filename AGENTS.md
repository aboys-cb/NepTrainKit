# NepTrainKit Agent Notes

## Project Skill

This repo has a local skill at `skills/make-dataset-card-dev/SKILL.md`.

Before changing Make Dataset cards, card operations, card docs, or card tests:

1. Read `skills/make-dataset-card-dev/SKILL.md` from this checkout.
2. Follow its operation/params/UI/docs/tests contract.
3. If the global Codex skill is not installed or does not auto-trigger, say so briefly and use the repo-local `SKILL.md` as the source of truth.

The skill should trigger naturally when the task is about adding, migrating, refactoring, or testing Make Dataset cards. Do not assume every developer has it installed globally; the repo-local file is the fallback.

## Change Style

- Keep changes surgical. Touch only files needed for the requested behavior.
- Prefer existing helpers, card patterns, canvas backends, and result-data contracts over new abstractions.
- Do not add optional flexibility, new dependencies, or broad refactors unless the user asks.
- Preserve user-visible workflows unless the task explicitly changes them.

## Tests

- For ordinary fixes, run the smallest targeted pytest command that covers the behavior, then broaden if risk warrants it.
- Before committing a repo change, run `git diff --check`.
- For Make Dataset card work, also run `python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick` when the script is available.
- Keep ad hoc generated files out of Git. Use temp paths for fixtures and clean up local outputs created by tests.

## Local Artifacts

Treat `tmp/`, `outputs/`, `diagnostics/`, generated descriptors, copied datasets, compiled local binaries, and similar files as local artifacts unless the user explicitly asks to version them.
