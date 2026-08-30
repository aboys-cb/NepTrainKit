from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "make-dataset-card-dev"
    / "scripts"
    / "run_card_checks.py"
)
SPEC = importlib.util.spec_from_file_location("run_card_checks", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_card_checks = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("run_card_checks", run_card_checks)
SPEC.loader.exec_module(run_card_checks)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _initialized_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Card Skill Test")
    _git(repo, "config", "user.email", "card-skill@example.invalid")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "baseline")
    return repo


def test_worktree_snapshot_detects_new_untracked_artifact(tmp_path):
    repo = _initialized_repo(tmp_path)
    before = run_card_checks.capture_worktree_snapshot(repo)

    (repo / "generated.out").write_text("test artifact\n", encoding="utf-8")
    after = run_card_checks.capture_worktree_snapshot(repo)

    assert before != after
    assert run_card_checks.report_worktree_drift(before, after) == 1


def test_worktree_snapshot_detects_rewrite_of_already_modified_file(tmp_path):
    repo = _initialized_repo(tmp_path)
    tracked = repo / "tracked.txt"
    tracked.write_text("local edit before tests\n", encoding="utf-8")
    before = run_card_checks.capture_worktree_snapshot(repo)

    tracked.write_text("rewritten by test\n", encoding="utf-8")
    after = run_card_checks.capture_worktree_snapshot(repo)

    assert before is not None and after is not None
    assert before.status == after.status
    assert before.tracked_diff != after.tracked_diff
    assert run_card_checks.report_worktree_drift(before, after) == 1


def test_ui_mode_adds_real_workflow_interface_regressions():
    commands = run_card_checks.build_commands(
        full=False,
        with_docs=False,
        ui=True,
    )
    ui_command = next(
        command
        for command in commands
        if "tests/test_workflow_branching.py" in command
    )

    assert "tests/test_compact_form_widgets.py" in ui_command
    assert "tests/test_card_library_dialog.py" in ui_command
    assert "tests/test_workflow_library.py" in ui_command
    assert "tests/test_i18n.py" in ui_command


def test_quick_mode_keeps_ui_suite_opt_in():
    commands = run_card_checks.build_commands(
        full=False,
        with_docs=False,
    )

    assert all(
        "tests/test_workflow_branching.py" not in command
        for command in commands
    )
