"""Model library scorecard, benchmark import dialog, viewer chip and settings."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.core.validation import SetTooLargeError
from NepTrainKit.core.validation import library as library_module
from NepTrainKit.ui.views.model_library import (
    VERDICT_COLORS,
    ModelLibraryWidget,
    ScorecardTableModel,
    summarize_model_status,
)
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from test_validation_framework import MODEL_HEADER, _import, efv_frames, eos_frames, write_xyz
from test_validation_framework import make_service as make_service, model_file as model_file


def _wait_until(predicate, timeout=10.0):
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _capture(service, tmp_path, name):
    path = tmp_path / name / "nep.txt"
    path.parent.mkdir()
    path.write_text(MODEL_HEADER + name)
    return service.capture_model(path)


@pytest.fixture
def scored(make_service, tmp_path):
    """Baseline ``old``, improved ``new`` and never-evaluated ``fresh`` on an EOS + EFV suite."""
    service = make_service()
    old, new, fresh = (_capture(service, tmp_path, name) for name in ("old", "new", "fresh"))
    make_service(**{old["content_hash"]: (1.10, 0.0), new["content_hash"]: (1.02, 0.0)})
    eos = _import(service, tmp_path, eos_frames(), "eos")
    efv = _import(service, tmp_path, efv_frames(), "efv")
    suite = service.create_suite("PbTe", [eos["id"], efv["id"]])
    for model in (old, new):
        service.run_suite(model["id"], suite["id"])
    service.set_baseline(suite["id"], old["id"], "first production model")
    return SimpleNamespace(service=service, suite=suite, old=old, new=new, fresh=fresh, eos=eos, efv=efv)


# -- scorecard model ------------------------------------------------------------

def test_scorecard_model_marks_baseline_and_colours_verdicts(scored):
    service = scored.service
    card = service.scorecard(scored.suite["id"], [scored.new["id"], scored.fresh["id"]])
    model = ScorecardTableModel()
    model.set_scorecard(card)
    assert model.columnCount() == 1 + len(card["columns"]) and model.rowCount() == 3
    assert model.headerData(0, Qt.Orientation.Horizontal) == "Model"
    assert model.headerData(1, Qt.Orientation.Horizontal).startswith("eos\n")

    base = model.row_of_model(scored.old["id"])
    assert model.data(model.index(base, 0)).endswith("(baseline)")
    assert model.data(model.index(base, 0), Qt.ItemDataRole.FontRole).bold()
    assert model.data(model.index(base, 1), Qt.ItemDataRole.BackgroundRole) is None

    new = model.row_of_model(scored.new["id"])
    energy = next(i + 1 for i, c in enumerate(card["columns"]) if c["metric"] == "energy_rmse")
    assert model.data(model.index(new, energy), Qt.ItemDataRole.BackgroundRole) == VERDICT_COLORS["better"]
    assert float(model.data(model.index(new, energy))) > 0
    assert model.data(model.index(new, energy), ScorecardTableModel.CellRole)["verdict"] == "better"

    fresh = model.row_of_model(scored.fresh["id"])
    assert model.data(model.index(fresh, 1)) == "not run"
    assert model.data(model.index(fresh, 1), Qt.ItemDataRole.BackgroundRole) is None
    model.set_scorecard(None)
    assert model.rowCount() == 0 and model.columnCount() == 1


def test_library_runs_all_in_background_and_overlays_details(scored, monkeypatch):
    service = scored.service
    widget = ModelLibraryWidget(service=service)
    widget.refresh(scored.suite["id"])
    assert widget.model.rowCount() == 3
    assert "first production model" in widget.baseline_label.text()
    assert (scored.fresh["id"], scored.eos["id"]) in widget.run_jobs()

    task = widget.run_all()
    assert widget.run_button.text() == "Cancel"
    assert _wait_until(lambda: widget._run_task is None)
    assert task.outcome == "succeeded" and widget.run_button.text() == "Run all"
    assert widget.model.row_of_model(scored.fresh["id"]) >= 0

    # Ctrl+click two cells of the EOS column: both curves are overlaid on the reference.
    rows = [widget.model.row_of_model(model["id"]) for model in (scored.old, scored.new)]
    widget.table.setCurrentIndex(widget.model.index(rows[0], 1))
    widget.table.selectionModel().select(widget.model.index(rows[1], 1), QItemSelectionModel.SelectionFlag.Select)
    assert widget.detail.title_label.text().startswith("eos")
    names = [item.name() for item in widget.detail.plot.listDataItems()]
    assert names == ["Reference", "old", "new"]

    widget.table.setCurrentIndex(widget.model.index(rows[1], 0))
    widget.toggle_star()
    assert service.get_model(scored.new["id"])["starred"]
    assert widget.star_button.text() == "Unstar"

    from NepTrainKit.ui.widgets import GetStrMessageBox

    monkeypatch.setattr(GetStrMessageBox, "exec", lambda box: (box.lineEdit.setText("better forces"), 1)[1])
    widget.set_baseline()
    suite = service.get_suite(scored.suite["id"])
    assert suite["baseline_model_id"] == scored.new["id"] and suite["baseline_note"] == "better forces"
    assert widget.model.row(widget.model.row_of_model(scored.new["id"]))["is_baseline"]


def test_library_run_all_can_be_cancelled(scored):
    widget = ModelLibraryWidget(service=scored.service)
    widget.refresh(scored.suite["id"])
    total = len(widget.run_jobs())
    task = widget.run_all()
    widget._on_run_clicked()  # the Run button becomes Cancel while running
    assert _wait_until(lambda: widget._run_task is None)
    assert task.outcome == "canceled"
    runs = [scored.service.latest_run(m, s) for m, s in widget.run_jobs()]
    assert sum(run is not None for run in runs) < total


def test_open_scorecard_selects_suite_and_model(scored):
    service = scored.service
    other = service.create_suite("EOS only", [scored.eos["id"]])
    widget = ModelLibraryWidget(service=service)
    widget.refresh(other["id"])
    widget.open_scorecard(scored.suite["id"], scored.new["id"])
    assert widget.current_suite_id() == scored.suite["id"]
    assert widget.current_model_row()["model_id"] == scored.new["id"]


# -- import dialog ----------------------------------------------------------------

def test_import_dialog_requires_provenance_and_lists_issues(make_service, tmp_path):
    service = make_service()
    host = QWidget()
    dialog = BenchmarkImportDialog(host, service.suites())
    assert not dialog.yesButton.isEnabled() and "Choose a benchmark file" in dialog.status_label.text()

    dialog.load_file(write_xyz(tmp_path / "few.xyz", eos_frames()[:3]))
    assert dialog.current_protocol().kind == "efv"  # too few points for EOS
    dialog.protocol_combo.setCurrentIndex(0)

    dialog.load_file(write_xyz(tmp_path / "scan.xyz", eos_frames()))
    assert dialog.current_protocol().kind == "eos" and dialog.name_edit.text() == "few"
    assert dialog.issues == [] and "No issues found" in dialog.issues_label.text()
    dialog.name_edit.setText("PbTe rocksalt EOS")
    assert dialog.validation_errors() == [
        "Declare where the reference labels come from.",
        "Describe the reference calculation conditions.",
    ]
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData("DFT"))
    assert not dialog.yesButton.isEnabled()
    dialog.conditions_edit.setText("VASP PBE 520 eV")
    assert dialog.yesButton.isEnabled() and dialog.validate()

    shifted = eos_frames(minimum=1.2)  # reference minimum outside the scan
    dialog.load_file(write_xyz(tmp_path / "edge.xyz", shifted))
    assert [issue.level for issue in dialog.issues] == ["warning"]
    assert "Warning" in dialog.issues_label.text() and dialog.yesButton.isEnabled()

    (tmp_path / "bad.xyz").write_text("not an xyz file\n")
    dialog.load_file(tmp_path / "bad.xyz")
    assert not dialog.yesButton.isEnabled() and "cannot be read" in dialog.status_label.text()

    broken = eos_frames()
    broken[2].info["_ref_energy"] = float("nan")
    dialog.load_file(write_xyz(tmp_path / "nan.xyz", broken))
    assert dialog.current_protocol().kind == "eos"
    assert [issue.level for issue in dialog.issues] == ["error"] and "Error" in dialog.issues_label.text()
    assert not dialog.yesButton.isEnabled()


def test_import_dialog_asks_before_copying_large_files(make_service, tmp_path):
    service = make_service()
    original = service.import_set
    service.import_set = lambda path, **kwargs: original(path, max_bytes=10, **kwargs)
    host = QWidget()
    dialog = BenchmarkImportDialog(host, [], "PbTe")
    dialog.load_file(write_xyz(tmp_path / "scan.xyz", eos_frames()))
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData("DFT"))
    dialog.conditions_edit.setText("VASP PBE")

    asked = []
    assert dialog.import_into(service, lambda error: asked.append(error) or False) is None
    assert isinstance(asked[0], SetTooLargeError) and service.sets() == []

    record = dialog.import_into(service, lambda error: True)
    assert record["provenance"] == {"conditions": "VASP PBE", "source_kind": "DFT"}
    suite = service.suites()[0]
    assert suite["name"] == "PbTe" and [s["id"] for s in service.suite_sets(suite["id"])] == [record["id"]]


# -- viewer chip ------------------------------------------------------------------

def test_status_summary_counts_regressions_on_the_preferred_suite(scored):
    service = scored.service
    service.set_baseline(scored.suite["id"], scored.new["id"])
    status = summarize_model_status(service, scored.old["id"])
    assert status["suite_name"] == "PbTe" and status["evaluated"] and status["regressions"] > 0
    other = service.create_suite("EOS only", [scored.eos["id"]])
    assert summarize_model_status(service, scored.old["id"], other["id"])["suite_id"] == other["id"]
    assert not summarize_model_status(service, scored.fresh["id"])["evaluated"]


def _viewer_stub(model_id, opened, imported=None):
    from qfluentwidgets import Action, DotInfoBadge, InfoBadge, TransparentDropDownToolButton

    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    def import_benchmark_set():
        if imported is not None:
            imported.append(True)

    stub = SimpleNamespace(
        validation_button=TransparentDropDownToolButton(),
        validation_status_action=Action("status"),
        validation_star_action=Action("star"),
        validation_save_action=Action("save"),
        validation_count_badge=InfoBadge.error("0"),
        validation_dot_badge=DotInfoBadge.info(),
        _validation_model_id=model_id,
        _validation_draft=None,
        _validation_library_root=library_module.resolve_library_root(),
        _validation_library_changed=False,
        _validation_status=None,
        _validation_starred=False,
        _parent=SimpleNamespace(
            open_model_scorecard=lambda suite, model=None, **kwargs: opened.append((suite, model, kwargs["draft"]) if kwargs else (suite, model)),
            import_benchmark_set=import_benchmark_set,
        ),
        tr=lambda text: text,
        window=lambda: QWidget(),
    )
    for name in (
        "_validation_service",
        "refresh_validation_chip",
        "_show_validation_badge",
        "open_validation_scorecard",
        "import_validation_benchmark",
        "toggle_validation_star",
        "save_validation_model",
    ):
        setattr(stub, name, getattr(ShowNepWidget, name).__get__(stub))
    return stub


def _badge(stub):
    """What the footer shows next to the icon: a count, a dot, or nothing."""
    if not stub.validation_count_badge.isHidden():
        return int(stub.validation_count_badge.text())
    return "dot" if not stub.validation_dot_badge.isHidden() else None


def test_viewer_status_line_and_badge_follow_the_suite(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    service, opened = scored.service, []
    # With the newer model as baseline, the old one regresses: a red count.
    service.set_baseline(scored.suite["id"], scored.new["id"])
    stub = _viewer_stub(scored.old["id"], opened)
    stub.refresh_validation_chip()
    count = summarize_model_status(service, scored.old["id"])["regressions"]
    assert count > 0 and _badge(stub) == count
    assert stub.validation_status_action.text() == f"Validation: PbTe · {count} regressions"
    assert stub.validation_button.toolTip() == stub.validation_status_action.text()
    stub.open_validation_scorecard()
    assert opened == [(scored.suite["id"], scored.old["id"])]

    stub._validation_model_id = scored.new["id"]
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: PbTe · baseline" and _badge(stub) is None

    # No regressions against the baseline: no badge at all.
    service.set_baseline(scored.suite["id"], scored.old["id"])
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: PbTe · 0 regressions" and _badge(stub) is None

    stub._validation_model_id = scored.fresh["id"]
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: PbTe · not run" and _badge(stub) == "dot"
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


def test_viewer_entry_guides_first_import(scored, tmp_path, monkeypatch):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    opened, imported = [], []
    # A model no suite covers: the menu says so, the dot asks for attention.
    nickel = tmp_path / "ni" / "nep.txt"
    nickel.parent.mkdir()
    nickel.write_text("nep4 1 Ni\ncutoff 8 4 74 10\n")
    stub = _viewer_stub(scored.service.capture_model(nickel)["id"], opened, imported)
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: no matching set" and _badge(stub) == "dot"
    assert "Import a DFT benchmark set" in stub.validation_button.toolTip()
    stub.import_validation_benchmark()
    assert imported == [True]

    # No captured model: a neutral status, nothing to star.
    stub._validation_model_id = None
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: no NEP model open" and _badge(stub) is None
    assert not stub.validation_star_action.isEnabled()
    stub.open_validation_scorecard()
    assert opened == [(None, None)]

    # An empty library says it is not set up yet.
    empty = ValidationService(Library(tmp_path / "empty-lib"))
    monkeypatch.setenv(library_module.ENV_VAR, str(empty.library.root))
    path = tmp_path / "first" / "nep.txt"
    path.parent.mkdir()
    path.write_text(MODEL_HEADER)
    stub._validation_model_id = empty.capture_model(path)["id"]
    stub._validation_library_root = empty.library.root
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: not set up" and _badge(stub) == "dot"
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


def test_fresh_model_is_selected_and_can_be_validated_alone(scored):
    service = scored.service
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(scored.suite["id"], scored.fresh["id"])
    row = widget.current_model_row()
    assert row["model_id"] == scored.fresh["id"]
    assert all(cell["status"] == "missing" for cell in row["cells"])
    assert widget.run_current_button.isEnabled()
    widget.refresh()  # Refresh/showEvent must retain the newly opened version.
    assert widget.current_model_row()["model_id"] == scored.fresh["id"]
    with patch.object(service, "evaluate", wraps=service.evaluate) as evaluate:
        widget.run_current_button.click()
        assert not widget.run_current_button.isEnabled()
        assert _wait_until(lambda: widget._run_task is None)
        assert {call.args for call in evaluate.call_args_list} == {
            (scored.fresh["id"], scored.eos["id"]), (scored.fresh["id"], scored.efv["id"]),
        }
    row = widget.current_model_row()
    assert row["model_id"] == scored.fresh["id"] and row["comparison_complete"]
    assert widget.run_current_button.isEnabled()


def test_unsaved_viewer_to_temporary_validation_then_save(scored, tmp_path, monkeypatch):
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    service = scored.service
    monkeypatch.setenv(library_module.ENV_VAR, str(service.library.root))
    path = tmp_path / "unsaved.txt"
    path.write_text(MODEL_HEADER + "unsaved")
    stub = _viewer_stub(None, [])
    stub.nep_result_data = SimpleNamespace(nep_txt_path=path, data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)
    original_count = len(service.models())
    assert stub._validation_model_id is None and stub.validation_save_action.isEnabled()
    assert not stub.validation_star_action.isEnabled()
    assert "unsaved" in stub.validation_status_action.text()

    widget = ModelLibraryWidget(service=service)
    stub._parent.open_model_scorecard = widget.open_scorecard
    stub.open_validation_scorecard()
    assert widget.current_model_row()["model_id"] == 0
    assert len(service.models()) == original_count  # entering the scorecard is still read-only
    assert not widget.baseline_button.isEnabled() and not widget.star_button.isEnabled()
    with patch.object(SaveModelDialog, "exec", return_value=0):
        widget.save_current()
    assert len(service.models()) == original_count  # cancelling Save leaves no project/version
    old_bytes = path.read_bytes()
    path.write_text(MODEL_HEADER + "changed while viewing")
    widget.run_current_button.click()
    assert _wait_until(lambda: widget._run_task is None)
    row = widget.current_model_row()
    temporary_id = row["model_id"]
    assert row["temporary"] and row["complete"] and temporary_id > 0
    assert service.get_model(temporary_id)["project_id"] is None
    assert service.model_file(temporary_id).read_bytes() == old_bytes
    stub.refresh_validation_chip()
    assert stub._validation_model_id == temporary_id
    assert stub.validation_status_action.text().startswith("Temporary")

    def accept(dialog):
        dialog.project_combo.setText("New project")
        dialog.name_edit.setText("v1")
        return 1

    with patch.object(SaveModelDialog, "exec", accept), patch("NepTrainKit.ui.views.model_library.MessageManager"):
        widget.save_current()
    saved = service.get_model(temporary_id)
    assert saved["name"] == "v1" and saved["project_id"] is not None
    assert saved["dataset_ref"] == {} and not saved["auto_captured"]
    assert widget.current_model_row()["model_id"] == temporary_id
    assert widget.current_model_row()["complete"] and widget.baseline_button.isEnabled()
    stub.refresh_validation_chip()
    assert not stub.validation_status_action.text().startswith("Temporary")


def test_save_dialog_requires_explicit_project_and_version(make_service):
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    host = QWidget()
    dialog = SaveModelDialog(host, make_service(), "v1")
    assert not dialog.yesButton.isEnabled()
    dialog.project_combo.setText("FeRh")
    assert dialog.yesButton.isEnabled()
    dialog.name_edit.clear()
    assert not dialog.yesButton.isEnabled()


@pytest.mark.parametrize("accept", [False, True])
def test_save_dialog_can_choose_library_without_retargeting_on_cancel(make_service, model_file, tmp_path, monkeypatch, accept):
    from NepTrainKit.config import Config
    from NepTrainKit.core.validation import Library, ValidationService
    from NepTrainKit.core.validation.service import ModelDraft
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog, save_model_to_project

    monkeypatch.delenv(library_module.ENV_VAR, raising=False)
    service = make_service()
    target = tmp_path / "chosen-library"
    previous = Config.get("validation", "library_path", "")
    Config.set("validation", "library_path", str(service.library.root))
    monkeypatch.setattr("NepTrainKit.ui.widgets.validation_dialogs.QFileDialog.getExistingDirectory", lambda *args: str(target))

    def choose(dialog):
        assert dialog.location_button.isEnabled()
        dialog._choose_library()
        dialog.project_combo.setText("Chosen project")
        dialog.name_edit.setText("v2")
        return int(accept)

    try:
        with patch.object(SaveModelDialog, "exec", choose):
            record = save_model_to_project(QWidget(), service, ModelDraft.from_file(model_file))
        destination = ValidationService(Library(target))
        assert service.models() == [] and service.projects() == []
        if accept:
            assert record["library_root"] == str(target.resolve())
            assert destination.get_model(record["id"])["name"] == "v2"
            assert Config.get("validation", "library_path") == str(target.resolve())
        else:
            assert record is None and destination.models() == [] and destination.projects() == []
            assert Config.get("validation", "library_path") == str(service.library.root)
    finally:
        Config.set("validation", "library_path", previous)


def test_viewer_save_can_add_a_second_project_without_reopening(model_file, make_service, monkeypatch):
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    service = make_service()
    monkeypatch.setenv(library_module.ENV_VAR, str(service.library.root))
    stub = _viewer_stub(None, [])
    stub.nep_result_data = SimpleNamespace(nep_txt_path=model_file, data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)

    def accept(dialog):
        dialog.project_combo.setText("A" if not service.projects() else "B")
        dialog.name_edit.setText("v1")
        return 1

    with patch.object(SaveModelDialog, "exec", accept), patch("NepTrainKit.ui.pages.show_nep.MessageManager"):
        stub.save_validation_model()
        first = stub._validation_model_id
        stub.save_validation_model()
    assert first != stub._validation_model_id
    assert len(service.models()) == 2 and len(service.projects()) == 2
    assert service.get_model(first)["content_hash"] == service.get_model(stub._validation_model_id)["content_hash"]


def test_unsaved_model_is_visible_without_any_suites(make_service, model_file):
    from NepTrainKit.core.validation.service import ModelDraft
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    service = make_service()
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(None, draft=ModelDraft.from_file(model_file))
    assert widget.current_model_row()["model_id"] == 0
    assert widget.save_button.isEnabled() and not widget.run_current_button.isEnabled()
    assert service.models() == [] and service.projects() == []

    def accept(dialog):
        dialog.project_combo.setText("First project")
        dialog.name_edit.setText("v1")
        return 1

    with patch.object(SaveModelDialog, "exec", accept), patch("NepTrainKit.ui.views.model_library.MessageManager"):
        widget.save_current()
    assert widget.current_model_row()["name"] == "v1"
    assert not widget.current_model_row()["temporary"]
    assert not widget.run_current_button.isEnabled()


def test_refresh_clears_stale_actions_when_library_cannot_open(scored):
    widget = ModelLibraryWidget(service=scored.service)
    widget.open_scorecard(scored.suite["id"], scored.new["id"])
    with patch.object(widget, "service", side_effect=OSError("unavailable")), patch("NepTrainKit.ui.views.model_library.MessageManager"):
        widget.refresh()
    assert widget.model.rowCount() == 0
    assert not widget.save_button.isEnabled() and not widget.run_current_button.isEnabled()


def test_partial_or_failed_suite_never_looks_complete(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    service = scored.service
    stub = _viewer_stub(scored.fresh["id"], [])
    service.evaluate(scored.fresh["id"], scored.eos["id"])
    stub.refresh_validation_chip()
    assert "1/2 sets complete" in stub.validation_status_action.text()
    assert _badge(stub) == "dot"
    with patch.object(service, "predictor_factory", side_effect=ValueError("test failure")):
        service.evaluate(scored.fresh["id"], scored.efv["id"])
    stub.refresh_validation_chip()
    assert "1 failed" in stub.validation_status_action.text() and _badge(stub) == "dot"
    service.evaluate(scored.fresh["id"], scored.efv["id"])
    stub.refresh_validation_chip()
    assert "0 regressions" in stub.validation_status_action.text() and _badge(stub) is None


def test_missing_baseline_runs_never_looks_comparable(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    service = scored.service
    service.set_baseline(scored.suite["id"], scored.fresh["id"])
    stub = _viewer_stub(scored.new["id"], [])
    stub.refresh_validation_chip()
    assert "baseline comparison incomplete" in stub.validation_status_action.text()
    assert _badge(stub) == "dot"
    service.evaluate(scored.fresh["id"], scored.eos["id"])
    stub.refresh_validation_chip()
    assert "baseline comparison incomplete" in stub.validation_status_action.text()


@pytest.mark.parametrize("action", ["refresh_validation_chip", "toggle_validation_star", "open_validation_scorecard"])
def test_library_switch_invalidates_viewer_ids(scored, tmp_path, monkeypatch, action):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    opened = []
    stub = _viewer_stub(scored.old["id"], opened)
    stub.refresh_validation_chip()
    other_service = ValidationService(Library(tmp_path / "other-lib"))
    other = _capture(other_service, tmp_path, "unrelated")
    assert other["id"] == scored.old["id"]
    monkeypatch.setenv(library_module.ENV_VAR, str(other_service.library.root))
    with patch("NepTrainKit.ui.pages.show_nep.MessageManager"):
        getattr(stub, action)()
    assert not other_service.get_model(other["id"])["starred"]
    assert stub._validation_model_id is None and not stub.validation_star_action.isEnabled()
    assert stub.validation_status_action.text() == "Validation: library changed"
    if action == "open_validation_scorecard":
        assert opened == [(None, None)]
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    stub.nep_result_data = SimpleNamespace(nep_txt_path=scored.old["source_path"], data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)
    assert not stub._validation_library_changed
    assert stub._validation_model_id is None
    assert stub._validation_draft.content_hash == scored.old["content_hash"]
    assert len(other_service.models()) == 1  # reopening does not save into the new library


def test_library_switch_does_not_star_an_old_scorecard_id(scored, tmp_path, monkeypatch):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    widget = ModelLibraryWidget()
    widget.open_scorecard(scored.suite["id"], scored.old["id"])
    other_service = ValidationService(Library(tmp_path / "other-lib"))
    other = _capture(other_service, tmp_path, "unrelated")
    monkeypatch.setenv(library_module.ENV_VAR, str(other_service.library.root))
    widget.toggle_star()
    assert not other_service.get_model(other["id"])["starred"]
    assert widget.current_model_row() is None


def test_viewer_menu_stars_the_current_version(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    stub = _viewer_stub(scored.fresh["id"], [])
    stub.refresh_validation_chip()
    assert stub.validation_star_action.isEnabled()
    assert stub.validation_star_action.text() == "Star this version"
    with patch("NepTrainKit.ui.pages.show_nep.MessageManager"):
        stub.toggle_validation_star()
        assert scored.service.get_model(scored.fresh["id"])["starred"] is True
        assert stub.validation_star_action.text() == "Unstar this version"
        stub.toggle_validation_star()
    assert scored.service.get_model(scored.fresh["id"])["starred"] is False
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    monkeypatch.setenv(library_module.ENV_VAR, str(tmp_path / "lib"))
    from NepTrainKit.config import Config
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    Config()
    widget = ShowNepWidget()
    yield widget
    widget.deleteLater()


def test_viewer_footer_has_compact_validation_entry(viewer):
    assert not viewer.validation_button.isHidden()
    actions = [action.text() for action in viewer.validation_menu.actions() if not action.isSeparator()]
    assert actions == [
        "Validation: no NEP model open",
        "Save to project…",
        "Star this version",
        "Open scorecard",
        "Import benchmark set…",
    ]
    assert not viewer.validation_status_action.isEnabled()  # a status line, not a command
    assert not viewer.validation_star_action.isEnabled()
    assert viewer.validation_count_badge.isHidden() and viewer.validation_dot_badge.isHidden()
    # Counts keep their natural width; only the left side stretches.
    layout = viewer.plot_widget_layout
    assert (layout.columnStretch(0), layout.columnStretch(1)) == (1, 0)


def test_viewer_elides_only_the_file_name(viewer):
    from PySide6.QtGui import QResizeEvent

    label = viewer.path_label
    name = "FeNiCoCrMn_high_entropy_alloy_train_v14.xyz"
    viewer._set_path_label_text("Current file: {file_name}", file_name=name)
    assert label.toolTip() == f"Current file: {name}"

    def at_width(width):
        label.resize(width, 30)
        QApplication.sendEvent(label, QResizeEvent(label.size(), label.size()))
        return label.text()

    metrics = label.fontMetrics()
    assert at_width(label.maximumWidth()) == f"Current file: {name}"
    roomy = at_width(metrics.horizontalAdvance("Current file: ") * 3)
    assert roomy.startswith("Current file: ") and "…" in roomy and roomy.endswith(".xyz")
    tight = at_width(metrics.horizontalAdvance("Current file: ") + 20)
    assert not tight.startswith("Current file") and tight  # the wording yields to the name


def test_viewer_capture_feeds_the_badge(scored, tmp_path, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    stub = _viewer_stub(None, [])
    model_path = tmp_path / "new" / "nep.txt"
    stub.nep_result_data = SimpleNamespace(nep_txt_path=model_path, data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)
    assert stub._validation_model_id == scored.new["id"]
    assert stub.validation_status_action.text().startswith("Validation: PbTe")
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


# -- settings ---------------------------------------------------------------------

def test_settings_cleanup_previews_before_deleting(make_service, tmp_path, monkeypatch):
    import NepTrainKit.ui.pages.settings as settings_module

    service = make_service()
    monkeypatch.setenv(library_module.ENV_VAR, str(service.library.root))
    orphan = service.library.put_bytes(b"orphan bytes")
    widget = settings_module.SettingsWidget(None)
    widget.refresh_library_usage()
    content = widget.library_location_card.contentLabel.text()
    assert str(service.library.root) in content and "used" in content and "NEPTRAINKIT_LIBRARY" in content
    assert not widget.library_location_card.button.isEnabled()
    with patch.object(settings_module.QFileDialog, "getExistingDirectory") as chooser:
        widget.choose_library_location()
    chooser.assert_not_called()

    answers = []

    class FakeBox:
        def __init__(self, title, text, parent):
            answers.append(text)

        def exec(self):
            return len(answers) > 1

    with patch.object(settings_module, "MessageBox", FakeBox):
        widget.clean_up_library()  # declined after the dry run
        assert service.library.has_object(orphan) and "1 files" in answers[0]
        widget.clean_up_library()
    assert not service.library.has_object(orphan)


def test_settings_switches_library_and_keeps_old_files(make_service, tmp_path, monkeypatch):
    from NepTrainKit.config import Config
    from NepTrainKit.core.validation import ValidationService
    import NepTrainKit.ui.pages.settings as settings_module

    monkeypatch.delenv(library_module.ENV_VAR, raising=False)
    service = make_service()
    model = _capture(service, tmp_path, "original")
    previous = Config.get("validation", "library_path", "")
    Config.set("validation", "library_path", str(service.library.root))
    target = tmp_path / "new-library"
    target.mkdir()
    try:
        widget = settings_module.SettingsWidget(None)
        widget.refresh_library_usage()
        assert widget.library_location_card.button.isEnabled()
        with patch.object(settings_module.QFileDialog, "getExistingDirectory", return_value=str(target)), patch.object(settings_module, "MessageManager") as messages:
            widget.choose_library_location()
        assert Config.get("validation", "library_path") == str(target.resolve())
        assert ValidationService().library.root == target.resolve()
        assert ValidationService().models() == []
        assert service.model_file(model["id"]).exists()
        assert str(target.resolve()) in widget.library_location_card.contentLabel.text()
        assert "restart" not in messages.send_info_message.call_args.args[0]

        with patch.object(settings_module.QFileDialog, "getExistingDirectory", return_value=""):
            widget.choose_library_location()
        assert Config.get("validation", "library_path") == str(target.resolve())
        broken = tmp_path / "broken-library"
        broken.mkdir()
        (broken / "library.db").write_bytes(b"not a database")
        with patch.object(settings_module.QFileDialog, "getExistingDirectory", return_value=str(broken)), patch.object(settings_module, "MessageManager") as messages:
            widget.choose_library_location()
        messages.send_error_message.assert_called_once()
        assert Config.get("validation", "library_path") == str(target.resolve())
    finally:
        Config.set("validation", "library_path", previous)


# -- localisation -----------------------------------------------------------------

def test_scorecard_translates_stored_run_errors_but_keeps_unknown_diagnostics():
    from NepTrainKit import i18n

    column = {"set_name": "EOS", "label": "RMSE", "unit": "meV/atom", "better": "lower", "protocol": "eos"}
    errors = [
        ("failed", "Library object 0123456789ab is missing from /lib."),
        ("unsupported", "This protocol does not support spin models yet."),
        ("failed", "native crash 0x1f"),
    ]
    cells = [{"value": None, "status": status, "error": error, "verdict": None} for status, error in errors]
    card = {
        "columns": [dict(column, set_id=f"s{i}") for i in range(len(cells))],
        "rows": [
            {"model_id": 1, "name": "v1", "starred": False, "is_baseline": False, "regressions": 0, "cells": cells}
        ],
    }
    model = ScorecardTableModel()
    model.set_scorecard(card)
    app = QApplication.instance()

    def tooltip(column_index):
        return model.data(model.index(0, column_index), Qt.ItemDataRole.ToolTipRole)

    i18n.install_translator(app, "zh_CN")
    try:
        assert tooltip(1) == "模型库对象 0123456789ab 不在 /lib 中。"
        assert tooltip(2) == "此验证协议暂不支持自旋模型。"
        assert tooltip(3) == "native crash 0x1f"  # unknown diagnostics stay readable, not hidden
    finally:
        i18n.install_translator(app, "en_US")
    assert tooltip(1) == "Library object 0123456789ab is missing from /lib."
