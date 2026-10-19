"""Model library scorecard, benchmark import dialog, viewer chip and settings."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import TransparentPushButton

from NepTrainKit.core.validation import SetTooLargeError
from NepTrainKit.core.validation import library as library_module
from NepTrainKit.ui.views.model_library import (
    VERDICT_COLORS,
    ModelLibraryWidget,
    ScorecardTableModel,
    summarize_model_status,
)
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from test_validation_framework import MODEL_HEADER, _import, efv_frames, eos_frames, make_service, write_xyz  # noqa: F401


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
    assert widget.model.rowCount() == 2
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
    from qfluentwidgets import Action

    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    def import_benchmark_set():
        if imported is not None:
            imported.append(True)

    stub = SimpleNamespace(
        validation_chip=TransparentPushButton(""),
        validation_star_action=Action("star"),
        _validation_model_id=model_id,
        _validation_status=None,
        _validation_starred=False,
        _parent=SimpleNamespace(
            open_model_scorecard=lambda suite, model: opened.append((suite, model)),
            import_benchmark_set=import_benchmark_set,
        ),
        tr=lambda text: text,
    )
    for name in (
        "refresh_validation_chip",
        "open_validation_scorecard",
        "import_validation_benchmark",
        "toggle_validation_star",
    ):
        setattr(stub, name, getattr(ShowNepWidget, name).__get__(stub))
    stub.validation_chip.clicked.connect(lambda: ShowNepWidget._on_validation_chip_clicked(stub))
    return stub


def test_viewer_chip_shows_regressions_and_opens_scorecard(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    opened = []
    stub = _viewer_stub(scored.new["id"], opened)
    stub.refresh_validation_chip()
    count = summarize_model_status(scored.service, scored.new["id"])["regressions"]
    assert stub.validation_chip.text() == f"Validation: PbTe · {count} regressions"
    stub.validation_chip.click()
    assert opened == [(scored.suite["id"], scored.new["id"])]

    stub._validation_model_id = scored.old["id"]
    stub.refresh_validation_chip()
    assert stub.validation_chip.text() == "Validation: PbTe · baseline"
    stub._validation_model_id = scored.fresh["id"]
    stub.refresh_validation_chip()
    assert stub.validation_chip.text() == "Validation: PbTe · not run"


def test_viewer_chip_stays_visible_and_guides_first_import(scored, tmp_path, monkeypatch):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    opened, imported = [], []
    # A model no suite covers: clicking guides to an import instead of hiding.
    nickel = tmp_path / "ni" / "nep.txt"
    nickel.parent.mkdir()
    nickel.write_text("nep4 1 Ni\ncutoff 8 4 74 10\n")
    stub = _viewer_stub(scored.service.capture_model(nickel)["id"], opened, imported)
    stub.validation_chip.show()
    stub.refresh_validation_chip()
    assert not stub.validation_chip.isHidden()  # refreshing never hides the entrance
    assert stub.validation_chip.text() == "Validation: no matching set"
    stub.validation_chip.click()
    assert imported == [True] and opened == []

    # No captured model: a plain entrance that opens the scorecard.
    stub._validation_model_id = None
    stub.refresh_validation_chip()
    assert stub.validation_chip.text() == "Validation" and not stub.validation_star_action.isEnabled()
    stub.validation_chip.click()
    assert opened == [(None, None)] and not stub.validation_chip.isHidden()

    # An empty library says it is not set up yet.
    empty = ValidationService(Library(tmp_path / "empty-lib"))
    monkeypatch.setenv(library_module.ENV_VAR, str(empty.library.root))
    path = tmp_path / "first" / "nep.txt"
    path.parent.mkdir()
    path.write_text(MODEL_HEADER)
    stub._validation_model_id = empty.capture_model(path)["id"]
    stub.refresh_validation_chip()
    assert stub.validation_chip.text() == "Validation: not set up"
    stub.validation_chip.close()


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


def test_viewer_builds_chip_with_menu(monkeypatch, tmp_path):
    monkeypatch.setenv(library_module.ENV_VAR, str(tmp_path / "lib"))
    from NepTrainKit.config import Config
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    Config()
    viewer = ShowNepWidget()
    try:
        assert not viewer.validation_chip.isHidden() and not viewer.validation_menu_button.isHidden()
        assert viewer.validation_chip.text() == "Validation"
        assert [action.text() for action in viewer.validation_menu.actions()] == [
            "Star this version",
            "Open scorecard",
            "Import benchmark set…",
        ]
        assert not viewer.validation_star_action.isEnabled()
    finally:
        viewer.deleteLater()


def test_viewer_capture_feeds_the_chip(scored, tmp_path, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    stub = _viewer_stub(None, [])
    model_path = tmp_path / "new" / "nep.txt"
    stub.nep_result_data = SimpleNamespace(nep_txt_path=model_path, data_xyz_path=None)
    ShowNepWidget._capture_model_version(stub)
    assert stub._validation_model_id == scored.new["id"]
    assert stub.validation_chip.text().startswith("Validation: PbTe")


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
