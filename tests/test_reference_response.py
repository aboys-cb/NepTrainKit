"""Signed combinations, immutable recipes, reusable predictions and public UI paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.core.validation import Library, ValidationService, SetRejectedError
from NepTrainKit.core.validation.protocols import get_protocol
from NepTrainKit.core.validation.protocols.response import ReferenceResponseProtocol, ResponseSpec
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from test_validation_framework import eos_frames
from test_validation_accuracy import accuracy_case as accuracy_case


def spin_frame(**metadata):
    a = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5] * 3, pbc=True)
    a.set_initial_magnetic_moments([[0, 0, 2], [0, 0, -2]])
    a.info.update(metadata)
    return a


def four_states(geometry="A", coupling=0.001):
    result = []
    for si, sj in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        a = spin_frame(geometry=geometry, sign_i=si, sign_j=sj)
        a.info["energy"] = -10 + si * 0.01 + sj * 0.02 + coupling * si * sj
        result.append(a)
    return result


class SignedPredictor:
    backend = "analytic"

    def __init__(self):
        self.calls = 0

    def predict(self, frames):
        self.calls += 1
        energy = [
            -9 + a.info["sign_i"] * 0.01 + a.info["sign_j"] * 0.02 + 0.002 * a.info["sign_i"] * a.info["sign_j"]
            for a in frames
        ]
        return SimpleNamespace(
            energy=np.array(energy),
            forces=np.zeros((sum(map(len, frames)), 3)),
            structure_virials=np.zeros((len(frames), 9)),
        )

    def close(self):
        pass


def four_spec(**kwargs):
    return ResponseSpec(
        mode="four_state",
        group_fields=("geometry",),
        state_fields=("sign_i", "sign_j"),
        normalization="total",
        **kwargs,
    )


def test_four_states_preserve_signed_responses_and_validate_group_completeness():
    frames = four_states() + four_states("B", -0.001)
    protocol = ReferenceResponseProtocol(four_spec().to_dict())
    assert not protocol.check(frames)
    evaluation = protocol.evaluate(SignedPredictor(), frames)
    assert evaluation.details["reference"] == pytest.approx([4, -4])
    assert evaluation.details["predicted"] == pytest.approx([8, 8])
    assert evaluation.metrics["response_rmse"] == pytest.approx(np.sqrt(80))
    assert evaluation.metrics["sign_mismatches"] == 1
    assert [row["terms"] for row in evaluation.details["rows"]][0] == [(0, 1), (1, 1), (2, -1), (3, -1)]
    assert protocol.check(frames[:-1])[0].level == "error"
    skipped = ReferenceResponseProtocol(four_spec(skip_incomplete=True).to_dict())
    summary = skipped.summarize_reference(frames[:-1])
    assert len(summary["series"]) == 1 and summary["excluded"][0]["indices"] == [4, 5, 6]
    assert protocol.check(frames + [frames[0]])[0].level == "error"
    changed = [a.copy() for a in frames[:4]]
    changed[1].positions[0, 0] += 0.01
    assert "geometry" in protocol.check(changed)[0].message


def test_difference_scan_and_two_dimensional_response_use_reference_zero():
    frames = [
        spin_frame(geometry="A", direction=state, energy=energy)
        for state, energy in [("p", -10), ("q", -9.9998), ("t", -9.9996)]
    ]
    spec = ResponseSpec(
        mode="difference",
        group_fields=("geometry",),
        state_fields=("direction",),
        reference="p",
        normalization="element",
        element="Fe",
        energy_unit="μeV",
    )
    summary = ReferenceResponseProtocol(spec.to_dict()).summarize_reference(frames)
    assert summary["reference"] == pytest.approx([100, 200])
    assert summary["unit"] == "μeV/Fe"
    frames = [spin_frame(x=x, energy=x * x * 0.001, sign_i=1, sign_j=1) for x in [-1, 0, 1]]
    protocol = ReferenceResponseProtocol(ResponseSpec(x_field="x").to_dict())
    evaluation = protocol.evaluate(SignedPredictor(), frames)
    assert evaluation.details["reference"] == pytest.approx([0.5, 0, 0.5])
    assert evaluation.details["series"][0]["reference_extremum"]["curvature"] == pytest.approx(1)
    assert protocol.check(frames + [frames[0]])[0].level == "error"
    for i, a in enumerate(frames):
        a.info["y"] = i
    two = ReferenceResponseProtocol(ResponseSpec(x_field="x", second_axis="y").to_dict()).summarize_reference(frames)
    assert [row["second_axis"] for row in two["rows"]] == [0, 1, 2]


def test_cross_response_and_arbitrary_signed_terms():
    frames = []
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        frames.append(spin_frame(cell_state=a, spin_state=b, energy=-10 + 0.1 * a + 0.2 * b + 0.003 * a * b))
    spec = ResponseSpec(mode="cross", state_fields=("cell_state", "spin_state"), normalization="total")
    assert ReferenceResponseProtocol(spec.to_dict()).summarize_reference(frames)["reference"] == pytest.approx([3])
    spec = replace(spec, mode="combination", terms=(([1, 0], 2), ([0, 0], -2)))
    assert ReferenceResponseProtocol(spec.to_dict()).summarize_reference(frames[:2])["reference"] == pytest.approx(
        [200]
    )
    with pytest.raises(ValueError):
        ResponseSpec.from_dict(dict(spec.to_dict(), terms=[[[1, 0], float("nan")]]))


def test_pressure_reference_conversion_is_explicit():
    a = spin_frame(dft_pressure_kbar=10, energy=-10)
    spec = ResponseSpec(
        x_field="volume", quantity="pressure", reference_field="dft_pressure_kbar", reference_scale=0.1, relative=False
    )
    protocol = ReferenceResponseProtocol(spec.to_dict())
    virial = np.eye(3).reshape(1, 9) * a.get_volume() / len(a) / 160.21766208
    predictor = SimpleNamespace(predict=lambda frames: SimpleNamespace(structure_virials=virial))
    result = protocol.evaluate(predictor, [a])
    assert result.details["reference"] == pytest.approx([1])
    assert result.details["predicted"] == pytest.approx([1])
    assert result.metrics["response_rmse"] == pytest.approx(0)


def test_analysis_identity_separates_recipes_but_predictions_are_reused(tmp_path):
    engine = SignedPredictor()
    service = ValidationService(Library(tmp_path / "library"), predictor_factory=lambda *_: engine)
    path = tmp_path / "frames.xyz"
    write(path, four_states(), format="extxyz")
    model = tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\n")
    model = service.capture_model(model)
    spec = four_spec()
    first = service.import_set(path, name="four", protocol="response", provenance={}, analysis=spec.to_dict())
    second = service.import_set(
        path,
        name="per atom",
        protocol="response",
        provenance={},
        analysis=replace(spec, normalization="atom").to_dict(),
    )
    assert first["id"] != second["id"] and first["object_hash"] == second["object_hash"]
    run = service.evaluate(model["id"], first["id"])
    assert run["status"] == "succeeded", run["error"]
    run2 = service.evaluate(model["id"], second["id"])
    assert run2["status"] == "succeeded", run2["error"]
    assert run2["metrics"]["response_rmse"] == pytest.approx(run["metrics"]["response_rmse"] / 2)
    assert engine.calls == 1
    details = service.run_details(run2)["details"]
    assert details["prediction_source"]["kind"] == "prediction_cache"
    assert run2["environment"]["analysis"] == replace(spec, normalization="atom").to_dict()
    assert details["evaluation_scope"]["frame_indices"] == [0, 1, 2, 3]
    service.evaluate(model["id"], second["id"], force=True)
    assert engine.calls == 2
    (cached,) = list((service.library.root / "cache" / "predictions").glob("*.npz"))
    cached.write_bytes(b"corrupted")
    third = service.import_set(
        path, name="micro", protocol="response", provenance={}, analysis=replace(spec, energy_unit="μeV").to_dict()
    )
    assert service.evaluate(model["id"], third["id"])["status"] == "succeeded"
    assert engine.calls == 3


def test_spin_eos_accepts_fixed_order_and_rejects_moment_changes():
    frames = eos_frames()
    for a in frames:
        a.info["energy"] = a.info["_ref_energy"]
        a.calc = None
        a.set_initial_magnetic_moments([[0, 0, 2], [0, 0, -2]])
    protocol = get_protocol("eos")
    assert not any(i.level == "error" for i in protocol.check(frames))
    frames[-1].set_initial_magnetic_moments([[0, 0, 2], [0, 0, -1.9]])
    assert any("magnetic order" in i.message for i in protocol.check(frames))


def test_public_import_preview_saves_recipe_and_draws_small_cards(tmp_path):
    service = ValidationService(Library(tmp_path / "library"))
    path = tmp_path / "jd.xyz"
    frames = four_states()
    for a in frames:
        a.info.update(observable="JD", theta_deg=5, axis_i=1, axis_j=2, site_i=0, site_j=1)
    write(path, frames, format="extxyz")
    host = QWidget()
    host.resize(1180, 800)
    dialog = BenchmarkImportDialog(host)
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("response"))
    dialog.load_file(path)
    dialog.suite_combo.setText("Response")
    assert dialog.analysis_editor.preset.currentData() == "four_state"
    assert not dialog.validation_errors(), dialog.validation_errors()
    assert "1 complete groups" in dialog.preview_summary.text()
    (record,) = dialog.import_into(service, lambda _: True)
    assert record["source_ref"]["analysis"]["mode"] == "four_state"
    from NepTrainKit.ui.views.model_library import ModelLibraryWidget

    widget = ModelLibraryWidget(service=service)
    widget.resize(1180, 800)
    widget.open_scorecard(service.suites()[0]["id"])
    widget.show()
    QApplication.processEvents()
    assert len(widget.detail.eos_plots) == 1
    assert widget.detail.eos_cards[0].height() == 300
    widget.close()
    host.close()


def test_accuracy_slices_keep_partial_labels_and_counts(accuracy_case):
    case = accuracy_case
    for i, a in enumerate(case.frames):
        a.info["slice"] = "A" if i < 2 else "B"
    protocol = ReferenceResponseProtocol(ResponseSpec(mode="slices", group_fields=("slice",)).to_dict())
    result = protocol.evaluate(SimpleNamespace(predict=lambda _: case.prediction), case.frames)
    assert result.metrics["mforce_rmse"] == pytest.approx(4)
    first, second = result.details["accuracy_slices"]
    assert first["counts"]["energy"]["components"] == 1
    assert first["counts"]["mforce"]["components"] == 9
    assert second["metrics"]["force_rmse"] is None
    assert second["counts"]["mforce"]["evaluated_components"] == 0


def test_metadata_csv_maps_missing_cross_states_without_changing_source(tmp_path):
    from NepTrainKit.ui.widgets.reference_analysis import ReferenceAnalysisEditor

    frames = []
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        frames.append(spin_frame(frame_id=f"{a}-{b}", energy=-10 + 0.003 * a * b))
    editor = ReferenceAnalysisEditor()
    editor.set_frames(frames)
    editor.preset.setCurrentIndex(editor.preset.findData("cross"))
    editor.state_fields.setText("cell_state, spin_state")
    path = tmp_path / "metadata.csv"
    path.write_text("frame_id,cell_state,spin_state\n0-0,0,0\n1-0,1,0\n0-1,0,1\n1-1,1,1\n")
    editor.load_metadata(path)
    protocol = ReferenceResponseProtocol(editor.settings())
    assert protocol.summarize_reference(frames)["reference"] == pytest.approx([3])
    assert all("cell_state" not in a.info for a in frames)
    bad = tmp_path / "bad.csv"
    bad.write_text("frame_index,energy\n0,100\n")
    with pytest.raises(ValueError, match="never reference labels"):
        editor.load_metadata(bad)
    assert protocol.spec.frame_metadata["0"] == {"cell_state": 0, "spin_state": 0}
    editor.deleteLater()


def test_response_package_round_trip_preserves_analysis(tmp_path):
    from NepTrainKit.core.validation.package import build_package, import_package

    service = ValidationService(Library(tmp_path / "source"))
    data = tmp_path / "frames.xyz"
    write(data, four_states(), format="extxyz")
    spec = four_spec()
    record = service.import_set(data, name="pairs", protocol="response", provenance={}, analysis=spec.to_dict())
    suite = service.create_suite("response", [record["id"]])
    target = ValidationService(Library(tmp_path / "target"))
    imported = import_package(target, build_package(service, suite["id"]), name="shared")
    (restored,) = target.suite_sets(imported["id"])
    assert restored["source_ref"]["analysis"] == spec.to_dict()
    assert restored["summary"]["reference"] == pytest.approx([4])


def test_explicit_unit_spin_projections_recover_symmetric_and_chiral_parts():
    theta = 0.13
    reference = np.array([[0.002, 0.0007, 0], [0.0001, 0.003, 0], [0, 0, 0.004]])
    frames = []
    for ai, aj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for si, sj in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            a = spin_frame(geometry="same", axis_i=ai, axis_j=aj, site_i=0, site_j=1, sign_i=si, sign_j=sj)
            ui = np.array([0.0, 0.0, np.cos(theta)])
            uj = -ui.copy()
            ui[ai] = si * np.sin(theta)
            uj[aj] = sj * np.sin(theta)
            a.set_initial_magnetic_moments(np.array([ui, uj]) * 2)
            a.info["energy"] = -10 + ui @ reference @ uj
            frames.append(a)
    spec = ResponseSpec(
        mode="four_state",
        group_fields=("geometry", "axis_i", "axis_j", "site_i", "site_j"),
        state_fields=("sign_i", "sign_j"),
        normalization="total",
        extract_exchange=True,
    )
    protocol = ReferenceResponseProtocol(spec.to_dict())
    assert not protocol.check(frames)
    summary = protocol.summarize_reference(frames)
    projections = summary["exchange"]["projections"]
    assert [p["reference"] for p in projections] == pytest.approx([2, 0.7, 0.1, 3])
    (chiral,) = summary["exchange"]["symmetric_chiral"]
    assert chiral["reference_symmetric"] == pytest.approx(0.4)
    assert chiral["reference_d_projection"] == pytest.approx(0.3)
    assert chiral["direction"] == pytest.approx([0, 0, 1])
    broken = [a.copy() for a in frames]
    broken[0].set_initial_magnetic_moments(np.array([[0, 0, 2], [0, 0, -1.9]]))
    assert any(i.level == "error" for i in protocol.check(broken))


def test_batch_preview_and_atomic_registration(tmp_path):
    from NepTrainKit.ui.widgets.reference_batch import inspect_references

    paths = []
    for name in ["one", "two"]:
        path = tmp_path / (name + ".xyz")
        write(path, four_states(), format="extxyz")
        paths.append(path)
    entries = inspect_references(paths)
    # Default grouping suggestions for unmarked data are accuracy slices.
    assert all(not row["error"] and row["analysis"]["mode"] == "slices" for row in entries)
    service = ValidationService(Library(tmp_path / "library"))
    paths[1].write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        service.import_reference_batch(entries)
    assert not service.sets()
    write(paths[1], four_states(), format="extxyz")
    records = service.import_reference_batch(inspect_references(paths))
    assert records and service.sets()


def test_combined_difference_states_include_angle_only_points():
    frames = [
        spin_frame(geometry="A", direction="p", energy=-10),
        spin_frame(geometry="A", direction="q", energy=-9.999),
        spin_frame(geometry="A", plane="pq", angle_deg=15, energy=-9.9998),
    ]
    spec = ResponseSpec(
        mode="difference", group_fields=("geometry",), state_fields=("direction", "plane", "angle_deg"), reference="p"
    )
    summary = ReferenceResponseProtocol(spec.to_dict()).summarize_reference(frames)
    assert summary["reference"] == pytest.approx([0.5, 0.1])
    assert [row["label"] for row in summary["rows"]] == ["q", "pq / 15"]


def test_batch_dialog_loads_and_cancel_does_not_register(tmp_path):
    from NepTrainKit.ui.widgets.reference_batch import ReferenceBatchDialog
    from test_validation_ui import _wait_until

    path = tmp_path / "frames.xyz"
    write(path, four_states(), format="extxyz")
    service = ValidationService(Library(tmp_path / "library"))
    host = QWidget()
    dialog = ReferenceBatchDialog(host, service, [path], suite_name="preview")
    assert _wait_until(lambda: dialog.task is None)
    assert dialog.table.rowCount() == 1
    dialog.reject()
    assert service.sets() == [] and service.suites() == []
    host.close()


def test_cli_reference_analysis_recipe(tmp_path, capsys):
    import json
    from NepTrainKit.core.validation.cli import main

    path = tmp_path / "frames.xyz"
    write(path, four_states(), format="extxyz")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(four_spec().to_dict()))
    assert (
        main(
            [
                "--library",
                str(tmp_path / "library"),
                "--json",
                "import",
                str(path),
                "--name",
                "four",
                "--source",
                "DFT",
                "--analysis",
                str(recipe),
            ]
        )
        == 0
    )
    assert "four_state" in capsys.readouterr().out


def test_numeric_details_dialog_exposes_curvature_and_slice_counts(accuracy_case, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from NepTrainKit.ui.widgets.reference_response_plot import _show_values
    from NepTrainKit.ui.views.model_library import RunDetailPanel
    from qfluentwidgets import TableWidget

    case = accuracy_case
    for frame in case.frames:
        frame.info["group"] = "partial"
    protocol = ReferenceResponseProtocol(ResponseSpec(mode="slices", group_fields=("group",)).to_dict())
    summary = protocol.summarize_reference(case.frames)
    evaluation = protocol.evaluate(SimpleNamespace(predict=lambda _: case.prediction), case.frames)
    checked = []

    def inspect(dialog):
        tables = dialog.findChildren(TableWidget)
        assert len(tables) == 2
        stats = tables[-1]
        assert stats.rowCount() == 4 and stats.columnCount() == 7
        assert stats.item(3, 5).text() == "9" and stats.item(3, 6).text() == "9"
        checked.append(True)
        return 0

    monkeypatch.setattr(QDialog, "exec", inspect)
    panel = RunDetailPanel()
    _show_values(panel, summary, list(range(3)), [(1, "model", {"status": "succeeded"}, evaluation.details)])
    assert checked
    panel.close()
    panel.deleteLater()


@pytest.mark.parametrize(
    "bad", [{"mode": 3}, {"relative": "true"}, {"multiplicity": float("nan")}, {"frame_metadata": {"0": {"energy": 1}}}]
)
def test_analysis_recipe_rejects_invalid_json_types(bad):
    with pytest.raises(ValueError):
        ResponseSpec.from_dict(bad)


def test_batch_overlaps_are_symmetric_and_preserved_in_provenance(tmp_path):
    from NepTrainKit.ui.widgets.reference_batch import inspect_references

    one = tmp_path / "one.xyz"
    two = tmp_path / "two.xyz"
    write(one, four_states(), format="extxyz")
    write(two, four_states(), format="extxyz")
    rows = inspect_references([one, two])
    for row in rows:
        (other,) = row["provenance"]["batch_overlap"]["datasets"]
        assert other["exact_inputs"] == 4 and other["geometries"] == 4
