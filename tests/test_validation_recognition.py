"""Automatic proposals are exercised through preview and persisted imports."""

from dataclasses import replace

import numpy as np
import pytest
from ase.io import write
from PySide6.QtWidgets import QWidget

from NepTrainKit.core.validation.protocols.response import ReferenceResponseProtocol, ResponseSpec, suggest_response_settings
from NepTrainKit.core.validation.recognition import (
    AUTO_EOS_GROUP, eos_scan_groups, four_state_candidates, magnitude_scan, shear_scan,
)
from NepTrainKit.ui.widgets.reference_batch import inspect_references
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from test_reference_response import spin_frame
from test_validation_framework import _label, eos_frames, write_xyz
from test_validation_framework import make_service as make_service


def test_eos_geometry_groups_preview_and_batch_import_without_labels(make_service, tmp_path):
    frames = eos_frames()
    other = eos_frames()
    for a in other:
        fractional = a.get_scaled_positions()
        fractional[1, 0] += .02
        a.set_scaled_positions(fractional)
        _label(a, a.info["_ref_energy"])
    path = write_xyz(tmp_path / "two-scans.xyz", frames + other)
    groups = eos_scan_groups(frames + other)
    assert list(groups.values()) == [list(range(7)), list(range(7, 14))]
    host = QWidget()
    dialog = BenchmarkImportDialog(host)
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    dialog.load_file(path)
    dialog.suite_combo.setText("Automatic EOS")
    assert dialog.group_field_combo.currentData() == AUTO_EOS_GROUP
    assert len(dialog.eos_groups) == 2 and not dialog.validation_errors()
    records = dialog.import_into(make_service(), lambda _: True)
    assert len(records) == 2
    rows = inspect_references([path])
    assert rows[0]["protocol"] == "eos" and not rows[0]["error"]
    assert rows[0]["summary"]["groups"] == 2
    service = make_service()
    saved = service.import_reference_batch(rows)
    assert len(saved) == 2
    for record, indices in zip(saved, groups.values()):
        assert record["source_ref"]["eos_group"]["indices"] == indices
        assert len(service.set_frames(record["id"])) == 7
    dialog.close()
    host.close()


def test_eos_rejects_duplicate_volume_branches_and_separates_magnetic_order():
    assert eos_scan_groups(eos_frames() + eos_frames()) == {}
    frames, other = eos_frames(), eos_frames()
    for a in frames:
        a.set_initial_magnetic_moments([[0, 0, 2], [0, 0, 2]])
        _label(a, a.info["_ref_energy"])
    for a in other:
        a.set_initial_magnetic_moments([[0, 0, 2], [0, 0, -2]])
        _label(a, a.info["_ref_energy"])
    assert len(eos_scan_groups(frames + other)) == 2


def test_magnitude_scan_recovers_zero_endpoint_and_volume_grid(make_service, tmp_path):
    frames = []
    for moment in (0., 1., 2.):
        a = spin_frame(energy=-10 + moment * .01)
        a.set_initial_magnetic_moments([[0, 0, moment], [0, 0, -moment]])
        frames.append(a)
    settings = suggest_response_settings(frames)
    assert settings["mode"] == "scan" and settings["x_field"] == "moment_ratio"
    assert settings["frame_metadata"]["0"]["moment_ratio"] == 0.
    assert settings["frame_metadata"]["2"]["moment_ratio"] == 2.
    assert settings["frame_metadata"]["0"]["scan_reference_frame"] == 1
    summary = ReferenceResponseProtocol(settings).summarize_reference(frames)
    assert [row["x"] for row in summary["rows"]] == [0., 1., 2.]
    grid = []
    for scale in (.9, 1.1):
        for a in frames[1:]:
            a = a.copy()
            a.set_cell(a.cell * scale, scale_atoms=True)
            grid.append(a)
    settings = magnitude_scan(grid)
    assert settings["x_field"] == "volume_ratio" and settings["second_axis"] == "moment_ratio"
    summary = ReferenceResponseProtocol(settings).summarize_reference(grid)
    assert len(summary["rows"]) == 4
    assert all("volume_ratio" not in a.info for a in grid)
    path = tmp_path / "moments.xyz"
    write(path, grid, format="extxyz")
    rows = inspect_references([path])
    assert rows[0]["protocol"] == "response" and not rows[0]["error"]
    saved, = make_service().import_reference_batch(rows)
    assert saved["source_ref"]["analysis"]["second_axis"] == "moment_ratio"
    assert len(saved["summary"]["rows"]) == 4


def test_shear_scan_recovers_engineering_coordinate_and_reference(make_service, tmp_path):
    frames = []
    for shear in (0., -.02, .02):
        a = spin_frame(energy=-10 + shear * shear)
        deformation = np.eye(3)
        deformation[1, 2] = shear
        a.set_cell((deformation @ a.cell.array.T).T, scale_atoms=True)
        frames.append(a)
    settings = suggest_response_settings(frames)
    assert settings["x_field"] == "shear_yz"
    summary = ReferenceResponseProtocol(settings).summarize_reference(frames)
    assert [row["x"] for row in summary["rows"]] == pytest.approx([-.02, 0, .02])
    assert all(row["strain_reference_frame"] == 0 for row in settings["frame_metadata"].values())
    assert all("shear_yz" not in a.info for a in frames)
    path = tmp_path / "shear.xyz"
    write(path, frames, format="extxyz")
    rows = inspect_references([path])
    assert not rows[0]["error"]
    saved, = make_service().import_reference_batch(rows)
    assert saved["source_ref"]["analysis"]["x_field"] == "shear_yz"
    assert [r["x"] for r in saved["summary"]["rows"]] == pytest.approx([-.02, 0, .02])
    extra = frames[-1].copy()
    cell = extra.cell.array.copy()
    cell[0, 1] += .1
    extra.set_cell(cell, scale_atoms=True)
    assert shear_scan(frames + [extra]) is None


def actual_four_states():
    frames = []
    for angle, coupling in ((2.5, .001), (5., .002)):
        s, c = np.sin(np.radians(angle)), np.cos(np.radians(angle))
        for first, second in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            a = spin_frame(energy=-10 + coupling * first * second)
            a += a.copy()
            a.positions[2:] += [0, 2, 0]
            a.set_initial_magnetic_moments([[2*s*first, 0, 2*c], [0, 2*s*second, -2*c],
                                          [0, 0, 2], [0, 0, -2]])
            frames.append(a)
    return frames


def test_four_state_candidates_preserve_signed_combinations_and_are_reviewable(make_service, tmp_path):
    frames = actual_four_states()
    settings = suggest_response_settings(frames)
    assert settings["mode"] == "four_state" and not settings["extract_exchange"]
    summary = ReferenceResponseProtocol(settings).summarize_reference(frames)
    assert len(summary["series"]) == 2 and summary["reference"] == pytest.approx([4., 8.])
    assert {(m["site_i"], m["site_j"]) for m in settings["frame_metadata"].values()} == {(0, 1)}
    assert all("sign_i" not in a.info for a in frames)
    path = tmp_path / "pairs.xyz"
    write(path, frames, format="extxyz")
    rows = inspect_references([path])
    assert not rows[0]["error"]
    saved, = make_service().import_reference_batch(rows)
    assert saved["summary"]["reference"] == pytest.approx([4., 8.])
    assert not saved["source_ref"]["analysis"]["extract_exchange"]
    assert four_state_candidates(frames[:-1]) is None
    assert four_state_candidates(frames + [frames[0].copy()]) is None
    spec = ResponseSpec.from_dict(settings)
    assert not ReferenceResponseProtocol(replace(spec, extract_exchange=True).to_dict()).check(frames)
    for a in frames:
        a.info["observable"] = "JD"
    assert suggest_response_settings(frames)["frame_metadata"] == settings["frame_metadata"]


def test_four_state_product_takes_precedence_over_two_reversed_magnetic_orders():
    frames = []
    for first, second in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        a = spin_frame(energy=-10 + .001 * first * second)
        a.set_initial_magnetic_moments([[0, 0, 2 * first], [0, 0, 2 * second]])
        frames.append(a)
    settings = suggest_response_settings(frames)
    assert settings["mode"] == "four_state"
    assert ReferenceResponseProtocol(settings).summarize_reference(frames)["reference"] == pytest.approx([4.])
