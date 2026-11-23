from unittest.mock import patch

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.audit import build_training_set_audit
from NepTrainKit.core.io.nep import WardReferenceResultData
from NepTrainKit.core.structure_inspection import inspect_structure
from NepTrainKit.core.ward import VIRIAL_ORDER_KEY, select_ward_residuals, ward_labels
from NepTrainKit.ui.pages.training_set_audit import TrainingSetAuditWidget
from NepTrainKit.ui.views.nep import NepResultPlotWidget
from NepTrainKit.ui.views.structure import StructureInfoWidget


def _frame(virial, *, order="", magnetic_force="mforce", force="0 0 -0.000006"):
    return (
        '1\nLattice="5 0 0 0 5 0 0 0 5" Config_type=test '
        f'Properties=species:S:1:pos:R:3:spin:R:3:{magnetic_force}:R:3 '
        f'{order} virial="{virial}"\nFe 0 0 0 0 1 0 {force}\n'
    )


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ward_data(tmp_path, app):
    xyz = tmp_path / "train.xyz"
    xyz.write_text(
        _frame("0 0 0 0 0 0 0 0.000010 0")   # ASE: A=(5, 0, 0), P=(3, 0, 0)
        + _frame("1 2 3 4 5 6")             # no antisymmetric information
        + _frame("0 0 0 0 0 0.000016 0 0 0", order=f"{VIRIAL_ORDER_KEY}=row_major",
                 magnetic_force="force_mag")
    )
    data = WardReferenceResultData(xyz, "column_major")
    with patch("NepTrainKit.core.io.base.NepCalculator") as calculator:
        data.load()
        calculator.assert_not_called()
    assert data.load_flag
    return data


def test_signed_ward_and_strict_threshold_keep_original_indices(ward_data):
    first, partial, last = ward_data.structure.all_data
    values = ward_labels(first)
    assert values.virial == pytest.approx((5.0, 0.0, 0.0))
    assert values.magnetic == pytest.approx((3.0, 0.0, 0.0))
    assert values.residual == pytest.approx((2.0, 0.0, 0.0))
    assert ward_labels(last).norm == pytest.approx(5.0)
    assert ward_labels(partial).magnetic == pytest.approx((3.0, 0.0, 0.0))
    assert ward_labels(partial).residual is None
    assert first.additional_fields[VIRIAL_ORDER_KEY] == "row_major"
    assert ward_data.datasets[0].structure_index.tolist() == [0, 2]
    assert select_ward_residuals(enumerate([first, partial, last]), values.norm) == ([2], 2)

    del first.additional_fields[VIRIAL_ORDER_KEY]
    assert ward_labels(first).issues == ("Virial order is unconfirmed",)
    first.atomic_properties["force_mag"] = -first.atomic_properties["mforce"]
    assert "mforce and force_mag disagree" in ward_labels(first).issues


def test_ward_filter_replaces_selection_and_can_be_undone(ward_data, app, monkeypatch):
    widget = NepResultPlotWidget()
    widget.set_dataset(ward_data)
    ward_data.select([1])
    ward_data.remove([0])
    def accept_threshold(dialog):
        assert dialog.doubleSpinBox.value() == 0.25
        dialog.doubleSpinBox.setValue(4.0)
        return True

    monkeypatch.setattr("NepTrainKit.ui.views.nep.GetFloatMessageBox.exec", accept_threshold)
    widget.select_by_ward_residual()
    for _ in range(50):
        if ward_data.select_index == {2}:
            break
        QTest.qWait(20)
    assert ward_data.select_index == {2}
    assert ward_data.undo_selection()
    assert ward_data.select_index == {1}


def test_audit_and_right_inspector_show_same_ward_values(ward_data, app):
    result = build_training_set_audit(ward_data)
    dimension = next(item for item in result.dimensions if item.id == "label_ranges")
    plot = next(plot for plot in dimension.plots if plot["id"] == "label_ranges:ward_residual")
    assert plot["labeled_count"] == 2
    assert plot["total_count"] == 3
    audit = TrainingSetAuditWidget()
    audit._result = result
    audit._active_plots = [plot]
    audit._show_selected_plot(0)
    assert audit.ward_table.item(0, 0).text() == "2"
    assert audit.ward_table.item(0, 2).text() == "5"
    assert audit.ward_table.item(2, 0).text() == "1"
    assert "Full nine-component virial required" in audit.ward_table.item(2, 2).toolTip()

    inspector = StructureInfoWidget()
    inspector.show_analysis(inspect_structure(ward_data.get_atoms(2)), None)
    assert inspector.ward_text.text() == "5 μeV/atom"
    assert "P = (+3, 0, 0)" in inspector.ward_text.toolTip()
    inspector.show_analysis(inspect_structure(ward_data.get_atoms(1)), None)
    assert inspector.ward_text.text() == "Unavailable"
    assert "P = (+3, 0, 0)" in inspector.ward_text.toolTip()
    inspector.set_analysis_pending()
    assert inspector.ward_caption.isHidden()
