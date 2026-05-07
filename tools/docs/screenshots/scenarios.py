"""Scenario setup functions for documentation screenshots."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import CaptionLabel, MessageBoxBase

from NepTrainKit.main import create_app, create_main_window
from NepTrainKit.ui.widgets.dialog import (
    ArrowMessageBox,
    DatasetSummaryMessageBox,
    DFTD3MessageBox,
    DistributionInspectorMessageBox,
    EditInfoMessageBox,
    ExportFormatMessageBox,
    GetFloatMessageBox,
    GetIntMessageBox,
    IndexSelectMessageBox,
    LatticeRangeSelectMessageBox,
    RangeSelectMessageBox,
    ShiftEnergyMessageBox,
    SparseMessageBox,
)


@dataclass
class ScenarioContext:
    """Runtime objects shared by screenshot scenarios."""

    app: QApplication
    window: object
    repo_root: Path
    capture_widget: QWidget | None = None


def pump_events(app: QApplication, cycles: int = 80, delay_ms: int = 5) -> None:
    """Let Qt finish layout, paint, and pending signal delivery."""
    for _ in range(cycles):
        app.processEvents()
        if delay_ms > 0:
            QThread.msleep(delay_ms)


def wait_until(app: QApplication, predicate, *, cycles: int = 1200, delay_ms: int = 5) -> bool:
    """Process Qt events until predicate returns true or the cycle budget ends."""
    for _ in range(cycles):
        app.processEvents()
        if predicate():
            return True
        if delay_ms > 0:
            QThread.msleep(delay_ms)
    return False


def dismiss_transient_notifications(app: QApplication) -> None:
    """Close qfluentwidgets InfoBar notices before documentation capture."""
    for widget in app.allWidgets():
        if type(widget).__name__ == "InfoBar" and widget.isVisible():
            widget.close()
    pump_events(app, 20)


def create_context(repo_root: Path, window_size: tuple[int, int]) -> ScenarioContext:
    """Create a configured main window without entering the Qt event loop."""
    app = create_app(["capture-ui"])
    window = create_main_window(show=True)
    window.resize(*window_size)
    pump_events(app, 80)
    return ScenarioContext(app=app, window=window, repo_root=repo_root)


def prepare_nep_demo_data(ctx: ScenarioContext) -> Path:
    """Copy the tracked NEP fixture to a scratch directory before loading it."""
    source_dir = ctx.repo_root / "tests/data/nep"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing NEP demo fixture: {source_dir}")

    work_dir = ctx.repo_root / ".tmp/docs-screenshots/fixtures/nep"
    resolved_tmp = work_dir.resolve()
    resolved_root = (ctx.repo_root / ".tmp/docs-screenshots").resolve()
    if resolved_root not in resolved_tmp.parents:
        raise RuntimeError(f"Refusing to replace unexpected screenshot fixture path: {work_dir}")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train.xyz", "nep.txt", "descriptor.npy", "energy.npy", "forces.npy", "virial.npy"):
        shutil.copy2(source_dir / name, work_dir / name)
    return work_dir / "train.xyz"


def show_nep_overview(ctx: ScenarioContext) -> None:
    """Prepare the default NEP Dataset Display page."""
    ctx.window.switchTo(ctx.window.show_nep_interface)
    pump_events(ctx.app, 100)
    data_path = prepare_nep_demo_data(ctx)
    ctx.window.show_nep_interface.set_work_path(str(data_path))

    def loaded() -> bool:
        data = getattr(ctx.window.show_nep_interface, "nep_result_data", None)
        return bool(data is not None and getattr(data, "load_flag", False))

    if not wait_until(ctx.app, loaded, cycles=1600):
        raise RuntimeError(f"Timed out loading demo dataset: {data_path}")
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 220)


def make_data_empty(ctx: ScenarioContext) -> None:
    """Prepare an empty Make Data workspace."""
    ctx.window.switchTo(ctx.window.make_data_interface)
    pump_events(ctx.app, 100)


def make_data_lattice_strain(ctx: ScenarioContext) -> None:
    """Prepare a Make Data quickstart scene with a configured strain card."""
    make_data_empty(ctx)
    card = ctx.window.make_data_interface.add_card("CellStrainCard")
    if card is None:
        raise RuntimeError("CellStrainCard is not registered")
    card.strain_x_frame.set_input_value([-2, 2, 1])
    card.strain_y_frame.set_input_value([-2, 2, 1])
    card.strain_z_frame.set_input_value([-2, 2, 1])
    pump_events(ctx.app, 100)


def _show_dialog(ctx: ScenarioContext, dialog: QWidget, *, width: int | None = None) -> None:
    """Show a dialog and mark it as the screenshot target."""
    if width is not None:
        dialog.resize(width, dialog.height())
    dialog.show()
    pump_events(ctx.app, 120)
    ctx.capture_widget = getattr(dialog, "widget", dialog)


def show_nep_index_dialog(ctx: ScenarioContext) -> None:
    _show_dialog(ctx, IndexSelectMessageBox(ctx.window, "Specify index or slice"))


def show_nep_range_dialog(ctx: ScenarioContext) -> None:
    _show_dialog(ctx, RangeSelectMessageBox(ctx.window, "Specify x/y range"))


def show_nep_lattice_dialog(ctx: ScenarioContext) -> None:
    _show_dialog(ctx, LatticeRangeSelectMessageBox(ctx.window, "Specify lattice parameters range"))


def show_nep_max_error_dialog(ctx: ScenarioContext) -> None:
    dialog = GetIntMessageBox(ctx.window, "Enter the number of maximum-error structures")
    dialog.intSpinBox.setValue(10)
    _show_dialog(ctx, dialog)


def show_nep_sparse_dialog(ctx: ScenarioContext) -> None:
    dialog = SparseMessageBox(ctx.window, "Sparse samples")
    dialog.intSpinBox.setValue(100)
    dialog.doubleSpinBox.setValue(0.05)
    _show_dialog(ctx, dialog)


def show_nep_force_dialog(ctx: ScenarioContext) -> None:
    dialog = GetFloatMessageBox(ctx.window, "Net force threshold")
    dialog.doubleSpinBox.setValue(0.1)
    _show_dialog(ctx, dialog)


def show_nep_edit_info_dialog(ctx: ScenarioContext) -> None:
    dialog = EditInfoMessageBox(ctx.window)
    dialog.add_tag("Config_type", "bulk")
    dialog.add_tag("source", "candidate_pool")
    _show_dialog(ctx, dialog)


def show_nep_shift_dialog(ctx: ScenarioContext) -> None:
    dialog = ShiftEnergyMessageBox(ctx.window, "Group regex patterns (comma separated)")
    dialog.groupEdit.setText("bulk.*,surface.*")
    dialog.genSpinBox.setValue(200)
    dialog.sizeSpinBox.setValue(20)
    dialog.tolSpinBox.setValue(0.0001)
    _show_dialog(ctx, dialog)


def show_nep_dftd3_dialog(ctx: ScenarioContext) -> None:
    dialog = DFTD3MessageBox(ctx.window, "DFTD3 correction")
    dialog.functionEdit.setCurrentText("pbe")
    _show_dialog(ctx, dialog)


def show_nep_summary_dialog(ctx: ScenarioContext) -> None:
    summary = {
        "data_file": "candidate_pool_clean.xyz",
        "model_file": "nep.txt",
        "group_by": "tag",
        "total": 128,
        "groups": [
            {"name": "bulk", "count": 64, "fraction": 0.5},
            {"name": "surface", "count": 48, "fraction": 0.375},
            {"name": "defect", "count": 16, "fraction": 0.125},
        ],
    }
    _show_dialog(ctx, DatasetSummaryMessageBox(ctx.window, summary))


def show_nep_distribution_dialog(ctx: ScenarioContext) -> None:
    dialog = DistributionInspectorMessageBox(ctx.window, data=None, canvas_type="pyqtgraph")
    dialog.resize(760, 430)
    _show_dialog(ctx, dialog)


def show_nep_arrow_dialog(ctx: ScenarioContext) -> None:
    dialog = ArrowMessageBox(ctx.window, props=["forces", "magmom", "dipole"])
    _show_dialog(ctx, dialog)


def show_nep_export_format_dialog(ctx: ScenarioContext) -> None:
    _show_dialog(ctx, ExportFormatMessageBox(ctx.window, "xyz"))


def show_nep_drop_bad_dialog(ctx: ScenarioContext) -> None:
    dialog = MessageBoxBase(ctx.window)
    dialog.titleLabel = CaptionLabel("Confirm", dialog)
    dialog.contentLabel = CaptionLabel(
        "This will delete 1 structures marked as bad.\nDo you want to continue?",
        dialog,
    )
    dialog.contentLabel.setWordWrap(True)
    dialog.viewLayout.addWidget(dialog.titleLabel)
    dialog.viewLayout.addWidget(dialog.contentLabel)
    dialog.widget.setMinimumWidth(360)
    dialog.yesButton.setText("OK")
    dialog.cancelButton.setText("Cancel")
    _show_dialog(ctx, dialog)


RUNNERS: dict[str, Callable[[ScenarioContext], None]] = {
    "show_nep_overview": show_nep_overview,
    "show_nep_index_dialog": show_nep_index_dialog,
    "show_nep_range_dialog": show_nep_range_dialog,
    "show_nep_lattice_dialog": show_nep_lattice_dialog,
    "show_nep_max_error_dialog": show_nep_max_error_dialog,
    "show_nep_sparse_dialog": show_nep_sparse_dialog,
    "show_nep_force_dialog": show_nep_force_dialog,
    "show_nep_edit_info_dialog": show_nep_edit_info_dialog,
    "show_nep_shift_dialog": show_nep_shift_dialog,
    "show_nep_dftd3_dialog": show_nep_dftd3_dialog,
    "show_nep_summary_dialog": show_nep_summary_dialog,
    "show_nep_distribution_dialog": show_nep_distribution_dialog,
    "show_nep_arrow_dialog": show_nep_arrow_dialog,
    "show_nep_export_format_dialog": show_nep_export_format_dialog,
    "show_nep_drop_bad_dialog": show_nep_drop_bad_dialog,
    "make_data_empty": make_data_empty,
    "make_data_lattice_strain": make_data_lattice_strain,
}
