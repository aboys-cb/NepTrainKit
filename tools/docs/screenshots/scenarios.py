"""Scenario setup functions for documentation screenshots."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from ase.io import read as ase_read
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import CaptionLabel, MessageBoxBase

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import RandomDopingParams
from NepTrainKit.core.cards.defect import VacancyDefectParams
from NepTrainKit.core.cards.filter import GeometryFilterParams
from NepTrainKit.core.cards.lattice import PerturbParams, SuperCellParams
from NepTrainKit.core.energy_shift import EnergyBaselinePreset, apply_energy_baseline
from NepTrainKit.i18n import install_translator
from NepTrainKit.main import create_app, create_main_window
from NepTrainKit.ui.widgets.dialog import (
    ArrowMessageBox,
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
    language: str
    capture_widget: QWidget | None = None

    def text(self, english: str, chinese: str) -> str:
        return chinese if self.language == "zh_CN" else english


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


def create_context(repo_root: Path, window_size: tuple[int, int], language: str) -> ScenarioContext:
    """Create a configured main window without entering the Qt event loop."""
    app = create_app(["capture-ui"])
    install_translator(app, language)
    window = create_main_window(show=True)
    window.resize(*window_size)
    pump_events(app, 80)
    return ScenarioContext(app=app, window=window, repo_root=repo_root, language=language)


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


def _focus_nep_plot(ctx: ScenarioContext, title: str) -> None:
    """Make one named NEP plot the large active plot."""
    canvas = ctx.window.show_nep_interface.graph_widget.canvas
    datasets = canvas.nep_result_data.datasets

    # The VisPy backend keeps the large axes fixed in slot zero and swaps the
    # datasets rendered in that slot when a thumbnail is double-clicked.
    if hasattr(canvas, "_plot_dataset_indices"):
        canvas._ensure_plot_dataset_indices()
        for dataset_index, dataset in enumerate(datasets):
            if getattr(dataset, "title", "") != title:
                continue
            target_slot = canvas._plot_dataset_indices.index(dataset_index)
            if target_slot != 0:
                old_dataset_index = canvas._plot_dataset_indices[0]
                canvas._plot_dataset_indices[0], canvas._plot_dataset_indices[target_slot] = (
                    dataset_index,
                    old_dataset_index,
                )
                main_axes = canvas.axes_list[0]
                preview_axes = canvas.axes_list[target_slot]
                canvas._render_plot(main_axes, datasets[dataset_index], True)
                canvas._render_plot(preview_axes, datasets[old_dataset_index], False)
                for axes in (main_axes, preview_axes):
                    axes.clear_overlays()
                canvas._refresh_current_axes_annotations()
                canvas._refresh_current_point_marker()
            pump_events(ctx.app, 100)
            return

    for axes, dataset in zip(canvas.axes_list, datasets):
        if getattr(dataset, "title", "") == title:
            canvas.set_current_axes(axes)
            pump_events(ctx.app, 100)
            return
    raise RuntimeError(f"NEP plot is unavailable: {title}")


def _expose_nep_toolbar_button(
    ctx: ScenarioContext,
    action_key: str,
    attribute_name: str,
) -> None:
    """Expose a visible toolbar button for annotation targeting."""
    toolbar = ctx.window.show_nep_interface.graph_toolbar
    action = toolbar._actions[action_key]
    for widget in toolbar.findChildren(QWidget):
        if widget.isVisible() and widget.toolTip() == action.text():
            setattr(ctx.window, attribute_name, widget)
            return
    raise RuntimeError(f"Toolbar button is hidden in the overflow: {action_key}")


def training_set_audit_overview(ctx: ScenarioContext) -> None:
    """Open the deterministic audit summary for the tracked NEP fixture."""
    show_nep_overview(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    ctx.window.open_training_set_audit(data)

    def loaded() -> bool:
        return getattr(ctx.window.training_set_audit_interface, "_result", None) is not None

    if not wait_until(ctx.app, loaded, cycles=2400):
        raise RuntimeError("Timed out building the Training Set Audit overview")
    audit = ctx.window.training_set_audit_interface
    audit.dataset_label.setText(
        audit.tr("{dataset} · {scope} scope · {count}/{total} structures").format(
            dataset="train.xyz",
            scope="active",
            count=25,
            total=25,
        )
    )
    audit.page_tabs.setCurrentIndex(0)
    ctx.capture_widget = audit
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 220)


def training_set_audit_structure_map(ctx: ScenarioContext) -> None:
    """Run on-demand phase evidence and show it on the composition map."""
    training_set_audit_overview(ctx)
    audit = ctx.window.training_set_audit_interface
    audit.page_tabs.setCurrentIndex(1)
    audit.data_map_tabs.setCurrentIndex(0)
    audit.composition_evidence_button.click()

    def analysis_complete() -> bool:
        result = getattr(audit, "_result", None)
        return bool(
            result is not None
            and result.phase_inventory is not None
            and audit.composition_view_selector.currentData() == "structural"
        )

    if not wait_until(ctx.app, analysis_complete, cycles=6000):
        raise RuntimeError("Timed out building structural-phase evidence for the audit map")
    audit.page_tabs.setCurrentIndex(1)
    audit.data_map_tabs.setCurrentIndex(0)
    ctx.capture_widget = audit
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 260)


def training_set_audit_magnetic_shares(ctx: ScenarioContext) -> None:
    """Show frame-normalized magnetic-type shares with deterministic spin data."""
    show_nep_overview(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    structures = data.structure.all_data
    for index, structure in enumerate(structures):
        atom_count = len(structure.atomic_properties["species"])
        if index < 10:
            spins = np.zeros((atom_count, 3), dtype=np.float32)
            spins[:, 2] = 2.0
            structure.atomic_properties["spin"] = spins
        elif index < 18:
            structure.atomic_properties["spin"] = np.zeros(
                (atom_count, 3), dtype=np.float32
            )
        else:
            structure.atomic_properties.pop("spin", None)

    ctx.window.open_training_set_audit(data)
    audit = ctx.window.training_set_audit_interface

    def loaded() -> bool:
        return getattr(audit, "_result", None) is not None

    if not wait_until(ctx.app, loaded, cycles=2400):
        raise RuntimeError("Timed out building the magnetic screenshot audit")
    audit.composition_evidence_button.click()

    def analysis_complete() -> bool:
        result = getattr(audit, "_result", None)
        return bool(
            result is not None
            and result.phase_inventory is not None
            and result.magnetic_inventory is not None
        )

    if not wait_until(ctx.app, analysis_complete, cycles=6000):
        raise RuntimeError("Timed out building magnetic evidence for the audit map")
    audit.page_tabs.setCurrentIndex(1)
    audit.data_map_tabs.setCurrentIndex(1)
    for row in range(audit.dimension_list.count()):
        item = audit.dimension_list.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == "magnetic_evidence":
            audit.dimension_list.setCurrentItem(item)
            break
    else:
        raise RuntimeError("Magnetic evidence dimension is unavailable")
    ctx.capture_widget = audit
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 260)


def make_data_empty(ctx: ScenarioContext) -> None:
    """Prepare an empty Make Data workspace."""
    ctx.window.switchTo(ctx.window.make_data_interface)
    pump_events(ctx.app, 100)
    ctx.window.make_data_run_button = _find_make_data_run_button(
        ctx,
        ctx.window.make_data_interface,
    )


def _find_make_data_run_button(ctx: ScenarioContext, page) -> QWidget:
    """Return the visible Run button used by the card tutorial annotations."""
    for widget in page.setting_group.setting_command.findChildren(QWidget):
        if widget.isVisible() and widget.toolTip() == ctx.text(
            "Run selected cards",
            "运行选中的卡片",
        ):
            return widget
    raise RuntimeError("Make Dataset run button is unavailable")


def _find_make_data_console_button(ctx: ScenarioContext, page, english: str, chinese: str) -> QWidget:
    """Return a visible console control by its translated tooltip."""
    expected = ctx.text(english, chinese)
    for widget in page.setting_group.setting_command.findChildren(QWidget):
        if widget.isVisible() and widget.toolTip() == expected:
            return widget
    raise RuntimeError(f"Make Dataset console button is unavailable: {expected}")


def _collapse_card(card) -> None:
    """Collapse one card without depending on its current height."""
    if getattr(card, "window_state", "expand") != "collapse":
        card.collapse()


def _prepare_card_system_workflow(ctx: ScenarioContext):
    """Build one realistic linear workflow with a two-branch card group."""
    make_data_empty(ctx)
    page = ctx.window.make_data_interface
    page.load_base_structure([str(ctx.repo_root / "tests/data/Si2.vasp")])

    supercell = page.add_card("SuperCellCard")
    group = page.add_card("CardGroup")
    perturb = page.add_card("PerturbCard")
    geometry_filter = page.add_card("GeometryFilterCard")
    if None in (supercell, group, perturb, geometry_filter):
        raise RuntimeError("Card-system tutorial workflow could not be created")

    supercell.set_params(SuperCellParams(mode="scale", super_scale=(2, 2, 2)))

    doping = CardManager.card_info_dict["RandomDopingCard"](group)
    doping.set_params(
        RandomDopingParams(
            rules=[
                {
                    "target": "Si",
                    "dopants": {"Ge": 1.0},
                    "use": "count",
                    "count": [1, 1],
                    "count_mode": "fixed",
                }
            ],
            doping_type="Exact",
            max_structures=2,
            use_seed=True,
            seed=42,
        )
    )
    vacancy = CardManager.card_info_dict["VacancyDefectCard"](group)
    vacancy.set_params(
        VacancyDefectParams(
            engine_type=1,
            num_condition=1,
            use_num=True,
            concentration_condition=0.0,
            count_mode="fixed",
            max_structures=2,
            use_seed=True,
            seed=42,
        )
    )
    group.add_card(doping)
    group.add_card(vacancy)
    page._connect_card_output_actions(doping)
    page._connect_card_output_actions(vacancy)

    perturb.set_params(
        PerturbParams(
            engine_type=1,
            max_distance=0.05,
            max_num=2,
            identify_organic=False,
            use_element_scaling=False,
            element_scalings={},
            use_seed=True,
            seed=42,
        )
    )
    geometry_filter.set_params(
        GeometryFilterParams(
            min_pair_distance=1.5,
            require_finite_cell=True,
        )
    )

    ctx.window.card_workflow_supercell = supercell
    ctx.window.card_workflow_group = group
    ctx.window.card_workflow_doping = doping
    ctx.window.card_workflow_vacancy = vacancy
    ctx.window.card_workflow_perturb = perturb
    ctx.window.card_workflow_filter = geometry_filter
    ctx.window.make_data_run_button = _find_make_data_run_button(ctx, page)
    ctx.window.make_data_paste_button = _find_make_data_console_button(
        ctx,
        page,
        "Create card(s) from clipboard JSON",
        "从剪贴板 JSON 创建卡片",
    )
    ctx.window.make_data_copy_button = _find_make_data_console_button(
        ctx,
        page,
        "Copy current workflow card JSON",
        "复制当前工作流卡片 JSON",
    )
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 160)
    return page, supercell, group, doping, vacancy, perturb, geometry_filter


def make_data_card_system_controls(ctx: ScenarioContext) -> None:
    """Show the controls shared by every Make Dataset card."""
    page, supercell, *_rest = _prepare_card_system_workflow(ctx)
    for card in list(page.workspace_card_widget.cards)[1:]:
        card.hide()
    supercell.setMinimumWidth(520)
    pump_events(ctx.app, 140)


def make_data_card_system_workflow(ctx: ScenarioContext) -> None:
    """Show the ordered chain and the independent branches inside Card Group."""
    _page, supercell, _group, doping, vacancy, perturb, geometry_filter = _prepare_card_system_workflow(ctx)
    for card in (supercell, vacancy, perturb, geometry_filter):
        _collapse_card(card)
    pump_events(ctx.app, 180)


def make_data_card_system_result(ctx: ScenarioContext) -> None:
    """Run the complete workflow and verify every intermediate count and formula."""
    page, supercell, group, doping, vacancy, perturb, geometry_filter = _prepare_card_system_workflow(ctx)
    page.run_card()

    def finished() -> bool:
        return bool(
            getattr(page, "_last_completed_card_index", None) == 3
            and len(getattr(geometry_filter, "result_dataset", []) or []) == 8
        )

    if not wait_until(ctx.app, finished, cycles=3200):
        counts = [
            len(getattr(card, "result_dataset", []) or [])
            for card in (supercell, group, perturb, geometry_filter)
        ]
        raise RuntimeError(f"Timed out running card-system tutorial: {counts}")

    if len(supercell.result_dataset) != 1 or len(supercell.result_dataset[0]) != 16:
        raise RuntimeError("Card-system tutorial supercell result is invalid")
    if len(doping.result_dataset) != 2 or any(
        atoms.get_chemical_symbols().count("Ge") != 1 for atoms in doping.result_dataset
    ):
        raise RuntimeError("Card-system tutorial doping branch is invalid")
    if len(vacancy.result_dataset) != 2 or any(len(atoms) != 15 for atoms in vacancy.result_dataset):
        raise RuntimeError("Card-system tutorial vacancy branch is invalid")
    if len(group.result_dataset) != 4 or len(perturb.result_dataset) != 8:
        raise RuntimeError(
            f"Card-system tutorial branch/perturb counts are invalid: "
            f"group={len(group.result_dataset)}, perturb={len(perturb.result_dataset)}"
        )
    if len(geometry_filter.result_dataset) != 8:
        raise RuntimeError("Card-system tutorial geometry filter result is invalid")

    for card in (supercell, doping, vacancy, perturb, geometry_filter):
        _collapse_card(card)
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 220)


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
    dialog = IndexSelectMessageBox(ctx.window, ctx.text("Specify index or slice", "指定索引或切片"))
    dialog.indexEdit.setText("0, 5, 10:13")
    dialog.checkBox.setChecked(True)
    _show_dialog(ctx, dialog)


def show_nep_range_dialog(ctx: ScenarioContext) -> None:
    _show_dialog(ctx, RangeSelectMessageBox(ctx.window, ctx.text("Specify x/y range", "指定 x/y 范围")))


def show_nep_lattice_dialog(ctx: ScenarioContext) -> None:
    _show_dialog(ctx, LatticeRangeSelectMessageBox(ctx.window, ctx.text("Specify lattice-parameter range", "指定晶格参数范围")))


def show_nep_max_error_dialog(ctx: ScenarioContext) -> None:
    dialog = GetIntMessageBox(ctx.window, ctx.text("Enter the number of maximum-error structures", "输入最大误差结构数量"))
    dialog.intSpinBox.setValue(5)
    _show_dialog(ctx, dialog)


def show_nep_sparse_dialog(ctx: ScenarioContext) -> None:
    dialog = SparseMessageBox(ctx.window, ctx.text("Configure farthest point sampling", "配置最远点采样"))
    dialog.intSpinBox.setValue(8)
    dialog.doubleSpinBox.setValue(0.01)
    dialog.strategyCombo.setCurrentIndex(dialog.strategyCombo.findData("global"))
    dialog.modeCombo.setCurrentIndex(dialog.modeCombo.findData("count"))
    dialog.descriptorCombo.setCurrentIndex(dialog.descriptorCombo.findData("reduced"))
    dialog.trainingPathEdit.clear()
    _show_dialog(ctx, dialog)


def fps_sampling_entry(ctx: ScenarioContext) -> None:
    """Show the descriptor plot and FPS toolbar entry."""
    show_nep_overview(ctx)
    _focus_nep_plot(ctx, "descriptor")
    _expose_nep_toolbar_button(ctx, "sparse_samples", "sparse_samples_button")


def fps_sampling_result(ctx: ScenarioContext) -> None:
    """Run fixed-count global FPS and show the selected representatives."""
    fps_sampling_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    selected, reverse = data.sparse_point_selection(
        n_samples=8,
        distance=0.01,
        descriptor_source="reduced",
        restrict_to_selection=False,
        training_path=None,
        sampling_mode="count",
        r2_threshold=0.9,
        selection_strategy="global",
    )
    if len(selected) != 8 or reverse:
        raise RuntimeError(f"Unexpected FPS demo result: selected={len(selected)}, reverse={reverse}")
    ctx.window.show_nep_interface.graph_widget.canvas.select_index(selected, reverse)
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 140)


def max_error_review_entry(ctx: ScenarioContext) -> None:
    """Show the active energy plot and maximum-error toolbar entry."""
    show_nep_overview(ctx)
    _focus_nep_plot(ctx, "energy")
    _expose_nep_toolbar_button(ctx, "find_max_error", "find_max_error_button")


def max_error_review_result(ctx: ScenarioContext) -> None:
    """Select the five structures with the largest energy error."""
    max_error_review_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    selected = data.energy.get_max_error_index(5)
    if len(selected) != 5:
        raise RuntimeError(f"Unexpected maximum-error demo result: {len(selected)}")
    ctx.window.show_nep_interface.graph_widget.canvas.select_index(selected, False)
    pump_events(ctx.app, 140)


def select_by_index_entry(ctx: ScenarioContext) -> None:
    """Show the stable index-selection entry on a loaded dataset."""
    show_nep_overview(ctx)
    _expose_nep_toolbar_button(ctx, "select_by_index", "select_by_index_button")


def select_by_index_result(ctx: ScenarioContext) -> None:
    """Select a reproducible mix of individual indices and a slice."""
    select_by_index_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    selected = data.select_structures_by_index("0, 5, 10:13", use_origin=True)
    if selected != [0, 5, 10, 11, 12]:
        raise RuntimeError(f"Unexpected index-selection result: {selected}")
    ctx.window.show_nep_interface.graph_widget.canvas.select_index(selected, False)
    pump_events(ctx.app, 140)


def suspicious_structure_scan_entry(ctx: ScenarioContext) -> None:
    """Show the geometry and net-force quality checks."""
    show_nep_overview(ctx)
    _expose_nep_toolbar_button(ctx, "find_non_physical", "find_non_physical_button")
    _expose_nep_toolbar_button(ctx, "check_net_force", "check_net_force_button")


def suspicious_structure_scan_result(ctx: ScenarioContext) -> None:
    """Run both quality checks against two deliberately corrupted demo frames."""
    suspicious_structure_scan_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data

    collision_structure = data.structure.all_data[0]
    collision_positions = np.array(collision_structure.positions, copy=True)
    collision_positions[1] = collision_positions[0] + np.array([0.01, 0.0, 0.0])
    collision_structure.positions = collision_positions
    data.structure._geometry_cache = None

    force_structure = data.structure.all_data[1]
    force_values = np.array(force_structure.forces, copy=True)
    force_values[0, 0] += 1.0
    force_structure.forces = force_values

    for _ in data.iter_non_physical_structure_indices(radius_coefficient=0.7):
        pass
    geometry_indices = data.consume_non_physical_structure_indices()
    for _ in data.iter_unbalanced_force_indices(threshold=0.01):
        pass
    force_indices = data.consume_unbalanced_force_indices()
    if 0 not in geometry_indices or 1 not in force_indices:
        raise RuntimeError(
            f"Unexpected quality-check result: geometry={geometry_indices}, force={force_indices}"
        )
    selected = sorted(set(geometry_indices).union(force_indices))
    ctx.window.show_nep_interface.graph_widget.canvas.select_index(selected, False)
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 160)


def edit_metadata_entry(ctx: ScenarioContext) -> None:
    """Show metadata editing after selecting a small structure subset."""
    show_nep_overview(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    selected = data.select_structures_by_index("0:3", use_origin=True)
    ctx.window.show_nep_interface.graph_widget.canvas.select_index(selected, False)
    _expose_nep_toolbar_button(ctx, "edit_info", "edit_info_button")
    pump_events(ctx.app, 120)


def edit_metadata_result(ctx: ScenarioContext) -> None:
    """Apply a real Config_type update to the selected structures."""
    edit_metadata_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    data.update_structure_metadata([], {"Config_type": "reviewed_bulk"}, {})
    updated = [data.structure.all_data[index].tag for index in (0, 1, 2)]
    if updated != ["reviewed_bulk"] * 3:
        raise RuntimeError(f"Unexpected metadata result: {updated}")
    page = ctx.window.show_nep_interface
    page.struct_index_spinbox.setValue(1)
    page.struct_index_spinbox.setValue(0)
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 160)


def distribution_analysis_result(ctx: ScenarioContext) -> None:
    """Run a real force-distribution analysis and show its histogram."""
    show_nep_overview(ctx)
    data = ctx.window.show_nep_interface.nep_result_data

    def run_analysis(request):
        for _ in data.iter_distribution_analysis(request=request):
            pass
        return data.get_distribution_analysis()

    dialog = DistributionInspectorMessageBox(
        ctx.window,
        data=data,
        run_analysis_callback=run_analysis,
        apply_selection_callback=lambda _indices, _mode: None,
        canvas_type="pyqtgraph",
    )
    explorer = dialog.explorer
    preferred_index = -1
    for index in range(explorer.fieldCombo.count()):
        key = str(explorer.fieldCombo.itemData(index) or "")
        if key == "atomic.force":
            preferred_index = index
            break
    if preferred_index < 0:
        for index in range(explorer.fieldCombo.count()):
            if "force" in str(explorer.fieldCombo.itemData(index) or ""):
                preferred_index = index
                break
    if preferred_index < 0:
        raise RuntimeError("Force field is unavailable for the distribution tutorial")
    explorer.fieldCombo.setCurrentIndex(preferred_index)
    explorer.binsSpin.setValue(30)
    explorer.curveCombo.setCurrentIndex(explorer.curveCombo.findData("none"))
    explorer._run_analysis()
    if explorer.metricCombo.count() == 0:
        raise RuntimeError("Distribution tutorial produced no metrics")
    dialog.resize(800, 650)
    _show_dialog(ctx, dialog)


def show_nep_force_dialog(ctx: ScenarioContext) -> None:
    dialog = GetFloatMessageBox(ctx.window, ctx.text("Net force threshold", "净力阈值"))
    dialog.doubleSpinBox.setValue(0.1)
    _show_dialog(ctx, dialog)


def show_nep_edit_info_dialog(ctx: ScenarioContext) -> None:
    dialog = EditInfoMessageBox(ctx.window)
    dialog.init_tags(["Config_type", "energy", "force"])
    dialog.add_tag("review_status", "checked")
    _show_dialog(ctx, dialog)


def show_nep_shift_dialog(ctx: ScenarioContext) -> None:
    dialog = ShiftEnergyMessageBox(
        ctx.window,
        ctx.text(
            "Use .* for one shared baseline; separate different Config_type baseline groups with semicolons.",
            "同一能量基线填写 .*；不同 Config_type 基线组用英文分号分隔。",
        ),
    )
    dialog.set_defaults([".*"], 100000, 40, 1e-8)
    _show_dialog(ctx, dialog)


def energy_baseline_shift_entry(ctx: ScenarioContext) -> None:
    """Show where to inspect energy errors and open baseline shifting."""
    show_nep_overview(ctx)
    _focus_nep_plot(ctx, "energy")
    _expose_nep_toolbar_button(
        ctx,
        "energy_baseline_shift",
        "energy_baseline_shift_button",
    )
    data = ctx.window.show_nep_interface.nep_result_data
    for structure in data.structure.now_data:
        structure.energy = float(structure.energy) - 5000.0 * len(structure.elements)
    data.sync_structures(["energy"])
    ctx.window.show_nep_interface.graph_widget.canvas.plot_nep_result()
    pump_events(ctx.app, 160)
    dismiss_transient_notifications(ctx.app)


def energy_baseline_shift_result(ctx: ScenarioContext) -> None:
    """Show the same demo dataset after removing a known atomic baseline."""
    energy_baseline_shift_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    structures = list(data.structure.now_data)
    elements = sorted({element for structure in structures for element in structure.elements})
    config_types = sorted({str(structure.tag) for structure in structures})
    baseline = EnergyBaselinePreset(
        alignment_mode="DFT_TO_NEP",
        elements=elements,
        group_to_ref={config_type: [-5000.0] * len(elements) for config_type in config_types},
        config_to_group={config_type: config_type for config_type in config_types},
    )
    apply_energy_baseline(structures, baseline)
    data.sync_structures(["energy"])
    ctx.window.show_nep_interface.graph_widget.canvas.plot_nep_result()
    pump_events(ctx.app, 160)
    dismiss_transient_notifications(ctx.app)


def show_nep_dftd3_dialog(ctx: ScenarioContext) -> None:
    dialog = DFTD3MessageBox(ctx.window, ctx.text("DFT-D3 correction", "DFT-D3 校正"))
    dialog.functionEdit.setCurrentText("pbe")
    dialog.d1SpinBox.setValue(12.0)
    dialog.d1cnSpinBox.setValue(6.0)
    dialog.modeCombo.setCurrentIndex(0)
    _show_dialog(ctx, dialog)


def dft_d3_entry(ctx: ScenarioContext) -> None:
    """Show the energy plot and DFT-D3 toolbar entry."""
    show_nep_overview(ctx)
    _focus_nep_plot(ctx, "energy")
    _expose_nep_toolbar_button(ctx, "dft_d3", "dft_d3_button")


def dft_d3_result(ctx: ScenarioContext) -> None:
    """Apply a real PBE DFT-D3 correction to the tracked demo dataset."""
    dft_d3_entry(ctx)
    data = ctx.window.show_nep_interface.nep_result_data
    data.apply_dft_d3_correction(
        mode=0,
        functional="pbe",
        cutoff=12.0,
        cutoff_cn=6.0,
    )
    ctx.window.show_nep_interface.graph_widget.canvas.plot_nep_result()
    dismiss_transient_notifications(ctx.app)
    pump_events(ctx.app, 180)


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
    dialog.titleLabel = CaptionLabel(ctx.text("Confirm", "确认"), dialog)
    dialog.contentLabel = CaptionLabel(
        ctx.text(
            "This will delete 1 structure marked as bad.\nDo you want to continue?",
            "这将删除 1 个标记为不良的结构。\n是否继续？",
        ),
        dialog,
    )
    dialog.contentLabel.setWordWrap(True)
    dialog.viewLayout.addWidget(dialog.titleLabel)
    dialog.viewLayout.addWidget(dialog.contentLabel)
    dialog.widget.setMinimumWidth(360)
    dialog.yesButton.setText(ctx.text("OK", "确定"))
    dialog.cancelButton.setText(ctx.text("Cancel", "取消"))
    _show_dialog(ctx, dialog)


RUNNERS: dict[str, Callable[[ScenarioContext], None]] = {
    "show_nep_overview": show_nep_overview,
    "training_set_audit_overview": training_set_audit_overview,
    "training_set_audit_structure_map": training_set_audit_structure_map,
    "training_set_audit_magnetic_shares": training_set_audit_magnetic_shares,
    "show_nep_index_dialog": show_nep_index_dialog,
    "show_nep_range_dialog": show_nep_range_dialog,
    "show_nep_lattice_dialog": show_nep_lattice_dialog,
    "show_nep_max_error_dialog": show_nep_max_error_dialog,
    "show_nep_sparse_dialog": show_nep_sparse_dialog,
    "fps_sampling_entry": fps_sampling_entry,
    "fps_sampling_result": fps_sampling_result,
    "max_error_review_entry": max_error_review_entry,
    "max_error_review_result": max_error_review_result,
    "select_by_index_entry": select_by_index_entry,
    "select_by_index_result": select_by_index_result,
    "suspicious_structure_scan_entry": suspicious_structure_scan_entry,
    "suspicious_structure_scan_result": suspicious_structure_scan_result,
    "edit_metadata_entry": edit_metadata_entry,
    "edit_metadata_result": edit_metadata_result,
    "distribution_analysis_result": distribution_analysis_result,
    "show_nep_force_dialog": show_nep_force_dialog,
    "show_nep_edit_info_dialog": show_nep_edit_info_dialog,
    "show_nep_shift_dialog": show_nep_shift_dialog,
    "energy_baseline_shift_entry": energy_baseline_shift_entry,
    "energy_baseline_shift_result": energy_baseline_shift_result,
    "show_nep_dftd3_dialog": show_nep_dftd3_dialog,
    "dft_d3_entry": dft_d3_entry,
    "dft_d3_result": dft_d3_result,
    "show_nep_distribution_dialog": show_nep_distribution_dialog,
    "show_nep_arrow_dialog": show_nep_arrow_dialog,
    "show_nep_export_format_dialog": show_nep_export_format_dialog,
    "show_nep_drop_bad_dialog": show_nep_drop_bad_dialog,
    "make_data_empty": make_data_empty,
    "make_data_lattice_strain": make_data_lattice_strain,
    "make_data_card_system_controls": make_data_card_system_controls,
    "make_data_card_system_workflow": make_data_card_system_workflow,
    "make_data_card_system_result": make_data_card_system_result,
}
