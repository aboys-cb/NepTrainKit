"""Visualization widgets and analysis helpers for NEP evaluation results."""

import json
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget, QProgressDialog
from loguru import logger
from qfluentwidgets import MessageBox

from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.core import MessageManager
from NepTrainKit.config import Config
from NepTrainKit.ui.canvas.canvas_factory import create_result_canvas, resolve_canvas_host_widget

from NepTrainKit.ui.widgets import (
    GetIntMessageBox,
    GetFloatMessageBox,
    SparseMessageBox,
    IndexSelectMessageBox,
    RangeSelectMessageBox,
    LatticeRangeSelectMessageBox,
    EditInfoMessageBox,
    ShiftEnergyMessageBox,
    DFTD3MessageBox,
    DistributionInspectorMessageBox,
    TrainingOverlayDialog,
)
from NepTrainKit.core.types import SearchType, CanvasMode
from NepTrainKit.ui.views import NepDisplayGraphicsToolBar
from NepTrainKit.core.energy_shift import (
    EnergyBaselinePreset,
    delete_energy_baseline_preset,
    list_energy_baseline_preset_names,
    load_energy_baseline_preset,
    save_energy_baseline_preset,
    suggest_group_patterns,
)

AUTO_VISPY_POINT_THRESHOLD = 100000
AUTO_VISPY_THRESHOLD_OPTION = "auto_vispy_total_point_threshold"
LARGE_DATASET_CANVAS_PROMPTED_OPTION = "large_dataset_canvas_prompted"
AUTO_VISPY_NOTICE_SHOWN_OPTION = "auto_vispy_notice_shown"
SPARSE_BACKGROUND_THRESHOLD = 2000


class NepResultPlotWidget(QWidget):
    """Plot widget that visualizes NEP evaluation results and provides analysis helpers.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget used to manage modality for dialogs and progress windows.

    Attributes
    ----------
    canvas : object
        Active plotting canvas for the NEP results (PyqtgraphCanvas or VispyCanvas).
    tool_bar : NepDisplayGraphicsToolBar
        Toolbar whose actions manipulate the canvas and underlying dataset.
    """

    structureIndexChanged = Signal(int)

    def __init__(self, parent=None):
        """Create the widget layout and load the canvas defined in user preferences.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget used for signal propagation and dialog ownership.
        """
        super().__init__(parent)
        self._parent = parent
        self.tool_bar: NepDisplayGraphicsToolBar
        self.draw_mode = False
        # self.setRenderHint(QPainter.Antialiasing, False)
        self._layout = QHBoxLayout(self)
        self.setLayout(self._layout)
        canvas_type = Config.get("widget", "canvas_type", CanvasMode.AUTO)

        self.last_figure_num = None
        self._distribution_inspector = None
        self._canvas_fallback_warned = False
        self._vispy_unavailable = False
        self._large_dataset_vispy_override = None
        self._overlay_dialog_refs: list[TrainingOverlayDialog] = []
        self.swith_canvas(canvas_type)

    @staticmethod
    def _canvas_mode_value(canvas_type: object) -> str:
        """Return a supported canvas mode string from config-like values."""
        text = str(canvas_type or "").strip().lower()
        if text in {CanvasMode.AUTO.value, "canvasmode.auto", CanvasMode.AUTO.name.lower()}:
            return CanvasMode.AUTO.value
        if text in {CanvasMode.VISPY.value, "canvasmode.vispy", CanvasMode.VISPY.name.lower()}:
            return CanvasMode.VISPY.value
        return CanvasMode.PYQTGRAPH.value

    @staticmethod
    def _canvas_dataset_index(canvas, axes, dataset) -> int | None:
        """Return the result-dataset index displayed by one backend axes."""
        try:
            axes_dataset = canvas.get_axes_dataset(axes)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        for index, item in enumerate(getattr(dataset, "datasets", []) or []):
            if item is axes_dataset:
                return index
        return None

    @classmethod
    def _capture_canvas_state(cls, canvas, dataset) -> dict:
        """Capture backend-neutral interaction state before replacing a canvas."""
        state = {
            "current_dataset_index": None,
            "ranges": {},
            "search_indices": set(getattr(canvas, "_search_highlight_indices", set()) or set()),
            "structure_index": int(getattr(canvas, "structure_index", 0) or 0),
            "draw_mode": bool(getattr(canvas, "draw_mode", False)),
        }
        current_axes = getattr(canvas, "current_axes", None)
        state["current_dataset_index"] = cls._canvas_dataset_index(canvas, current_axes, dataset)
        for axes in getattr(canvas, "axes_list", []) or []:
            dataset_index = cls._canvas_dataset_index(canvas, axes, dataset)
            if dataset_index is None:
                continue
            try:
                if hasattr(axes, "viewRange"):
                    x_range, y_range = axes.viewRange()
                    state["ranges"][dataset_index] = (
                        (float(x_range[0]), float(x_range[1])),
                        (float(y_range[0]), float(y_range[1])),
                    )
                    continue
                rect = axes.view.camera.rect
                state["ranges"][dataset_index] = (
                    (float(rect.left), float(rect.right)),
                    (float(rect.bottom), float(rect.top)),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return state

    @classmethod
    def _restore_canvas_state(cls, canvas, dataset, state: dict) -> None:
        """Restore focus, ranges, current point and overlays on a new backend."""
        canvas.structure_index = int(state.get("structure_index", 0))
        canvas.draw_mode = bool(state.get("draw_mode", False))
        current_dataset_index = state.get("current_dataset_index")
        axes_by_dataset = {}
        for axes in getattr(canvas, "axes_list", []) or []:
            dataset_index = cls._canvas_dataset_index(canvas, axes, dataset)
            if dataset_index is not None:
                axes_by_dataset[dataset_index] = axes

        current_axes = axes_by_dataset.get(current_dataset_index)
        if current_axes is not None:
            canvas.set_current_axes(current_axes)

        for dataset_index, ranges in state.get("ranges", {}).items():
            axes = axes_by_dataset.get(dataset_index)
            if axes is None:
                continue
            x_range, y_range = ranges
            try:
                if hasattr(axes, "setRange"):
                    axes.setRange(xRange=x_range, yRange=y_range, padding=0)
                else:
                    axes.view.camera.rect = (
                        x_range[0],
                        y_range[0],
                        x_range[1] - x_range[0],
                        y_range[1] - y_range[0],
                    )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        plot_current_point = getattr(canvas, "plot_current_point", None)
        if callable(plot_current_point):
            plot_current_point(state.get("structure_index", 0))
        search_indices = state.get("search_indices") or set()
        if search_indices:
            canvas.set_search_highlight(search_indices)

    @staticmethod
    def _dispose_canvas(canvas, host_widget) -> None:
        """Release Qt and OpenGL resources owned by a replaced canvas."""
        if canvas is not None:
            try:
                canvas.close()
            except (AttributeError, RuntimeError):
                pass
        if host_widget is not None:
            try:
                host_widget.setParent(None)
            except (AttributeError, RuntimeError):
                pass
            try:
                host_widget.deleteLater()
            except (AttributeError, RuntimeError):
                pass

    def _connect_canvas_signals(self, canvas) -> None:
        try:
            canvas.structureIndexChanged.connect(self.structureIndexChanged.emit)
        except (AttributeError, RuntimeError):
            pass

    def _disconnect_canvas_signals(self, canvas) -> None:
        try:
            canvas.structureIndexChanged.disconnect(self.structureIndexChanged.emit)
        except (AttributeError, RuntimeError):
            pass

    def swith_canvas(self, canvas_type: CanvasMode = "pyqtgraph", dataset=None, preserve_state=False) -> bool:
        """Instantiate the requested plotting backend and attach it to the layout.

        Parameters
        ----------
        canvas_type : CanvasMode, default=CanvasMode.PYQTGRAPH
            Backend identifier used to select between the supported canvases.
        """
        old_canvas = getattr(self, "canvas", None)
        old_host = getattr(self, "_canvas_host_widget", None)
        state = self._capture_canvas_state(old_canvas, dataset) if preserve_state and old_canvas is not None else None
        requested_mode = self._canvas_mode_value(canvas_type)
        candidate = None
        candidate_host = None
        try:
            candidate, fallback = create_result_canvas(canvas_type, self)
            candidate_host = resolve_canvas_host_widget(candidate)
            if dataset is not None:
                candidate.init_axes(len(dataset.datasets))
                candidate.set_nep_result_data(dataset)
                if state is not None:
                    candidate.structure_index = state["structure_index"]
                    candidate.draw_mode = state["draw_mode"]
                candidate.plot_nep_result(preserve_selection=bool(preserve_state))
                if state is not None:
                    self._restore_canvas_state(candidate, dataset, state)
        except Exception:
            logger.exception("Failed to prepare replacement result canvas")
            self._dispose_canvas(candidate, candidate_host)
            MessageManager.send_warning_message(
                self.tr("Failed to switch canvas backend; the current canvas was kept.")
            )
            return False

        if old_canvas is not None:
            self._disconnect_canvas_toolbar(old_canvas)
            self._disconnect_canvas_signals(old_canvas)
        if old_host is not None:
            try:
                self._layout.removeWidget(old_host)
            except (AttributeError, RuntimeError):
                pass

        self.canvas = candidate
        self._canvas_host_widget = candidate_host
        self._layout.addWidget(candidate_host)
        self._canvas_type = CanvasMode.PYQTGRAPH.value if fallback or requested_mode == CanvasMode.AUTO.value else requested_mode
        if fallback:
            self._vispy_unavailable = True
        if fallback and not self._canvas_fallback_warned:
            MessageManager.send_warning_message(
                "Current canvas backend is vispy, but vispy canvas failed to initialize; fallback to pyqtgraph."
            )
            self._canvas_fallback_warned = True
        self._connect_canvas_toolbar(self.canvas)
        self._connect_canvas_signals(self.canvas)
        if old_canvas is not None:
            self._dispose_canvas(old_canvas, old_host)
        if dataset is not None:
            self.last_figure_num = len(dataset.datasets)
        return True

    def _connect_canvas_toolbar(self, canvas):
        """Connect canvas-specific toolbar actions to the active canvas."""
        tool_bar = getattr(self, "tool_bar", None)
        if tool_bar is None:
            return
        tool_bar.panSignal.connect(canvas.pan)
        tool_bar.resetSignal.connect(canvas.auto_range)
        tool_bar.deleteSignal.connect(canvas.delete)
        tool_bar.revokeSignal.connect(canvas.revoke)
        tool_bar.undoSelectionSignal.connect(canvas.undo_selection)
        tool_bar.penSignal.connect(canvas.pen)
        canvas.tool_bar = tool_bar

    def _disconnect_canvas_toolbar(self, canvas):
        """Disconnect toolbar actions from a canvas that is being replaced."""
        tool_bar = getattr(self, "tool_bar", None)
        if tool_bar is None:
            return
        for signal, slot in (
            (tool_bar.panSignal, canvas.pan),
            (tool_bar.resetSignal, canvas.auto_range),
            (tool_bar.deleteSignal, canvas.delete),
            (tool_bar.revokeSignal, canvas.revoke),
            (tool_bar.undoSelectionSignal, canvas.undo_selection),
            (tool_bar.penSignal, canvas.pen),
        ):
            try:
                signal.disconnect(slot)
            except Exception:
                pass
        try:
            canvas.tool_bar = None
        except Exception:
            pass

    # def clear(self):
    #     self.canvas.clear_axes()
    # self.last_figure_num=None

    def set_tool_bar(self, tool):
        """Connect toolbar signals to canvas slots and store the toolbar reference.

        Parameters
        ----------
        tool : NepDisplayGraphicsToolBar
            Toolbar instance whose actions manipulate the canvas.
        """
        self.tool_bar: NepDisplayGraphicsToolBar = tool
        self._connect_canvas_toolbar(self.canvas)
        self.tool_bar.exportSignal.connect(self.export_descriptor_data)
        self.tool_bar.findMaxSignal.connect(self.find_max_error_point)
        self.tool_bar.discoverySignal.connect(self.find_non_physical_structures)
        self.tool_bar.sparseSignal.connect(self.sparse_point)
        self.tool_bar.shiftEnergySignal.connect(self.shift_energy_baseline)
        self.tool_bar.inverseSignal.connect(self.inverse_select)
        self.tool_bar.selectIndexSignal.connect(self.select_by_index)
        self.tool_bar.rangeSignal.connect(self.select_by_range)
        self.tool_bar.latticeRangeSignal.connect(self.select_by_lattice_range)
        self.tool_bar.dftd3Signal.connect(self.calc_dft_d3)
        self.tool_bar.editInfoSignal.connect(self.edit_structure_info)
        self.tool_bar.forceBalanceSignal.connect(self.check_force_balance)

    def closeEvent(self, event):
        """Ensure auxiliary non-modal inspectors are closed with this widget."""
        dlg = getattr(self, "_distribution_inspector", None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
            self._distribution_inspector = None
        super().closeEvent(event)

    def find_non_physical_structures(self):
        """Launch a background scan for structures that violate distance constraints."""
        data = self.canvas.nep_result_data
        if data is None:
            return
        radius = Config.getfloat("widget", "radius_coefficient", 0.7)
        progress_diag = QProgressDialog("", "Cancel", 0, data.structure.num, self._parent)
        thread = BackgroundTask(self._parent, show_tip=False)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle(self.tr("Finding non-physical structures"))
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        thread.succeeded.connect(lambda _result: self._apply_non_physical_selection(data))
        progress_diag.canceled.connect(thread.stop_work)
        thread.start_work(
            data.iter_non_physical_structure_indices,
            radius_coefficient=radius,
        )
        progress_diag.exec()

    def _apply_non_physical_selection(self, data):
        """Select any structures flagged by the background non-physical scan."""
        indices = data.consume_non_physical_structure_indices()
        if indices:
            self.canvas.select_index(indices, False)

    def find_max_error_point(self):
        """Select the highest-error structures on the active axes based on user input."""
        dataset = self.canvas.get_axes_dataset(self.canvas.current_axes)

        if dataset is None:
            return

        box = GetIntMessageBox(
            self._parent,
            self.tr("Enter an integer N to find the top N structures with the largest errors."),
        )
        n = Config.getint("widget", "max_error_value", 10)
        box.intSpinBox.setValue(n)

        if not box.exec():
            return
        nmax = box.intSpinBox.value()
        Config.set("widget", "max_error_value", nmax)
        index = dataset.get_max_error_index(nmax)

        self.canvas.select_index(index, False)

    def sparse_point(self):
        """Run farthest point sampling with simple and advanced strategies."""
        data = self.canvas.nep_result_data
        if data is None:
            return

        box = SparseMessageBox(self._parent, self.tr("Configure farthest point sampling"))
        n_samples_default = Config.getint("widget", "sparse_num_value", 10)
        distance_default = Config.getfloat("widget", "sparse_distance_value", 0.01)

        descriptor_source_default = Config.get("widget", "sparse_descriptor_source", "reduced").lower()
        sampling_mode_default = Config.get("widget", "sparse_sampling_mode", "count").lower()
        selection_strategy_default = Config.get(
            "widget",
            "sparse_selection_strategy",
            "global",
        ).lower()
        r2_threshold_default = Config.getfloat("widget", "sparse_r2_threshold", 0.9)
        physics_count_mode_default = Config.get(
            "widget",
            "sparse_physics_count_mode",
            "limit",
        ).lower()

        training_path_default = Config.get("widget", "sparse_training_path", "")

        box.intSpinBox.setValue(n_samples_default)
        box.doubleSpinBox.setValue(distance_default)

        box.descriptorCombo.setCurrentIndex(1 if descriptor_source_default == "raw" else 0)
        box.modeCombo.setCurrentIndex(1 if sampling_mode_default == "r2" else 0)
        box.r2SpinBox.setValue(r2_threshold_default if r2_threshold_default is not None else 0.9)
        strategy_index = box.strategyCombo.findData(selection_strategy_default)
        box.strategyCombo.setCurrentIndex(strategy_index if strategy_index >= 0 else 0)
        physics_count_mode_index = box.physicsCountModeCombo.findData(
            physics_count_mode_default
        )
        box.physicsCountModeCombo.setCurrentIndex(
            physics_count_mode_index if physics_count_mode_index >= 0 else 0
        )

        box.trainingPathEdit.setText(training_path_default)

        if not box.exec():
            return

        n_samples = box.intSpinBox.value()
        distance = box.doubleSpinBox.value()
        use_selection_region = bool(getattr(box, "regionCheck", None) and box.regionCheck.isChecked())

        selection_strategy = str(box.strategyCombo.currentData() or "global")
        descriptor_source = str(box.descriptorCombo.currentData() or "reduced")
        sampling_mode = str(box.modeCombo.currentData() or "count")
        physics_count_mode = str(
            box.physicsCountModeCombo.currentData() or "limit"
        )
        if selection_strategy in {"element_set", "physics"}:
            descriptor_source = "raw"
            sampling_mode = "count"
        r2_threshold = box.r2SpinBox.value()

        training_path = box.trainingPathEdit.text().strip()

        Config.set("widget", "sparse_num_value", n_samples)
        Config.set("widget", "sparse_distance_value", distance)

        Config.set("widget", "sparse_selection_strategy", selection_strategy)
        Config.set("widget", "sparse_physics_count_mode", physics_count_mode)
        if selection_strategy == "global":
            Config.set("widget", "sparse_descriptor_source", descriptor_source)
            Config.set("widget", "sparse_sampling_mode", sampling_mode)
        Config.set("widget", "sparse_r2_threshold", r2_threshold)

        Config.set("widget", "sparse_training_path", training_path)

        sampling_kwargs = dict(
            n_samples=n_samples,
            distance=distance,
            descriptor_source=descriptor_source,
            restrict_to_selection=use_selection_region,
            training_path=training_path or None,
            sampling_mode=sampling_mode,
            r2_threshold=r2_threshold,
            selection_strategy=selection_strategy,
            physics_count_mode=physics_count_mode,
        )
        candidate_count = int(
            getattr(getattr(data, "structure", None), "num", 0) or 0
        )
        if candidate_count >= SPARSE_BACKGROUND_THRESHOLD:
            progress_dialog = QProgressDialog(
                self.tr("Analyzing descriptor and physical coverage..."),
                "",
                0,
                0,
                self._parent,
            )
            progress_dialog.setCancelButton(None)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setWindowTitle(
                self.tr("Sampling representative structures")
            )
            task = BackgroundTask(self._parent, show_tip=False)
            task.finished.connect(progress_dialog.accept)
            task.start_work(data.sparse_point_selection, **sampling_kwargs)
            progress_dialog.exec()
            if task.outcome == "failed":
                MessageManager.send_warning_message(
                    self.tr("FPS sampling failed: {message}").format(
                        message=task.error_message
                    )
                )
                return
            if task.outcome != "succeeded" or task.result is None:
                return
            structures, reverse = task.result
        else:
            structures, reverse = data.sparse_point_selection(**sampling_kwargs)
        if structures:
            self.canvas.select_index(structures, reverse)
            if selection_strategy == "element_set":
                report = getattr(data, "_last_sparse_group_report", {}) or {}
                MessageManager.send_info_message(
                    self.tr(
                        "Balanced FPS selected {selected} structures across {groups} element sets."
                    ).format(selected=len(structures), groups=len(report))
                )
            elif selection_strategy == "physics":
                report = getattr(data, "_last_sparse_group_report", {}) or {}
                plan = getattr(data, "_last_sparse_physics_plan", None)
                covered = sum(
                    int(group.get("selected_count", 0)) > 0
                    for group in report.values()
                )
                MessageManager.send_info_message(
                    self.tr(
                        "Physics-aware FPS selected {selected} structures; covered "
                        "{covered}/{strata} element-set/phase/spin strata across "
                        "{element_sets} element sets."
                    ).format(
                        selected=len(structures),
                        covered=covered,
                        strata=getattr(plan, "group_count", len(report)),
                        element_sets=getattr(plan, "element_set_count", 0),
                    )
                )
            elif sampling_mode == "r2":
                coverage_r2 = getattr(data, "_last_sparse_coverage_r2", None)
                if coverage_r2 is not None:
                    MessageManager.send_info_message(
                        self.tr(
                            "Coverage-R^2 FPS selected {selected} structures; final coverage R^2: {score:.4f}."
                        ).format(
                            selected=len(structures),
                            score=float(coverage_r2),
                        )
                    )

            # Show training overlay if requested - pre-compute PCA then show dialog
            show_overlay = bool(getattr(box, "trainingOverlayCheck", None) and box.trainingOverlayCheck.isChecked())
            if show_overlay and training_path:
                pca_data = TrainingOverlayDialog.compute_pca_data(
                    training_path=training_path,
                    result_data=data,
                    selected_indices=structures,
                )
                if pca_data is not None:
                    overlay_dialog = TrainingOverlayDialog(
                        parent=None,
                        pca_data=pca_data,
                        canvas_type=str(Config.get("widget", "canvas_type", CanvasMode.PYQTGRAPH.value)),
                    )
                    overlay_refs = getattr(self, "_overlay_dialog_refs", None)
                    if overlay_refs is None:
                        overlay_refs = []
                        self._overlay_dialog_refs = overlay_refs
                    overlay_refs.append(overlay_dialog)
                    destroyed_signal = getattr(overlay_dialog, "destroyed", None)
                    if destroyed_signal is not None:
                        destroyed_signal.connect(
                            lambda *_args, dlg=overlay_dialog, refs=overlay_refs: (
                                refs.remove(dlg) if dlg in refs else None
                            )
                        )
                    overlay_dialog.show()

    def edit_structure_info(self):
        """Open the metadata editor for the current selection and apply the changes."""
        data = self.canvas.nep_result_data
        if data is None or len(data.select_index) == 0:
            MessageManager.send_info_message("No data selected!")
            return
        editable_tags = data.get_editable_structure_tags()
        box = EditInfoMessageBox(self._parent)

        box.init_tags(sorted(editable_tags))
        if not box.exec():
            return

        data.update_structure_metadata(box.remove_tag, box.new_tag_info, box.rename_tag_map)

    def export_descriptor_data(self):
        """Prompt for a destination file and export the selected descriptor rows."""
        data = self.canvas.nep_result_data
        if data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        path = call_path_dialog(self, "Choose a file save ", "file", default_filename="export_descriptor_data.out")
        if path:
            thread = BackgroundTask(self, show_tip=True, title="Exporting descriptor data")
            thread.start_work(data.export_descriptor_data, path)

    def _build_shift_energy_dialog(
        self,
        suggested_patterns: list[str],
        max_generations: int,
        population_size: int,
        convergence_tol: float,
    ) -> ShiftEnergyMessageBox:
        """Create and wire the shift-energy dialog."""
        box = ShiftEnergyMessageBox(
            self._parent,
            self.tr(
                "Use .* for one shared baseline; separate different Config_type baseline groups with semicolons."
            ),
        )
        preset_placeholder = "None"
        box.set_defaults(suggested_patterns, max_generations, population_size, convergence_tol)
        self._refresh_shift_preset_combo(box, preset_placeholder)
        box.importButton.clicked.connect(lambda: self._on_shift_preset_import(box, preset_placeholder))
        box.exportButton.clicked.connect(lambda: self._on_shift_preset_export(box, preset_placeholder))
        if hasattr(box, "deleteButton"):
            box.deleteButton.clicked.connect(lambda: self._on_shift_preset_delete(box, preset_placeholder))
        box.presetNameEdit.setText("")
        box.savePresetCheck.setChecked(False)
        box.presetCombo.currentTextChanged.connect(
            lambda selected_name: self._apply_selected_preset_to_dialog(
                box,
                selected_name,
                suggested_patterns,
                preset_placeholder,
            )
        )
        return box

    def _refresh_shift_preset_combo(self, box: ShiftEnergyMessageBox, preset_placeholder: str) -> None:
        """Refresh preset names in the shift-energy dialog."""
        box.set_preset_names(list_energy_baseline_preset_names(), preset_placeholder)

    def _on_shift_preset_import(self, box: ShiftEnergyMessageBox, preset_placeholder: str) -> None:
        """Import baseline preset from a JSON file."""
        path = call_path_dialog(
            self,
            "Import baseline preset",
            "file",
            file_filter="JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                preset_data = json.load(handle)
            preset = EnergyBaselinePreset.from_dict(preset_data)
            preset_name = preset.metadata.get("name") or Path(path).stem
            save_energy_baseline_preset(preset_name, preset)
            self._refresh_shift_preset_combo(box, preset_placeholder)
            box.presetCombo.setCurrentText(preset_name)
            MessageManager.send_info_message(f"Imported preset: {preset_name}")
        except Exception:  # noqa: BLE001
            MessageManager.send_warning_message("Failed to import baseline preset.")

    def _on_shift_preset_export(self, box: ShiftEnergyMessageBox, preset_placeholder: str) -> None:
        """Export selected baseline preset to a JSON file."""
        selected = box.presetCombo.currentText().strip()
        if selected in {"", preset_placeholder}:
            MessageManager.send_info_message("Please select a preset to export.")
            return
        preset = load_energy_baseline_preset(selected)
        if preset is None:
            MessageManager.send_warning_message("Preset not found.")
            return
        default_name = f"{selected}.json"
        path = call_path_dialog(
            self,
            "Export baseline preset",
            "file",
            default_filename=default_name,
            file_filter="JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(preset.to_dict(), handle, indent=2)
            MessageManager.send_info_message(f"Preset exported to {path}")
        except Exception:  # noqa: BLE001
            MessageManager.send_warning_message("Failed to export preset.")

    def _on_shift_preset_delete(self, box: ShiftEnergyMessageBox, preset_placeholder: str) -> None:
        """Delete selected baseline preset after user confirmation."""
        selected = box.presetCombo.currentText().strip()
        if selected in {"", preset_placeholder}:
            MessageManager.send_info_message("Please select a preset to delete.")
            return
        w = MessageBox("Delete baseline preset", f"Delete preset '{selected}'?", box)
        w.setClosableOnMaskClicked(True)
        if not w.exec():
            return
        if delete_energy_baseline_preset(selected):
            self._refresh_shift_preset_combo(box, preset_placeholder)
            box.presetCombo.setCurrentText(preset_placeholder)
            MessageManager.send_info_message(f"Deleted preset: {selected}")
        else:
            MessageManager.send_warning_message("Failed to delete preset.")

    def _apply_selected_preset_to_dialog(
        self,
        box: ShiftEnergyMessageBox,
        selected_name: str,
        suggested_patterns: list[str],
        preset_placeholder: str,
    ) -> None:
        """Load preset values into dialog inputs when selection changes."""
        selected_name = (selected_name or "").strip()
        if not selected_name or selected_name == preset_placeholder:
            box.apply_preset_to_inputs(None, suggested_patterns)
            return
        preset = load_energy_baseline_preset(selected_name)
        if preset is None:
            return
        box.apply_preset_to_inputs(preset, suggested_patterns)

    def _run_shift_energy_task(self, data, ref_index: list[int], values, selected_preset):
        """Run baseline shifting asynchronously and return captured baseline outputs."""
        Config.set("widget", "max_generation_value", values.max_generations)
        Config.set("widget", "population_size", values.population_size)
        Config.set("widget", "convergence_tol", values.convergence_tol)

        config_set = set(data.structure.get_all_config(SearchType.TAG))
        progress_diag = QProgressDialog("", self.tr("Cancel"), 0, len(config_set), self._parent)
        thread = BackgroundTask(self._parent, show_tip=False)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle(self.tr("Shift energies"))
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        progress_diag.canceled.connect(thread.stop_work)
        baseline_store: dict[str, object] = {}
        source_summary = {
            "config_types": list(config_set),
            "selected_refs": len(ref_index),
            "total_structures": len(data.structure.now_data),
        }
        thread.start_work(
            data.iter_shift_energy_baseline,
            values.group_patterns,
            values.alignment_mode,
            values.max_generations,
            values.population_size,
            values.convergence_tol,
            reference_indices=ref_index,
            precomputed_baseline=selected_preset,
            baseline_store=baseline_store,
            source_summary=source_summary,
        )
        progress_diag.exec()
        if thread.outcome == "canceled":
            return None
        if thread.outcome == "failed":
            MessageManager.send_warning_message(
                self.tr("Energy shift failed: {message}").format(message=thread.error_message)
            )
            return None
        return baseline_store

    def _post_shift_energy_messages(self, data, selected_preset, baseline_store, values) -> None:
        """Show user-facing messages after baseline shifting."""
        apply_stats = baseline_store.get("apply_stats")
        if selected_preset is not None and isinstance(apply_stats, dict):
            shifted = int(apply_stats.get("shifted_structures", 0) or 0)
            total = int(apply_stats.get("total_structures", len(data.structure.now_data)) or 0)
            unmatched = apply_stats.get("unmatched_config_types") or []
            if not isinstance(unmatched, list):
                unmatched = []
            if shifted == 0 and total > 0:
                examples = ", ".join(map(str, unmatched[:5]))
                suffix = f" Unmatched examples: {examples}" if examples else ""
                MessageManager.send_warning_message(
                    f"Preset did not match current dataset (0/{total} structures shifted).{suffix}"
                )
            elif unmatched:
                examples = ", ".join(map(str, unmatched[:5]))
                MessageManager.send_info_message(
                    f"Preset shifted {shifted}/{total} structures; unmatched examples: {examples}"
                )
        if selected_preset is None and values.save_preset:
            baseline = baseline_store.get("baseline")
            if baseline is not None:
                preset_name = values.preset_name or f"baseline_{len(list_energy_baseline_preset_names()) + 1}"
                baseline.metadata.setdefault("name", preset_name)
                save_energy_baseline_preset(preset_name, baseline)
                MessageManager.send_info_message(f"Baseline preset saved: {preset_name}")

    def shift_energy_baseline(self):
        """Fit and apply an energy baseline shift using the configured search strategy."""
        data = self.canvas.nep_result_data
        if data is None:
            return
        ref_index = list(data.select_index)
        max_generations = Config.getint("widget", "max_generation_value", 100000)
        population_size = Config.getint("widget", "population_size", 40)
        convergence_tol = Config.getfloat("widget", "convergence_tol", 1e-8)
        config_set = set(data.structure.get_all_config(SearchType.TAG))
        suggested = suggest_group_patterns(list(config_set))
        box = self._build_shift_energy_dialog(
            suggested,
            max_generations,
            population_size,
            convergence_tol,
        )
        if not box.exec():
            return
        values = box.collect_values()
        selected_preset = None
        if values.selected_preset_name:
            selected_preset = load_energy_baseline_preset(values.selected_preset_name)
            if selected_preset is None:
                MessageManager.send_warning_message("Selected preset unavailable.")
                return

        baseline_store = self._run_shift_energy_task(data, ref_index, values, selected_preset)
        if baseline_store is None:
            return
        self._post_shift_energy_messages(data, selected_preset, baseline_store, values)
        self.canvas.plot_nep_result()

    def calc_dft_d3(self):
        """Collect DFT-D3 parameters from the user and start the calculation asynchronously."""
        data = self.canvas.nep_result_data
        if data is None:
            return
        function = Config.get("widget", "functional", "scan")
        cutoff = Config.getfloat("widget", "cutoff", 12)
        cutoff_cn = Config.getfloat("widget", "cutoff_cn", 6)
        mode = Config.getint("widget", "d3_mode", 0)

        box = DFTD3MessageBox(self._parent, "DFT D3")
        box.functionEdit.setText(function)
        box.d1SpinBox.setValue(cutoff)
        box.d1cnSpinBox.setValue(cutoff_cn)
        box.modeCombo.setCurrentIndex(mode)
        if not box.exec():
            return

        mode = box.modeCombo.currentIndex()
        d3_cutoff = box.d1SpinBox.value()
        d3_cutoff_cn = box.d1cnSpinBox.value()
        functional = box.functionEdit.text().strip()
        Config.set("widget", "cutoff", d3_cutoff)
        Config.set("widget", "cutoff_cn", d3_cutoff_cn)
        Config.set("widget", "functional", functional)
        Config.set("widget", "d3_mode", mode)

        thread = BackgroundTask(self._parent, show_tip=True, title=self.tr("Calculating DFT-D3"))
        thread.start_work(data.apply_dft_d3_correction, mode, functional, d3_cutoff, d3_cutoff_cn)
        thread.succeeded.connect(lambda _result: self.canvas.plot_nep_result())

    def _run_distribution_analysis_task(self, data, request) -> dict:
        """Run distribution analysis in a worker thread and return the payload."""
        structures = getattr(data, "structure", None)
        total_structures = int(getattr(structures, "now_data", np.array([])).shape[0]) if structures is not None else 0
        progress_diag = QProgressDialog("", self.tr("Cancel"), 0, max(total_structures, 1), self._parent)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle(self.tr("Building distributions"))
        thread = BackgroundTask(self._parent, show_tip=False)
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        progress_diag.canceled.connect(thread.stop_work)
        thread.start_work(data.iter_distribution_analysis, request=request)
        progress_diag.exec()
        if thread.outcome == "canceled":
            return {}
        if thread.outcome == "failed":
            MessageManager.send_warning_message(
                self.tr("Distribution analysis failed: {message}").format(
                    message=thread.error_message
                )
            )
            return {}
        return data.get_distribution_analysis()

    def _apply_distribution_selection(self, data, indices: list[int], select_mode: str) -> None:
        """Apply selected structure indices from distribution bins to the canvas."""
        if not indices:
            MessageManager.send_info_message("No structures found in this bin.")
            return
        mode = str(select_mode or "replace").strip().lower()
        if mode == "add":
            self.canvas.select_index(indices, False)
            return
        if mode == "intersect":
            current = set(getattr(data, "select_index", set()))
            target = sorted(current.intersection(int(i) for i in indices))
            if current:
                self.canvas.select_index(list(current), True)
            if target:
                self.canvas.select_index(target, False)
            return

        current = list(getattr(data, "select_index", set()))
        if current:
            self.canvas.select_index(current, True)
        self.canvas.select_index(indices, False)

    def show_distribution_inspector(self):
        """Open the unified audit distribution explorer when available."""
        if hasattr(self._parent, "open_training_set_distribution"):
            self._parent.open_training_set_distribution()
            return
        data = self.canvas.nep_result_data
        if data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        dlg = self._distribution_inspector
        need_recreate = True
        if dlg is not None:
            try:
                need_recreate = getattr(dlg, "_data", None) is not data
            except RuntimeError:
                need_recreate = True

        if need_recreate:
            if dlg is not None:
                try:
                    dlg.close()
                except Exception:
                    pass
            host_parent = self._parent if isinstance(self._parent, QWidget) else self
            dlg = DistributionInspectorMessageBox(
                parent=host_parent,
                data=data,
                run_analysis_callback=lambda req: self._run_distribution_analysis_task(data, req),
                apply_selection_callback=lambda indices, mode: self._apply_distribution_selection(data, indices, mode),
                canvas_type=str(Config.get("widget", "canvas_type", CanvasMode.PYQTGRAPH.value)),
            )
            dlg.setWindowModality(Qt.WindowModality.NonModal)
            dlg.setModal(False)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self._distribution_inspector = dlg
            try:
                host_parent.destroyed.connect(lambda *_: dlg.close())
            except Exception:
                pass

            # Place inspector near the right side of the parent to avoid covering center plot.
            try:
                host = host_parent
                if host is not None:
                    geo = host.frameGeometry()
                    x = geo.x() + max(20, geo.width() - dlg.width() - 24)
                    y = geo.y() + 72
                    dlg.move(x, y)
            except Exception:
                pass

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def inverse_select(self):
        """Invert the current structure selection on the canvas."""
        self.canvas.inverse_select()

    def select_by_index(self):
        """Select structures by index, optionally mapping plot rows back to source indices."""
        data = self.canvas.nep_result_data
        if data is None:
            return
        box = IndexSelectMessageBox(self._parent, "Select structures by index")
        if not box.exec():
            return
        text_value = box.indexEdit.text().strip()
        use_origin = box.checkBox.isChecked()
        indices = data.select_structures_by_index(text_value, use_origin)
        if indices:
            self.canvas.select_index(indices, False)

    def select_by_range(self):
        """Select structures whose projected coordinates fall within a user-defined range."""
        data = self.canvas.nep_result_data
        if data is None:
            return
        dataset = self.canvas.get_axes_dataset(self.canvas.current_axes)
        if dataset is None or dataset.now_data.size == 0:
            return
        box = RangeSelectMessageBox(self._parent, "Select structures by range")
        box.xMinSpin.setValue(float(np.min(dataset.x)))
        box.xMaxSpin.setValue(float(np.max(dataset.x)))
        box.yMinSpin.setValue(float(np.min(dataset.y)))
        box.yMaxSpin.setValue(float(np.max(dataset.y)))
        if not box.exec():
            return
        x_min, x_max = box.xMinSpin.value(), box.xMaxSpin.value()
        y_min, y_max = box.yMinSpin.value(), box.yMaxSpin.value()
        logic_and = box.logicCombo.currentText() == "AND"
        indices = data.select_structures_by_range(dataset, x_min, x_max, y_min, y_max, logic_and)
        if indices:
            self.canvas.select_index(indices, False)

    def select_by_lattice_range(self):
        """Select structures by lattice parameters range."""
        data = self.canvas.nep_result_data
        if data is None:
            return
        structures = data.structure.now_data
        if structures.size == 0:
            return

        # Use cached lattice parameters from the dataset
        now_indices = data.structure.now_indices
        abcs = data.abcs[now_indices]
        angles = data.angles[now_indices]

        box = LatticeRangeSelectMessageBox(self._parent, "Select structures by lattice range")
        box.aMinSpin.setValue(float(np.min(abcs[:, 0])))
        box.aMaxSpin.setValue(float(np.max(abcs[:, 0])))
        box.bMinSpin.setValue(float(np.min(abcs[:, 1])))
        box.bMaxSpin.setValue(float(np.max(abcs[:, 1])))
        box.cMinSpin.setValue(float(np.min(abcs[:, 2])))
        box.cMaxSpin.setValue(float(np.max(abcs[:, 2])))

        box.alphaMinSpin.setValue(float(np.min(angles[:, 0])))
        box.alphaMaxSpin.setValue(float(np.max(angles[:, 0])))
        box.betaMinSpin.setValue(float(np.min(angles[:, 1])))
        box.betaMaxSpin.setValue(float(np.max(angles[:, 1])))
        box.gammaMinSpin.setValue(float(np.min(angles[:, 2])))
        box.gammaMaxSpin.setValue(float(np.max(angles[:, 2])))

        if not box.exec():
            return

        a_range = (box.aMinSpin.value(), box.aMaxSpin.value())
        b_range = (box.bMinSpin.value(), box.bMaxSpin.value())
        c_range = (box.cMinSpin.value(), box.cMaxSpin.value())
        alpha_range = (box.alphaMinSpin.value(), box.alphaMaxSpin.value())
        beta_range = (box.betaMinSpin.value(), box.betaMaxSpin.value())
        gamma_range = (box.gammaMinSpin.value(), box.gammaMaxSpin.value())

        indices = data.select_structures_by_lattice_range(
            a_range, b_range, c_range, alpha_range, beta_range, gamma_range
        )
        if indices:
            self.canvas.select_index(indices, False)

    def check_force_balance(self):
        """Scan for structures whose net force exceeds a configurable threshold.

        The user is prompted for the |ΣF| threshold; the value is persisted
        under the ``widget.force_balance_threshold`` config key. Structures
        with net force above this threshold are selected on the scatter plot.
        """
        data = self.canvas.nep_result_data
        if data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        default_threshold = Config.getfloat("widget", "force_balance_threshold", 1e-3)
        box = GetFloatMessageBox(self._parent, "Threshold for |ΣF| (eV/Å):")
        box.doubleSpinBox.setValue(default_threshold)
        if not box.exec():
            return
        threshold = float(box.doubleSpinBox.value())
        if threshold <= 0.0:
            MessageManager.send_warning_message("Threshold must be positive.")
            return
        Config.set("widget", "force_balance_threshold", threshold)

        total_structures = int(getattr(data.structure, "num", 0) or data.structure.now_data.shape[0])
        if total_structures == 0:
            MessageManager.send_info_message(self.tr("No active structures to scan."))
            return

        progress_diag = QProgressDialog("", self.tr("Cancel"), 0, total_structures, self._parent)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle(self.tr("Checking net forces"))
        thread = BackgroundTask(self._parent, show_tip=False)
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        thread.succeeded.connect(
            lambda _result: self._apply_force_balance_selection(data, threshold)
        )
        progress_diag.canceled.connect(thread.stop_work)
        thread.start_work(data.iter_unbalanced_force_indices, threshold=threshold)
        progress_diag.exec()

    def _apply_force_balance_selection(self, data, threshold: float):
        """Select structures flagged by the net-force scan and report counts."""
        try:
            indices = data.consume_unbalanced_force_indices()
        except Exception:  # noqa: BLE001
            logger.debug(traceback.format_exc())
            MessageManager.send_warning_message("Failed to consume force-balance results.")
            return
        if indices:
            self.canvas.select_index(indices, False)
            MessageManager.send_info_message(f"{len(indices)} structures with |ΣF| > {threshold:g}")
        else:
            MessageManager.send_info_message("All scanned structures satisfy the net-force threshold.")

    @staticmethod
    def _plot_point_count(plot_data) -> int:
        """Return the number of scatter points a dataset will draw."""
        try:
            return int(np.asarray(plot_data.x).size)
        except Exception:
            pass
        rows = getattr(plot_data, "now_data", None)
        if rows is None:
            return int(getattr(plot_data, "num", 0) or 0)
        try:
            cols = int(getattr(plot_data, "cols", 0) or 0)
            row_count = int(np.asarray(rows).shape[0])
            return row_count * cols if cols > 0 else row_count
        except Exception:
            return int(getattr(plot_data, "num", 0) or 0)

    @classmethod
    def _total_plot_point_count(cls, dataset) -> int:
        """Return the total number of points rendered across all result plots."""
        return sum(cls._plot_point_count(item) for item in getattr(dataset, "datasets", []) or [])

    @staticmethod
    def _auto_vispy_threshold() -> int:
        threshold = Config.getint("widget", AUTO_VISPY_THRESHOLD_OPTION, AUTO_VISPY_POINT_THRESHOLD)
        if threshold is None or threshold <= 0:
            return AUTO_VISPY_POINT_THRESHOLD
        return threshold

    def _confirm_large_dataset_vispy_switch(self, point_count: int) -> bool:
        """Offer a one-time VisPy switch when PyQtGraph was explicitly selected."""
        if Config.getboolean("widget", LARGE_DATASET_CANVAS_PROMPTED_OPTION, False):
            return False
        Config.set("widget", LARGE_DATASET_CANVAS_PROMPTED_OPTION, True)
        parent = self._parent if isinstance(self._parent, QWidget) else self
        box = MessageBox(
            self.tr("Large plot detected"),
            self.tr(
                "This result contains {point_count:,} plotted points. "
                "PyQtGraph may become less responsive at this size. Switch this result view to VisPy?"
            ).format(point_count=point_count),
            parent,
        )
        box.yesButton.setText(self.tr("Switch to VisPy"))
        box.cancelButton.setText(self.tr("Keep PyQtGraph"))
        box.setClosableOnMaskClicked(True)
        return bool(box.exec())

    def _show_auto_vispy_notice_once(self, point_count: int) -> None:
        if Config.getboolean("widget", AUTO_VISPY_NOTICE_SHOWN_OPTION, False):
            return
        Config.set("widget", AUTO_VISPY_NOTICE_SHOWN_OPTION, True)
        MessageManager.send_info_message(
            self.tr(
                "Large result ({point_count:,} plotted points): switched to VisPy to keep the plot responsive."
            ).format(point_count=point_count)
        )

    def _desired_canvas_type_for_dataset(self, dataset) -> str:
        """Resolve the active canvas backend for this dataset and user setting."""
        configured = self._canvas_mode_value(Config.get("widget", "canvas_type", CanvasMode.AUTO))
        if (
            configured == CanvasMode.PYQTGRAPH.value
            and getattr(self, "_large_dataset_vispy_override", None) is dataset
            and not getattr(self, "_vispy_unavailable", False)
        ):
            return CanvasMode.VISPY.value
        if configured == CanvasMode.VISPY.value and getattr(self, "_vispy_unavailable", False):
            return CanvasMode.PYQTGRAPH.value
        if configured != CanvasMode.AUTO.value:
            return configured

        if getattr(self, "_vispy_unavailable", False):
            return CanvasMode.PYQTGRAPH.value

        if self._total_plot_point_count(dataset) >= self._auto_vispy_threshold():
            return CanvasMode.VISPY.value
        return CanvasMode.PYQTGRAPH.value

    def apply_canvas_mode(self, canvas_type: object) -> bool:
        """Apply a settings change immediately to the currently displayed result."""
        self._large_dataset_vispy_override = None
        configured = self._canvas_mode_value(canvas_type)
        dataset = getattr(getattr(self, "canvas", None), "nep_result_data", None)
        if dataset is None:
            desired = CanvasMode.PYQTGRAPH.value if configured == CanvasMode.AUTO.value else configured
            if getattr(self, "_canvas_type", None) == desired:
                return False
            return self.swith_canvas(desired)

        desired = self._desired_canvas_type_for_dataset(dataset)
        if getattr(self, "_canvas_type", None) == desired:
            return False
        switched = self.swith_canvas(desired, dataset=dataset, preserve_state=True)
        if switched and configured == CanvasMode.AUTO.value and self._canvas_type == CanvasMode.VISPY.value:
            self._show_auto_vispy_notice_once(self._total_plot_point_count(dataset))
        return switched

    def set_dataset(self, dataset):
        """Attach a NEP result dataset to the canvas and refresh the plots.

        Parameters
        ----------
        dataset : Any
            Loaded NEP result container exposing descriptors and structures.
        """
        configured = self._canvas_mode_value(Config.get("widget", "canvas_type", CanvasMode.AUTO))
        point_count = self._total_plot_point_count(dataset)
        desired_canvas_type = self._desired_canvas_type_for_dataset(dataset)
        if (
            configured == CanvasMode.PYQTGRAPH.value
            and desired_canvas_type == CanvasMode.PYQTGRAPH.value
            and not getattr(self, "_vispy_unavailable", False)
            and point_count >= self._auto_vispy_threshold()
            and self._confirm_large_dataset_vispy_switch(point_count)
        ):
            self._large_dataset_vispy_override = dataset
            desired_canvas_type = CanvasMode.VISPY.value

        if getattr(self, "_canvas_type", None) != desired_canvas_type:
            switched = self.swith_canvas(desired_canvas_type, dataset=dataset, preserve_state=True)
            if switched:
                if configured == CanvasMode.AUTO.value and self._canvas_type == CanvasMode.VISPY.value:
                    self._show_auto_vispy_notice_once(point_count)
                return

        if self.last_figure_num != len(dataset.datasets):
            self.canvas.init_axes(len(dataset.datasets))
            self.last_figure_num = len(dataset.datasets)

        self.canvas.set_nep_result_data(dataset)
        self.canvas.plot_nep_result()
