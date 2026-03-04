"""Visualization widgets and analysis helpers for NEP evaluation results."""
import json
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget, QProgressDialog
from loguru import logger
from qfluentwidgets import MessageBox

from NepTrainKit.ui.threads import LoadingThread
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.core import MessageManager
from NepTrainKit.config import Config

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
    DatasetSummaryMessageBox,
    DistributionInspectorMessageBox,
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

    def __init__(self,parent=None):
        """Create the widget layout and load the canvas defined in user preferences.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget used for signal propagation and dialog ownership.
        """
        super().__init__(parent)
        self._parent=parent
        self.tool_bar: NepDisplayGraphicsToolBar
        self.draw_mode=False
        # self.setRenderHint(QPainter.Antialiasing, False)
        self._layout = QHBoxLayout(self)
        self.setLayout(self._layout)
        canvas_type = Config.get("widget","canvas_type",CanvasMode.PYQTGRAPH)

        self.last_figure_num=None
        self._distribution_inspector = None
        self.swith_canvas(canvas_type)

    def swith_canvas(self,canvas_type:CanvasMode="pyqtgraph"):
        """Instantiate the requested plotting backend and attach it to the layout.
        
        Parameters
        ----------
        canvas_type : CanvasMode, default=CanvasMode.PYQTGRAPH
            Backend identifier used to select between the supported canvases.
        """
        if canvas_type == CanvasMode.PYQTGRAPH:
            from NepTrainKit.ui.canvas.pyqtgraph import PyqtgraphCanvas
            self.canvas = PyqtgraphCanvas(self)
            self._layout.addWidget(self.canvas)

        elif canvas_type == CanvasMode.VISPY:


            from NepTrainKit.ui.canvas.vispy import VispyCanvas
            self.canvas = VispyCanvas(parent=self, bgcolor='white')
            self._layout.addWidget(self.canvas.native)
            # self.window().windowHandle().screenChanged.connect(self.canvas.native.screen_changed)
        else:
            from NepTrainKit.ui.canvas.vispy import VispyCanvas
            self.canvas = VispyCanvas(parent=self, bgcolor='white')
            self._layout.addWidget(self.canvas.native)



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
        self.tool_bar.panSignal.connect(self.canvas.pan)
        self.tool_bar.resetSignal.connect(self.canvas.auto_range)
        self.tool_bar.deleteSignal.connect(self.canvas.delete)
        self.tool_bar.revokeSignal.connect(self.canvas.revoke)
        self.tool_bar.penSignal.connect(self.canvas.pen)
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
        self.tool_bar.summarySignal.connect(self.show_dataset_summary)
        self.tool_bar.forceBalanceSignal.connect(self.check_force_balance)
        self.tool_bar.distributionSignal.connect(self.show_distribution_inspector)
        self.canvas.tool_bar = self.tool_bar

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
        thread = LoadingThread(self._parent, show_tip=False)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Finding non-physical structures")
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        thread.finished.connect(lambda: self._apply_non_physical_selection(data))
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
        """Select the highest-error structures on the active axes based on user input.
        """
        dataset = self.canvas.get_axes_dataset(self.canvas.current_axes)

        if dataset is None:
            return

        box= GetIntMessageBox(self._parent,"Please enter an integer N, it will find the top N structures with the largest errors")
        n = Config.getint("widget","max_error_value",10)
        box.intSpinBox.setValue(n)

        if not box.exec():
            return
        nmax= box.intSpinBox.value()
        Config.set("widget","max_error_value",nmax)
        index= (dataset.get_max_error_index(nmax))

        self.canvas.select_index(index,False)

    def sparse_point(self):
        """Run farthest point sampling with simple and advanced strategies."""
        data = self.canvas.nep_result_data
        if data is None:
            return

        box = SparseMessageBox(self._parent, "Configure farthest point sampling")
        n_samples_default = Config.getint("widget", "sparse_num_value", 10)
        distance_default = Config.getfloat("widget", "sparse_distance_value", 0.01)

        descriptor_source_default = Config.get("widget", "sparse_descriptor_source", "reduced").lower()
        sampling_mode_default = Config.get("widget", "sparse_sampling_mode", "count").lower()
        r2_threshold_default = Config.getfloat("widget", "sparse_r2_threshold", 0.9)

        training_path_default = Config.get("widget", "sparse_training_path", "")

        box.intSpinBox.setValue(n_samples_default)
        box.doubleSpinBox.setValue(distance_default)

        box.descriptorCombo.setCurrentIndex(1 if descriptor_source_default == "raw" else 0)
        box.modeCombo.setCurrentIndex(1 if sampling_mode_default == "r2" else 0)
        box.r2SpinBox.setValue(r2_threshold_default if r2_threshold_default is not None else 0.9)

        box.trainingPathEdit.setText(training_path_default)

        if not box.exec():
            return

        n_samples = box.intSpinBox.value()
        distance = box.doubleSpinBox.value()
        use_selection_region = bool(getattr(box, "regionCheck", None) and box.regionCheck.isChecked())

        descriptor_source = "raw" if box.descriptorCombo.currentIndex() == 1 else "reduced"
        sampling_mode = "r2" if box.modeCombo.currentIndex() == 1 else "count"
        r2_threshold = box.r2SpinBox.value()

        training_path = box.trainingPathEdit.text().strip()

        Config.set("widget", "sparse_num_value", n_samples)
        Config.set("widget", "sparse_distance_value", distance)

        Config.set("widget", "sparse_descriptor_source", descriptor_source)
        Config.set("widget", "sparse_sampling_mode", sampling_mode)
        Config.set("widget", "sparse_r2_threshold", r2_threshold)

        Config.set("widget", "sparse_training_path", training_path)

        structures, reverse = data.sparse_point_selection(
            n_samples=n_samples,
            distance=distance,
            descriptor_source=descriptor_source,
            restrict_to_selection=use_selection_region,
            training_path=training_path or None,
            sampling_mode=sampling_mode,
            r2_threshold=r2_threshold,
        )
        if structures:
            self.canvas.select_index(structures, reverse)

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
            thread = LoadingThread(self, show_tip=True, title="Exporting descriptor data")
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
            "Specify regex groups for Config_type (comma separated)",
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
        progress_diag = QProgressDialog("", "Cancel", 0, len(config_set), self._parent)
        thread = LoadingThread(self._parent, show_tip=False)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Shift energies")
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

        box = DFTD3MessageBox(
            self._parent,
            "DFT D3"
        )
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

        thread = LoadingThread(self._parent, show_tip=True, title="calculating dftd3")
        thread.start_work(
            data.apply_dft_d3_correction,
            mode, functional, d3_cutoff, d3_cutoff_cn
        )
        thread.finished.connect(self.canvas.plot_nep_result)

    def show_dataset_summary(self):
        """Compute and display dataset-wide summary statistics."""
        data = self.canvas.nep_result_data
        if data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        group_by = SearchType.TAG
        parent = getattr(self, "_parent", None)
        search = getattr(parent, "search_lineEdit", None) if parent is not None else None
        search_type = getattr(search, "search_type", None)
        if isinstance(search_type, SearchType):
            group_by = search_type
        structures = getattr(data, "structure", None)
        if structures is None or structures.now_data.size == 0:
            MessageManager.send_info_message("No active structures to summarise.")
            return
        total_structures = int(structures.now_data.shape[0])

        progress_diag = QProgressDialog("", "Cancel", 0, total_structures, self._parent)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Summarising dataset")
        progress_diag.setAutoClose(True)
        progress_diag.setAutoReset(True)
        thread = LoadingThread(self._parent, show_tip=False)
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(lambda: progress_diag.setValue(total_structures))
        thread.finished.connect(progress_diag.accept)
        thread.finished.connect(lambda: self._show_dataset_summary_dialog(data))
        progress_diag.canceled.connect(thread.stop_work)
        thread.start_work(data.iter_dataset_summary, group_by=group_by)
        progress_diag.exec()

    def _show_dataset_summary_dialog(self, data):
        """Instantiate and execute the dataset summary dialog."""
        try:
            summary = data.get_dataset_summary()
        except Exception:  # noqa: BLE001
            MessageManager.send_warning_message("Failed to build dataset summary.")
            logger.debug(traceback.format_exc())
            return
        if not summary:
            MessageManager.send_info_message("Dataset summary is empty.")
            return
        dlg = DatasetSummaryMessageBox(self._parent, summary)
        dlg.exec()

    def _run_distribution_analysis_task(self, data, request) -> dict:
        """Run distribution analysis in a worker thread and return the payload."""
        structures = getattr(data, "structure", None)
        total_structures = int(getattr(structures, "now_data", np.array([])).shape[0]) if structures is not None else 0
        progress_diag = QProgressDialog("", "Cancel", 0, max(total_structures, 1), self._parent)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Building distributions")
        thread = LoadingThread(self._parent, show_tip=False)
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        progress_diag.canceled.connect(thread.stop_work)
        thread.start_work(data.iter_distribution_analysis, request=request)
        progress_diag.exec()
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
        """Open the distribution-inspector dialog for numeric fields."""
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
        """Invert the current structure selection on the canvas.
        """
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
        indices = data.select_structures_by_range( dataset, x_min, x_max, y_min, y_max, logic_and)
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
            MessageManager.send_info_message("No active structures to scan.")
            return

        progress_diag = QProgressDialog("", "Cancel", 0, total_structures, self._parent)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Checking net forces")
        thread = LoadingThread(self._parent, show_tip=False)
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        thread.finished.connect(lambda: self._apply_force_balance_selection(data, threshold))
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
            MessageManager.send_info_message(
                f"{len(indices)} structures with |ΣF| > {threshold:g}"
            )
        else:
            MessageManager.send_info_message("All scanned structures satisfy the net-force threshold.")

    def set_dataset(self,dataset):
        """Attach a NEP result dataset to the canvas and refresh the plots.
        
        Parameters
        ----------
        dataset : Any
            Loaded NEP result container exposing descriptors and structures.
        """
        if self.last_figure_num !=len(dataset.datasets):

            self.canvas.init_axes(len(dataset.datasets))
            self.last_figure_num = len(dataset.datasets)

        self.canvas.set_nep_result_data(dataset)
        self.canvas.plot_nep_result()

















