"""Visualization widgets and analysis helpers for NEP evaluation results."""

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QWidget, QProgressDialog


from NepTrainKit.ui.threads import LoadingThread
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.core import MessageManager
from NepTrainKit.config import Config

from NepTrainKit.ui.widgets import (
    GetIntMessageBox,
    SparseMessageBox,
    IndexSelectMessageBox,
    RangeSelectMessageBox,
    EditInfoMessageBox,
    ShiftEnergyMessageBox,
    DFTD3MessageBox,
)
from NepTrainKit.core.types import SearchType, CanvasMode
from NepTrainKit.ui.views import NepDisplayGraphicsToolBar
from NepTrainKit.core.energy_shift import suggest_group_patterns


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
        self.tool_bar.dftd3Signal.connect(self.calc_dft_d3)
        self.tool_bar.editInfoSignal.connect(self.edit_structure_info)
        self.canvas.tool_bar = self.tool_bar


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

        training_path_default = Config.get("widget", "sparse_training_path", "")

        box.intSpinBox.setValue(n_samples_default)
        box.doubleSpinBox.setValue(distance_default)

        box.descriptorCombo.setCurrentIndex(1 if descriptor_source_default == "raw" else 0)

        box.trainingPathEdit.setText(training_path_default)

        if not box.exec():
            return

        n_samples = box.intSpinBox.value()
        distance = box.doubleSpinBox.value()
        use_selection_region = bool(getattr(box, "regionCheck", None) and box.regionCheck.isChecked())

        descriptor_source = "raw" if box.descriptorCombo.currentIndex() == 1 else "reduced"

        training_path = box.trainingPathEdit.text().strip()

        Config.set("widget", "sparse_num_value", n_samples)
        Config.set("widget", "sparse_distance_value", distance)

        Config.set("widget", "sparse_descriptor_source", descriptor_source)

        Config.set("widget", "sparse_training_path", training_path)

        structures, reverse = data.sparse_point_selection(
            n_samples=n_samples,
            distance=distance,
            descriptor_source=descriptor_source,
            restrict_to_selection=use_selection_region,
            training_path=training_path or None,
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

        data.update_structure_metadata(box.remove_tag, box.new_tag_info)

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
        box = ShiftEnergyMessageBox(
            self._parent,
            "Specify regex groups for Config_type (comma separated)"
        )
        box.groupEdit.setText(";".join(suggested))
        box.genSpinBox.setValue(max_generations)
        box.sizeSpinBox.setValue(population_size)
        box.tolSpinBox.setValue(convergence_tol)

        if not box.exec():
            return

        pattern_text = box.groupEdit.text().strip()
        group_patterns = [p.strip() for p in pattern_text.split(';') if p.strip()]
        alignment_mode = box.modeCombo.currentText()
        max_generations = box.genSpinBox.value()
        population_size = box.sizeSpinBox.value()
        convergence_tol = box.tolSpinBox.value()

        Config.set("widget", "max_generation_value", max_generations)
        Config.set("widget", "population_size", population_size)
        Config.set("widget", "convergence_tol", convergence_tol)

        config_set = set(data.structure.get_all_config(SearchType.TAG))
        progress_diag = QProgressDialog("", "Cancel", 0, len(config_set), self._parent)
        thread = LoadingThread(self._parent, show_tip=False)
        progress_diag.setFixedSize(300, 100)
        progress_diag.setWindowTitle("Shift energies")
        thread.progressSignal.connect(progress_diag.setValue)
        thread.finished.connect(progress_diag.accept)
        progress_diag.canceled.connect(thread.stop_work)
        thread.start_work(
            data.iter_shift_energy_baseline,
            group_patterns,
            alignment_mode,
            max_generations,
            population_size,
            convergence_tol,
            reference_indices=ref_index,
        )
        progress_diag.exec()
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

















