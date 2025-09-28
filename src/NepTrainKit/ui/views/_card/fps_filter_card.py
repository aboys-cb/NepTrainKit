"""Filter card that keeps representative points via farthest point sampling."""

import os
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox, LineEdit

from NepTrainKit import module_path
from NepTrainKit.ui.threads import FilterProcessingThread
from NepTrainKit.config import Config
from NepTrainKit.core import CardManager,   MessageManager
from NepTrainKit.core.calculator import  NepCalculator
from NepTrainKit.core.io import farthest_point_sampling
from NepTrainKit.core.types import NepBackend
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard, FilterDataCard


@CardManager.register_card

class FPSFilterDataCard(FilterDataCard):
    """Filter dataset entries via farthest point sampling computed from NEP descriptors.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget managing the card lifecycle.
    """
    separator=True
    card_name= "FPS Filter"
    menu_icon=r":/images/src/images/fps.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Filter by FPS")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("fps_filter_card_widget")
        self.nep_path_label = BodyLabel("NEP file path: ", self.setting_widget)

        self.nep_path_lineedit = LineEdit(self.setting_widget)
        self.nep_path_lineedit.setPlaceholderText("nep.txt path")
        self.nep_path_label.setToolTip("Path to NEP model")
        self.nep_path_label.installEventFilter(ToolTipFilter(self.nep_path_label, 300, ToolTipPosition.TOP))

        self.nep89_path = str(module_path/ "Config/nep89.txt" )
        self.nep_path_lineedit.setText(self.nep89_path )


        self.num_label = BodyLabel("Max selected", self.setting_widget)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit", 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([100])
        self.num_label.setToolTip("Number of structures to keep")
        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.min_distance_condition_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_condition_frame.set_input("", 1,"float")
        self.min_distance_condition_frame.setRange(0, 100)
        self.min_distance_condition_frame.object_list[0].setDecimals(4)   # pyright:ignore
        self.min_distance_condition_frame.set_input_value([0.01])

        self.min_distance_label = BodyLabel("Min distance", self.setting_widget)
        self.min_distance_label.setToolTip("Minimum distance between samples")

        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.num_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.min_distance_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.min_distance_condition_frame, 1, 1, 1, 2)


        self.settingLayout.addWidget(self.nep_path_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.nep_path_lineedit, 2, 1, 1, 2)

    def process_structure(self,*args, **kwargs ):
        """Compute NEP descriptors and apply farthest point sampling to the current dataset.
        
        Parameters
        ----------
        *args
            Unused positional arguments.
        **kwargs
            Unused keyword arguments.
        
        Returns
        -------
        None
            The filtered dataset is stored on ``result_dataset``.
        """
        nep_path=self.nep_path_lineedit.text()
        n_samples=self.num_condition_frame.get_input_value()[0]
        distance=self.min_distance_condition_frame.get_input_value()[0]

        nep_calc = NepCalculator(
            model_file=nep_path,
            backend=NepBackend(Config.get("nep", "backend", "auto")),
            batch_size=Config.getint("nep", "gpu_batch_size", 1000)
        )
        desc_array=nep_calc.get_structures_descriptor(self.dataset)



        remaining_indices = farthest_point_sampling(desc_array, n_samples=n_samples, min_dist=distance)

        self.result_dataset = [self.dataset[i] for i in remaining_indices]

    def stop(self):
        """Stop background processing and release any worker threads.
        """
        super().stop()
        if hasattr(self, "nep_thread"):
            self.nep_thread.stop()
            del self.nep_thread

    def run(self):
        """Execute the card logic, launching a worker thread when the card is enabled.
        """
        nep_path=self.nep_path_lineedit.text()

        if not os.path.exists(nep_path):
            MessageManager.send_warning_message(  "NEP file not exists!")
            self.runFinishedSignal.emit(self.index)

            return
        if self.check_state:
            self.worker_thread = FilterProcessingThread(

                self.process_structure
            )
            self.status_label.set_colors(["#59745A"])

            self.worker_thread.progressSignal.connect(self.update_progress)
            self.worker_thread.finishSignal.connect(self.on_processing_finished)
            self.worker_thread.errorSignal.connect(self.on_processing_error)

            self.worker_thread.start()
        else:
            self.result_dataset = self.dataset
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)

    def update_progress(self, progress):
        """Update the visual progress indicators during background execution.
        
        Parameters
        ----------
        progress : float | int
            Latest progress value emitted by the worker thread.
        """
        self.status_label.setText(f"generate descriptors ...")
        self.status_label.set_progress(progress)

    def to_dict(self):
        """Serialize the current configuration to a plain dictionary.
        
        Returns
        -------
        dict
            Dictionary that can be fed into ``from_dict`` to rebuild the state.
        """
        data_dict = super().to_dict()

        data_dict['nep_path']=self.nep_path_lineedit.text()
        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        data_dict['min_distance_condition'] = self.min_distance_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        try:
            super().from_dict(data_dict)

            if os.path.exists(data_dict['nep_path']):
                self.nep_path_lineedit.setText(data_dict['nep_path'])
            else:
                self.nep_path_lineedit.setText(self.nep89_path )
            self.num_condition_frame.set_input_value(data_dict['num_condition'])
            self.min_distance_condition_frame.set_input_value(data_dict['min_distance_condition'])
        except:
            pass

