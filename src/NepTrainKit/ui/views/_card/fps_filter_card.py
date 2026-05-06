"""Filter card that keeps representative points via farthest point sampling."""

from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, LineEdit

from NepTrainKit import module_path
from NepTrainKit.config import Config
from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.filter import FPSFilterOperation, FPSFilterParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import FilterDataCard


@CardManager.register_card

class FPSFilterDataCard(FilterDataCard):
    """Filter dataset entries via farthest point sampling computed from NEP descriptors.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget managing the card lifecycle.
    """
    separator=True
    group = "Filter"
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

    def create_operation(self):
        """Return the UI-independent FPS operation."""
        return FPSFilterOperation()

    def get_params(self) -> FPSFilterParams:
        """Read FPS parameters from UI controls."""
        return FPSFilterParams(
            nep_path=self.nep_path_lineedit.text(),
            n_samples=int(self.num_condition_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_condition_frame.get_input_value()[0]),
            backend=Config.get("nep", "backend", "auto"),
            batch_size=Config.getint("nep", "gpu_batch_size", 1000),
        )

    def set_params(self, params: FPSFilterParams) -> None:
        """Apply FPS parameters to UI controls."""
        self.nep_path_lineedit.setText(params.nep_path)
        self.num_condition_frame.set_input_value([int(params.n_samples)])
        self.min_distance_condition_frame.set_input_value([float(params.min_distance)])

    def process_structure(self, *args, **kwargs):
        """Run dataset-level FPS filtering for legacy direct callers."""
        self.result_dataset = self.create_operation().run_dataset(self.dataset, self.get_params())
        return self.result_dataset

    def stop(self):
        """Stop background processing and release any worker threads.
        """
        super().stop()
        if hasattr(self, "nep_thread"):
            self.nep_thread.stop()
            del self.nep_thread

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
        params = self.get_params()
        data_dict["params"] = params_to_dict(params)
        data_dict['nep_path'] = params.nep_path
        data_dict['num_condition'] = [params.n_samples]
        data_dict['min_distance_condition'] = [params.min_distance]
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = FPSFilterParams(
                nep_path=raw_params.get("nep_path", self.nep89_path),
                n_samples=raw_params.get("n_samples", 100),
                min_distance=raw_params.get("min_distance", 0.01),
                backend=raw_params.get("backend", Config.get("nep", "backend", "auto")),
                batch_size=raw_params.get("batch_size", Config.getint("nep", "gpu_batch_size", 1000)),
            )
        else:
            params = FPSFilterParams(
                nep_path=data_dict.get("nep_path", self.nep89_path),
                n_samples=data_dict.get("num_condition", [100])[0],
                min_distance=data_dict.get("min_distance_condition", [0.01])[0],
                backend=Config.get("nep", "backend", "auto"),
                batch_size=Config.getint("nep", "gpu_batch_size", 1000),
            )
        self.set_params(params)

