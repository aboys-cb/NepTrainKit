"""Card for enumerating crystal slabs across Miller indices."""

from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import RandomSlabOperation, RandomSlabParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard

@CardManager.register_card
class RandomSlabCard(MakeDataCard):
    """Construct surface slabs across multiple Miller indices and thicknesses.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Surface"

    card_name = "Random Slab"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Random Slab Generation")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("random_slab_card_widget")

        # Miller index ranges for h, k, l
        self.h_label = BodyLabel("h", self.setting_widget)
        self.h_label.setToolTip("h index range")
        self.h_label.installEventFilter(ToolTipFilter(self.h_label, 0, ToolTipPosition.TOP))
        self.h_frame = SpinBoxUnitInputFrame(self)
        self.h_frame.set_input(["-", "step", ""], 3, "int")
        self.h_frame.setRange(-10, 10)
        self.h_frame.set_input_value([0, 1, 1])

        self.k_label = BodyLabel("k", self.setting_widget)
        self.k_label.setToolTip("k index range")
        self.k_label.installEventFilter(ToolTipFilter(self.k_label, 0, ToolTipPosition.TOP))
        self.k_frame = SpinBoxUnitInputFrame(self)
        self.k_frame.set_input(["-", "step", ""], 3, "int")
        self.k_frame.setRange(-10, 10)
        self.k_frame.set_input_value([0, 1, 1])

        self.l_label = BodyLabel("l", self.setting_widget)
        self.l_label.setToolTip("l index range")
        self.l_label.installEventFilter(ToolTipFilter(self.l_label, 0, ToolTipPosition.TOP))
        self.l_frame = SpinBoxUnitInputFrame(self)
        self.l_frame.set_input(["-", "step", ""], 3, "int")
        self.l_frame.setRange(-10, 10)
        self.l_frame.set_input_value([1, 3, 1])

        # Layer number range
        self.layer_label = BodyLabel("Layers", self.setting_widget)
        self.layer_label.setToolTip("Layer range")
        self.layer_label.installEventFilter(ToolTipFilter(self.layer_label, 0, ToolTipPosition.TOP))
        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input(["-", "step", ""], 3, "int")
        self.layer_frame.setRange(1, 50)
        self.layer_frame.set_input_value([3, 6, 1])

        # Vacuum thickness range
        self.vacuum_label = BodyLabel("Vacuum", self.setting_widget)
        self.vacuum_label.setToolTip("Vacuum thickness range in Å")
        self.vacuum_label.installEventFilter(ToolTipFilter(self.vacuum_label, 0, ToolTipPosition.TOP))
        self.vacuum_frame = SpinBoxUnitInputFrame(self)
        self.vacuum_frame.set_input(["-", "step", "Å"], 3, "int")
        self.vacuum_frame.setRange(0, 100)
        self.vacuum_frame.set_input_value([10, 10, 1])

        self.settingLayout.addWidget(self.h_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.h_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.k_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.k_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.l_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.l_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.vacuum_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.vacuum_frame, 4, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent slab operation."""
        return RandomSlabOperation()

    def get_params(self) -> RandomSlabParams:
        """Read slab generation parameters from UI controls."""
        return RandomSlabParams(
            h_range=tuple(int(v) for v in self.h_frame.get_input_value()),
            k_range=tuple(int(v) for v in self.k_frame.get_input_value()),
            l_range=tuple(int(v) for v in self.l_frame.get_input_value()),
            layer_range=tuple(int(v) for v in self.layer_frame.get_input_value()),
            vacuum_range=tuple(int(v) for v in self.vacuum_frame.get_input_value()),
        )

    def set_params(self, params: RandomSlabParams) -> None:
        """Apply slab generation parameters to UI controls."""
        self.h_frame.set_input_value([int(v) for v in params.h_range])
        self.k_frame.set_input_value([int(v) for v in params.k_range])
        self.l_frame.set_input_value([int(v) for v in params.l_range])
        self.layer_frame.set_input_value([int(v) for v in params.layer_range])
        self.vacuum_frame.set_input_value([int(v) for v in params.vacuum_range])

    def process_structure(self, structure):
        """Build surface slabs from UI-independent parameters.
        
        Parameters
        ----------
        structure : ase.Atoms
            Bulk structure used as the source for slab generation.
        
        Returns
        -------
        list[ase.Atoms]
            Slab structures created from the specified index and thickness combinations.
        """
        return self.create_operation().run_structure(structure, self.get_params())

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
        data_dict["h_range"] = list(params.h_range)
        data_dict["k_range"] = list(params.k_range)
        data_dict["l_range"] = list(params.l_range)
        data_dict["layer_range"] = list(params.layer_range)
        data_dict["vacuum_range"] = list(params.vacuum_range)
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
            params = RandomSlabParams(
                h_range=raw_params.get("h_range", [0, 1, 1]),
                k_range=raw_params.get("k_range", [0, 1, 1]),
                l_range=raw_params.get("l_range", [1, 3, 1]),
                layer_range=raw_params.get("layer_range", [3, 6, 1]),
                vacuum_range=raw_params.get("vacuum_range", [10, 10, 1]),
            )
        else:
            params = RandomSlabParams(
                h_range=data_dict.get("h_range", [0, 1, 1]),
                k_range=data_dict.get("k_range", [0, 1, 1]),
                l_range=data_dict.get("l_range", [1, 3, 1]),
                layer_range=data_dict.get("layer_range", [3, 6, 1]),
                vacuum_range=data_dict.get("vacuum_range", [10, 10, 1]),
            )
        self.set_params(params)




