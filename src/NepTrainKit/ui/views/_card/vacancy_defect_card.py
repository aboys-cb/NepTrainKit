"""Card for generating vacancy defects using stochastic sampling."""

from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, RadioButton

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import VacancyDefectOperation, VacancyDefectParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard

@CardManager.register_card
class VacancyDefectCard(MakeDataCard):
    """Sample vacancy defects either by concentration or by explicit counts.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Defect"
    card_name= "Vacancy Defect Generation"
    menu_icon=r":/images/src/images/defect.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Vacancy Defect")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("vacancy_defect_card_widget")

        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_type_combo.setCurrentIndex(1)

        self.num_radio_button = RadioButton("Vacancy count",self.setting_widget)
        self.num_radio_button.setChecked(True)
        self.num_radio_button.setToolTip("Use atom count as the vacancy amount control")
        self.num_radio_button.installEventFilter(ToolTipFilter(self.num_radio_button, 300, ToolTipPosition.TOP))

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1)
        self.num_condition_frame.setRange(1,10000)


        self.concentration_radio_button = RadioButton("Vacancy fraction",self.setting_widget)
        self.concentration_radio_button.setToolTip("Use fraction of atoms as the vacancy amount control")
        self.concentration_radio_button.installEventFilter(ToolTipFilter(self.concentration_radio_button, 300, ToolTipPosition.TOP))


        self.concentration_condition_frame = SpinBoxUnitInputFrame(self)
        self.concentration_condition_frame.set_input("",1,"float")
        self.concentration_condition_frame.setRange(0,1)

        self.count_mode_label = BodyLabel("Count mode", self.setting_widget)
        self.count_mode_label.setToolTip("Fixed removes exactly the requested amount. Random samples from 1 to that amount.")
        self.count_mode_label.installEventFilter(ToolTipFilter(self.count_mode_label, 300, ToolTipPosition.TOP))
        self.count_mode_combo = ComboBox(self.setting_widget)
        self.count_mode_combo.addItems(["Fixed count", "Random up to value"])
        self.count_mode_combo.setCurrentText("Fixed count")


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)

        self.max_atoms_label= BodyLabel("Structures",self.setting_widget)
        self.max_atoms_label.setToolTip("Number of vacancy structures to generate")

        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip("Enable reproducible random sampling")
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        #
        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.num_radio_button, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.concentration_radio_button, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.concentration_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.count_mode_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.count_mode_combo, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 5, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent vacancy-defect operation."""
        return VacancyDefectOperation()

    def get_params(self) -> VacancyDefectParams:
        """Read vacancy-defect parameters from UI controls."""
        return VacancyDefectParams(
            engine_type=self.engine_type_combo.currentIndex(),
            num_condition=int(self.num_condition_frame.get_input_value()[0]),
            use_num=self.num_radio_button.isChecked(),
            concentration_condition=float(self.concentration_condition_frame.get_input_value()[0]),
            count_mode="fixed" if self.count_mode_combo.currentText() == "Fixed count" else "random",
            max_structures=int(self.max_atoms_condition_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: VacancyDefectParams) -> None:
        """Apply vacancy-defect parameters to UI controls."""
        self.engine_type_combo.setCurrentIndex(int(params.engine_type))
        self.num_condition_frame.set_input_value([int(params.num_condition)])
        self.concentration_condition_frame.set_input_value([float(params.concentration_condition)])
        self.max_atoms_condition_frame.set_input_value([int(params.max_structures)])
        self.num_radio_button.setChecked(bool(params.use_num))
        self.concentration_radio_button.setChecked(not bool(params.use_num))
        self.count_mode_combo.setCurrentText("Fixed count" if params.count_mode == "fixed" else "Random up to value")
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self,structure):
        """Create vacancy defect structures from UI-independent parameters.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure to modify.
        
        Returns
        -------
        list[ase.Atoms]
            Structures with randomly placed vacancies.
        """
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

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
            params = VacancyDefectParams(
                engine_type=raw_params.get("engine_type", 1),
                num_condition=raw_params.get("num_condition", 1),
                use_num=raw_params.get("use_num", True),
                concentration_condition=raw_params.get("concentration_condition", 0.0),
                count_mode=raw_params.get("count_mode", "random"),
                max_structures=raw_params.get("max_structures", 1),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = VacancyDefectParams(
                engine_type=data_dict.get("engine_type", 1),
                num_condition=data_dict.get("num_condition", [1])[0],
                use_num=data_dict.get("num_radio_button", True),
                concentration_condition=data_dict.get("concentration_condition", [0.0])[0],
                count_mode=data_dict.get("count_mode", "random"),
                max_structures=data_dict.get("max_atoms_condition", [1])[0],
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)

