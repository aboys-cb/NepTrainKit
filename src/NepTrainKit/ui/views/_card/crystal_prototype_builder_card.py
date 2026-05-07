"""Card for generating crystal prototype structures (fcc/bcc/hcp)."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, RadioButton

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import CrystalPrototypeBuilderOperation, CrystalPrototypeBuilderParams
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CrystalPrototypeBuilderCard(MakeDataCard):
    """Generate simple bulk crystal prototypes without requiring input structures."""

    group = "Lattice"
    card_name = "Crystal Prototype Builder"
    menu_icon = r":/images/src/images/supercell.svg"

    requires_input_dataset = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Crystal Prototype Builder")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("crystal_prototype_builder_card_widget")

        self.structure_label = BodyLabel("Lattice", self.setting_widget)
        self.structure_combo = ComboBox(self.setting_widget)
        self.structure_combo.addItems(["fcc", "bcc", "hcp"])
        self.structure_label.setToolTip("Crystal structure prototype")
        self.structure_label.installEventFilter(ToolTipFilter(self.structure_label, 300, ToolTipPosition.TOP))

        self.element_label = BodyLabel("Base element", self.setting_widget)
        self.element_edit = LineEdit(self.setting_widget)
        self.element_edit.setPlaceholderText("Cu")
        self.element_edit.setText("Cu")
        self.element_label.setToolTip("Temporary element used to build the lattice sites")
        self.element_label.installEventFilter(ToolTipFilter(self.element_label, 300, ToolTipPosition.TOP))

        self.a_label = BodyLabel("a (Å)", self.setting_widget)
        self.a_frame = SpinBoxUnitInputFrame(self)
        self.a_frame.set_input(["-", "step", "Å"], 3, "float")
        self.a_frame.setDecimals(6)
        self.a_frame.setRange(0.1, 100.0)
        self.a_frame.set_input_value([3.6, 3.6, 0.1])
        self.a_label.setToolTip("Lattice parameter a range [min, max, step]")
        self.a_label.installEventFilter(ToolTipFilter(self.a_label, 300, ToolTipPosition.TOP))

        self.covera_label = BodyLabel("c/a", self.setting_widget)
        self.covera_frame = SpinBoxUnitInputFrame(self)
        self.covera_frame.set_input("", 1, "float")
        self.covera_frame.setDecimals(6)
        self.covera_frame.setRange(1.0, 5.0)
        self.covera_frame.set_input_value([1.633])
        self.covera_label.setToolTip("hcp only: c/a ratio (ideal ~1.633)")
        self.covera_label.installEventFilter(ToolTipFilter(self.covera_label, 300, ToolTipPosition.TOP))

        self.auto_supercell_button = RadioButton("Auto supercell (max atoms)", self.setting_widget)
        self.auto_supercell_button.setChecked(True)
        self.manual_supercell_button = RadioButton("Manual supercell", self.setting_widget)

        self.max_atoms_label = BodyLabel("Max atoms", self.setting_widget)
        self.max_atoms_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_frame.set_input("unit", 1, "int")
        self.max_atoms_frame.setRange(1, 500000)
        self.max_atoms_frame.set_input_value([512])

        self.rep_label = BodyLabel("Rep (na,nb,nc)", self.setting_widget)
        self.rep_frame = SpinBoxUnitInputFrame(self)
        self.rep_frame.set_input("", 3, "int")
        self.rep_frame.setRange(1, 999)
        self.rep_frame.set_input_value([4, 4, 4])

        self.max_output_label = BodyLabel("Max outputs", self.setting_widget)
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("unit", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([200])

        self.settingLayout.addWidget(self.structure_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.structure_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.element_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.element_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.a_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.a_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.covera_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.covera_frame, 3, 1, 1, 2)

        self.settingLayout.addWidget(self.auto_supercell_button, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_label, 4, 1, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_frame, 4, 2, 1, 1)
        self.settingLayout.addWidget(self.manual_supercell_button, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.rep_label, 5, 1, 1, 1)
        self.settingLayout.addWidget(self.rep_frame, 5, 2, 1, 1)
        self.settingLayout.addWidget(self.max_output_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 6, 1, 1, 2)

    def create_operation(self):
        return CrystalPrototypeBuilderOperation()

    def get_params(self) -> CrystalPrototypeBuilderParams:
        return CrystalPrototypeBuilderParams(
            lattice=self.structure_combo.currentText(),
            element=self.element_edit.text(),
            a_range=tuple(float(value) for value in self.a_frame.get_input_value()),
            covera=float(self.covera_frame.get_input_value()[0]),
            auto_supercell=self.auto_supercell_button.isChecked(),
            max_atoms=int(self.max_atoms_frame.get_input_value()[0]),
            rep=tuple(int(value) for value in self.rep_frame.get_input_value()),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: CrystalPrototypeBuilderParams) -> None:
        self.structure_combo.setCurrentText(params.lattice)
        self.element_edit.setText(params.element)
        self.a_frame.set_input_value([float(value) for value in params.a_range])
        self.covera_frame.set_input_value([float(params.covera)])
        self.auto_supercell_button.setChecked(bool(params.auto_supercell))
        self.manual_supercell_button.setChecked(not bool(params.auto_supercell))
        self.max_atoms_frame.set_input_value([int(params.max_atoms)])
        self.rep_frame.set_input_value([int(value) for value in params.rep])
        self.max_output_frame.set_input_value([int(params.max_outputs)])

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            raw_params["a_range"] = tuple(raw_params.get("a_range", [3.6, 3.6, 0.1]))
            raw_params["rep"] = tuple(raw_params.get("rep", [4, 4, 4]))
            params = CrystalPrototypeBuilderParams(**raw_params)
        else:
            params = CrystalPrototypeBuilderParams(
                lattice=data_dict.get("lattice", "fcc"),
                element=data_dict.get("element", "Cu"),
                a_range=tuple(data_dict.get("a_range", [3.6, 3.6, 0.1])),
                covera=data_dict.get("covera", [1.633])[0],
                auto_supercell=data_dict.get("auto_supercell", True),
                max_atoms=data_dict.get("max_atoms", [512])[0],
                rep=tuple(data_dict.get("rep", [4, 4, 4])),
                max_outputs=data_dict.get("max_outputs", [200])[0],
            )
        self.set_params(params)
