"""Card for strict GSFE paths."""

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import StrictGSFEPathOperation, StrictGSFEPathParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class StrictGSFEPathCard(MakeDataCard):
    """Generate unrelaxed GSFE structures with explicit plane and slip direction."""

    group = "Defect"
    card_name = "Strict GSFE Path"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Make Strict GSFE Path"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("strict_gsfe_path_card_widget")

        self.hkl_label = BodyLabel(self.tr("h k l"), self.setting_widget)
        self.hkl_frame = SpinBoxUnitInputFrame(self)
        self.hkl_frame.set_input("", 3, "int")
        self.hkl_frame.setRange(-10, 10)
        self.hkl_frame.set_input_value([0, 0, 1])

        self.uvw_label = BodyLabel(self.tr("u v w"), self.setting_widget)
        self.uvw_frame = SpinBoxUnitInputFrame(self)
        self.uvw_frame.set_input("", 3, "int")
        self.uvw_frame.setRange(-10, 10)
        self.uvw_frame.set_input_value([1, 0, 0])
        self.uvw_label.setToolTip(self.tr("Slip direction. It is projected into the selected plane."))
        self.uvw_label.installEventFilter(ToolTipFilter(self.uvw_label, 300, ToolTipPosition.TOP))

        self.disp_label = BodyLabel(self.tr("Displacement"), self.setting_widget)
        self.disp_frame = SpinBoxUnitInputFrame(self)
        self.disp_frame.set_input(["-", "step", ""], 3, "float")
        self.disp_frame.setDecimals(4)
        self.disp_frame.setRange(-100.0, 100.0)
        self.disp_frame.set_input_value([0.0, 1.0, 0.5])

        self.unit_label = BodyLabel(self.tr("Unit"), self.setting_widget)
        self.unit_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.unit_combo,
            [("fraction_of_vector", "fraction of vector"), ("angstrom", "angstrom")],
        )

        self.cut_label = BodyLabel(self.tr("Cut"), self.setting_widget)
        self.cut_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.cut_combo,
            [("middle", "middle"), ("fractional", "fractional"), ("layer_index", "layer index")],
        )

        self.cut_fraction_label = BodyLabel(self.tr("Cut fraction"), self.setting_widget)
        self.cut_fraction_frame = SpinBoxUnitInputFrame(self)
        self.cut_fraction_frame.set_input("", 1, "float")
        self.cut_fraction_frame.setDecimals(4)
        self.cut_fraction_frame.setRange(0.0, 1.0)
        self.cut_fraction_frame.set_input_value([0.5])

        self.layer_label = BodyLabel(self.tr("Layer index"), self.setting_widget)
        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input("", 1, "int")
        self.layer_frame.setRange(0, 999999)
        self.layer_frame.set_input_value([0])

        self.wrap_checkbox = CheckBox(self.tr("Wrap"), self.setting_widget)
        self.wrap_checkbox.setChecked(True)

        self.settingLayout.addWidget(self.hkl_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.hkl_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.uvw_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.uvw_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.disp_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.disp_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.unit_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.unit_combo, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.cut_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.cut_combo, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.cut_fraction_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.cut_fraction_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.wrap_checkbox, 7, 0, 1, 2)

    def create_operation(self):
        return StrictGSFEPathOperation()

    def get_params(self) -> StrictGSFEPathParams:
        return StrictGSFEPathParams(
            plane_hkl=tuple(int(v) for v in self.hkl_frame.get_input_value()),
            slip_uvw=tuple(int(v) for v in self.uvw_frame.get_input_value()),
            displacement_range=tuple(float(v) for v in self.disp_frame.get_input_value()),
            displacement_unit=combo_value(self.unit_combo),
            cut_mode=combo_value(self.cut_combo),
            cut_fraction=float(self.cut_fraction_frame.get_input_value()[0]),
            layer_index=int(self.layer_frame.get_input_value()[0]),
            wrap=self.wrap_checkbox.isChecked(),
        )

    def set_params(self, params: StrictGSFEPathParams) -> None:
        self.hkl_frame.set_input_value([int(v) for v in params.plane_hkl])
        self.uvw_frame.set_input_value([int(v) for v in params.slip_uvw])
        self.disp_frame.set_input_value([float(v) for v in params.displacement_range])
        set_combo_value(self.unit_combo, params.displacement_unit)
        set_combo_value(self.cut_combo, params.cut_mode)
        self.cut_fraction_frame.set_input_value([float(params.cut_fraction)])
        self.layer_frame.set_input_value([int(params.layer_index)])
        self.wrap_checkbox.setChecked(bool(params.wrap))

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.set_params(StrictGSFEPathParams(**data_dict.get("params", {})))
