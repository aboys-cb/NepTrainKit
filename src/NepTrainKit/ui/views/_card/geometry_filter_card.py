"""Card for explicit distance, volume, and density geometry filtering."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.filter import GeometryFilterOperation, GeometryFilterParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class GeometryFilterCard(MakeDataCard):
    """Reject structures that violate explicit geometry-quality thresholds."""

    group = "Filter"
    card_name = "Geometry Filter"
    menu_icon = r":/images/src/images/check.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Geometry Filter")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("geometry_filter_card_widget")

        self.min_pair_label = BodyLabel("Min pair distance", self.setting_widget)
        self.min_pair_label.setToolTip("Reject structures whose shortest interatomic distance is below this value; 0 disables")
        self.min_pair_label.installEventFilter(ToolTipFilter(self.min_pair_label, 300, ToolTipPosition.TOP))
        self.min_pair_frame = SpinBoxUnitInputFrame(self)
        self.min_pair_frame.set_input("A", 1, "float")
        self.min_pair_frame.setRange(0.0, 20.0)
        self.min_pair_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_pair_frame.set_input_value([1.0])

        self.min_vpa_label = BodyLabel("Min volume/atom", self.setting_widget)
        self.min_vpa_label.setToolTip("Reject structures below this per-atom volume in A^3; 0 disables")
        self.min_vpa_label.installEventFilter(ToolTipFilter(self.min_vpa_label, 300, ToolTipPosition.TOP))
        self.min_vpa_frame = SpinBoxUnitInputFrame(self)
        self.min_vpa_frame.set_input("A^3", 1, "float")
        self.min_vpa_frame.setRange(0.0, 10000.0)
        self.min_vpa_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_vpa_frame.set_input_value([0.0])

        self.max_vpa_label = BodyLabel("Max volume/atom", self.setting_widget)
        self.max_vpa_label.setToolTip("Reject structures above this per-atom volume in A^3; 0 disables")
        self.max_vpa_label.installEventFilter(ToolTipFilter(self.max_vpa_label, 300, ToolTipPosition.TOP))
        self.max_vpa_frame = SpinBoxUnitInputFrame(self)
        self.max_vpa_frame.set_input("A^3", 1, "float")
        self.max_vpa_frame.setRange(0.0, 10000.0)
        self.max_vpa_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.max_vpa_frame.set_input_value([0.0])

        self.min_density_label = BodyLabel("Min density", self.setting_widget)
        self.min_density_label.setToolTip("Reject structures below this mass density in g/cm^3; 0 disables")
        self.min_density_label.installEventFilter(ToolTipFilter(self.min_density_label, 300, ToolTipPosition.TOP))
        self.min_density_frame = SpinBoxUnitInputFrame(self)
        self.min_density_frame.set_input("g/cm3", 1, "float")
        self.min_density_frame.setRange(0.0, 1000.0)
        self.min_density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_density_frame.set_input_value([0.0])

        self.max_density_label = BodyLabel("Max density", self.setting_widget)
        self.max_density_label.setToolTip("Reject structures above this mass density in g/cm^3; 0 disables")
        self.max_density_label.installEventFilter(ToolTipFilter(self.max_density_label, 300, ToolTipPosition.TOP))
        self.max_density_frame = SpinBoxUnitInputFrame(self)
        self.max_density_frame.set_input("g/cm3", 1, "float")
        self.max_density_frame.setRange(0.0, 1000.0)
        self.max_density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.max_density_frame.set_input_value([0.0])

        self.require_cell_checkbox = CheckBox("Require finite cell", self.setting_widget)
        self.require_cell_checkbox.setChecked(False)
        self.require_cell_checkbox.setToolTip("Reject zero-volume or invalid-cell structures even when volume/density thresholds are disabled")
        self.require_cell_checkbox.installEventFilter(ToolTipFilter(self.require_cell_checkbox, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.min_pair_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.min_pair_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.min_vpa_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.min_vpa_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.max_vpa_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.max_vpa_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.min_density_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.min_density_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.max_density_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.max_density_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.require_cell_checkbox, 5, 0, 1, 3)

    def create_operation(self):
        return GeometryFilterOperation()

    def get_params(self) -> GeometryFilterParams:
        return GeometryFilterParams(
            min_pair_distance=float(self.min_pair_frame.get_input_value()[0]),
            min_volume_per_atom=float(self.min_vpa_frame.get_input_value()[0]),
            max_volume_per_atom=float(self.max_vpa_frame.get_input_value()[0]),
            min_density=float(self.min_density_frame.get_input_value()[0]),
            max_density=float(self.max_density_frame.get_input_value()[0]),
            require_finite_cell=self.require_cell_checkbox.isChecked(),
        )

    def set_params(self, params: GeometryFilterParams) -> None:
        self.min_pair_frame.set_input_value([float(params.min_pair_distance)])
        self.min_vpa_frame.set_input_value([float(params.min_volume_per_atom)])
        self.max_vpa_frame.set_input_value([float(params.max_volume_per_atom)])
        self.min_density_frame.set_input_value([float(params.min_density)])
        self.max_density_frame.set_input_value([float(params.max_density)])
        self.require_cell_checkbox.setChecked(bool(params.require_finite_cell))

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = GeometryFilterParams(**raw_params) if raw_params else GeometryFilterParams()
        self.set_params(params)
