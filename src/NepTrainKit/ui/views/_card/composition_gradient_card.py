"""Card for applying composition gradients along a structure axis."""

from __future__ import annotations

from PySide6.QtCore import Qt
from qfluentwidgets import BodyLabel, CaptionLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import CompositionGradientOperation, CompositionGradientParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import CompositionPathTableInput, MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CompositionGradientCard(MakeDataCard):
    """Assign atom types from a layerwise composition gradient."""

    group = "Alloy"
    card_name = "Composition Gradient"
    description = "Build a one-dimensional composition transition along lattice a, b, or c without moving atoms."
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Composition Gradient"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_gradient_card_widget")

        self.elements_label = BodyLabel(self.tr("Elements"), self.setting_widget)
        self.elements_label.setToolTip(self.tr("Elements participating in the gradient"))
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setText("Ni,Co")

        self.start_label = BodyLabel(self.tr("Start composition"), self.setting_widget)
        self.start_label.setToolTip(self.tr("Composition at the low-coordinate end, e.g. Ni:1,Co:0"))
        self.start_label.installEventFilter(ToolTipFilter(self.start_label, 300, ToolTipPosition.TOP))
        self.start_edit = LineEdit(self.setting_widget)
        self.start_edit.setText("Ni:1,Co:0")

        self.end_label = BodyLabel(self.tr("End composition"), self.setting_widget)
        self.end_label.setToolTip(self.tr("Composition at the high-coordinate end, e.g. Ni:0,Co:1"))
        self.end_label.installEventFilter(ToolTipFilter(self.end_label, 300, ToolTipPosition.TOP))
        self.end_edit = LineEdit(self.setting_widget)
        self.end_edit.setText("Ni:0,Co:1")
        self.start_label.hide()
        self.end_label.hide()
        self.elements_edit.hide()
        self.start_edit.hide()
        self.end_edit.hide()

        self.composition_table = CompositionPathTableInput(self.setting_widget)
        self.composition_table.set_values(
            self.elements_edit.text(), self.start_edit.text(), self.end_edit.text()
        )
        self.elements_edit.textChanged.connect(self._sync_legacy_composition_fields)
        self.start_edit.textChanged.connect(self._sync_legacy_composition_fields)
        self.end_edit.textChanged.connect(self._sync_legacy_composition_fields)

        self.axis_label = BodyLabel(self.tr("Gradient direction"), self.setting_widget)
        self.axis_label.setToolTip(self.tr("Lattice-coordinate direction used to order and layer atoms"))
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.axis_combo,
            [("a", "Lattice a"), ("b", "Lattice b"), ("c", "Lattice c")],
        )
        set_combo_value(self.axis_combo, "a")
        self.direction_hint_label = CaptionLabel("", self.setting_widget)
        self.direction_hint_label.setWordWrap(True)
        self.direction_hint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.bins_label = BodyLabel(self.tr("Composition layers"), self.setting_widget)
        self.bins_label.setToolTip(
            self.tr("Number of equal-atom groups used to approximate the gradient")
        )
        self.bins_label.installEventFilter(ToolTipFilter(self.bins_label, 300, ToolTipPosition.TOP))
        self.bins_frame = SpinBoxUnitInputFrame(self)
        self.bins_frame.set_input("", 1, "int")
        self.bins_frame.setRange(1, 10000)
        self.bins_frame.set_input_value([8])

        self.target_label = BodyLabel(self.tr("Replace existing elements"), self.setting_widget)
        self.target_label.setToolTip(self.tr("Optional existing elements eligible for replacement; empty means all atoms"))
        self.target_label.installEventFilter(ToolTipFilter(self.target_label, 300, ToolTipPosition.TOP))
        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText(self.tr("Ni,Co"))
        self.target_hint_label = CaptionLabel(
            self.tr(
                "Leave empty to replace every atom. List existing elements such as Ni,Co "
                "to preserve all other sublattices."
            ),
            self.setting_widget,
        )
        self.target_hint_label.setWordWrap(True)
        self.target_hint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.samples_label = BodyLabel(self.tr("Random arrangements"), self.setting_widget)
        self.samples_label.setToolTip(self.tr("Number of random assignments emitted for the same layer compositions"))
        self.samples_label.installEventFilter(ToolTipFilter(self.samples_label, 300, ToolTipPosition.TOP))
        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("", 1, "int")
        self.samples_frame.setRange(1, 10000)
        self.samples_frame.set_input_value([1])

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.elements_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.composition_table, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.direction_hint_label, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.bins_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.bins_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.target_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.target_edit, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.target_hint_label, 5, 0, 1, 3)
        self.settingLayout.addWidget(self.samples_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.samples_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 7, 1, 1, 2)

        self.axis_combo.currentIndexChanged.connect(self._update_direction_hint)
        self._update_direction_hint()

    def _sync_legacy_composition_fields(self, *_args) -> None:
        self.composition_table.set_values(
            self.elements_edit.text(), self.start_edit.text(), self.end_edit.text()
        )

    def _update_direction_hint(self) -> None:
        direction = combo_value(self.axis_combo, "a")
        self.direction_hint_label.setText(
            self.tr(
                "Start applies to the low fractional coordinate along lattice {direction}; "
                "end applies to the high side. Periodic boundaries join the two ends."
            ).format(direction=direction)
        )

    def create_operation(self):
        return CompositionGradientOperation()

    def get_params(self) -> CompositionGradientParams:
        elements, start_composition, end_composition = self.composition_table.values()
        return CompositionGradientParams(
            elements=elements,
            start_composition=start_composition,
            end_composition=end_composition,
            axis=combo_value(self.axis_combo),
            bins=int(self.bins_frame.get_input_value()[0]),
            target_elements=self.target_edit.text(),
            samples=int(self.samples_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CompositionGradientParams) -> None:
        self.elements_edit.setText(params.elements)
        self.start_edit.setText(params.start_composition)
        self.end_edit.setText(params.end_composition)
        self.composition_table.set_values(
            params.elements, params.start_composition, params.end_composition
        )
        legacy_axis = {"x": "a", "y": "b", "z": "c"}.get(
            str(params.axis).strip().lower(),
            str(params.axis).strip().lower(),
        )
        set_combo_value(self.axis_combo, legacy_axis)
        self.bins_frame.set_input_value([int(params.bins)])
        self.target_edit.setText(params.target_elements)
        self.samples_frame.set_input_value([int(params.samples)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = CompositionGradientParams(**raw_params) if raw_params else CompositionGradientParams()
        self.set_params(params)
