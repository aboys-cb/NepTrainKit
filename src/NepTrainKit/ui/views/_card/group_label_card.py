"""Card for labeling atoms into groups (e.g., A/B sublattices) for downstream rules."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import GroupLabelOperation, GroupLabelParams
from NepTrainKit.ui.widgets import MakeDataCard
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class GroupLabelCard(MakeDataCard):
    """Attach ``atoms.arrays['group']`` labels using common, lattice-agnostic rules.

    The default strategy labels atoms by a commensurate k-vector layering in
    fractional coordinates, producing two groups (even/odd) suitable for
    two-sublattice AFM patterns.
    """

    group = "Alloy"
    card_name = "Group Label"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Group Label (A/B Sublattice)"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("group_label_card_widget")

        self.mode_label = BodyLabel(self.tr("Mode"), self.setting_widget)
        self.mode_label.setToolTip(self.tr("How to assign group labels"))
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            ["k-vector layers (recommended)", "fractional parity (2x rounding)"],
        )

        self.kvec_label = BodyLabel(self.tr("k-vector"), self.setting_widget)
        self.kvec_label.setToolTip(self.tr("Layering vector in fractional coordinates: 100, 010, 001, 110, 111"))
        self.kvec_label.installEventFilter(ToolTipFilter(self.kvec_label, 300, ToolTipPosition.TOP))
        self.kvec_combo = ComboBox(self.setting_widget)
        self.kvec_combo.addItems(["100", "010", "001", "110", "111"])
        self.kvec_combo.setCurrentText("111")

        self.group_a_label = BodyLabel(self.tr("Group A"), self.setting_widget)
        self.group_a_label.setToolTip(self.tr("Label assigned to even layer/parity"))
        self.group_a_label.installEventFilter(ToolTipFilter(self.group_a_label, 300, ToolTipPosition.TOP))
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")

        self.group_b_label = BodyLabel(self.tr("Group B"), self.setting_widget)
        self.group_b_label.setToolTip(self.tr("Label assigned to odd layer/parity"))
        self.group_b_label.installEventFilter(ToolTipFilter(self.group_b_label, 300, ToolTipPosition.TOP))
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")

        self.overwrite_checkbox = CheckBox(self.tr("Overwrite existing group"), self.setting_widget)
        self.overwrite_checkbox.setChecked(True)

        self.settingLayout.addWidget(self.mode_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.kvec_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.kvec_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.overwrite_checkbox, 4, 0, 1, 2)

    def create_operation(self):
        return GroupLabelOperation()

    def get_params(self) -> GroupLabelParams:
        return GroupLabelParams(
            mode=combo_value(self.mode_combo),
            kvec=self.kvec_combo.currentText(),
            group_a=self.group_a_edit.text(),
            group_b=self.group_b_edit.text(),
            overwrite=self.overwrite_checkbox.isChecked(),
        )

    def set_params(self, params: GroupLabelParams) -> None:
        set_combo_value(self.mode_combo, params.mode)
        self.kvec_combo.setCurrentText(params.kvec)
        self.group_a_edit.setText(params.group_a)
        self.group_b_edit.setText(params.group_b)
        self.overwrite_checkbox.setChecked(bool(params.overwrite))

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = GroupLabelParams(**raw_params)
        else:
            params = GroupLabelParams(
                mode=data_dict.get("mode", "k-vector layers (recommended)"),
                kvec=data_dict.get("kvec", "111"),
                group_a=data_dict.get("group_a", "A"),
                group_b=data_dict.get("group_b", "B"),
                overwrite=data_dict.get("overwrite", True),
            )
        self.set_params(params)
