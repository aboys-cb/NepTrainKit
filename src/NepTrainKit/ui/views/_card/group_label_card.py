"""Card for labeling atoms into two coordinate-based groups for downstream rules."""

from __future__ import annotations

from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import GroupLabelOperation, GroupLabelParams
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
)


@CardManager.register_card
class GroupLabelCard(MakeDataCard):
    """Attach ``atoms.arrays['group']`` labels using fractional-coordinate rules.

    This metadata is independent from ``atoms.arrays['sublattice']``.
    """

    group = "Alloy"
    card_name = "Group Label"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Group Label"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("group_label_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.mode_combo = SegmentedControl(parent=self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("k_vector", "Alternating layers"),
                ("fractional_parity", "Half-grid parity"),
            ],
        )
        self.mode_combo.setMinimumWidth(0)
        self.mode_combo.setFixedHeight(28)

        self.kvec_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.kvec_combo,
            [
                ("100", "100 (along lattice a)"),
                ("010", "010 (along lattice b)"),
                ("001", "001 (along lattice c)"),
                ("110", "110 (along lattice a+b)"),
                ("111", "111 (along lattice a+b+c)"),
            ],
        )
        set_combo_value(self.kvec_combo, "111")
        self.kvec_combo.setMinimumWidth(0)
        self.kvec_combo.setFixedHeight(28)

        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")
        self.group_a_edit.setMinimumWidth(0)
        self.group_a_edit.setFixedHeight(28)
        self.group_a_edit.setAccessibleName(self.tr("Group A label (even phase)"))

        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")
        self.group_b_edit.setMinimumWidth(0)
        self.group_b_edit.setFixedHeight(28)
        self.group_b_edit.setAccessibleName(self.tr("Group B label (odd phase)"))

        self.overwrite_checkbox = CheckBox(
            self.tr("Overwrite existing group labels"),
            self.setting_widget,
        )
        self.overwrite_checkbox.setChecked(False)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("groupLabelPreview")

        rule_section = InspectorSection(self.tr("Grouping"), self.setting_widget)
        mode_field = CompactField(
            self.tr("Grouping rule"),
            self.mode_combo,
            rule_section,
            self.tr("Assign labels from fractional coordinates in the current cell."),
        )
        self.kvec_field = CompactField(
            self.tr("Layer vector"),
            self.kvec_combo,
            rule_section,
            self.tr("Direction of the alternating phase."),
        )
        rule_section.addWidget(mode_field)
        rule_section.addWidget(self.kvec_field)

        labels_section = InspectorSection(self.tr("Output labels"), self.setting_widget)
        labels_grid = ResponsiveFormGrid(labels_section)
        self.group_a_field = CompactField(self.tr("Even phase"), self.group_a_edit, labels_section)
        self.group_b_field = CompactField(self.tr("Odd phase"), self.group_b_edit, labels_section)
        labels_grid.add_field(self.group_a_field)
        labels_grid.add_field(self.group_b_field)
        labels_section.addWidget(labels_grid)
        labels_section.addWidget(self.overwrite_checkbox)
        labels_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(rule_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(labels_section, 1, 0, 1, 3)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.kvec_combo.currentIndexChanged.connect(self._refresh_preview)
        self.group_a_edit.editingFinished.connect(self._refresh_preview)
        self.group_b_edit.editingFinished.connect(self._refresh_preview)
        self.overwrite_checkbox.stateChanged.connect(self._refresh_preview)
        self._on_mode_changed()
        self._update_tab_order()

    def _on_mode_changed(self) -> None:
        uses_kvec = combo_value(self.mode_combo) == "k_vector"
        self.kvec_field.setEnabled(uses_kvec)
        self.kvec_combo.setEnabled(uses_kvec)
        self._refresh_preview()
        self._update_tab_order()

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_scaled_positions"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._refresh_preview()

    @staticmethod
    def _format_counts(values) -> str:
        counts = Counter(str(value) for value in values)
        return " · ".join(
            f"{label}={count}"
            for label, count in sorted(counts.items())
        )

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream structure to preview group counts.")
            )
            return

        existing_group = "group" in self._input_structure.arrays
        try:
            output = self.create_operation().run_structure(
                self._input_structure,
                self.get_params(),
            )[0]
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=str(exc))
            )
            return

        if existing_group and not self.overwrite_checkbox.isChecked():
            self.preview_label.setText(
                self.tr(
                    "First input already has group labels. Overwrite is off, so output "
                    "will be unchanged · Existing counts: {counts}"
                ).format(counts=self._format_counts(output.arrays["group"]))
            )
            return

        a_label = self.group_a_edit.text().strip()
        b_label = self.group_b_edit.text().strip()
        counts = Counter(str(value) for value in output.arrays["group"])
        message = self.tr("First input preview: {a}={a_count} · {b}={b_count}").format(
            a=a_label,
            a_count=counts.get(a_label, 0),
            b=b_label,
            b_count=counts.get(b_label, 0),
        )
        if counts.get(a_label, 0) == 0 or counts.get(b_label, 0) == 0:
            message = (
                "⚠ "
                + message
                + " · "
                + self.tr(
                    "Only one group would be produced; expand the cell or choose another rule."
                )
            )
        if existing_group:
            message += " · " + self.tr("Existing group labels will be overwritten.")
        self.preview_label.setText(message)

    def _update_tab_order(self) -> None:
        widgets = [
            self.mode_combo,
            self.kvec_combo,
            self.group_a_edit,
            self.group_b_edit,
            self.overwrite_checkbox,
        ]
        self.tab_order_widgets = [
            widget for widget in widgets if widget.isEnabled() and not widget.isHidden()
        ]
        for previous, current in zip(self.tab_order_widgets, self.tab_order_widgets[1:]):
            QWidget.setTabOrder(previous, current)

    def create_operation(self):
        return GroupLabelOperation()

    def get_params(self) -> GroupLabelParams:
        return GroupLabelParams(
            mode=combo_value(self.mode_combo),
            kvec=combo_value(self.kvec_combo),
            group_a=self.group_a_edit.text(),
            group_b=self.group_b_edit.text(),
            overwrite=self.overwrite_checkbox.isChecked(),
        )

    def set_params(self, params: GroupLabelParams) -> None:
        set_combo_value(
            self.mode_combo,
            GroupLabelOperation.normalize_mode(params.mode),
        )
        set_combo_value(self.kvec_combo, params.kvec)
        self.group_a_edit.setText(params.group_a)
        self.group_b_edit.setText(params.group_b)
        self.overwrite_checkbox.setChecked(bool(params.overwrite))
        self._on_mode_changed()

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
