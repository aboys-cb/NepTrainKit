"""Card for labeling detected atomic layers into alternating groups."""

from __future__ import annotations

from collections import Counter

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import GroupLabelOperation, GroupLabelParams
from NepTrainKit.ui.messages import MessageManager, translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class GroupLabelCard(MakeDataCard):
    """Attach ``atoms.arrays['group']`` labels by detected atomic plane.

    This metadata is independent from ``atoms.arrays['sublattice']``.
    """

    group = "Structure"
    card_name = "Layer Groups"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count = None
        self.setTitle(self.tr("Layer Groups"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("group_label_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.plane_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.plane_combo,
            [
                ("100", "(100) planes"),
                ("010", "(010) planes"),
                ("001", "(001) planes"),
                ("110", "(110) planes"),
                ("111", "(111) planes"),
            ],
        )
        set_combo_value(self.plane_combo, "001")
        self.plane_combo.setMinimumWidth(0)
        self.plane_combo.setFixedHeight(28)

        self.tolerance_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.tolerance_frame.set_input("Å", 1, "float")
        self.tolerance_frame.setRange(0.001, 10.0)
        self.tolerance_frame.setDecimals(3)
        self.tolerance_frame.setSingleStep(0.01)
        self.tolerance_frame.set_input_value([0.05])

        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")
        self.group_a_edit.setMinimumWidth(0)
        self.group_a_edit.setFixedHeight(28)
        self.group_a_edit.setAccessibleName(self.tr("Group A label (even layer)"))

        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")
        self.group_b_edit.setMinimumWidth(0)
        self.group_b_edit.setFixedHeight(28)
        self.group_b_edit.setAccessibleName(self.tr("Group B label (odd layer)"))

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

        self.legacy_notice = CaptionLabel("", self.setting_widget)
        self.legacy_notice.setWordWrap(True)
        self.legacy_notice.setStyleSheet("color:#c56a00; font-weight:600;")
        self.legacy_notice.hide()

        rule_section = InspectorSection(
            self.tr("Layer detection"),
            self.setting_widget,
            self.tr("Atoms in each detected plane receive the same label; adjacent planes alternate A/B."),
        )
        plane_field = CompactField(
            self.tr("Crystal plane (hkl)"),
            self.plane_combo,
            rule_section,
            self.tr("The plane normal is computed from the reciprocal lattice, including non-orthogonal cells."),
        )
        tolerance_field = CompactField(
            self.tr("Layer tolerance"),
            self.tolerance_frame,
            rule_section,
            self.tr("Atoms whose normal projections differ by no more than this distance share one layer."),
        )
        rule_grid = ResponsiveFormGrid(rule_section)
        rule_grid.add_field(plane_field)
        rule_grid.add_field(tolerance_field)
        rule_section.addWidget(rule_grid)

        labels_section = InspectorSection(self.tr("Output labels"), self.setting_widget)
        labels_grid = ResponsiveFormGrid(labels_section)
        self.group_a_field = CompactField(self.tr("Even layers"), self.group_a_edit, labels_section)
        self.group_b_field = CompactField(self.tr("Odd layers"), self.group_b_edit, labels_section)
        labels_grid.add_field(self.group_a_field)
        labels_grid.add_field(self.group_b_field)
        labels_section.addWidget(labels_grid)
        labels_section.addWidget(self.overwrite_checkbox)
        labels_section.addWidget(self.preview_label)
        labels_section.addWidget(self.legacy_notice)

        self.settingLayout.addWidget(rule_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(labels_section, 1, 0, 1, 3)

        self.plane_combo.currentIndexChanged.connect(self._refresh_preview)
        for control in self.tolerance_frame.object_list:
            control.valueChanged.connect(self._refresh_preview)
        self.group_a_edit.editingFinished.connect(self._refresh_preview)
        self.group_b_edit.editingFinished.connect(self._refresh_preview)
        self.overwrite_checkbox.stateChanged.connect(self._refresh_preview)
        self._update_tab_order()
        self._refresh_preview()

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

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self._refresh_preview()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    @staticmethod
    def _format_counts(values) -> str:
        counts = Counter(str(value) for value in values)
        return " · ".join(
            f"{label}={count}"
            for label, count in sorted(counts.items())
        )

    def _format_layer_count(self, count: int) -> str:
        if count == 1:
            return self.tr("1 layer")
        return self.tr("{layers} layers").format(layers=count)

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        self.refresh_compact_presentation()
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream structure to preview group counts.")
            )
            return

        existing_group = "group" in self._input_structure.arrays
        if existing_group and not self.overwrite_checkbox.isChecked():
            self.preview_label.setText(
                self.tr(
                    "First input already has group labels. Overwrite is off, so output "
                    "will be unchanged · Existing counts: {counts}"
                ).format(counts=self._format_counts(self._input_structure.arrays["group"]))
            )
            return

        a_label = self.group_a_edit.text().strip()
        b_label = self.group_b_edit.text().strip()
        try:
            operation = self.create_operation()
            operation._validated_labels(a_label, b_label)
            layer_ids = operation.layer_ids(
                self._input_structure,
                combo_value(self.plane_combo),
                float(self.tolerance_frame.get_input_value()[0]),
            )
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return

        layer_count = int(layer_ids.max()) + 1 if layer_ids.size else 0
        counts = Counter(
            a_label if layer_id % 2 == 0 else b_label
            for layer_id in layer_ids
        )
        sequence = []
        for layer_id in range(min(layer_count, 8)):
            label = a_label if layer_id % 2 == 0 else b_label
            sequence.append(f"{label}({int(np.count_nonzero(layer_ids == layer_id))})")
        sequence_text = " → ".join(sequence)
        if layer_count > 8:
            sequence_text += " → …"
        message = self.tr(
            "First input: {layers} · Layer sequence (atoms): {sequence} · {a}={a_count} · {b}={b_count}"
        ).format(
            layers=self._format_layer_count(layer_count),
            sequence=sequence_text,
            a=a_label,
            a_count=counts.get(a_label, 0),
            b=b_label,
            b_count=counts.get(b_label, 0),
        )
        if layer_count < 2:
            message = (
                "⚠ "
                + message
                + " · "
                + self.tr(
                    "At least two layers are required; expand the cell, choose another plane, or reduce the tolerance."
                )
            )
        elif (
            layer_count % 2
            and operation.has_periodic_layer_axis(
                self._input_structure,
                combo_value(self.plane_combo),
            )
        ):
            message += " · " + self.tr(
                "Odd layer count: periodic A/B order does not close across the boundary."
            )
        if existing_group:
            message += " · " + self.tr("Existing group labels will be overwritten.")
        self.preview_label.setText(message)

    def _update_tab_order(self) -> None:
        widgets = [
            self.plane_combo,
            *self.tolerance_frame.object_list,
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
            miller_index=combo_value(self.plane_combo),
            layer_tolerance=float(self.tolerance_frame.get_input_value()[0]),
            group_a=self.group_a_edit.text(),
            group_b=self.group_b_edit.text(),
            overwrite=self.overwrite_checkbox.isChecked(),
        )

    def set_params(self, params: GroupLabelParams) -> None:
        set_combo_value(self.plane_combo, params.miller_index)
        self.tolerance_frame.set_input_value([float(params.layer_tolerance)])
        self.group_a_edit.setText(params.group_a)
        self.group_b_edit.setText(params.group_b)
        self.overwrite_checkbox.setChecked(bool(params.overwrite))
        self._refresh_preview()
        self._update_tab_order()

    def get_summary_text(self) -> str:
        params = self.get_params()
        try:
            self.create_operation()._validated_labels(params.group_a, params.group_b)
        except ValueError:
            return self.tr("Complete the two group labels")
        if self._input_structure is not None:
            if "group" in self._input_structure.arrays and not params.overwrite:
                return self.tr("Keep existing groups · 1 output/input")
            try:
                layer_ids = self.create_operation().layer_ids(
                    self._input_structure,
                    params.miller_index,
                    params.layer_tolerance,
                )
            except ValueError:
                pass
            else:
                layer_count = int(layer_ids.max()) + 1 if layer_ids.size else 0
                return self.tr("({hkl}) · {layers} · {a}/{b} · 1/input").format(
                    hkl=params.miller_index,
                    layers=self._format_layer_count(layer_count),
                    a=params.group_a.strip(),
                    b=params.group_b.strip(),
                )
        return self.tr("({hkl}) · {tolerance} Å · {a}/{b}").format(
            hkl=params.miller_index,
            tolerance=f"{params.layer_tolerance:.4g}",
            a=params.group_a.strip(),
            b=params.group_b.strip(),
        )

    def get_guidance_text(self) -> str:
        input_count = self._preview_input_count
        if input_count is None or input_count <= 0:
            output_text = self.tr("Outputs/input: 1.")
        else:
            output_text = self.tr(
                "Inputs {inputs} × 1 output/input = outputs {total}."
            ).format(inputs=input_count, total=input_count)

        params = self.get_params()
        try:
            operation = self.create_operation()
            operation._validated_labels(params.group_a, params.group_b)
            if self._input_structure is None:
                return output_text + " " + self.tr(
                    "Load an upstream structure to check the detected layers."
                )
            if "group" in self._input_structure.arrays and not params.overwrite:
                return output_text + " " + self.tr(
                    "Existing group labels are preserved because overwrite is off."
                )
            layer_ids = operation.layer_ids(
                self._input_structure,
                params.miller_index,
                params.layer_tolerance,
            )
        except ValueError as exc:
            return output_text + " " + translate_runtime_message(exc)

        layer_count = int(layer_ids.max()) + 1 if layer_ids.size else 0
        if layer_count < 2:
            return output_text + " " + self.tr(
                "Only {layers} layer is detected; expand the cell, choose another plane, or reduce the tolerance."
            ).format(layers=layer_count)
        guidance = output_text + " " + self.tr(
            "The first input has {layers} detected layers; verify the layer sequence below."
        ).format(layers=layer_count)
        if layer_count % 2 and operation.has_periodic_layer_axis(
            self._input_structure,
            params.miller_index,
        ):
            guidance += " " + self.tr(
                "Periodic A/B magnetic order does not close with an odd layer count."
            )
        return guidance

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.legacy_notice.clear()
        self.legacy_notice.hide()
        raw_params = dict(data_dict.get("params") or {})
        legacy = bool(
            "mode" in raw_params
            or "kvec" in raw_params
            or "mode" in data_dict
            or "kvec" in data_dict
        )
        if raw_params:
            params = GroupLabelParams(
                miller_index=raw_params.get(
                    "miller_index",
                    raw_params.get("kvec", "001"),
                ),
                layer_tolerance=raw_params.get("layer_tolerance", 0.05),
                group_a=raw_params.get("group_a", "A"),
                group_b=raw_params.get("group_b", "B"),
                overwrite=raw_params.get("overwrite", False),
            )
        else:
            params = GroupLabelParams(
                miller_index=data_dict.get("kvec", "001"),
                group_a=data_dict.get("group_a", "A"),
                group_b=data_dict.get("group_b", "B"),
                overwrite=data_dict.get("overwrite", True),
            )
        self.set_params(params)
        if legacy:
            migration_message = self.tr(
                "Legacy Group Label loaded: the old cell-phase and half-grid rules were removed. "
                "This card now detects real atomic layers; verify the preview before rerunning the workflow."
            )
            self.legacy_notice.setText("⚠ " + migration_message)
            self.legacy_notice.show()
            MessageManager.send_warning_message(migration_message)
