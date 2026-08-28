"""Card for interdiffusing near-interface layers of a bilayer."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import InterfaceLayerMixOperation, InterfaceLayerMixParams
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class InterfaceLayerMixCard(MakeDataCard):
    """Exchange unlike species between selected layers on two interface sides."""

    group = "Alloy"
    card_name = "Interface Layer Mixing"
    description = (
        "Select one interface and exchange unlike species between nearby layers while preserving the total composition."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "Sun Xiaojian", "role": "author"},
        {"name": "NepTrainKit", "role": "maintainer"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count: int | None = None
        self.setTitle(self.tr("Interface Layer Mixing"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("interface_layer_mix_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.axis_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.axis_combo,
            [("auto", "Auto detect"), ("a", "Fractional a"), ("b", "Fractional b"), ("c", "Fractional c")],
        )
        set_combo_value(self.axis_combo, "auto")
        self.axis_field = CompactField(self.tr("Layer direction"), self.axis_combo, self.setting_widget)

        self.position_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.position_mode_combo,
            [("auto", "Auto locate"), ("manual", "Manual position")],
        )
        set_combo_value(self.position_mode_combo, "auto")
        self.position_mode_field = CompactField(
            self.tr("Interface position"), self.position_mode_combo, self.setting_widget
        )
        self.interface_position_frame = self._float_frame("", 0.000001, 0.999999, 0.5, 6)
        self.interface_position_field = CompactField(
            self.tr("Fractional position"),
            self.interface_position_frame,
            self.setting_widget,
            self.tr("Lower coordinates form the L side; the rest form the R side."),
            inline=True,
            input_max_width=150,
        )
        self.interface_position_field.hide()
        interface_section = InspectorSection(
            self.tr("Interface"),
            self.setting_widget,
            self.tr(
                "Layers follow a fractional lattice coordinate; its constant-coordinate planes have a reciprocal-lattice normal."
            ),
        )
        interface_grid = ResponsiveFormGrid(interface_section, two_column_threshold=520)
        interface_grid.add_field(self.axis_field)
        interface_grid.add_field(self.position_mode_field)
        interface_grid.add_field(self.interface_position_field, span=2)
        interface_section.addWidget(interface_grid)

        self.left_layers_frame = self._integer_frame(1, 100, 2)
        self.left_layers_field = CompactField(
            self.tr("L-side layers"),
            self.left_layers_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        self.right_layers_frame = self._integer_frame(1, 100, 2)
        self.right_layers_field = CompactField(
            self.tr("R-side layers"),
            self.right_layers_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        self.layer_tolerance_frame = self._float_frame("Å", 0.001, 10.0, 0.25, 3)
        self.layer_tolerance_field = CompactField(
            self.tr("Layer tolerance"),
            self.layer_tolerance_frame,
            self.setting_widget,
            self.tr("Coordinate separations no larger than this value are clustered into one layer."),
            inline=True,
            input_max_width=150,
        )
        layer_section = InspectorSection(
            self.tr("Mixed layers"),
            self.setting_widget,
            self.tr("Counts start at the selected interface and extend toward lower/higher fractional coordinates."),
        )
        layer_grid = ResponsiveFormGrid(layer_section, two_column_threshold=420)
        layer_grid.add_field(self.left_layers_field)
        layer_grid.add_field(self.right_layers_field)
        layer_grid.add_field(self.layer_tolerance_field, span=2)
        layer_section.addWidget(layer_grid)

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.mode_combo, [("fixed", "Fixed"), ("gradient", "Gradient")])
        set_combo_value(self.mode_combo, "fixed")
        self.mode_field = CompactField(self.tr("Mixing schedule"), self.mode_combo, self.setting_widget)
        self.concentration_frame = self._float_frame("%", 0.0, 100.0, 50.0, 2)
        self.concentration_field = CompactField(
            self.tr("Target changed atoms"),
            self.concentration_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.gradient_start_frame = self._float_frame("%", 0.0, 100.0, 0.0, 2)
        self.gradient_start_field = CompactField(
            self.tr("Start"),
            self.gradient_start_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.gradient_end_frame = self._float_frame("%", 0.0, 100.0, 100.0, 2)
        self.gradient_end_field = CompactField(
            self.tr("End"),
            self.gradient_end_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.gradient_start_field.hide()
        self.gradient_end_field.hide()
        self.num_structures_frame = self._integer_frame(1, 10000, 1)
        self.num_structures_field = CompactField(
            self.tr("Outputs per input"),
            self.num_structures_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        schedule_section = InspectorSection(
            self.tr("Mixing amount"),
            self.setting_widget,
            self.tr(
                "Only unlike species are paired. Finite atom counts may make the realized percentage differ from the target."
            ),
        )
        schedule_grid = ResponsiveFormGrid(schedule_section, two_column_threshold=420)
        schedule_grid.add_field(self.mode_field)
        schedule_grid.add_field(self.num_structures_field)
        schedule_grid.add_field(self.concentration_field, span=2)
        schedule_grid.add_field(self.gradient_start_field)
        schedule_grid.add_field(self.gradient_end_field)
        schedule_section.addWidget(schedule_grid)

        self.seed_checkbox = CheckBox(self.tr("Use fixed random seed"), self.setting_widget)
        self.seed_frame = self._integer_frame(0, 2**31 - 1, 0)
        self.seed_field = CompactField(
            self.tr("Random seed"),
            self.seed_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.seed_field.hide()
        random_section = InspectorSection(self.tr("Randomness"), self.setting_widget)
        random_section.addWidget(self.seed_checkbox)
        random_section.addWidget(self.seed_field)

        self.settingLayout.addWidget(interface_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(layer_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(schedule_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(random_section, 3, 0, 1, 3)

        self.axis_combo.currentIndexChanged.connect(self.refresh_compact_presentation)
        self.position_mode_combo.currentIndexChanged.connect(self._update_position_visibility)
        self.mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        self.seed_checkbox.toggled.connect(self._update_seed_visibility)
        for frame in (
            self.interface_position_frame,
            self.left_layers_frame,
            self.right_layers_frame,
            self.layer_tolerance_frame,
            self.concentration_frame,
            self.gradient_start_frame,
            self.gradient_end_frame,
            self.num_structures_frame,
            self.seed_frame,
        ):
            frame.object_list[0].valueChanged.connect(self.refresh_compact_presentation)
        self._update_position_visibility()
        self._update_mode_visibility()
        self._update_seed_visibility(False)

    def _integer_frame(self, minimum: int, maximum: int, value: int):
        frame = SpinBoxUnitInputFrame(self.setting_widget)
        frame.set_input("", 1, "int")
        frame.setRange(minimum, maximum)
        frame.set_input_value([value])
        return frame

    def _float_frame(self, unit: str, minimum: float, maximum: float, value: float, decimals: int):
        frame = SpinBoxUnitInputFrame(self.setting_widget)
        frame.set_input(unit, 1, "float")
        frame.setRange(minimum, maximum)
        frame.setDecimals(decimals)
        frame.set_input_value([value])
        return frame

    def _update_position_visibility(self, *_args) -> None:
        self.interface_position_field.setVisible(combo_value(self.position_mode_combo, "auto") == "manual")
        self.refresh_compact_presentation()

    def _update_mode_visibility(self, *_args) -> None:
        is_gradient = combo_value(self.mode_combo, "fixed") == "gradient"
        self.concentration_field.setVisible(not is_gradient)
        self.gradient_start_field.setVisible(is_gradient)
        self.gradient_end_field.setVisible(is_gradient)
        self.refresh_compact_presentation()

    def _update_seed_visibility(self, checked: bool) -> None:
        self.seed_field.setVisible(bool(checked))
        self.refresh_compact_presentation()

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    @staticmethod
    def _dataset_count(dataset) -> int:
        if dataset is None:
            return 0
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return 1
        try:
            return len(dataset)
        except TypeError:
            return 0

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self.refresh_compact_presentation()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self.refresh_compact_presentation()

    def create_operation(self):
        return InterfaceLayerMixOperation()

    def _summary(self):
        if self._input_structure is None:
            return None
        return self.create_operation().interface_summary(self._input_structure, self.get_params())

    def get_summary_text(self) -> str:
        try:
            summary = self._summary()
        except (CardOperationError, ValueError) as exc:
            return self.tr("Preview unavailable: {reason}").format(reason=translate_runtime_message(exc))
        if summary is None:
            return self.tr("Select one interface · {count} output(s)/input").format(
                count=self.get_params().num_structures
            )
        amount = self._format_amount(summary["requested_concentrations"], summary["effective_concentrations"])
        return self.tr("fractional {axis} @ {pos} · L {left}/{right} R sites · {amount}").format(
            axis=summary["axis"],
            pos=f"{summary['position']:.3f}",
            left=summary["n_left"],
            right=summary["n_right"],
            amount=amount,
        )

    def get_guidance_text(self) -> str:
        parts: list[str] = []
        input_count = self._preview_input_count
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        outputs = self.get_params().num_structures
        if input_count is None:
            parts.append(self.tr("Outputs/input: {outputs}.").format(outputs=outputs))
        else:
            parts.append(
                self.tr("Inputs {inputs} × {outputs}/input = {total} outputs.").format(
                    inputs=input_count, outputs=outputs, total=input_count * outputs
                )
            )
        try:
            summary = self._summary()
        except (CardOperationError, ValueError) as exc:
            parts.append(translate_runtime_message(exc))
            return " ".join(parts)
        if summary is not None:
            parts.append(
                self.tr(
                    "First input: L {left_layers} layer(s) {left_formula}, R {right_layers} layer(s) {right_formula}; at most {pairs} unlike pairs."
                ).format(
                    left_layers=summary["left_layers"],
                    left_formula=summary["left_formula"],
                    right_layers=summary["right_layers"],
                    right_formula=summary["right_formula"],
                    pairs=summary["pair_capacity"],
                )
            )
            requested = summary["requested_concentrations"]
            effective = summary["effective_concentrations"]
            if any(not np.isclose(a, b) for a, b in zip(requested, effective)):
                parts.append(
                    self.tr("Discrete realization: {amount}.").format(amount=self._format_amount(requested, effective))
                )
            pbc = np.asarray(self._input_structure.get_pbc(), dtype=bool)
            axis_idx = {"a": 0, "b": 1, "c": 2}[summary["axis"]]
            if bool(pbc[axis_idx]):
                parts.append(
                    self.tr(
                        "This edits the selected interface only; a periodic boundary along {axis} creates a second interface."
                    ).format(axis=summary["axis"])
                )
        parts.append(
            self.tr("Changed structures discard stale energy, force, stress, virial, and calculated magnetic labels.")
        )
        return " ".join(parts)

    def _format_amount(self, requested, effective) -> str:
        def pct(value):
            return f"{100.0 * float(value):.3g}%"

        if len(requested) == 1:
            if np.isclose(requested[0], effective[0]):
                return self.tr("realized {value}").format(value=pct(effective[0]))
            return self.tr("target {target} → realized {actual}").format(
                target=pct(requested[0]), actual=pct(effective[0])
            )
        return self.tr("target {start}–{end} → realized {actual_start}–{actual_end}").format(
            start=pct(requested[0]),
            end=pct(requested[-1]),
            actual_start=pct(effective[0]),
            actual_end=pct(effective[-1]),
        )

    def get_params(self) -> InterfaceLayerMixParams:
        return InterfaceLayerMixParams(
            axis=combo_value(self.axis_combo, "auto"),
            auto_position=combo_value(self.position_mode_combo, "auto") == "auto",
            interface_position=float(self.interface_position_frame.get_input_value()[0]),
            layer_tolerance=float(self.layer_tolerance_frame.get_input_value()[0]),
            left_layers=int(self.left_layers_frame.get_input_value()[0]),
            right_layers=int(self.right_layers_frame.get_input_value()[0]),
            mode=combo_value(self.mode_combo, "fixed"),
            concentration=float(self.concentration_frame.get_input_value()[0]) / 100.0,
            gradient_start=float(self.gradient_start_frame.get_input_value()[0]) / 100.0,
            gradient_end=float(self.gradient_end_frame.get_input_value()[0]) / 100.0,
            num_structures=int(self.num_structures_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: InterfaceLayerMixParams) -> None:
        set_combo_value(self.axis_combo, params.axis)
        set_combo_value(self.position_mode_combo, "auto" if params.auto_position else "manual")
        self.interface_position_frame.set_input_value([float(params.interface_position)])
        self.layer_tolerance_frame.set_input_value([float(params.layer_tolerance)])
        self.left_layers_frame.set_input_value([int(params.left_layers)])
        self.right_layers_frame.set_input_value([int(params.right_layers)])
        set_combo_value(self.mode_combo, params.mode)
        self.concentration_frame.set_input_value([float(params.concentration) * 100.0])
        self.gradient_start_frame.set_input_value([float(params.gradient_start) * 100.0])
        self.gradient_end_frame.set_input_value([float(params.gradient_end) * 100.0])
        self.num_structures_frame.set_input_value([int(params.num_structures)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_position_visibility()
        self._update_mode_visibility()
        self._update_seed_visibility(bool(params.use_seed))

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            normalized = dict(raw_params)
            normalized.setdefault("layer_tolerance", 0.25)
            params = InterfaceLayerMixParams(**normalized)
        else:
            params = InterfaceLayerMixParams()
        self.set_params(params)
