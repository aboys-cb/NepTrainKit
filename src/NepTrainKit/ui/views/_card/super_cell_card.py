"""Card for building diagonal supercells with explicit output semantics."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import SuperCellOperation, SuperCellParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import (
    add_translated_items,
    combo_value,
    set_combo_value,
)
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class SuperCellCard(MakeDataCard):
    """Create diagonal supercells from repeat, length, or atom-budget targets."""

    group = "Lattice"
    card_name = "Super Cell"
    menu_icon = r":/images/src/images/supercell.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]
    separator = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Super Cell"))
        self.init_ui()

    def init_ui(self):
        """Build a mode-driven form that only exposes parameters in effect."""
        self.setObjectName("super_cell_card_widget")

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("scale", "Repeat factors"),
                ("cell", "Target lengths"),
                ("max_atoms", "Atom budget"),
            ],
        )
        self.mode_field = CompactField(
            self.tr("Expansion basis"),
            self.mode_combo,
            self.setting_widget,
            self.tr("Choose explicit repeats, target lattice-vector lengths, or a strict atom limit."),
        )

        self.super_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_scale_condition_frame.set_input("×", 3)
        self.super_scale_condition_frame.setRange(1, 999)
        self.super_scale_condition_frame.set_input_value([3, 3, 3])
        self.super_scale_field = CompactField(
            self.tr("Repeat factors (a, b, c)"),
            self.super_scale_condition_frame,
            self.setting_widget,
            self.tr("Each value repeats the complete input cell along one lattice vector."),
        )

        self.super_cell_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_cell_condition_frame.set_input("Å", 3, "float")
        self.super_cell_condition_frame.setRange(0.001, 9999.0)
        self.super_cell_condition_frame.setDecimals(3)
        self.super_cell_condition_frame.set_input_value([20.0, 20.0, 20.0])
        self.target_cell_field = CompactField(
            self.tr("Target lengths (a, b, c)"),
            self.super_cell_condition_frame,
            self.setting_widget,
            self.tr("Integer repeats are derived from the norms of the three input lattice vectors."),
        )

        self.target_policy_control = SegmentedControl(parent=self.setting_widget)
        self.target_policy_control.addItem(self.tr("At least"), userData="at_least")
        self.target_policy_control.addItem(self.tr("At most"), userData="at_most")
        self.target_policy_field = CompactField(
            self.tr("Length constraint"),
            self.target_policy_control,
            self.setting_widget,
            self.tr(
                "At least uses ceiling; at most uses floor, and exact multiples give the same factor. "
                "This card cannot shrink a cell, so an input already longer than an at-most target stays unchanged."
            ),
        )

        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input(self.tr("atoms"), 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_condition_frame.set_input_value([100])
        self.max_atoms_field = CompactField(
            self.tr("Atom limit"),
            self.max_atoms_condition_frame,
            self.setting_widget,
            self.tr(
                "A strict upper bound. Single output first maximizes atom count; ties minimize the "
                "longest/shortest output-vector ratio. A limit below the input atom count is invalid."
            ),
        )

        self.output_mode_control = SegmentedControl(parent=self.setting_widget)
        self.output_mode_control.addItem(self.tr("One supercell"), userData="single")
        self.output_mode_control.addItem(self.tr("Enumerate sizes"), userData="enumerate")
        self.output_mode_field = CompactField(
            self.tr("Output mode"),
            self.output_mode_control,
            self.setting_widget,
            self.tr("Enumeration includes every integer repeat combination up to the selected target."),
        )

        strategy_section = InspectorSection(
            self.tr("Expansion strategy"),
            self.setting_widget,
            self.tr("The operation only repeats complete cells; it never strains lattice vectors."),
        )
        strategy_grid = ResponsiveFormGrid(strategy_section, two_column_threshold=520)
        for field in (
            self.mode_field,
            self.super_scale_field,
            self.target_cell_field,
            self.target_policy_field,
            self.max_atoms_field,
            self.output_mode_field,
        ):
            strategy_grid.add_field(field, span=2 if field is self.mode_field else 1)
        strategy_section.addWidget(strategy_grid)

        axis_row = QWidget(self.setting_widget)
        axis_layout = QHBoxLayout(axis_row)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.setSpacing(14)
        self.fixed_axis_a_checkbox = CheckBox(self.tr("a"), axis_row)
        self.fixed_axis_b_checkbox = CheckBox(self.tr("b"), axis_row)
        self.fixed_axis_c_checkbox = CheckBox(self.tr("c"), axis_row)
        for checkbox in (
            self.fixed_axis_a_checkbox,
            self.fixed_axis_b_checkbox,
            self.fixed_axis_c_checkbox,
        ):
            axis_layout.addWidget(checkbox)
        axis_layout.addStretch(1)
        self.fixed_axes_field = CompactField(
            self.tr("Lock lattice-vector repeats"),
            axis_row,
            self.setting_widget,
            self.tr("A locked axis ignores the active target and uses its multiplier below."),
        )

        self.fixed_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.fixed_scale_condition_frame.set_input("×", 3)
        self.fixed_scale_condition_frame.setRange(1, 999)
        self.fixed_scale_condition_frame.set_input_value([1, 1, 1])
        self.fixed_scale_field = CompactField(
            self.tr("Locked multipliers (a, b, c)"),
            self.fixed_scale_condition_frame,
            self.setting_widget,
        )

        axes_section = InspectorSection(
            self.tr("Axis overrides"),
            self.setting_widget,
            self.tr("Useful for slabs: lock the non-periodic normal while expanding in-plane."),
        )
        axes_section.addWidget(self.fixed_axes_field)
        axes_section.addWidget(self.fixed_scale_field)

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.output_preview)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(10)
        self.settingLayout.addWidget(strategy_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(axes_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 2, 0, 1, 3)

        self.mode_combo.currentIndexChanged.connect(self._update_widgets)
        self.output_mode_control.currentIndexChanged.connect(self._update_widgets)
        self.target_policy_control.currentIndexChanged.connect(self._update_widgets)
        for checkbox in (
            self.fixed_axis_a_checkbox,
            self.fixed_axis_b_checkbox,
            self.fixed_axis_c_checkbox,
        ):
            checkbox.toggled.connect(self._update_widgets)
        for frame in (
            self.super_scale_condition_frame,
            self.super_cell_condition_frame,
            self.max_atoms_condition_frame,
            self.fixed_scale_condition_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._update_output_preview)
        self._update_widgets()

    @staticmethod
    def _set_segment_value(control: SegmentedControl, value: str) -> None:
        for index in range(control.count()):
            if control.itemData(index) == value:
                control.setCurrentIndex(index)
                return

    def _get_fixed_axis_flags(self) -> tuple[bool, bool, bool]:
        return (
            self.fixed_axis_a_checkbox.isChecked(),
            self.fixed_axis_b_checkbox.isChecked(),
            self.fixed_axis_c_checkbox.isChecked(),
        )

    def _get_fixed_axis_values(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.fixed_scale_condition_frame.get_input_value())

    def _update_widgets(self, *_args) -> None:
        mode = combo_value(self.mode_combo, "scale")
        self.super_scale_field.setVisible(mode == "scale")
        self.target_cell_field.setVisible(mode == "cell")
        self.target_policy_field.setVisible(mode == "cell")
        self.max_atoms_field.setVisible(mode == "max_atoms")

        flags = self._get_fixed_axis_flags()
        self.fixed_scale_field.setVisible(any(flags))
        for enabled, control in zip(flags, self.fixed_scale_condition_frame.object_list):
            control.setEnabled(enabled)
        self.output_mode_field.set_helper_text(
            self.tr("Enumeration lists every feasible integer repeat triple and stops above 1000 outputs.")
            if mode == "max_atoms"
            else self.tr("Enumeration includes every integer repeat combination from 1 up to the target factors.")
        )
        self._update_output_preview()

    def _first_input_structure(self):
        try:
            return self.dataset[0] if self.dataset else None
        except (TypeError, KeyError, IndexError):
            return next(iter(self.dataset), None) if self.dataset is not None else None

    def _update_output_preview(self, *_args) -> None:
        params = self.get_params()
        structure = self._first_input_structure()
        if structure is None:
            if params.mode == "scale":
                factors = tuple(
                    params.fixed_axis_scale[index]
                    if params.fixed_axis_flags[index]
                    else params.super_scale[index]
                    for index in range(3)
                )
                count = (
                    math.prod(
                        1 if params.fixed_axis_flags[index] else factors[index]
                        for index in range(3)
                    )
                    if params.output_mode == "enumerate"
                    else 1
                )
                self.output_preview.setText(
                    self.tr(
                        "Per input: {count} output(s); largest repeat is {a} × {b} × {c}; "
                        "atom count scales by {factor}."
                    ).format(
                        count=count,
                        a=factors[0],
                        b=factors[1],
                        c=factors[2],
                        factor=math.prod(factors),
                    )
                )
            else:
                self.output_preview.setText(
                    self.tr("Attach an input structure to resolve exact repeat factors and atom counts.")
                )
            return

        try:
            factors = self.create_operation().plan_factors(structure, params)
            largest = max(factors, key=math.prod)
            output_atoms = len(structure) * math.prod(largest)
            self.output_preview.setText(
                self.tr(
                    "First input: {count} output(s); largest repeat is {a} × {b} × {c}; "
                    "{input_atoms} → {output_atoms} atoms."
                ).format(
                    count=len(factors),
                    a=largest[0],
                    b=largest[1],
                    c=largest[2],
                    input_atoms=len(structure),
                    output_atoms=output_atoms,
                )
            )
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def set_dataset(self, dataset):
        super().set_dataset(dataset)
        self._update_output_preview()

    def create_operation(self):
        return SuperCellOperation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        if params.mode == "scale":
            detail = " × ".join(str(value) for value in params.super_scale)
        elif params.mode == "cell":
            detail = self.tr("{a:g} × {b:g} × {c:g} Å").format(
                a=params.target_cell[0], b=params.target_cell[1], c=params.target_cell[2]
            )
        else:
            detail = self.tr("≤ {count} atoms").format(count=params.max_atoms)
        return self.tr("{mode} · {detail} · {outputs}").format(
            mode=self.mode_combo.currentText(),
            detail=detail,
            outputs=self.output_mode_control.currentText(),
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Supercell repeats use lattice vectors a, b, and c, not Cartesian x, y, and z. "
            "Lock a non-periodic slab normal explicitly before expanding in-plane."
        )

    def get_params(self) -> SuperCellParams:
        return SuperCellParams(
            mode=combo_value(self.mode_combo, "scale"),
            output_mode=str(self.output_mode_control.currentData() or "single"),
            target_policy=str(self.target_policy_control.currentData() or "at_least"),
            super_scale=tuple(
                int(value) for value in self.super_scale_condition_frame.get_input_value()
            ),
            target_cell=tuple(
                float(value) for value in self.super_cell_condition_frame.get_input_value()
            ),
            max_atoms=int(self.max_atoms_condition_frame.get_input_value()[0]),
            fixed_axis_flags=self._get_fixed_axis_flags(),
            fixed_axis_scale=self._get_fixed_axis_values(),
        )

    def set_params(self, params: SuperCellParams) -> None:
        set_combo_value(self.mode_combo, params.mode)
        self._set_segment_value(self.output_mode_control, params.output_mode)
        self._set_segment_value(self.target_policy_control, params.target_policy)
        self.super_scale_condition_frame.set_input_value(list(params.super_scale))
        self.super_cell_condition_frame.set_input_value(list(params.target_cell))
        self.max_atoms_condition_frame.set_input_value([int(params.max_atoms)])
        for checkbox, checked in zip(
            (
                self.fixed_axis_a_checkbox,
                self.fixed_axis_b_checkbox,
                self.fixed_axis_c_checkbox,
            ),
            params.fixed_axis_flags,
        ):
            checkbox.setChecked(bool(checked))
        self.fixed_scale_condition_frame.set_input_value(list(params.fixed_axis_scale))
        self._update_widgets()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    @staticmethod
    def _migrate_behavior(behavior_type: int) -> tuple[str, str]:
        behavior = int(behavior_type)
        return (
            "enumerate" if behavior == 1 else "single",
            "at_least" if behavior == 2 else "at_most",
        )

    def from_dict(self, data_dict):
        """Restore current parameters and migrate legacy behavior/radio fields."""
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            legacy_output, legacy_policy = self._migrate_behavior(
                raw_params.get("behavior_type", 0)
            )
            params = SuperCellParams(
                mode=raw_params.get("mode", "scale"),
                output_mode=raw_params.get("output_mode", legacy_output),
                target_policy=raw_params.get("target_policy", legacy_policy),
                super_scale=tuple(raw_params.get("super_scale", [3, 3, 3])),
                target_cell=tuple(raw_params.get("target_cell", [20.0, 20.0, 20.0])),
                max_atoms=raw_params.get("max_atoms", 100),
                fixed_axis_flags=tuple(raw_params.get("fixed_axis_flags", [False, False, False])),
                fixed_axis_scale=tuple(raw_params.get("fixed_axis_scale", [1, 1, 1])),
            )
        else:
            legacy_output, legacy_policy = self._migrate_behavior(
                data_dict.get("super_cell_type", data_dict.get("behavior_type", 0))
            )
            mode = data_dict.get("mode")
            if mode not in {"scale", "cell", "max_atoms"}:
                if data_dict.get("super_cell_radio_button", False):
                    mode = "cell"
                elif data_dict.get("max_atoms_radio_button", False):
                    mode = "max_atoms"
                else:
                    mode = "scale"
            raw_max_atoms = data_dict.get("max_atoms_condition", [100])
            max_atoms = raw_max_atoms[0] if isinstance(raw_max_atoms, (list, tuple)) else raw_max_atoms
            params = SuperCellParams(
                mode=mode,
                output_mode=data_dict.get("output_mode", legacy_output),
                target_policy=data_dict.get("target_policy", legacy_policy),
                super_scale=tuple(data_dict.get("super_scale_condition", data_dict.get("super_scale", [3, 3, 3]))),
                target_cell=tuple(
                    data_dict.get(
                        "super_cell_condition",
                        data_dict.get("target_cell", [20.0, 20.0, 20.0]),
                    )
                ),
                max_atoms=data_dict.get("max_atoms", max_atoms),
                fixed_axis_flags=tuple(data_dict.get("fixed_axis_flags", [False, False, False])),
                fixed_axis_scale=tuple(data_dict.get("fixed_axis_scale", [1, 1, 1])),
            )
        self.set_params(params)
