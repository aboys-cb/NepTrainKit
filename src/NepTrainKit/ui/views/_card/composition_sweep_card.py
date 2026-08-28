"""Card for planning unique target compositions on element simplexes."""

from __future__ import annotations

from qfluentwidgets import CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import CompositionSweepOperation, CompositionSweepParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import (
    add_translated_items,
    combo_value,
    set_combo_value,
)
from NepTrainKit.ui.widgets import CompactField, InspectorSection, MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CompositionSweepCard(MakeDataCard):
    """Create unique Comp targets without changing atomic occupancies."""

    group = "Alloy"
    card_name = "Composition Space Sampling"
    description = (
        "Plan unique binary-to-quinary target compositions. This card writes "
        "Comp(...) tags only; add Random Occupancy to change atoms."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Composition Space Sampling"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_sweep_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText(self.tr("For example: Co,Cr,Ni,Al"))
        self.elements_edit.setText("Co,Cr,Ni")
        elements_field = CompactField(
            self.tr("Candidate elements"), self.elements_edit, self.setting_widget
        )

        self.order_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.order_combo,
            [
                ("2", "Binary (2)"),
                ("3", "Ternary (3)"),
                ("4", "Quaternary (4)"),
                ("5", "Quinary (5)"),
                ("2,3", "Binary + ternary (2,3)"),
                ("2,3,4", "Binary to quaternary (2-4)"),
                ("4,5", "Quaternary + quinary (4,5)"),
                ("2,3,4,5", "All feasible (2-5)"),
                ("5,4,3,2", "All feasible, high-order first (5-2)"),
            ],
        )
        set_combo_value(self.order_combo, "2,3,4,5")
        order_field = CompactField(
            self.tr("Component counts"), self.order_combo, self.setting_widget
        )

        space_section = InspectorSection(
            self.tr("Composition space"),
            self.setting_widget,
            self.tr(
                "Creates unique target ratios only. Atomic species change after a downstream Random Occupancy card."
            ),
        )
        space_section.addWidget(elements_field)
        space_section.addWidget(order_field)

        self.method_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.method_combo, ["Grid", "Sobol"])
        method_field = CompactField(
            self.tr("Sampling method"), self.method_combo, self.setting_widget
        )

        self.step_frame = self._number_frame("float", 0.001, 1.0, 0.1, decimals=6)
        self.step_field = CompactField(
            self.tr("Grid fraction step"),
            self.step_frame,
            self.setting_widget,
            self.tr("For four or five components, use 1/n such as 0.1 or 0.05."),
            inline=True,
            input_max_width=144,
        )
        self.n_points_frame = self._number_frame("int", 1, 999999, 50)
        self.n_points_field = CompactField(
            self.tr("Sobol points per element combination"),
            self.n_points_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )

        self.minfrac_frame = self._number_frame(
            "float", 0.0, 1.0, 0.0, decimals=6
        )
        self.minfrac_field = CompactField(
            self.tr("Minimum fraction per included element"),
            self.minfrac_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        self.include_endpoints_checkbox = CheckBox(
            self.tr("Include simplex boundary points"), self.setting_widget
        )
        self.include_endpoints_checkbox.setChecked(True)

        sampling_section = InspectorSection(
            self.tr("Simplex sampling"), self.setting_widget
        )
        sampling_section.addWidget(method_field)
        sampling_section.addWidget(self.step_field)
        sampling_section.addWidget(self.n_points_field)
        sampling_section.addWidget(self.minfrac_field)
        sampling_section.addWidget(self.include_endpoints_checkbox)

        self.max_output_frame = self._number_frame(
            "int", 1, CompositionSweepOperation.MAX_OUTPUTS_PER_INPUT, 500
        )
        self.max_output_field = CompactField(
            self.tr("Maximum unique targets per input"),
            self.max_output_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        self.budget_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.budget_mode_combo,
            [
                ("Equal+Reflow", "Balance component counts"),
                ("Capacity-weighted", "Favor larger composition spaces"),
                ("Equal (legacy)", "Legacy equal split"),
            ],
        )
        set_combo_value(self.budget_mode_combo, "Equal+Reflow")
        budget_mode_field = CompactField(
            self.tr("Budget allocation"),
            self.budget_mode_combo,
            self.setting_widget,
        )
        budget_section = InspectorSection(
            self.tr("Output budget"),
            self.setting_widget,
            self.tr("The limit applies independently to every input structure."),
        )
        budget_section.addWidget(self.max_output_field)
        budget_section.addWidget(budget_mode_field)

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = self._number_frame("int", 0, 2**31 - 1, 0)
        self.seed_field = CompactField(
            self.tr("Random seed"),
            self.seed_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        self.seed_field.hide()
        reproducibility_section = InspectorSection(
            self.tr("Reproducibility"),
            self.setting_widget,
            self.tr(
                "The seed controls Sobol points and target ordering when the budget truncates the space."
            ),
        )
        reproducibility_section.addWidget(self.seed_checkbox)
        reproducibility_section.addWidget(self.seed_field)

        self.settingLayout.addWidget(space_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(sampling_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(budget_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(reproducibility_section, 3, 0, 1, 3)

        self.method_combo.currentIndexChanged.connect(self._update_method_widgets)
        self.seed_checkbox.toggled.connect(self._update_seed_visibility)
        self.elements_edit.textChanged.connect(self.refresh_compact_presentation)
        self.order_combo.currentIndexChanged.connect(self.refresh_compact_presentation)
        self.budget_mode_combo.currentIndexChanged.connect(
            self.refresh_compact_presentation
        )
        self.include_endpoints_checkbox.toggled.connect(
            self.refresh_compact_presentation
        )
        for frame in (
            self.step_frame,
            self.n_points_frame,
            self.minfrac_frame,
            self.seed_frame,
            self.max_output_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self.refresh_compact_presentation)

        self._update_method_widgets()
        self._update_seed_visibility(False)

    def _number_frame(
        self,
        kind: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int | None = None,
    ):
        frame = SpinBoxUnitInputFrame(self)
        frame.set_input("", 1, kind)
        if decimals is not None:
            frame.setDecimals(decimals)
        frame.setRange(minimum, maximum)
        frame.set_input_value([value])
        return frame

    def _update_method_widgets(self, *_args) -> None:
        is_sobol = combo_value(self.method_combo, "Grid") == "Sobol"
        self.n_points_field.setVisible(is_sobol)
        self.step_field.setVisible(not is_sobol)
        self.include_endpoints_checkbox.setVisible(not is_sobol)
        self.refresh_compact_presentation()

    def _update_seed_visibility(self, checked: bool) -> None:
        self.seed_field.setVisible(bool(checked))
        self.refresh_compact_presentation()

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

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def create_operation(self):
        return CompositionSweepOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError:
            return self.tr("Complete the composition-space definition")
        orders = ",".join(str(value) for value in summary["active_orders"])
        return self.tr("{method} · orders {orders} · {count} unique/input").format(
            method=summary["method"],
            orders=orders,
            count=summary["outputs_per_input"],
        )

    def get_guidance_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError as exc:
            return translate_runtime_message(exc)
        per_input = int(summary["outputs_per_input"])
        input_count = getattr(self, "_preview_input_count", None)
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        parts = []
        if input_count is None:
            parts.append(
                self.tr("Unique targets/input: {count}.").format(count=per_input)
            )
        else:
            parts.append(
                self.tr(
                    "Inputs {inputs} × targets/input {count} = outputs {total}."
                ).format(
                    inputs=input_count,
                    count=per_input,
                    total=input_count * per_input,
                )
            )
        emitted = ", ".join(
            f"{order}:{count}"
            for order, count in summary["emitted_by_order"].items()
        )
        parts.append(
            self.tr("Unique targets by nominal order: {counts}.").format(
                counts=emitted
            )
        )
        if summary["skipped_orders"]:
            skipped = ",".join(str(value) for value in summary["skipped_orders"])
            parts.append(
                self.tr("Skipped infeasible component counts: {orders}.").format(
                    orders=skipped
                )
            )
        parts.append(
            self.tr(
                "Only Comp(...) targets are written; add Random Occupancy next to change atomic species."
            )
        )
        return " ".join(parts)

    def get_params(self) -> CompositionSweepParams:
        return CompositionSweepParams(
            elements=self.elements_edit.text(),
            order=combo_value(self.order_combo, "2,3,4,5"),
            method=combo_value(self.method_combo, "Grid"),
            step=float(self.step_frame.get_input_value()[0]),
            n_points=int(self.n_points_frame.get_input_value()[0]),
            min_fraction=float(self.minfrac_frame.get_input_value()[0]),
            include_endpoints=self.include_endpoints_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
            budget_mode=combo_value(self.budget_mode_combo, "Equal+Reflow"),
        )

    def set_params(self, params: CompositionSweepParams) -> None:
        self.elements_edit.setText(params.elements)
        normalized_order = ",".join(
            str(value) for value in self.create_operation()._target_orders(params.order)
        )
        set_combo_value(self.order_combo, normalized_order)
        set_combo_value(self.method_combo, params.method)
        self.step_frame.set_input_value([float(params.step)])
        self.n_points_frame.set_input_value([int(params.n_points)])
        self.minfrac_frame.set_input_value([float(params.min_fraction)])
        self.include_endpoints_checkbox.setChecked(bool(params.include_endpoints))
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        set_combo_value(self.budget_mode_combo, params.budget_mode)
        self._update_method_widgets()
        self._update_seed_visibility(bool(params.use_seed))

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
            params = CompositionSweepParams(
                elements=raw_params.get("elements", "Co,Cr,Ni"),
                order=raw_params.get("order", "2,3,4,5"),
                method=raw_params.get("method", "Grid"),
                step=raw_params.get("step", 0.1),
                n_points=raw_params.get("n_points", 50),
                min_fraction=raw_params.get("min_fraction", 0.0),
                include_endpoints=raw_params.get("include_endpoints", True),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
                max_outputs=raw_params.get("max_outputs", 500),
                budget_mode=raw_params.get("budget_mode", "Equal+Reflow"),
            )
        else:
            params = CompositionSweepParams(
                elements=data_dict.get("elements", "Co,Cr,Ni"),
                order=data_dict.get("order", "2,3,4,5"),
                method=data_dict.get("method", "Grid"),
                step=data_dict.get("step", [0.1])[0],
                n_points=data_dict.get("n_points", [50])[0],
                min_fraction=data_dict.get("min_fraction", [0.0])[0],
                include_endpoints=data_dict.get("include_endpoints", True),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
                max_outputs=data_dict.get("max_outputs", [500])[0],
                budget_mode=data_dict.get("budget_mode", "Equal+Reflow"),
            )
        self.set_params(params)
