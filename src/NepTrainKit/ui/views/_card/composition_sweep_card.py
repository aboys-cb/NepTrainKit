"""Card for sampling target ratios in binary-to-quinary composition spaces."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.alloy import CompositionSweepOperation, CompositionSweepParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CompositionSweepCard(MakeDataCard):
    """Sample target compositions without changing atomic occupancies."""

    group = "Alloy"
    card_name = "Composition Space Sampling"
    description = (
        "Sample target ratios in binary-to-quinary composition spaces. "
        "This card only writes Comp(...) tags; add Random Occupancy to change atoms."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Composition Space Sampling"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_sweep_card_widget")

        self.elements_label = BodyLabel(self.tr("Elements"), self.setting_widget)
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText(self.tr("Co,Cr,Ni,Al"))
        self.elements_edit.setText("Co,Cr,Ni")
        self.elements_label.setToolTip(self.tr("Candidate elements for binary/ternary/quaternary/quinary combinations"))
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))

        self.order_label = BodyLabel(self.tr("Component counts"), self.setting_widget)
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
            ]
        )
        set_combo_value(self.order_combo, "2,3,4,5")
        self.order_label.setToolTip(
            self.tr("Select binary, ternary, quaternary, or quinary composition spaces")
        )
        self.order_label.installEventFilter(ToolTipFilter(self.order_label, 300, ToolTipPosition.TOP))

        self.method_label = BodyLabel(self.tr("Sampling method"), self.setting_widget)
        self.method_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.method_combo, ["Grid", "Sobol"])
        self.method_label.setToolTip(self.tr("Grid step scan or Sobol low-discrepancy sampling on simplex"))
        self.method_label.installEventFilter(ToolTipFilter(self.method_label, 300, ToolTipPosition.TOP))

        self.step_label = BodyLabel(self.tr("Composition step"), self.setting_widget)
        self.step_frame = SpinBoxUnitInputFrame(self)
        self.step_frame.set_input("", 1, "float")
        self.step_frame.setDecimals(6)
        self.step_frame.setRange(1e-6, 1.0)
        self.step_frame.set_input_value([0.1])
        self.step_label.setToolTip(
            self.tr("Grid spacing in fraction units; 0.1 means 10 percentage points")
        )
        self.step_label.installEventFilter(ToolTipFilter(self.step_label, 300, ToolTipPosition.TOP))

        self.n_points_label = BodyLabel(self.tr("Points per combination"), self.setting_widget)
        self.n_points_frame = SpinBoxUnitInputFrame(self)
        self.n_points_frame.set_input("", 1, "int")
        self.n_points_frame.setRange(1, 999999)
        self.n_points_frame.set_input_value([50])
        self.n_points_label.setToolTip(
            self.tr("Number of Sobol target ratios generated for each element combination")
        )
        self.n_points_label.installEventFilter(ToolTipFilter(self.n_points_label, 300, ToolTipPosition.TOP))

        self.minfrac_label = BodyLabel(self.tr("Minimum element fraction"), self.setting_widget)
        self.minfrac_frame = SpinBoxUnitInputFrame(self)
        self.minfrac_frame.set_input("", 1, "float")
        self.minfrac_frame.setDecimals(6)
        self.minfrac_frame.setRange(0.0, 1.0)
        self.minfrac_frame.set_input_value([0.0])

        self.include_endpoints_checkbox = CheckBox(self.tr("Include boundary points"), self.setting_widget)
        self.include_endpoints_checkbox.setChecked(True)
        self.include_endpoints_checkbox.setToolTip(
            self.tr("Include pure-element endpoints and compositions with zero fraction")
        )
        self.include_endpoints_checkbox.installEventFilter(
            ToolTipFilter(self.include_endpoints_checkbox, 300, ToolTipPosition.TOP)
        )

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.max_output_label = BodyLabel(self.tr("Max target compositions/input"), self.setting_widget)
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("", 1, "int")
        self.max_output_frame.setRange(1, 9999999)
        self.max_output_frame.set_input_value([500])

        self.budget_mode_label = BodyLabel(self.tr("Budget allocation"), self.setting_widget)
        self.budget_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.budget_mode_combo,
            [
                ("Equal+Reflow", "Balance component counts"),
                ("Capacity-weighted", "Favor larger composition spaces"),
                ("Equal (legacy)", "Legacy equal split"),
            ]
        )
        set_combo_value(self.budget_mode_combo, "Equal+Reflow")
        self.budget_mode_label.setToolTip(
            self.tr("How the output limit is split across selected component counts")
        )
        self.budget_mode_label.installEventFilter(ToolTipFilter(self.budget_mode_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.elements_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_edit, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.order_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.order_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.method_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.method_combo, 2, 1, 1, 2)

        self.settingLayout.addWidget(self.step_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.step_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.n_points_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.n_points_frame, 4, 1, 1, 2)

        self.settingLayout.addWidget(self.minfrac_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.minfrac_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.include_endpoints_checkbox, 6, 0, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.budget_mode_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.budget_mode_combo, 9, 1, 1, 2)

        self.method_combo.currentTextChanged.connect(self._update_method_widgets)
        self._update_method_widgets()

    def _update_method_widgets(self) -> None:
        is_sobol = combo_value(self.method_combo) == "Sobol"
        self.n_points_label.setVisible(is_sobol)
        self.n_points_frame.setVisible(is_sobol)
        self.step_label.setVisible(not is_sobol)
        self.step_frame.setVisible(not is_sobol)
        self.include_endpoints_checkbox.setVisible(not is_sobol)

        # Keep enabled-state gating as a defensive fallback.
        self.n_points_label.setEnabled(is_sobol)
        self.n_points_frame.setEnabled(is_sobol)
        self.step_label.setEnabled(not is_sobol)
        self.step_frame.setEnabled(not is_sobol)
        self.include_endpoints_checkbox.setEnabled(not is_sobol)

    def create_operation(self):
        """Return the UI-independent composition sweep operation."""
        return CompositionSweepOperation()

    def get_params(self) -> CompositionSweepParams:
        """Read composition sweep parameters from UI controls."""
        return CompositionSweepParams(
            elements=self.elements_edit.text(),
            order=combo_value(self.order_combo),
            method=combo_value(self.method_combo),
            step=float(self.step_frame.get_input_value()[0]),
            n_points=int(self.n_points_frame.get_input_value()[0]),
            min_fraction=float(self.minfrac_frame.get_input_value()[0]),
            include_endpoints=self.include_endpoints_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
            budget_mode=combo_value(self.budget_mode_combo),
        )

    def set_params(self, params: CompositionSweepParams) -> None:
        """Apply composition sweep parameters to UI controls."""
        self.elements_edit.setText(params.elements)
        set_combo_value(self.order_combo, params.order)
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

    def process_structure(self, structure):
        """Create composition-tagged structures from UI-independent parameters."""
        try:
            return self.create_operation().run_structure(structure, self.get_params())
        except NotImplementedError as exc:
            MessageManager.send_warning_message(f"CompositionSweep Grid: {exc} (use Sobol for order>=4)")
            return [structure]

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
