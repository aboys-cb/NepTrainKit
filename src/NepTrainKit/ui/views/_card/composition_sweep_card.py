"""Card for expanding structures into binary/ternary composition batches."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.alloy import parse_element_list, simplex_grid_points, simplex_sobol_points
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CompositionSweepCard(MakeDataCard):
    """Create multiple copies per input structure, each annotated with a target composition."""

    group = "Alloy"
    card_name = "Composition Sweep"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Composition Sweep")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_sweep_card_widget")

        self.elements_label = BodyLabel("Elements", self.setting_widget)
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText("Co,Cr,Ni,Al")
        self.elements_edit.setText("Co,Cr,Ni")
        self.elements_label.setToolTip("Candidate elements for binary/ternary combinations")
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))

        self.order_label = BodyLabel("Order", self.setting_widget)
        self.order_combo = ComboBox(self.setting_widget)
        self.order_combo.addItems(
            [
                "2",
                "3",
                "4",
                "5",
                "2,3",
                "4,5",
                "2,3,4,5",
                "5,4,3,2",
            ]
        )
        self.order_combo.setCurrentText("2,3,4,5")
        self.order_label.setToolTip("Orders to generate, e.g. 2,3,4,5")
        self.order_label.installEventFilter(ToolTipFilter(self.order_label, 300, ToolTipPosition.TOP))

        self.method_label = BodyLabel("Method", self.setting_widget)
        self.method_combo = ComboBox(self.setting_widget)
        self.method_combo.addItems(["Grid", "Sobol"])
        self.method_label.setToolTip("Grid step scan or Sobol low-discrepancy sampling on simplex")
        self.method_label.installEventFilter(ToolTipFilter(self.method_label, 300, ToolTipPosition.TOP))

        self.step_label = BodyLabel("Step", self.setting_widget)
        self.step_frame = SpinBoxUnitInputFrame(self)
        self.step_frame.set_input("", 1, "float")
        self.step_frame.setDecimals(6)
        self.step_frame.setRange(1e-6, 1.0)
        self.step_frame.set_input_value([0.1])

        self.n_points_label = BodyLabel("N points", self.setting_widget)
        self.n_points_frame = SpinBoxUnitInputFrame(self)
        self.n_points_frame.set_input("unit", 1, "int")
        self.n_points_frame.setRange(1, 999999)
        self.n_points_frame.set_input_value([50])

        self.minfrac_label = BodyLabel("Min fraction", self.setting_widget)
        self.minfrac_frame = SpinBoxUnitInputFrame(self)
        self.minfrac_frame.set_input("", 1, "float")
        self.minfrac_frame.setDecimals(6)
        self.minfrac_frame.setRange(0.0, 1.0)
        self.minfrac_frame.set_input_value([0.0])

        self.include_endpoints_checkbox = CheckBox("Include endpoints", self.setting_widget)
        self.include_endpoints_checkbox.setChecked(True)

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.max_output_label = BodyLabel("Max outputs/frame", self.setting_widget)
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("unit", 1, "int")
        self.max_output_frame.setRange(1, 9999999)
        self.max_output_frame.set_input_value([500])

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

    def _target_orders(self) -> list[int]:
        text = (self.order_combo.currentText() or "").strip()
        legacy_map = {
            "Binary": [2],
            "Ternary": [3],
            "Quaternary": [4],
            "Quinary": [5],
            "Quaternary+Quinary": [4, 5],
            "Binary+Ternary+Quaternary+Quinary": [2, 3, 4, 5],
        }
        if text in legacy_map:
            return legacy_map[text]

        orders: list[int] = []
        for tok in text.replace(" ", "").split(","):
            if not tok:
                continue
            try:
                val = int(tok)
            except ValueError:
                continue
            if val not in (2, 3, 4, 5):
                continue
            if val not in orders:
                orders.append(val)
        return orders or [2, 3]

    def _simplex_points(self, order: int) -> list[tuple[float, ...]]:
        method = self.method_combo.currentText()
        min_fraction = float(self.minfrac_frame.get_input_value()[0])
        seed = int(self.seed_frame.get_input_value()[0]) if self.seed_checkbox.isChecked() else None
        if method == "Sobol":
            n_points = int(self.n_points_frame.get_input_value()[0])
            return simplex_sobol_points(order, n_points, seed=seed, min_fraction=min_fraction)
        step = float(self.step_frame.get_input_value()[0])
        include_endpoints = self.include_endpoints_checkbox.isChecked()
        try:
            pts = simplex_grid_points(order, step, include_endpoints=include_endpoints, min_fraction=min_fraction)
        except NotImplementedError as exc:
            MessageManager.send_warning_message(f"CompositionSweep Grid: {exc} (use Sobol for order>=4)")
            pts = []
        if seed is not None and pts:
            rng = np.random.default_rng(seed)
            rng.shuffle(pts)
        return pts

    def process_structure(self, structure):
        elements = parse_element_list(self.elements_edit.text())
        if len(elements) < 2:
            MessageManager.send_warning_message("CompositionSweep: need at least 2 elements.")
            return [structure]

        orders = self._target_orders()
        max_outputs = int(self.max_output_frame.get_input_value()[0])
        if max_outputs <= 0:
            return [structure]

        # Filter orders that are feasible given current element list.
        orders = [o for o in orders if len(elements) >= o]
        if not orders:
            MessageManager.send_warning_message("CompositionSweep: not enough elements for requested order.")
            return [structure]

        # Allocate output budget across orders.
        budgets: dict[int, int] = {}
        base = max_outputs // len(orders)
        rem = max_outputs - base * len(orders)
        for i, o in enumerate(orders):
            budgets[o] = base + (1 if i < rem else 0)
        # Ensure at least 1 sample for the highest-priority order when possible.
        if budgets.get(orders[0], 0) == 0 and max_outputs > 0:
            budgets[orders[0]] = 1

        out = []
        sweep_index = 0
        seed = int(self.seed_frame.get_input_value()[0]) if self.seed_checkbox.isChecked() else None
        combo_rng = np.random.default_rng(seed) if seed is not None else None

        for order in orders:
            points = self._simplex_points(order)
            order_budget = int(budgets.get(order, 0))
            if order_budget <= 0:
                continue
            if not points:
                continue
            combos = list(combinations(elements, order))
            if combo_rng is not None and combos:
                combo_rng.shuffle(combos)

            unique_total = len(combos) * len(points)
            n_emit = min(order_budget, unique_total)
            for k in range(n_emit):
                combo_idx = k % len(combos)
                point_idx = k // len(combos)
                elems = combos[combo_idx]
                frac = points[point_idx]
                comp = {e: float(f) for e, f in zip(elems, frac)}
                new_structure = structure.copy()
                sweep_index += 1

                tag = ",".join(f"{e}={comp[e]:.4g}" for e in elems)
                append_config_tag(new_structure, f"Comp({tag})")
                out.append(new_structure)
                if len(out) >= max_outputs:
                    return out
        return out or [structure]

    def to_dict(self):
        data = super().to_dict()
        data["elements"] = self.elements_edit.text()
        data["order"] = self.order_combo.currentText()
        data["method"] = self.method_combo.currentText()
        data["step"] = self.step_frame.get_input_value()
        data["n_points"] = self.n_points_frame.get_input_value()
        data["min_fraction"] = self.minfrac_frame.get_input_value()
        data["include_endpoints"] = self.include_endpoints_checkbox.isChecked()
        data["use_seed"] = self.seed_checkbox.isChecked()
        data["seed"] = self.seed_frame.get_input_value()
        data["max_outputs"] = self.max_output_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.elements_edit.setText(data_dict.get("elements", "Co,Cr,Ni"))
        self.order_combo.setCurrentText(data_dict.get("order", "2,3,4,5"))
        self.method_combo.setCurrentText(data_dict.get("method", "Grid"))
        self.step_frame.set_input_value(data_dict.get("step", [0.1]))
        self.n_points_frame.set_input_value(data_dict.get("n_points", [50]))
        self.minfrac_frame.set_input_value(data_dict.get("min_fraction", [0.0]))
        self.include_endpoints_checkbox.setChecked(bool(data_dict.get("include_endpoints", True)))
        self.seed_checkbox.setChecked(bool(data_dict.get("use_seed", False)))
        self.seed_frame.set_input_value(data_dict.get("seed", [0]))
        self.max_output_frame.set_input_value(data_dict.get("max_outputs", [500]))
