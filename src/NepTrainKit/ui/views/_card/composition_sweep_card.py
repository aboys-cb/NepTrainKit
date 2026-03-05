"""Card for expanding structures into binary/ternary composition batches."""

from __future__ import annotations

import math
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
        self._shown_plan_notice_key: tuple | None = None
        self._shown_warning_keys: set[tuple] = set()
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_sweep_card_widget")

        self.elements_label = BodyLabel("Elements", self.setting_widget)
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText("Co,Cr,Ni,Al")
        self.elements_edit.setText("Co,Cr,Ni")
        self.elements_label.setToolTip("Candidate elements for binary/ternary/quaternary/quinary combinations")
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

        self.budget_mode_label = BodyLabel("Budget mode", self.setting_widget)
        self.budget_mode_combo = ComboBox(self.setting_widget)
        self.budget_mode_combo.addItems(
            [
                "Equal+Reflow",
                "Capacity-weighted",
                "Equal (legacy)",
            ]
        )
        self.budget_mode_combo.setCurrentText("Equal+Reflow")
        self.budget_mode_label.setToolTip("How Max outputs/frame is split across selected orders")
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
        for spin in self.step_frame.object_list + self.n_points_frame.object_list:
            spin.valueChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.minfrac_frame.object_list[0].valueChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.max_output_frame.object_list[0].valueChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.include_endpoints_checkbox.stateChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.seed_checkbox.stateChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.seed_frame.object_list[0].valueChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.budget_mode_combo.currentTextChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.order_combo.currentTextChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self.elements_edit.textChanged.connect(lambda *_args: self._invalidate_runtime_notices())
        self._update_method_widgets()

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

    def _invalidate_runtime_notices(self) -> None:
        self._shown_plan_notice_key = None
        self._shown_warning_keys.clear()

    def _warn_once(self, key: tuple, message: str) -> None:
        if key in self._shown_warning_keys:
            return
        self._shown_warning_keys.add(key)
        MessageManager.send_warning_message(message)

    def _update_method_widgets(self) -> None:
        is_sobol = self.method_combo.currentText() == "Sobol"
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

    @staticmethod
    def _is_near_rational_step(step: float, tol: float = 1e-9) -> tuple[bool, int]:
        if step <= 0:
            return False, 0
        n = int(round(1.0 / float(step)))
        if n <= 0:
            return False, 0
        return abs(float(step) - 1.0 / n) <= tol, n

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

    def _budget_mode(self) -> str:
        txt = (self.budget_mode_combo.currentText() or "").strip().lower()
        if "legacy" in txt:
            return "equal_legacy"
        if "weight" in txt:
            return "weighted_reflow"
        return "equal_reflow"

    @staticmethod
    def _allocate_equal(orders: list[int], max_outputs: int) -> dict[int, int]:
        budgets: dict[int, int] = {o: 0 for o in orders}
        if not orders or max_outputs <= 0:
            return budgets
        base = max_outputs // len(orders)
        rem = max_outputs - base * len(orders)
        for i, o in enumerate(orders):
            budgets[o] = base + (1 if i < rem else 0)
        return budgets

    @staticmethod
    def _allocate_weighted(
        orders: list[int],
        capacities: dict[int, int],
        max_outputs: int,
    ) -> dict[int, int]:
        budgets: dict[int, int] = {o: 0 for o in orders}
        if not orders or max_outputs <= 0:
            return budgets
        total_cap = int(sum(max(0, int(capacities.get(o, 0))) for o in orders))
        if total_cap <= 0:
            return budgets

        raw = [float(max_outputs) * float(max(0, int(capacities.get(o, 0)))) / float(total_cap) for o in orders]
        floors = [int(np.floor(v)) for v in raw]
        for o, v in zip(orders, floors):
            budgets[o] = int(v)
        remaining = int(max_outputs - sum(floors))
        if remaining > 0:
            frac_rank = sorted(
                range(len(orders)),
                key=lambda i: (raw[i] - floors[i], -i),
                reverse=True,
            )
            for i in frac_rank[:remaining]:
                budgets[orders[i]] += 1
        return budgets

    @staticmethod
    def _reflow_budget(
        orders: list[int],
        budget: dict[int, int],
        capacities: dict[int, int],
        max_outputs: int,
    ) -> dict[int, int]:
        emit = {o: int(min(max(0, int(budget.get(o, 0))), max(0, int(capacities.get(o, 0))))) for o in orders}
        remaining = int(max_outputs) - int(sum(emit.values()))
        if remaining <= 0:
            return emit

        active = [o for o in orders if int(capacities.get(o, 0)) > emit[o]]
        while remaining > 0 and active:
            n_active = len(active)
            share = max(remaining // n_active, 1)
            next_active: list[int] = []
            progressed = False
            for o in active:
                room = int(capacities.get(o, 0)) - emit[o]
                if room <= 0:
                    continue
                add = int(min(room, share))
                if add > 0:
                    emit[o] += add
                    remaining -= add
                    progressed = True
                if emit[o] < int(capacities.get(o, 0)):
                    next_active.append(o)
                if remaining <= 0:
                    break
            if not progressed:
                break
            active = next_active
        return emit

    @staticmethod
    def _coprime_stride(total: int, hint: int) -> int:
        total = int(total)
        if total <= 1:
            return 1
        stride = int(hint) % total
        if stride <= 0:
            stride = 1
        while math.gcd(stride, total) != 1:
            stride += 1
            if stride >= total:
                stride = 1
        return stride

    @classmethod
    def _spread_slots(cls, total: int, n_pick: int, *, seed: int | None = None) -> list[int]:
        """Pick ``n_pick`` slots from ``[0, total)`` with low front bias and no duplicates."""
        total = int(total)
        n_pick = int(n_pick)
        if total <= 0 or n_pick <= 0:
            return []
        if n_pick >= total:
            return list(range(total))
        if seed is None:
            start = 0
            stride_hint = int(total * 0.6180339887498949)
        else:
            rng = np.random.default_rng(int(seed))
            start = int(rng.integers(0, total))
            stride_hint = int(rng.integers(1, total))
        stride = cls._coprime_stride(total, stride_hint)
        return [int((start + i * stride) % total) for i in range(n_pick)]

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

        out = []
        sweep_index = 0
        seed = int(self.seed_frame.get_input_value()[0]) if self.seed_checkbox.isChecked() else None
        combo_rng = np.random.default_rng(seed) if seed is not None else None
        method = self.method_combo.currentText()
        if method == "Grid":
            high_orders = [o for o in orders if o >= 4]
            if high_orders:
                step = float(self.step_frame.get_input_value()[0])
                near, _ = self._is_near_rational_step(step)
                if not near:
                    warn_key = (
                        "grid-step-order4plus",
                        tuple(high_orders),
                        round(step, 12),
                    )
                    self._warn_once(
                        warn_key,
                        "CompositionSweep: Grid with order>=4 requires step ~= 1/n; "
                        f"step={step:.8g} cannot cover order {','.join(str(o) for o in high_orders)}. "
                        "Those orders will be skipped (use Sobol for robust high-order sampling).",
                    )

        order_data: list[dict] = []
        capacities: dict[int, int] = {}
        for order in orders:
            points = self._simplex_points(order)
            if not points:
                continue
            combos = list(combinations(elements, order))
            if combo_rng is not None and combos:
                combo_rng.shuffle(combos)
            unique_total = len(combos) * len(points)
            if unique_total <= 0:
                continue
            capacities[order] = int(unique_total)
            order_data.append(
                {
                    "order": order,
                    "points": points,
                    "combos": combos,
                    "capacity": int(unique_total),
                }
            )

        if not order_data:
            MessageManager.send_warning_message("CompositionSweep: no valid composition points can be generated.")
            return [structure]

        active_orders = [int(item["order"]) for item in order_data]
        mode = self._budget_mode()
        if mode == "weighted_reflow":
            budgets = self._allocate_weighted(active_orders, capacities, max_outputs)
        else:
            budgets = self._allocate_equal(active_orders, max_outputs)

        if budgets.get(active_orders[0], 0) == 0 and max_outputs > 0:
            budgets[active_orders[0]] = 1

        if mode == "equal_legacy":
            emit = {
                o: int(min(max(0, int(budgets.get(o, 0))), max(0, int(capacities.get(o, 0)))))
                for o in active_orders
            }
        else:
            emit = self._reflow_budget(active_orders, budgets, capacities, max_outputs)

        plan_key = (
            tuple(elements),
            tuple(active_orders),
            method,
            mode,
            round(float(self.step_frame.get_input_value()[0]), 12),
            int(self.n_points_frame.get_input_value()[0]),
            round(float(self.minfrac_frame.get_input_value()[0]), 12),
            bool(self.include_endpoints_checkbox.isChecked()),
            bool(self.seed_checkbox.isChecked()),
            int(self.seed_frame.get_input_value()[0]),
            int(max_outputs),
            tuple((o, int(emit.get(o, 0)), int(capacities.get(o, 0))) for o in active_orders),
        )
        if self._shown_plan_notice_key != plan_key:
            self._shown_plan_notice_key = plan_key
            total_emit = int(sum(emit.values()))
            details = ", ".join(f"o{o}:{int(emit[o])}/{int(capacities[o])}" for o in active_orders)
            suffix = ""
            if total_emit < int(max_outputs):
                suffix = " (limited by available unique combinations)"
            MessageManager.send_info_message(
                "CompositionSweep plan: "
                f"budget={mode}, target={int(max_outputs)}, emit={total_emit}. {details}{suffix}"
            )

        for item in order_data:
            order = int(item["order"])
            points = item["points"]
            combos = item["combos"]
            order_budget = int(emit.get(order, 0))
            if order_budget <= 0:
                continue
            unique_total = int(item["capacity"])
            n_emit = min(order_budget, unique_total)
            slot_seed = None if seed is None else int(seed + order * 104729)
            slots = self._spread_slots(unique_total, n_emit, seed=slot_seed)
            for slot in slots:
                combo_idx = int(slot % len(combos))
                point_idx = int(slot // len(combos))
                elems = combos[combo_idx]
                frac = points[point_idx]
                comp = {e: float(f) for e, f in zip(elems, frac)}
                new_structure = structure.copy()
                sweep_index += 1

                tag = ",".join(f"{e}={comp[e]:.4g}" for e in elems)
                # Card-generated workflow markers belong in Config_type only.
                # Do not persist card parameters into atoms.info (especially JSON blobs).
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
        data["budget_mode"] = self.budget_mode_combo.currentText()
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
        self.budget_mode_combo.setCurrentText(data_dict.get("budget_mode", "Equal+Reflow"))
        self._update_method_widgets()
        self._invalidate_runtime_notices()
