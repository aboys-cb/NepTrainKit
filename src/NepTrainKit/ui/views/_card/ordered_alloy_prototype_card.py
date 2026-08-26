"""Card for generating ordered-alloy base cells with explicit sublattices."""

from __future__ import annotations

from PySide6.QtCore import Qt
from qfluentwidgets import CaptionLabel, ComboBox, LineEdit

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.alloy import (
    OrderedAlloyPrototypeOperation,
    OrderedAlloyPrototypeParams,
)
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
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
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class OrderedAlloyPrototypeCard(MakeDataCard):
    """Generate A1/A2/A3/L12/B2/L10 base cells with sublattice labels."""

    group = "Alloy"
    card_name = "Ordered Alloy Prototype"
    menu_icon = r":/images/src/images/supercell.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]
    requires_input_dataset = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._current_prototype = ""
        self._covera_cache = {"A3/hcp": 1.633, "L10/AB": 1.0}
        self.setTitle(self.tr("Ordered Alloy Prototype"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("ordered_alloy_prototype_card_widget")

        self.prototype_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.prototype_combo,
            [
                ("A1/fcc", "A1 / FCC (single sublattice)"),
                ("A2/bcc", "A2 / BCC (single sublattice)"),
                ("A3/hcp", "A3 / HCP (single sublattice)"),
                ("L12/A3B", "L1₂ / A₃B"),
                ("B2/AB", "B2 / AB"),
                ("L10/AB", "L1₀ / AB"),
            ],
        )
        set_combo_value(self.prototype_combo, "L12/A3B")
        self.prototype_field = CompactField(
            self.tr("Ordered prototype"),
            self.prototype_combo,
            self.setting_widget,
            self.tr("Choose the fixed crystallographic sites and A/B sublattice topology."),
        )

        self.sublattice_hint_label = CaptionLabel("", self.setting_widget)
        self.sublattice_hint_label.setWordWrap(True)
        self.sublattice_hint_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.single_sublattice_tip = CaptionLabel("", self.setting_widget)
        self.single_sublattice_tip.setWordWrap(True)
        self.single_sublattice_tip.setStyleSheet("color:#8a6d20;")

        self.a_frame = SpinBoxUnitInputFrame(self)
        self.a_frame.set_input(["–", self.tr("step"), "Å"], 3, "float")
        self.a_frame.setDecimals(4)
        self.a_frame.setRange(0.1, 100.0)
        self.a_frame.set_input_value([3.6, 3.6, 0.1])
        self.a_field = CompactField(
            self.tr("Lattice constant a (min, max, step)"),
            self.a_frame,
            self.setting_widget,
            self.tr("Each sampled a value produces one unexpanded base cell."),
        )

        self.covera_frame = SpinBoxUnitInputFrame(self)
        self.covera_frame.set_input("", 1, "float")
        self.covera_frame.setDecimals(4)
        self.covera_frame.setRange(0.1, 10.0)
        self.covera_frame.set_input_value([1.0])
        self.covera_field = CompactField(
            self.tr("c/a ratio"),
            self.covera_frame,
            self.setting_widget,
            self.tr("Only A3/HCP and L1₀ use c/a; cubic prototypes keep c/a = 1."),
        )

        prototype_section = InspectorSection(
            self.tr("Prototype"),
            self.setting_widget,
            self.tr("The output is an ideal, fully periodic base cell with a per-atom sublattice array."),
        )
        prototype_grid = ResponsiveFormGrid(prototype_section, two_column_threshold=520)
        prototype_grid.add_field(self.prototype_field, span=2)
        prototype_grid.add_field(self.a_field)
        prototype_grid.add_field(self.covera_field)
        prototype_section.addWidget(prototype_grid)
        prototype_section.addWidget(self.sublattice_hint_label)
        prototype_section.addWidget(self.single_sublattice_tip)

        self.element_a_edit = LineEdit(self.setting_widget)
        self.element_a_edit.setText("X")
        self.element_a_edit.setPlaceholderText(self.tr("Element or X"))
        self.element_a_field = CompactField(
            self.tr("Sublattice A element"),
            self.element_a_edit,
            self.setting_widget,
            self.tr("Use a real element for direct ordered occupancy, or X as a placeholder for a later occupancy card."),
        )

        self.element_b_edit = LineEdit(self.setting_widget)
        self.element_b_edit.setText("X")
        self.element_b_edit.setPlaceholderText(self.tr("Element or X"))
        self.element_b_field = CompactField(
            self.tr("Sublattice B element"),
            self.element_b_edit,
            self.setting_widget,
            self.tr("Shown only for two-sublattice prototypes; A and B are crystallographic site identities."),
        )

        occupant_section = InspectorSection(
            self.tr("Sublattice occupants"),
            self.setting_widget,
            self.tr("X is an unresolved site placeholder and is not a trainable chemical element."),
        )
        occupant_grid = ResponsiveFormGrid(occupant_section, two_column_threshold=520)
        occupant_grid.add_field(self.element_a_field)
        occupant_grid.add_field(self.element_b_field)
        occupant_section.addWidget(occupant_grid)

        self.max_outputs_frame = SpinBoxUnitInputFrame(self)
        self.max_outputs_frame.set_input(self.tr("structures"), 1, "int")
        self.max_outputs_frame.setRange(1, 999999)
        self.max_outputs_frame.set_input_value([200])
        self.max_outputs_field = CompactField(
            self.tr("Maximum outputs"),
            self.max_outputs_frame,
            self.setting_widget,
            self.tr("If the a scan has more points, only the first values in ascending scan order are kept."),
        )
        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.max_outputs_field)
        output_section.addWidget(self.output_preview)

        self.next_step_tip = CaptionLabel("", self.setting_widget)
        self.next_step_tip.setWordWrap(True)
        self.next_step_tip.setStyleSheet("color:#4078a8; font-weight:600;")
        self.legacy_expansion_notice = CaptionLabel("", self.setting_widget)
        self.legacy_expansion_notice.setWordWrap(True)
        self.legacy_expansion_notice.setStyleSheet("color:#c56a00; font-weight:600;")
        self.legacy_expansion_notice.hide()
        next_step_section = InspectorSection(self.tr("Next step"), self.setting_widget)
        next_step_section.addWidget(self.next_step_tip)
        next_step_section.addWidget(self.legacy_expansion_notice)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(10)
        self.settingLayout.addWidget(prototype_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(occupant_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(next_step_section, 3, 0, 1, 3)

        self.prototype_combo.currentIndexChanged.connect(self._on_prototype_changed)
        self.element_a_edit.textChanged.connect(self._on_elements_changed)
        self.element_b_edit.textChanged.connect(self._on_elements_changed)
        for frame in (self.a_frame, self.covera_frame, self.max_outputs_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self._update_output_preview)
        self._on_prototype_changed()

    @staticmethod
    def _prototype_requirements(prototype: str) -> tuple[tuple[str, ...], str, bool]:
        requirements = {
            "A1/fcc": (("A",), "A × 4", False),
            "A2/bcc": (("A",), "A × 2", False),
            "A3/hcp": (("A",), "A × 2", True),
            "L12/A3B": (("A", "B"), "A × 3, B × 1", False),
            "B2/AB": (("A", "B"), "A × 1, B × 1", False),
            "L10/AB": (("A", "B"), "A × 2, B × 2", True),
        }
        return requirements.get(prototype, requirements["L12/A3B"])

    @staticmethod
    def _element_mapping(text: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for token in str(text or "").split(","):
            if ":" not in token:
                continue
            label, element = token.split(":", 1)
            if label.strip() and element.strip():
                mapping[label.strip()] = element.strip()
        return mapping

    def _on_prototype_changed(self, *_args) -> None:
        if self._loading:
            return
        prototype = combo_value(self.prototype_combo, "L12/A3B")
        if self._current_prototype in self._covera_cache:
            self._covera_cache[self._current_prototype] = float(
                self.covera_frame.get_input_value()[0]
            )
        required, ratio, uses_covera = self._prototype_requirements(prototype)
        self._current_prototype = prototype
        self.element_b_field.setVisible("B" in required)
        self.covera_field.setVisible(uses_covera)
        if uses_covera:
            self.covera_frame.set_input_value([self._covera_cache.get(prototype, 1.0)])
        self.sublattice_hint_label.setText(
            self.tr("Base-cell sites: {ratio}. The sublattice array stores these sublattice identities.").format(
                ratio=ratio
            )
        )
        single = len(required) == 1
        self.single_sublattice_tip.setVisible(single)
        self.single_sublattice_tip.setText(
            self.tr(
                "A1/A2/A3 overlap geometrically with the Crystal Prototype Builder. "
                "Use them here only when downstream steps need sublattice A labels or X placeholders."
            )
            if single
            else ""
        )
        self._update_output_preview()

    def _on_elements_changed(self, *_args) -> None:
        self._update_output_preview()

    def _sublattice_text(self) -> str:
        required, _ratio, _uses_covera = self._prototype_requirements(
            combo_value(self.prototype_combo, "L12/A3B")
        )
        values = {"A": self.element_a_edit.text(), "B": self.element_b_edit.text()}
        return ",".join(f"{label}:{values[label]}" for label in required)

    def _update_output_preview(self, *_args) -> None:
        params = self.get_params()
        try:
            plan = self.create_operation().plan(params)
            shown = min(len(plan.a_values), params.max_outputs)
            la, lb, lc = plan.cell_lengths
            sites = ", ".join(
                f"{label}={count} ({plan.sublattice_elements[label]})"
                for label, count in plan.sublattice_counts.items()
            )
            text = self.tr(
                "{shown} base-cell output(s); {atoms} sites each: {sites}; "
                "first cell lengths {la:.3f} × {lb:.3f} × {lc:.3f} Å."
            ).format(
                shown=shown,
                atoms=plan.atoms_per_output,
                sites=sites,
                la=la,
                lb=lb,
                lc=lc,
            )
            if plan.truncated:
                text += " " + self.tr(
                    "The scan has {total} points; later a values are truncated."
                ).format(total=len(plan.a_values))
            self.output_preview.setText(text)
            has_placeholder = "X" in plan.sublattice_elements.values()
            self.next_step_tip.setText(
                self.tr(
                    "X placeholders are not ready for training. Add Super Cell if a larger cell is needed, "
                    "then use Finite-Cell Alloy Occupancy to assign real elements."
                )
                if has_placeholder
                else self.tr(
                    "Real elements produce a fixed-stoichiometry ordered base cell. Add Super Cell next when a larger cell is needed."
                )
            )
        except ValueError as exc:
            self.output_preview.setText(translate_runtime_message(exc))
            self.next_step_tip.setText(
                self.tr("Fix the highlighted parameter meaning before continuing downstream.")
            )
        self.refresh_compact_presentation()

    def create_operation(self):
        return OrderedAlloyPrototypeOperation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        try:
            plan = self.create_operation().plan(params)
            occupants = "/".join(plan.sublattice_elements.values())
            count = min(len(plan.a_values), params.max_outputs)
            return self.tr("{prototype} · {occupants} · {count} base-cell output(s)").format(
                prototype=self.prototype_combo.currentText(),
                occupants=occupants,
                count=count,
            )
        except ValueError:
            return self.tr("{prototype} · parameters need attention").format(
                prototype=self.prototype_combo.currentText()
            )

    def get_guidance_text(self) -> str:
        return self.tr(
            "This card defines crystallographic A/B site identities. It does not expand the cell; "
            "use Super Cell afterward, and replace every X before training."
        )

    def get_params(self) -> OrderedAlloyPrototypeParams:
        prototype = combo_value(self.prototype_combo, "L12/A3B")
        _required, _ratio, uses_covera = self._prototype_requirements(prototype)
        return OrderedAlloyPrototypeParams(
            prototype=prototype,
            a_range=tuple(float(value) for value in self.a_frame.get_input_value()),
            covera=float(self.covera_frame.get_input_value()[0]) if uses_covera else 1.0,
            sublattice_elements=self._sublattice_text(),
            max_outputs=int(self.max_outputs_frame.get_input_value()[0]),
        )

    def set_params(self, params: OrderedAlloyPrototypeParams) -> None:
        self._loading = True
        try:
            set_combo_value(self.prototype_combo, params.prototype)
            self.a_frame.set_input_value([float(value) for value in params.a_range])
            self.covera_frame.set_input_value([float(params.covera)])
            mapping = self._element_mapping(params.sublattice_elements)
            self.element_a_edit.setText(mapping.get("A", "X"))
            self.element_b_edit.setText(mapping.get("B", "X"))
            self.max_outputs_frame.set_input_value([int(params.max_outputs)])
            if params.prototype in self._covera_cache:
                self._covera_cache[params.prototype] = float(params.covera)
        finally:
            self._loading = False
        self._current_prototype = ""
        self._on_prototype_changed()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = dict(data_dict.get("params") or {})
        legacy_expansion = any(key in raw for key in ("auto_supercell", "max_atoms", "rep"))
        for key in ("auto_supercell", "max_atoms", "rep"):
            raw.pop(key, None)
        if raw:
            raw["a_range"] = tuple(raw.get("a_range", [3.6, 3.6, 0.1]))
            params = OrderedAlloyPrototypeParams(**raw)
        else:
            params = OrderedAlloyPrototypeParams()
        self.set_params(params)
        if legacy_expansion:
            migration_message = self.tr(
                "This saved Ordered Alloy Prototype used the removed expansion settings. "
                "They were ignored; add a Super Cell card after it to restore the intended cell size."
            )
            self.legacy_expansion_notice.setText("⚠ " + migration_message)
            self.legacy_expansion_notice.show()
            MessageManager.send_warning_message(migration_message)
