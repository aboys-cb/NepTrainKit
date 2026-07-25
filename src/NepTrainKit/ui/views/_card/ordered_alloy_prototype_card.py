"""Card for generating ordered-alloy prototypes with explicit sublattices."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, ComboBox, LineEdit, RadioButton, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import OrderedAlloyPrototypeOperation, OrderedAlloyPrototypeParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class OrderedAlloyPrototypeCard(MakeDataCard):
    """Generate A1/A2/A3/L12/B2/L10 cells with crystallographic site labels."""

    group = "Alloy"
    card_name = "Ordered Alloy Prototype"
    menu_icon = r":/images/src/images/supercell.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]
    requires_input_dataset = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._current_required_labels: tuple[str, ...] = ()
        self._current_prototype = ""
        self._element_cache = {"A": "X", "B": "X"}
        self._covera_cache = {"A3/hcp": 1.633, "L10/AB": 1.0}
        self.setTitle(self.tr("Ordered Alloy Prototype"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("ordered_alloy_prototype_card_widget")

        self.prototype_label = BodyLabel(self.tr("Prototype"), self.setting_widget)
        self.prototype_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.prototype_combo,
            ["A1/fcc", "A2/bcc", "A3/hcp", "L12/A3B", "B2/AB", "L10/AB"],
        )
        set_combo_value(self.prototype_combo, "L12/A3B")
        self.prototype_label.setToolTip(self.tr("Ordered or elemental crystal prototype"))
        self.prototype_label.installEventFilter(ToolTipFilter(self.prototype_label, 300, ToolTipPosition.TOP))
        self.sublattice_hint_label = CaptionLabel("", self.setting_widget)
        self.sublattice_hint_label.setWordWrap(True)
        self.sublattice_hint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.a_label = BodyLabel(self.tr("a (Å)"), self.setting_widget)
        self.a_frame = SpinBoxUnitInputFrame(self)
        self.a_frame.set_input(["-", "step", "Å"], 3, "float")
        self.a_frame.setDecimals(6)
        self.a_frame.setRange(0.1, 100.0)
        self.a_frame.set_input_value([3.6, 3.6, 0.1])

        self.covera_label = BodyLabel(self.tr("c/a"), self.setting_widget)
        self.covera_frame = SpinBoxUnitInputFrame(self)
        self.covera_frame.set_input("", 1, "float")
        self.covera_frame.setDecimals(6)
        self.covera_frame.setRange(0.1, 10.0)
        self.covera_frame.set_input_value([1.0])
        self.covera_label.setToolTip(self.tr("Used by A3/hcp and L10/AB; cubic prototypes use 1"))
        self.covera_label.installEventFilter(ToolTipFilter(self.covera_label, 300, ToolTipPosition.TOP))

        self.elements_label = BodyLabel(self.tr("Sublattice elements"), self.setting_widget)
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setText("A:X,B:X")
        self.elements_edit.setPlaceholderText(self.tr("A:X,B:X (use X as a placeholder)"))
        self.elements_label.setToolTip(self.tr("Element or X placeholder assigned to each crystallographic sublattice"))
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))

        self.auto_supercell_button = RadioButton(self.tr("Auto supercell (max atoms)"), self.setting_widget)
        self.auto_supercell_button.setChecked(True)
        self.manual_supercell_button = RadioButton(self.tr("Manual supercell"), self.setting_widget)

        self.max_atoms_label = BodyLabel(self.tr("Max atoms"), self.setting_widget)
        self.max_atoms_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_frame.set_input("unit", 1, "int")
        self.max_atoms_frame.setRange(1, 500000)
        self.max_atoms_frame.set_input_value([128])

        self.rep_label = BodyLabel(self.tr("Rep (na,nb,nc)"), self.setting_widget)
        self.rep_frame = SpinBoxUnitInputFrame(self)
        self.rep_frame.set_input("", 3, "int")
        self.rep_frame.setRange(1, 999)
        self.rep_frame.set_input_value([2, 2, 2])

        self.max_outputs_label = BodyLabel(self.tr("Max outputs"), self.setting_widget)
        self.max_outputs_frame = SpinBoxUnitInputFrame(self)
        self.max_outputs_frame.set_input("unit", 1, "int")
        self.max_outputs_frame.setRange(1, 999999)
        self.max_outputs_frame.set_input_value([200])

        self.settingLayout.addWidget(self.prototype_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.prototype_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.sublattice_hint_label, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.a_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.a_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.covera_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.covera_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.elements_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_edit, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.auto_supercell_button, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_label, 5, 1, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_frame, 5, 2, 1, 1)
        self.settingLayout.addWidget(self.manual_supercell_button, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.rep_label, 6, 1, 1, 1)
        self.settingLayout.addWidget(self.rep_frame, 6, 2, 1, 1)
        self.settingLayout.addWidget(self.max_outputs_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.max_outputs_frame, 7, 1, 1, 2)

        self.prototype_combo.currentIndexChanged.connect(self._on_prototype_changed)
        self.auto_supercell_button.toggled.connect(self._update_supercell_controls)
        self.manual_supercell_button.toggled.connect(self._update_supercell_controls)
        self._on_prototype_changed()
        self._update_supercell_controls()
        self._update_tab_order()

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

    def _on_prototype_changed(self) -> None:
        if self._loading:
            return
        prototype = combo_value(self.prototype_combo)
        if self._current_prototype in self._covera_cache and self.covera_frame.isEnabled():
            self._covera_cache[self._current_prototype] = float(
                self.covera_frame.get_input_value()[0]
            )
        if self._current_required_labels:
            current = self._element_mapping(self.elements_edit.text())
            for label in self._current_required_labels:
                if label in current:
                    self._element_cache[label] = current[label]

        required, ratio, uses_covera = self._prototype_requirements(prototype)
        self._current_prototype = prototype
        self._current_required_labels = required
        self.elements_edit.setText(
            ",".join(f"{label}:{self._element_cache.get(label, 'X')}" for label in required)
        )
        self.sublattice_hint_label.setText(
            self.tr("Required sublattices: {labels}. Conventional-cell sites: {ratio}.").format(
                labels=", ".join(required),
                ratio=ratio,
            )
        )
        if uses_covera:
            self.covera_label.setText(self.tr("c/a"))
            self.covera_frame.setEnabled(True)
            self.covera_frame.set_input_value([self._covera_cache.get(prototype, 1.0)])
        else:
            self.covera_label.setText(self.tr("c/a (fixed at 1)"))
            self.covera_frame.set_input_value([1.0])
            self.covera_frame.setEnabled(False)
        self._update_tab_order()

    def _update_supercell_controls(self) -> None:
        auto = self.auto_supercell_button.isChecked()
        self.max_atoms_label.setEnabled(auto)
        self.max_atoms_frame.setEnabled(auto)
        self.rep_label.setEnabled(not auto)
        self.rep_frame.setEnabled(not auto)
        self._update_tab_order()

    def _update_tab_order(self) -> None:
        widgets = [
            self.prototype_combo,
            *self.a_frame.object_list,
            *self.covera_frame.object_list,
            self.elements_edit,
            self.auto_supercell_button,
            *self.max_atoms_frame.object_list,
            self.manual_supercell_button,
            *self.rep_frame.object_list,
            *self.max_outputs_frame.object_list,
        ]
        self.tab_order_widgets = [
            widget for widget in widgets if widget.isEnabled() and not widget.isHidden()
        ]
        for previous, current in zip(self.tab_order_widgets, self.tab_order_widgets[1:]):
            QWidget.setTabOrder(previous, current)

    def create_operation(self):
        return OrderedAlloyPrototypeOperation()

    def get_params(self) -> OrderedAlloyPrototypeParams:
        prototype = combo_value(self.prototype_combo)
        _, _, uses_covera = self._prototype_requirements(prototype)
        return OrderedAlloyPrototypeParams(
            prototype=prototype,
            a_range=tuple(float(value) for value in self.a_frame.get_input_value()),
            covera=float(self.covera_frame.get_input_value()[0]) if uses_covera else 1.0,
            sublattice_elements=self.elements_edit.text(),
            auto_supercell=self.auto_supercell_button.isChecked(),
            max_atoms=int(self.max_atoms_frame.get_input_value()[0]),
            rep=tuple(int(value) for value in self.rep_frame.get_input_value()),
            max_outputs=int(self.max_outputs_frame.get_input_value()[0]),
        )

    def set_params(self, params: OrderedAlloyPrototypeParams) -> None:
        self._loading = True
        try:
            set_combo_value(self.prototype_combo, params.prototype)
            self.a_frame.set_input_value([float(value) for value in params.a_range])
            self.covera_frame.set_input_value([float(params.covera)])
            mapping = self._element_mapping(params.sublattice_elements)
            self._element_cache.update(mapping)
            self.elements_edit.setText(params.sublattice_elements)
            self.auto_supercell_button.setChecked(bool(params.auto_supercell))
            self.manual_supercell_button.setChecked(not bool(params.auto_supercell))
            self.max_atoms_frame.set_input_value([int(params.max_atoms)])
            self.rep_frame.set_input_value([int(value) for value in params.rep])
            self.max_outputs_frame.set_input_value([int(params.max_outputs)])
        finally:
            self._loading = False
        self._current_required_labels = ()
        prototype = combo_value(self.prototype_combo)
        if prototype in self._covera_cache:
            self._covera_cache[prototype] = float(params.covera)
        self._on_prototype_changed()
        self._update_supercell_controls()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = dict(data_dict.get("params") or {})
        if raw:
            raw["a_range"] = tuple(raw.get("a_range", [3.6, 3.6, 0.1]))
            raw["rep"] = tuple(raw.get("rep", [2, 2, 2]))
            params = OrderedAlloyPrototypeParams(**raw)
        else:
            params = OrderedAlloyPrototypeParams()
        self.set_params(params)
