"""Card for generating crystal prototype structures (fcc/bcc/hcp)."""

from __future__ import annotations

import json

import numpy as np
from ase.build import bulk, make_supercell
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, RadioButton

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.alloy import best_supercell_factors_max_atoms
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CrystalPrototypeBuilderCard(MakeDataCard):
    """Generate simple bulk crystal prototypes without requiring input structures."""

    group = "Lattice"
    card_name = "Crystal Prototype Builder"
    menu_icon = r":/images/src/images/supercell.svg"

    requires_input_dataset = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Crystal Prototype Builder")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("crystal_prototype_builder_card_widget")

        self.structure_label = BodyLabel("Lattice", self.setting_widget)
        self.structure_combo = ComboBox(self.setting_widget)
        self.structure_combo.addItems(["fcc", "bcc", "hcp"])
        self.structure_label.setToolTip("Crystal structure prototype")
        self.structure_label.installEventFilter(ToolTipFilter(self.structure_label, 300, ToolTipPosition.TOP))

        self.element_label = BodyLabel("Base element", self.setting_widget)
        self.element_edit = LineEdit(self.setting_widget)
        self.element_edit.setPlaceholderText("Cu")
        self.element_edit.setText("Cu")
        self.element_label.setToolTip("Temporary element used to build the lattice sites")
        self.element_label.installEventFilter(ToolTipFilter(self.element_label, 300, ToolTipPosition.TOP))

        self.a_label = BodyLabel("a (Å)", self.setting_widget)
        self.a_frame = SpinBoxUnitInputFrame(self)
        self.a_frame.set_input(["-", "step", "Å"], 3, "float")
        self.a_frame.setDecimals(6)
        self.a_frame.setRange(0.1, 100.0)
        self.a_frame.set_input_value([3.6, 3.6, 0.1])
        self.a_label.setToolTip("Lattice parameter a range [min, max, step]")
        self.a_label.installEventFilter(ToolTipFilter(self.a_label, 300, ToolTipPosition.TOP))

        self.covera_label = BodyLabel("c/a", self.setting_widget)
        self.covera_frame = SpinBoxUnitInputFrame(self)
        self.covera_frame.set_input("", 1, "float")
        self.covera_frame.setDecimals(6)
        self.covera_frame.setRange(1.0, 5.0)
        self.covera_frame.set_input_value([1.633])
        self.covera_label.setToolTip("hcp only: c/a ratio (ideal ~1.633)")
        self.covera_label.installEventFilter(ToolTipFilter(self.covera_label, 300, ToolTipPosition.TOP))

        self.auto_supercell_button = RadioButton("Auto supercell (max atoms)", self.setting_widget)
        self.auto_supercell_button.setChecked(True)
        self.manual_supercell_button = RadioButton("Manual supercell", self.setting_widget)

        self.max_atoms_label = BodyLabel("Max atoms", self.setting_widget)
        self.max_atoms_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_frame.set_input("unit", 1, "int")
        self.max_atoms_frame.setRange(1, 500000)
        self.max_atoms_frame.set_input_value([512])

        self.rep_label = BodyLabel("Rep (na,nb,nc)", self.setting_widget)
        self.rep_frame = SpinBoxUnitInputFrame(self)
        self.rep_frame.set_input("", 3, "int")
        self.rep_frame.setRange(1, 999)
        self.rep_frame.set_input_value([4, 4, 4])

        self.max_output_label = BodyLabel("Max outputs", self.setting_widget)
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("unit", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([200])

        self.settingLayout.addWidget(self.structure_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.structure_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.element_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.element_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.a_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.a_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.covera_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.covera_frame, 3, 1, 1, 2)

        self.settingLayout.addWidget(self.auto_supercell_button, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_label, 4, 1, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_frame, 4, 2, 1, 1)
        self.settingLayout.addWidget(self.manual_supercell_button, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.rep_label, 5, 1, 1, 1)
        self.settingLayout.addWidget(self.rep_frame, 5, 2, 1, 1)
        self.settingLayout.addWidget(self.max_output_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 6, 1, 1, 2)

    def _a_values(self) -> list[float]:
        a_min, a_max, a_step = [float(v) for v in self.a_frame.get_input_value()]
        if a_step <= 0:
            return [a_min]
        if a_max < a_min:
            a_min, a_max = a_max, a_min
        if abs(a_max - a_min) <= 1e-12:
            return [a_min]
        values = list(np.arange(a_min, a_max + 1e-12, a_step, dtype=float))
        if not values:
            values = [a_min]
        return [float(v) for v in values]

    def _build_prototypes(self) -> list:
        element = self.element_edit.text().strip()
        if not element:
            element = "Cu"
        element = element[0].upper() + element[1:].lower()

        lattice = self.structure_combo.currentText().strip().lower()
        covera = float(self.covera_frame.get_input_value()[0])

        max_outputs = int(self.max_output_frame.get_input_value()[0])
        if max_outputs <= 0:
            return []

        out = []
        for a in self._a_values():
            try:
                if lattice == "hcp":
                    base = bulk(element, "hcp", a=float(a), covera=covera)
                else:
                    base = bulk(element, lattice, a=float(a), cubic=True)
                base.pbc = True
                base.wrap()

                if self.manual_supercell_button.isChecked():
                    na, nb, nc = [int(v) for v in self.rep_frame.get_input_value()]
                else:
                    factors = best_supercell_factors_max_atoms(base, int(self.max_atoms_frame.get_input_value()[0]))
                    na, nb, nc = factors.na, factors.nb, factors.nc

                mat = np.diag([max(na, 1), max(nb, 1), max(nc, 1)])
                atoms = make_supercell(base, mat)
                atoms.wrap()
                append_config_tag(atoms, f"Proto({lattice},a={float(a):.6g},rep={int(na)}x{int(nb)}x{int(nc)})")
                out.append(atoms)
                if len(out) >= max_outputs:
                    break
            except Exception as exc:  # noqa: BLE001
                MessageManager.send_warning_message(f"Prototype build failed: {exc}")
                continue
        return out

    def process_structure(self, structure):  # noqa: ARG002
        return self._build_prototypes()

    def run(self):
        """Override to support generation without iterating an input dataset."""
        if self.check_state:
            if self.dataset is None:
                self.dataset = []
            self.result_dataset = self._build_prototypes()
            self.update_dataset_info()
            self.status_label.setText(f"Generated: {len(self.result_dataset)}")
            self.runFinishedSignal.emit(self.index)
        else:
            self.result_dataset = self.dataset or []
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)

    def to_dict(self):
        data = super().to_dict()
        data["lattice"] = self.structure_combo.currentText()
        data["element"] = self.element_edit.text()
        data["a_range"] = self.a_frame.get_input_value()
        data["covera"] = self.covera_frame.get_input_value()
        data["auto_supercell"] = self.auto_supercell_button.isChecked()
        data["max_atoms"] = self.max_atoms_frame.get_input_value()
        data["rep"] = self.rep_frame.get_input_value()
        data["max_outputs"] = self.max_output_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.structure_combo.setCurrentText(data_dict.get("lattice", "fcc"))
        self.element_edit.setText(data_dict.get("element", "Cu"))
        self.a_frame.set_input_value(data_dict.get("a_range", [3.6, 3.6, 0.1]))
        self.covera_frame.set_input_value(data_dict.get("covera", [1.633]))
        auto = bool(data_dict.get("auto_supercell", True))
        self.auto_supercell_button.setChecked(auto)
        self.manual_supercell_button.setChecked(not auto)
        self.max_atoms_frame.set_input_value(data_dict.get("max_atoms", [512]))
        self.rep_frame.set_input_value(data_dict.get("rep", [4, 4, 4]))
        self.max_output_frame.set_input_value(data_dict.get("max_outputs", [200]))
