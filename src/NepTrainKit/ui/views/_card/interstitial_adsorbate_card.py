"""Card for inserting interstitial and adsorbate species into structures."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
from loguru import logger
from ase import Atom
from ase.geometry import geometry
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


def _parse_species(tokens: str, fallback: Sequence[str]) -> Tuple[List[str], List[float]]:
    """Parse ``Element[:weight]`` tokens into species and weights."""
    species: list[str] = []
    weights: list[float] = []
    for token in tokens.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" in item:
            symbol, weight_str = item.split(":", 1)
            symbol = symbol.strip()
            try:
                weight = float(weight_str.strip())
            except ValueError:
                weight = 1.0
        else:
            symbol = item
            weight = 1.0
        if symbol:
            species.append(symbol)
            weights.append(weight)
    if not species:
        default = list(dict.fromkeys(fallback))
        species = default or ["H"]
        weights = [1.0] * len(species)
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(species)] * len(species)
    else:
        weights = [w / total for w in weights]
    return species, weights


@CardManager.register_card
class InsertDefectCard(MakeDataCard):
    """Create interstitial or surface-adsorbate configurations."""

    group = "Defect"
    card_name = "Insert Defect"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Insert Defect")
        self._init_ui()

    def _init_ui(self):
        """Build configuration widgets."""
        self.setObjectName("insert_defect_card_widget")

        row = 0
        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_label.setToolTip("Interstitial: insert inside bulk. Adsorption: place species above a surface.")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["Interstitial", "Adsorption"])
        self.mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        self.settingLayout.addWidget(self.mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        self.species_label = BodyLabel("Species (comma separated)", self.setting_widget)
        self.species_label.setToolTip("Insert element list, optionally with weights, e.g. 'Li, Na:2'")
        self.species_label.installEventFilter(ToolTipFilter(self.species_label, 300, ToolTipPosition.TOP))
        self.species_edit = LineEdit(self.setting_widget)
        self.settingLayout.addWidget(self.species_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.species_edit, row, 1, 1, 2)
        row += 1

        self.insert_count_label = BodyLabel("Atoms per structure", self.setting_widget)
        self.insert_count_label.setToolTip("Number of atoms to insert per generated structure")
        self.insert_count_label.installEventFilter(ToolTipFilter(self.insert_count_label, 300, ToolTipPosition.TOP))
        self.insert_count_frame = SpinBoxUnitInputFrame(self)
        self.insert_count_frame.set_input("unit", 1, "int")
        self.insert_count_frame.setRange(1, 20)
        self.insert_count_frame.set_input_value([1])
        self.settingLayout.addWidget(self.insert_count_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.insert_count_frame, row, 1, 1, 2)
        row += 1

        self.structures_label = BodyLabel("Structures to generate", self.setting_widget)
        self.structures_label.setToolTip("Number of augmented structures to create")
        self.structures_label.installEventFilter(ToolTipFilter(self.structures_label, 300, ToolTipPosition.TOP))
        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("unit", 1, "int")
        self.structures_frame.setRange(1, 1000)
        self.structures_frame.set_input_value([10])
        self.settingLayout.addWidget(self.structures_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.structures_frame, row, 1, 1, 2)
        row += 1

        self.min_distance_label = BodyLabel("Min distance (Å)", self.setting_widget)
        self.min_distance_label.setToolTip("Reject insertions closer than this distance to existing atoms")
        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))
        self.min_distance_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_frame.set_input("Å", 1, "float")
        self.min_distance_frame.setRange(0.0, 10.0)
        self.min_distance_frame.object_list[0].setDecimals(3)  # pyright: ignore
        self.min_distance_frame.set_input_value([1.4])
        self.settingLayout.addWidget(self.min_distance_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.min_distance_frame, row, 1, 1, 2)
        row += 1

        self.max_attempts_label = BodyLabel("Max attempts", self.setting_widget)
        self.max_attempts_label.setToolTip("Maximum random trials per inserted atom")
        self.max_attempts_label.installEventFilter(ToolTipFilter(self.max_attempts_label, 300, ToolTipPosition.TOP))
        self.max_attempts_frame = SpinBoxUnitInputFrame(self)
        self.max_attempts_frame.set_input("unit", 1, "int")
        self.max_attempts_frame.setRange(1, 1000)
        self.max_attempts_frame.set_input_value([200])
        self.settingLayout.addWidget(self.max_attempts_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.max_attempts_frame, row, 1, 1, 2)
        row += 1

        # Adsorption-specific controls
        self.axis_label = BodyLabel("Surface axis", self.setting_widget)
        self.axis_label.setToolTip("Crystal axis treated as surface normal for adsorption")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_combo = ComboBox(self.setting_widget)
        self.axis_combo.addItems(["a (x)", "b (y)", "c (z)"])
        self.settingLayout.addWidget(self.axis_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_combo, row, 1, 1, 2)
        row += 1

        self.offset_label = BodyLabel("Offset distance (Å)", self.setting_widget)
        self.offset_label.setToolTip("Height above the surface plane when placing adsorbates")
        self.offset_label.installEventFilter(ToolTipFilter(self.offset_label, 300, ToolTipPosition.TOP))
        self.offset_frame = SpinBoxUnitInputFrame(self)
        self.offset_frame.set_input("Å", 1, "float")
        self.offset_frame.setRange(0.0, 10.0)
        self.offset_frame.object_list[0].setDecimals(3)  # pyright: ignore
        self.offset_frame.set_input_value([1.5])
        self.settingLayout.addWidget(self.offset_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.offset_frame, row, 1, 1, 2)
        row += 1

        self.adsorption_controls = [
            self.axis_label,
            self.axis_combo,
            self.offset_label,
            self.offset_frame,
        ]
        self._update_mode_visibility(self.mode_combo.currentIndex())

    def _update_mode_visibility(self, mode: int):
        is_adsorption = mode == 1
        for widget in self.adsorption_controls:
            widget.setVisible(is_adsorption)

    def process_structure(self, structure):
        """Insert atoms according to the current configuration."""
        mode = self.mode_combo.currentIndex()
        count = int(self.insert_count_frame.get_input_value()[0])
        max_structs = int(self.structures_frame.get_input_value()[0])
        min_distance = float(self.min_distance_frame.get_input_value()[0])
        max_attempts = int(self.max_attempts_frame.get_input_value()[0])

        species_tokens = self.species_edit.text()
        species, weights = _parse_species(species_tokens, structure.get_chemical_symbols())

        axis = self.axis_combo.currentIndex()
        offset = float(self.offset_frame.get_input_value()[0])

        base_positions = structure.get_positions()
        cell = structure.cell.array
        pbc = structure.get_pbc()

        results = []
        for _ in range(max_structs):
            new_structure = structure.copy()
            positions_reference = np.array(base_positions, dtype=float)
            inserted = 0
            inserted_species: list[str] = []

            for _ in range(count):
                success = False
                for attempt in range(max_attempts):
                    if mode == 0:
                        candidate = self._sample_interstitial(positions_reference, cell, pbc)
                    else:
                        candidate = self._sample_adsorbate(structure, positions_reference, cell, axis, offset)

                    if candidate is None:
                        continue

                    _, dists = geometry.get_distances(candidate, positions_reference, cell=cell, pbc=pbc)
                    if len(dists.ravel()) and np.min(dists) < max(min_distance, 0.0):
                        continue

                    element = random.choices(species, weights=weights, k=1)[0]
                    new_structure.append(Atom(element, candidate))
                    positions_reference = np.vstack([positions_reference, candidate])
                    inserted += 1
                    inserted_species.append(element)
                    success = True
                    break

                if not success:
                    logger.warning(
                        f'InsertDefectCard: failed to place atom after {max_attempts} attempts (mode={"adsorption" if mode == 1 else "interstitial"})',

                    )
                    break

            if inserted:
                tag = new_structure.info.get("Config_type", "")
                mode_name = "adsorbate" if mode == 1 else "interstitial"
                species_summary = ",".join(inserted_species)
                tag += f" Insert({mode_name},count={inserted},species={species_summary})"
                new_structure.info["Config_type"] = tag
            new_structure.wrap()
            results.append(new_structure)
        return results

    def _sample_interstitial(self, positions: np.ndarray, cell: np.ndarray, pbc: Sequence[bool]) -> np.ndarray:
        """Sample a random position inside the unit cell."""
        frac = np.random.rand(3)
        candidate = frac @ cell
        return candidate

    def _sample_adsorbate(
        self,
        structure,
        positions: np.ndarray,
        cell: np.ndarray,
        axis: int,
        offset: float,
    ) -> np.ndarray | None:
        """Sample a position above the surface along the selected axis."""
        if positions.size == 0:
            return None

        scaled = structure.get_scaled_positions(wrap=False)
        top_frac = scaled[:, axis].max()

        frac = np.random.rand(3)
        frac[axis] = top_frac
        in_plane = frac @ cell

        axis_vec = cell[axis]
        norm = np.linalg.norm(axis_vec)
        if norm < 1e-8:
            return None
        direction = axis_vec / norm
        candidate = in_plane + direction * offset
        return candidate

    def to_dict(self):
        data = super().to_dict()
        data.update(
            {
                "mode": self.mode_combo.currentIndex(),
                "species": self.species_edit.text(),
                "insert_count": self.insert_count_frame.get_input_value(),
                "structure_count": self.structures_frame.get_input_value(),
                "min_distance": self.min_distance_frame.get_input_value(),
                "max_attempts": self.max_attempts_frame.get_input_value(),
                "axis": self.axis_combo.currentIndex(),
                "offset": self.offset_frame.get_input_value(),
            }
        )
        return data

    def from_dict(self, data):
        super().from_dict(data)
        self.mode_combo.setCurrentIndex(data.get("mode", 0))
        self.species_edit.setText(data.get("species", ""))
        self.insert_count_frame.set_input_value(data.get("insert_count", [1]))
        self.structures_frame.set_input_value(data.get("structure_count", [10]))
        self.min_distance_frame.set_input_value(data.get("min_distance", [1.4]))
        self.max_attempts_frame.set_input_value(data.get("max_attempts", [200]))
        self.axis_combo.setCurrentIndex(data.get("axis", 2))
        self.offset_frame.set_input_value(data.get("offset", [1.5]))
        self._update_mode_visibility(self.mode_combo.currentIndex())
