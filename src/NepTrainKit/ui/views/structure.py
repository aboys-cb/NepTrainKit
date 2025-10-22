"""Widgets for displaying structural metadata inside the NEP UI."""

import numpy as np
from PySide6.QtWidgets import QWidget, QGridLayout, QSizePolicy

from qfluentwidgets import BodyLabel


class StructureInfoWidget(QWidget):
    """Display atomic counts, lattice metrics, and configuration metadata.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget used to embed the info panel.
    """

    def __init__(self, parent=None):
        """Initialize labels and layout placeholders."""
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        """Construct the labels that show structure metadata."""
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

        self.atom_label = BodyLabel(self)
        self.atom_label.setText("Atoms:")
        self.atom_num_text = BodyLabel(self)

        self.formula_label = BodyLabel(self)
        self.formula_label.setText("Formula:")
        self.formula_text = BodyLabel(self)
        self.formula_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.formula_text.setWordWrap(True)

        self.lattice_label = BodyLabel(self)
        self.lattice_label.setText("Lattice:")
        self.lattice_text = BodyLabel(self)
        self.lattice_text.setWordWrap(True)

        self.length_label = BodyLabel(self)
        self.length_label.setText("a b c:")
        self.length_text = BodyLabel(self)

        self.angle_label = BodyLabel(self)
        self.angle_label.setText("Angles:")
        self.angle_text = BodyLabel(self)

        self.config_label = BodyLabel(self)
        self.config_label.setText("Config_type:")
        self.config_text = BodyLabel(self)
        self.config_text.setMaximumWidth(400)
        self.config_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.config_text.setWordWrap(True)

        self._layout.addWidget(self.atom_label, 0, 0, 1, 1)
        self._layout.addWidget(self.atom_num_text, 0, 1, 1, 3)
        self._layout.addWidget(self.formula_label, 1, 0, 1, 1)
        self._layout.addWidget(self.formula_text, 1, 1, 1, 3)
        self._layout.addWidget(self.config_label, 2, 0, 1, 1)
        self._layout.addWidget(self.config_text, 2, 1, 1, 3)
        self._layout.addWidget(self.lattice_label, 3, 0, 1, 1)
        self._layout.addWidget(self.lattice_text, 3, 1, 1, 3)
        self._layout.addWidget(self.length_label, 4, 0, 1, 1)
        self._layout.addWidget(self.length_text, 4, 1, 1, 3)
        self._layout.addWidget(self.angle_label, 5, 0, 1, 1)
        self._layout.addWidget(self.angle_text, 5, 1, 1, 3)

    def show_structure_info(self, structure) -> None:
        """Populate the labels using the provided ASE structure.

        Parameters
        ----------
        structure : ase.Atoms
            Structure whose metadata will be rendered.
        """
        self.atom_num_text.setText(str(len(structure)))
        self.formula_text.setText(structure.html_formula)
        self.lattice_text.setText(str(np.round(structure.lattice, 3)))
        self.length_text.setText(" ".join(f"{x:.3f}" for x in structure.abc))
        self.angle_text.setText(" ".join(f"{x:.2f} °" for x in structure.angles))
        self.config_text.setText('\n'.join(structure.tag[i:i + 50] for i in range(0, len(structure.tag), 50)))
