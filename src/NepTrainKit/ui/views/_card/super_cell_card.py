"""Card for building supercells using several expansion strategies."""

from itertools import combinations
from typing import List, Tuple

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from ase.build import make_supercell
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox, RadioButton

from NepTrainKit.core import CardManager

from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard
from scipy.stats.qmc import Sobol

@CardManager.register_card
class SuperCellCard(MakeDataCard):
    """Create supercells based on fixed scale factors, target lattice lengths, or atom limits.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Lattice"
    card_name= "Super Cell"
    menu_icon=r":/images/src/images/supercell.svg"
    separator = False
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Supercell")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("super_cell_card_widget")
        self.behavior_type_combo=ComboBox(self.setting_widget)
        self.behavior_type_combo.addItem("Maximum")
        self.behavior_type_combo.addItem("Iteration")

        self.combo_label=BodyLabel("Behavior:",self.setting_widget)
        self.combo_label.setToolTip("Select supercell generation method")
        self.combo_label.installEventFilter(ToolTipFilter(self.combo_label, 300, ToolTipPosition.TOP))

        self.super_scale_radio_button = RadioButton("Super scale",self.setting_widget)
        self.super_scale_radio_button.setChecked(True)
        self.super_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_scale_condition_frame.set_input("",3)
        self.super_scale_condition_frame.setRange(1,100)
        self.super_scale_condition_frame.set_input_value([3,3,3])
        self.super_scale_radio_button.setToolTip("Scale factors along axes")
        self.super_scale_radio_button.installEventFilter(ToolTipFilter(self.super_scale_radio_button, 300, ToolTipPosition.TOP))

        self.super_cell_radio_button = RadioButton("Super cell",self.setting_widget)
        self.super_cell_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_cell_condition_frame.set_input("Å",3)
        self.super_cell_condition_frame.setRange(1,100)
        self.super_cell_condition_frame.set_input_value([20,20,20])

        self.super_cell_radio_button.setToolTip("Target lattice constant in Å")
        self.super_cell_radio_button.installEventFilter(ToolTipFilter(self.super_cell_radio_button, 300, ToolTipPosition.TOP))


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)
        # self.max_atoms_condition_frame.setToolTip("Maximum allowed atoms")
        self.max_atoms_condition_frame.set_input_value([100])

        self.max_atoms_radio_button = RadioButton("Max atoms",self.setting_widget)
        self.max_atoms_radio_button.setToolTip("Limit cell size by atom count")
        self.max_atoms_radio_button.installEventFilter(ToolTipFilter(self.max_atoms_radio_button, 300, ToolTipPosition.TOP))


        self.settingLayout.addWidget(self.combo_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.behavior_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.super_scale_radio_button, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.super_scale_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.super_cell_radio_button, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.super_cell_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_radio_button, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 3, 1, 1, 2)

    def _get_scale_factors(self) -> List[Tuple[int, int, int]]:
        """Return the user-specified scale factors for direct supercell expansion.
        
        Returns
        -------
        list[tuple[int, int, int]]
            Scale factors along the a, b, and c axes.
        """
        na, nb, nc = self.super_scale_condition_frame.get_input_value()
        return [(int(na), int(nb), int(nc))]

    def _get_max_cell_factors(self, structure) -> List[Tuple[int, int, int]]:
        """Calculate expansion factors that keep lattice lengths below the configured maxima.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure whose cell metrics are analysed.
        
        Returns
        -------
        list[tuple[int, int, int]]
            Candidate scale factors that satisfy the length limits.
        """
        max_a, max_b, max_c = self.super_cell_condition_frame.get_input_value()
        lattice = structure.cell.array


        a_len = np.linalg.norm(lattice[0])
        b_len = np.linalg.norm(lattice[1])
        c_len = np.linalg.norm(lattice[2])


        na = max(int(max_a / a_len) if a_len > 0 else 0, 1)
        nb = max(int(max_b / b_len) if b_len > 0 else 0, 1)
        nc = max(int(max_c / c_len) if c_len > 0 else 0, 1)


        na = na - 1 if na * a_len > max_a else na
        nb = nb - 1 if nb * b_len > max_b else nb
        nc = nc - 1 if nc * c_len > max_c else nc

        return [(max(na, 1), max(nb, 1), max(nc, 1))]


    def _get_max_atoms_factors(self, structure) -> List[Tuple[int, int, int]]:
        """Enumerate expansion factors that keep the total atom count within the configured limit.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure providing the base atom count.
        
        Returns
        -------
        list[tuple[int, int, int]]
            Sorted list of scale factors ordered by resulting atom count.
        """
        max_atoms = self.max_atoms_condition_frame.get_input_value()[0]
        num_atoms_orig = len(structure)
        max_n = int(max_atoms / num_atoms_orig)
        max_n_a = max_n_b = max_n_c = max(max_n, 1)

        expansion_factors = []
        for na in range(1, max_n_a + 1):
            for nb in range(1, max_n_b + 1):
                for nc in range(1, max_n_c + 1):
                    total_atoms = num_atoms_orig * na * nb * nc
                    if total_atoms <= max_atoms:
                        expansion_factors.append((na, nb, nc))
                    else:
                        break

        expansion_factors.sort(key=lambda x: num_atoms_orig * x[0] * x[1] * x[2])
        if len(expansion_factors)==0:
            return [(1, 1, 1)]

        return expansion_factors

    def _generate_structures(self, structure, expansion_factors, super_cell_type) :
        """Build supercells using the supplied expansion factors and behaviour mode.
        
        Parameters
        ----------
        structure : ase.Atoms
            Base structure to expand.
        expansion_factors : list[tuple[int, int, int]]
            Candidate scale factors produced by helper methods.
        super_cell_type : int
            Behaviour index selected in the UI.
        
        Returns
        -------
        list[ase.Atoms]
            Generated supercell structures.
        """
        structure_list = []
        if super_cell_type == 0:
            na, nb, nc = expansion_factors[-1]

            if na == 1 and nb == 1 and nc == 1:


                return [structure.copy()]

            supercell = make_supercell(structure,np.diag([na, nb, nc]),order="atom-major")
            supercell.info["Config_type"] = structure.info.get("Config_type","") + f" supercell({na, nb, nc})"

            structure_list.append(supercell)

        elif super_cell_type == 1:
            if self.max_atoms_radio_button.isChecked():

                for na, nb, nc in expansion_factors:


                    if na==1 and nb==1 and nc==1:
                        supercell=structure.copy()


                    else:
                        supercell = make_supercell(structure, np.diag([na, nb, nc]),order="atom-major")
                        supercell.info["Config_type"] = structure.info.get("Config_type","") + f" supercell({na, nb, nc})"

                        # supercell = structure.supercell([na, nb, nc])
                    structure_list.append(supercell)
            else:

                na, nb, nc = expansion_factors[0]
                for i in range(1, na + 1):
                    for j in range(1, nb + 1):
                        for k in range(1, nc + 1):

                            if na == 1 and nb == 1 and nc == 1:
                                supercell = structure.copy()
                            else:
                                supercell = make_supercell(structure, np.diag([i, j, k]),order="atom-major")
                                supercell.info["Config_type"]=structure.info.get("Config_type","") +f" supercell({i,j,k})"
                                # supercell = structure.supercell((i, j, k))

                            structure_list.append(supercell)


        # super_cell_type == 2
        return structure_list

    def process_structure(self,structure):
        """Generate supercells according to the selected behaviour (maximum, iteration, or atom limit).
        
        Parameters
        ----------
        structure : ase.Atoms
            Base structure used to create supercells.
        
        Returns
        -------
        list[ase.Atoms]
            Supercell structures produced by the chosen expansion strategy.
        """
        super_cell_type = self.behavior_type_combo.currentIndex()

        if self.super_scale_radio_button.isChecked():
            expansion_factors = self._get_scale_factors()
        elif self.super_cell_radio_button.isChecked():
            expansion_factors = self._get_max_cell_factors(structure)
        elif self.max_atoms_radio_button.isChecked():
            expansion_factors = self._get_max_atoms_factors(structure)
        else:
            expansion_factors = [(1, 1, 1)]


        structure_list = self._generate_structures(structure, expansion_factors, super_cell_type)
        return structure_list

    def to_dict(self):
        """Serialize the current configuration to a plain dictionary.
        
        Returns
        -------
        dict
            Dictionary that can be fed into ``from_dict`` to rebuild the state.
        """
        data_dict = super().to_dict()


        data_dict['super_cell_type'] = self.behavior_type_combo.currentIndex()
        data_dict['super_scale_radio_button'] = self.super_scale_radio_button.isChecked()
        data_dict['super_scale_condition'] = self.super_scale_condition_frame.get_input_value()
        data_dict['super_cell_radio_button'] = self.super_cell_radio_button.isChecked()
        data_dict['super_cell_condition'] = self.super_cell_condition_frame.get_input_value()
        data_dict['max_atoms_radio_button'] = self.max_atoms_radio_button.isChecked()
        data_dict['max_atoms_condition'] = self.max_atoms_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)

        self.behavior_type_combo.setCurrentIndex(data_dict['super_cell_type'])
        self.super_scale_radio_button.setChecked(data_dict['super_scale_radio_button'])
        self.super_scale_condition_frame.set_input_value(data_dict['super_scale_condition'])
        self.super_cell_radio_button.setChecked(data_dict['super_cell_radio_button'])
        self.super_cell_condition_frame.set_input_value(data_dict['super_cell_condition'])
        self.max_atoms_radio_button.setChecked(data_dict['max_atoms_radio_button'])
        self.max_atoms_condition_frame.set_input_value(data_dict['max_atoms_condition'])
