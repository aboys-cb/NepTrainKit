"""Card that wraps the torsion-guard PBC configurator for organic molecules."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.torsion_guard_pbc import (
    TorsionGuardParams,
    process_single as tg_process_single,
)
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard


@CardManager.register_card
class OrganicMolConfigPBCCard(MakeDataCard):
    """Create torsion-driven molecular configurations using the TorsionGuard PBC workflow.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the configuration card.
    """

    group = "Perturbation"
    card_name = "Organic Mol Config"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Organic Molecular Configuration(Zherui Chen)")
        self._init_ui()

    # ---------- UI ----------
    def _init_ui(self):
        """Create all of the widgets required to configure the torsion-guard workflow.
        """
        self.setObjectName("organic_mol_config_pbc_card")

        row = 0

        # perturb_per_frame
        self.perturb_label = BodyLabel("Confs per structure:", self.setting_widget)
        self.perturb_label.setToolTip("Number of perturbed conformations generated per input structure")
        self.perturb_label.installEventFilter(ToolTipFilter(self.perturb_label, 300, ToolTipPosition.TOP))
        self.perturb_frame = SpinBoxUnitInputFrame(self)
        self.perturb_frame.set_input("count", 1, "int")
        self.perturb_frame.setRange(1, 100000)
        self.perturb_frame.set_input_value([100])
        self.settingLayout.addWidget(self.perturb_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.perturb_frame, row, 1, 1, 2)
        row += 1

        # torsion_range_deg
        self.torsion_label = BodyLabel("Torsion range:", self.setting_widget)
        self.torsion_label.setToolTip("Torsion angle range (degrees)")
        self.torsion_label.installEventFilter(ToolTipFilter(self.torsion_label, 300, ToolTipPosition.TOP))
        self.torsion_frame = SpinBoxUnitInputFrame(self)
        self.torsion_frame.set_input(["°", "°"], 2, ["float", "float"])
        self.torsion_frame.setRange(-360, 360)
        self.torsion_frame.set_input_value([-180.0, 180.0])
        self.settingLayout.addWidget(self.torsion_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.torsion_frame, row, 1, 1, 2)
        row += 1

        # max_torsions_per_conf
        self.max_torsions_label = BodyLabel("Max torsions/conf:", self.setting_widget)
        self.max_torsions_label.setToolTip("Maximum number of torsions applied per conformation")
        self.max_torsions_label.installEventFilter(ToolTipFilter(self.max_torsions_label, 300, ToolTipPosition.TOP))
        self.max_torsions_frame = SpinBoxUnitInputFrame(self)
        self.max_torsions_frame.set_input("", 1, "int")
        self.max_torsions_frame.setRange(0, 10000)
        self.max_torsions_frame.set_input_value([50])
        self.settingLayout.addWidget(self.max_torsions_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.max_torsions_frame, row, 1, 1, 2)
        row += 1

        # gaussian_sigma
        self.sigma_label = BodyLabel("Gaussian sigma:", self.setting_widget)
        self.sigma_label.setToolTip("Std dev of added Gaussian noise (Å)")
        self.sigma_label.installEventFilter(ToolTipFilter(self.sigma_label, 300, ToolTipPosition.TOP))
        self.sigma_frame = SpinBoxUnitInputFrame(self)
        self.sigma_frame.set_input("Å", 1, "float")
        self.sigma_frame.setRange(0, 5)
        self.sigma_frame.set_input_value([0.03])
        self.settingLayout.addWidget(self.sigma_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.sigma_frame, row, 1, 1, 2)
        row += 1

        # pbc mode
        self.pbc_label = BodyLabel("PBC mode:", self.setting_widget)
        self.pbc_label.setToolTip("auto: use cell if present; yes: force PBC; no: non-PBC")
        self.pbc_label.installEventFilter(ToolTipFilter(self.pbc_label, 300, ToolTipPosition.TOP))
        self.pbc_combo = ComboBox(self.setting_widget)
        for opt in ("auto", "yes", "no"):
            self.pbc_combo.addItem(opt)
        self.pbc_combo.setCurrentIndex(0)
        self.settingLayout.addWidget(self.pbc_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.pbc_combo, row, 1, 1, 2)
        row += 1

        # local_mode_cutoff_atoms
        self.local_cut_label = BodyLabel("Local-mode cutoff atoms:", self.setting_widget)
        self.local_cut_label.setToolTip("Use local subtree rotations if atoms > this threshold")
        self.local_cut_label.installEventFilter(ToolTipFilter(self.local_cut_label, 300, ToolTipPosition.TOP))
        self.local_cut_frame = SpinBoxUnitInputFrame(self)
        self.local_cut_frame.set_input("atoms", 1, "int")
        self.local_cut_frame.setRange(0, 1000000)
        self.local_cut_frame.set_input_value([200])
        self.settingLayout.addWidget(self.local_cut_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.local_cut_frame, row, 1, 1, 2)
        row += 1

        # local_torsion_max_subtree
        self.local_sub_label = BodyLabel("Max subtree size:", self.setting_widget)
        self.local_sub_label.setToolTip("Maximum atoms in rotated subtree for local mode")
        self.local_sub_label.installEventFilter(ToolTipFilter(self.local_sub_label, 300, ToolTipPosition.TOP))
        self.local_sub_frame = SpinBoxUnitInputFrame(self)
        self.local_sub_frame.set_input("atoms", 1, "int")
        self.local_sub_frame.setRange(1, 100000)
        self.local_sub_frame.set_input_value([100])
        self.settingLayout.addWidget(self.local_sub_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.local_sub_frame, row, 1, 1, 2)
        row += 1

        # bond_detect_factor
        self.bond_detect_label = BodyLabel("Bond detect factor:", self.setting_widget)
        self.bond_detect_label.setToolTip("Bond detection cutoff multiplier (ri+rj)")
        self.bond_detect_label.installEventFilter(ToolTipFilter(self.bond_detect_label, 300, ToolTipPosition.TOP))
        self.bond_detect_frame = SpinBoxUnitInputFrame(self)
        self.bond_detect_frame.set_input("x", 1, "float")
        self.bond_detect_frame.setRange(0, 5)
        self.bond_detect_frame.set_input_value([1.15])
        self.settingLayout.addWidget(self.bond_detect_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_detect_frame, row, 1, 1, 2)
        row += 1

        # bond_keep_min_factor
        self.bond_min_label = BodyLabel("Bond min factor:", self.setting_widget)
        self.bond_min_label.setToolTip("Lower bound for bonded distances; 0 disables")
        self.bond_min_label.installEventFilter(ToolTipFilter(self.bond_min_label, 300, ToolTipPosition.TOP))
        self.bond_min_frame = SpinBoxUnitInputFrame(self)
        self.bond_min_frame.set_input("x", 1, "float")
        self.bond_min_frame.setRange(0, 5)
        self.bond_min_frame.set_input_value([0.60])
        self.settingLayout.addWidget(self.bond_min_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_min_frame, row, 1, 1, 2)
        row += 1

        # Pauling bond-order params
        self.bo_c_label = BodyLabel("Pauling c constant:", self.setting_widget)
        self.bo_c_label.setToolTip("Bond order constant c in exp((r0-r)/c)")
        self.bo_c_label.installEventFilter(ToolTipFilter(self.bo_c_label, 300, ToolTipPosition.TOP))
        self.bo_c_frame = SpinBoxUnitInputFrame(self)
        self.bo_c_frame.set_input("", 1, "float")
        self.bo_c_frame.setRange(0.01, 2.0)
        self.bo_c_frame.set_input_value([0.3])
        self.settingLayout.addWidget(self.bo_c_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bo_c_frame, row, 1, 1, 2)
        row += 1

        self.bo_thr_label = BodyLabel("BondOrder threshold:", self.setting_widget)
        self.bo_thr_label.setToolTip("Minimum bond order to form bond (default 0.2)")
        self.bo_thr_label.installEventFilter(ToolTipFilter(self.bo_thr_label, 300, ToolTipPosition.TOP))
        self.bo_thr_frame = SpinBoxUnitInputFrame(self)
        self.bo_thr_frame.set_input("", 1, "float")
        self.bo_thr_frame.setRange(0.0, 1.0)
        self.bo_thr_frame.set_input_value([0.2])
        self.settingLayout.addWidget(self.bo_thr_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bo_thr_frame, row, 1, 1, 2)
        row += 1

        # bond_keep_max_factor (optional)
        self.bond_max_label = BodyLabel("Bond max factor:", self.setting_widget)
        self.bond_max_label.setToolTip("Upper bound for bonded distances; uncheck to disable")
        self.bond_max_label.installEventFilter(ToolTipFilter(self.bond_max_label, 300, ToolTipPosition.TOP))
        self.bond_max_frame = SpinBoxUnitInputFrame(self)
        self.bond_max_frame.set_input("x", 1, "float")
        self.bond_max_frame.setRange(0, 5)
        self.bond_max_frame.set_input_value([1.15])
        self.bond_max_enable = CheckBox("Enable upper bound", self.setting_widget)
        self.bond_max_enable.setChecked(False)
        self.settingLayout.addWidget(self.bond_max_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_max_frame, row, 1, 1, 1)
        self.settingLayout.addWidget(self.bond_max_enable, row, 2, 1, 1)
        row += 1

        # nonbond_min_factor
        self.nonbond_min_label = BodyLabel("Non-bonded min factor:", self.setting_widget)
        self.nonbond_min_label.setToolTip("Minimum separation for non-bonded atoms (ri+rj) factor")
        self.nonbond_min_label.installEventFilter(ToolTipFilter(self.nonbond_min_label, 300, ToolTipPosition.TOP))
        self.nonbond_min_frame = SpinBoxUnitInputFrame(self)
        self.nonbond_min_frame.set_input("x", 1, "float")
        self.nonbond_min_frame.setRange(0, 5)
        self.nonbond_min_frame.set_input_value([0.80])
        self.settingLayout.addWidget(self.nonbond_min_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.nonbond_min_frame, row, 1, 1, 2)
        row += 1

        # max_retries_per_frame
        self.retries_label = BodyLabel("Max retries:", self.setting_widget)
        self.retries_label.setToolTip("Backoff retries per conformation if guards fail")
        self.retries_label.installEventFilter(ToolTipFilter(self.retries_label, 300, ToolTipPosition.TOP))
        self.retries_frame = SpinBoxUnitInputFrame(self)
        self.retries_frame.set_input("tries", 1, "int")
        self.retries_frame.setRange(0, 100)
        self.retries_frame.set_input_value([12])
        self.settingLayout.addWidget(self.retries_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.retries_frame, row, 1, 1, 2)
        row += 1

        # MULT_BOND_FACTOR
        self.multbond_label = BodyLabel("Multi-bond factor:", self.setting_widget)
        self.multbond_label.setToolTip("Exclude suspected multiple bonds if d < factor*(ri+rj)")
        self.multbond_label.installEventFilter(ToolTipFilter(self.multbond_label, 300, ToolTipPosition.TOP))
        self.multbond_frame = SpinBoxUnitInputFrame(self)
        self.multbond_frame.set_input("x", 1, "float")
        self.multbond_frame.setRange(0, 2)
        self.multbond_frame.set_input_value([0.87])
        self.settingLayout.addWidget(self.multbond_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.multbond_frame, row, 1, 1, 2)
        row += 1

        # nonpbc_box_size
        self.box_label = BodyLabel("Non-PBC box size:", self.setting_widget)
        self.box_label.setToolTip("Box edge for non-periodic output (Å)")
        self.box_label.installEventFilter(ToolTipFilter(self.box_label, 300, ToolTipPosition.TOP))
        self.box_frame = SpinBoxUnitInputFrame(self)
        self.box_frame.set_input("Å", 1, "float")
        self.box_frame.setRange(1, 100000)
        self.box_frame.set_input_value([100.0])
        self.settingLayout.addWidget(self.box_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.box_frame, row, 1, 1, 2)

    def _current_pbc_mode(self) -> str:
        """Return the currently selected periodic boundary mode.
        
        Returns
        -------
        str
            One of ``"auto"``, ``"yes"``, or ``"no"``.
        """
        return self.pbc_combo.currentText()

    # ---------- Core ----------
    def process_structure(self, structure) -> list[Any]:
        """Generate torsion-driven molecular conformers using the TorsionGuard PBC workflow.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure providing the initial molecular coordinates and cell.
        
        Returns
        -------
        list[ase.Atoms]
            Structures returned by the torsion-guard generator.
        """
        # structure is an ASE Atoms
        from ase import Atoms  # local import to avoid hard dep in module import time

        pbc_mode = self._current_pbc_mode()

        symbols: List[str] = structure.get_chemical_symbols()
        coords: np.ndarray = structure.get_positions().astype(float)

        # Decide cell: always pass a 3x3 even for non-PBC if available; module decides using pbc_mode
        cell_mat = None
        try:
            cell_arr = structure.get_cell().array  # type: ignore[attr-defined]
            if cell_arr is not None and np.array(cell_arr).shape == (3, 3):
                cell_mat = np.array(cell_arr, dtype=float)
        except Exception:
            cell_mat = None

        params = TorsionGuardParams(
            perturb_per_frame=int(self.perturb_frame.get_input_value()[0]),
            torsion_range_deg=tuple(map(float, self.torsion_frame.get_input_value())),
            max_torsions_per_conf=int(self.max_torsions_frame.get_input_value()[0]),
            gaussian_sigma=float(self.sigma_frame.get_input_value()[0]),
            pbc_mode=pbc_mode,
            local_mode_cutoff_atoms=int(self.local_cut_frame.get_input_value()[0]),
            local_torsion_max_subtree=int(self.local_sub_frame.get_input_value()[0]),
            bond_detect_factor=float(self.bond_detect_frame.get_input_value()[0]),
            bond_keep_min_factor=float(self.bond_min_frame.get_input_value()[0]),
            bond_keep_max_factor=float(self.bond_max_frame.get_input_value()[0]) if self.bond_max_enable.isChecked() else None,
            nonbond_min_factor=float(self.nonbond_min_frame.get_input_value()[0]),
            max_retries_per_frame=int(self.retries_frame.get_input_value()[0]),
            mult_bond_factor=float(self.multbond_frame.get_input_value()[0]),
            nonpbc_box_size=float(self.box_frame.get_input_value()[0]),
            bo_c_const=float(self.bo_c_frame.get_input_value()[0]),
            bo_threshold=float(self.bo_thr_frame.get_input_value()[0]),
        )

        result_list = tg_process_single(symbols, coords, cell_mat, params)

        structures_out: list[Atoms] = []
        for sym, new_coords, cell, pbc_active in result_list:
            new_atoms: Atoms = structure.copy()
            new_atoms.set_positions(np.array(new_coords, dtype=float))

            # Set periodicity and cell
            if pbc_active and cell is not None:
                new_atoms.set_cell(np.array(cell, dtype=float))
                new_atoms.set_pbc(True)
                try:
                    new_atoms.wrap()
                except Exception:
                    pass
            else:
                # Centered in a non-PBC box already by module; attach a large box for visualization
                L = float(self.box_frame.get_input_value()[0])
                new_atoms.set_cell(np.diag([L, L, L]))
                new_atoms.set_pbc(False)

            # Tagging
            cfg = new_atoms.info.get("Config_type", "")
            cfg += f" TorsionGuard(n={self.perturb_frame.get_input_value()[0]}, sigma={self.sigma_frame.get_input_value()[0]}, pbc={pbc_mode})"
            new_atoms.info["Config_type"] = cfg.strip()

            structures_out.append(new_atoms)

        return structures_out

    # ---------- Persistence ----------
    def to_dict(self):
        """Serialize the current configuration to a plain dictionary.
        
        Returns
        -------
        dict
            Dictionary that can be fed into ``from_dict`` to rebuild the state.
        """
        data = super().to_dict()
        data.update({
            "perturb_per_frame": self.perturb_frame.get_input_value(),
            "torsion_range_deg": self.torsion_frame.get_input_value(),
            "max_torsions_per_conf": self.max_torsions_frame.get_input_value(),
            "gaussian_sigma": self.sigma_frame.get_input_value(),
            "pbc_mode": self.pbc_combo.currentText(),
            "local_cutoff": self.local_cut_frame.get_input_value(),
            "local_subtree": self.local_sub_frame.get_input_value(),
            "bond_detect_factor": self.bond_detect_frame.get_input_value(),
            "bond_keep_min_factor": self.bond_min_frame.get_input_value(),
            "bond_keep_max_factor": self.bond_max_frame.get_input_value(),
            "bond_keep_max_enable": self.bond_max_enable.isChecked(),
            "nonbond_min_factor": self.nonbond_min_frame.get_input_value(),
            "max_retries": self.retries_frame.get_input_value(),
            "mult_bond_factor": self.multbond_frame.get_input_value(),
            "nonpbc_box_size": self.box_frame.get_input_value(),
            "bo_c_const": self.bo_c_frame.get_input_value(),
            "bo_threshold": self.bo_thr_frame.get_input_value(),
        })
        return data

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)
        self.perturb_frame.set_input_value(data_dict.get("perturb_per_frame", [100]))
        self.torsion_frame.set_input_value(data_dict.get("torsion_range_deg", [-180.0, 180.0]))
        self.max_torsions_frame.set_input_value(data_dict.get("max_torsions_per_conf", [5]))
        self.sigma_frame.set_input_value(data_dict.get("gaussian_sigma", [0.03]))

        pbc_mode = data_dict.get("pbc_mode", "auto")
        idx = {"auto": 0, "yes": 1, "no": 2}.get(pbc_mode, 0)
        self.pbc_combo.setCurrentIndex(idx)

        self.local_cut_frame.set_input_value(data_dict.get("local_cutoff", [150]))
        self.local_sub_frame.set_input_value(data_dict.get("local_subtree", [40]))
        self.bond_detect_frame.set_input_value(data_dict.get("bond_detect_factor", [1.15]))
        self.bond_min_frame.set_input_value(data_dict.get("bond_keep_min_factor", [0.60]))
        self.bond_max_frame.set_input_value(data_dict.get("bond_keep_max_factor", [1.15]))
        self.bond_max_enable.setChecked(bool(data_dict.get("bond_keep_max_enable", False)))
        self.nonbond_min_frame.set_input_value(data_dict.get("nonbond_min_factor", [0.80]))
        self.retries_frame.set_input_value(data_dict.get("max_retries", [12]))
        self.multbond_frame.set_input_value(data_dict.get("mult_bond_factor", [0.87]))
        self.box_frame.set_input_value(data_dict.get("nonpbc_box_size", [100.0]))
        self.bo_c_frame.set_input_value(data_dict.get("bo_c_const", [0.3]))
        self.bo_thr_frame.set_input_value(data_dict.get("bo_threshold", [0.2]))
