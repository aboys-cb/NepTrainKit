"""Card that wraps the torsion-guard PBC configurator for organic molecules."""

from __future__ import annotations

from typing import Any

from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import (
    OrganicMolConfigPBCOperation,
    OrganicMolConfigPBCParams,
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

    group = "Organic"
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
        row += 1

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip("Enable reproducible torsion/noise sampling")
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))
        self.settingLayout.addWidget(self.seed_checkbox, row, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, row, 1, 1, 2)

    def _current_pbc_mode(self) -> str:
        """Return the currently selected periodic boundary mode.
        
        Returns
        -------
        str
            One of ``"auto"``, ``"yes"``, or ``"no"``.
        """
        return self.pbc_combo.currentText()

    def create_operation(self):
        return OrganicMolConfigPBCOperation()

    def get_params(self) -> OrganicMolConfigPBCParams:
        return OrganicMolConfigPBCParams(
            perturb_per_frame=int(self.perturb_frame.get_input_value()[0]),
            torsion_range_deg=tuple(map(float, self.torsion_frame.get_input_value())),
            max_torsions_per_conf=int(self.max_torsions_frame.get_input_value()[0]),
            gaussian_sigma=float(self.sigma_frame.get_input_value()[0]),
            pbc_mode=self._current_pbc_mode(),
            local_cutoff=int(self.local_cut_frame.get_input_value()[0]),
            local_subtree=int(self.local_sub_frame.get_input_value()[0]),
            bond_detect_factor=float(self.bond_detect_frame.get_input_value()[0]),
            bond_keep_min_factor=float(self.bond_min_frame.get_input_value()[0]),
            bond_keep_max_factor=float(self.bond_max_frame.get_input_value()[0]),
            bond_keep_max_enable=self.bond_max_enable.isChecked(),
            nonbond_min_factor=float(self.nonbond_min_frame.get_input_value()[0]),
            max_retries=int(self.retries_frame.get_input_value()[0]),
            mult_bond_factor=float(self.multbond_frame.get_input_value()[0]),
            nonpbc_box_size=float(self.box_frame.get_input_value()[0]),
            bo_c_const=float(self.bo_c_frame.get_input_value()[0]),
            bo_threshold=float(self.bo_thr_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: OrganicMolConfigPBCParams) -> None:
        self.perturb_frame.set_input_value([int(params.perturb_per_frame)])
        self.torsion_frame.set_input_value([float(value) for value in params.torsion_range_deg])
        self.max_torsions_frame.set_input_value([int(params.max_torsions_per_conf)])
        self.sigma_frame.set_input_value([float(params.gaussian_sigma)])
        self.pbc_combo.setCurrentIndex({"auto": 0, "yes": 1, "no": 2}.get(params.pbc_mode, 0))
        self.local_cut_frame.set_input_value([int(params.local_cutoff)])
        self.local_sub_frame.set_input_value([int(params.local_subtree)])
        self.bond_detect_frame.set_input_value([float(params.bond_detect_factor)])
        self.bond_min_frame.set_input_value([float(params.bond_keep_min_factor)])
        self.bond_max_frame.set_input_value([float(params.bond_keep_max_factor)])
        self.bond_max_enable.setChecked(bool(params.bond_keep_max_enable))
        self.nonbond_min_frame.set_input_value([float(params.nonbond_min_factor)])
        self.retries_frame.set_input_value([int(params.max_retries)])
        self.multbond_frame.set_input_value([float(params.mult_bond_factor)])
        self.box_frame.set_input_value([float(params.nonpbc_box_size)])
        self.bo_c_frame.set_input_value([float(params.bo_c_const)])
        self.bo_thr_frame.set_input_value([float(params.bo_threshold)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

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
        return self.create_operation().run_structure(structure, self.get_params())

    # ---------- Persistence ----------
    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            raw_params["torsion_range_deg"] = tuple(raw_params.get("torsion_range_deg", [-180.0, 180.0]))
            params = OrganicMolConfigPBCParams(**raw_params)
        else:
            params = OrganicMolConfigPBCParams(
                perturb_per_frame=data_dict.get("perturb_per_frame", [100])[0],
                torsion_range_deg=tuple(data_dict.get("torsion_range_deg", [-180.0, 180.0])),
                max_torsions_per_conf=data_dict.get("max_torsions_per_conf", [50])[0],
                gaussian_sigma=data_dict.get("gaussian_sigma", [0.03])[0],
                pbc_mode=data_dict.get("pbc_mode", "auto"),
                local_cutoff=data_dict.get("local_cutoff", [200])[0],
                local_subtree=data_dict.get("local_subtree", [100])[0],
                bond_detect_factor=data_dict.get("bond_detect_factor", [1.15])[0],
                bond_keep_min_factor=data_dict.get("bond_keep_min_factor", [0.60])[0],
                bond_keep_max_factor=data_dict.get("bond_keep_max_factor", [1.15])[0],
                bond_keep_max_enable=data_dict.get("bond_keep_max_enable", False),
                nonbond_min_factor=data_dict.get("nonbond_min_factor", [0.80])[0],
                max_retries=data_dict.get("max_retries", [12])[0],
                mult_bond_factor=data_dict.get("mult_bond_factor", [0.87])[0],
                nonpbc_box_size=data_dict.get("nonpbc_box_size", [100.0])[0],
                bo_c_const=data_dict.get("bo_c_const", [0.3])[0],
                bo_threshold=data_dict.get("bo_threshold", [0.2])[0],
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
