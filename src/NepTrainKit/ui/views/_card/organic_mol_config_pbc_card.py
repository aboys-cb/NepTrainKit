"""Card that wraps the torsion-guard PBC configurator for organic molecules."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    ToolTipFilter,
    ToolTipPosition,
    CheckBox,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import (
    OrganicMolConfigPBCOperation,
    OrganicMolConfigPBCParams,
)
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
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
    contributors = [
        {"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"},
    ]

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Organic Conformer Sampling"))
        self._init_ui()

    # ---------- UI ----------
    def _init_ui(self):
        """Create all of the widgets required to configure the torsion-guard workflow.
        """
        self.setObjectName("organic_mol_config_pbc_card")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        row = 0

        # perturb_per_frame
        self.perturb_label = BodyLabel(self.tr("Requested outputs per input"), self.setting_widget)
        self.perturb_label.setToolTip(self.tr("Failed geometry checks are skipped, so the actual count can be lower"))
        self.perturb_label.installEventFilter(ToolTipFilter(self.perturb_label, 300, ToolTipPosition.TOP))
        self.perturb_frame = SpinBoxUnitInputFrame(self)
        self.perturb_frame.set_input("", 1, "int")
        self.perturb_frame.setRange(1, 100000)
        self.perturb_frame.set_input_value([100])
        self.settingLayout.addWidget(self.perturb_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.perturb_frame, row, 1, 1, 2)
        row += 1

        # torsion_range_deg
        self.torsion_label = BodyLabel(self.tr("Torsion angle increment"), self.setting_widget)
        self.torsion_label.setToolTip(self.tr("Random rotation added around each selected rotatable bond, in degrees"))
        self.torsion_label.installEventFilter(ToolTipFilter(self.torsion_label, 300, ToolTipPosition.TOP))
        self.torsion_frame = SpinBoxUnitInputFrame(self)
        self.torsion_frame.set_input(["°", "°"], 2, ["float", "float"])
        self.torsion_frame.setDecimals(3)
        self.torsion_frame.setSingleStep(1.0)
        self.torsion_frame.setRange(-360, 360)
        self.torsion_frame.set_input_value([-180.0, 180.0])
        self.settingLayout.addWidget(self.torsion_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.torsion_frame, row, 1, 1, 2)
        row += 1

        # max_torsions_per_conf
        self.max_torsions_label = BodyLabel(self.tr("Rotatable bonds per output"), self.setting_widget)
        self.max_torsions_label.setToolTip(self.tr("Maximum number of distinct rotatable bonds changed in one output"))
        self.max_torsions_label.installEventFilter(ToolTipFilter(self.max_torsions_label, 300, ToolTipPosition.TOP))
        self.max_torsions_frame = SpinBoxUnitInputFrame(self)
        self.max_torsions_frame.set_input("", 1, "int")
        self.max_torsions_frame.setRange(0, 10000)
        self.max_torsions_frame.set_input_value([5])
        self.settingLayout.addWidget(self.max_torsions_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.max_torsions_frame, row, 1, 1, 2)
        row += 1

        # gaussian_sigma
        self.sigma_label = BodyLabel(self.tr("Gaussian coordinate noise"), self.setting_widget)
        self.sigma_label.setToolTip(self.tr("Independent Cartesian noise applied to every atom after torsion rotations"))
        self.sigma_label.installEventFilter(ToolTipFilter(self.sigma_label, 300, ToolTipPosition.TOP))
        self.sigma_frame = SpinBoxUnitInputFrame(self)
        self.sigma_frame.set_input("Å", 1, "float")
        self.sigma_frame.setDecimals(4)
        self.sigma_frame.setSingleStep(0.005)
        self.sigma_frame.setRange(0, 5)
        self.sigma_frame.set_input_value([0.03])
        self.settingLayout.addWidget(self.sigma_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.sigma_frame, row, 1, 1, 2)
        row += 1

        # pbc mode
        self.pbc_label = BodyLabel(self.tr("Boundary handling"), self.setting_widget)
        self.pbc_label.setToolTip(self.tr("Auto follows full 3D input PBC; mixed periodic boundaries are not supported"))
        self.pbc_label.installEventFilter(ToolTipFilter(self.pbc_label, 300, ToolTipPosition.TOP))
        self.pbc_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.pbc_combo,
            [
                ("auto", "Auto (follow input PBC)"),
                ("yes", "Force full 3D PBC"),
                ("no", "Nonperiodic molecule"),
            ],
        )
        set_combo_value(self.pbc_combo, "auto")
        self.settingLayout.addWidget(self.pbc_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.pbc_combo, row, 1, 1, 2)
        row += 1

        # local_mode_cutoff_atoms
        self.local_cut_label = BodyLabel(self.tr("Local rotation threshold"), self.setting_widget)
        self.local_cut_label.setToolTip(self.tr("Use capped local subtrees when the input has more atoms than this value"))
        self.local_cut_label.installEventFilter(ToolTipFilter(self.local_cut_label, 300, ToolTipPosition.TOP))
        self.local_cut_frame = SpinBoxUnitInputFrame(self)
        self.local_cut_frame.set_input(self.tr("atoms"), 1, "int")
        self.local_cut_frame.setRange(0, 1000000)
        self.local_cut_frame.set_input_value([150])
        self.settingLayout.addWidget(self.local_cut_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.local_cut_frame, row, 1, 1, 2)
        row += 1

        # local_torsion_max_subtree
        self.local_sub_label = BodyLabel(self.tr("Local subtree atom cap"), self.setting_widget)
        self.local_sub_label.setToolTip(self.tr("Maximum atoms rotated on one side of a bond in local mode"))
        self.local_sub_label.installEventFilter(ToolTipFilter(self.local_sub_label, 300, ToolTipPosition.TOP))
        self.local_sub_frame = SpinBoxUnitInputFrame(self)
        self.local_sub_frame.set_input(self.tr("atoms"), 1, "int")
        self.local_sub_frame.setRange(1, 100000)
        self.local_sub_frame.set_input_value([40])
        self.settingLayout.addWidget(self.local_sub_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.local_sub_frame, row, 1, 1, 2)
        row += 1

        # bond_detect_factor
        self.bond_detect_label = BodyLabel(self.tr("Bond detection factor"), self.setting_widget)
        self.bond_detect_label.setToolTip(self.tr("Maximum detected bond distance as a multiple of the covalent-radius sum"))
        self.bond_detect_label.installEventFilter(ToolTipFilter(self.bond_detect_label, 300, ToolTipPosition.TOP))
        self.bond_detect_frame = SpinBoxUnitInputFrame(self)
        self.bond_detect_frame.set_input("x", 1, "float")
        self.bond_detect_frame.setDecimals(4)
        self.bond_detect_frame.setSingleStep(0.01)
        self.bond_detect_frame.setRange(0, 5)
        self.bond_detect_frame.set_input_value([1.15])
        self.settingLayout.addWidget(self.bond_detect_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_detect_frame, row, 1, 1, 2)
        row += 1

        # bond_keep_min_factor
        self.bond_min_label = BodyLabel(self.tr("Minimum bond-length factor"), self.setting_widget)
        self.bond_min_label.setToolTip(self.tr("Reject a candidate if an original bond is shorter than this covalent-radius factor; 0 disables"))
        self.bond_min_label.installEventFilter(ToolTipFilter(self.bond_min_label, 300, ToolTipPosition.TOP))
        self.bond_min_frame = SpinBoxUnitInputFrame(self)
        self.bond_min_frame.set_input("x", 1, "float")
        self.bond_min_frame.setDecimals(4)
        self.bond_min_frame.setSingleStep(0.01)
        self.bond_min_frame.setRange(0, 5)
        self.bond_min_frame.set_input_value([0.60])
        self.settingLayout.addWidget(self.bond_min_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_min_frame, row, 1, 1, 2)
        row += 1

        # Pauling bond-order params
        self.bo_c_label = BodyLabel(self.tr("Pauling decay constant"), self.setting_widget)
        self.bo_c_label.setToolTip(self.tr("Bond order constant c in exp((r0-r)/c)"))
        self.bo_c_label.installEventFilter(ToolTipFilter(self.bo_c_label, 300, ToolTipPosition.TOP))
        self.bo_c_frame = SpinBoxUnitInputFrame(self)
        self.bo_c_frame.set_input("", 1, "float")
        self.bo_c_frame.setDecimals(4)
        self.bo_c_frame.setSingleStep(0.01)
        self.bo_c_frame.setRange(0.01, 2.0)
        self.bo_c_frame.set_input_value([0.3])
        self.settingLayout.addWidget(self.bo_c_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bo_c_frame, row, 1, 1, 2)
        row += 1

        self.bo_thr_label = BodyLabel(self.tr("Bond-order threshold"), self.setting_widget)
        self.bo_thr_label.setToolTip(self.tr("Minimum estimated Pauling bond order required to form a topology edge"))
        self.bo_thr_label.installEventFilter(ToolTipFilter(self.bo_thr_label, 300, ToolTipPosition.TOP))
        self.bo_thr_frame = SpinBoxUnitInputFrame(self)
        self.bo_thr_frame.set_input("", 1, "float")
        self.bo_thr_frame.setDecimals(6)
        self.bo_thr_frame.setSingleStep(0.001)
        self.bo_thr_frame.setRange(0.0, 1.0)
        self.bo_thr_frame.set_input_value([0.2])
        self.settingLayout.addWidget(self.bo_thr_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bo_thr_frame, row, 1, 1, 2)
        row += 1

        # bond_keep_max_factor (optional)
        self.bond_max_label = BodyLabel(self.tr("Maximum bond-length factor"), self.setting_widget)
        self.bond_max_label.setToolTip(self.tr("Optional upper bound for original bonded pairs"))
        self.bond_max_label.installEventFilter(ToolTipFilter(self.bond_max_label, 300, ToolTipPosition.TOP))
        self.bond_max_frame = SpinBoxUnitInputFrame(self)
        self.bond_max_frame.set_input("x", 1, "float")
        self.bond_max_frame.setDecimals(4)
        self.bond_max_frame.setSingleStep(0.01)
        self.bond_max_frame.setRange(0, 5)
        self.bond_max_frame.set_input_value([1.15])
        self.bond_max_enable = CheckBox(self.tr("Enable upper bound"), self.setting_widget)
        self.bond_max_enable.setChecked(False)
        self.bond_max_frame.setEnabled(False)
        self.settingLayout.addWidget(self.bond_max_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_max_frame, row, 1, 1, 1)
        self.settingLayout.addWidget(self.bond_max_enable, row, 2, 1, 1)
        row += 1

        # nonbond_min_factor
        self.nonbond_min_label = BodyLabel(self.tr("Nonbonded distance factor"), self.setting_widget)
        self.nonbond_min_label.setToolTip(self.tr("Reject nonbonded pairs closer than this covalent-radius-sum factor"))
        self.nonbond_min_label.installEventFilter(ToolTipFilter(self.nonbond_min_label, 300, ToolTipPosition.TOP))
        self.nonbond_min_frame = SpinBoxUnitInputFrame(self)
        self.nonbond_min_frame.set_input("x", 1, "float")
        self.nonbond_min_frame.setDecimals(4)
        self.nonbond_min_frame.setSingleStep(0.01)
        self.nonbond_min_frame.setRange(0, 5)
        self.nonbond_min_frame.set_input_value([0.80])
        self.settingLayout.addWidget(self.nonbond_min_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.nonbond_min_frame, row, 1, 1, 2)
        row += 1

        # max_retries_per_frame
        self.retries_label = BodyLabel(self.tr("Guard retries per output"), self.setting_widget)
        self.retries_label.setToolTip(self.tr("Each retry halves both torsion increments and Gaussian noise"))
        self.retries_label.installEventFilter(ToolTipFilter(self.retries_label, 300, ToolTipPosition.TOP))
        self.retries_frame = SpinBoxUnitInputFrame(self)
        self.retries_frame.set_input("", 1, "int")
        self.retries_frame.setRange(0, 100)
        self.retries_frame.set_input_value([12])
        self.settingLayout.addWidget(self.retries_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.retries_frame, row, 1, 1, 2)
        row += 1

        # MULT_BOND_FACTOR
        self.multbond_label = BodyLabel(self.tr("Short-bond rotation cutoff"), self.setting_widget)
        self.multbond_label.setToolTip(self.tr("Do not rotate bonds shorter than this covalent-radius-sum factor"))
        self.multbond_label.installEventFilter(ToolTipFilter(self.multbond_label, 300, ToolTipPosition.TOP))
        self.multbond_frame = SpinBoxUnitInputFrame(self)
        self.multbond_frame.set_input("x", 1, "float")
        self.multbond_frame.setDecimals(4)
        self.multbond_frame.setSingleStep(0.01)
        self.multbond_frame.setRange(0, 2)
        self.multbond_frame.set_input_value([0.87])
        self.settingLayout.addWidget(self.multbond_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.multbond_frame, row, 1, 1, 2)
        row += 1

        # nonpbc_box_size
        self.box_label = BodyLabel(self.tr("Nonperiodic display box"), self.setting_widget)
        self.box_label.setToolTip(self.tr("Cubic cell edge assigned to nonperiodic outputs; it is not a physical boundary"))
        self.box_label.installEventFilter(ToolTipFilter(self.box_label, 300, ToolTipPosition.TOP))
        self.box_frame = SpinBoxUnitInputFrame(self)
        self.box_frame.set_input("Å", 1, "float")
        self.box_frame.setDecimals(3)
        self.box_frame.setSingleStep(1.0)
        self.box_frame.setRange(1, 100000)
        self.box_frame.set_input_value([100.0])
        self.settingLayout.addWidget(self.box_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.box_frame, row, 1, 1, 2)
        row += 1

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip(self.tr("Enable reproducible torsion/noise sampling"))
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.settingLayout.addWidget(self.seed_checkbox, row, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, row, 1, 1, 2)
        row += 1

        self.advanced_checkbox = CheckBox(
            self.tr("Show topology and geometry-guard settings"),
            self.setting_widget,
        )
        self.advanced_checkbox.setChecked(False)
        self.settingLayout.addWidget(self.advanced_checkbox, row, 0, 1, 3)
        row += 1

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("organicConformerPreview")
        self.settingLayout.addWidget(self.preview_label, row, 0, 1, 3)

        self.advanced_controls = (
            self.local_cut_label,
            self.local_cut_frame,
            self.local_sub_label,
            self.local_sub_frame,
            self.bond_detect_label,
            self.bond_detect_frame,
            self.bond_min_label,
            self.bond_min_frame,
            self.bo_c_label,
            self.bo_c_frame,
            self.bo_thr_label,
            self.bo_thr_frame,
            self.bond_max_label,
            self.bond_max_frame,
            self.bond_max_enable,
            self.nonbond_min_label,
            self.nonbond_min_frame,
            self.retries_label,
            self.retries_frame,
            self.multbond_label,
            self.multbond_frame,
            self.box_label,
            self.box_frame,
        )
        self.advanced_checkbox.stateChanged.connect(
            self._update_advanced_visibility
        )
        self.bond_max_enable.stateChanged.connect(
            self._on_bond_max_changed
        )
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.pbc_combo.currentIndexChanged.connect(self._refresh_preview)
        for frame in (
            self.perturb_frame,
            self.torsion_frame,
            self.max_torsions_frame,
            self.sigma_frame,
            self.local_cut_frame,
            self.local_sub_frame,
            self.bond_detect_frame,
            self.bond_min_frame,
            self.bo_c_frame,
            self.bo_thr_frame,
            self.bond_max_frame,
            self.nonbond_min_frame,
            self.retries_frame,
            self.multbond_frame,
            self.box_frame,
            self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._refresh_preview)

        self._update_advanced_visibility()
        self._on_bond_max_changed()
        self._on_seed_changed()
        self._refresh_preview()

    def _update_advanced_visibility(self, *_args) -> None:
        visible = self.advanced_checkbox.isChecked()
        for widget in self.advanced_controls:
            widget.setVisible(visible)
        self._update_tab_order()

    def _on_bond_max_changed(self, *_args) -> None:
        self.bond_max_frame.setEnabled(self.bond_max_enable.isChecked())
        self._update_tab_order()
        self._refresh_preview()

    def _on_seed_changed(self, *_args) -> None:
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._update_tab_order()

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an upstream molecule to preview detected bonds and rotatable torsions."
                )
            )
            return
        try:
            summary = self.create_operation().topology_summary(
                self._input_structure,
                self.get_params(),
            )
        except (TypeError, ValueError, IndexError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return

        boundary = self.tr("3D periodic") if summary["pbc_active"] else self.tr("nonperiodic")
        message = self.tr(
            "First input: {atoms} atoms · {bonds} detected bonds / {torsions} rotatable · {components} molecular components · {boundary} · request {outputs} outputs"
        ).format(
            atoms=summary["atom_count"],
            bonds=summary["bond_count"],
            torsions=summary["torsion_count"],
            components=summary["component_count"],
            boundary=boundary,
            outputs=summary["requested_outputs"],
        )
        if not summary["torsion_active"]:
            message += " · " + self.tr(
                "no active torsion; outputs use Gaussian noise only"
            )
        elif summary["local_mode"]:
            message += " · " + self.tr("local subtree rotation is active")
        self.preview_label.setText(message)

    def _update_tab_order(self) -> None:
        if not hasattr(self, "advanced_checkbox"):
            return
        widgets = [
            *self.perturb_frame.object_list,
            *self.torsion_frame.object_list,
            *self.max_torsions_frame.object_list,
            *self.sigma_frame.object_list,
            self.pbc_combo,
            self.seed_checkbox,
        ]
        if self.seed_frame.isEnabled():
            widgets.extend(self.seed_frame.object_list)
        widgets.append(self.advanced_checkbox)
        if self.advanced_checkbox.isChecked():
            widgets.extend(
                [
                    *self.local_cut_frame.object_list,
                    *self.local_sub_frame.object_list,
                    *self.bond_detect_frame.object_list,
                    *self.bond_min_frame.object_list,
                    *self.bo_c_frame.object_list,
                    *self.bo_thr_frame.object_list,
                    self.bond_max_enable,
                ]
            )
            if self.bond_max_frame.isEnabled():
                widgets.extend(self.bond_max_frame.object_list)
            widgets.extend(
                [
                    *self.nonbond_min_frame.object_list,
                    *self.retries_frame.object_list,
                    *self.multbond_frame.object_list,
                    *self.box_frame.object_list,
                ]
            )
        self.tab_order_widgets = widgets

    def _current_pbc_mode(self) -> str:
        """Return the currently selected periodic boundary mode.
        
        Returns
        -------
        str
            One of ``"auto"``, ``"yes"``, or ``"no"``.
        """
        return combo_value(self.pbc_combo)

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
        set_combo_value(self.pbc_combo, params.pbc_mode)
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
        self._on_bond_max_changed()
        self._on_seed_changed()
        self._refresh_preview()

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
            raw_params = dict(raw_params)
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
