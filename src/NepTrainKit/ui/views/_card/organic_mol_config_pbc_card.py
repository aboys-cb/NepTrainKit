"""Card that wraps the torsion-guard PBC configurator for organic molecules."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer
from qfluentwidgets import (
    CaptionLabel,
    ComboBox,
    CheckBox,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import (
    OrganicMolConfigPBCOperation,
    OrganicMolConfigPBCParams,
)
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


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
    _PREVIEW_DEBOUNCE_MS = 120

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self._input_structure = None
        self._preview_generation = 0
        self._preview_task: BackgroundTask | None = None
        self._active_preview_generation: int | None = None
        self._pending_preview = None
        self._preview_closing = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self._PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._start_preview)
        self.setTitle(self.tr("Molecular Conformers"))
        self._init_ui()

    # ---------- UI ----------
    def _init_ui(self):
        """Build compact sampling, boundary, topology, and guard sections."""
        self.setObjectName("organic_mol_config_pbc_card")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.perturb_frame = SpinBoxUnitInputFrame(self)
        self.perturb_frame.set_input("", 1, "int")
        self.perturb_frame.setRange(1, 100000)
        self.perturb_frame.set_input_value([100])
        self.perturb_frame.setAccessibleName(self.tr("Maximum outputs per input"))
        self.perturb_field = CompactField(
            self.tr("Maximum outputs per input"),
            self.perturb_frame,
            self.setting_widget,
            self.tr("Geometry guards can reduce the actual count."),
            inline=True,
            input_max_width=144,
        )

        self.torsion_frame = SpinBoxUnitInputFrame(self)
        self.torsion_frame.set_input(["°", "°"], 2, ["float", "float"])
        self.torsion_frame.setDecimals(3)
        self.torsion_frame.setSingleStep(1.0)
        self.torsion_frame.setRange(-360, 360)
        self.torsion_frame.set_input_value([-180.0, 180.0])
        self.torsion_frame.setAccessibleName(self.tr("Torsion increment range"))
        self.torsion_field = CompactField(
            self.tr("Torsion increment range"),
            self.torsion_frame,
            self.setting_widget,
            self.tr("Random angle added around each selected rotatable bond."),
        )

        self.max_torsions_frame = SpinBoxUnitInputFrame(self)
        self.max_torsions_frame.set_input("", 1, "int")
        self.max_torsions_frame.setRange(0, 10000)
        self.max_torsions_frame.set_input_value([5])
        self.max_torsions_frame.setAccessibleName(self.tr("Bonds rotated per output"))
        self.max_torsions_field = CompactField(
            self.tr("Bonds rotated per output"),
            self.max_torsions_frame,
            self.setting_widget,
            self.tr("0 disables torsion rotation but keeps coordinate noise."),
            inline=True,
            input_max_width=144,
        )

        self.sigma_frame = SpinBoxUnitInputFrame(self)
        self.sigma_frame.set_input("Å", 1, "float")
        self.sigma_frame.setDecimals(4)
        self.sigma_frame.setSingleStep(0.005)
        self.sigma_frame.setRange(0, 5)
        self.sigma_frame.set_input_value([0.03])
        self.sigma_frame.setAccessibleName(self.tr("Coordinate noise σ"))
        self.sigma_field = CompactField(
            self.tr("Coordinate noise σ"),
            self.sigma_frame,
            self.setting_widget,
            self.tr("Independent Cartesian noise applied after torsion rotations."),
            inline=True,
            input_max_width=144,
        )

        sampling_section = InspectorSection(
            self.tr("Conformer sampling"), self.setting_widget
        )
        sampling_grid = ResponsiveFormGrid(sampling_section)
        sampling_grid.add_field(self.perturb_field, span=2)
        sampling_grid.add_field(self.torsion_field, span=2)
        sampling_grid.add_field(self.max_torsions_field, span=2)
        sampling_grid.add_field(self.sigma_field, span=2)
        sampling_section.addWidget(sampling_grid)

        self.pbc_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.pbc_combo,
            [
                ("auto", "Follow input"),
                ("yes", "3D periodic"),
                ("no", "Nonperiodic"),
            ],
        )
        set_combo_value(self.pbc_combo, "auto")
        self.pbc_combo.setAccessibleName(self.tr("Output boundary"))
        self.pbc_field = CompactField(
            self.tr("Output boundary"),
            self.pbc_combo,
            self.setting_widget,
            self.tr("Follow input accepts either full 3D PBC or a nonperiodic molecule."),
        )

        self.box_frame = SpinBoxUnitInputFrame(self)
        self.box_frame.set_input("Å", 1, "float")
        self.box_frame.setDecimals(3)
        self.box_frame.setSingleStep(1.0)
        self.box_frame.setRange(1, 100000)
        self.box_frame.set_input_value([100.0])
        self.box_frame.setAccessibleName(self.tr("Nonperiodic display box"))
        self.box_field = CompactField(
            self.tr("Nonperiodic display box"),
            self.box_frame,
            self.setting_widget,
            self.tr("Cubic display cell only; output PBC remains off."),
            inline=True,
            input_max_width=144,
        )

        boundary_section = InspectorSection(self.tr("Boundary"), self.setting_widget)
        boundary_grid = ResponsiveFormGrid(boundary_section)
        boundary_grid.add_field(self.pbc_field, span=2)
        boundary_grid.add_field(self.box_field, span=2)
        boundary_section.addWidget(boundary_grid)

        self.seed_checkbox = CheckBox(self.tr("Use reproducible seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_field = CompactField(
            self.tr("Random seed"),
            self.seed_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        randomness_section = InspectorSection(self.tr("Randomness"), self.setting_widget)
        randomness_section.addWidget(self.seed_checkbox)
        randomness_section.addWidget(self.seed_field)

        self.local_cut_frame = SpinBoxUnitInputFrame(self)
        self.local_cut_frame.set_input(self.tr("atoms"), 1, "int")
        self.local_cut_frame.setRange(0, 1000000)
        self.local_cut_frame.set_input_value([150])
        self.local_cut_field = CompactField(
            self.tr("Large-molecule threshold"),
            self.local_cut_frame,
            self.setting_widget,
            self.tr("Above this atom count, rotations use capped local subtrees."),
            inline=True,
            input_max_width=144,
        )

        self.local_sub_frame = SpinBoxUnitInputFrame(self)
        self.local_sub_frame.set_input(self.tr("atoms"), 1, "int")
        self.local_sub_frame.setRange(1, 100000)
        self.local_sub_frame.set_input_value([40])
        self.local_sub_field = CompactField(
            self.tr("Local subtree cap"),
            self.local_sub_frame,
            self.setting_widget,
            self.tr("Maximum atoms rotated on one side of a bond in local mode."),
            inline=True,
            input_max_width=144,
        )

        self.bond_detect_frame = SpinBoxUnitInputFrame(self)
        self.bond_detect_frame.set_input("×", 1, "float")
        self.bond_detect_frame.setDecimals(4)
        self.bond_detect_frame.setSingleStep(0.01)
        self.bond_detect_frame.setRange(0.0001, 5)
        self.bond_detect_frame.set_input_value([1.15])
        self.bond_detect_field = CompactField(
            self.tr("Bond detection radius"),
            self.bond_detect_frame,
            self.setting_widget,
            self.tr("Maximum distance as a multiple of the covalent-radius sum."),
            inline=True,
            input_max_width=144,
        )

        self.bond_min_frame = SpinBoxUnitInputFrame(self)
        self.bond_min_frame.set_input("×", 1, "float")
        self.bond_min_frame.setDecimals(4)
        self.bond_min_frame.setSingleStep(0.01)
        self.bond_min_frame.setRange(0, 5)
        self.bond_min_frame.set_input_value([0.60])
        self.bond_min_field = CompactField(
            self.tr("Minimum bond length"),
            self.bond_min_frame,
            self.setting_widget,
            self.tr("0 disables the lower bond-length guard."),
            inline=True,
            input_max_width=144,
        )

        self.bo_c_frame = SpinBoxUnitInputFrame(self)
        self.bo_c_frame.set_input("Å", 1, "float")
        self.bo_c_frame.setDecimals(4)
        self.bo_c_frame.setSingleStep(0.01)
        self.bo_c_frame.setRange(0.01, 2.0)
        self.bo_c_frame.set_input_value([0.3])
        self.bo_c_field = CompactField(
            self.tr("Pauling decay length"),
            self.bo_c_frame,
            self.setting_widget,
            self.tr("Length c in exp((r₀-r)/c)."),
            inline=True,
            input_max_width=144,
        )

        self.bo_thr_frame = SpinBoxUnitInputFrame(self)
        self.bo_thr_frame.set_input("", 1, "float")
        self.bo_thr_frame.setDecimals(6)
        self.bo_thr_frame.setSingleStep(0.001)
        self.bo_thr_frame.setRange(0.0, 1.0)
        self.bo_thr_frame.set_input_value([0.2])
        self.bo_thr_field = CompactField(
            self.tr("Bond-order threshold"),
            self.bo_thr_frame,
            self.setting_widget,
            self.tr("Minimum estimated order required to create a topology edge."),
            inline=True,
            input_max_width=144,
        )

        self.bond_max_frame = SpinBoxUnitInputFrame(self)
        self.bond_max_frame.set_input("×", 1, "float")
        self.bond_max_frame.setDecimals(4)
        self.bond_max_frame.setSingleStep(0.01)
        self.bond_max_frame.setRange(0, 5)
        self.bond_max_frame.set_input_value([1.15])
        self.bond_max_enable = CheckBox(
            self.tr("Limit maximum bond length"), self.setting_widget
        )
        self.bond_max_enable.setChecked(False)
        self.bond_max_frame.setEnabled(False)
        self.bond_max_field = CompactField(
            self.tr("Maximum bond length"),
            self.bond_max_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )

        self.nonbond_min_frame = SpinBoxUnitInputFrame(self)
        self.nonbond_min_frame.set_input("×", 1, "float")
        self.nonbond_min_frame.setDecimals(4)
        self.nonbond_min_frame.setSingleStep(0.01)
        self.nonbond_min_frame.setRange(0, 5)
        self.nonbond_min_frame.set_input_value([0.80])
        self.nonbond_min_field = CompactField(
            self.tr("Minimum nonbonded distance"),
            self.nonbond_min_frame,
            self.setting_widget,
            self.tr("Reject closer nonbonded pairs; 0 disables this guard."),
            inline=True,
            input_max_width=144,
        )

        self.retries_frame = SpinBoxUnitInputFrame(self)
        self.retries_frame.set_input("", 1, "int")
        self.retries_frame.setRange(0, 100)
        self.retries_frame.set_input_value([12])
        self.retries_field = CompactField(
            self.tr("Retries per output"),
            self.retries_frame,
            self.setting_widget,
            self.tr("Each retry halves both torsion increments and coordinate noise."),
            inline=True,
            input_max_width=144,
        )

        self.multbond_frame = SpinBoxUnitInputFrame(self)
        self.multbond_frame.set_input("×", 1, "float")
        self.multbond_frame.setDecimals(4)
        self.multbond_frame.setSingleStep(0.01)
        self.multbond_frame.setRange(0, 2)
        self.multbond_frame.set_input_value([0.87])
        self.multbond_field = CompactField(
            self.tr("Short-bond rotation cutoff"),
            self.multbond_frame,
            self.setting_widget,
            self.tr("Shorter bonds are treated as non-rotatable."),
            inline=True,
            input_max_width=144,
        )

        self.advanced_checkbox = CheckBox(
            self.tr("Show topology and guards"),
            self.setting_widget,
        )
        self.advanced_checkbox.setChecked(False)

        topology_section = InspectorSection(
            self.tr("Topology recognition"),
            self.setting_widget,
            self.tr("These heuristics decide which geometric contacts are treated as bonds."),
        )
        topology_grid = ResponsiveFormGrid(topology_section)
        for field in (
            self.local_cut_field,
            self.local_sub_field,
            self.bond_detect_field,
            self.bo_c_field,
            self.bo_thr_field,
            self.multbond_field,
        ):
            topology_grid.add_field(field, span=2)
        topology_section.addWidget(topology_grid)

        guard_section = InspectorSection(
            self.tr("Geometry guards"),
            self.setting_widget,
            self.tr("Candidates that violate these distance checks are retried or skipped."),
        )
        guard_grid = ResponsiveFormGrid(guard_section)
        guard_grid.add_field(self.bond_min_field, span=2)
        guard_section.addWidget(self.bond_max_enable)
        guard_grid.add_field(self.bond_max_field, span=2)
        guard_grid.add_field(self.nonbond_min_field, span=2)
        guard_grid.add_field(self.retries_field, span=2)
        guard_section.addWidget(guard_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("organicConformerPreview")
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        self.topology_section = topology_section
        self.guard_section = guard_section
        self.advanced_controls = (topology_section, guard_section)

        self.settingLayout.addWidget(sampling_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(boundary_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(randomness_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 3, 0, 1, 3)
        self.settingLayout.addWidget(topology_section, 4, 0, 1, 3)
        self.settingLayout.addWidget(guard_section, 5, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 6, 0, 1, 3)

        self.advanced_checkbox.stateChanged.connect(
            self._update_advanced_visibility
        )
        self.bond_max_enable.stateChanged.connect(
            self._on_bond_max_changed
        )
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.pbc_combo.currentIndexChanged.connect(self._on_boundary_changed)
        self.bond_min_frame.object_list[0].valueChanged.connect(
            self._sync_bond_max_minimum
        )
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
        self._on_boundary_changed()
        self._refresh_preview()

    def _update_advanced_visibility(self, *_args) -> None:
        visible = self.advanced_checkbox.isChecked()
        for widget in self.advanced_controls:
            widget.setVisible(visible)
        self._update_tab_order()

    def _on_bond_max_changed(self, *_args) -> None:
        self.bond_max_frame.setEnabled(self.bond_max_enable.isChecked())
        self._sync_bond_max_minimum()
        self._update_tab_order()
        self._refresh_preview()

    def _on_seed_changed(self, *_args) -> None:
        self.seed_field.setVisible(self.seed_checkbox.isChecked())
        self._update_tab_order()
        self._refresh_preview()

    def _sync_bond_max_minimum(self, *_args) -> None:
        minimum = (
            max(0.0001, float(self.bond_min_frame.get_input_value()[0]))
            if self.bond_max_enable.isChecked()
            else 0.0
        )
        self.bond_max_frame.object_list[0].setMinimum(minimum)

    def _resolved_nonperiodic_output(self) -> bool | None:
        mode = self._current_pbc_mode()
        if mode == "no":
            return True
        if mode == "yes":
            return False
        if self._input_structure is None:
            return None
        flags = self._input_structure.get_pbc()
        return not bool(all(flags))

    def _on_boundary_changed(self, *_args) -> None:
        resolved_nonperiodic = self._resolved_nonperiodic_output()
        self.box_field.setVisible(resolved_nonperiodic is not False)
        self._update_tab_order()
        self._refresh_preview()

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
        if dataset is None:
            self._input_count = 0
        elif hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            self._input_count = 1
        else:
            try:
                self._input_count = len(dataset)
            except TypeError:
                self._input_count = None
        self._on_boundary_changed()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        self._preview_generation += 1
        self._pending_preview = None
        self._preview_timer.stop()
        if self._input_structure is None:
            if self._preview_task is not None:
                self._preview_task.stop_work()
            self.preview_label.setText(
                self.tr(
                    "Load an upstream molecule to preview detected bonds and rotatable torsions."
                )
            )
            return
        self.preview_label.setText(self.tr("Calculating topology preview in background…"))
        self._preview_timer.start()

    def _start_preview(self) -> None:
        if self._preview_closing or self._input_structure is None:
            return
        request = (
            self._preview_generation,
            self._input_structure.copy(),
            self.get_params(),
        )
        if self._preview_task is not None:
            self._pending_preview = request
            self._preview_task.stop_work()
            return
        self._start_preview_task(request)

    def _start_preview_task(self, request) -> None:
        request_id, structure, params = request
        task = BackgroundTask(self, show_tip=False)
        self._preview_task = task
        self._active_preview_generation = request_id
        task.succeeded.connect(self._on_preview_succeeded)
        task.failed.connect(self._on_preview_failed)
        task.finished.connect(self._on_preview_task_finished)
        task.start_work(
            self._calculate_preview,
            request_id,
            structure,
            params,
        )

    @staticmethod
    def _calculate_preview(request_id, structure, params):
        try:
            summary = OrganicMolConfigPBCOperation.topology_summary(structure, params)
        except CardOperationError as exc:
            return request_id, exc
        return request_id, summary

    def _on_preview_succeeded(self, result) -> None:
        request_id, summary = result
        if self._preview_closing or request_id != self._preview_generation:
            return
        if isinstance(summary, CardOperationError):
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(summary)
                )
            )
            return
        self._apply_preview_summary(summary)

    def _on_preview_failed(self, message: str) -> None:
        task = self._preview_task
        if (
            self._preview_closing
            or task is None
            or self._active_preview_generation != self._preview_generation
            or self._pending_preview is not None
        ):
            return
        self.preview_label.setText(
            "⚠ "
            + self.tr("Preview unavailable: {error}").format(
                error=translate_runtime_message(message)
            )
        )

    def _on_preview_task_finished(self) -> None:
        task = self._preview_task
        if task is None:
            return
        task.wait()
        task.deleteLater()
        self._preview_task = None
        self._active_preview_generation = None
        if self._preview_closing:
            QTimer.singleShot(0, self.close)
            return
        pending = self._pending_preview
        self._pending_preview = None
        if pending is not None and pending[0] == self._preview_generation:
            self._start_preview_task(pending)

    def _apply_preview_summary(self, summary) -> None:
        boundary = self.tr("3D periodic") if summary.pbc_active else self.tr("nonperiodic")
        dataset_maximum = ""
        input_count = getattr(self, "_input_count", None)
        if input_count:
            dataset_maximum = self.tr(" · dataset maximum {total}").format(
                total=input_count * summary.requested_outputs
            )
        component_text = (
            self.tr("{count} component")
            if summary.component_count == 1
            else self.tr("{count} components")
        ).format(count=summary.component_count)
        message = self.tr(
            "First input: {atoms} atoms · {bonds} detected bonds / {torsions} rotatable · {components} · {boundary} · up to {outputs} outputs{dataset_maximum}"
        ).format(
            atoms=summary.atom_count,
            bonds=summary.bond_count,
            torsions=summary.torsion_count,
            components=component_text,
            boundary=boundary,
            outputs=summary.requested_outputs,
            dataset_maximum=dataset_maximum,
        )
        if not summary.torsion_active:
            message += " · " + self.tr(
                "no active torsion; outputs use Gaussian noise only"
            )
        elif summary.local_mode:
            message += " · " + self.tr("local subtree rotation is active")
        self.preview_label.setText(message)

    def closeEvent(self, event) -> None:
        self._preview_closing = True
        self._preview_generation += 1
        self._pending_preview = None
        self._preview_timer.stop()
        task = self._preview_task
        if task is not None:
            task.stop_work()
            if task.isRunning() and not task.wait(200):
                event.ignore()
                return
            task.deleteLater()
            self._preview_task = None
            self._active_preview_generation = None
        super().closeEvent(event)

    def _update_tab_order(self) -> None:
        if not hasattr(self, "advanced_checkbox"):
            return
        widgets = [
            *self.perturb_frame.object_list,
            *self.torsion_frame.object_list,
            *self.max_torsions_frame.object_list,
            *self.sigma_frame.object_list,
            self.pbc_combo,
        ]
        if not self.box_field.isHidden():
            widgets.extend(self.box_frame.object_list)
        widgets.extend([
            self.seed_checkbox,
        ])
        if not self.seed_field.isHidden():
            widgets.extend(self.seed_frame.object_list)
        widgets.append(self.advanced_checkbox)
        if self.advanced_checkbox.isChecked():
            widgets.extend(
                [
                    *self.local_cut_frame.object_list,
                    *self.local_sub_frame.object_list,
                    *self.bond_detect_frame.object_list,
                    *self.bo_c_frame.object_list,
                    *self.bo_thr_frame.object_list,
                    *self.multbond_frame.object_list,
                    *self.bond_min_frame.object_list,
                    self.bond_max_enable,
                ]
            )
            if self.bond_max_frame.isEnabled():
                widgets.extend(self.bond_max_frame.object_list)
            widgets.extend(
                [
                    *self.nonbond_min_frame.object_list,
                    *self.retries_frame.object_list,
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
        self._on_boundary_changed()
        self._refresh_preview()

    def get_summary_text(self) -> str:
        params = self.get_params()
        return self.tr(
            "≤{outputs} outputs · {minimum}°→{maximum}° · {boundary}"
        ).format(
            outputs=params.perturb_per_frame,
            minimum=f"{params.torsion_range_deg[0]:g}",
            maximum=f"{params.torsion_range_deg[1]:g}",
            boundary={
                "auto": self.tr("follow input"),
                "yes": self.tr("3D periodic"),
                "no": self.tr("nonperiodic"),
            }.get(params.pbc_mode, params.pbc_mode),
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Check the detected-bond and rotatable-bond counts before generating conformers."
        )

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
