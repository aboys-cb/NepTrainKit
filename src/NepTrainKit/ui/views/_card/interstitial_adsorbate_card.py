"""Card for sampling bulk interstitial and upper-surface adsorbate candidates."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import InsertDefectOperation, InsertDefectParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    ElementLineEdit,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class InsertDefectCard(MakeDataCard):
    """Create random bulk interstitial or upper-surface adsorbate candidates."""

    group = "Defect"
    card_name = "Insert Defect"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Insert / Adsorb"))
        self._init_ui()

    def _init_ui(self):
        """Build mode, species, geometry, randomness, and preview controls."""
        self.setObjectName("insert_defect_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItem(
            self.tr("Random bulk interstitial"),
            userData=0,
        )
        self.mode_combo.addItem(
            self.tr("Random upper-surface adsorption"),
            userData=1,
        )
        self.mode_combo.setAccessibleName(self.tr("Insertion mode"))
        self.mode_field = CompactField(
            self.tr("Insertion mode"),
            self.mode_combo,
            self.setting_widget,
        )

        self.species_edit = ElementLineEdit(self.setting_widget, multiple=True)
        self.species_edit.setPlaceholderText(self.tr("e.g. Li or Li:7, Na:3"))
        self.species_edit.setClearButtonEnabled(True)
        self.species_edit.setAccessibleName(self.tr("Inserted species and weights"))
        self.species_field = CompactField(
            self.tr("Inserted species and weights"),
            self.species_edit,
            self.setting_widget,
            self.tr("Each inserted atom is sampled independently from these relative weights."),
        )

        self.insert_count_frame = SpinBoxUnitInputFrame(self)
        self.insert_count_frame.set_input("", 1, "int")
        self.insert_count_frame.setRange(1, 20)
        self.insert_count_frame.set_input_value([1])
        self.insert_count_frame.setFixedWidth(144)
        self.insert_count_frame.setAccessibleName(self.tr("Atoms inserted per output"))
        self.insert_count_field = CompactField(
            self.tr("Atoms per output"),
            self.insert_count_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )

        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("", 1, "int")
        self.structures_frame.setRange(1, 1000)
        self.structures_frame.set_input_value([10])
        self.structures_frame.setFixedWidth(144)
        self.structures_frame.setAccessibleName(self.tr("Outputs per input"))
        self.structures_field = CompactField(
            self.tr("Outputs per input"),
            self.structures_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )

        self.min_distance_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_frame.set_input("Å", 1, "float")
        self.min_distance_frame.setRange(0.001, 10.0)
        self.min_distance_frame.object_list[0].setDecimals(3)
        self.min_distance_frame.set_input_value([1.4])
        self.min_distance_frame.setFixedWidth(144)
        self.min_distance_frame.setAccessibleName(self.tr("Minimum atom distance"))
        self.min_distance_field = CompactField(
            self.tr("Minimum atom distance"),
            self.min_distance_frame,
            self.setting_widget,
            self.tr("Checked against host atoms and atoms inserted earlier in the same output."),
            inline=True,
            input_max_width=144,
        )

        self.max_attempts_frame = SpinBoxUnitInputFrame(self)
        self.max_attempts_frame.set_input("", 1, "int")
        self.max_attempts_frame.setRange(1, 1000)
        self.max_attempts_frame.set_input_value([200])
        self.max_attempts_frame.setFixedWidth(144)
        self.max_attempts_frame.setAccessibleName(self.tr("Placement attempts per atom"))
        self.max_attempts_field = CompactField(
            self.tr("Placement attempts per atom"),
            self.max_attempts_frame,
            self.setting_widget,
            self.tr("A search budget; increasing it does not change the distance rule."),
            inline=True,
            input_max_width=144,
        )

        self.seed_checkbox = CheckBox(
            self.tr("Use seed"),
            self.setting_widget,
        )
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setFixedWidth(144)
        self.seed_frame.setEnabled(False)
        self.seed_frame.setAccessibleName(self.tr("Random seed"))
        self.seed_row = QWidget(self.setting_widget)
        seed_layout = QHBoxLayout(self.seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(8)
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_frame)
        seed_layout.addStretch(1)

        self.axis_combo = ComboBox(self.setting_widget)
        for axis, label in enumerate(
            (
                self.tr("Lattice a"),
                self.tr("Lattice b"),
                self.tr("Lattice c"),
            )
        ):
            self.axis_combo.addItem(label, userData=axis)
        self.axis_combo.setCurrentIndex(2)
        self.axis_combo.setAccessibleName(self.tr("Vacuum / surface-normal direction"))
        self.axis_field = CompactField(
            self.tr("Vacuum direction"),
            self.axis_combo,
            self.setting_widget,
            self.tr("Choose the lattice direction that contains the slab vacuum."),
        )
        # Compatibility aliases for callers that used the former label attributes.
        self.axis_label = self.axis_field

        self.offset_frame = SpinBoxUnitInputFrame(self)
        self.offset_frame.set_input("Å", 1, "float")
        self.offset_frame.setRange(0.001, 10.0)
        self.offset_frame.object_list[0].setDecimals(3)
        self.offset_frame.set_input_value([1.5])
        self.offset_frame.setFixedWidth(144)
        self.offset_frame.setAccessibleName(self.tr("Height above top atomic plane"))
        self.offset_field = CompactField(
            self.tr("Adsorption height"),
            self.offset_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        self.offset_label = self.offset_field

        self.placement_section = InspectorSection(
            self.tr("Placement"),
            self.setting_widget,
        )
        placement_grid = ResponsiveFormGrid(self.placement_section)
        placement_grid.add_field(self.mode_field, span=2)
        placement_grid.add_field(self.species_field, span=2)
        placement_grid.add_field(self.axis_field)
        placement_grid.add_field(self.offset_field)
        self.placement_section.addWidget(placement_grid)

        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(self.insert_count_field)
        generation_grid.add_field(self.structures_field)
        generation_grid.add_field(self.min_distance_field, span=2)
        generation_grid.add_field(self.seed_row, span=2)
        self.advanced_checkbox = CheckBox(self.tr("Advanced settings"), self.setting_widget)
        generation_grid.add_field(self.advanced_checkbox, span=2)
        generation_grid.add_field(self.max_attempts_field, span=2)
        generation_section.addWidget(generation_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setObjectName("insertDefectPreview")
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(self.placement_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 2, 0, 1, 3)

        self.adsorption_controls = (
            self.axis_field,
            self.offset_field,
        )
        self.mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        self.species_edit.textChanged.connect(self._refresh_preview)
        self.insert_count_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.structures_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.min_distance_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.max_attempts_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.axis_combo.currentIndexChanged.connect(self._refresh_preview)
        self.offset_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.advanced_checkbox.toggled.connect(self._update_advanced_visibility)

        self._update_mode_visibility(self.mode_combo.currentIndex())
        self._on_seed_changed()
        self._update_advanced_visibility()
        self._refresh_preview()

    def _update_mode_visibility(self, _index: int) -> None:
        is_adsorption = self.mode_combo.currentData() == 1
        for widget in self.adsorption_controls:
            widget.setVisible(is_adsorption)
        description = (
            self.tr("Samples above the original upper atomic plane along the selected lattice-normal direction.")
            if is_adsorption
            else self.tr("Samples the full periodic cell uniformly; for a slab, this includes its vacuum region.")
        )
        self.placement_section.description_label.setText(description)
        self.placement_section.description_label.setVisible(True)
        self._refresh_preview()
        self._update_tab_order()

    def _on_seed_changed(self) -> None:
        enabled = self.seed_checkbox.isChecked()
        self.seed_frame.setEnabled(enabled)
        self.seed_frame.setVisible(enabled)
        self._update_tab_order()

    def _update_advanced_visibility(self, *_args) -> None:
        self.max_attempts_field.setVisible(self.advanced_checkbox.isChecked())
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

    @staticmethod
    def _dataset_count(dataset) -> int:
        if dataset is None:
            return 0
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return 1
        try:
            return len(dataset)
        except TypeError:
            return 0

    @staticmethod
    def _species_summary(species, weights) -> str:
        if len(species) == 1:
            return str(species[0])
        return " / ".join(f"{symbol} {weight * 100:.3g}%" for symbol, weight in zip(species, weights))

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        if not self.species_edit.text().strip():
            self.preview_label.setText(
                "⚠ " + self.tr("Enter at least one inserted species, for example Li or Li:7, Na:3.")
            )
            return
        if self._input_structure is None:
            self.preview_label.setText(self.tr("Load an upstream structure to preview insertion geometry."))
            return

        try:
            summary = self.create_operation().sampling_summary(
                self._input_structure,
                self.get_params(),
            )
        except ValueError as exc:
            self.preview_label.setText("⚠ " + self.tr("Preview unavailable: {error}").format(error=str(exc)))
            return

        species_text = self._species_summary(
            summary["species"],
            summary["weights"],
        )
        input_count = self._dataset_count(self.dataset)
        output_count = input_count * summary["structure_count"]
        common = self.tr(
            "insert {count} × {species} · {inputs} inputs × {per_input} = {total} outputs · minimum distance {distance} Å"  # noqa: E501
        ).format(
            count=summary["count"],
            species=species_text,
            inputs=input_count,
            per_input=summary["structure_count"],
            total=output_count,
            distance=f"{summary['min_distance']:.3g}",
        )
        if summary["mode"] == 0:
            detail = self.tr("random positions inside the cell")
        else:
            axis_name = ("a", "b", "c")[summary["axis"]]
            detail = self.tr("upper surface along lattice {axis} · height {height} Å").format(
                axis=axis_name,
                height=f"{summary['offset']:.3g}",
            )
        self.preview_label.setText(
            self.tr("First input: {atoms} atoms · {detail} · {common}").format(
                atoms=len(self._input_structure),
                detail=detail,
                common=common,
            )
        )

    def _update_tab_order(self) -> None:
        if not hasattr(self, "adsorption_controls"):
            return
        widgets = [
            self.mode_combo,
            self.species_edit,
        ]
        if self.mode_combo.currentData() == 1:
            widgets.extend(
                [
                    self.axis_combo,
                    self.offset_frame.object_list[0],
                ]
            )
        widgets.extend(
            [
                self.insert_count_frame.object_list[0],
                self.structures_frame.object_list[0],
                self.min_distance_frame.object_list[0],
                self.seed_checkbox,
            ]
        )
        if self.seed_frame.isEnabled():
            widgets.append(self.seed_frame.object_list[0])
        widgets.append(self.advanced_checkbox)
        if self.advanced_checkbox.isChecked():
            widgets.append(self.max_attempts_frame.object_list[0])
        self.tab_order_widgets = widgets

    def create_operation(self):
        """Return the UI-independent insertion operation."""
        return InsertDefectOperation()

    def get_params(self) -> InsertDefectParams:
        """Read insertion parameters from UI controls."""
        return InsertDefectParams(
            mode=int(self.mode_combo.currentData()),
            species=self.species_edit.text(),
            insert_count=int(self.insert_count_frame.get_input_value()[0]),
            structure_count=int(self.structures_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            max_attempts=int(self.max_attempts_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            axis=int(self.axis_combo.currentData()),
            offset=float(self.offset_frame.get_input_value()[0]),
        )

    def set_params(self, params: InsertDefectParams) -> None:
        """Apply insertion parameters to UI controls."""
        mode_index = self.mode_combo.findData(int(params.mode))
        self.mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self.species_edit.setText(str(params.species))
        self.insert_count_frame.set_input_value([int(params.insert_count)])
        self.structures_frame.set_input_value([int(params.structure_count)])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.max_attempts_frame.set_input_value([int(params.max_attempts)])
        self.advanced_checkbox.setChecked(int(params.max_attempts) != 200)
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        axis_index = self.axis_combo.findData(int(params.axis))
        self.axis_combo.setCurrentIndex(axis_index if axis_index >= 0 else 2)
        self.offset_frame.set_input_value([float(params.offset)])
        self._update_mode_visibility(self.mode_combo.currentIndex())
        self._on_seed_changed()
        self._refresh_preview()

    def get_summary_text(self) -> str:
        params = self.get_params()
        mode = self.tr("Upper adsorption") if params.mode == 1 else self.tr("Bulk interstitial")
        return self.tr("{mode} · {atoms} atoms × {outputs}/input").format(
            mode=mode,
            atoms=params.insert_count,
            outputs=params.structure_count,
        )

    def get_guidance_text(self) -> str:
        params = self.get_params()
        input_count = self._dataset_count(self.dataset)
        if input_count:
            total = input_count * params.structure_count
            count_text = self.tr("{inputs} inputs × {outputs} outputs/input = {total} outputs.").format(
                inputs=input_count,
                outputs=params.structure_count,
                total=total,
            )
        else:
            count_text = self.tr("Load input structures; each input produces {outputs} outputs.").format(
                outputs=params.structure_count
            )
        mode_text = (
            self.tr("Choose the slab vacuum direction and adsorption height.")
            if params.mode == 1
            else self.tr("Bulk mode samples the full cell, including slab vacuum.")
        )
        return f"{count_text} {mode_text}"

    def process_structure(self, structure):
        """Insert atoms according to the current configuration."""
        return self.create_operation().run_structure(
            structure,
            self.get_params(),
        )

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data):
        """Restore current or legacy insertion parameters."""
        super().from_dict(data)
        raw_params = data.get("params")
        if raw_params is not None:
            params = InsertDefectParams(
                mode=raw_params.get("mode", 0),
                species=raw_params.get("species", ""),
                insert_count=raw_params.get("insert_count", 1),
                structure_count=raw_params.get("structure_count", 10),
                min_distance=raw_params.get("min_distance", 1.4),
                max_attempts=raw_params.get("max_attempts", 200),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
                axis=raw_params.get("axis", 2),
                offset=raw_params.get("offset", 1.5),
            )
        else:
            params = InsertDefectParams(
                mode=self._legacy_scalar(data.get("mode", 0), 0),
                species=data.get("species", ""),
                insert_count=self._legacy_scalar(
                    data.get("insert_count", 1),
                    1,
                ),
                structure_count=self._legacy_scalar(
                    data.get("structure_count", 10),
                    10,
                ),
                min_distance=self._legacy_scalar(
                    data.get("min_distance", 1.4),
                    1.4,
                ),
                max_attempts=self._legacy_scalar(
                    data.get("max_attempts", 200),
                    200,
                ),
                use_seed=data.get("use_seed", False),
                seed=self._legacy_scalar(data.get("seed", 0), 0),
                axis=self._legacy_scalar(data.get("axis", 2), 2),
                offset=self._legacy_scalar(data.get("offset", 1.5), 1.5),
            )
        self.set_params(params)

    @staticmethod
    def _legacy_scalar(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default
