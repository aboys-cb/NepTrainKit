"""Card for inserting interstitial and adsorbate species into structures."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import InsertDefectOperation, InsertDefectParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


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

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip("Enable reproducible random sampling")
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))
        self.settingLayout.addWidget(self.seed_checkbox, row, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, row, 1, 1, 2)
        row += 1

        # Adsorption-specific controls
        self.axis_label = BodyLabel("Surface axis", self.setting_widget)
        self.axis_label.setToolTip("Crystal axis treated as surface normal for adsorption")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_combo = ComboBox(self.setting_widget)
        self.axis_combo.addItems(["a (x)", "b (y)", "c (z)"])
        self.axis_combo.setCurrentIndex(2)
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

    def create_operation(self):
        """Return the UI-independent insertion operation."""
        return InsertDefectOperation()

    def get_params(self) -> InsertDefectParams:
        """Read insertion parameters from UI controls."""
        return InsertDefectParams(
            mode=self.mode_combo.currentIndex(),
            species=self.species_edit.text(),
            insert_count=int(self.insert_count_frame.get_input_value()[0]),
            structure_count=int(self.structures_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            max_attempts=int(self.max_attempts_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            axis=self.axis_combo.currentIndex(),
            offset=float(self.offset_frame.get_input_value()[0]),
        )

    def set_params(self, params: InsertDefectParams) -> None:
        """Apply insertion parameters to UI controls."""
        self.mode_combo.setCurrentIndex(int(params.mode))
        self.species_edit.setText(params.species)
        self.insert_count_frame.set_input_value([int(params.insert_count)])
        self.structures_frame.set_input_value([int(params.structure_count)])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.max_attempts_frame.set_input_value([int(params.max_attempts)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.axis_combo.setCurrentIndex(int(params.axis))
        self.offset_frame.set_input_value([float(params.offset)])
        self._update_mode_visibility(self.mode_combo.currentIndex())

    def process_structure(self, structure):
        """Insert atoms according to the current configuration."""
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        params = self.get_params()
        data.update(
            {
                "params": params_to_dict(params),
                "mode": params.mode,
                "species": params.species,
                "insert_count": [params.insert_count],
                "structure_count": [params.structure_count],
                "min_distance": [params.min_distance],
                "max_attempts": [params.max_attempts],
                "use_seed": params.use_seed,
                "seed": [params.seed],
                "axis": params.axis,
                "offset": [params.offset],
            }
        )
        return data

    def from_dict(self, data):
        super().from_dict(data)
        raw_params = data.get("params")
        if raw_params:
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
                mode=data.get("mode", 0),
                species=data.get("species", ""),
                insert_count=data.get("insert_count", [1])[0],
                structure_count=data.get("structure_count", [10])[0],
                min_distance=data.get("min_distance", [1.4])[0],
                max_attempts=data.get("max_attempts", [200])[0],
                use_seed=data.get("use_seed", False),
                seed=data.get("seed", [0])[0],
                axis=data.get("axis", 2),
                offset=data.get("offset", [1.5])[0],
            )
        self.set_params(params)
