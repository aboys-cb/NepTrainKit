"""Card for generating lattice perturbations via stochastic scaling."""

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import CellScalingOperation, CellScalingParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class CellScalingCard(MakeDataCard):
    """Generate stochastic lattice-length and optional angle variations."""

    group = "Lattice"
    card_name = "Lattice Perturb"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Lattice Perturb"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("cell_scaling_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.engine_type_combo = SegmentedControl(parent=self.setting_widget)
        self.engine_type_combo.addItem(self.tr("Sobol"), userData=0)
        self.engine_type_combo.addItem(self.tr("Uniform"), userData=1)
        self.engine_type_combo.setCurrentIndex(1)
        engine_field = CompactField(
            self.tr("Sampling sequence"),
            self.engine_type_combo,
            self.setting_widget,
            self.tr(
                "Uniform gives independent random samples. Sobol spreads small sample sets more evenly; 32, 64, … outputs are preferred."
            ),
        )

        # The serialized parameter remains a fraction: 4% <-> max_scaling=0.04.
        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("%", 1, "float")
        self.scaling_condition_frame.setDecimals(4)
        self.scaling_condition_frame.setSingleStep(0.5)
        self.scaling_condition_frame.setRange(0, 20)
        self.scaling_condition_frame.set_input_value([4.0])
        self.scaling_condition_frame.setFixedWidth(132)
        scaling_field = CompactField(
            self.tr("Maximum relative change"),
            self.scaling_condition_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )

        self.perturb_angle_checkbox = CheckBox(
            self.tr("Also vary cell angles"), self.setting_widget
        )
        self.perturb_angle_checkbox.setChecked(True)
        self.organic_checkbox = CheckBox(
            self.tr("Keep detected molecules rigid"), self.setting_widget
        )
        self.organic_checkbox.setChecked(False)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("", 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([50])
        self.num_condition_frame.setFixedWidth(132)
        num_field = CompactField(
            self.tr("Outputs per input"),
            self.num_condition_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setFixedWidth(132)
        self.seed_frame.setEnabled(False)
        self.seed_frame.hide()
        seed_row = QWidget(self.setting_widget)
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(6)
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_frame)
        seed_layout.addStretch(1)

        sampling_section = InspectorSection(
            self.tr("Lattice sampling"),
            self.setting_widget,
            self.tr(
                "Each lattice length is multiplied independently within the selected ± range."
            ),
        )
        sampling_grid = ResponsiveFormGrid(sampling_section)
        sampling_grid.add_field(engine_field, span=2)
        sampling_grid.add_field(scaling_field, span=2)
        sampling_grid.add_field(self.perturb_angle_checkbox, span=2)
        sampling_section.addWidget(sampling_grid)

        molecule_section = InspectorSection(
            self.tr("Molecular handling"),
            self.setting_widget,
            self.tr(
                "After the cell changes, restore the internal geometry of detected molecular clusters."
            ),
        )
        molecule_section.addWidget(self.organic_checkbox)

        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(num_field, span=2)
        generation_grid.add_field(seed_row, span=2)
        generation_section.addWidget(generation_grid)

        self.settingLayout.addWidget(sampling_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(molecule_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 2, 0, 1, 3)

        self.engine_type_combo.currentIndexChanged.connect(
            self.refresh_compact_presentation
        )
        for control in (
            self.scaling_condition_frame.object_list
            + self.num_condition_frame.object_list
        ):
            control.valueChanged.connect(self.refresh_compact_presentation)
        self.perturb_angle_checkbox.toggled.connect(self.refresh_compact_presentation)
        self.organic_checkbox.toggled.connect(self.refresh_compact_presentation)
        self.seed_checkbox.toggled.connect(self._on_seed_changed)
        for control in self.seed_frame.object_list:
            control.valueChanged.connect(self.refresh_compact_presentation)

    def _on_seed_changed(self, checked: bool | None = None) -> None:
        enabled = self.seed_checkbox.isChecked() if checked is None else bool(checked)
        self.seed_frame.setEnabled(enabled)
        self.seed_frame.setVisible(enabled)
        self.refresh_compact_presentation()

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

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        engine = self.tr("Sobol") if params.engine_type == 0 else self.tr("Uniform")
        target = (
            self.tr("lengths + angles")
            if params.perturb_angle
            else self.tr("lengths")
        )
        return self.tr("{engine} · {target} · ±{percent:g}% · {count}/input").format(
            engine=engine,
            target=target,
            percent=params.max_scaling * 100.0,
            count=params.max_num,
        )

    def get_guidance_text(self) -> str:
        params = self.get_params()
        input_count = getattr(self, "_preview_input_count", None)
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        if input_count is None:
            count_text = self.tr("Each input produces exactly {count} structures.").format(
                count=params.max_num
            )
        else:
            count_text = self.tr("{inputs} × {count} = {total} outputs.").format(
                inputs=input_count,
                count=params.max_num,
                total=input_count * params.max_num,
            )
        angle_text = (
            self.tr("Cell angles use the same relative ± range.")
            if params.perturb_angle
            else self.tr("Cell angles stay unchanged.")
        )
        notes = [count_text, angle_text]
        if params.engine_type == 0 and params.max_num & (params.max_num - 1):
            notes.append(
                self.tr("Sobol balance is best with 4, 8, 16, 32, 64, … outputs.")
            )
        return " ".join(notes)

    def create_operation(self):
        return CellScalingOperation()

    def get_params(self) -> CellScalingParams:
        return CellScalingParams(
            engine_type=int(self.engine_type_combo.currentIndex()),
            max_scaling=float(self.scaling_condition_frame.get_input_value()[0])
            / 100.0,
            max_num=int(self.num_condition_frame.get_input_value()[0]),
            perturb_angle=self.perturb_angle_checkbox.isChecked(),
            identify_organic=self.organic_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CellScalingParams) -> None:
        self.engine_type_combo.setCurrentIndex(int(params.engine_type))
        self.perturb_angle_checkbox.setChecked(bool(params.perturb_angle))
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        percent = float(params.max_scaling) * 100.0
        if percent > 20.0:
            # Do not silently clamp an older workflow that used the former
            # 0..1 fraction control. Core validation still rejects bad cells.
            self.scaling_condition_frame.setRange(0, percent)
        self.scaling_condition_frame.set_input_value([percent])
        self.num_condition_frame.set_input_value([int(params.max_num)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._on_seed_changed()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = CellScalingParams(
                engine_type=raw_params.get("engine_type", 1),
                max_scaling=raw_params.get("max_scaling", 0.04),
                max_num=raw_params.get("max_num", 50),
                perturb_angle=raw_params.get("perturb_angle", True),
                identify_organic=raw_params.get("identify_organic", False),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = CellScalingParams(
                engine_type=data_dict.get("engine_type", 1),
                max_scaling=data_dict.get("scaling_condition", [0.04])[0],
                max_num=data_dict.get("num_condition", [50])[0],
                perturb_angle=data_dict.get("perturb_angle", True),
                identify_organic=data_dict.get("organic", False),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
