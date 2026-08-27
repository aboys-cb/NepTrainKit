"""Card for sampling global vacancy patterns by count or fraction."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import VacancyDefectOperation, VacancyDefectParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class VacancyDefectCard(MakeDataCard):
    """Sample global vacancy patterns without distinguishing elements."""

    group = "Defect"
    # Keep the serialized card identity stable; only the displayed title is shorter.
    card_name = "Vacancy Defect Generation"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Global Vacancy"))
        self.init_ui()

    def init_ui(self):
        """Build a mode-driven inspector that exposes only active values."""
        self.setObjectName("vacancy_defect_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.amount_mode_control = SegmentedControl(parent=self.setting_widget)
        self.amount_mode_control.addItem(self.tr("Count"), userData="count")
        self.amount_mode_control.addItem(self.tr("Fraction"), userData="fraction")
        self.amount_mode_control.setAccessibleName(self.tr("Vacancy amount basis"))
        amount_mode_field = CompactField(self.tr("Amount basis"), self.amount_mode_control, self.setting_widget)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input(self.tr("atoms"), 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([1])
        self.num_condition_frame.setFixedWidth(144)
        self.num_condition_frame.setAccessibleName(self.tr("Vacancies"))
        self.num_condition_field = CompactField(
            self.tr("Vacancies"),
            self.num_condition_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )

        self.concentration_condition_frame = SpinBoxUnitInputFrame(self)
        self.concentration_condition_frame.set_input("", 1, "float")
        self.concentration_condition_frame.setDecimals(6)
        self.concentration_condition_frame.setSingleStep(0.001)
        self.concentration_condition_frame.setRange(0.000001, 0.999999)
        self.concentration_condition_frame.set_input_value([0.01])
        self.concentration_condition_frame.setFixedWidth(144)
        self.concentration_condition_frame.setAccessibleName(self.tr("Vacancy fraction"))
        self.concentration_condition_field = CompactField(
            self.tr("Vacancy fraction"),
            self.concentration_condition_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )

        amount_section = InspectorSection(
            self.tr("Vacancy amount"),
            self.setting_widget,
            self.tr(
                "Choose an absolute number or a fraction of all atoms. "
                "Fractions are multiplied by the input atom count and rounded down."
            ),
        )
        amount_grid = ResponsiveFormGrid(amount_section)
        amount_grid.add_field(amount_mode_field, span=2)
        amount_grid.add_field(self.num_condition_field, span=2)
        amount_grid.add_field(self.concentration_condition_field, span=2)
        amount_section.addWidget(amount_grid)

        self.count_mode_control = SegmentedControl(parent=self.setting_widget)
        self.count_mode_control.addItem(self.tr("Fixed"), userData="fixed")
        self.count_mode_control.addItem(self.tr("Variable"), userData="random")
        self.count_mode_control.setAccessibleName(self.tr("Vacancies per output"))
        count_mode_field = CompactField(
            self.tr("Vacancies per output"),
            self.count_mode_control,
            self.setting_widget,
            self.tr(
                "Fixed uses the resolved amount. Variable samples an integer from 1 to that amount for each output."
            ),
        )

        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("", 1, "int")
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_condition_frame.set_input_value([1])
        self.max_atoms_condition_frame.setFixedWidth(144)
        self.max_atoms_condition_frame.setAccessibleName(self.tr("Maximum outputs per input"))
        max_outputs_field = CompactField(
            self.tr("Maximum outputs per input"),
            self.max_atoms_condition_frame,
            self.setting_widget,
            self.tr("Duplicate deletion patterns are removed."),
            inline=True,
            input_max_width=144,
        )

        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(count_mode_field, span=2)
        generation_grid.add_field(max_outputs_field, span=2)
        generation_section.addWidget(generation_grid)

        self.engine_type_combo = ComboBox(self.setting_widget)
        self.engine_type_combo.addItem(self.tr("Uniform"), userData=1)
        self.engine_type_combo.addItem(self.tr("Sobol"), userData=0)
        self.engine_type_combo.setCurrentIndex(0)
        self.engine_type_combo.setAccessibleName(self.tr("Site sampling"))
        engine_field = CompactField(
            self.tr("Site sampling"),
            self.engine_type_combo,
            self.setting_widget,
            self.tr(
                "Uniform is the general default. Sobol gives quasi-random coverage and supports up to 21,200 atoms."
            ),
        )

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setFixedWidth(144)
        self.seed_frame.setAccessibleName(self.tr("Random seed"))
        seed_row = QWidget(self.setting_widget)
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(8)
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_frame)
        seed_layout.addStretch(1)

        sampling_section = InspectorSection(self.tr("Sampling and reproducibility"), self.setting_widget)
        sampling_grid = ResponsiveFormGrid(sampling_section)
        sampling_grid.add_field(engine_field, span=2)
        sampling_grid.add_field(seed_row, span=2)
        sampling_section.addWidget(sampling_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setObjectName("vacancyDefectPreview")
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(amount_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(sampling_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 3, 0, 1, 3)

        self.amount_mode_control.currentIndexChanged.connect(self._on_amount_mode_changed)
        self.count_mode_control.currentIndexChanged.connect(self._on_count_mode_changed)
        self.engine_type_combo.currentIndexChanged.connect(self._refresh_preview)
        self.max_atoms_condition_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.num_condition_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.concentration_condition_frame.object_list[0].valueChanged.connect(self._refresh_preview)
        self.seed_checkbox.toggled.connect(self._on_seed_changed)
        self._on_amount_mode_changed()
        self._on_count_mode_changed()
        self._on_seed_changed()

    def _on_amount_mode_changed(self, *_args) -> None:
        use_count = self.amount_mode_control.currentData() == "count"
        self.num_condition_field.setVisible(use_count)
        self.concentration_condition_field.setVisible(not use_count)
        self._update_amount_labels()
        self._refresh_preview()
        self._update_tab_order()

    def _on_count_mode_changed(self, *_args) -> None:
        self._update_amount_labels()
        self._refresh_preview()
        self._update_tab_order()

    def _update_amount_labels(self) -> None:
        variable = self.count_mode_control.currentData() == "random"
        self.num_condition_field.set_label(self.tr("Maximum vacancies") if variable else self.tr("Vacancies"))
        self.concentration_condition_field.set_label(
            self.tr("Maximum vacancy fraction") if variable else self.tr("Vacancy fraction")
        )

    def _on_seed_changed(self, *_args) -> None:
        enabled = self.seed_checkbox.isChecked()
        self.seed_frame.setEnabled(enabled)
        self.seed_frame.setVisible(enabled)
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
            self.preview_label.setText(self.tr("Load an upstream structure to preview the resolved vacancy count."))
            return
        try:
            summary = self.create_operation().sampling_summary(self._input_structure, self.get_params())
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=translate_runtime_message(exc))
            )
            return

        if summary["min_defects"] == summary["max_defects"]:
            vacancy_text = self.tr("remove {count} atoms").format(count=summary["max_defects"])
        else:
            vacancy_text = self.tr("remove {minimum}–{maximum} atoms").format(
                minimum=summary["min_defects"], maximum=summary["max_defects"]
            )
        self.preview_label.setText(
            self.tr(
                "First input: {atoms} atoms · {vacancies} · up to {outputs} unique outputs · all elements eligible"
            ).format(
                atoms=summary["n_atoms"],
                vacancies=vacancy_text,
                outputs=summary["target_outputs"],
            )
        )

    def _update_tab_order(self) -> None:
        widgets = [self.amount_mode_control]
        widgets.extend(
            self.num_condition_frame.object_list
            if self.amount_mode_control.currentData() == "count"
            else self.concentration_condition_frame.object_list
        )
        widgets.extend(
            [
                self.count_mode_control,
                self.max_atoms_condition_frame.object_list[0],
                self.engine_type_combo,
                self.seed_checkbox,
            ]
        )
        if self.seed_checkbox.isChecked():
            widgets.append(self.seed_frame.object_list[0])
        self.tab_order_widgets = widgets

    def create_operation(self):
        return VacancyDefectOperation()

    def get_params(self) -> VacancyDefectParams:
        return VacancyDefectParams(
            engine_type=int(self.engine_type_combo.currentData()),
            num_condition=int(self.num_condition_frame.get_input_value()[0]),
            use_num=self.amount_mode_control.currentData() == "count",
            concentration_condition=float(self.concentration_condition_frame.get_input_value()[0]),
            count_mode=str(self.count_mode_control.currentData() or "fixed"),
            max_structures=int(self.max_atoms_condition_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: VacancyDefectParams) -> None:
        engine_index = self.engine_type_combo.findData(int(params.engine_type))
        self.engine_type_combo.setCurrentIndex(engine_index if engine_index >= 0 else 0)
        self.num_condition_frame.set_input_value([int(params.num_condition)])
        self.concentration_condition_frame.set_input_value([float(params.concentration_condition)])
        self.max_atoms_condition_frame.set_input_value([int(params.max_structures)])
        self.amount_mode_control.setCurrentIndex(0 if params.use_num else 1)
        count_mode_index = self.count_mode_control.findData(str(params.count_mode))
        self.count_mode_control.setCurrentIndex(count_mode_index if count_mode_index >= 0 else 0)
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._on_amount_mode_changed()
        self._on_count_mode_changed()
        self._on_seed_changed()

    def get_summary_text(self) -> str:
        params = self.get_params()
        amount = (
            self.tr("{count} vacancies").format(count=params.num_condition)
            if params.use_num
            else self.tr("fraction {fraction}").format(fraction=f"{params.concentration_condition:.6g}")
        )
        mode = self.tr("fixed") if params.count_mode == "fixed" else self.tr("variable")
        return self.tr("{amount} · {mode} · up to {outputs} outputs").format(
            amount=amount, mode=mode, outputs=params.max_structures
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "All elements are eligible. Use Targeted Vacancy when deletion must be "
            "restricted by element or existing group labels."
        )

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        """Restore card settings from current or legacy serialized values."""
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = VacancyDefectParams(
                engine_type=raw_params.get("engine_type", 1),
                num_condition=raw_params.get("num_condition", 1),
                use_num=raw_params.get("use_num", True),
                concentration_condition=raw_params.get("concentration_condition", 0.01),
                count_mode=raw_params.get("count_mode", "random"),
                max_structures=raw_params.get("max_structures", 1),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = VacancyDefectParams(
                engine_type=self._legacy_scalar(data_dict.get("engine_type", 1), 1),
                num_condition=self._legacy_scalar(data_dict.get("num_condition", 1), 1),
                use_num=data_dict.get("num_radio_button", True),
                concentration_condition=self._legacy_scalar(data_dict.get("concentration_condition", 0.01), 0.01),
                count_mode=data_dict.get("count_mode", "random"),
                max_structures=self._legacy_scalar(data_dict.get("max_atoms_condition", 1), 1),
                use_seed=data_dict.get("use_seed", False),
                seed=self._legacy_scalar(data_dict.get("seed", 0), 0),
            )
        self.set_params(params)

    @staticmethod
    def _legacy_scalar(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default
