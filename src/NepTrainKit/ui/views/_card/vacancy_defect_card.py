"""Card for sampling global vacancy patterns by count or fraction."""

from PySide6.QtCore import Qt
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    RadioButton,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import VacancyDefectOperation, VacancyDefectParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class VacancyDefectCard(MakeDataCard):
    """Sample global vacancy patterns without distinguishing elements."""

    group = "Defect"
    card_name = "Vacancy Defect Generation"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Global Random Vacancy"))
        self.init_ui()

    def init_ui(self):
        """Build amount, distribution, sampling, seed, and preview controls."""
        self.setObjectName("vacancy_defect_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.num_radio_button = RadioButton(
            self.tr("Vacancy count"),
            self.setting_widget,
        )
        self.num_radio_button.setChecked(True)
        self.num_radio_button.setToolTip(
            self.tr("Use an absolute vacancy count for every input structure")
        )
        self._install_tooltip(self.num_radio_button)
        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("", 1)
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([1])
        self.num_condition_frame.setAccessibleName(self.tr("Vacancy count"))

        self.concentration_radio_button = RadioButton(
            self.tr("Vacancy fraction (0–1)"),
            self.setting_widget,
        )
        self.concentration_radio_button.setToolTip(
            self.tr("The atom count multiplied by this fraction is rounded down")
        )
        self._install_tooltip(self.concentration_radio_button)
        self.concentration_condition_frame = SpinBoxUnitInputFrame(self)
        self.concentration_condition_frame.set_input("", 1, "float")
        self.concentration_condition_frame.setDecimals(6)
        self.concentration_condition_frame.setSingleStep(0.001)
        self.concentration_condition_frame.setRange(0.000001, 0.999999)
        self.concentration_condition_frame.set_input_value([0.01])
        self.concentration_condition_frame.setAccessibleName(
            self.tr("Vacancy fraction (0–1)")
        )

        self.count_mode_label = BodyLabel(
            self.tr("Vacancies per output"),
            self.setting_widget,
        )
        self.count_mode_label.setToolTip(
            self.tr("Use the resolved amount exactly, or sample from 1 up to that amount")
        )
        self._install_tooltip(self.count_mode_label)
        self.count_mode_combo = ComboBox(self.setting_widget)
        self.count_mode_combo.addItem(
            self.tr("Fixed at the set value"),
            userData="fixed",
        )
        self.count_mode_combo.addItem(
            self.tr("Random from 1 to the set value"),
            userData="random",
        )
        self.count_mode_combo.setCurrentIndex(0)
        self.count_mode_combo.setAccessibleName(self.tr("Vacancies per output"))

        self.max_atoms_label = BodyLabel(
            self.tr("Maximum outputs per input"),
            self.setting_widget,
        )
        self.max_atoms_label.setToolTip(
            self.tr("Duplicate vacancy placements are removed, so the actual count can be lower")
        )
        self._install_tooltip(self.max_atoms_label)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("", 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_condition_frame.set_input_value([1])
        self.max_atoms_condition_frame.setAccessibleName(
            self.tr("Maximum outputs per input")
        )

        self.engine_label = BodyLabel(
            self.tr("Site sampling"),
            self.setting_widget,
        )
        self.engine_label.setToolTip(
            self.tr("Uniform is the general default; Sobol gives quasi-random coverage for up to 21,200 atoms")
        )
        self._install_tooltip(self.engine_label)
        self.engine_type_combo = ComboBox(self.setting_widget)
        self.engine_type_combo.addItem(
            self.tr("Uniform random (recommended)"),
            userData=1,
        )
        self.engine_type_combo.addItem(
            self.tr("Sobol quasi-random"),
            userData=0,
        )
        self.engine_type_combo.setCurrentIndex(0)
        self.engine_type_combo.setAccessibleName(self.tr("Site sampling"))

        self.seed_checkbox = CheckBox(
            self.tr("Use seed"),
            self.setting_widget,
        )
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip(
            self.tr("Enable reproducible per-structure vacancy sampling")
        )
        self._install_tooltip(self.seed_checkbox)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_frame.setAccessibleName(self.tr("Random seed"))

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("vacancyDefectPreview")

        self.settingLayout.addWidget(self.num_radio_button, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(
            self.concentration_radio_button,
            1,
            0,
            1,
            1,
        )
        self.settingLayout.addWidget(
            self.concentration_condition_frame,
            1,
            1,
            1,
            2,
        )
        self.settingLayout.addWidget(self.count_mode_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.count_mode_combo, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(
            self.max_atoms_condition_frame,
            3,
            1,
            1,
            2,
        )
        self.settingLayout.addWidget(self.engine_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.engine_type_combo, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.preview_label, 6, 0, 1, 3)

        self.num_radio_button.toggled.connect(self._on_amount_mode_changed)
        self.concentration_radio_button.toggled.connect(
            self._on_amount_mode_changed
        )
        self.count_mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.engine_type_combo.currentIndexChanged.connect(self._refresh_preview)
        self.max_atoms_condition_frame.object_list[0].valueChanged.connect(
            self._refresh_preview
        )
        self.num_condition_frame.object_list[0].valueChanged.connect(
            self._refresh_preview
        )
        self.concentration_condition_frame.object_list[0].valueChanged.connect(
            self._refresh_preview
        )
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self._on_amount_mode_changed()
        self._on_seed_changed()
        self._refresh_preview()

    @staticmethod
    def _install_tooltip(widget) -> None:
        widget.installEventFilter(
            ToolTipFilter(widget, 300, ToolTipPosition.TOP)
        )

    def _on_amount_mode_changed(self) -> None:
        use_count = self.num_radio_button.isChecked()
        self.num_condition_frame.setEnabled(use_count)
        self.concentration_condition_frame.setEnabled(not use_count)
        self._refresh_preview()
        self._update_tab_order()

    def _on_seed_changed(self) -> None:
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

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an upstream structure to preview the resolved vacancy count."
                )
            )
            return
        try:
            summary = self.create_operation().sampling_summary(
                self._input_structure,
                self.get_params(),
            )
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=str(exc))
            )
            return

        if summary["min_defects"] == summary["max_defects"]:
            vacancy_text = self.tr("remove {count} atoms").format(
                count=summary["max_defects"]
            )
        else:
            vacancy_text = self.tr("remove {minimum}–{maximum} atoms").format(
                minimum=summary["min_defects"],
                maximum=summary["max_defects"],
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
        widgets = [self.num_radio_button]
        if self.num_condition_frame.isEnabled():
            widgets.extend(self.num_condition_frame.object_list)
        widgets.append(self.concentration_radio_button)
        if self.concentration_condition_frame.isEnabled():
            widgets.extend(self.concentration_condition_frame.object_list)
        widgets.extend(
            [
                self.count_mode_combo,
                self.max_atoms_condition_frame.object_list[0],
                self.engine_type_combo,
                self.seed_checkbox,
            ]
        )
        if self.seed_frame.isEnabled():
            widgets.append(self.seed_frame.object_list[0])
        self.tab_order_widgets = widgets

    def create_operation(self):
        """Return the UI-independent vacancy-defect operation."""
        return VacancyDefectOperation()

    def get_params(self) -> VacancyDefectParams:
        """Read vacancy-defect parameters from UI controls."""
        return VacancyDefectParams(
            engine_type=int(self.engine_type_combo.currentData()),
            num_condition=int(self.num_condition_frame.get_input_value()[0]),
            use_num=self.num_radio_button.isChecked(),
            concentration_condition=float(
                self.concentration_condition_frame.get_input_value()[0]
            ),
            count_mode=str(self.count_mode_combo.currentData() or "fixed"),
            max_structures=int(
                self.max_atoms_condition_frame.get_input_value()[0]
            ),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: VacancyDefectParams) -> None:
        """Apply vacancy-defect parameters to UI controls."""
        engine_index = self.engine_type_combo.findData(int(params.engine_type))
        self.engine_type_combo.setCurrentIndex(
            engine_index if engine_index >= 0 else 0
        )
        self.num_condition_frame.set_input_value([int(params.num_condition)])
        self.concentration_condition_frame.set_input_value(
            [float(params.concentration_condition)]
        )
        self.max_atoms_condition_frame.set_input_value(
            [int(params.max_structures)]
        )
        self.num_radio_button.setChecked(bool(params.use_num))
        self.concentration_radio_button.setChecked(not bool(params.use_num))
        count_mode_index = self.count_mode_combo.findData(str(params.count_mode))
        self.count_mode_combo.setCurrentIndex(
            count_mode_index if count_mode_index >= 0 else 0
        )
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._on_amount_mode_changed()
        self._on_seed_changed()
        self._refresh_preview()

    def process_structure(self, structure):
        """Create globally sampled vacancy structures."""
        return self.create_operation().run_structure(
            structure,
            self.get_params(),
        )

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
                concentration_condition=raw_params.get(
                    "concentration_condition",
                    0.01,
                ),
                count_mode=raw_params.get("count_mode", "random"),
                max_structures=raw_params.get("max_structures", 1),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = VacancyDefectParams(
                engine_type=self._legacy_scalar(
                    data_dict.get("engine_type", 1),
                    1,
                ),
                num_condition=self._legacy_scalar(
                    data_dict.get("num_condition", 1),
                    1,
                ),
                use_num=data_dict.get("num_radio_button", True),
                concentration_condition=self._legacy_scalar(
                    data_dict.get("concentration_condition", 0.01),
                    0.01,
                ),
                count_mode=data_dict.get("count_mode", "random"),
                max_structures=self._legacy_scalar(
                    data_dict.get("max_atoms_condition", 1),
                    1,
                ),
                use_seed=data_dict.get("use_seed", False),
                seed=self._legacy_scalar(data_dict.get("seed", 0), 0),
            )
        self.set_params(params)

    @staticmethod
    def _legacy_scalar(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default
