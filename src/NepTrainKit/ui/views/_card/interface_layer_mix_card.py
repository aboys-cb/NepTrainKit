"""Card for interdiffusing near-interface layers of a bilayer (界面随机互混)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import (
    InterfaceLayerMixOperation,
    InterfaceLayerMixParams,
)
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class InterfaceLayerMixCard(MakeDataCard):
    """Detect a bilayer interface and swap species between near-interface layers."""

    group = "Alloy"
    card_name = "Interface Layer Mixing"
    description = (
        "Detect a bilayer interface, pick near-interface layers on both sides, "
        "and swap their atom species at a target or gradient concentration."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "Sun Xiaojian", "role": "author"},
        {"name": "NepTrainKit", "role": "maintainer"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Interface Layer Mixing"))
        self.init_ui()

    def init_ui(self):
        """Build the interface, layer, concentration, and seed controls."""
        self.setObjectName("interface_layer_mix_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(5)
        self.settingLayout.setColumnStretch(1, 1)

        self.axis_label = BodyLabel(self.tr("Interface normal"), self.setting_widget)
        self.axis_label.setToolTip(self.tr("Lattice axis perpendicular to the bilayer interface"))
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.axis_combo,
            [
                ("auto", "Auto detect"),
                ("a", "Lattice a"),
                ("b", "Lattice b"),
                ("c", "Lattice c"),
            ],
        )
        set_combo_value(self.axis_combo, "auto")
        self.axis_hint_label = CaptionLabel("", self.setting_widget)
        self.axis_hint_label.setWordWrap(True)
        self.axis_hint_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.left_layers_label = BodyLabel(self.tr("L-side layers from interface"), self.setting_widget)
        self.left_layers_label.setToolTip(
            self.tr("Number of atomic layers selected below the interface")
        )
        self.left_layers_label.installEventFilter(
            ToolTipFilter(self.left_layers_label, 300, ToolTipPosition.TOP)
        )
        self.left_layers_frame = SpinBoxUnitInputFrame(self)
        self.left_layers_frame.set_input("", 1, "int")
        self.left_layers_frame.setRange(1, 100)
        self.left_layers_frame.set_input_value([2])

        self.right_layers_label = BodyLabel(self.tr("R-side layers from interface"), self.setting_widget)
        self.right_layers_label.setToolTip(
            self.tr("Number of atomic layers selected above the interface")
        )
        self.right_layers_label.installEventFilter(
            ToolTipFilter(self.right_layers_label, 300, ToolTipPosition.TOP)
        )
        self.right_layers_frame = SpinBoxUnitInputFrame(self)
        self.right_layers_frame.set_input("", 1, "int")
        self.right_layers_frame.setRange(1, 100)
        self.right_layers_frame.set_input_value([2])
        self.side_note_label = CaptionLabel(
            self.tr("Relative to the interface, the L side is below and the R side is above."),
            self.setting_widget,
        )
        self.side_note_label.setWordWrap(True)
        self.side_note_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.mode_label = BodyLabel(self.tr("Concentration mode"), self.setting_widget)
        self.mode_label.setToolTip(
            self.tr("Fixed target concentration, or a linear gradient across the generated structures")
        )
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("fixed", "Fixed concentration"),
                ("gradient", "Gradient concentration"),
            ],
        )
        set_combo_value(self.mode_combo, "fixed")

        self.concentration_label = BodyLabel(self.tr("Target concentration"), self.setting_widget)
        self.concentration_label.setToolTip(
            self.tr("Fraction of the selected-layer atoms that exchange species across the interface")
        )
        self.concentration_label.installEventFilter(
            ToolTipFilter(self.concentration_label, 300, ToolTipPosition.TOP)
        )
        self.concentration_frame = SpinBoxUnitInputFrame(self)
        self.concentration_frame.set_input("%", 1, "float")
        self.concentration_frame.setRange(0.0, 100.0)
        self.concentration_frame.setDecimals(6)
        self.concentration_frame.set_input_value([50.0])

        self.gradient_start_frame = SpinBoxUnitInputFrame(self)
        self.gradient_start_frame.set_input("~", 1, "float")
        self.gradient_start_frame.setRange(0.0, 100.0)
        self.gradient_start_frame.setDecimals(6)
        self.gradient_start_frame.set_input_value([0.0])

        self.gradient_end_frame = SpinBoxUnitInputFrame(self)
        self.gradient_end_frame.set_input("%", 1, "float")
        self.gradient_end_frame.setRange(0.0, 100.0)
        self.gradient_end_frame.setDecimals(6)
        self.gradient_end_frame.set_input_value([100.0])

        self.gradient_container = QWidget(self.setting_widget)
        self.gradient_layout = QHBoxLayout(self.gradient_container)
        self.gradient_layout.setContentsMargins(0, 0, 0, 0)
        self.gradient_layout.setSpacing(0)
        self.gradient_layout.addWidget(self.gradient_start_frame)
        self.gradient_layout.addWidget(self.gradient_end_frame)
        self.gradient_layout.addStretch(1)
        # zero the frames' internal gaps so the boxes sit back to back, and
        # align the digits toward the separator so 0.000000~100.000000% reads flush.
        # Cap both boxes to 70% of the end box's width so the 6-decimal
        # start box stays the same length as the end box instead of being
        # stretched wider by its "100.000000" size hint.
        self.gradient_start_frame.layout().setSpacing(0)
        self.gradient_end_frame.layout().setSpacing(0)
        _end_box = self.gradient_end_frame.object_list[0]
        _box_width = int(_end_box.sizeHint().width() * 0.7)
        self.gradient_start_frame.object_list[0].setFixedWidth(_box_width)
        _end_box.setFixedWidth(_box_width)
        self.gradient_start_frame.object_list[0].setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.gradient_end_frame.object_list[0].setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.num_structures_label = BodyLabel(self.tr("Number of structures"), self.setting_widget)
        self.num_structures_label.setToolTip(
            self.tr("Number of new structures generated for each input structure")
        )
        self.num_structures_label.installEventFilter(
            ToolTipFilter(self.num_structures_label, 300, ToolTipPosition.TOP)
        )
        self.num_structures_frame = SpinBoxUnitInputFrame(self)
        self.num_structures_frame.set_input("", 1, "int")
        self.num_structures_frame.setRange(1, 10000)
        self.num_structures_frame.set_input_value([1])

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip(self.tr("Enable reproducible random swapping"))
        self.seed_checkbox.installEventFilter(
            ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP)
        )
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_frame.setAccessibleName(self.tr("Random seed"))

        self.summary_label = CaptionLabel("", self.setting_widget)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.settingLayout.addWidget(self.axis_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_hint_label, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.left_layers_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.left_layers_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.right_layers_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.right_layers_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.side_note_label, 4, 0, 1, 3)
        self.settingLayout.addWidget(self.mode_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.concentration_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.concentration_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.gradient_container, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.num_structures_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.num_structures_frame, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.summary_label, 9, 0, 1, 3)

        self.axis_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.left_layers_frame.object_list[0].valueChanged.connect(self._on_controls_changed)
        self.right_layers_frame.object_list[0].valueChanged.connect(self._on_controls_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.concentration_frame.object_list[0].valueChanged.connect(self._on_controls_changed)
        self.gradient_start_frame.object_list[0].valueChanged.connect(self._on_controls_changed)
        self.gradient_end_frame.object_list[0].valueChanged.connect(self._on_controls_changed)
        self.num_structures_frame.object_list[0].valueChanged.connect(self._on_controls_changed)
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self._on_controls_changed()

    def _on_controls_changed(self) -> None:
        is_gradient = combo_value(self.mode_combo) == "gradient"
        self.concentration_frame.setVisible(not is_gradient)
        self.gradient_container.setVisible(is_gradient)
        self._update_axis_hint()
        self._update_summary()

    def _update_axis_hint(self) -> None:
        axis = combo_value(self.axis_combo)
        if axis == "auto":
            self.axis_hint_label.setText(
                self.tr("Auto: the axis with the sharpest composition split is chosen.")
            )
        else:
            self.axis_hint_label.setText(
                self.tr("Atoms are layered along lattice {axis}.").format(axis=axis)
            )

    def _on_seed_changed(self) -> None:
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._update_summary()

    def _update_summary(self) -> None:
        """Live, exact preview of the detected interface and output count."""
        first = None
        if self.dataset is not None:
            try:
                first = self.dataset[0]
            except (TypeError, IndexError, KeyError):
                first = None
        if first is None:
            self.summary_label.setText(
                self.tr("Preview appears after attaching an input dataset.")
            )
            return
        try:
            summary = self.create_operation().interface_summary(first, self.get_params())
        except CardOperationError as exc:
            self.summary_label.setText(
                self.tr("Preview unavailable: {reason}").format(
                    reason=translate_runtime_message(exc)
                )
            )
            return
        except Exception as exc:
            self.summary_label.setText(
                self.tr("Preview unavailable: {reason}").format(reason=str(exc))
            )
            return
        self.summary_label.setText(self._format_summary(summary))

    def _format_summary(self, summary: dict) -> str:
        return self.tr(
            "{axis}-axis interface @ {pos}; L={l_layers} ({left_formula}) "
            "<-> R={r_layers} ({right_formula}); c_max={c_max}; outputs: {count}"
        ).format(
            axis=summary["axis"],
            pos=f"{summary['position']:.3f}",
            l_layers=int(summary["left_layers"]),
            left_formula=summary["left_formula"],
            r_layers=int(summary["right_layers"]),
            right_formula=summary["right_formula"],
            c_max=f"{summary['c_max']:.3g}",
            count=int(summary["num_structures"]),
        )

    def create_operation(self):
        return InterfaceLayerMixOperation()

    def get_params(self) -> InterfaceLayerMixParams:
        return InterfaceLayerMixParams(
            axis=combo_value(self.axis_combo),
            auto_position=True,
            interface_position=0.5,
            left_layers=int(self.left_layers_frame.get_input_value()[0]),
            right_layers=int(self.right_layers_frame.get_input_value()[0]),
            mode=combo_value(self.mode_combo),
            concentration=float(self.concentration_frame.get_input_value()[0]) / 100.0,
            gradient_start=float(self.gradient_start_frame.get_input_value()[0]) / 100.0,
            gradient_end=float(self.gradient_end_frame.get_input_value()[0]) / 100.0,
            num_structures=int(self.num_structures_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: InterfaceLayerMixParams) -> None:
        set_combo_value(self.axis_combo, params.axis)
        self.left_layers_frame.set_input_value([int(params.left_layers)])
        self.right_layers_frame.set_input_value([int(params.right_layers)])
        set_combo_value(self.mode_combo, params.mode)
        self.concentration_frame.set_input_value([float(params.concentration) * 100.0])
        self.gradient_start_frame.set_input_value([float(params.gradient_start) * 100.0])
        self.gradient_end_frame.set_input_value([float(params.gradient_end) * 100.0])
        self.num_structures_frame.set_input_value([int(params.num_structures)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._on_controls_changed()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = InterfaceLayerMixParams(
                axis=raw_params.get("axis", "auto"),
                auto_position=raw_params.get("auto_position", True),
                interface_position=float(raw_params.get("interface_position", 0.5)),
                left_layers=int(raw_params.get("left_layers", 2)),
                right_layers=int(raw_params.get("right_layers", 2)),
                mode=raw_params.get("mode", "fixed"),
                concentration=float(raw_params.get("concentration", 0.5)),
                gradient_start=float(raw_params.get("gradient_start", 0.0)),
                gradient_end=float(raw_params.get("gradient_end", 1.0)),
                num_structures=int(raw_params.get("num_structures", 1)),
                use_seed=raw_params.get("use_seed", False),
                seed=int(raw_params.get("seed", 0)),
            )
        else:
            params = InterfaceLayerMixParams()
        self.set_params(params)
