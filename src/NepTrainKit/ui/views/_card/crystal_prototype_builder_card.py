"""Card for generating single-element crystal prototype structures."""

from __future__ import annotations

from qfluentwidgets import CaptionLabel, ComboBox, LineEdit

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import (
    CrystalPrototypeBuilderOperation,
    CrystalPrototypeBuilderParams,
)
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class CrystalPrototypeBuilderCard(MakeDataCard):
    """Generate a single-element crystal prototype without an input dataset."""

    group = "Lattice"
    card_name = "Crystal Prototype Builder"
    menu_icon = r":/images/src/images/supercell.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    requires_input_dataset = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Crystal Prototype Builder"))
        self.init_ui()

    def init_ui(self):
        """Build a compact form with only active parameters visible."""
        self.setObjectName("crystal_prototype_builder_card_widget")

        self.structure_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.structure_combo,
            [
                ("fcc", "FCC conventional cell"),
                ("bcc", "BCC conventional cell"),
                ("hcp", "HCP primitive cell"),
                ("fcc111", "FCC (111)-oriented periodic cell"),
            ],
        )
        self.structure_field = CompactField(
            self.tr("Crystal prototype"),
            self.structure_combo,
            self.setting_widget,
            self.tr("Choose one single-element ideal prototype; generated coordinates are not relaxed."),
        )

        self.element_edit = LineEdit(self.setting_widget)
        self.element_edit.setPlaceholderText(self.tr("e.g. Cu"))
        self.element_edit.setText("Cu")
        self.element_field = CompactField(
            self.tr("Element symbol"),
            self.element_edit,
            self.setting_widget,
            self.tr("Enter exactly one real chemical element symbol, such as Cu, Fe, or Mg."),
            inline=True,
            input_max_width=132,
        )
        self.element_edit.setFixedWidth(132)

        self.a_frame = SpinBoxUnitInputFrame(self)
        self.a_frame.set_input(["–", self.tr("step"), "Å"], 3, "float")
        self.a_frame.setDecimals(4)
        self.a_frame.setRange(0.1, 100.0)
        self.a_frame.set_input_value([3.6, 3.6, 0.1])
        self.a_field = CompactField(
            self.tr("Lattice constant a (min, max, step)"),
            self.a_frame,
            self.setting_widget,
            self.tr("Endpoints are included when reached by the positive step; reversed endpoints are normalized."),
        )

        self.covera_frame = SpinBoxUnitInputFrame(self)
        self.covera_frame.set_input("", 1, "float")
        self.covera_frame.setDecimals(4)
        self.covera_frame.setRange(0.1, 5.0)
        self.covera_frame.set_input_value([1.633])
        self.covera_field = CompactField(
            self.tr("HCP c/a ratio"),
            self.covera_frame,
            self.setting_widget,
            self.tr("Only HCP uses this ratio; c = a × (c/a)."),
            inline=True,
            input_max_width=132,
        )
        self.covera_frame.setFixedWidth(132)

        prototype_section = InspectorSection(
            self.tr("Prototype"),
            self.setting_widget,
            self.tr("This generator creates a fully periodic, single-element starting structure."),
        )
        prototype_grid = ResponsiveFormGrid(prototype_section, two_column_threshold=520)
        prototype_grid.add_field(self.structure_field, span=2)
        prototype_grid.add_field(self.element_field)
        prototype_grid.add_field(self.a_field)
        prototype_grid.add_field(self.covera_field)
        prototype_section.addWidget(prototype_grid)

        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input(self.tr("structures"), 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([200])
        self.max_output_field = CompactField(
            self.tr("Maximum outputs"),
            self.max_output_frame,
            self.setting_widget,
            self.tr("If the a scan has more points, only the first values in ascending scan order are kept."),
            inline=True,
            input_max_width=176,
        )
        self.max_output_frame.setFixedWidth(176)

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.max_output_field)
        output_section.addWidget(self.output_preview)

        self.expansion_tip = CaptionLabel(
            self.tr(
                "Need a larger cell? Add a Super Cell card after this card to choose repeats, target lengths, or an atom budget."
            ),
            self.setting_widget,
        )
        self.expansion_tip.setWordWrap(True)
        self.expansion_tip.setStyleSheet("color:#4078a8; font-weight:600;")
        self.legacy_expansion_notice = CaptionLabel("", self.setting_widget)
        self.legacy_expansion_notice.setWordWrap(True)
        self.legacy_expansion_notice.setStyleSheet("color:#c56a00; font-weight:600;")
        self.legacy_expansion_notice.hide()
        next_step_section = InspectorSection(self.tr("Next step"), self.setting_widget)
        next_step_section.addWidget(self.expansion_tip)
        next_step_section.addWidget(self.legacy_expansion_notice)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(prototype_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(next_step_section, 2, 0, 1, 3)

        self.structure_combo.currentIndexChanged.connect(self._update_widgets)
        self.element_edit.textChanged.connect(self._update_output_preview)
        for frame in (
            self.a_frame,
            self.covera_frame,
            self.max_output_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._update_output_preview)
        self._update_widgets()

    def _update_widgets(self, *_args) -> None:
        lattice = combo_value(self.structure_combo, "fcc")
        self.covera_field.setVisible(lattice == "hcp")
        self.structure_field.set_helper_text(
            self.tr("Fully periodic (PBC x/y/z), with no vacuum; the third cell vector is normal to FCC (111).")
            if lattice == "fcc111"
            else self.tr("Choose one single-element ideal prototype; generated coordinates are not relaxed.")
        )
        self._update_output_preview()

    def _update_output_preview(self, *_args) -> None:
        params = self.get_params()
        try:
            plan = self.create_operation().plan(params)
            shown = min(len(plan.a_values), params.max_outputs)
            la, lb, lc = plan.cell_lengths
            text = self.tr(
                "{shown} base-cell output(s); {atoms} atoms each; "
                "first cell lengths {la:.3f} × {lb:.3f} × {lc:.3f} Å."
            ).format(
                shown=shown,
                atoms=plan.atoms_per_output,
                la=la,
                lb=lb,
                lc=lc,
            )
            if plan.truncated:
                text += " " + self.tr("The scan has {total} points; later a values are truncated.").format(
                    total=len(plan.a_values)
                )
            self.output_preview.setText(text)
        except ValueError as exc:
            self.output_preview.setText(translate_runtime_message(exc))
        self.refresh_compact_presentation()

    def create_operation(self):
        return CrystalPrototypeBuilderOperation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        try:
            count = min(len(self.create_operation().plan(params).a_values), params.max_outputs)
            return self.tr("{element} · {lattice} · {count} base-cell output(s)").format(
                element=params.element.strip() or self.tr("invalid element"),
                lattice=self.structure_combo.currentText(),
                count=count,
            )
        except ValueError:
            return self.tr("{lattice} · parameters need attention").format(
                lattice=self.structure_combo.currentText()
            )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Use prototype-specific lattice constants. In particular, HCP a is not the FCC conventional-cell a. "
            "Add Super Cell next when downstream operations need a larger structure."
        )

    def get_params(self) -> CrystalPrototypeBuilderParams:
        return CrystalPrototypeBuilderParams(
            lattice=combo_value(self.structure_combo, "fcc"),
            element=self.element_edit.text(),
            a_range=tuple(float(value) for value in self.a_frame.get_input_value()),
            covera=float(self.covera_frame.get_input_value()[0]),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: CrystalPrototypeBuilderParams) -> None:
        set_combo_value(self.structure_combo, params.lattice)
        self.element_edit.setText(params.element)
        self.a_frame.set_input_value([float(value) for value in params.a_range])
        self.covera_frame.set_input_value([float(params.covera)])
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self._update_widgets()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            raw_params = dict(raw_params)
            legacy_expansion = any(
                key in raw_params for key in ("auto_supercell", "max_atoms", "rep")
            )
            for key in ("auto_supercell", "max_atoms", "rep"):
                raw_params.pop(key, None)
            raw_params["a_range"] = tuple(raw_params.get("a_range", [3.6, 3.6, 0.1]))
            params = CrystalPrototypeBuilderParams(**raw_params)
        else:
            legacy_expansion = any(
                key in data_dict for key in ("auto_supercell", "max_atoms", "rep")
            )
            params = CrystalPrototypeBuilderParams(
                lattice=data_dict.get("lattice", "fcc"),
                element=data_dict.get("element", "Cu"),
                a_range=tuple(data_dict.get("a_range", [3.6, 3.6, 0.1])),
                covera=data_dict.get("covera", [1.633])[0],
                max_outputs=data_dict.get("max_outputs", [200])[0],
            )
        self.set_params(params)
        if legacy_expansion:
            migration_message = self.tr(
                "This saved Crystal Prototype Builder used the removed expansion settings. "
                "They were ignored; add a Super Cell card after it to restore the intended cell size."
            )
            self.legacy_expansion_notice.setText("⚠ " + migration_message)
            self.legacy_expansion_notice.show()
            MessageManager.send_warning_message(migration_message)
