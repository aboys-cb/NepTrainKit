"""Card for explicit distance, volume, and density geometry filtering."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CaptionLabel, CheckBox, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.filter import GeometryFilterOperation, GeometryFilterParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import FilterDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class GeometryFilterCard(FilterDataCard):
    """Reject structures that violate explicit geometry-quality thresholds."""

    group = "Filter"
    card_name = "Geometry Filter"
    menu_icon = r":/images/src/images/check.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_dataset = []
        self.setTitle(self.tr("Geometry Sanity Filter"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("geometry_filter_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.min_pair_label = BodyLabel(self.tr("Shortest allowed pair distance"), self.setting_widget)
        self.min_pair_label.setToolTip(
            self.tr(
                "Reject any structure with a closer atom pair; 0 disables this chemistry-independent hard cutoff"
            )
        )
        self.min_pair_label.installEventFilter(ToolTipFilter(self.min_pair_label, 300, ToolTipPosition.TOP))
        self.min_pair_frame = SpinBoxUnitInputFrame(self)
        self.min_pair_frame.set_input("Å", 1, "float")
        self.min_pair_frame.setRange(0.0, 20.0)
        self.min_pair_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_pair_frame.set_input_value([0.0])

        self.bulk_checkbox = CheckBox(
            self.tr("Show bulk cell, volume, and density limits"),
            self.setting_widget,
        )
        self.bulk_checkbox.setChecked(False)

        self.min_vpa_label = BodyLabel(self.tr("Min volume/atom"), self.setting_widget)
        self.min_vpa_label.setToolTip(self.tr("Reject structures below this per-atom volume in A^3; 0 disables"))
        self.min_vpa_label.installEventFilter(ToolTipFilter(self.min_vpa_label, 300, ToolTipPosition.TOP))
        self.min_vpa_frame = SpinBoxUnitInputFrame(self)
        self.min_vpa_frame.set_input("Å³", 1, "float")
        self.min_vpa_frame.setRange(0.0, 10000.0)
        self.min_vpa_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_vpa_frame.set_input_value([0.0])

        self.max_vpa_label = BodyLabel(self.tr("Max volume/atom"), self.setting_widget)
        self.max_vpa_label.setToolTip(self.tr("Reject structures above this per-atom volume in A^3; 0 disables"))
        self.max_vpa_label.installEventFilter(ToolTipFilter(self.max_vpa_label, 300, ToolTipPosition.TOP))
        self.max_vpa_frame = SpinBoxUnitInputFrame(self)
        self.max_vpa_frame.set_input("Å³", 1, "float")
        self.max_vpa_frame.setRange(0.0, 10000.0)
        self.max_vpa_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.max_vpa_frame.set_input_value([0.0])

        self.min_density_label = BodyLabel(self.tr("Min density"), self.setting_widget)
        self.min_density_label.setToolTip(self.tr("Reject structures below this mass density in g/cm^3; 0 disables"))
        self.min_density_label.installEventFilter(ToolTipFilter(self.min_density_label, 300, ToolTipPosition.TOP))
        self.min_density_frame = SpinBoxUnitInputFrame(self)
        self.min_density_frame.set_input("g/cm³", 1, "float")
        self.min_density_frame.setRange(0.0, 1000.0)
        self.min_density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_density_frame.set_input_value([0.0])

        self.max_density_label = BodyLabel(self.tr("Max density"), self.setting_widget)
        self.max_density_label.setToolTip(self.tr("Reject structures above this mass density in g/cm^3; 0 disables"))
        self.max_density_label.installEventFilter(ToolTipFilter(self.max_density_label, 300, ToolTipPosition.TOP))
        self.max_density_frame = SpinBoxUnitInputFrame(self)
        self.max_density_frame.set_input("g/cm³", 1, "float")
        self.max_density_frame.setRange(0.0, 1000.0)
        self.max_density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.max_density_frame.set_input_value([0.0])

        self.require_cell_checkbox = CheckBox(
            self.tr("Require a finite, nonzero-volume cell"),
            self.setting_widget,
        )
        self.require_cell_checkbox.setChecked(False)
        self.require_cell_checkbox.setToolTip(self.tr("Reject zero-volume or invalid-cell structures even when volume/density thresholds are disabled"))
        self.require_cell_checkbox.installEventFilter(ToolTipFilter(self.require_cell_checkbox, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.min_pair_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.min_pair_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.bulk_checkbox, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.min_vpa_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.min_vpa_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_vpa_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.max_vpa_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.min_density_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.min_density_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.max_density_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.max_density_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.require_cell_checkbox, 6, 0, 1, 3)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("geometryFilterPreview")
        self.settingLayout.addWidget(self.preview_label, 7, 0, 1, 3)

        self.bulk_controls = (
            self.min_vpa_label,
            self.min_vpa_frame,
            self.max_vpa_label,
            self.max_vpa_frame,
            self.min_density_label,
            self.min_density_frame,
            self.max_density_label,
            self.max_density_frame,
            self.require_cell_checkbox,
        )
        self.bulk_checkbox.stateChanged.connect(self._update_bulk_visibility)
        for frame in (
            self.min_pair_frame,
            self.min_vpa_frame,
            self.max_vpa_frame,
            self.min_density_frame,
            self.max_density_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._refresh_preview)
        self.require_cell_checkbox.stateChanged.connect(self._refresh_preview)
        self._update_bulk_visibility()
        self._refresh_preview()

    def _update_bulk_visibility(self, *_args) -> None:
        visible = self.bulk_checkbox.isChecked()
        for widget in self.bulk_controls:
            widget.setVisible(visible)
        self._refresh_preview()

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_dataset = list(dataset) if dataset is not None else []
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if not self._input_dataset:
            self.preview_label.setText(
                self.tr(
                    "Load upstream structures to preview how many pass each active geometry limit."
                )
            )
            return
        try:
            summary = self.create_operation().filter_summary(
                self._input_dataset,
                self.get_params(),
            )
        except (TypeError, ValueError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return
        labels = {
            "empty": self.tr("empty structures"),
            "nonfinite_positions": self.tr("non-finite positions"),
            "invalid_cell": self.tr("invalid cells"),
            "pair_distance": self.tr("short pairs"),
            "volume_per_atom": self.tr("volume/atom"),
            "density": self.tr("density"),
        }
        details = [
            f"{labels[key]} {count}"
            for key, count in summary["reasons"].items()
            if count
        ]
        rejected = (
            self.tr("rejected: {details}").format(details=", ".join(details))
            if details
            else self.tr("no structures rejected")
        )
        self.preview_label.setText(
            self.tr(
                "First pass: input {input} → keep {kept} / reject {rejected_count} · {reasons}"
            ).format(
                input=summary["input_count"],
                kept=summary["kept_count"],
                rejected_count=summary["rejected_count"],
                reasons=rejected,
            )
        )

    def create_operation(self):
        return GeometryFilterOperation()

    def get_params(self) -> GeometryFilterParams:
        return GeometryFilterParams(
            min_pair_distance=float(self.min_pair_frame.get_input_value()[0]),
            min_volume_per_atom=float(self.min_vpa_frame.get_input_value()[0]),
            max_volume_per_atom=float(self.max_vpa_frame.get_input_value()[0]),
            min_density=float(self.min_density_frame.get_input_value()[0]),
            max_density=float(self.max_density_frame.get_input_value()[0]),
            require_finite_cell=self.require_cell_checkbox.isChecked(),
        )

    def set_params(self, params: GeometryFilterParams) -> None:
        self.min_pair_frame.set_input_value([float(params.min_pair_distance)])
        self.min_vpa_frame.set_input_value([float(params.min_volume_per_atom)])
        self.max_vpa_frame.set_input_value([float(params.max_volume_per_atom)])
        self.min_density_frame.set_input_value([float(params.min_density)])
        self.max_density_frame.set_input_value([float(params.max_density)])
        self.require_cell_checkbox.setChecked(bool(params.require_finite_cell))
        show_bulk = (
            float(params.min_volume_per_atom) > 0.0
            or float(params.max_volume_per_atom) > 0.0
            or float(params.min_density) > 0.0
            or float(params.max_density) > 0.0
            or bool(params.require_finite_cell)
        )
        self.bulk_checkbox.setChecked(show_bulk)
        self._update_bulk_visibility()
        self._refresh_preview()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = GeometryFilterParams(**raw_params) if raw_params else GeometryFilterParams()
        self.set_params(params)
