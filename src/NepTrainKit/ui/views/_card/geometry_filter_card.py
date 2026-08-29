"""Card for explicit distance, volume, and density geometry filtering."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from qfluentwidgets import CaptionLabel, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.filter import GeometryFilterOperation, GeometryFilterParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.widgets import (
    CompactField,
    FilterDataCard,
    InspectorSection,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class GeometryFilterCard(FilterDataCard):
    """Reject structures that violate explicit geometry-quality thresholds."""

    _PREVIEW_DEBOUNCE_MS = 120

    group = "Filter"
    card_name = "Geometry Filter"
    menu_icon = r":/images/src/images/check.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_dataset = []
        self._preview_generation = 0
        self._preview_task: BackgroundTask | None = None
        self._active_preview_generation: int | None = None
        self._pending_preview = None
        self._preview_closing = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self._PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._start_preview)
        self.setTitle(self.tr("Geometry Filter"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("geometry_filter_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.min_pair_frame = SpinBoxUnitInputFrame(self)
        self.min_pair_frame.set_input("Å", 1, "float")
        self.min_pair_frame.setRange(0.0, 20.0)
        self.min_pair_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_pair_frame.set_input_value([0.5])
        self.min_pair_frame.setFixedWidth(150)
        self.min_pair_frame.setAccessibleName(self.tr("Minimum pair distance"))
        self.min_pair_field = CompactField(
            self.tr("Minimum pair distance"),
            self.min_pair_frame,
            self.setting_widget,
            self.tr("Rejects pairs strictly below this value; 0 disables the limit."),
            inline=True,
            input_max_width=150,
        )
        distance_section = InspectorSection(
            self.tr("Atomic distances"),
            self.setting_widget,
            self.tr(
                "Empty structures and non-finite coordinates are always removed. "
                "The distance limit is the same for every element pair."
            ),
        )
        distance_section.addWidget(self.min_pair_field)

        self.bulk_checkbox = CheckBox(
            "",
            self.setting_widget,
        )
        self.bulk_checkbox.setChecked(False)

        self.min_vpa_frame = SpinBoxUnitInputFrame(self)
        self.min_vpa_frame.set_input("Å³", 1, "float")
        self.min_vpa_frame.setRange(0.0, 10000.0)
        self.min_vpa_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_vpa_frame.set_input_value([0.0])
        self.min_vpa_frame.setAccessibleName(self.tr("Minimum volume per atom"))

        self.max_vpa_frame = SpinBoxUnitInputFrame(self)
        self.max_vpa_frame.set_input("Å³", 1, "float")
        self.max_vpa_frame.setRange(0.0, 10000.0)
        self.max_vpa_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.max_vpa_frame.set_input_value([0.0])
        self.max_vpa_frame.setAccessibleName(self.tr("Maximum volume per atom"))

        self.min_density_frame = SpinBoxUnitInputFrame(self)
        self.min_density_frame.set_input("g/cm³", 1, "float")
        self.min_density_frame.setRange(0.0, 1000.0)
        self.min_density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_density_frame.set_input_value([0.0])
        self.min_density_frame.setAccessibleName(self.tr("Minimum density"))

        self.max_density_frame = SpinBoxUnitInputFrame(self)
        self.max_density_frame.set_input("g/cm³", 1, "float")
        self.max_density_frame.setRange(0.0, 1000.0)
        self.max_density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.max_density_frame.set_input_value([0.0])
        self.max_density_frame.setAccessibleName(self.tr("Maximum density"))

        self.min_vpa_field = CompactField(self.tr("Min volume / atom"), self.min_vpa_frame, self.setting_widget)
        self.max_vpa_field = CompactField(self.tr("Max volume / atom"), self.max_vpa_frame, self.setting_widget)
        self.min_density_field = CompactField(self.tr("Min density"), self.min_density_frame, self.setting_widget)
        self.max_density_field = CompactField(self.tr("Max density"), self.max_density_frame, self.setting_widget)

        self.require_cell_checkbox = CheckBox(
            self.tr("Reject invalid cells"),
            self.setting_widget,
        )
        self.require_cell_checkbox.setChecked(False)
        self.require_cell_checkbox.setToolTip(
            self.tr("Reject zero-volume, singular, or non-finite cells even when no bulk limit is active.")
        )

        self.bulk_section = InspectorSection(
            self.tr("Cell and bulk limits"),
            self.setting_widget,
            self.tr(
                "Volume and density use the full cell, including vacuum. "
                "A value of 0 disables that limit."
            ),
        )
        bulk_grid = ResponsiveFormGrid(self.bulk_section, two_column_threshold=300)
        bulk_grid.add_field(self.require_cell_checkbox, span=2)
        bulk_grid.add_field(self.min_vpa_field)
        bulk_grid.add_field(self.max_vpa_field)
        bulk_grid.add_field(self.min_density_field)
        bulk_grid.add_field(self.max_density_field)
        self.bulk_section.addWidget(bulk_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("geometryFilterPreview")
        self.preview_section = InspectorSection(self.tr("Exact preview"), self.setting_widget)
        self.preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(distance_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.bulk_checkbox, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.bulk_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.preview_section, 3, 0, 1, 3)

        self.bulk_controls = (
            self.bulk_section,
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
        for frame in (
            self.min_vpa_frame,
            self.max_vpa_frame,
            self.min_density_frame,
            self.max_density_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._update_bulk_label)
        self.require_cell_checkbox.stateChanged.connect(self._refresh_preview)
        self.require_cell_checkbox.stateChanged.connect(self._update_bulk_label)
        self._update_bulk_visibility()
        self._refresh_preview()

    def _update_bulk_visibility(self, *_args) -> None:
        visible = self.bulk_checkbox.isChecked()
        for widget in self.bulk_controls:
            widget.setVisible(visible)
        self._update_bulk_label()
        self._refresh_preview()

    def _active_bulk_limit_count(self) -> int:
        params = self.get_params()
        return sum(
            (
                params.min_volume_per_atom > 0.0,
                params.max_volume_per_atom > 0.0,
                params.min_density > 0.0,
                params.max_density > 0.0,
                params.require_finite_cell,
            )
        )

    def _update_bulk_label(self, *_args) -> None:
        count = self._active_bulk_limit_count()
        self.bulk_checkbox.setText(
            self.tr("Cell and bulk limits · {count} active").format(count=count)
        )

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_dataset = list(dataset) if dataset is not None else []
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        self._preview_generation += 1
        self._pending_preview = None
        self._preview_timer.stop()
        if not self._input_dataset:
            if self._preview_task is not None:
                self._preview_task.stop_work()
            self._set_preview_text(
                self.tr(
                    "Load upstream structures to preview how many pass each active geometry limit."
                )
            )
            return
        self._set_preview_text(self.tr("Calculating preview in background…"))
        self._preview_timer.start()

    def _set_preview_text(self, text: str) -> None:
        """Update wrapped preview text and propagate its new height immediately."""
        self.preview_label.setText(text)
        self.preview_label.updateGeometry()
        self.preview_section.layout().invalidate()
        self.preview_section.layout().activate()
        self.preview_section.updateGeometry()
        self.settingLayout.invalidate()
        self.settingLayout.activate()
        self.setting_widget.updateGeometry()
        self.updateGeometry()

    def _start_preview(self) -> None:
        if self._preview_closing or not self._input_dataset:
            return
        request = (
            self._preview_generation,
            list(self._input_dataset),
            self.get_params(),
        )
        if self._preview_task is not None:
            self._pending_preview = request
            self._preview_task.stop_work()
            return
        self._start_preview_task(request)

    def _start_preview_task(self, request) -> None:
        request_id, dataset, params = request
        task = BackgroundTask(self, show_tip=False)
        self._preview_task = task
        self._active_preview_generation = request_id
        task.succeeded.connect(self._on_preview_succeeded)
        task.failed.connect(self._on_preview_failed)
        task.finished.connect(self._on_preview_task_finished)
        task.start_work(
            self._calculate_preview,
            request_id,
            dataset,
            params,
        )

    @staticmethod
    def _calculate_preview(request_id, dataset, params):
        summary = GeometryFilterOperation().filter_summary(dataset, params)
        return request_id, summary

    def _on_preview_succeeded(self, result) -> None:
        request_id, summary = result
        if self._preview_closing or request_id != self._preview_generation:
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
        self._set_preview_text(
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
        labels = {
            "empty": self.tr("empty structures"),
            "nonfinite_positions": self.tr("non-finite positions"),
            "invalid_cell": self.tr("invalid cells"),
            "pair_distance": self.tr("short pairs"),
            "volume_too_small": self.tr("volume below minimum"),
            "volume_too_large": self.tr("volume above maximum"),
            "density_too_low": self.tr("density below minimum"),
            "density_too_high": self.tr("density above maximum"),
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
        self._set_preview_text(
            self.tr(
                "Current settings: input {input} → keep {kept} / reject {rejected_count} · {reasons}"
            ).format(
                input=summary["input_count"],
                kept=summary["kept_count"],
                rejected_count=summary["rejected_count"],
                reasons=rejected,
            )
        )

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

    def get_summary_text(self) -> str:
        params = self.get_params()
        distance = (
            self.tr("pair distance ≥ {value} Å").format(value=f"{params.min_pair_distance:g}")
            if params.min_pair_distance > 0.0
            else self.tr("pair limit off")
        )
        return self.tr("{distance} · {count} cell/bulk limits").format(
            distance=distance,
            count=self._active_bulk_limit_count(),
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Preview checks the complete input. For slabs or molecules with vacuum, "
            "leave volume and density limits off unless the full-cell values are meaningful."
        )

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = GeometryFilterParams(**raw_params) if raw_params else GeometryFilterParams()
        self.set_params(params)
