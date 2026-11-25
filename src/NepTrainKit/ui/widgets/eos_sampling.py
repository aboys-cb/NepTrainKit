"""EOS context inside the card canvas; no independent generation or export path."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon,
    TableWidget,
    TransparentPushButton,
    TransparentToolButton,
)

from NepTrainKit import palette
from NepTrainKit.core.validation.sampling import EOSSamplingParams, sampling_volumes, strain_params
from NepTrainKit.ui.widgets import AdaptiveCompactDoubleSpinBox, AdaptiveCompactSpinBox, CompactField


class EOSSamplingPanel(QFrame):
    parametersChanged = Signal()

    def __init__(self, parent, *, card, atoms, context):
        """Bind an EOS selection and volume preview to an existing strain card."""
        super().__init__(parent)
        self.card, self.atoms, self.context = card, atoms, context
        self._updating = False
        self._from_region = False
        self.valid = False
        self.volumes = np.array([])
        self.setObjectName("eosSamplingPanel")
        self.setStyleSheet(
            f"#eosSamplingPanel {{background: {palette.SURFACE}; "
            f"border: 1px solid {palette.BORDER}; border-radius: 8px;}}"
        )
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        heading = QHBoxLayout()
        heading.addWidget(BodyLabel(self.tr("EOS sampling range"), self), 1)
        reset = TransparentToolButton(FluentIcon.SYNC, self)
        reset.setToolTip(self.tr("Reset view"))
        heading.addWidget(reset)
        layout.addLayout(heading)
        label = CaptionLabel(context["suggestion"]["summary"]["formula"], self)
        label.setToolTip(context["benchmark"]["name"])
        layout.addWidget(label)
        suggestion = context["suggestion"]
        reasons = {
            "manual": self.tr("No current-model result; choose an interval manually."),
            "offset_only": self.tr("Constant offset only: check the energy reference before adding data."),
            "reference_incomplete": self.tr("Reference minimum is not bracketed; complete the reference scan first."),
            "shape_error": self.tr("Suggested near the largest shape residual: {peak:.2f} meV/atom.").format(
                peak=suggestion["peak_mev"] or 0
            ),
        }
        reason = CaptionLabel(reasons[suggestion["reason"]], self)
        reason.setWordWrap(True)
        layout.addWidget(reason)
        import pyqtgraph as pg

        self.plot = pg.PlotWidget(self)
        self.plot.setBackground(palette.SURFACE)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", self.tr("Volume (Å³/atom)"))
        self.plot.setLabel("left", self.tr("Relative energy (meV/atom)"))
        self.plot.setMinimumSize(0, 170)
        self.plot.addLegend(labelTextSize="9pt")
        summary = suggestion["summary"]
        reference = np.asarray(summary["energies"])
        self.plot.plot(
            summary["volumes"],
            (reference - reference.min()) * 1000,
            pen=pg.mkPen(palette.PLOT_AXIS, width=1.5),
            symbol="o",
            symbolSize=4,
            name=self.tr("DFT reference"),
        )
        details = context.get("details")
        if details:
            prediction = np.asarray(details["predicted"])
            prediction = prediction - np.mean(prediction - reference)
            self.plot.plot(
                summary["volumes"],
                (prediction - reference.min()) * 1000,
                pen=pg.mkPen(palette.ACCENT, width=2),
                name=self.tr("Current model · offset removed"),
            )
        self.region = pg.LinearRegionItem(
            movable=True, brush=pg.mkBrush(0, 160, 160, 28), pen=pg.mkPen(palette.ACCENT, width=1.5)
        )
        self.region.setZValue(5)
        self.plot.addItem(self.region)
        self.sample_marks = pg.ScatterPlotItem(pen=pg.mkPen("#E8912B"), brush="#E8912B", symbol="t", size=8)
        self.sample_marks.setZValue(6)
        self.plot.addItem(self.sample_marks)
        reset.clicked.connect(lambda: self.plot.getViewBox().enableAutoRange(x=True, y=True))
        layout.addWidget(self.plot, 1)
        drag_hint = CaptionLabel(
            self.tr("Drag edges to resize; drag the shaded band to move. Triangles mark volumes, not energies."), self
        )
        drag_hint.setWordWrap(True)
        layout.addWidget(drag_hint)
        self.start_volume, self.stop_volume = AdaptiveCompactDoubleSpinBox(self), AdaptiveCompactDoubleSpinBox(self)
        for control in (self.start_volume, self.stop_volume):
            control.setRange(0.00001, 100000.0)
            control.setDecimals(5)
            control.setSingleStep(0.1)
            control.setMinimumWidth(0)
        self.count = AdaptiveCompactSpinBox(self)
        self.count.setRange(2, 100)
        self.count.setMinimumWidth(0)
        form = QHBoxLayout()
        form.setSpacing(6)
        for text, control in (
            (self.tr("Start (Å³/atom)"), self.start_volume),
            (self.tr("End (Å³/atom)"), self.stop_volume),
            (self.tr("Points"), self.count),
        ):
            field = CompactField(text, control, self)
            field.setMinimumWidth(0)
            form.addWidget(field, 1 if control is self.count else 2)
        layout.addLayout(form)
        self.table = TableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            [self.tr("Point"), self.tr("Volume (Å³/atom)"), self.tr("Lattice strain (%)")]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setMinimumSectionSize(24)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(125)
        details_button = TransparentPushButton(self.tr("Volume point details"), self)
        details_button.setCheckable(True)
        details_button.toggled.connect(self.table.setVisible)
        layout.addWidget(details_button)
        layout.addWidget(self.table)
        self.table.hide()
        self.status_label = CaptionLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.region.sigRegionChanged.connect(self._region_changed)
        self.region.sigRegionChangeFinished.connect(self._region_finished)
        for control in (self.start_volume, self.stop_volume, self.count):
            control.valueChanged.connect(self._inputs_changed)
        self.sync_from_card()

    def params(self):
        return EOSSamplingParams(self.start_volume.value(), self.stop_volume.value(), self.count.value())

    def _region_changed(self):
        if self._updating:
            return
        low, high = self.region.getRegion()
        with QSignalBlocker(self.start_volume), QSignalBlocker(self.stop_volume):
            self.start_volume.setValue(low)
            self.stop_volume.setValue(high)
        self._from_region = True
        try:
            self._inputs_changed()
        finally:
            self._from_region = False

    def _region_finished(self):
        if not self._updating and self.valid:
            self.sync_from_card()

    def _inputs_changed(self, *_args):
        if self._updating:
            return
        self.valid = False
        self.sample_marks.clear()
        self.table.setRowCount(0)
        try:
            params = strain_params(self.atoms, self.params())
        except ValueError:
            self.status_label.setText(self.tr("Choose an increasing positive range and 2–100 resolvable points."))
            self.parametersChanged.emit()
            return
        self._updating = True
        try:
            self.card.set_params(params)
        finally:
            self._updating = False
        self.sync_from_card()
        self.parametersChanged.emit()

    def sync_from_card(self):
        if self._updating:
            return
        self.valid = False
        self.sample_marks.clear()
        self.table.setRowCount(0)
        try:
            self.volumes = sampling_volumes(self.atoms, self.card.get_params())
        except ValueError:
            self.status_label.setText(
                self.tr("EOS preview needs isotropic strain, 2–100 points and molecular handling disabled.")
            )
            return
        self._updating = True
        try:
            with QSignalBlocker(self.start_volume), QSignalBlocker(self.stop_volume), QSignalBlocker(self.count):
                self.start_volume.setValue(float(self.volumes[0]))
                self.stop_volume.setValue(float(self.volumes[-1]))
                self.count.setValue(len(self.volumes))
            if not self._from_region:
                self.region.setRegion((self.volumes[0], self.volumes[-1]))
            self._show_volumes(self.volumes)
        finally:
            self._updating = False
        self.valid = True
        self.status_label.setText(
            self.tr(
                "{count} planned points · Run the workflow to generate structures. Equal lattice-scale steps."
            ).format(count=len(self.volumes))
        )

    def _show_volumes(self, volumes):
        self.sample_marks.setData(volumes, np.full(len(volumes), -1.0))
        base = self.atoms.get_volume() / len(self.atoms)
        self.table.setRowCount(len(volumes))
        for row, volume in enumerate(volumes):
            for col, text in enumerate((str(row + 1), f"{volume:.5f}", f"{100 * (np.cbrt(volume / base) - 1):+.6f}")):
                self.table.setItem(row, col, QTableWidgetItem(text))

    def show_outputs(self, structures):
        self._show_volumes([a.get_volume() / len(a) for a in structures])
        self.status_label.setText(
            self.tr("{count} unlabelled structures generated by the card. Use its output and export actions.").format(
                count=len(structures)
            )
        )

    def set_editing_enabled(self, enabled):
        for control in (self.start_volume, self.stop_volume, self.count):
            control.setEnabled(enabled)
        self.region.setMovable(enabled)
