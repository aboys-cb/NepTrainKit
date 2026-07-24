#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distribution analysis controls and their non-modal inspector window."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    ComboBox,
    EditableComboBox,
    LineEdit,
    MessageBoxBase,
    PrimaryPushButton,
    PushButton,
    SpinBox,
)

from NepTrainKit.config import Config
from NepTrainKit.core.io.base import DistributionRequest
from NepTrainKit.core.types import (
    CanvasMode,
    DistributionCurveStyle,
    DistributionGroupMode,
    DistributionScope,
    DistributionSelectMode,
    DistributionValueView,
)
from NepTrainKit.ui.canvas.distribution_factory import create_distribution_plot_adapter


class DistributionExplorerWidget(QWidget):
    """Embeddable explorer for numeric dataset and atomic-field distributions."""

    _ALL_SERIES_KEY = "__all__"

    def __init__(
        self,
        parent=None,
        data=None,
        run_analysis_callback=None,
        apply_selection_callback=None,
        canvas_type: str | None = None,
    ):
        super().__init__(parent)
        self._data = data
        self._run_analysis_callback = run_analysis_callback
        self._apply_selection_callback = apply_selection_callback
        self._analysis: dict[str, Any] = {}
        self._field_specs: list[Any] = []
        self._field_by_key: dict[str, Any] = {}
        self._metric_by_key: dict[str, dict[str, Any]] = {}
        self._canvas_type = str(canvas_type or Config.get("widget", "canvas_type", CanvasMode.PYQTGRAPH.value)).strip()
        self._plot_adapter, self._vispy_fallback_warned = create_distribution_plot_adapter(self._canvas_type, self)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)
        self.setLayout(root_layout)

        control_frame = QFrame(self)
        control_layout = QGridLayout(control_frame)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)

        self.fieldCombo = EditableComboBox(self)
        self.groupCombo = ComboBox(self)
        self.groupCombo.addItem(self.tr("Formula"), userData=DistributionGroupMode.FORMULA.value)
        self.groupCombo.addItem(self.tr("Element"), userData=DistributionGroupMode.ELEMENT.value)
        self.groupCombo.addItem(self.tr("Value source"), userData=DistributionGroupMode.VALUE_VIEW.value)
        self.groupCombo.addItem(self.tr("Custom group data"), userData=DistributionGroupMode.CUSTOM.value)
        self.groupCombo.setCurrentIndex(2)

        self.scopeCombo = ComboBox(self)
        self.scopeCombo.addItem(self.tr("All data"), userData=DistributionScope.ACTIVE.value)
        self.scopeCombo.addItem(self.tr("Selected data"), userData=DistributionScope.SELECTED.value)
        self.scopeCombo.setCurrentIndex(0)

        # Value source checkboxes (visible in VALUE_VIEW mode, and in CUSTOM mode with single group)
        self.refCheck = CheckBox(self.tr("Reference"), self)
        self.refCheck.setChecked(True)
        self.predCheck = CheckBox(self.tr("Prediction"), self)
        self.predCheck.setChecked(False)
        self.errCheck = CheckBox(self.tr("Error (pred - ref)"), self)
        self.errCheck.setChecked(False)
        self.refCheck.toggled.connect(self._on_value_view_check_changed)
        self.predCheck.toggled.connect(self._on_value_view_check_changed)
        self.errCheck.toggled.connect(self._on_value_view_check_changed)

        self.viewCombo = ComboBox(self)
        self.viewCombo.addItem(self.tr("Reference"), userData=DistributionValueView.REFERENCE.value)
        self.viewCombo.addItem(self.tr("Prediction"), userData=DistributionValueView.PREDICTION.value)
        self.viewCombo.addItem(self.tr("Error (pred - ref)"), userData=DistributionValueView.ERROR.value)
        self.viewCombo.setCurrentIndex(0)

        # Custom group management
        self.customGroupBtn = PushButton(self.tr("Edit groups"), self)
        self.customGroupBtn.clicked.connect(self._edit_custom_groups)
        self._custom_groups: list[dict[str, Any]] = []

        self.curveCombo = ComboBox(self)
        self.curveCombo.addItem(self.tr("None"), userData=DistributionCurveStyle.NONE.value)
        self.curveCombo.addItem(self.tr("KDE"), userData=DistributionCurveStyle.KDE.value)
        self.curveCombo.addItem(self.tr("Normal"), userData=DistributionCurveStyle.NORMAL.value)
        self.curveCombo.setCurrentIndex(0)

        self.binsSpin = SpinBox(self)
        self.binsSpin.setRange(2, 2000)
        self.binsSpin.setValue(120)

        self.selectModeCombo = ComboBox(self)
        self.selectModeCombo.addItem(self.tr("Replace"), userData=DistributionSelectMode.REPLACE.value)
        self.selectModeCombo.addItem(self.tr("Add"), userData=DistributionSelectMode.ADD.value)
        self.selectModeCombo.addItem(self.tr("Intersect"), userData=DistributionSelectMode.INTERSECT.value)

        self.includeNormCheck = CheckBox(self.tr("Include norm"), self)
        self.includeNormCheck.setChecked(True)
        self.advancedCheck = CheckBox(self.tr("Advanced options"), self)
        self.advancedCheck.setChecked(False)
        self.advancedCheck.toggled.connect(self._set_advanced_visible)

        self.analyzeButton = PrimaryPushButton(self.tr("Analyze"), self)
        self.analyzeButton.clicked.connect(self._run_analysis)

        self.fieldLabel = CaptionLabel(self.tr("Field"), self)
        self.groupLabel = CaptionLabel(self.tr("Group by"), self)
        self.scopeLabel = CaptionLabel(self.tr("Scope"), self)
        self.viewLabel = CaptionLabel(self.tr("View"), self)
        self.selectModeLabel = CaptionLabel(self.tr("Select mode"), self)
        self.binsLabel = CaptionLabel(self.tr("Bins"), self)
        self.curveLabel = CaptionLabel(self.tr("Curve"), self)
        control_layout.addWidget(self.fieldLabel, 0, 0)
        control_layout.addWidget(self.fieldCombo, 0, 1, 1, 3)
        control_layout.addWidget(self.groupLabel, 1, 0)
        control_layout.addWidget(self.groupCombo, 1, 1)
        control_layout.addWidget(self.scopeLabel, 1, 2)
        control_layout.addWidget(self.scopeCombo, 1, 3)
        # Row 2: value source checkboxes (visible in VALUE_VIEW, or CUSTOM with single group)
        self.valueViewsLabel = CaptionLabel(self.tr("Value source"), self)
        control_layout.addWidget(self.valueViewsLabel, 2, 0)
        control_layout.addWidget(self.refCheck, 2, 1)
        control_layout.addWidget(self.predCheck, 2, 2)
        control_layout.addWidget(self.errCheck, 2, 3)
        control_layout.addWidget(self.viewLabel, 2, 0)
        control_layout.addWidget(self.viewCombo, 2, 1, 1, 3)
        # Row 3: custom group editor (visible only in CUSTOM mode)
        self.customGroupLabel = CaptionLabel(self.tr("Custom group data"), self)
        control_layout.addWidget(self.customGroupLabel, 3, 0)
        control_layout.addWidget(self.customGroupBtn, 3, 1, 1, 3)
        # Row 4: advanced options
        control_layout.addWidget(self.binsLabel, 4, 0)
        control_layout.addWidget(self.binsSpin, 4, 1)
        control_layout.addWidget(self.curveLabel, 4, 2)
        control_layout.addWidget(self.curveCombo, 4, 3)
        control_layout.addWidget(self.selectModeLabel, 5, 0)
        control_layout.addWidget(self.selectModeCombo, 5, 1)
        control_layout.addWidget(self.advancedCheck, 6, 0, 1, 2)
        control_layout.addWidget(self.includeNormCheck, 6, 2)
        control_layout.addWidget(self.analyzeButton, 6, 3)

        root_layout.addWidget(control_frame)

        result_frame = QFrame(self)
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(2)

        result_selector = QWidget(self)
        result_selector_layout = QHBoxLayout(result_selector)
        result_selector_layout.setContentsMargins(0, 0, 0, 0)
        result_selector_layout.setSpacing(6)
        self.metricCombo = ComboBox(self)
        self.seriesCombo = ComboBox(self)
        self.metricCombo.currentIndexChanged.connect(self._refresh_series_combo)
        self.seriesCombo.currentIndexChanged.connect(self._refresh_plot)
        result_selector_layout.addWidget(CaptionLabel(self.tr("Metric"), self))
        result_selector_layout.addWidget(self.metricCombo, 1)
        result_selector_layout.addWidget(CaptionLabel(self.tr("Series"), self))
        result_selector_layout.addWidget(self.seriesCombo, 1)
        result_layout.addWidget(result_selector)

        self.plotHintLabel = CaptionLabel("", self)
        self.plotHintLabel.setWordWrap(True)
        result_layout.addWidget(self.plotHintLabel)
        result_layout.addWidget(self._plot_adapter.widget(), 10)

        self.statusLabel = CaptionLabel("", self)
        self.statusLabel.setWordWrap(True)
        result_layout.addWidget(self.statusLabel)

        root_layout.addWidget(result_frame)

        self.setMinimumWidth(720)
        self._plot_adapter.set_bin_click_callback(self._select_bin)
        self.groupCombo.currentIndexChanged.connect(self._on_group_mode_changed)
        self.scopeCombo.currentIndexChanged.connect(self._reload_fields)

        self._set_advanced_visible(False)
        self._on_group_mode_changed()
        self._reload_fields()

    def _set_advanced_visible(self, visible: bool) -> None:
        for widget in (
            self.selectModeLabel,
            self.selectModeCombo,
            self.binsLabel,
            self.binsSpin,
            self.curveLabel,
            self.curveCombo,
            self.includeNormCheck,
        ):
            widget.setVisible(bool(visible))

    def _on_group_mode_changed(self) -> None:
        """Show/hide widgets based on the selected group mode."""
        mode = str(self.groupCombo.currentData() or "")
        is_value_view = mode == DistributionGroupMode.VALUE_VIEW.value
        is_custom = mode == DistributionGroupMode.CUSTOM.value
        is_legacy = mode in {
            DistributionGroupMode.FORMULA.value,
            DistributionGroupMode.ELEMENT.value,
        }

        # Custom group editor
        for w in (self.customGroupLabel, self.customGroupBtn):
            w.setVisible(is_custom)

        # Value source checkboxes:
        # - Always visible in VALUE_VIEW mode (multi-select)
        # - In CUSTOM mode: visible, but single-select when >1 group, multi when <=1
        show_value_checks = is_value_view or is_custom
        for w in (self.valueViewsLabel, self.refCheck, self.predCheck, self.errCheck):
            w.setVisible(show_value_checks)
        for w in (self.viewLabel, self.viewCombo):
            w.setVisible(is_legacy)

        if is_custom:
            active_groups = [g for g in self._custom_groups if g.get("enabled", True)]
            if len(active_groups) > 1:
                checked = [w for w in (self.refCheck, self.predCheck, self.errCheck) if w.isChecked()]
                if not checked:
                    self.refCheck.setChecked(True)
                elif len(checked) > 1:
                    for widget in checked[1:]:
                        widget.setChecked(False)

    def _on_value_view_check_changed(self) -> None:
        """Ensure at least one value view is checked; force single-select for CUSTOM multi-group."""
        mode = str(self.groupCombo.currentData() or "")
        if mode == DistributionGroupMode.CUSTOM.value:
            active_groups = [g for g in self._custom_groups if g.get("enabled", True)]
            if len(active_groups) > 1:
                # Force single-select: keep the last toggled one
                sender = self.sender()
                if sender is self.refCheck and self.refCheck.isChecked():
                    self.predCheck.setChecked(False)
                    self.errCheck.setChecked(False)
                elif sender is self.predCheck and self.predCheck.isChecked():
                    self.refCheck.setChecked(False)
                    self.errCheck.setChecked(False)
                elif sender is self.errCheck and self.errCheck.isChecked():
                    self.refCheck.setChecked(False)
                    self.predCheck.setChecked(False)
                # Ensure at least one
                if not any((self.refCheck.isChecked(), self.predCheck.isChecked(), self.errCheck.isChecked())):
                    self.refCheck.setChecked(True)
                return
        if not any((self.refCheck.isChecked(), self.predCheck.isChecked(), self.errCheck.isChecked())):
            self.refCheck.setChecked(True)

    def _edit_custom_groups(self) -> None:
        """Open a dialog to manage custom groups with checkboxes and filters."""
        from NepTrainKit.core.types import StructureFilterSpec

        dlg = MessageBoxBase(self)
        dlg.titleLabel = CaptionLabel(self.tr("Edit custom groups"), dlg)
        dlg.viewLayout.addWidget(dlg.titleLabel)

        groups_list = list(self._custom_groups)
        group_widgets: list[tuple[Any, Any, Any]] = []

        container = QWidget(dlg)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        def _add_group_row(label: str = "", spec: dict | None = None, enabled: bool = True):
            row = QWidget(container)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            check = CheckBox(row)
            check.setChecked(enabled)
            check.setToolTip(self.tr("Include in plot"))
            name_edit = LineEdit(row)
            name_edit.setPlaceholderText(self.tr("Group name"))
            name_edit.setText(label)
            filter_btn = PushButton(self.tr("Filter"), row)
            row_layout.addWidget(check)
            row_layout.addWidget(name_edit, 1)
            row_layout.addWidget(filter_btn)
            container_layout.addWidget(row)
            stored_spec = spec or {"logic": "all", "conditions": []}
            group_widgets.append((check, name_edit, stored_spec))
            row._stored_spec = stored_spec

            def _open_filter(_checked=False, _row=row):
                from NepTrainKit.ui.widgets.structure_filter_bar import StructureFilterEditorPopup
                popup = StructureFilterEditorPopup(self)
                # Build suggestions from loaded data (same as main panel)
                if self._data is not None:
                    from NepTrainKit.core.types import SearchType as _ST
                    max_items = 50000
                    suggestions = {}
                    for st in (_ST.TAG, _ST.FORMULA, _ST.ELEMENTS, _ST.EXPRESSION):
                        try:
                            if hasattr(self._data, "has_completer_cache") and self._data.has_completer_cache(st, max_items=max_items):
                                suggestions[st] = self._data.get_completer_cache(st, max_items=max_items)
                        except Exception:
                            pass
                    popup.set_suggestions(suggestions)
                try:
                    popup.set_spec(StructureFilterSpec.from_dict(getattr(_row, "_stored_spec", {"logic": "all", "conditions": []})))
                except Exception:
                    pass
                popup.specChanged.connect(lambda s, _row=_row: _save_spec(s, _row))
                # Center popup on screen
                from PySide6.QtWidgets import QApplication as _QApp
                screen = _QApp.primaryScreen().availableGeometry()
                popup.adjustSize()
                popup.show()
                pw, ph = popup.width(), popup.height()
                popup.move(screen.center().x() - pw // 2, screen.center().y() - ph // 2)

            def _save_spec(spec, _row):
                _row._stored_spec = spec.to_dict()

            filter_btn.clicked.connect(_open_filter)

        for g in groups_list:
            _add_group_row(
                g.get("label", ""),
                g.get("spec", {"logic": "all", "conditions": []}),
                g.get("enabled", True),
            )

        add_btn = PushButton(self.tr("+ Add group"), container)
        add_btn.clicked.connect(lambda: _add_group_row())
        container_layout.addWidget(add_btn)

        dlg.viewLayout.addWidget(container)
        dlg.yesButton.setText(self.tr("OK"))
        dlg.cancelButton.setText(self.tr("Cancel"))
        dlg.widget.setMinimumWidth(420)

        if dlg.exec():
            self._custom_groups = []
            for check, name_edit, spec in group_widgets:
                label = name_edit.text().strip()
                if label:
                    row_widget = name_edit.parent()
                    stored = getattr(row_widget, "_stored_spec", spec) if row_widget else spec
                    self._custom_groups.append({
                        "label": label,
                        "spec": stored,
                        "enabled": check.isChecked(),
                    })
            self._on_group_mode_changed()


    def set_context(
        self,
        *,
        data=None,
        run_analysis_callback=None,
        apply_selection_callback=None,
    ) -> None:
        """Replace the dataset and callbacks without rebuilding the explorer UI."""
        self._data = data
        self._run_analysis_callback = run_analysis_callback
        self._apply_selection_callback = apply_selection_callback
        self._analysis = {}
        self._metric_by_key.clear()
        self.metricCombo.clear()
        self.seriesCombo.clear()
        self._plot_adapter.clear()
        self._reload_fields()

    def _reload_fields(self) -> None:
        self.fieldCombo.clear()
        self._field_specs = []
        self._field_by_key.clear()
        if self._vispy_fallback_warned:
            self.plotHintLabel.setText(
                self.tr("Current canvas backend is vispy, but vispy plot failed to initialize; fallback to pyqtgraph.")
            )
        else:
            self.plotHintLabel.setText("")

        if self._data is None or not hasattr(self._data, "discover_atomic_numeric_fields"):
            self.statusLabel.setText(self.tr("Dataset does not support distribution analysis."))
            return

        try:
            specs = self._data.discover_atomic_numeric_fields(
                scope=DistributionScope(str(self.scopeCombo.currentData() or DistributionScope.ACTIVE.value))
            )
        except Exception:  # noqa: BLE001
            specs = []
        self._field_specs = list(specs or [])

        for spec in self._field_specs:
            key = str(getattr(spec, "key", "") or "")
            if not key:
                continue
            source = str(getattr(spec, "source", ""))
            label = str(getattr(spec, "label", key) or key)
            shape = getattr(spec, "shape", None)
            shape_text = shape.value if hasattr(shape, "value") else str(shape or "")
            unit = str(getattr(spec, "unit_guess", "unknown") or "unknown")
            display = f"[{source}] {label} ({shape_text}, unit={unit})"
            self.fieldCombo.addItem(display, userData=key)
            self._field_by_key[key] = spec

        if self.fieldCombo.count() == 0:
            self.statusLabel.setText(self.tr("No numeric fields found in current scope."))
        else:
            self.statusLabel.setText(
                self.tr("{count} fields ready. Click Analyze.").format(count=self.fieldCombo.count())
            )

    def _current_field_key(self) -> str:
        data = self.fieldCombo.currentData()
        if data:
            return str(data)
        text = self.fieldCombo.currentText().strip()
        for i in range(self.fieldCombo.count()):
            if text == self.fieldCombo.itemText(i):
                return str(self.fieldCombo.itemData(i) or "")
        return ""

    def _run_analysis(self) -> None:
        field_key = self._current_field_key()
        if not field_key:
            self.statusLabel.setText(self.tr("Please select a field."))
            return
        if self._run_analysis_callback is None:
            self.statusLabel.setText(self.tr("Analyze callback is unavailable."))
            return

        group_mode = DistributionGroupMode(
            str(self.groupCombo.currentData() or DistributionGroupMode.ELEMENT.value)
        )

        # Determine selected value views
        # - VALUE_VIEW: always use checkboxes
        # - CUSTOM with <=1 enabled group: also use checkboxes (multi-value on single group)
        # - CUSTOM with >1 enabled groups: only reference (single value per group)
        selected_views: list[str] = []
        active_custom_count = 0
        if group_mode == DistributionGroupMode.CUSTOM:
            active_custom_count = sum(1 for g in self._custom_groups if g.get("enabled", True))

        value_view = DistributionValueView.REFERENCE
        if group_mode in {DistributionGroupMode.FORMULA, DistributionGroupMode.ELEMENT}:
            value_view = DistributionValueView(
                str(self.viewCombo.currentData() or DistributionValueView.REFERENCE.value)
            )

        use_value_checks = group_mode == DistributionGroupMode.VALUE_VIEW or (
            group_mode == DistributionGroupMode.CUSTOM and active_custom_count <= 1
        )
        if use_value_checks:
            if self.refCheck.isChecked():
                selected_views.append(DistributionValueView.REFERENCE.value)
            if self.predCheck.isChecked():
                selected_views.append(DistributionValueView.PREDICTION.value)
            if self.errCheck.isChecked():
                selected_views.append(DistributionValueView.ERROR.value)
            if not selected_views:
                selected_views.append(DistributionValueView.REFERENCE.value)
        elif group_mode == DistributionGroupMode.CUSTOM:
            if self.predCheck.isChecked():
                value_view = DistributionValueView.PREDICTION
            elif self.errCheck.isChecked():
                value_view = DistributionValueView.ERROR

        # Determine custom group specs for CUSTOM mode (only enabled groups)
        custom_specs: list[dict] = []
        if group_mode == DistributionGroupMode.CUSTOM:
            custom_specs = [g for g in self._custom_groups if g.get("enabled", True)]

        req = DistributionRequest(
            field_keys=(field_key,),
            include_norm=bool(self.includeNormCheck.isChecked()),
            value_view=value_view,
            group_mode=group_mode,
            scope=DistributionScope(str(self.scopeCombo.currentData() or DistributionScope.ACTIVE.value)),
            bins=int(self.binsSpin.value()),
            select_mode=DistributionSelectMode(
                str(self.selectModeCombo.currentData() or DistributionSelectMode.REPLACE.value)
            ),
            groups=(),
            curve_style=DistributionCurveStyle(str(self.curveCombo.currentData() or DistributionCurveStyle.KDE.value)),
            curve_points=240,
            selected_value_views=tuple(selected_views),
            custom_group_specs=tuple(custom_specs),
        )
        try:
            analysis = self._run_analysis_callback(req)
        except Exception:  # noqa: BLE001
            analysis = {}

        self._analysis = dict(analysis or {})
        self._metric_by_key.clear()
        self.metricCombo.clear()
        self.seriesCombo.clear()
        self._plot_adapter.clear()

        metrics = self._analysis.get("metrics", []) or []
        for metric in metrics:
            m_key = str(metric.get("metric_key", "") or "")
            if not m_key:
                continue
            self._metric_by_key[m_key] = metric
            label = (
                f"{metric.get('field_label', metric.get('field_key', ''))}"
                f" :: {metric.get('component', '')}"
                f" [{metric.get('value_view', 'reference')}]"
            )
            self.metricCombo.addItem(label, userData=m_key)

        msgs = self._analysis.get("messages", []) or []
        if self.metricCombo.count() == 0:
            base = self.tr("No metrics produced for current request.")
            if msgs:
                base += f" {msgs[0]}"
            self.statusLabel.setText(base)
            return

        self._refresh_series_combo()
        status = self.tr("{count} metrics generated.").format(count=self.metricCombo.count())
        if msgs:
            status += f" {msgs[0]}"
        self.statusLabel.setText(status)

    def _current_metric(self) -> dict[str, Any] | None:
        m_key = str(self.metricCombo.currentData() or "")
        if not m_key:
            return None
        return self._metric_by_key.get(m_key)

    def _current_series(self, metric: dict[str, Any] | None) -> dict[str, Any] | None:
        if metric is None:
            return None
        s_key = str(self.seriesCombo.currentData() or "")
        if not s_key:
            return None
        if s_key == self._ALL_SERIES_KEY:
            return {"series_key": self._ALL_SERIES_KEY, "name": self.tr("All groups")}
        for item in metric.get("series", []) or []:
            if str(item.get("series_key", item.get("name", "")) or "") == s_key:
                return item
        return None

    def _refresh_series_combo(self) -> None:
        self.seriesCombo.clear()
        metric = self._current_metric()
        if metric is None:
            self._plot_adapter.clear()
            return
        series = metric.get("series", []) or []
        if len(series) > 1:
            self.seriesCombo.addItem(self.tr("All groups (overlay)"), userData=self._ALL_SERIES_KEY)
        for item in series:
            s_key = str(item.get("series_key", item.get("name", "")) or "")
            self.seriesCombo.addItem(str(item.get("name", s_key)), userData=s_key)
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        metric = self._current_metric()
        series = self._current_series(metric)
        self._plot_adapter.set_payload(metric, series)

    def _select_bin(self, bin_index: int) -> None:
        if self._data is None or self._apply_selection_callback is None:
            return
        metric = self._current_metric()
        if metric is None:
            return
        analysis_id = int(self._analysis.get("analysis_id", 0) or 0)
        if analysis_id <= 0:
            return
        metric_key = str(metric.get("metric_key", "") or "")
        series_key = str(self.seriesCombo.currentData() or "")
        if not metric_key or not series_key:
            return

        indices: list[int] = []
        sample_count = 0
        if series_key == self._ALL_SERIES_KEY:
            merged: set[int] = set()
            for item in metric.get("series", []) or []:
                s_key = str(item.get("series_key", item.get("name", "")) or "")
                if not s_key:
                    continue
                hist = list(item.get("hist", []) or [])
                if 0 <= int(bin_index) < len(hist):
                    sample_count += int(hist[int(bin_index)] or 0)
                try:
                    vals = self._data.resolve_distribution_bin_indices(
                        analysis_id, metric_key, s_key, int(bin_index)
                    )
                except Exception:  # noqa: BLE001
                    vals = []
                merged.update(int(i) for i in vals)
            indices = sorted(merged)
        else:
            try:
                indices = self._data.resolve_distribution_bin_indices(
                    analysis_id, metric_key, series_key, int(bin_index)
                )
            except Exception:  # noqa: BLE001
                indices = []
            series = self._current_series(metric)
            if series is not None:
                hist = list(series.get("hist", []) or [])
                if 0 <= int(bin_index) < len(hist):
                    sample_count = int(hist[int(bin_index)] or 0)

        mode = str(self.selectModeCombo.currentData() or DistributionSelectMode.REPLACE.value)
        self._apply_selection_callback(list(indices), mode)
        series_label = self.tr("all groups") if series_key == self._ALL_SERIES_KEY else series_key
        self.statusLabel.setText(
            self.tr("Applied bin {bin_index} ({series_label}): {sample_count} samples -> {count} structures, mode='{mode}'.").format(
                bin_index=bin_index,
                series_label=series_label,
                sample_count=sample_count,
                count=len(indices),
                mode=mode,
            )
        )


class DistributionInspectorMessageBox(QDialog):
    """Compatibility window around :class:`DistributionExplorerWidget`."""

    def __init__(
        self,
        parent=None,
        data=None,
        run_analysis_callback=None,
        apply_selection_callback=None,
        canvas_type: str | None = None,
    ):
        super().__init__(parent)
        self._data = data
        self.setWindowTitle(self.tr("Distribution inspector"))
        self.setWindowIcon(QIcon(":/images/src/images/distribution_inspector.svg"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.explorer = DistributionExplorerWidget(
            self,
            data=data,
            run_analysis_callback=run_analysis_callback,
            apply_selection_callback=apply_selection_callback,
            canvas_type=canvas_type,
        )
        layout.addWidget(self.explorer)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setMinimumWidth(720)
        self.setMaximumWidth(840)
        self.resize(780, 620)
