"""Per-frame regression changes against one explicitly chosen baseline."""
from functools import partial
from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, ComboBox, FluentIcon, PushButton, SegmentedWidget, TransparentToolButton

from NepTrainKit import palette
from NepTrainKit.core.validation.accuracy import compare_frame_errors
from NepTrainKit.ui.widgets.eos_plot import energy_curve_canvas
from NepTrainKit.ui.widgets.validation_values import TABLE_ICON, values_dialog, values_table


QUANTITIES = (("force", "Force", "meV/Å"), ("energy", "Energy", "meV/atom"),
              ("virial", "Virial", "meV/atom"), ("mforce", "Magnetic force", "meV/μB"))


def accuracy_controls(panel):
    tools = QWidget(panel)
    layout = QHBoxLayout(tools)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    panel.accuracy_quantity = "force"
    panel.accuracy_picker = SegmentedWidget(tools)

    def choose(key):
        panel.accuracy_quantity = key
        panel._redraw()

    for key, title, _ in QUANTITIES:
        panel.accuracy_picker.addItem(key, panel.tr(title), lambda checked=False, q=key: choose(q))
    layout.addWidget(panel.accuracy_picker)
    layout.addStretch(1)
    panel.accuracy_filter = ComboBox(tools)
    panel.accuracy_filter.setAccessibleName(panel.tr("Configuration group"))
    panel.accuracy_filter.setMinimumWidth(130)
    panel.accuracy_filter.setMaximumWidth(240)
    panel.accuracy_filter.currentIndexChanged.connect(lambda *_: panel._redraw())
    layout.addWidget(panel.accuracy_filter)
    tools.hide()
    return tools


def _frame_text(panel, point, unit):
    text = panel.tr("Frame {frame} (zero based)").format(frame=point["frame"])
    text += f" · {panel.tr('Error')} {point['error']:.4g} {unit}"
    if point["baseline"] is not None:
        text += f" · {panel.tr('Baseline')} {point['baseline']:.4g}"
    return text


def _pin_frame(panel, readout, points, unit, curve, index):
    point = next(p for p in points if p["rank"] == curve.xData[index])
    readout.setText(_frame_text(panel, point, unit) + (f" · Δ {point['delta']:+.4g} {unit}" if point["baseline"] is not None else ""))
    readout.show()


def _show_frames(panel, name, points, groups, unit, statistics=None):
    dialog, layout = values_dialog(panel, name + " · " + panel.tr("Frame errors"))
    layout.addWidget(BodyLabel(panel.tr("Largest deterioration first") if any(p["baseline"] is not None for p in points)
                               else panel.tr("Largest error first"), dialog))
    table = values_table(dialog)
    headers = [panel.tr("Frame (zero based)"), panel.tr("Rank"), panel.tr("Error") + f" ({unit})",
               panel.tr("Baseline") + f" ({unit})", "Δ " + f"({unit})"] + list(groups)
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    ordered = sorted(points, key=lambda p: (-p["delta"], p["frame"]))
    table.setRowCount(len(ordered))
    for row, point in enumerate(ordered):
        values = [point["frame"], point["rank"], point["error"], point["baseline"],
                  point["delta"] if point["baseline"] is not None else None]
        values += [labels[point["frame"]] for labels in groups.values()]
        for column, value in enumerate(values):
            cell = QTableWidgetItem()
            cell.setData(Qt.ItemDataRole.DisplayRole, "—" if value is None else value)
            cell.setToolTip(str(value) if value is not None else "—")
            table.setItem(row, column, cell)
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSortIndicator(4 if any(p["baseline"] is not None for p in points) else 2,
                                             Qt.SortOrder.DescendingOrder)
    table.setSortingEnabled(True)
    layout.addWidget(table)
    if statistics is not None:
        button = PushButton(panel.tr("Dataset statistics and export"), dialog, icon=TABLE_ICON)
        button.clicked.connect(lambda checked=False: statistics())
        layout.addWidget(button, 0, Qt.AlignmentFlag.AlignRight)
    bounds = dialog.screen().availableGeometry()
    dialog.resize(min(1000, bounds.width() - 32), min(600, bounds.height() - 64))
    dialog.exec()


def draw_accuracy_cards(panel, benchmark, entries):
    from NepTrainKit.ui.widgets.reference_response_plot import _show_values

    panel.legend_panel.hide()
    panel.title_label.hide()
    panel.info_label.hide()
    panel.accuracy_tools.show()
    groups = next((d.get("frame_groups", {}) for _, _, _, d in entries if d.get("frame_groups")),
                  benchmark["summary"].get("frame_groups", {}))
    options = [(panel.tr("All configurations"), None)]
    for field, labels in groups.items():
        for label in sorted(set(labels), key=lambda v: (v is None, v or "")):
            options.append((f"{field} · {label or panel.tr('Unassigned')}", (field, label)))
    combo = panel.accuracy_filter
    previous = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for title, key in options:
        combo.addItem(title, userData=key)
    combo.setCurrentIndex(next((i for i, (_, key) in enumerate(options) if key == previous), 0))
    combo.blockSignals(False)
    combo.setVisible(len(options) > 1)
    selection = combo.currentData()
    selected = [i for i, label in enumerate(groups[selection[0]]) if label == selection[1]] if selection else None
    ready = {key: details for key, _, run, details in entries if run and run["status"] == "succeeded" and details}
    available = {q for details in ready.values() for q, values in details.get("frame_errors", {}).items()
                 if any(v is not None for v in values)}
    if panel.accuracy_quantity not in available and available:
        panel.accuracy_quantity = next(key for key, _, _ in QUANTITIES if key in available)
    for key, _, _ in QUANTITIES:
        panel.accuracy_picker.items[key].setEnabled(key in available)
    panel.accuracy_picker.setCurrentItem(panel.accuracy_quantity)
    quantity, quantity_title, unit = next(q for q in QUANTITIES if q[0] == panel.accuracy_quantity)
    baseline = ready.get(panel.baseline_id, {}).get("frame_errors", {}).get(quantity)
    if baseline is not None and not any(v is not None for v in baseline):
        baseline = None
    baseline_name = next((name for key, name, _, _ in entries if key == panel.baseline_id), "")
    targets = [e for e in entries if e[0] not in panel._hidden_series and e[0] != panel.baseline_id]
    if not targets:
        targets = [e for e in entries if e[0] not in panel._hidden_series]
    all_values = []
    for key, name, run, details in targets:
        card = QFrame(panel.grid_container)
        card.setObjectName("accuracyComparisonCard")
        color = panel._series_pen(key).color().name()
        card.setStyleSheet(f"#accuracyComparisonCard {{background:{palette.SURFACE};border:1px solid {palette.BORDER};border-radius:10px;}}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 8, 6)
        layout.setSpacing(4)
        heading = QHBoxLayout()
        title = BodyLabel(card)
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setText(f'<span style="color:{color}">●</span> <b>{escape(name)}</b>')
        title.setMinimumWidth(0)
        heading.addWidget(title, 1)
        reset = TransparentToolButton(FluentIcon.SYNC, card)
        reset.setToolTip(panel.tr("Reset view"))
        reset.setAccessibleName(reset.toolTip())
        heading.addWidget(reset)
        data = TransparentToolButton(TABLE_ICON, card)
        data.setToolTip(panel.tr("Frame errors"))
        data.setAccessibleName(data.toolTip())
        heading.addWidget(data)
        layout.addLayout(heading)
        errors = ready.get(key, {}).get("frame_errors", {}).get(quantity, [])
        paired = baseline is not None and key != panel.baseline_id
        points = compare_frame_errors(errors, baseline if paired else None, selected) if errors else []
        description = CaptionLabel(card)
        description.setWordWrap(True)
        if paired:
            improved = sum(p["delta"] < -1e-9 for p in points)
            worse = sum(p["delta"] > 1e-9 for p in points)
            description.setText(f"{panel.tr('Baseline')} · {baseline_name}  |  " + panel.tr("{count} paired frames").format(count=len(points)))
            stats = CaptionLabel(card)
            stats.setTextFormat(Qt.TextFormat.RichText)
            stats.setText(f'<span style="color:#16875b">↓ {panel.tr("Improved")} {improved} ({improved / max(1, len(points)):.0%})</span>　'
                          f'<span style="color:#c03936">↑ {panel.tr("Worse")} {worse} ({worse / max(1, len(points)):.0%})</span>　'
                          + panel.tr("Unchanged {count}").format(count=len(points) - improved - worse))
            layout.addWidget(stats)
        else:
            description.setText(panel.tr("Absolute frame errors · {count} frames").format(count=len(points)))
            description.setToolTip(panel.tr("Choose and evaluate a baseline model to compare frame errors."))
        roles = {"training": "Training regression", "development": "Development validation",
                 "validation": "Validation (user declared)", "unknown": "Not declared"}
        description.setText(description.text() + " · " + panel.tr(roles[benchmark["summary"]["analysis"]["role"]]))
        layout.addWidget(description)
        xlabel = panel.tr("Rank · baseline error descending") if paired else panel.tr("Rank · model error descending")
        metric = panel.tr("Absolute energy error") if quantity == "energy" else panel.tr("Frame RMS")
        ylabel = (panel.tr("{metric} change").format(metric=metric) if paired else metric) + f" ({unit})"
        plot, readout = energy_curve_canvas(panel, card, layout, reset, xlabel, ylabel,
                                            panel.tr("Rank {}"), unit, paired)
        plot.comparison_points = points
        if points:
            curve = plot.plot([p["rank"] for p in points], [p["delta"] for p in points], pen=None,
                              symbol="o", symbolSize=7, symbolBrush=color, symbolPen=color, name=escape(name))
            panel._register_curve(key, curve)
            plot.category_labels = {p["rank"]: _frame_text(panel, p, unit) for p in points}
            if readout is not None:
                curve.read_point = partial(_pin_frame, panel, readout, points, unit)
            all_values.extend(p["delta"] for p in points)
            plot.finish()
        else:
            plot.hide()
            if paired:
                plot.residual_placeholder.hide()
            empty = BodyLabel(panel.tr("No valid frame errors. Double-click the item to evaluate it."), card)
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(empty, 1)
        data.setEnabled(bool(points))
        statistics = partial(_show_values, panel, benchmark["summary"],
                             list(range(len(benchmark["summary"]["rows"]))), entries)
        data.clicked.connect(lambda checked=False, n=name + " · " + panel.tr(quantity_title), p=points:
                             _show_frames(panel, n, p, groups, unit, statistics))
        panel.eos_cards.append(card)
        panel.eos_plots.append(plot)
    # Same limits across cards are essential for comparing the size of a regression.
    if all_values:
        lo, hi = min(0., min(all_values)), max(0., max(all_values))
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        for plot in panel.eos_plots:
            plot.comparison_y_range = (lo, hi)
            plot.fit()
