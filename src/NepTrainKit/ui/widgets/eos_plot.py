"""EOS chart cards and compact names for independent volume scans."""

import re
from functools import partial
from html import escape
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget
from qfluentwidgets import FlowLayout, BodyLabel, CaptionLabel, FluentIcon, TransparentToolButton

from NepTrainKit import palette
from NepTrainKit.ui.canvas.curve_factory import create_curve_plot
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets.validation_model_picker import GroupMarkerButton
from NepTrainKit.ui.widgets.validation_plot_hover import PlotPointHover
from NepTrainKit.ui.widgets.validation_values import TABLE_ICON, show_curve_values


EOS_POINT_SIZE = 10
EOS_REFERENCE_WIDTH = 2.6


def eos_group_title(name, formula):
    """Keep the occupancy seed visible when a formula has multiple EOS branches."""
    if not formula or not name.startswith(formula + "_"):
        return name, ""
    branch = re.sub(r"seed(\d{7,})", lambda m: "seed…" + m[1][-4:], name[len(formula) + 1:])
    seed = re.search(r"(?:^|_)(seed[…\d]+)$", branch, re.IGNORECASE)
    if seed:
        return f"{formula} · {seed[1]}", branch[:seed.start()].replace("_", " ")
    return name.replace("_", " "), ""


def _read_eos_point(panel, label, name, reference, predicted, offset, curve, index):
    volume = curve.xData[index]
    fields = [name, f"V = {volume:.4f} Å³/atom"]
    if predicted is not None:
        fields.append(panel.tr("Predicted {value:.6f} eV/atom").format(value=predicted[index]))
    if reference is not None:
        fields.append(panel.tr("Reference {value:.6f} eV/atom").format(value=reference[index]))
        if predicted is not None:
            delta = (predicted[index] - reference[index] - offset) * 1000
            delta_label = panel.tr("Aligned ΔE") if panel.offset_checkbox.isChecked() else "ΔE"
            fields.append(f"{delta_label} = {delta:+.3f} meV/atom")
    label.setText("  ·  ".join(fields))
    label.setToolTip(label.text())
    label.show()


def _eos_canvas(panel, card, layout, reset, residual, relative):
    unit = "meV/atom" if residual or relative else "eV/atom"
    ylabel = (panel.tr("Residual ΔE (meV/atom)") if residual else
              panel.tr("Relative energy (meV/atom)") if relative else panel.tr("Energy (eV/atom)"))
    return energy_curve_canvas(panel, card, layout, reset, panel.tr("Volume (Å³/atom)"), ylabel,
                               "V = {:.4f} Å³/atom", unit, residual)


def energy_curve_canvas(panel, card, layout, reset, xlabel, ylabel, x_format, unit, residual):
    plot = create_curve_plot(card)
    plot.viewport().installEventFilter(panel)
    reset.clicked.connect(plot.fit)
    plot.axis_titles = (xlabel, ylabel)
    plot.configure(xlabel, ylabel, False, panel._series_pen(None) if residual else None)
    readout = None
    if panel._adaptive_layout:
        plot.point_hover = PlotPointHover(plot, x_format, unit)
        readout = CaptionLabel(card)
        readout.hide()
        readout.setObjectName("eosPointReadout")
        readout.setWordWrap(True)
        readout.setTextFormat(Qt.TextFormat.PlainText)
        readout.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
    plot.setMinimumHeight(190)
    plot.setMinimumWidth(0)
    layout.addWidget(plot, 1)
    if residual:
        plot.residual_placeholder = CaptionLabel(card)
        plot.residual_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot.residual_placeholder.setWordWrap(True)
        layout.addWidget(plot.residual_placeholder, 1)
        plot.residual_placeholder.hide()
    if readout is not None:
        layout.addWidget(readout)
    return plot, readout


def update_residual_placeholder(plot):
    label = getattr(plot, "residual_placeholder", None)
    if label is None:
        return
    curves = [c for c in plot.listDataItems() if c.xData is not None and len(c.xData)]
    visible = any(c.isVisible() for c in curves)
    if not visible:
        label.setText(QCoreApplication.translate("RunDetailPanel", "Show a model curve above to see its residuals.") if curves else
                      QCoreApplication.translate("RunDetailPanel", "No paired prediction and reference values. Double-click the validation item to evaluate it."))
    plot.setVisible(visible)
    label.setVisible(not visible)


def _plot_group(panel, plot, readout, benchmark, entries, residual, relative, *, group_name=""):
    summary = benchmark["summary"]
    symbol = panel.group_markers.get(benchmark["id"], "o")

    prefix = group_name + " · " if group_name else ""

    zero = min(summary["energies"]) if relative else 0.0
    scale = 1000.0 if relative else 1.0
    reference = [(value - zero) * scale for value in summary["energies"]]
    if not residual:
        curve = plot.plot(
            summary["volumes"], reference, pen=panel._series_pen(None),
            symbol=symbol, symbolBrush=palette.PLOT_AXIS, symbolPen=palette.PLOT_AXIS,
            symbolSize=EOS_POINT_SIZE if panel._adaptive_layout else 5, name=panel.tr("Reference"),
        )
        curve.setProperty("groupId", benchmark["id"])
        panel._register_curve(None, curve)
        if panel._adaptive_layout:
            curve.read_point = partial(_read_eos_point, panel, readout, prefix + panel.tr("Reference"),
                                                   summary["energies"], None, 0)
        _draw_eos_fit(panel, plot, None, summary["volumes"], summary["energies"],
                      summary.get("fit", {}), zero, 0., scale)
    for key, name, run, details in entries:
        if not run or run["status"] != "succeeded" or not details:
            continue
        offset = details.get("offset_ev_atom", 0.0) if panel.offset_checkbox.isChecked() else 0.0
        predicted = details.get("predicted", [])
        volumes = details.get("volumes", [])
        ref = details.get("reference")
        if ref is None and volumes == summary["volumes"]:
            ref = summary["energies"]
        if ref is not None and len(ref) != len(predicted):
            ref = None
        if residual:
            if ref is None:
                continue
            energies = [(p - r - offset) * 1000 for p, r in zip(predicted, ref)]
        else:
            energies = [(value - zero - offset) * scale for value in predicted]
        pen = panel._series_pen(key)
        curve = plot.plot(volumes, energies, pen=pen, name=escape(name),
                          symbol=symbol if panel._adaptive_layout or benchmark["id"] in panel.group_markers else None, symbolSize=EOS_POINT_SIZE if panel._adaptive_layout else 7,
                          symbolBrush=pen.color(), symbolPen=pen.color())
        curve.setProperty("groupId", benchmark["id"])
        panel._register_curve(key, curve)
        if panel._adaptive_layout:
            curve.read_point = partial(_read_eos_point, panel, readout, prefix + name, ref, predicted, offset)
        if not residual:
            _draw_eos_fit(panel, plot, key, volumes, predicted, details.get("model_fit", {}), zero, offset, scale)


def _draw_eos_fit(panel, plot, key, volumes, energies, fit, zero, offset, scale):
    from NepTrainKit.core.validation.protocols.eos import eos_fit_curve

    sampled = eos_fit_curve(volumes, energies, fit)
    if sampled is None:
        return
    x, y = sampled
    y = (y - zero - offset) * scale
    pen = QPen(panel._series_pen(key))
    pen.setWidthF(1.4)
    pen.setStyle(Qt.PenStyle.DotLine)
    line = plot.plot(x, y, pen=pen, auxiliary=True)
    line.setProperty("annotation", "eos_fit")
    panel._register_curve(key, line)
    marker = plot.plot([fit["v0"]], [(fit["e0"] - zero - offset) * scale], pen=None,
                       symbol="d", symbolSize=12, symbolPen=pen.color(), symbolBrush=pen.color(), auxiliary=True)
    marker.setProperty("annotation", "equilibrium")
    panel._register_curve(key, marker)
    guide = plot.plot([fit["v0"], fit["v0"]], [min(y), max(y)], pen=pen, auxiliary=True)
    guide.setProperty("annotation", "equilibrium_volume")
    panel._register_curve(key, guide)


def _missing_status(panel, name, run, details):
    if not run and "previous_version" in details:
        return panel.tr("{name}: v{version} result · double-click the item to update").format(
            name=name, version=details["previous_version"])
    if not run:
        return panel.tr("{name}: not run").format(name=name)
    if run["status"] != "succeeded" or not details:
        return panel.tr("{name}: unavailable").format(name=name)
    return ""


class GroupStrip(QScrollArea):
    """Wrap group labels and reserve at most two rows of chart space."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidget(QWidget())
        self.flow = FlowLayout(self.widget(), needAni=False)
        self.flow.setContentsMargins(0, 0, 0, 0)
        self.flow.setHorizontalSpacing(8)
        self.flow.setVerticalSpacing(3)

    def fit_groups(self):
        if not self.flow.count():
            return
        width = self.width()
        limit = 2 * self.flow.itemAt(0).widget().sizeHint().height() + self.flow.verticalSpacing()
        if self.flow.heightForWidth(width) > limit:
            width -= self.verticalScrollBar().sizeHint().width()
        height = self.flow.heightForWidth(width)
        self.widget().resize(width, height)
        self.flow.setGeometry(self.widget().rect())
        self.setFixedHeight(min(height, limit))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.fit_groups()


def add_overlay_group(panel, strip, key, name, tooltip):
    chip = QWidget(strip.widget())
    row = QHBoxLayout(chip)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    marker = GroupMarkerButton(panel.group_markers.get(key, "o"), chip)
    marker.setProperty("groupId", key)
    marker.markerChanged.connect(lambda symbol, group_id=key: panel.markerRequested.emit(group_id, symbol))
    row.addWidget(marker)
    label = BodyLabel(name, chip)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setToolTip(tooltip)
    label.setText(label.fontMetrics().elidedText(label.text(), Qt.TextElideMode.ElideMiddle, 235))
    row.addWidget(label)
    strip.flow.addWidget(chip)


def energy_action(heading, icon, name, text):
    button = TransparentToolButton(icon, heading.parentWidget())
    button.setObjectName(name)
    button.setToolTip(text)
    button.setAccessibleName(text)
    button.setFixedSize(24, 24)
    button.setIconSize(QSize(15, 15))
    heading.addWidget(button)
    return button


def energy_actions(panel, heading, key, *, sample_name=""):
    sample = energy_action(heading, QIcon(":/images/src/images/make.svg"), "eosSamplingButton",
                           panel.tr("Create sampling workflow"))
    if sample_name:
        sample.setToolTip(sample.toolTip() + " · " + sample_name)
        sample.setAccessibleName(sample.toolTip())
    sample.clicked.connect(lambda: panel.samplingRequested.emit(key))
    return energy_action(heading, TABLE_ICON, "energyValues", panel.tr("Curve values"))


def overlay_header(panel, card, layout, key, name):
    heading = QHBoxLayout()
    heading.setSpacing(4)
    layout.addLayout(heading)
    strip = GroupStrip(card)
    heading.addWidget(strip, 1)
    reset = energy_action(heading, FluentIcon.SYNC.colored(palette.ACCENT, palette.ACCENT),
                          "resetEosView", panel.tr("Reset view: fit all visible curves"))
    values = energy_actions(panel, heading, key, sample_name=name)
    return strip, reset, values


def _draw_overlay(panel, residual, relative):
    card = QFrame(panel.grid_container)
    card.setObjectName("eosOverlayCard")
    card.setStyleSheet(f"#eosOverlayCard {{ background: {palette.PLOT_BACKGROUND}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(4, 3, 4, 3)
    target = next((item for _, item, _ in panel._groups if item["id"] == panel.eos_current_id), panel._groups[0][1])
    strip, reset, values = overlay_header(panel, card, layout, target["id"], target["name"])
    plot, readout = _eos_canvas(panel, card, layout, reset, residual, relative)
    values.clicked.connect(lambda: show_curve_values(panel, plot))
    for tag, benchmark, entries in panel._groups:
        if not (benchmark.get("summary") or {}).get("volumes"):
            continue
        key = benchmark["id"]
        group = (benchmark.get("source_ref") or {}).get("eos_group") or {}
        title, branch = eos_group_title(str(group.get("value", benchmark["name"])), benchmark["summary"].get("formula"))
        name = " · ".join(filter(None, (title, branch)))
        missing = list(filter(None, (_missing_status(panel, n, r, d) for _, n, r, d in entries)))
        caption = name + " · " + panel.tr("Reference only") if missing and len(missing) == len(entries) else name
        add_overlay_group(panel, strip, key, caption, "\n".join([benchmark["name"], *missing]))
        _plot_group(panel, plot, readout, benchmark, entries, residual, relative,
                    group_name=name)
    strip.fit_groups()
    plot.finish()
    update_residual_placeholder(plot)
    panel.eos_cards.append(card)
    panel.eos_plots.append(plot)
    panel._layout_cards()


def energy_card_header(panel, card, card_layout, benchmark, key, tag, title_text, grouped, can_focus, focus_back, *, focus_key=None):
    heading = QHBoxLayout()
    card_layout.addLayout(heading)
    current = key == panel.eos_current_id
    role = panel.tr("Current group") if current else panel.tr("Comparison group")
    badge = CaptionLabel(role if grouped else tag, card)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setFixedSize(badge.fontMetrics().horizontalAdvance(role) + 12 if grouped else 30, 24)
    badge.setStyleSheet(
        f"color: {palette.INFO if current else palette.ACCENT_PRESSED}; background: {palette.INFO_SOFT if current else palette.ACCENT_SOFT}; border-radius: 5px; font-weight: 600;"
    )
    symbol = panel.group_markers.get(key, "o")
    if panel._adaptive_layout:
        card.setProperty("groupId", key)
        marker = GroupMarkerButton(symbol, card)
        marker.setProperty("groupId", key)
        marker.markerChanged.connect(lambda value, key=key: panel.markerRequested.emit(key, value))
        heading.addWidget(marker)
    heading.addWidget(badge)
    heading.setSpacing(4 if panel._adaptive_layout else 8)
    title = BodyLabel(title_text, card)
    title_font = title.font()
    title_font.setBold(True)
    title.setFont(title_font)
    title.setWordWrap(True)
    title.setToolTip(benchmark["name"])
    title.setTextFormat(Qt.TextFormat.PlainText)
    heading.addWidget(title, 1)
    reset = TransparentToolButton(FluentIcon.SYNC.colored(palette.ACCENT, palette.ACCENT), card)
    reset.setObjectName("resetEosView")
    reset.setToolTip(panel.tr("Reset view: fit all visible curves"))
    reset.setAccessibleName(panel.tr("Reset view"))
    heading.addWidget(reset)
    focus = TransparentToolButton(
        (FluentIcon.ZOOM_OUT if focus_back else FluentIcon.ZOOM_IN).colored(
            palette.PLOT_LOADED_OVERLAY, palette.PLOT_LOADED_OVERLAY
        ),
        card,
    )
    focus.setObjectName("toggleEosGroupFocus")
    focus.setToolTip(panel.tr("Back to all groups") if focus_back else panel.tr("Focus this group"))
    focus.setAccessibleName(focus.toolTip())
    if focus_back:
        focus.clicked.connect(lambda _checked=False: panel.allGroupsRequested.emit())
    else:
        focus.clicked.connect(lambda _checked=False, group=focus_key or tag: panel.groupFocused.emit(group))
    focus.setVisible(can_focus)
    heading.addWidget(focus)
    values = energy_actions(panel, heading, key)
    if grouped and not current:
        remove = TransparentToolButton(FluentIcon.CLOSE, card)
        remove.setObjectName("removeEosGroup")
        remove.setToolTip(panel.tr("Remove comparison group"))
        remove.setAccessibleName(remove.toolTip())
        remove.clicked.connect(lambda _checked=False, key=key: panel.groupRemoved.emit(key))
        heading.addWidget(remove)
    if panel._adaptive_layout:
        for tool in card.findChildren(TransparentToolButton):
            tool.setFixedSize(24, 24)
            tool.setIconSize(QSize(15, 15))
    return reset, values


def redraw_eos(panel):
    for card in panel.eos_cards:
        panel.grid_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
    panel.eos_cards = []
    panel.eos_plots = []
    panel._grid_columns = 0
    panel.offset_checkbox.setVisible(bool(panel._groups))
    panel.title_label.setText(panel.tr("EOS · {count} groups").format(count=len(panel._groups)))
    residual = panel._adaptive_layout and panel._eos_residual
    if panel._adaptive_layout and len(panel._groups) == 1:
        item = panel._groups[0][1]
        path = (item.get("source_ref") or {}).get("path")
        panel.title_label.setText(Path(path).name if panel.eos_current_id and path else item["name"])
    panel.relative_checkbox.setVisible(not residual and bool(panel._groups))
    relative = panel.relative_checkbox.isChecked() and not residual
    overlay = panel._adaptive_layout and panel._eos_overlay and len(panel._groups) > 1
    notes = [panel.tr("Shared axes · Color by model, marker by group") if overlay else panel.tr("Independent axes · Energy per atom")]
    if relative:
        notes.append(panel.tr("Reference minimum = 0"))
    if panel.offset_checkbox.isChecked():
        notes.append(panel.tr("Constant offset removed; shape comparison"))
    if len(panel._groups) == 1:
        benchmark = panel._groups[0][1]
        provenance = benchmark.get("provenance") or {}
        if provenance.get("source_kind"):
            notes.append(
                panel.tr("Reference: {source}, {conditions}").format(
                    source=provenance["source_kind"], conditions=provenance.get("conditions", "-")
                )
            )
        if provenance.get("notes") or provenance.get("conditions"):
            notes.append(
                panel.tr("Notes: {notes}").format(
                    notes=provenance.get("notes") or provenance["conditions"]
                )
            )
    panel.info_label.setText(" · ".join(notes))
    if overlay:
        _draw_overlay(panel, residual, relative)
        return
    for tag, benchmark, entries in panel._groups:
        summary = benchmark.get("summary") or {}
        if not summary.get("volumes"):
            continue
        card = QFrame(panel.grid_container)
        card.setObjectName("eosGroupCard")
        card.setStyleSheet(
            f"#eosGroupCard {{ background: {palette.PLOT_BACKGROUND}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(4, 3, 4, 3)
        card_layout.setSpacing(4)
        name = benchmark["name"].rsplit(" / ", 1)[-1]
        formula = summary.get("formula")
        group = (benchmark.get("source_ref") or {}).get("eos_group")
        grouped = panel.eos_current_id is not None and bool(group)
        title_text, branch = eos_group_title(str(group["value"]) if grouped else name, formula)
        if not grouped:
            title_text = formula or name
        reset, values = energy_card_header(panel, card, card_layout, benchmark, benchmark["id"], tag,
                                      title_text, grouped, panel._total_groups > 1,
                                      len(panel._groups) == 1 and panel._total_groups > 1)
        status = [panel.tr("{count} volume points").format(count=len(summary["volumes"]))]
        if branch:
            status.insert(0, branch.replace("_", " "))
        warnings = list(summary.get("warnings", []))
        if warnings:
            status.append(panel.tr("Fit warning"))
        for _key, name, run, details in entries:
            missing = _missing_status(panel, name, run, details)
            if missing:
                status.append(missing)
            if run and run["status"] != "succeeded":
                warnings.append(run.get("error") or run["status"])
        caption = CaptionLabel(" · ".join(status), card)
        caption.setWordWrap(True)
        caption.setMaximumHeight(caption.fontMetrics().lineSpacing() * 2)
        caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
        caption.setToolTip(
            benchmark["name"]
            + "\n"
            + " · ".join(status)
            + "\n"
            + "\n".join(translate_runtime_message(w) for w in warnings)
        )
        card_layout.addWidget(caption)
        plot, readout = _eos_canvas(panel, card, card_layout, reset, residual, relative)
        values.clicked.connect(lambda checked=False, chart=plot: show_curve_values(panel, chart))
        _plot_group(panel, plot, readout, benchmark, entries, residual, relative)
        plot.finish()
        update_residual_placeholder(plot)
        if not panel._adaptive_layout:
            card.setFixedHeight(300)
            card.setMaximumWidth(460)
        panel.eos_cards.append(card)
        panel.eos_plots.append(plot)
    panel._layout_cards()


def preview_eos(dialog, summary, group):
    if not hasattr(dialog.preview_plot, "point_hover"):
        dialog.preview_plot.point_hover = PlotPointHover(dialog.preview_plot, "V = {:.4f} Å³/atom", "eV/atom")
    volumes = summary["volumes"]
    plotted = [g for g in dialog.eos_groups if g["include"] and g["summary"]] if dialog.overlay_groups.isChecked() else [group]
    dialog.preview_plot.configure(dialog.tr("Volume (Å³/atom)"), dialog.tr("Energy (eV/atom)"), len(plotted) > 1)
    for entry in plotted:
        curve = entry["summary"]
        color = palette.PLOT_AXIS
        dialog.preview_plot.plot(curve["volumes"], curve["energies"], pen=QPen(QColor(color), EOS_REFERENCE_WIDTH),
                               symbol=entry["marker"], symbolBrush=color, symbolPen=color, symbolSize=EOS_POINT_SIZE,
                               name=escape(entry["key"]) or dialog.tr("Whole file"))
    dialog.preview_plot.finish()
    text = dialog.tr("{formula} · {count} structures\nVolume: {low:.3f}–{high:.3f} Å³/atom").format(
        formula=summary["formula"], count=len(group["indices"]), low=volumes[0], high=volumes[-1])
    if group["key"]:
        title, branch = eos_group_title(group["key"], summary.get("formula"))
        dialog.preview_title.setText(title if not branch else f"{title} · {branch}")
        dialog.preview_title.setToolTip(group["key"])
    fit = summary["fit"]
    if "v0" in fit:
        text += "\n" + dialog.tr("Reference fit: V₀ = {v0:.3f} Å³/atom · B₀ = {b0:.2f} GPa").format(v0=fit["v0"], b0=fit["b0_gpa"])
    else:
        text += "\n" + dialog.tr("Fit unavailable: compare the E–V curve only; V₀/B₀ will not be compared.")
    return text
