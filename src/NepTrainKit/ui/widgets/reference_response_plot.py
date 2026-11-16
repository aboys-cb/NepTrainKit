"""Reference-response cards using the workbench's shared legend and wheel policy."""

import csv
from functools import partial
from html import escape

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QFileDialog, QTabWidget, QFrame, QHBoxLayout, QTableWidgetItem, QVBoxLayout
from qfluentwidgets import BodyLabel, CaptionLabel, FluentIcon, PrimaryPushButton, PushButton, TransparentToolButton

from NepTrainKit import palette
from NepTrainKit.ui.widgets.eos_plot import (EOS_POINT_SIZE, overlay_header, add_overlay_group, energy_card_header,
                                              energy_curve_canvas, update_residual_placeholder)
from NepTrainKit.ui.widgets.validation_values import TABLE_ICON, show_curve_values, values_dialog, values_table
from NepTrainKit.ui.widgets.validation_model_picker import default_group_marker
from NepTrainKit.core.io.atomic import atomic_text_output
from NepTrainKit.core import MessageManager


def _show_values(panel, summary, selected, entries):
    dialog, layout = values_dialog(panel, panel.tr("Response values and frame combinations"))
    available = dialog.screen().availableGeometry()
    dialog.resize(min(1000, available.width() - 32), min(540, available.height() - 64))
    names = [
        panel.tr("Group / state"),
        "X",
        panel.tr("Second axis"),
        panel.tr("Reference") + " (" + summary["unit"] + ")",
        panel.tr("Frame coefficients (zero based)"),
    ]
    ready = [(name, details) for _, name, run, details in entries if run and run["status"] == "succeeded" and details]
    for name, _ in ready:
        names += [name, name + " " + panel.tr("Error")]
    if summary.get("exchange"):
        names += [panel.tr("Projected unit-spin response (reference)")]
        names += [name + " " + panel.tr("Projected response") for name, _ in ready]
    table = values_table(dialog)
    table.setColumnCount(len(names))
    table.setHorizontalHeaderLabels(names)
    table.setRowCount(len(selected))
    for row, index in enumerate(selected):
        item = summary["rows"][index]
        values = [
            item["label"],
            item["x"],
            item.get("second_axis"),
            summary["reference"][index],
            ", ".join(f"{i}:{coefficient:+g}" for i, coefficient in item["terms"]),
        ]
        for _, details in ready:
            values += [details["predicted"][index], details["errors"][index]]
        if summary.get("exchange"):
            values += [summary["exchange"]["projections"][index]["reference"]]
            values += [details.get("exchange", {}).get("projections", [])[index]["predicted"] for _, details in ready]
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem("—" if value is None else str(value)))
    pages = [(panel.tr("Frame values"), table)]
    if summary.get("exchange"):
        chiral = summary["exchange"]["symmetric_chiral"]
        names2 = [
            panel.tr(label)
            for label in ("Ordered pair / group", "D projection direction", "Symmetric reference", "D reference")
        ]
        for name, _ in ready:
            names2 += [name + " " + panel.tr("Symmetric"), name + " D"]
        extra = values_table(dialog)
        extra.setColumnCount(len(names2))
        extra.setHorizontalHeaderLabels(names2)
        extra.setRowCount(len(chiral))
        for row, item in enumerate(chiral):
            values = [
                str(item["parent"]),
                str(item["direction"]),
                item["reference_symmetric"],
                item["reference_d_projection"],
            ]
            for _, details in ready:
                other = details["exchange"]["symmetric_chiral"][row]
                values += [other["predicted_symmetric"], other["predicted_d_projection"]]
            for col, value in enumerate(values):
                extra.setItem(row, col, QTableWidgetItem(str(value)))
        pages.append((panel.tr("Exchange components"), extra))
    if summary["analysis"]["mode"] == "scan" and not summary["analysis"]["second_axis"]:
        headers = [panel.tr("Group"), panel.tr("Reference sampled minimum"), panel.tr("Reference local curvature")]
        for name, _ in ready:
            headers += [name + " " + panel.tr("Sampled minimum"), name + " " + panel.tr("Local curvature")]
        extra = values_table(dialog)
        extra.setColumnCount(len(headers))
        extra.setHorizontalHeaderLabels(headers)
        curves = [group for group in summary["series"] if set(group["indices"]) & set(selected)]
        extra.setRowCount(len(curves))
        for row, group in enumerate(curves):
            ref = group.get("reference_extremum", {})
            values = [group["name"], ref.get("minimum_x"), ref.get("curvature")]
            for _, details in ready:
                model = next(s for s in details["series"] if s["name"] == group["name"]).get("model_extremum", {})
                values += [model.get("minimum_x"), model.get("curvature")]
            for column, value in enumerate(values):
                extra.setItem(row, column, QTableWidgetItem("—" if value is None else str(value)))
        extra.setToolTip(
            panel.tr(
                "Minima refer to sampled points. Curvature is a local three-point estimate and is unavailable at a boundary minimum."
            )
        )
        pages.append((panel.tr("Minima and curvature"), extra))
    if summary["analysis"]["mode"] == "slices":
        names = [
            panel.tr(label)
            for label in ("Group", "Quantity", "Model", "RMSE", "MAE", "Valid components", "Compared components")
        ]
        rows = []
        for name, details in ready:
            for group in details["accuracy_slices"]:
                for quantity, count in group["counts"].items():
                    rows.append(
                        [
                            group["group"],
                            quantity,
                            name,
                            group["metrics"][quantity + "_rmse"],
                            group["metrics"][quantity + "_mae"],
                            count["components"],
                            count["evaluated_components"],
                        ]
                    )
        extra = values_table(dialog)
        extra.setColumnCount(len(names))
        extra.setHorizontalHeaderLabels(names)
        extra.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                extra.setItem(row, column, QTableWidgetItem("—" if value is None else str(value)))
        pages.append((panel.tr("Error statistics"), extra))
    tabs = QTabWidget(dialog)
    tabs.setDocumentMode(True)
    tabs.setTabBarAutoHide(True)
    for title, grid in pages:
        grid.resizeColumnsToContents()
        grid.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(grid, title)
    layout.addWidget(tabs, 1)
    buttons = QHBoxLayout()
    layout.addLayout(buttons)
    export = PushButton(FluentIcon.SAVE, panel.tr("Export CSV"), dialog)
    buttons.addWidget(export)
    buttons.addStretch(1)
    close = PrimaryPushButton(panel.tr("Close"), dialog)
    close.clicked.connect(dialog.accept)
    buttons.addWidget(close)

    def save():
        path, _ = QFileDialog.getSaveFileName(dialog, panel.tr("Export response values"), "", "CSV (*.csv)")
        if path:
            try:
                with atomic_text_output(path) as handle:
                    handle.write("\ufeff")
                    writer = csv.writer(handle, lineterminator="\n")
                    grid = tabs.currentWidget()
                    writer.writerow(grid.horizontalHeaderItem(c).text() for c in range(grid.columnCount()))
                    for row in range(grid.rowCount()):
                        values = [grid.item(row, col).text() for col in range(grid.columnCount())]
                        writer.writerow("" if v == "—" else v for v in values)
            except OSError as error:
                MessageManager.send_error_message(str(error))

    export.clicked.connect(save)
    dialog.exec()


RATIO_AXES = {"moment_ratio": "M/M₀", "volume_ratio": "V/V₀"}


def is_magnetic_eos(item):
    if item["protocol"] != "response":
        return False
    summary = item.get("summary") or {}
    spec = summary.get("analysis") or {}
    axes = {spec.get("x_field"), spec.get("second_axis")} - {None, ""}
    if spec.get("mode") != "scan" or spec.get("quantity") != "energy":
        return False
    if spec.get("x_field") not in RATIO_AXES or "moment_ratio" not in axes or not axes <= RATIO_AXES.keys():
        return False
    # Scan coordinates have one type within each series. Keep categorical scans on their labelled axis.
    return all(isinstance(summary["rows"][group["indices"][0]]["x"], (int, float))
               for group in summary.get("series", ()))


def _pin_response(readout, name, axis, unit, curve, index):
    readout.setText(f"{name}  ·  {axis} = {curve.xData[index]:.4g}  ·  {curve.yData[index]:.6g} {unit}")
    readout.show()


def magnetic_branches(benchmark):
    summary = benchmark["summary"]
    spec = summary["analysis"]
    branches = []
    for group in summary["series"]:
        slices = {}
        for i in group["indices"]:
            slices.setdefault(summary["rows"][i]["second_axis"], []).append(i)
        for value, indices in sorted(slices.items()):
            key = f"{benchmark['id']}/{len(branches)}"
            parts = [group["name"]] if spec["group_fields"] else []
            if value is not None:
                parts.append(f"{RATIO_AXES[spec['second_axis']]} = {value}")
            branches.append((key, " · ".join(parts) or benchmark["name"].rsplit(" / ", 1)[-1], indices))
    if len(branches) == 1:
        _, label, indices = branches[0]
        return [(benchmark["id"], label, indices)]
    return branches


def _draw_magnetic_scan(panel, benchmark, entries):
    summary = benchmark["summary"]
    spec, rows, unit = summary["analysis"], summary["rows"], summary["unit"]
    axis = RATIO_AXES[spec["x_field"]]
    branches = magnetic_branches(benchmark)
    symbols = {key: panel.group_markers.get(key, default_group_marker(n))
               for n, (key, _, _) in enumerate(branches)}
    selection = panel.magnetic_selection
    visible = [b for b in branches if selection is None or b[0] in selection]
    batches = [visible] if panel._eos_overlay or not panel._adaptive_layout else [[b] for b in visible]
    residual = panel._adaptive_layout and panel._eos_residual
    relative = panel.relative_checkbox.isChecked() and not residual
    absolute = spec["relative"] and not (relative or residual)
    ordinate = "ΔE" if relative else "E"
    if residual:
        ordinate = QCoreApplication.translate("ValidationWorkspace", "Residual")

    def energies(source, key, run=None):
        if not absolute:
            return source[key]
        values = source.get("absolute_" + key)
        if values is None and panel.load_scan_energies is not None:
            values = panel.load_scan_energies(benchmark, run)
        return values

    reference = energies(summary, "reference")
    series_values = [] if residual or reference is None else [(None, panel.tr("Reference"), reference)]
    unavailable = []
    if reference is None:
        unavailable.append(panel.tr("Reference"))
    for key, name, run, details in entries:
        if run and run["status"] == "succeeded" and details:
            values = details["errors"] if residual else energies(details, "predicted", run)
            if values is None:
                unavailable.append(name)
            else:
                series_values.append((key, name, values))
    for batch in batches:
        card = QFrame(panel.grid_container)
        card.setObjectName("magneticScanCard")
        card.setStyleSheet(f"#magneticScanCard {{background:{palette.PLOT_BACKGROUND};border:1px solid {palette.BORDER};border-radius:10px;}}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(4)
        if len(batch) == 1:
            key, label, indices = batch[0]
            grouped = panel.eos_current_id is not None
            reset, data = energy_card_header(
                panel, card, layout, benchmark, key, "G1", summary.get("formula") or label,
                grouped, panel.magnetic_can_focus, panel.magnetic_can_focus and len(visible) == 1, focus_key=key,
            )
            count_text = QCoreApplication.translate("ValidationWorkspace", "{count} scan points")
            caption = CaptionLabel(label + " · " + count_text.format(count=len(indices)), card)
            caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
            layout.addWidget(caption)
        else:
            target = next((branch for branch in batch if branch[0] == panel.eos_current_id), batch[0])
            strip, reset, data = overlay_header(panel, card, layout, target[0], target[1])
            for key, label, _ in batch:
                add_overlay_group(panel, strip, key, label, benchmark["name"] + "\n" + label)
            strip.fit_groups()
        selected = [i for _, _, indices in batch for i in indices]
        plot, readout = energy_curve_canvas(panel, card, layout, reset, axis, f"{ordinate} ({unit})",
                                            axis + " = {:.4g}", unit, residual)
        names = {key: label for key, label, _ in batch}
        details = partial(_show_values, panel, summary, selected, entries)
        data.clicked.connect(lambda _=False, chart=plot, groups=names, more=details:
                             show_curve_values(panel, chart, groups, more))
        if unavailable:
            note = CaptionLabel(panel.tr("Absolute energy unavailable: ") + ", ".join(unavailable), card)
            note.setWordWrap(True)
            layout.insertWidget(1, note)
        for branch_key, label, indices in batch:
            symbol = symbols[branch_key]
            x = [rows[i]["x"] for i in indices]
            zero = min(reference[i] for i in indices) if relative and not residual and not spec["relative"] else 0.
            for key, name, values in series_values:
                offset = 0.
                if key is not None and panel.offset_checkbox.isChecked() and (residual or reference is not None):
                    offset = sum(values[i] if residual else values[i] - reference[i] for i in indices) / len(indices)
                pen = panel._series_pen(key)
                curve = plot.plot(x, [values[i] - zero - offset for i in indices], pen=pen, symbol=symbol,
                                  symbolSize=EOS_POINT_SIZE if panel._adaptive_layout else 5,
                                  symbolPen=pen.color(), symbolBrush=pen.color(), name=escape(name))
                curve.setProperty("groupId", branch_key)
                panel._register_curve(key, curve)
                if readout is not None:
                    curve.read_point = partial(_pin_response, readout, " · ".join(filter(None, (name, label))), axis, unit)
        plot.finish()
        update_residual_placeholder(plot)
        panel.eos_cards.append(card)
        panel.eos_plots.append(plot)


def redraw_reference_responses(panel):
    for card in panel.eos_cards:
        panel.grid_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
    panel.eos_cards, panel.eos_plots = [], []
    panel._grid_columns = 0
    panel.relative_checkbox.hide()
    panel.offset_checkbox.hide()
    panel.title_label.setText(panel.tr("Reference response analysis"))
    panel.info_label.setText(
        panel.tr(
            "Fixed structures · signed responses · normalization and reference states are stored with each dataset"
        )
    )
    for tag, benchmark, entries in panel._groups:
        summary = benchmark["summary"]
        recipe = summary["analysis"]
        grouped_points = recipe["mode"] in ("combination", "four_state", "cross")
        series = (
            [dict(name=benchmark["name"], indices=list(range(len(summary["rows"]))))]
            if grouped_points
            else summary["series"]
        )
        if is_magnetic_eos(benchmark):
            _draw_magnetic_scan(panel, benchmark, entries)
            continue
        for group in series:
            selected = group["indices"]
            card = QFrame(panel.grid_container)
            card.setObjectName("referenceResponseCard")
            card.setStyleSheet(
                f"#referenceResponseCard {{background:{palette.SURFACE};border:1px solid {palette.BORDER};border-radius:10px;}}"
            )
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(4)
            heading = QHBoxLayout()
            title = BodyLabel(f"{tag} · {group['name']}", card)
            title.setMinimumWidth(0)
            title.setToolTip(benchmark["name"] + "\n" + group["name"])
            heading.addWidget(title, 1)
            reset = TransparentToolButton(FluentIcon.SYNC, card)
            reset.setToolTip(panel.tr("Reset view"))
            reset.setAccessibleName(reset.toolTip())
            heading.addWidget(reset)
            data = TransparentToolButton(TABLE_ICON, card)
            data.setToolTip(panel.tr("Response values and frame combinations"))
            data.clicked.connect(
                lambda _=False, s=summary, indices=selected, versions=entries: _show_values(panel, s, indices, versions)
            )
            data.setAccessibleName(data.toolTip())
            heading.addWidget(data)
            layout.addLayout(heading)
            roles = {
                "unknown": panel.tr("Not declared"),
                "training": panel.tr("Training regression"),
                "development": panel.tr("Development validation"),
                "validation": panel.tr("Validation (user declared)"),
            }
            description = CaptionLabel(f"{summary['unit']} · {len(selected)} · {roles[recipe['role']]}", card)
            notes = list(summary.get("warnings", []))
            for _, name, run, details in entries:
                if not run or run["status"] != "succeeded" or not details:
                    notes.append(name + ": " + panel.tr("Not run"))
                    continue
                if recipe["mode"] == "slices":
                    matching = next(
                        (item for item in details["accuracy_slices"] if item["group"] == group["name"]), None
                    )
                    if matching:
                        notes.append(
                            name
                            + ": "
                            + " / ".join(
                                key + "=" + ("—" if value is None else f"{value:.4g}")
                                for key, value in matching["metrics"].items()
                                if key.endswith("_rmse")
                            )
                        )
                else:
                    matching = next((item for item in details["series"] if item["name"] == group["name"]), None)
                    if matching:
                        notes.append(name + ": RMSE " + f"{matching['rmse']:.4g}")
            description.setToolTip("\n".join(notes))
            layout.addWidget(description)
            plot = panel._pg.PlotWidget(card)
            panel._configure_subplot(plot)
            plot.setMinimumWidth(0)
            plot.setMinimumHeight(190)
            plot.setLabel("left", summary["unit"])
            x = [summary["rows"][i]["x"] for i in selected]
            categorical = grouped_points or not all(isinstance(value, (int, float)) for value in x)
            if categorical:
                x = list(range(1, len(selected) + 1))
                labels = [summary["rows"][i]["label"] for i in selected]
                if len(labels) <= 12 and not grouped_points:
                    plot.getAxis("bottom").setTicks([list(enumerate(labels, 1))])
            plot.setLabel("bottom", summary["x_label"] if recipe["mode"] == "scan" else panel.tr("State / combination"))
            if recipe["mode"] == "slices":
                selected = [i for i in selected if summary["reference"][i] is not None]
                x = [summary["rows"][i]["x"] for i in selected]
            if recipe["second_axis"]:
                # Sparse two-dimensional scans are shown as separate measured slices.
                ys = sorted({summary["rows"][i]["second_axis"] for i in selected})
            else:
                ys = [None]
            for y in ys:
                subset = [offset for offset, i in enumerate(selected) if summary["rows"][i]["second_axis"] == y]
                xx = [x[i] for i in subset]
                label = "" if y is None else f" · {recipe['second_axis']}={y}"
                curve = plot.plot(
                    xx,
                    [summary["reference"][selected[i]] for i in subset],
                    pen=None
                    if grouped_points
                    else panel._pg.mkPen(palette.PLOT_AXIS, width=1.5, style=Qt.PenStyle.DashLine),
                    symbol="o",
                    symbolSize=5,
                    antialias=True,
                    name=panel.tr("Reference") + label,
                )
                panel._register_curve(None, curve)
                for key, name, run, details in entries:
                    if not run or run["status"] != "succeeded" or not details:
                        continue
                    curve = plot.plot(
                        xx,
                        [details["predicted"][selected[i]] for i in subset],
                        pen=None if grouped_points else panel._series_pen(key),
                        symbol="o",
                        symbolSize=4,
                        antialias=True,
                        symbolBrush=panel._series_pen(key).color(),
                        symbolPen=panel._series_pen(key).color(),
                        name=name + label,
                    )
                    panel._register_curve(key, curve)
            reset.clicked.connect(lambda _=False, target=plot: panel._reset_plot_view(target))
            layout.addWidget(plot, 1)
            card.setFixedHeight(300)
            card.setMaximumWidth(460)
            panel.eos_cards.append(card)
            panel.eos_plots.append(plot)
    panel._layout_cards()
