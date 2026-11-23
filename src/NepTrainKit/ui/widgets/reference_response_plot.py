"""Reference-response cards using the workbench's shared legend and wheel policy."""

import csv
from functools import partial
from html import escape

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QFileDialog, QTabWidget, QFrame, QHBoxLayout, QTableWidgetItem, QVBoxLayout
from qfluentwidgets import CaptionLabel, FluentIcon, PrimaryPushButton, PushButton, TransparentToolButton

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
            values += [details["exchange"]["projections"][index]["predicted"] if details.get("exchange") else None
                       for _, details in ready]
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
                other = details["exchange"]["symmetric_chiral"][row] if details.get("exchange") else {}
                values += [other.get("predicted_symmetric"), other.get("predicted_d_projection")]
            for col, value in enumerate(values):
                extra.setItem(row, col, QTableWidgetItem("—" if value is None else str(value)))
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
SCAN_AXES = {**RATIO_AXES, "shear_yz": "γyz", "shear_zx": "γzx", "shear_xy": "γxy"}


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
    x = getattr(curve, "sample_x", curve.xData)[index]
    readout.setText(f"{name}  ·  {axis} = {x:.4g}  ·  {curve.yData[index]:.6g} {unit}")
    readout.show()


def response_branches(benchmark):
    summary = benchmark["summary"]
    if summary.get("plot_series"):
        branches = [(f"{benchmark['id']}/{n}", branch["name"], branch["indices"])
                    for n, branch in enumerate(summary["plot_series"])]
    else:
        spec = summary["analysis"]
        branches = []
        series = summary["series"]
        if spec["mode"] in ("combination", "four_state", "cross"):
            series = [dict(name=benchmark["name"], indices=list(range(len(summary["rows"]))))]
        for group in series:
            slices = {}
            for i in group["indices"]:
                slices.setdefault(summary["rows"][i]["second_axis"], []).append(i)
            for value, indices in sorted(slices.items()):
                parts = [group["name"]] if spec["group_fields"] else []
                if value is not None:
                    parts.append(f"{RATIO_AXES.get(spec['second_axis'], spec['second_axis'])} = {value}")
                branches.append((f"{benchmark['id']}/{len(branches)}",
                                 " · ".join(parts) or benchmark["name"].rsplit(" / ", 1)[-1], indices))
    if len(branches) == 1:
        _, label, indices = branches[0]
        return [(benchmark["id"], label, indices)]
    return branches


def magnetic_branches(benchmark):
    return response_branches(benchmark)


def _draw_response_curves(panel, benchmark, entries):
    from NepTrainKit.core.validation.response_plot import exchange_plot_view

    source_summary, source_entries = benchmark["summary"], entries
    summary = exchange_plot_view(source_summary, panel.response_kind)
    if summary is not source_summary:
        projected = []
        for key, name, run, details in entries:
            view = exchange_plot_view(details, panel.response_kind) if details else details
            projected.append((key, name, run, view if view is not details else {}))
        entries = projected
        benchmark = dict(benchmark, summary=summary)
    spec, rows, unit = summary["analysis"], summary["rows"], summary["unit"]
    branches = response_branches(benchmark)
    metadata = {key: branch for (key, _, _), branch in zip(branches, summary.get("plot_series", []))}
    scan = spec["mode"] == "scan"
    categorical = spec["mode"] in ("combination", "four_state", "cross", "difference") and not metadata
    axis = SCAN_AXES.get(spec["x_field"], spec["x_field"]) if scan else panel.tr("State / combination")
    symbols = {key: panel.group_markers.get(key, default_group_marker(n))
               for n, (key, _, _) in enumerate(branches)}
    selection = panel.magnetic_selection
    visible = [b for b in branches if selection is None or b[0] in selection]
    batches = [visible] if panel._eos_overlay or not panel._adaptive_layout else [[b] for b in visible]
    residual = panel._adaptive_layout and panel._eos_residual
    relative = scan and spec["quantity"] == "energy" and panel.relative_checkbox.isChecked() and not residual
    absolute = scan and spec["quantity"] == "energy" and spec["relative"] and not (relative or residual)
    ordinate = "ΔE" if relative or not scan else "E"
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
        if run and run["status"] == "succeeded" and not details and summary is not source_summary:
            unavailable.append(name)
        if run and run["status"] == "succeeded" and details:
            values = details["errors"] if residual else energies(details, "predicted", run)
            if values is None:
                unavailable.append(name)
            else:
                series_values.append((key, name, values))
    # Separate incompatible coordinate types even when the user overlays groups.
    compatible_batches = []
    for batch in batches:
        by_axis = {}
        for branch in batch:
            by_axis.setdefault(metadata.get(branch[0], {}).get("axis", axis), []).append(branch)
        compatible_batches.extend(by_axis.values())
    batches = compatible_batches
    for batch in batches:
        branch_meta = metadata.get(batch[0][0], {})
        coordinate = branch_meta.get("axis", axis)
        xlabel = panel.tr("Rotation angle θ (°)") if coordinate == "angle_deg" else (
            panel.tr("Path image") if coordinate == "image" else
            panel.tr("Direction · angle unavailable") if coordinate == "direction" else coordinate)
        if coordinate == "structure":
            xlabel = panel.tr("Structure variant")
        is_category = categorical or branch_meta.get("connected") is False or (
            not branch_meta and any(not isinstance(rows[i]["x"], (int, float))
                                    for _, _, indices in batch for i in indices))
        indexed = is_category and coordinate != "structure" and any(len(indices) > 8 for _, _, indices in batch)
        if indexed:
            xlabel = panel.tr("Combination index")
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
                panel, card, layout, benchmark, key, "G1", label,
                grouped, panel.magnetic_can_focus, panel.magnetic_can_focus and len(visible) == 1, focus_key=key,
            )
            count_text = QCoreApplication.translate("ValidationWorkspace", "{count} scan points")
            context = label if is_magnetic_eos(benchmark) else (
                " → ".join(branch_meta["plane"]) if branch_meta.get("plane") else "")
            if "zero_frame" in branch_meta:
                context = panel.tr("0° = frame {index}").format(index=branch_meta["zero_frame"])
            caption = CaptionLabel(" · ".join(filter(None, (context, count_text.format(count=len(indices))))), card)
            caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
            layout.addWidget(caption)
        else:
            target = next((branch for branch in batch if branch[0] == panel.eos_current_id), batch[0])
            strip, reset, data = overlay_header(panel, card, layout, target[0], target[1])
            for key, label, _ in batch:
                add_overlay_group(panel, strip, key, label, benchmark["name"] + "\n" + label)
            strip.fit_groups()
        if not is_magnetic_eos(benchmark):
            sample = card.findChild(TransparentToolButton, "eosSamplingButton")
            if sample:
                sample.hide()
        selected = list(dict.fromkeys(i for _, _, indices in batch for i in indices if i is not None))
        quantity = ordinate if spec["quantity"] == "energy" else spec["quantity"]
        if summary is not source_summary:
            quantity = panel.tr({"projections": "Bilinear projection", "symmetric": "Symmetric projection",
                                 "d_projection": "Chiral D projection"}[panel.response_kind])
            if residual:
                quantity = ordinate + " · " + quantity
        reference_state = branch_meta.get("reference", spec.get("reference", ""))
        if "reference_frame" in branch_meta:
            reference_state = panel.tr("frame {index}").format(index=branch_meta["reference_frame"])
        if reference_state and not residual:
            quantity = f"ΔE · E − E({reference_state})" if spec["quantity"] == "energy" else quantity
        plot, readout = energy_curve_canvas(panel, card, layout, reset, xlabel, f"{quantity} ({unit})",
                                            xlabel + " = {:.4g}", unit, residual)
        if not scan or spec["quantity"] != "energy":
            plot.add_zero_line(panel._series_pen(None))

        names = {key: label for key, label, _ in batch}
        source_indices = sorted({i for row in selected for i in summary["source_rows"][row]}) if "source_rows" in summary else selected
        details = partial(_show_values, panel, source_summary, source_indices, source_entries)
        data.clicked.connect(lambda _=False, chart=plot, groups=names, more=details:
                             show_curve_values(panel, chart, groups, more))
        if unavailable:
            note = CaptionLabel(panel.tr("Projection unavailable: ") if summary is not source_summary else
                                panel.tr("Absolute energy unavailable: "), card)
            note.setText(note.text() + ", ".join(unavailable))
            note.setWordWrap(True)
            layout.insertWidget(1, note)
        for branch_key, label, indices in batch:
            symbol = symbols[branch_key]
            info = metadata.get(branch_key, {})
            x = info.get("x") or ([n + 1 for n in range(len(indices))] if is_category
                                  else [rows[i]["x"] for i in indices])
            if is_category:
                labels = info.get("labels") or [rows[i]["label"] for i in indices]
                plot.category_labels = dict(zip(x, labels))
                plot.set_category_ticks(list(zip(x, (str(n) for n in x) if indexed else labels)))
            def value_at(values, index):
                return 0. if index is None else values[index]

            zero = min(value_at(reference, i) for i in indices) if relative and not residual and not spec["relative"] else 0.
            for series_index, (key, name, values) in enumerate(series_values):
                offset = 0.
                if scan and spec["quantity"] == "energy" and key is not None and panel.offset_checkbox.isChecked() and (residual or reference is not None):
                    offset = sum(values[i] if residual else values[i] - reference[i] for i in indices) / len(indices)
                pen = panel._series_pen(key)
                shifted = [v + .52 * (series_index - (len(series_values) - 1) / 2) / max(1, len(series_values)) for v in x] if is_category else x
                y = [value_at(values, i) - zero - offset for i in indices]
                curve = plot.plot(shifted, y,
                                  pen=None if is_category else pen, symbol=symbol,
                                  symbolSize=EOS_POINT_SIZE if panel._adaptive_layout else 5,
                                  symbolPen=pen.color(), symbolBrush=pen.color(), name=escape(name))
                curve.setProperty("groupId", branch_key)
                if is_category:
                    curve.sample_x = x
                    curve.point_labels = labels
                    curve.point_ids = [(branch_key, value) for value in x]
                    curve.point_index = {value: i for i, value in enumerate(curve.point_ids)}
                panel._register_curve(key, curve)
                if scan and spec["quantity"] == "energy" and not residual and y:
                    minimum = min(range(len(y)), key=y.__getitem__)
                    marker = plot.plot([x[minimum]], [y[minimum]], pen=None, symbol="d", symbolSize=13,
                                       symbolPen=pen.color(), symbolBrush=pen.color(), auxiliary=True)
                    marker.setProperty("annotation", "sampled_minimum")
                    panel._register_curve(key, marker)
                if readout is not None:
                    curve.read_point = partial(_pin_response, readout, " · ".join(filter(None, (name, label))), xlabel, unit)
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
        if recipe["mode"] != "slices":
            _draw_response_curves(panel, benchmark, entries)
            continue
        from NepTrainKit.ui.widgets.accuracy_comparison_plot import draw_accuracy_cards

        draw_accuracy_cards(panel, benchmark, entries)
    panel._layout_cards()
