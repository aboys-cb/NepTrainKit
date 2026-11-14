"""Reference-response cards using the workbench's shared legend and wheel policy."""

import csv

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout, QTableWidgetItem, QVBoxLayout
from qfluentwidgets import BodyLabel, CaptionLabel, FluentIcon, PushButton, TableWidget, TransparentToolButton

from NepTrainKit import palette
from NepTrainKit.core.io.atomic import atomic_text_output
from NepTrainKit.core import MessageManager


def _show_values(panel, summary, selected, entries):
    dialog = QDialog(panel)
    dialog.setWindowTitle(panel.tr("Response values and frame combinations"))
    dialog.resize(1000, 600)
    layout = QVBoxLayout(dialog)
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
    table = TableWidget(dialog)
    table.setColumnCount(len(names))
    table.setHorizontalHeaderLabels(names)
    table.setRowCount(len(selected))
    rows = []
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
        rows.append(values)
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem("—" if value is None else str(value)))
    table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
    layout.addWidget(table)
    if summary.get("exchange"):
        chiral = summary["exchange"]["symmetric_chiral"]
        names2 = [
            panel.tr(label)
            for label in ("Ordered pair / group", "D projection direction", "Symmetric reference", "D reference")
        ]
        for name, _ in ready:
            names2 += [name + " " + panel.tr("Symmetric"), name + " D"]
        extra = TableWidget(dialog)
        extra.setColumnCount(len(names2))
        extra.setHorizontalHeaderLabels(names2)
        extra.setRowCount(len(chiral))
        extra.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
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
        layout.addWidget(extra)
    if summary["analysis"]["mode"] == "scan" and not summary["analysis"]["second_axis"]:
        headers = [panel.tr("Group"), panel.tr("Reference sampled minimum"), panel.tr("Reference local curvature")]
        for name, _ in ready:
            headers += [name + " " + panel.tr("Sampled minimum"), name + " " + panel.tr("Local curvature")]
        extra = TableWidget(dialog)
        extra.setColumnCount(len(headers))
        extra.setHorizontalHeaderLabels(headers)
        curves = [group for group in summary["series"] if set(group["indices"]) & set(selected)]
        extra.setRowCount(len(curves))
        extra.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
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
        layout.addWidget(extra)
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
        extra = TableWidget(dialog)
        extra.setColumnCount(len(names))
        extra.setHorizontalHeaderLabels(names)
        extra.setRowCount(len(rows))
        extra.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                extra.setItem(row, column, QTableWidgetItem("—" if value is None else str(value)))
        layout.addWidget(extra)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
    buttons.button(QDialogButtonBox.StandardButton.Close).setText(panel.tr("Close"))
    export = PushButton(panel.tr("Export CSV"), dialog)

    def save():
        path, _ = QFileDialog.getSaveFileName(dialog, panel.tr("Export response values"), "", "CSV (*.csv)")
        if path:
            try:
                with atomic_text_output(path) as handle:
                    handle.write("\ufeff")
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(names)
                    writer.writerows(rows)
            except OSError as error:
                MessageManager.send_error_message(str(error))

    export.clicked.connect(save)
    buttons.addButton(export, QDialogButtonBox.ButtonRole.ActionRole)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


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
            data = TransparentToolButton(FluentIcon.DOCUMENT, card)
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
