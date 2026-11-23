"""Native numeric dialogs for validation curves and their source frames."""
from collections import Counter
from html import unescape

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout
from qfluentwidgets import BodyLabel, CaptionLabel, PushButton, StrongBodyLabel, TableWidget

from NepTrainKit import palette
from NepTrainKit.ui.widgets.validation_model_picker import group_marker_icon

TABLE_ICON = ":/images/src/images/table.svg"


def values_dialog(panel, title):
    dialog = QDialog(panel, Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowCloseButtonHint)
    dialog.setWindowTitle(title)
    dialog.setObjectName("validationValues")
    dialog.setStyleSheet(f"""
        #validationValues {{ background: {palette.PLOT_BACKGROUND}; }}
        #validationValues QHeaderView::section {{
            background: {palette.SURFACE_SUBTLE}; color: {palette.TEXT_PRIMARY};
            border: 0; border-bottom: 1px solid {palette.BORDER}; padding: 5px 8px; font-weight: 600;
        }}
    """)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(16, 12, 16, 14)
    layout.setSpacing(8)
    return dialog, layout


def values_table(dialog):
    table = TableWidget(dialog)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(32)
    table.horizontalHeader().setFixedHeight(34)
    table.setBorderVisible(True)
    table.setBorderRadius(6)
    table.setWordWrap(False)
    table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
    table.setShowGrid(False)
    return table


def show_curve_values(panel, plot, group_names=None, response_details=None):
    curves = [curve for curve in plot.listDataItems() if curve.isVisible()]
    groups = group_names or {item["id"]: item["name"] for _, item, _ in panel._groups}
    group_ids = list(dict.fromkeys(c.property("groupId") for c in curves))
    models, points = {}, {}
    for curve in curves:
        key = curve.property("modelId")
        label = unescape(curve.label if hasattr(curve, "label") else curve.name())
        pen = QPen(curve.opts["pen"] or curve.opts["symbolPen"])
        models[key] = (label, pen.color())
        # Preserve separate measurements even when their X coordinates coincide.
        occurrence = Counter()
        for x, y in zip(getattr(curve, "sample_x", curve.xData), curve.yData):
            point = (curve.property("groupId"), x, occurrence[x])
            points.setdefault(point, {})[key] = y
            occurrence[x] += 1

    title = panel.tr("Curve values")
    if len(panel._groups) == 1:
        title += " · " + panel._groups[0][1]["name"]
    dialog, layout = values_dialog(panel, title)
    heading = QHBoxLayout()
    layout.addLayout(heading)
    context = QVBoxLayout()
    context.setSpacing(2)
    heading.addLayout(context, 1)
    context.addWidget(StrongBodyLabel(plot.axis_titles[1], dialog))
    caption = panel.tr("{count} scan points").format(count=len(points))
    if len(group_ids) == 1:
        caption = groups[group_ids[0]] + "  ·  " + caption
    elif group_ids:
        caption = panel.tr("{count} groups").format(count=len(group_ids)) + "  ·  " + caption
    subtitle = CaptionLabel(dialog)
    subtitle.setTextFormat(Qt.TextFormat.PlainText)
    subtitle.setText(caption)
    subtitle.setWordWrap(True)
    subtitle.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
    context.addWidget(subtitle)
    if response_details is not None:
        more = PushButton(panel.tr("Frame details"), dialog, icon=TABLE_ICON)
        more.clicked.connect(response_details)
        heading.addWidget(more)
    table = values_table(dialog)
    multi = len(group_ids) > 1
    categories = getattr(plot, "category_labels", {})
    headers = (([panel.tr("Group")] if multi else []) + [plot.axis_titles[0]]
               + ([panel.tr("State / combination")] if categories else []) + [label for label, _ in models.values()])
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(True)
    for column, (_, color) in enumerate(models.values(), int(multi) + 1 + bool(categories)):
        swatch = QPixmap(12, 12)
        swatch.fill(color)
        table.horizontalHeaderItem(column).setIcon(QIcon(swatch))
    table.setRowCount(len(points))
    widths = [min(270, header.fontMetrics().horizontalAdvance(label) + 50) for label in headers]
    if multi:
        widths[0] = min(310, max(table.fontMetrics().horizontalAdvance(groups[g]) + 68 for g in group_ids))
        table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    for row, ((group, x, _), values) in enumerate(points.items()):
        if multi:
            cell = QTableWidgetItem(group_marker_icon(panel.group_markers.get(group, "o")), groups[group])
            cell.setToolTip(groups[group])
            table.setItem(row, 0, cell)
        row_values = [x] + ([categories.get(x, "")] if categories else []) + [values.get(key) for key in models]
        for column, value in enumerate(row_values, int(multi)):
            text = value if isinstance(value, str) else "—" if value is None else f"{value:.8g}"
            cell = QTableWidgetItem(text)
            width = max(widths[column], table.fontMetrics().horizontalAdvance(text) + 52)
            widths[column] = min(310, width) if isinstance(value, str) else width
            alignment = Qt.AlignmentFlag.AlignLeft if isinstance(value, str) else Qt.AlignmentFlag.AlignRight
            cell.setTextAlignment(alignment | Qt.AlignmentFlag.AlignVCenter)
            if value is not None:
                cell.setToolTip(str(value))
            table.setItem(row, column, cell)
    for column, width in enumerate(widths):
        table.horizontalHeaderItem(column).setToolTip(headers[column])
        table.setColumnWidth(column, width)
    if points:
        layout.addWidget(table, 1)
    else:
        table.hide()
        empty = BodyLabel(panel.tr("No visible curve values."), dialog)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty, 1)
    bounds = dialog.screen().availableGeometry()
    dialog.resize(min(max(430, sum(widths) + 38), 1040, bounds.width() - 32),
                  min(120 + max(3, min(16, len(points))) * 32, bounds.height() - 64))
    dialog.exec()
