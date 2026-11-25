"""Diagnostic response cards, sharing the workbench legend and view controls."""
from html import escape
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QTextBrowser
from qfluentwidgets import BodyLabel, CaptionLabel, TransparentToolButton, TransparentPushButton, FluentIcon
from NepTrainKit import palette
from NepTrainKit.core.validation.protocols.diagnostic import scan_numeric_derivative
from NepTrainKit.ui.widgets.eos_plot import energy_curve_canvas
from NepTrainKit.ui.widgets.validation_plot_hover import PlotPointHover

STATUS = {"pass": "通过", "fail": "异常", "undetermined": "待确认", "not_applicable": "不适用"}
STATUS_COLOR = {"pass": QColor(palette.PLOT_SHOW).darker(125).name(), "fail": QColor(palette.PLOT_REJECT).darker(115).name(),
                "undetermined": QColor(palette.PLOT_SELECTED).darker(140).name(), "not_applicable": palette.TEXT_SECONDARY}


def _number(value):
    if value is None:
        return "未提供"
    if isinstance(value, (list, tuple)):
        return " / ".join(_number(item) for item in value)
    return f"{value:.6g}" if isinstance(value, (int, float)) else escape(str(value))


def _check_report(entries):
    """Human-readable evidence, with model names and all numerical fields retained."""
    labels = {"analytic": "解析导数", "numeric": "差分导数", "steps": "差分步长", "drift": "减半步长漂移",
              "tolerance": "比较容差", "residual": "残差", "unit": "单位"}
    blocks = []
    for _key, name, run, details in entries:
        blocks.append(f"<h3>{escape(name)}</h3>")
        if not run or run['status'] != 'succeeded':
            blocks.append(f"<p>{escape(run['error'] if run else '尚未运行')}</p>")
            continue
        for check in details.get('checks', []):
            state = check['status']
            blocks.append(f"<p><b>{escape(check['name'])}</b>　<span style='color:{STATUS_COLOR[state]}'>{STATUS[state]}</span><br>{escape(check['reason'])}</p>")
            values = [f"{labels.get(key, escape(key))}：{_number(value)}" for key, value in check.items()
                      if key not in ('name', 'status', 'reason')]
            blocks.append("<p>" + "<br>".join(values) + "</p>")
    return "".join(blocks)


def set_response_metric(panel):
    metric = panel.diagnostic_mode.currentData()
    for plot, curves in panel._diagnostic_charts:
        residual = metric == "derivative_residual"
        energy = metric == "relative_energy"
        unit = "meV/atom" if energy else "eV" + ("/" + plot.coordinate_unit if plot.coordinate_unit else "")
        ylabel = "ΔE (meV/atom)" if energy else panel.tr("Derivative residual") if residual else "dE/dq"
        if not energy:
            ylabel += " (" + unit + ")"
        plot.axis_titles = (plot.coordinate_label, ylabel)
        plot.configure(*plot.axis_titles, False)
        plot.point_hover.unit = unit
        plot.method_hint.setVisible(not energy)
        plot.method_hint.setText(panel.tr("● Analytic   ▲ Scan difference (approximate)   ■ Origin difference (h/4)") if not residual else
                                 panel.tr("● Analytic − scan difference   ■ Analytic − origin difference"))
        for curve, details in curves:
            xs = details["x"]
            numeric = details.get("scan_numeric_derivative")
            if numeric is None:
                numeric = scan_numeric_derivative(xs, details["energy"])
                details["scan_numeric_derivative"] = numeric
            analytic = details.get("derivative")
            values = details["relative_energy"] if energy else analytic
            if residual:
                values = [a - n for a, n in zip(analytic, numeric)] if analytic is not None and numeric is not None else None
            coarse, exact = curve.derivative_parts
            check = next((c for c in details.get("checks", ()) if "numeric" in c and "analytic" in c), {})
            origin_value = check.get("numeric", [])
            origin_value = origin_value[-1] if origin_value else None
            if residual:
                origin_value = (check["analytic"] - origin_value
                                if check.get("analytic") is not None and origin_value is not None else None)
            blocks = ((curve, xs, values), (coarse, xs, numeric if not energy and not residual else None),
                      (exact, [xs[details["origin_index"]]], [origin_value] if not energy and origin_value is not None else None))
            for part, x, y in blocks:
                part.setData(x if y is not None else [], y if y is not None else [])
                part.setProperty("viewVisible", y is not None)
                part.setVisible(y is not None and part.property("modelId") not in panel._hidden_series)
        visible = [v for c in plot.listDataItems() if c.isVisible() for v in c.yData]
        lo, hi = (min(visible), max(visible)) if visible else (0., 1.)
        plot.origin_line.setData([plot.origin, plot.origin], [lo, hi])
        plot.origin_line.setVisible(bool(visible))
        plot.finish()


def redraw_diagnostics(panel):
    for card in panel.eos_cards:
        panel.grid_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
    panel.eos_cards, panel.eos_plots, panel._diagnostic_charts = [], [], []
    panel._grid_columns = 0
    panel.relative_checkbox.hide()
    panel.offset_checkbox.hide()
    panel.title_label.setText(panel.tr("Model response  ·  {count} experiments").format(count=len(panel._groups)))
    panel.info_label.setText(panel.tr("Reference-free diagnostic · Internal consistency only · External accuracy not verified"))
    panel.info_label.setToolTip(panel.tr("Curves are model predictions. Passing an internal consistency check does not imply external accuracy, and a negative curvature is not automatically an error."))
    for tag, benchmark, entries in panel._groups:
        summary = benchmark['summary']
        magnetic = summary['kind'].startswith('spin')
        accent = palette.PLOT_CURRENT if magnetic else palette.PLOT_MARKER_EDGE
        card = QFrame(panel.grid_container)
        card.setObjectName('diagnosticCard')
        card.setStyleSheet(f"#diagnosticCard {{background:{palette.SURFACE};border:1px solid {palette.BORDER};border-radius:12px;}}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 12, 8)
        layout.setSpacing(4)
        heading = QHBoxLayout()
        heading.setSpacing(8)
        category = CaptionLabel('磁' if magnetic else '晶', card)
        category.setAlignment(Qt.AlignmentFlag.AlignCenter)
        category.setFixedSize(28, 28)
        category.setStyleSheet(f"color:{accent};background:{palette.SURFACE};border:1px solid {palette.BORDER};border-radius:8px;font-weight:600;")
        heading.addWidget(category)
        title = BodyLabel(summary['title'], card)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        heading.addWidget(title)
        meta = CaptionLabel(f"{summary['formula']} · {len(summary['coordinates'])} 点", card)
        meta.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        heading.addWidget(meta)
        heading.addStretch(1)
        spec = summary['spec']
        context = '固定分数坐标' if summary['kind'] == 'volume' else f"原子 {spec['atom'] + 1} · 方向 {spec['axis']}"
        title.setToolTip(context)
        reset = TransparentToolButton(FluentIcon.SYNC.colored(palette.TEXT_SECONDARY, palette.TEXT_SECONDARY), card)
        reset.setObjectName('resetDiagnosticView')
        reset.setToolTip(panel.tr('Reset the axes to fit the visible curves'))
        reset.setAccessibleName(reset.toolTip())
        reset.setFixedSize(28, 28)
        heading.addWidget(reset)
        focused = len(panel._groups) == 1 and panel._total_groups > 1
        focus = TransparentToolButton((FluentIcon.ZOOM_OUT if focused else FluentIcon.ZOOM_IN).colored(palette.TEXT_SECONDARY, palette.TEXT_SECONDARY), card)
        focus.setObjectName('toggleDiagnosticFocus')
        focus.setToolTip('返回全部实验' if focused else '放大这项实验')
        focus.setAccessibleName(focus.toolTip())
        focus.setFixedSize(28, 28)
        if focused:
            focus.clicked.connect(lambda _=False: panel.allGroupsRequested.emit())
        else:
            focus.clicked.connect(lambda _=False, t=tag: panel.groupFocused.emit(t))
        focus.setVisible(panel._total_groups > 1)
        heading.addWidget(focus)
        layout.addLayout(heading)
        plot, _ = energy_curve_canvas(panel, card, layout, reset, summary['x_label'], "ΔE (meV/atom)",
                                       "q = {:.6g}", "meV/atom", False)
        if not hasattr(plot, "point_hover"):
            plot.point_hover = PlotPointHover(plot, "q = {:.6g}", "meV/atom")
        plot.add_zero_line(panel._series_pen(None))
        plot.coordinate_label = summary['x_label']
        plot.coordinate_unit = {"volume": "%", "displacement": "Å", "spin_angle": "°", "spin_magnitude": ""}[summary['kind']]
        plot.origin = 1. if summary['kind'] == 'spin_magnitude' else 0.
        pen = QPen(panel._series_pen(None))
        pen.setWidthF(1.2)
        plot.origin_line = plot.plot([plot.origin, plot.origin], [0., 1.], pen=pen, auxiliary=True)
        plot.method_hint = CaptionLabel(card)
        plot.method_hint.setWordWrap(True)
        plot.method_hint.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        layout.insertWidget(layout.indexOf(plot), plot.method_hint)
        curves = []
        counts = dict.fromkeys(STATUS, 0)
        missing, failed = 0, 0
        for key, _name, run, details in entries:
            if not run:
                missing += 1
                continue
            if run['status'] != 'succeeded':
                failed += 1
                continue
            curve = plot.plot(details['x'], details['relative_energy'], pen=panel._series_pen(key),
                              symbol='o', symbolSize=8, symbolBrush=panel._series_pen(key).color(),
                              symbolPen=panel._series_pen(key).color(), name=escape(_name))
            curve.setProperty("diagnosticRole", "analytic")
            panel._register_curve(key, curve)
            pen = QPen(panel._series_pen(key))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.8)
            coarse = plot.plot([], [], pen=pen, symbol='t1', symbolSize=7,
                               symbolBrush=pen.color(), symbolPen=pen.color(), name=panel.tr("Scan difference"))
            exact = plot.plot([], [], pen=None, symbol='s', symbolSize=11,
                              symbolBrush=pen.color(), symbolPen=pen.color(), name=panel.tr("Origin difference"))
            coarse.setProperty("fixedStyle", True)
            for part in (coarse, exact):
                part.setProperty("viewVisible", False)
                part.setProperty("diagnosticRole", "numeric")
                panel._register_curve(key, part)
            curve.derivative_parts = (coarse, exact)
            curves.append((curve, details))
            for check in details.get('checks', []):
                counts[check['status']] += 1
        panel._diagnostic_charts.append((plot, curves))
        footer = QHBoxLayout()
        footer.setSpacing(8)
        label = CaptionLabel('一致性检查', card)
        label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        footer.addWidget(label)
        summary_parts = [f"<span style='color:{STATUS_COLOR[state]}'>{count} {STATUS[state]}</span>"
                         for state, count in counts.items() if count]
        if missing:
            summary_parts.append(f'{missing} 个版本待运行')
        if failed:
            summary_parts.append(f"<span style='color:{palette.PLOT_SELECTED}'>{failed} 个版本不可用</span>")
        status = CaptionLabel(' · '.join(summary_parts) or '选择势后运行', card)
        status.setObjectName('diagnosticCheckSummary')
        status.setToolTip(panel.tr('Summarises the checks for the selected model versions; expand a version for detail.'))
        status.setWordWrap(True)
        footer.addWidget(status, 1)
        more = TransparentPushButton(FluentIcon.CHEVRON_RIGHT, '详情', card)
        more.setObjectName('diagnosticEvidenceButton')
        more.setCheckable(True)
        more.setToolTip(panel.tr('View the check results, applicability conditions, and numeric evidence by model version'))
        footer.addWidget(more)
        layout.addLayout(footer)
        evidence = QTextBrowser(card)
        evidence.setObjectName('diagnosticEvidence')
        evidence.setOpenExternalLinks(False)
        evidence.setStyleSheet(f"QTextBrowser {{border:0;border-top:1px solid {palette.BORDER};background:{palette.SURFACE};color:{palette.TEXT_PRIMARY};padding:8px;}}")
        evidence.setHtml(_check_report(entries) or '运行后显示检查结果。')
        evidence.setMinimumHeight(180)
        evidence.setMaximumHeight(260)
        evidence.hide()
        def expand(checked, button=more, target=evidence):
            target.setVisible(checked)
            button.setIcon(FluentIcon.CHEVRON_DOWN_MED if checked else FluentIcon.CHEVRON_RIGHT)
            button.setText('收起' if checked else '详情')
        more.toggled.connect(expand)
        layout.addWidget(evidence)
        panel.eos_cards.append(card)
        panel.eos_plots.append(plot)
    set_response_metric(panel)
    panel._layout_cards()
