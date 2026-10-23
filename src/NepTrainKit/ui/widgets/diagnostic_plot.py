"""Diagnostic response cards, sharing the workbench legend and view controls."""
from html import escape
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QTextBrowser
from qfluentwidgets import BodyLabel, CaptionLabel, TransparentToolButton, TransparentPushButton, FluentIcon
from NepTrainKit import palette

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
        for curve, details in curves:
            values = details.get(metric)
            curve.setData(details['x'] if values is not None else [], values if values is not None else [])
        plot.setLabel('left', 'ΔE (meV/atom)' if metric == 'relative_energy' else 'dE/dq (eV / 横轴单位)')
        panel._reset_plot_view(plot)


def redraw_diagnostics(panel):
    for card in panel.eos_cards:
        panel.grid_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
    panel.eos_cards, panel.eos_plots, panel._diagnostic_charts = [], [], []
    panel._grid_columns = 0
    panel.relative_checkbox.hide()
    panel.offset_checkbox.hide()
    panel.title_label.setText(f"模型响应  ·  {len(panel._groups)} 项实验")
    panel.info_label.setText("无参考诊断  ·  相对初始态 / 原子  ·  准确性尚未验证")
    panel.info_label.setToolTip("曲线是模型预测；内部一致性检查通过不代表外部准确，负曲率也不自动判错。")
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
        reset.setToolTip('重置坐标轴：适应可见曲线')
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
        plot = panel._pg.PlotWidget(card)
        plot.setMinimumHeight(280 if focused else 175)
        plot.setBackground(palette.SURFACE)
        plot.showGrid(x=False, y=True, alpha=.09)
        for side in ('left', 'bottom'):
            axis = plot.getAxis(side)
            axis.setPen(panel._pg.mkPen(palette.BORDER))
            axis.setTextPen(panel._pg.mkPen(palette.PLOT_AXIS))
            axis.setStyle(maxTickLevel=0, tickLength=-3)
            axis.setTickDensity(.7)
            tick_font = QFont(card.font())
            tick_font.setPointSize(10)
            axis.setTickFont(tick_font)
        plot.setLabel('bottom', summary['x_label'])
        reset.clicked.connect(lambda _=False, target=plot: panel._reset_plot_view(target))
        layout.addWidget(plot, 1)
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
                              symbol='o', symbolSize=3, symbolBrush=panel._series_pen(key).color(), symbolPen=None)
            panel._register_curve(key, curve)
            curves.append((curve, details))
            for check in details.get('checks', []):
                counts[check['status']] += 1
        panel._diagnostic_charts.append((plot, curves))
        if not curves:
            plot.setXRange(min(summary['coordinates']), max(summary['coordinates']))
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
        status.setToolTip('汇总所选势版本的检查条目；详情按版本展开。')
        status.setWordWrap(True)
        footer.addWidget(status, 1)
        more = TransparentPushButton(FluentIcon.CHEVRON_RIGHT, '详情', card)
        more.setObjectName('diagnosticEvidenceButton')
        more.setCheckable(True)
        more.setToolTip('按势版本查看检查结果、适用条件和数值依据')
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
