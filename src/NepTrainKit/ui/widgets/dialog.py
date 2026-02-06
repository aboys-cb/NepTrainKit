#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 22:45
# @Author  : Bing
# @email    : 1747193328@qq.com
from pathlib import Path
from typing import Any, Dict

from PySide6.QtGui import QIcon, QDoubleValidator, QIntValidator, QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QFrame, QGridLayout,
    QPushButton, QWidget, QHBoxLayout, QFormLayout, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt, QUrl, QEvent
from qfluentwidgets import (
    MessageBoxBase,
    SpinBox,
    CaptionLabel,
    DoubleSpinBox,
    CheckBox,
    ProgressBar,
    ComboBox,
    FluentStyleSheet,
    FluentTitleBar, TransparentToolButton, ColorDialog,
    TitleLabel, HyperlinkLabel, LineEdit, EditableComboBox, PrimaryPushButton, Flyout, InfoBarIcon, MessageBox,
    TextEdit, FluentIcon,
    ToolTipFilter, ToolTipPosition
)
from qframelesswindow import FramelessDialog
import json
import html
import math
import os
from .button import TagPushButton, TagGroup

from NepTrainKit.core import MessageManager
from NepTrainKit.core.types import SearchType

from NepTrainKit import module_path

from NepTrainKit.utils import LoadingThread,call_path_dialog
from NepTrainKit.core.utils import get_xyz_nframe,  read_nep_out_file, get_rmse


class GetIntMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, parent=None,tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.intSpinBox = SpinBox(self)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.intSpinBox)

        self.widget.setMinimumWidth(100 )
        self.intSpinBox.setMaximum(100000000)


class GetFloatMessageBox(MessageBoxBase):
    """Message box that lets the user input a floating-point value."""

    def __init__(self, parent=None, tip: str = ""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.doubleSpinBox = DoubleSpinBox(self)
        self.doubleSpinBox.setDecimals(10)
        self.doubleSpinBox.setMinimum(0.0)
        self.doubleSpinBox.setMaximum(1e6)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.doubleSpinBox)
        self.widget.setMinimumWidth(160)


class ExportFormatMessageBox(MessageBoxBase):
    """Message box that lets the user pick an export format (XYZ vs DeepMD/NPY)."""

    def __init__(self, parent=None, default_format: str = "xyz"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel("Choose export format", self)
        self.titleLabel.setWordWrap(True)

        self.formatCombo = ComboBox(self)
        self.formatCombo.addItem("XYZ (.xyz / extxyz)", userData="xyz")
        self.formatCombo.addItem("DeepMD/NPY (deepmd/npy)", userData="deepmd/npy")

        default = (default_format or "xyz").strip().lower()
        if default in {"deepmd", "deepmd/npy", "npy", "dp"}:
            self.formatCombo.setCurrentIndex(1)
        else:
            self.formatCombo.setCurrentIndex(0)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.formatCombo)

        self.widget.setMinimumWidth(320)

    def selected_format(self) -> str:
        """Return the selected export format identifier."""
        data = self.formatCombo.currentData()
        if isinstance(data, str) and data:
            return data
        text = self.formatCombo.currentText().lower()
        return "deepmd/npy" if "deepmd" in text or "npy" in text else "xyz"


class DatasetSummaryMessageBox(MessageBoxBase):
    """Frameless dialog that presents dataset-wide summary statistics."""

    def __init__(self, parent=None, summary: dict | None = None):
        super().__init__(parent)
        self._summary: dict[str, Any] = summary or {}
        group_by = self._summary.get("group_by", SearchType.TAG.value)
        group_by_value = group_by.value if isinstance(group_by, SearchType) else str(group_by)
        try:
            group_by_enum = SearchType(group_by_value)
        except Exception:
            group_by_enum = SearchType.FORMULA if group_by_value.endswith(".FORMULA") else SearchType.TAG
        group_label = "Formula" if group_by_enum == SearchType.FORMULA else "Config_type"

        self.widget.setMinimumWidth(460)
        max_rows_display = 10  # limit rows shown in dialog to keep it compact

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.viewLayout.addLayout(layout)
        heager_row =   QHBoxLayout()
        title = TitleLabel("Dataset Summary", self)

        # Export HTML button
        self.exportButton = TransparentToolButton(":/images/src/images/export1.svg", self)

        self.exportButton.clicked.connect(self._export_html)
        heager_row.addWidget(title)
        heager_row.addWidget(self.exportButton,alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(heager_row)

        # Source info
        source_row = QHBoxLayout()
        data_file = self._summary.get("data_file", "")
        model_file = self._summary.get("model_file", "")
        data_label = CaptionLabel(f"Data: {data_file}", self)
        model_label = CaptionLabel(f"Model: {model_file}", self)
        for lbl in (data_label, model_label):
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_row.addWidget(data_label, 1)
        source_row.addWidget(model_label, 1)
        layout.addLayout(source_row)

        # Basic counts and atom statistics
        counts = self._summary.get("counts", {})
        atoms = self._summary.get("atoms", {})
        elements = self._summary.get("elements", [])

        # Top summary cards
        card_row = QHBoxLayout()
        card_row.setContentsMargins(0, 0, 0, 0)
        card_row.setSpacing(8)

        def _add_card(caption: str, value: str) -> None:
            frame = QFrame(self)
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(8, 4, 8, 4)
            frame_layout.setSpacing(2)
            value_label = TitleLabel(value, frame)
            value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            cap_label = CaptionLabel(caption, frame)
            cap_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            frame_layout.addWidget(value_label)
            frame_layout.addWidget(cap_label)
            card_row.addWidget(frame)

        active_structures = counts.get("active_structures", 0)
        total_atoms_active = atoms.get("total_atoms_active", 0)

        _add_card("Orig structures", str(counts.get("orig_structures", 0)))
        _add_card("Active structures", str(active_structures))
        _add_card("Removed structures", str(counts.get("removed_structures", 0)))
        _add_card("Selected structures", str(counts.get("selected_structures", 0)))
        layout.addLayout(card_row)

        atoms_row = QHBoxLayout()
        atoms_row.setContentsMargins(0, 0, 0, 0)
        atoms_row.setSpacing(12)
        atoms_row.addWidget(CaptionLabel(f"Total atoms (active): {total_atoms_active}", self))
        atoms_row.addWidget(
            CaptionLabel(
                f"Atoms per structure: min={atoms.get('min_atoms', 0)}, "
                f"max={atoms.get('max_atoms', 0)}, "
                f"mean={atoms.get('mean_atoms', 0.0):.1f}, "
                f"median={atoms.get('median_atoms', 0.0):.1f}",
                self,
            )
        )
        layout.addLayout(atoms_row)

        # Element distribution
        elements = sorted(self._summary.get("elements", []), key=lambda x: x.get("fraction", 0.0), reverse=True)
        if elements:
            elem_title = CaptionLabel("Element distribution (active structures):", self)
            layout.addWidget(elem_title)
            elem_grid = QGridLayout()
            elem_grid.setContentsMargins(0, 0, 0, 0)
            elem_grid.setSpacing(4)
            headers = ["Element", "Atoms", "Structures", "Fraction", ""]
            for c, h in enumerate(headers):
                elem_grid.addWidget(CaptionLabel(h, self), 0, c)
            for r, elem in enumerate(elements[:max_rows_display], start=1):
                elem_grid.addWidget(CaptionLabel(str(elem.get("symbol", "")), self), r, 0)
                elem_grid.addWidget(CaptionLabel(str(elem.get("atoms", 0)), self), r, 1)
                elem_grid.addWidget(CaptionLabel(str(elem.get("structures", 0)), self), r, 2)
                frac = elem.get("fraction", 0.0) * 100.0
                elem_grid.addWidget(CaptionLabel(f"{frac:.1f} %", self), r, 3)
                bar = ProgressBar(self)
                bar.setRange(0, 100)
                bar.setValue(int(max(0, min(100, frac))))
                bar.setFixedWidth(120)
                elem_grid.addWidget(bar, r, 4)
            layout.addLayout(elem_grid)

        # Config_type distribution
        cfg = self._summary.get("config_types", [])
        if cfg:
            cfg_title = CaptionLabel(f"{group_label} distribution (active structures):", self)
            layout.addWidget(cfg_title)
            cfg_grid = QGridLayout()
            cfg_grid.setContentsMargins(0, 0, 0, 0)
            cfg_grid.setSpacing(4)
            headers = [group_label, "Count", "Fraction", ""]
            for c, h in enumerate(headers):
                cfg_grid.addWidget(CaptionLabel(h, self), 0, c)
            for r, item in enumerate(cfg[:max_rows_display], start=1):
                cfg_grid.addWidget(CaptionLabel(str(item.get("name", "")), self), r, 0)
                cfg_grid.addWidget(CaptionLabel(str(item.get("count", 0)), self), r, 1)
                frac = item.get("fraction", 0.0) * 100.0
                cfg_grid.addWidget(CaptionLabel(f"{frac:.1f} %", self), r, 2)
                bar = ProgressBar(self)
                bar.setRange(0, 100)
                bar.setValue(int(max(0, min(100, frac))))
                bar.setFixedWidth(120)
                cfg_grid.addWidget(bar, r, 3)
            layout.addLayout(cfg_grid)



    def _export_html(self) -> None:
        """Export the full summary (all rows) to an HTML file."""
        path = call_path_dialog(
            self,
            "Export dataset summary",
            "file",
            default_filename="dataset_summary.html",
            file_filter="HTML files (*.html);;All files (*.*)",
        )
        if not path:
            return
        try:
            html = self._build_html()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(html)
            MessageManager.send_info_message(f"Exported dataset summary to: {path}")
        except Exception:  # noqa: BLE001
            MessageManager.send_warning_message("Failed to export dataset summary.")

    def _build_html(self) -> str:
        """Render the summary into a highly decorated, professional HTML dashboard."""
        counts = self._summary.get("counts", {})
        atoms = self._summary.get("atoms", {})
        elements = sorted(self._summary.get("elements", []) or [], key=lambda x: x.get("fraction", 0.0), reverse=True)
        cfg = self._summary.get("config_types", []) or []
        dist = self._summary.get("numeric_distributions", {}) or {}
        dist_metrics = dist.get("metrics", []) or []
        force_rms = self._summary.get("force_rms", {}) or {}
        energy_stats = self._summary.get("energy", {}) or {}
        
        # Handling Grouping Logic (Formula or Tag)
        group_by = self._summary.get("group_by", "tag") # Default to tag
        group_label = "Formula" if "FORMULA" in str(group_by).upper() else "Config ID"
        group_section_title = "Formulas" if "FORMULA" in str(group_by).upper() else "Configuration Types"
        
        data_file = self._summary.get("data_file", "N/A")
        model_file = self._summary.get("model_file", "N/A")

        force_rms_rows = ""
        try:
            atoms_with_forces = int(force_rms.get("atoms_with_forces", 0) or 0)
            if atoms_with_forces > 0:
                rms_all = float(force_rms.get("rms_all_atoms", 0.0) or 0.0)
                force_rms_rows = (
                    f"<tr><td class='stat-label'>Force RMS(|F|) (all atoms)</td>"
                    f"<td class='stat-val'>{rms_all:.4g} eV/Å</td></tr>"
                )
        except Exception:  # noqa: BLE001
            force_rms_rows = ""

        energy_rows = ""
        try:
            e_count = int(energy_stats.get("count", 0) or 0)
            if e_count > 0:
                e_mean = float(energy_stats.get("mean", 0.0) or 0.0)
                e_std = float(energy_stats.get("std", 0.0) or 0.0)
                energy_rows = (
                    f"<tr><td class='stat-label'>Energy/atom stats</td>"
                    f"<td class='stat-val'>{e_mean:.6g} ± {e_std:.3g} eV/atom</td></tr>"
                )
        except Exception:  # noqa: BLE001
            energy_rows = ""

        # CSS Styles for Decoration
        style = """
        <style>
            :root {
                --primary: #2563eb;
                --primary-gradient: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
                --secondary: #64748b;
                --bg: #f8fafc;
                --card: #ffffff;
                --border: #e2e8f0;
                --text-dark: #0f172a;
                --accent-green: #10b981;
                --dist-fill: #2563eb;
                --dist-fill-soft: rgba(37, 99, 235, 0.18);
                --s1: #2563eb;
                --s2: #10b981;
                --s3: #f59e0b;
                --s4: #ef4444;
                --s5: #a855f7;
                --s6: #06b6d4;
            }

            body {
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background-color: var(--bg);
                color: var(--text-dark);
                line-height: 1.6;
                margin: 0;
                padding: 40px 20px;
                background-image: radial-gradient(#e2e8f0 0.5px, transparent 0.5px);
                background-size: 24px 24px;
            }

            .container { max-width: 1000px; margin: 0 auto; }

            header { margin-bottom: 40px; padding-bottom: 20px; position: relative; }
            header::before {
                content: ""; position: absolute; left: -20px; top: 0; bottom: 20px;
                width: 4px; background: var(--primary-gradient); border-radius: 4px;
            }

            h1 { margin: 0; font-size: 32px; letter-spacing: -0.8px; font-weight: 800; }
            .subtitle { color: var(--secondary); font-size: 14px; margin-top: 8px; font-weight: 500; }

            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; margin-bottom: 30px; }
            
            .card {
                background: var(--card); border-radius: 16px; padding: 24px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
                border: 1px solid var(--border);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .card:hover { transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
            .card h2 { font-size: 14px; text-transform: uppercase; color: var(--secondary); margin: 0 0 20px 0; letter-spacing: 0.1em; display: flex; align-items: center; gap: 8px; }
            .card h2::before { content: ""; display: inline-block; width: 8px; height: 8px; background: var(--primary); border-radius: 50%; }

            table { width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
            th { text-align: left; padding: 12px; background: #f8fafc; color: var(--secondary); font-weight: 600; border-bottom: 2px solid var(--border); text-transform: uppercase; font-size: 12px; }
            td { padding: 14px 12px; border-bottom: 1px solid var(--border); }
            tr:last-child td { border-bottom: none; }
            tr:hover { background-color: #fcfdfe; }

            .bar-wrap { display: flex; align-items: center; gap: 12px; }
            .bar-bg { flex-grow: 1; height: 8px; background: #f1f5f9; border-radius: 10px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05); }
            .bar-fill { height: 100%; background: var(--primary-gradient); border-radius: 10px; }

            .badge {
                background: #f0f7ff; color: var(--primary); padding: 4px 10px; border-radius: 8px;
                font-weight: 700; font-family: monospace; border: 1px solid #dbeafe;
            }

            .scroll-area { max-height: 400px; overflow-y: auto; padding-right: 4px; }
            .scroll-area::-webkit-scrollbar { width: 6px; }
            .scroll-area::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }

            .stat-val { font-weight: 700; color: var(--text-dark); }
            .stat-label { color: var(--secondary); }

            .dist-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
            .dist-card {
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 14px 14px 12px 14px;
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            }
            .dist-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
            .dist-title { font-weight: 800; font-size: 14px; letter-spacing: -0.2px; }
            .dist-meta { color: var(--secondary); font-size: 12px; }
            .dist-meta code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; }
            .dist-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
            .dist-legend-item { display: inline-flex; align-items: center; gap: 6px; color: var(--secondary); font-size: 11px; }
            .dist-swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
            .dist-axis { display: flex; justify-content: space-between; margin-top: 6px; color: var(--secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
            canvas.dist-canvas {
                width: 100%;
                height: 84px;
                display: block;
                border-radius: 10px;
                background: #ffffff;
            }
            canvas.dist-canvas.dist-multi { height: 110px; }
        </style>
        """

        # Elements Section
        el_rows = []
        for item in elements:
            pct = item.get("fraction", 0.0) * 100.0
            el_rows.append(f"""
            <tr>
                <td><span class="badge">{item.get("symbol", "")}</span></td>
                <td>{item.get("atoms", 0):,}</td>
                <td>{item.get("structures", 0)}</td>
                <td width="45%">
                    <div class="bar-wrap">
                        <div class="bar-bg"><div class="bar-fill" style="width: {pct:.1f}%"></div></div>
                        <span class="stat-val">{pct:.1f}%</span>
                    </div>
                </td>
            </tr>""")

        # Config Types Section
        cfg_rows = []
        for item in cfg:
            cfg_rows.append(f"""
            <tr>
                <td style="color: var(--text-dark); font-family: monospace;">{item.get("name", "")}</td>
                <td class="stat-val">{item.get("count", 0)}</td>
                <td>{item.get("fraction", 0.0) * 100.0:.1f}%</td>
            </tr>""")

        # Numeric Distributions Section (histograms)
        dist_cards = []
        for m in dist_metrics:
            key = str(m.get("key", ""))
            label = str(m.get("label", key))
            unit = str(m.get("unit", ""))
            total = int(m.get("total", 0) or 0)
            series = m.get("series", None)
            mn = float(m.get("min", 0.0) or 0.0)
            mx = float(m.get("max", 0.0) or 0.0)
            mean = m.get("mean", None)
            std = m.get("std", None)
            bins = int(m.get("bins", 0) or 0)
            if bins <= 0:
                if isinstance(series, list) and series:
                    bins = len(series[0].get("hist", []) or [])
                else:
                    bins = len(m.get("hist", []) or [])
            unit_html = f"&nbsp;<code>{html.escape(unit)}</code>" if unit else ""
            stats_html = ""
            try:
                if mean is not None:
                    mu = float(mean)
                    sigma = float(std) if std is not None else 0.0
                    if math.isfinite(mu) and math.isfinite(sigma):
                        stats_html = f" &nbsp; μ={mu:.6g}, σ={sigma:.3g}"
            except Exception:  # noqa: BLE001
                stats_html = ""

            # Tick labels (more granular than just the two ends)
            axis_left = float(m.get("hist_left", mn) or mn)
            axis_right = float(m.get("hist_right", mx) or mx)
            if axis_right == axis_left:
                tick_vals = [axis_left, axis_left, axis_left, axis_left, axis_left]
            else:
                tick_vals = [
                    axis_left,
                    axis_left + 0.25 * (axis_right - axis_left),
                    axis_left + 0.5 * (axis_right - axis_left),
                    axis_left + 0.75 * (axis_right - axis_left),
                    axis_right,
                ]
            ticks_html = "".join(f"<span>{v:.6g}</span>" for v in tick_vals)

            legend_html = ""
            canvas_class = "dist-canvas"
            if isinstance(series, list) and series:
                canvas_class = "dist-canvas dist-multi"
                swatches = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"]
                parts = []
                for i, s in enumerate(series[:6]):
                    name = html.escape(str(s.get("name", f"s{i+1}")))
                    color_var = swatches[i]
                    parts.append(
                        f"<span class='dist-legend-item'><span class='dist-swatch' style='background: var({color_var});'></span>{name}</span>"
                    )
                legend_html = f"<div class='dist-legend'>{''.join(parts)}</div>"
            dist_cards.append(
                f"""
                <div class="dist-card">
                    <div class="dist-head">
                        <div class="dist-title">{html.escape(label)}</div>
                        <div class="dist-meta"><code>{html.escape(key)}</code>{unit_html} &nbsp; range=[{mn:.6g}, {mx:.6g}] &nbsp; N={total:,} &nbsp; bins={bins:,}{stats_html}</div>
                    </div>
                    {legend_html}
                    <canvas class="{canvas_class}" width="900" height="84" data-key="{html.escape(key)}"></canvas>
                    <div class="dist-axis">{ticks_html}</div>
                </div>
                """
            )

        dist_block = ""
        if dist_cards:
            # Embed raw JSON for the canvas renderer; escape only the closing script tag sequence.
            dist_json = json.dumps(dist, ensure_ascii=False).replace("</", "<\\/")
            dist_block = f"""
            <div class="card" style="margin-top: 30px;">
                <h2>Numeric Distributions</h2>
                <div class="dist-grid">
                    {''.join(dist_cards)}
                </div>
            </div>

            <script id="dist-data" type="application/json">{dist_json}</script>
            <script>
            (() => {{
                const payloadEl = document.getElementById('dist-data');
                if (!payloadEl) return;
                let payload = null;
                try {{
                    payload = JSON.parse(payloadEl.textContent || '{{}}');
                }} catch (e) {{
                    return;
                }}
                const metrics = new Map((payload.metrics || []).map(m => [m.key, m]));

                function clamp(v, lo, hi) {{ return Math.min(hi, Math.max(lo, v)); }}

                function computeHistogram(vals, lo, hi, bins) {{
                    const range = hi - lo;
                    const counts = new Array(bins).fill(0);
                    if (!(range > 0)) return counts;
                    for (const v of vals) {{
                        const t = (v - lo) / range;
                        if (t < 0 || t > 1) continue;
                        const i = clamp(Math.floor(t * bins), 0, bins - 1);
                        counts[i] += 1;
                    }}
                    return counts;
                }}

                function toRgba(color, alpha) {{
                    const c = (color || '').trim();
                    const a = Number(alpha);
                    if (!Number.isFinite(a)) return c;
                    if (c.startsWith('#')) {{
                        const h = c.slice(1);
                        let r = 37, g = 99, b = 235;
                        if (h.length === 3) {{
                            r = parseInt(h[0] + h[0], 16);
                            g = parseInt(h[1] + h[1], 16);
                            b = parseInt(h[2] + h[2], 16);
                        }} else if (h.length === 6) {{
                            r = parseInt(h.slice(0, 2), 16);
                            g = parseInt(h.slice(2, 4), 16);
                            b = parseInt(h.slice(4, 6), 16);
                        }}
                        return `rgba(${{r}}, ${{g}}, ${{b}}, ${{a}})`;
                    }}
                    if (c.startsWith('rgba(')) {{
                        return c.replace(/rgba\\(([^)]+)\\)/, (_m, inner) => `rgba(${{inner.split(',').slice(0,3).join(',')}}, ${{a}})`);
                    }}
                    if (c.startsWith('rgb(')) {{
                        return c.replace('rgb(', 'rgba(').replace(')', `, ${{a}})`);
                    }}
                    return c;
                }}

                function drawHistogram(canvas, metric) {{
                    const ctx = canvas.getContext('2d');
                    if (!ctx) return;

                    const rect = canvas.getBoundingClientRect();
                    const dpr = window.devicePixelRatio || 1;
                    const w = Math.max(1, rect.width);
                    const h = Math.max(1, rect.height);
                    const bw = Math.max(1, Math.round(w * dpr));
                    const bh = Math.max(1, Math.round(h * dpr));
                    if (canvas.width !== bw || canvas.height !== bh) {{
                        canvas.width = bw;
                        canvas.height = bh;
                    }}
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    ctx.clearRect(0, 0, w, h);

                    let lo = Number(metric.min);
                    let hi = Number(metric.max);
                    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
                    if (hi === lo) {{ hi = lo + 1; lo = lo - 1; }}

                    const css = getComputedStyle(document.documentElement);
                    const fill = (css.getPropertyValue('--dist-fill') || '#2563eb').trim();
                    const fillSoft = (css.getPropertyValue('--dist-fill-soft') || 'rgba(37, 99, 235, 0.18)').trim();
                    const border = (css.getPropertyValue('--border') || '#e2e8f0').trim();
                    const seriesColors = [
                        (css.getPropertyValue('--s1') || '#2563eb').trim(),
                        (css.getPropertyValue('--s2') || '#10b981').trim(),
                        (css.getPropertyValue('--s3') || '#f59e0b').trim(),
                        (css.getPropertyValue('--s4') || '#ef4444').trim(),
                        (css.getPropertyValue('--s5') || '#a855f7').trim(),
                        (css.getPropertyValue('--s6') || '#06b6d4').trim(),
                    ];

                    const padX = 10;
                    const padY = 10;
                    const barW = w - padX * 2;
                    const barH = h - padY * 2;

                    const series = Array.isArray(metric.series) ? metric.series : null;
                    const singleHist = Array.isArray(metric.hist) ? metric.hist : null;
                    const bins = (metric.bins && Number.isFinite(metric.bins) && metric.bins > 0)
                        ? Math.max(1, Math.floor(metric.bins))
                        : (singleHist ? singleHist.length : (series && series.length && Array.isArray(series[0].hist) ? series[0].hist.length : 60));

                    // Vertical grid lines (x ticks)
                    ctx.strokeStyle = 'rgba(100, 116, 139, 0.18)';
                    ctx.lineWidth = 1;
                    for (const t of [0, 0.25, 0.5, 0.75, 1]) {{
                        const x = padX + t * barW + 0.5;
                        ctx.beginPath();
                        ctx.moveTo(x, padY);
                        ctx.lineTo(x, padY + barH);
                        ctx.stroke();
                    }}
                    // Baseline
                    ctx.strokeStyle = 'rgba(100, 116, 139, 0.28)';
                    ctx.beginPath();
                    ctx.moveTo(padX, padY + barH + 0.5);
                    ctx.lineTo(padX + barW, padY + barH + 0.5);
                    ctx.stroke();

                    if (series && series.length) {{
                        // Overlay density curves for multiple series (same x-range)
                        const countsBySeries = [];
                        let maxC = 1;
                        for (const s of series.slice(0, 6)) {{
                            const counts = Array.isArray(s.hist) ? s.hist : computeHistogram((s.sample || []).map(Number).filter(v => Number.isFinite(v)), lo, hi, bins);
                            countsBySeries.push(counts);
                            for (const c of counts) maxC = Math.max(maxC, c);
                        }}
                        const dx = bins > 1 ? (barW / (bins - 1)) : 0;
                        for (let si = 0; si < countsBySeries.length; si++) {{
                            const counts = countsBySeries[si];
                            const col = seriesColors[si] || fill;
                            ctx.beginPath();
                            for (let i = 0; i < bins; i++) {{
                                const x = padX + i * dx;
                                const yNorm = counts[i] / maxC;
                                const y = padY + (1 - yNorm) * barH;
                                if (i === 0) ctx.moveTo(x, y);
                                else ctx.lineTo(x, y);
                            }}
                            ctx.lineTo(padX + barW, padY + barH);
                            ctx.lineTo(padX, padY + barH);
                            ctx.closePath();
                            ctx.fillStyle = toRgba(col, 0.14) || fillSoft;
                            ctx.fill();
                            ctx.strokeStyle = col;
                            ctx.lineWidth = 1.2;
                            ctx.stroke();
                        }}
                    }} else {{
                        // Single-series histogram bars
                        const counts = singleHist ? singleHist : computeHistogram((metric.sample || []).map(Number).filter(v => Number.isFinite(v)), lo, hi, bins);
                        let maxC = 1;
                        for (const c of counts) maxC = Math.max(maxC, c);

                        const bw = barW / bins;
                        const grad = ctx.createLinearGradient(0, padY, 0, padY + barH);
                        grad.addColorStop(0, fill);
                        grad.addColorStop(1, fillSoft);
                        ctx.fillStyle = grad;
                        for (let i = 0; i < bins; i++) {{
                            const density = counts[i] / maxC;
                            const hBar = density * barH;
                            const x0 = padX + i * bw;
                            const y0 = padY + (barH - hBar);
                            ctx.fillRect(x0, y0, Math.max(1, bw - 1), hBar);
                        }}
                    }}

                    // Border
                    ctx.strokeStyle = border;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(padX + 0.5, padY + 0.5, barW - 1, barH - 1);
                }}

                document.querySelectorAll('canvas.dist-canvas').forEach((c) => {{
                    const key = c.dataset.key || '';
                    const metric = metrics.get(key);
                    if (metric) drawHistogram(c, metric);
                }});
            }})();
            </script>
            """

        # Full HTML Template
        return f"""<!doctype html>
    <html lang="en">
    <head>
        <meta charset='utf-8'>
        <title>Dataset Summary</title>
        {style}
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Dataset Summary Report</h1>
                <div class="subtitle">SOURCE: <strong>{data_file}</strong> &nbsp;&bull;&nbsp; MODEL: <strong>{model_file}</strong></div>
            </header>

            <div class="stats-grid">
                <div class="card">
                    <h2>Structure Overview</h2>
                    <table>
                        <tr><td class="stat-label">Original Count</td><td class="stat-val">{counts.get('orig_structures', 0)}</td></tr>
                        <tr><td class="stat-label">Active Structures</td><td class="stat-val" style="color: var(--accent-green);">{counts.get('active_structures', 0)}</td></tr>
                        <tr><td class="stat-label">Removed / Selected</td><td class="stat-val">{counts.get('removed_structures', 0)} / {counts.get('selected_structures', 0)}</td></tr>
                        <tr><td class="stat-label">Total Atoms (Active)</td><td class="stat-val">{atoms.get('total_atoms_active', 0):,}</td></tr>
                        {energy_rows}
                        {force_rms_rows}
                    </table>
                </div>

                <div class="card">
                    <h2>Atoms per Structure</h2>
                    <table>
                        <tr><td class="stat-label">Minimum</td><td class="stat-val">{atoms.get('min_atoms', 0)}</td></tr>
                        <tr><td class="stat-label">Maximum</td><td class="stat-val">{atoms.get('max_atoms', 0)}</td></tr>
                        <tr><td class="stat-label">Mean Value</td><td class="stat-val" style="font-size: 1.1em; color: var(--primary);">{atoms.get('mean_atoms', 0.0):.1f}</td></tr>
                        <tr><td class="stat-label">Median Value</td><td class="stat-val">{atoms.get('median_atoms', 0.0):.1f}</td></tr>
                    </table>
                </div>
            </div>

            {dist_block}

            <div class="card" style="margin-bottom: 30px;">
                <h2>Elemental Composition</h2>
                <table>
                    <thead><tr><th>Element</th><th>Atoms</th><th>Structures</th><th>Distribution (%)</th></tr></thead>
                    <tbody>{"".join(el_rows)}</tbody>
                </table>
            </div>

            <div class="card">
                <h2>{group_section_title}</h2>
                <div class="scroll-area">
                    <table>
                        <thead><tr><th>{group_label}</th><th>Count</th><th>Fraction</th></tr></thead>
                        <tbody>{"".join(cfg_rows)}</tbody>
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>"""

class GetStrMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, parent=None,tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.lineEdit = LineEdit(self)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.lineEdit)

        self.widget.setMinimumWidth(100 )


class SparseMessageBox(MessageBoxBase):
    """Dialog for configuring sparsity-related parameters."""

    def __init__(self, parent=None,tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self._frame = QFrame(self)
        self.frame_layout=QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0,0,0,0)
        self.frame_layout.setSpacing(4)
        self.intSpinBox = SpinBox(self)

        self.intSpinBox.setMaximum(9999999)
        self.intSpinBox.setMinimum(0)
        self.doubleSpinBox = DoubleSpinBox(self)
        self.doubleSpinBox.setDecimals(5)
        self.doubleSpinBox.setMinimum(0)
        self.doubleSpinBox.setMaximum(10)

        self.modeCombo = ComboBox(self)
        self.modeCombo.addItems(["Fixed count (FPS)", "R^2 stop (FPS)"])
        self.frame_layout.addWidget(CaptionLabel("Sampling mode", self),0,0,1,1)
        self.frame_layout.addWidget(self.modeCombo,0,1,1,2)

        self.maxNumLabel = CaptionLabel("Max num", self)
        self.frame_layout.addWidget(self.maxNumLabel,1,0,1,1)
        self.frame_layout.addWidget(self.intSpinBox,1,1,1,2)
        self.frame_layout.addWidget(CaptionLabel("Min distance", self),2,0,1,1)

        self.frame_layout.addWidget(self.doubleSpinBox,2,1,1,2)

        self.r2Label = CaptionLabel("R^2 threshold", self)
        self.r2SpinBox = DoubleSpinBox(self)
        self.r2SpinBox.setDecimals(4)
        self.r2SpinBox.setRange(0.0, 1.0)
        self.r2SpinBox.setSingleStep(0.01)
        self.frame_layout.addWidget(self.r2Label,3,0,1,1)
        self.frame_layout.addWidget(self.r2SpinBox,3,1,1,2)



        self.descriptorCombo = ComboBox(self)
        self.descriptorCombo.addItems(["Reduced (PCA)", "Raw descriptor"])
        self.frame_layout.addWidget(CaptionLabel("Descriptor source", self),4,0,1,1)
        self.frame_layout.addWidget(self.descriptorCombo,4,1,1,2)

        self.advancedFrame = QFrame(self)
        self.advancedFrame.setVisible(False)
        self.advancedLayout = QGridLayout(self.advancedFrame)
        self.advancedLayout.setContentsMargins(0,0,0,0)
        self.advancedLayout.setSpacing(4)



        self.trainingPathEdit = LineEdit(self)
        self.trainingPathEdit.setPlaceholderText("Optional training dataset path (.xyz or folder)")
        self.trainingPathEdit.setClearButtonEnabled(True)
        trainingPathWidget = QWidget(self)
        trainingPathLayout = QHBoxLayout(trainingPathWidget)
        trainingPathLayout.setContentsMargins(0, 0, 0, 0)
        trainingPathLayout.setSpacing(4)
        trainingPathLayout.addWidget(self.trainingPathEdit, 1)
        self.trainingBrowseButton = TransparentToolButton(FluentIcon.FOLDER_ADD, trainingPathWidget)
        trainingPathLayout.addWidget(self.trainingBrowseButton, 0)
        self.trainingBrowseButton.clicked.connect(self._pick_training_path)
        self.trainingBrowseButton.setToolTip("Browse for an existing training dataset")

        self.advancedLayout.addWidget(CaptionLabel("Training dataset", self),1,0)
        self.advancedLayout.addWidget(trainingPathWidget,1,1)

        # region option: use current selection as FPS region
        self.regionCheck = CheckBox("Use current selection as region", self)
        self.regionCheck.setToolTip("When FPS sampling is performed in the designated area, the program will automatically deselect it, just click to delete!")
        self.regionCheck.installEventFilter(ToolTipFilter(self.regionCheck, 300, ToolTipPosition.TOP))

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame )
        self.viewLayout.addWidget(self.advancedFrame)
        self.viewLayout.addWidget(self.regionCheck)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')

        self.widget.setMinimumWidth(200)
        self.advancedFrame.setVisible(True)
        self.modeCombo.currentIndexChanged.connect(self._update_mode_visibility)
        self._update_mode_visibility()



    def _pick_training_path(self):
        """Prompt the user to choose a training dataset path."""
        path = call_path_dialog(
            self,
            "Select training dataset",
            "select",
            file_filter="XYZ files (*.xyz);;All files (*.*)",
        )
        if not path:
            path = call_path_dialog(self, "Select training dataset folder", "directory")
        if path:
            self.trainingPathEdit.setText(path)

    def _update_mode_visibility(self):
        """Toggle UI elements based on sampling mode selection."""
        r2_mode = self.modeCombo.currentIndex() == 1
        self.maxNumLabel.setVisible(True)
        self.intSpinBox.setVisible(True)
        self.r2Label.setVisible(r2_mode)
        self.r2SpinBox.setVisible(r2_mode)


class IndexSelectMessageBox(MessageBoxBase):
    """Dialog for selecting structures by index."""

    def __init__(self, parent=None, tip="Specify index or slice"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.indexEdit = LineEdit(self)
        self.checkBox = CheckBox("Use original indices", self)
        self.checkBox.setChecked(True)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.indexEdit)
        self.viewLayout.addWidget(self.checkBox)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(200)


class RangeSelectMessageBox(MessageBoxBase):
    """Dialog for selecting structures by axis range."""

    def __init__(self, parent=None, tip="Specify x/y range"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)

        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.xMinSpin = DoubleSpinBox(self)
        self.xMinSpin.setDecimals(6)
        self.xMinSpin.setRange(-1e8, 1e8)
        self.xMaxSpin = DoubleSpinBox(self)
        self.xMaxSpin.setDecimals(6)
        self.xMaxSpin.setRange(-1e8, 1e8)
        self.yMinSpin = DoubleSpinBox(self)
        self.yMinSpin.setDecimals(6)
        self.yMinSpin.setRange(-1e8, 1e8)
        self.yMaxSpin = DoubleSpinBox(self)
        self.yMaxSpin.setDecimals(6)
        self.yMaxSpin.setRange(-1e8, 1e8)

        self.logicCombo = ComboBox(self)
        self.logicCombo.addItems(["AND", "OR"])

        self.frame_layout.addWidget(CaptionLabel("X min", self), 0, 0)
        self.frame_layout.addWidget(self.xMinSpin, 0, 1)
        self.frame_layout.addWidget(CaptionLabel("X max", self), 0, 2)
        self.frame_layout.addWidget(self.xMaxSpin, 0, 3)
        self.frame_layout.addWidget(CaptionLabel("Y min", self), 1, 0)
        self.frame_layout.addWidget(self.yMinSpin, 1, 1)
        self.frame_layout.addWidget(CaptionLabel("Y max", self), 1, 2)
        self.frame_layout.addWidget(self.yMaxSpin, 1, 3)
        self.frame_layout.addWidget(CaptionLabel("Logic", self), 2, 0)
        self.frame_layout.addWidget(self.logicCombo, 2, 1, 1, 3)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(300)


class LatticeRangeSelectMessageBox(MessageBoxBase):
    """Dialog for selecting structures by lattice parameters range."""

    def __init__(self, parent=None, tip="Specify lattice parameters range"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)

        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.aMinSpin = DoubleSpinBox(self)
        self.aMaxSpin = DoubleSpinBox(self)
        self.bMinSpin = DoubleSpinBox(self)
        self.bMaxSpin = DoubleSpinBox(self)
        self.cMinSpin = DoubleSpinBox(self)
        self.cMaxSpin = DoubleSpinBox(self)

        self.alphaMinSpin = DoubleSpinBox(self)
        self.alphaMaxSpin = DoubleSpinBox(self)
        self.betaMinSpin = DoubleSpinBox(self)
        self.betaMaxSpin = DoubleSpinBox(self)
        self.gammaMinSpin = DoubleSpinBox(self)
        self.gammaMaxSpin = DoubleSpinBox(self)

        spins = [
            self.aMinSpin, self.aMaxSpin, self.bMinSpin, self.bMaxSpin, self.cMinSpin, self.cMaxSpin,
            self.alphaMinSpin, self.alphaMaxSpin, self.betaMinSpin, self.betaMaxSpin, self.gammaMinSpin, self.gammaMaxSpin
        ]
        for spin in spins:
            spin.setDecimals(4)
            spin.setRange(0, 1e6)

        # Lattice constants labels
        self.frame_layout.addWidget(CaptionLabel("a min", self), 0, 0)
        self.frame_layout.addWidget(self.aMinSpin, 0, 1)
        self.frame_layout.addWidget(CaptionLabel("a max", self), 0, 2)
        self.frame_layout.addWidget(self.aMaxSpin, 0, 3)

        self.frame_layout.addWidget(CaptionLabel("b min", self), 1, 0)
        self.frame_layout.addWidget(self.bMinSpin, 1, 1)
        self.frame_layout.addWidget(CaptionLabel("b max", self), 1, 2)
        self.frame_layout.addWidget(self.bMaxSpin, 1, 3)

        self.frame_layout.addWidget(CaptionLabel("c min", self), 2, 0)
        self.frame_layout.addWidget(self.cMinSpin, 2, 1)
        self.frame_layout.addWidget(CaptionLabel("c max", self), 2, 2)
        self.frame_layout.addWidget(self.cMaxSpin, 2, 3)

        # Lattice angles labels
        self.frame_layout.addWidget(CaptionLabel("α min", self), 3, 0)
        self.frame_layout.addWidget(self.alphaMinSpin, 3, 1)
        self.frame_layout.addWidget(CaptionLabel("α max", self), 3, 2)
        self.frame_layout.addWidget(self.alphaMaxSpin, 3, 3)

        self.frame_layout.addWidget(CaptionLabel("β min", self), 4, 0)
        self.frame_layout.addWidget(self.betaMinSpin, 4, 1)
        self.frame_layout.addWidget(CaptionLabel("β max", self), 4, 2)
        self.frame_layout.addWidget(self.betaMaxSpin, 4, 3)

        self.frame_layout.addWidget(CaptionLabel("γ min", self), 5, 0)
        self.frame_layout.addWidget(self.gammaMinSpin, 5, 1)
        self.frame_layout.addWidget(CaptionLabel("γ max", self), 5, 2)
        self.frame_layout.addWidget(self.gammaMaxSpin, 5, 3)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(400)


class ArrowMessageBox(MessageBoxBase):
    """Dialog for selecting arrow display options."""

    def __init__(self, parent=None, props=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel("Vector property", self)
        self.titleLabel.setWordWrap(True)

        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.propCombo = ComboBox(self)
        if props:
            self.propCombo.addItems(props)

        self.scaleSpin = DoubleSpinBox(self)
        self.scaleSpin.setDecimals(3)
        self.scaleSpin.setRange(0, 1000)
        self.scaleSpin.setValue(1.0)

        self.colorCombo = ComboBox(self)
        self.colorCombo.addItems(["viridis", "magma", "plasma", "inferno", "jet"])

        self.showCheck = CheckBox("Show arrows", self)
        self.showCheck.setChecked(True)

        self.frame_layout.addWidget(CaptionLabel("Property", self), 0, 0)
        self.frame_layout.addWidget(self.propCombo, 0, 1)
        self.frame_layout.addWidget(CaptionLabel("Scale", self), 1, 0)
        self.frame_layout.addWidget(self.scaleSpin, 1, 1)
        self.frame_layout.addWidget(CaptionLabel("Colormap", self), 2, 0)
        self.frame_layout.addWidget(self.colorCombo, 2, 1)
        self.frame_layout.addWidget(self.showCheck, 3, 0, 1, 2)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(250)
class InputInfoMessageBox(MessageBoxBase):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel("new structure info", self)
        self.titleLabel.setWordWrap(True)

        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.keyEdit = LineEdit(self)
        self.valueEdit = LineEdit(self)
        self.frame_layout.addWidget(CaptionLabel("Key", self), 1, 0)
        self.frame_layout.addWidget(self.keyEdit, 1, 1, 1, 3)
        self.frame_layout.addWidget(CaptionLabel("Value", self), 2, 0)
        self.frame_layout.addWidget(self.valueEdit, 2, 1, 1, 3)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(100)
    def validate(self):
        if self.keyEdit.text().strip() != "":
            return True
        Flyout.create(
            icon=InfoBarIcon.INFORMATION,
            title='Tip',
            content="A valid value must be entered",
            target=self.keyEdit,
            parent=self,
            isClosable=True
        )
        return False
class EditInfoMessageBox(MessageBoxBase):
    """Dialog for editing structure information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel("Edit info", self)
        self.titleLabel.setWordWrap(True)
        self.new_tag_button = PrimaryPushButton(QIcon(":/images/src/images/copy_figure.svg"),
                                                         "Add new tag", self)
        self.new_tag_button.setMaximumWidth(200)
        self.new_tag_button.setObjectName("new_tag_button")
        self.new_tag_button.clicked.connect(self.new_tag)
        self.tag_group = TagGroup(parent=self)
        self.tag_group.tagRemovedSignal.connect(self.tag_removed)
        self.viewLayout.addWidget(self.new_tag_button)

        self.viewLayout.addWidget(self.tag_group)
        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(600)
        self.remove_tag = set()
        self.new_tag_info = {}
        self.rename_tag_map = {}
        self._display_to_original = {}
        self._suppress_tag_removed = False
    def new_tag(self):
        box = InputInfoMessageBox(self)
        if not box.exec():
            return
        key=box.keyEdit.text()
        value=box.valueEdit.text()

        if key.strip():
            self.add_tag(key.strip(),value)
    def init_tags(self, tags):
        for tag in tags:
            if tag == "species_id":
                continue
            btn = self.tag_group.add_tag(tag)
            btn.installEventFilter(self)
            self._display_to_original[tag] = tag
    def tag_removed(self,tag):
        if self._suppress_tag_removed:
            return
        if tag in self.new_tag_info.keys():
            self.new_tag_info.pop(tag)
        self.remove_tag.add(tag)
    def add_tag(self,tag,value):
        if self.tag_group.has_tag(tag):
            MessageManager.send_message_box(f"{tag} already exists, please delete it first")
            return
        self.remove_tag.discard(tag)
        self.new_tag_info[tag] = value
        btn = self.tag_group.add_tag(tag)
        btn.installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(obj, TagPushButton) and event.type() == QEvent.ContextMenu:
            old_name = obj.text()
            dlg = RenameTagMessageBox(old_name, self)
            if dlg.exec():
                new_name = dlg.nameEdit.text().strip()
                if not new_name or new_name == old_name:
                    return True
                self._rename_tag(old_name, new_name, obj)
            return True
        return super().eventFilter(obj, event)

    def _confirm_merge(self, title: str, content: str) -> bool:
        w = MessageBox(title, content, self)
        w.setClosableOnMaskClicked(True)
        return bool(w.exec())

    def _redirect_rename_targets(self, old_target: str, new_target: str) -> None:
        if old_target == new_target:
            return
        for src, dst in list(self.rename_tag_map.items()):
            if dst == old_target:
                self.rename_tag_map[src] = new_target

    def _remove_tag_silently(self, tag: str) -> None:
        self._suppress_tag_removed = True
        try:
            self.tag_group.del_tag(tag)
        finally:
            self._suppress_tag_removed = False

    def _rename_tag(self, old_name: str, new_name: str, obj: TagPushButton) -> None:
        if old_name in self.new_tag_info:
            value = self.new_tag_info[old_name]
            if self.tag_group.has_tag(new_name):
                content = (
                    f"Merge rename detected because '{new_name}' already exists.\n\n"
                    f"Effect after clicking Ok:\n"
                    f"- The new tag '{old_name}' will be merged into '{new_name}'.\n"
                    f"- On apply, key '{new_name}' will be set to the value entered for '{old_name}'.\n"
                    f"- If '{new_name}' already has a value, it will be overwritten.\n"
                    f"- The temporary key '{old_name}' will be discarded.\n"
                )
                if not self._confirm_merge("Merge rename confirmation", content):
                    return
                self.remove_tag.discard(new_name)
                self.new_tag_info[new_name] = value
                self.new_tag_info.pop(old_name, None)
                self._remove_tag_silently(old_name)
                return

            self.new_tag_info.pop(old_name, None)
            self.new_tag_info[new_name] = value
            obj.setText(new_name)
            self.tag_group.tags[new_name] = self.tag_group.tags.pop(old_name)
            return

        original_old = self._display_to_original.get(old_name, old_name)
        if self.tag_group.has_tag(new_name):
            content = (
                f"Merge rename detected because '{new_name}' already exists.\n\n"
                f"Effect after clicking Ok:\n"
                f"- For each selected structure, value under key '{original_old}' will be moved to '{new_name}'.\n"
                f"- If '{new_name}' already exists, it will be overwritten by the value from '{original_old}'.\n"
                f"- Key '{original_old}' will be removed.\n"
            )
            if not self._confirm_merge("Merge rename confirmation", content):
                return
            self.remove_tag.discard(new_name)
            self.rename_tag_map[original_old] = new_name
            self._redirect_rename_targets(old_name, new_name)
            self._display_to_original.pop(old_name, None)
            self._remove_tag_silently(old_name)
            return

        self.remove_tag.discard(new_name)
        self.rename_tag_map[original_old] = new_name
        self._redirect_rename_targets(old_name, new_name)
        obj.setText(new_name)
        self.tag_group.tags[new_name] = self.tag_group.tags.pop(old_name)
        self._display_to_original.pop(old_name, None)
        self._display_to_original[new_name] = original_old
    def validate(self):
        if len(self.new_tag_info)!=0 or len(self.remove_tag)!=0 or len(self.rename_tag_map)!=0:
            title = 'Modify information confirmation'
            remove_info=";".join(self.remove_tag)
            add_info="\n".join([f"{k}={v}" for k,v in self.new_tag_info.items()])
            rename_info = "\n".join([f"{k} -> {v}" for k, v in self.rename_tag_map.items()])
            content = (
                f"You removed the following information from the structure:\n{remove_info}\n\n"
                f"You renamed the following information keys:\n{rename_info}\n\n"
                f"You added the following information to the structure:\n{add_info}"
            )

            w = MessageBox(title, content, self)

            w.setClosableOnMaskClicked(True)


            if w.exec():

                return True
            else:
                return False
        return True


class RenameTagMessageBox(MessageBoxBase):
    def __init__(self, old_name: str, parent=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(f"Rename tag: {old_name}", self)
        self.titleLabel.setWordWrap(True)
        self.nameEdit = LineEdit(self)
        self.nameEdit.setText(old_name)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameEdit)
        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(320)

    def validate(self):
        if self.nameEdit.text().strip() != "":
            return True
        Flyout.create(
            icon=InfoBarIcon.INFORMATION,
            title='Tip',
            content="A valid value must be entered",
            target=self.nameEdit,
            parent=self,
            isClosable=True
        )
        return False

class ShiftEnergyMessageBox(MessageBoxBase):
    """Dialog for energy baseline shift parameters."""

    def __init__(self, parent=None, tip="Group regex patterns (comma separated)"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.groupEdit = LineEdit(self)
        self.presetCombo = ComboBox(self)
        # self.presetCombo.setEnabled(False)
        self.importButton = TransparentToolButton(FluentIcon.FOLDER_ADD, self)
        self.exportButton = TransparentToolButton(FluentIcon.SAVE, self)
        self.deleteButton = TransparentToolButton(FluentIcon.DELETE, self)
        self.deleteButton.setToolTip("Delete selected preset")
        self.deleteButton.installEventFilter(ToolTipFilter(self.deleteButton, 300, ToolTipPosition.TOP))
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        preset_row.addWidget(self.presetCombo, 1)
        preset_row.addWidget(self.importButton, 0)
        preset_row.addWidget(self.exportButton, 0)
        preset_row.addWidget(self.deleteButton, 0)
        self.savePresetCheck = CheckBox("Save baseline as preset", self)
        self.presetNameEdit = LineEdit(self)
        self.presetNameEdit.setPlaceholderText("Preset name")
        self.presetNameEdit.setEnabled(False)
        self.savePresetCheck.toggled.connect(self.presetNameEdit.setEnabled)

        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.genSpinBox = SpinBox(self)
        self.genSpinBox.setMaximum(100000000)
        self.sizeSpinBox = SpinBox(self)
        self.sizeSpinBox.setMaximum(999999)
        self.tolSpinBox = DoubleSpinBox(self)
        self.tolSpinBox.setDecimals(10)
        self.tolSpinBox.setMinimum(0)
        self.modeCombo = ComboBox(self)
        self.modeCombo.addItems([
            "REF_GROUP",
            "ZERO_BASELINE",
            "DFT_TO_NEP",
        ])
        self.modeCombo.setCurrentText("DFT_TO_NEP")


        self.frame_layout.addWidget(CaptionLabel("Max generations", self), 0, 0)
        self.frame_layout.addWidget(self.genSpinBox, 0, 1)
        self.frame_layout.addWidget(CaptionLabel("Population size", self), 1, 0)
        self.frame_layout.addWidget(self.sizeSpinBox, 1, 1)
        self.frame_layout.addWidget(CaptionLabel("Convergence tol", self), 2, 0)
        self.frame_layout.addWidget(self.tolSpinBox, 2, 1)
        self.frame_layout.addWidget(HyperlinkLabel(QUrl("https://github.com/brucefan1983/GPUMD/tree/master/tools/Analysis_and_Processing/energy-reference-aligner"),
                                                   "Alignment mode", self), 3, 0)
        self.frame_layout.addWidget(self.modeCombo, 3, 1)


        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(CaptionLabel("Use existing preset (optional)", self))
        self.viewLayout.addLayout(preset_row)
        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.setSpacing(4)
        save_row.addWidget(self.savePresetCheck)
        save_row.addWidget(self.presetNameEdit)
        self.viewLayout.addLayout(save_row)
        self.viewLayout.addWidget(self.groupEdit)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(250)




class ProgressDialog(FramelessDialog):

    def __init__(self,parent=None,title=""):
        pass
        super().__init__(parent)
        self.setStyleSheet('ProgressDialog{background:white}')


        FluentStyleSheet.DIALOG.apply(self)


        self.setWindowTitle(title)
        self.setFixedSize(300,100)
        self.__layout = QVBoxLayout(self)
        self.__layout.setContentsMargins(0,0,0,0)
        self.progressBar = ProgressBar(self)
        self.progressBar.setRange(0,100)
        self.progressBar.setValue(0)
        self.__layout.addWidget(self.progressBar)
        self.setLayout(self.__layout)
        self.__thread = LoadingThread(self, show_tip=False)
        self.__thread.finished.connect(self.close)

        self.__thread.progressSignal.connect(self.progressBar.setValue)
    def closeEvent(self,event):
        if self.__thread.isRunning():
            self.__thread.stop_work()
    def run_task(self,task_function,*args,**kwargs):
        self.__thread.start_work(task_function, *args, **kwargs)


class PeriodicTableDialog(FramelessDialog):
    """Dialog showing a simple periodic table."""

    elementSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))
        self.setWindowTitle("Periodic Table")
        self.setWindowIcon(QIcon(':/images/src/images/logo.svg'))
        self.resize(400, 350)


        with open(module_path / "Config/ptable.json" , "r", encoding="utf-8") as f:
            self.table_data = {int(k): v for k, v in json.load(f).items()}

        self.group_colors = {}
        for info in self.table_data.values():
            g = info.get("group", 0)
            if g not in self.group_colors:
                self.group_colors[g] = info.get("color", "#FFFFFF")

        self.__layout = QGridLayout(self)
        self.__layout.setContentsMargins(2, 2,2, 2)
        self.__layout.setSpacing(1)
        self.setLayout(self.__layout)
        self.__layout.setMenuBar(self.titleBar)

        # self.__layout.addWidget(self.titleBar,0,0,1,18)
        for num in range(1, 119):
            info = self.table_data.get(num)
            if not info:
                continue
            group = info.get("group", 0)
            period = self._get_period(num)
            row, col = self._grid_position(num, group, period)
            btn = QPushButton(info["symbol"], self)
            btn.setFixedSize(30,30)
            btn.setStyleSheet(f'background-color: {info.get("color", "#FFFFFF")};')
            btn.clicked.connect(lambda _=False, sym=info["symbol"]: self.elementSelected.emit(sym))
            self.__layout.addWidget(btn, row+1, col)
    def _get_period(self, num: int) -> int:
        if num <= 2:
            return 1
        elif num <= 10:
            return 2
        elif num <= 18:
            return 3
        elif num <= 36:
            return 4
        elif num <= 54:
            return 5
        elif num <= 86:
            return 6
        else:
            return 7

    def _grid_position(self, num: int, group: int, period: int) -> tuple[int, int]:
        if group == 0:
            if 57 <= num <= 71:
                row = 8
                col = num - 53
            elif 89 <= num <= 103:
                row = 9
                col = num - 85
            else:
                row, col = period, 1
        else:
            row, col = period, group
        return row - 1, col - 1



class DFTD3MessageBox(MessageBoxBase):
    """Dialog for DFTD3 parameters."""

    def __init__(self, parent=None, tip="DFTD3 correction"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.functionEdit = EditableComboBox(self)
        self.functionEdit.setPlaceholderText("dft d3 functional")
        functionals = [
            "b1b95",
            "b2gpplyp",
            "b2plyp",
            "b3lyp",
            "b3pw91",
            "b97d",
            "bhlyp",
            "blyp",
            "bmk",
            "bop",
            "bp86",
            "bpbe",
            "camb3lyp",
            "dsdblyp",
            "hcth120",
            "hf",
            "hse-hjs",
            "lc-wpbe08",
            "lcwpbe",
            "m11",
            "mn12l",
            "mn12sx",
            "mpw1b95",
            "mpwb1k",
            "mpwlyp",
            "n12sx",
            "olyp",
            "opbe",
            "otpss",
            "pbe",
            "pbe0",
            "pbe38",
            "pbesol",
            "ptpss",
            "pw6b95",
            "pwb6k",
            "pwpb95",
            "revpbe",
            "revpbe0",
            "revpbe38",
            "revssb",
            "rpbe",
            "rpw86pbe",
            "scan",
            "sogga11x",
            "ssb",
            "tpss",
            "tpss0",
            "tpssh",
            "b2kplyp",
            "dsd-pbep86",
            "b97m",
            "wb97x",
            "wb97m"
        ]
        self.functionEdit.addItems(functionals)
        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.d1SpinBox = DoubleSpinBox(self)
        self.d1SpinBox.setMaximum(100000000)
        self.d1SpinBox.setDecimals(3)

        self.d1cnSpinBox = DoubleSpinBox(self)
        self.d1cnSpinBox.setMaximum(999999)


        self.modeCombo = ComboBox(self)
        self.modeCombo.addItems([
            # "NEP Only",
            # "DFT-D3 only",
            # "NEP with DFT-D3",
            "Add DFT-D3",
            "Subtract DFT-D3",

        ])
        self.modeCombo.setCurrentText("NEP Only")


        self.frame_layout.addWidget(CaptionLabel("D3 cutoff ", self), 0, 0)
        self.frame_layout.addWidget(self.d1SpinBox, 0, 1)
        self.frame_layout.addWidget(CaptionLabel("D3 cutoff _cn ", self), 1, 0)
        self.frame_layout.addWidget(self.d1cnSpinBox, 1, 1)

        self.frame_layout.addWidget(CaptionLabel("Alignment mode", self), 3, 0)
        self.frame_layout.addWidget(self.modeCombo, 3, 1)


        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.functionEdit)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(250)


    def validate(self):
        if self.modeCombo.currentIndex()!=0:
            if len(self.functionEdit.text()) == 0:

                self.functionEdit.setFocus()
                return False
        return True
class ProjectInfoMessageBox(MessageBoxBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._widget = QWidget(self)

        self.widget_layout = QGridLayout(self._widget)

        self.parent_combox=ComboBox(self._widget)
        self.project_name=LineEdit(self._widget)
        self.project_name.setPlaceholderText("The name of the project")

        self.project_note=TextEdit(self._widget)
        self.project_note.setMinimumSize(200,100)
        self.project_note.setPlaceholderText("Notes on the project")
        self.widget_layout.addWidget(CaptionLabel("Parent",self), 0, 0)

        self.widget_layout.addWidget(self.parent_combox, 0, 1)

        self.widget_layout.addWidget(CaptionLabel("Project Name",self), 1, 0)
        self.widget_layout.addWidget(self.project_name, 1, 1)
        self.widget_layout.addWidget(CaptionLabel("Project Note",self), 2, 0 )
        self.widget_layout.addWidget(self.project_note, 2, 1 )
        self.viewLayout.addWidget(self._widget)
    def validate(self):
        project_name=self.project_name.text().strip()
        if len(project_name)==0:
            return False
        return True



class ModelInfoMessageBox(MessageBoxBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)


        self._widget = QWidget(self)
        self.viewLayout.addWidget(self._widget)
        root = QVBoxLayout(self._widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)


        titleBar = QFrame(self._widget)
        tLayout = QHBoxLayout(titleBar)
        tLayout.setContentsMargins(0, 0, 0, 0)
        tLayout.setSpacing(0)
        self.titleLabel = TitleLabel("Create / Edit Model", titleBar)

        self.titleLabel.setAlignment(Qt.AlignCenter)
        tLayout.addWidget(self.titleLabel)
        root.addWidget(titleBar)


        infoCard = QFrame(self._widget)
        info = QFormLayout(infoCard)
        info.setLabelAlignment(Qt.AlignRight)
        info.setHorizontalSpacing(5)
        info.setVerticalSpacing(2)

        self.parent_combox = ComboBox(infoCard)
        self.model_type_combox = ComboBox(infoCard)
        self.model_type_combox.addItems(["NEP"])
        self.model_name_edit = LineEdit(infoCard)
        self.model_name_edit.setPlaceholderText("The name of the model")

        info.addRow(CaptionLabel("Parent", self), self.parent_combox)
        info.addRow(CaptionLabel("Type", self), self.model_type_combox)
        info.addRow(CaptionLabel("Name", self), self.model_name_edit)


        rmseCard = QFrame(self._widget)
        rmse = QGridLayout(rmseCard)
        rmse.setContentsMargins(0, 0, 0, 0)
        rmse.setHorizontalSpacing(5)
        rmse.setVerticalSpacing(2)

        titleRmse = CaptionLabel("RMSE (energy / force / virial)", self)
        tf = titleRmse.font()
        tf.setBold(True)
        titleRmse.setFont(tf)

        self.energy_spinBox = LineEdit(rmseCard)
        self.force_spinBox  = LineEdit(rmseCard)
        self.virial_spinBox = LineEdit(rmseCard)
        self.energy_spinBox.setText("0")
        self.force_spinBox.setText("0")
        self.virial_spinBox.setText("0")


        validator = QDoubleValidator(bottom=-1e12, top=1e12, decimals=2)
        for w in (self.energy_spinBox, self.force_spinBox, self.virial_spinBox):
            w.setValidator(validator)
            w.setPlaceholderText("0.0")

        r = 0
        rmse.addWidget(titleRmse, r, 0, 1, 3)
        r += 1
        rmse.addWidget(CaptionLabel("energy", self), r, 0)
        rmse.addWidget(self.energy_spinBox, r, 1)
        rmse.addWidget(CaptionLabel("meV/atom", self), r, 2)
        r += 1
        rmse.addWidget(CaptionLabel("force",  self), r, 0)
        rmse.addWidget(self.force_spinBox,  r, 1)
        rmse.addWidget(CaptionLabel("meV/Å",    self), r, 2)
        r += 1
        rmse.addWidget(CaptionLabel("virial", self), r, 0)
        rmse.addWidget(self.virial_spinBox, r, 1)
        rmse.addWidget(CaptionLabel("meV/atom", self), r, 2)
        r += 1
        rmse.setColumnStretch(1, 1)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(2)
        row1.addWidget(infoCard, 2)
        row1.addWidget(rmseCard, 1)
        root.addLayout(row1)

        pathCard = QFrame(self._widget)
        path = QFormLayout(pathCard)
        path.setLabelAlignment(Qt.AlignRight)
        path.setHorizontalSpacing(5); path.setVerticalSpacing(3)


        structureRow = QWidget(pathCard)
        h = QHBoxLayout(structureRow)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        self.train_path_edit = LineEdit(structureRow)
        self.train_path_edit.setPlaceholderText("model train path")
        self.train_path_edit.editingFinished.connect(self.check_path)
        browse = TransparentToolButton(FluentIcon.FOLDER_ADD, structureRow)
        browse.setFixedHeight(self.train_path_edit.sizeHint().height())
        browse.clicked.connect(self._pick_file)
        h.addWidget(self.train_path_edit, 1)
        h.addWidget(browse, 0)




        path.addRow(CaptionLabel("Path", self), structureRow)

        root.addWidget(pathCard)

        tagsCard = QFrame(self._widget)
        tags = QFormLayout(tagsCard)
        tags.setLabelAlignment(Qt.AlignRight)
        tags.setHorizontalSpacing(0)
        tags.setVerticalSpacing(0)

        self.new_tag_edit = LineEdit(tagsCard)
        self.new_tag_edit.setPlaceholderText("Enter the tag and press Enter")
        self.new_tag_edit.returnPressed.connect(lambda :self.add_tag(self.new_tag_edit.text()))
        self.tag_group = TagGroup(parent=self)

        tags.addRow(CaptionLabel("Tags", self), self.new_tag_edit )
        tags.addRow(CaptionLabel(""), self.tag_group)  # 鐠?TagGroup 閻欘剙宕版稉鈧悰?
        root.addWidget(tagsCard)

        notesCard = QFrame(self._widget)
        notes = QFormLayout(notesCard)
        notes.setLabelAlignment(Qt.AlignRight)
        notes.setHorizontalSpacing(5)
        notes.setVerticalSpacing(0)

        self.model_note_edit = TextEdit(notesCard)
        self.model_note_edit.setPlaceholderText("Notes on the model")
        self.model_note_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # self.model_note_edit.setMinimumHeight(30)

        notes.addRow(CaptionLabel("Notes", self), self.model_note_edit)
        root.addWidget(notesCard)

        root.addStretch(1)



    def _pick_file(self):
        path=call_path_dialog(self,"Select the model folder path","directory")

        if path:
            self.train_path_edit.setText(path)
            self.check_path()
    def add_tag(self,tag ):
        if self.tag_group.has_tag(tag):
            MessageManager.send_info_message(f"{tag} already exists!")
            return

        self.tag_group.add_tag(tag)
    def check_path(self):
        _path=self.train_path_edit.text()
        path=Path(_path)
        if not path.exists():
            MessageManager.send_message_box(f"{_path} does not exist!")
            return
        if self.model_type_combox.currentText()=="NEP":
            model_file=path.joinpath("nep.txt")
            if not model_file.exists():
                MessageManager.send_message_box("No 'nep.txt' found in the specified path. Its presence is not strictly required, but please make sure you know what you are doing.")

            data_file=path.joinpath("train.xyz")
            if not data_file.exists():
                MessageManager.send_message_box("No 'nep.txt' found in the specified path. Its presence is not strictly required, but please make sure you know what you are doing.")
                # data_size=0
                energy=0
                force=0
                virial=0
            else:

                # data_size=get_xyz_nframe(data_file)
                # if data_size
                energy_array=read_nep_out_file(path.joinpath("energy_train.out"))
                energy = get_rmse(energy_array[:,0],energy_array[:,1])*1000
                force_array=read_nep_out_file(path.joinpath("force_train.out"))
                force = get_rmse(force_array[:,:3],force_array[:,3:])*1000
                virial_array=read_nep_out_file(path.joinpath("virial_train.out"))
                virial = get_rmse(virial_array[:,:6],virial_array[:,6:])*1000

            self.force_spinBox.setText(str(round(force,2)))
            self.energy_spinBox.setText(str(round(energy,2)))
            self.virial_spinBox.setText(str(round(virial,2)))
    def get_dict(self):
        path=Path(self.train_path_edit.text())
        data_file=path.joinpath("train.xyz")
        data_size = get_xyz_nframe(data_file)
        return dict(
            # project_id=self.,
            name=self.model_name_edit.text().strip(),
            model_type=self.model_type_combox.currentText(),
            model_path=self.train_path_edit.text().strip(),
            # model_file=path.joinpath("nep.txt"),
            # data_file=data_file,
            data_size=data_size,
            energy=float(self.energy_spinBox.text().strip()),
            force=float(self.force_spinBox.text().strip()),
            virial=float(self.virial_spinBox.text().strip()),

            notes=self.model_note_edit.toPlainText(),
            tags=list(self.tag_group.tags.keys()),
            parent_id=self.parent_combox.currentData()
        )
class AdvancedModelSearchDialog(MessageBoxBase):

    searchRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Search - Models")
        # self.setDraggable(True)
        self.setModal(False)
        # self.resize(640, 520)
        self._build_ui()
        self._wire_events()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)
        self.viewLayout.addLayout(root)
        # Title
        titleBar = QFrame(self)
        tLay = QHBoxLayout(titleBar); tLay.setContentsMargins(0, 0, 0, 0)
        self.titleLabel = TitleLabel("Advanced Model Search", titleBar)
        # f = self.titleLabel.font(); f.setPointSize(f.pointSize() + 3); f.setBold(True)
        # self.titleLabel.setFont(f)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        tLay.addWidget(self.titleLabel)
        root.addWidget(titleBar)

        formCard = QFrame(self); form = QFormLayout(formCard)
        form.setLabelAlignment(Qt.AlignRight); form.setHorizontalSpacing(3); form.setVerticalSpacing(3)

        self.projectIdsEdit = LineEdit(formCard)
        self.projectIdsEdit.setPlaceholderText("e.g. 1 or 1,3,5")
        self.includeDescendantsChk = CheckBox("Include sub-projects", formCard)
        self.includeDescendantsChk.setChecked(True)

        # Parent id
        self.parentIdEdit = LineEdit(formCard)
        self.parentIdEdit.setPlaceholderText("None or integer")
        self.parentIdEdit.setValidator(QIntValidator())

        self.nameContainsEdit = LineEdit(formCard)
        self.nameContainsEdit.setPlaceholderText("contains in name")
        self.notesContainsEdit = LineEdit(formCard)
        self.notesContainsEdit.setPlaceholderText("contains in notes")

        self.modelTypeCombo = ComboBox(formCard)
        self.modelTypeCombo.addItems(["<Any>", "NEP", "DeepMD", "Other"])

        self.tagsAllEdit  = LineEdit(formCard); self.tagsAllEdit.setPlaceholderText("tag1, tag2 (AND)")
        self.tagsAnyEdit  = LineEdit(formCard); self.tagsAnyEdit.setPlaceholderText("tag1, tag2 (OR)")
        self.tagsNoneEdit = LineEdit(formCard); self.tagsNoneEdit.setPlaceholderText("tag1, tag2 (NOT)")

        self.orderAscChk = CheckBox("Order by created_at ascending", formCard)
        self.orderAscChk.setChecked(True)
        self.limitEdit  = LineEdit(formCard); self.limitEdit.setPlaceholderText("e.g. 100"); self.limitEdit.setValidator(QIntValidator(0, 10**9))
        self.offsetEdit = LineEdit(formCard); self.offsetEdit.setPlaceholderText("e.g. 0");   self.offsetEdit.setValidator(QIntValidator(0, 10**9))

        form.addRow(CaptionLabel("Project ID(s):",self), self.projectIdsEdit)
        form.addRow(CaptionLabel("",self), self.includeDescendantsChk)
        form.addRow(CaptionLabel("Parent ID:",self), self.parentIdEdit)
        form.addRow(CaptionLabel("Model Type:",self), self.modelTypeCombo)
        form.addRow(CaptionLabel("Name contains:",self), self.nameContainsEdit)
        form.addRow(CaptionLabel("Notes contains:",self), self.notesContainsEdit)
        form.addRow(CaptionLabel("Tags (ALL):",self), self.tagsAllEdit)
        form.addRow(CaptionLabel("Tags (ANY):",self), self.tagsAnyEdit)
        form.addRow(CaptionLabel("Tags (NOT):",self), self.tagsNoneEdit)
        form.addRow(CaptionLabel("Order:",self), self.orderAscChk)
        form.addRow(CaptionLabel("Limit:",self), self.limitEdit)
        form.addRow(CaptionLabel("Offset:",self), self.offsetEdit)

        root.addWidget(formCard)


        self.buttonLayout.removeWidget(self.yesButton)
        self.buttonLayout.removeWidget(self.cancelButton)
        self.yesButton.hide()
        self.cancelButton.hide()
        self.searchBtn = PrimaryPushButton("Search", self)
        self.resetBtn  = PrimaryPushButton("Reset", self)
        self.closeBtn  = PrimaryPushButton("Close", self)
        self.buttonLayout.addWidget(self.searchBtn)
        self.buttonLayout.addWidget(self.resetBtn)
        self.buttonLayout.addWidget(self.closeBtn)


        root.addStretch(1)


    def _wire_events(self):
        self.searchBtn.clicked.connect(self._emit_params)
        self.resetBtn.clicked.connect(self._on_reset)
        self.closeBtn.clicked.connect(self.reject)
        self.projectIdsEdit.returnPressed.connect(self._emit_params)
        self.nameContainsEdit.returnPressed.connect(self._emit_params)
        self.notesContainsEdit.returnPressed.connect(self._emit_params)
        self.tagsAllEdit.returnPressed.connect(self._emit_params)
        self.tagsAnyEdit.returnPressed.connect(self._emit_params)
        self.tagsNoneEdit.returnPressed.connect(self._emit_params)

    @staticmethod
    def _split_csv(text: str) -> list[str]:
        if not text:
            return []
        out, seen = [], set()
        for part in text.split(","):
            s = part.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @staticmethod
    def _parse_project_ids(text: str) -> list[int]:
        if not text.strip():
            return []
        ids = []
        for part in text.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                ids.append(int(p))
            except ValueError:
                pass
        return ids

    def build_params(self) -> Dict[str, Any]:
        """收集并返回与 search_models_advanced 对应的参数字典。"""
        project_ids = self._parse_project_ids(self.projectIdsEdit.text())
        mt_text = self.modelTypeCombo.currentText()
        model_type = None if mt_text == "<Any>" else mt_text

        parent_text = self.parentIdEdit.text().strip()
        parent_id_val = int(parent_text) if parent_text.isdigit() else None

        params: Dict[str, Any] = dict(
            project_id=(
                project_ids[0] if len(project_ids) == 1
                else (project_ids if project_ids else None)
            ),
            include_descendants=self.includeDescendantsChk.isChecked(),
            parent_id=parent_id_val,
            name_contains=(self.nameContainsEdit.text().strip() or None),
            notes_contains=(self.notesContainsEdit.text().strip() or None),
            model_type=model_type,
            tags_all=self._split_csv(self.tagsAllEdit.text()),
            tags_any=self._split_csv(self.tagsAnyEdit.text()),
            tags_none=self._split_csv(self.tagsNoneEdit.text()),
            order_by_created_asc=self.orderAscChk.isChecked(),
        )

        limit_text = self.limitEdit.text().strip()
        if limit_text:
            params["limit"] = int(limit_text)
        offset_text = self.offsetEdit.text().strip()
        if offset_text:
            params["offset"] = int(offset_text)

        return params

    def _emit_params(self):
        params = self.build_params()
        self.searchRequested.emit(params)

    def _on_reset(self):
        self.projectIdsEdit.clear()
        self.includeDescendantsChk.setChecked(True)
        self.parentIdEdit.clear()
        self.modelTypeCombo.setCurrentIndex(0)
        self.nameContainsEdit.clear()
        self.notesContainsEdit.clear()
        self.tagsAllEdit.clear()
        self.tagsAnyEdit.clear()
        self.tagsNoneEdit.clear()
        self.orderAscChk.setChecked(True)
        self.limitEdit.clear()
        self.offsetEdit.clear()


class TagEditDialog(MessageBoxBase):
    """Dialog for editing tag properties."""

    def __init__(self, name: str, color: str, notes: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Tag")
        # self.resize(300, 200)

        layout = QVBoxLayout()
        self.viewLayout.addLayout(layout)
        form = QFormLayout()
        self.nameEdit = LineEdit(self)
        self.nameEdit.setText(name)
        self.colorEdit = LineEdit(self)
        self.colorEdit.setText(color)
        self.colorBtn = PrimaryPushButton("...", self)
        self.colorBtn.setFixedWidth(30)
        colorLayout = QHBoxLayout()
        colorLayout.setContentsMargins(0, 0, 0, 0)
        colorLayout.setSpacing(3)
        colorLayout.addWidget(self.colorEdit)
        colorLayout.addWidget(self.colorBtn)
        colorWidget = QWidget(self)
        colorWidget.setLayout(colorLayout)
        self.notesEdit = TextEdit(self)
        self.notesEdit.setPlainText(notes)

        form.addRow("Name", self.nameEdit)
        form.addRow("Color", colorWidget)
        form.addRow("Notes", self.notesEdit)
        layout.addLayout(form)



        self.colorBtn.clicked.connect(self._choose_color)


    def _choose_color(self):
        color_dialog = ColorDialog(QColor(self.colorEdit.text()),"Edit Tag Color", self)
        if color_dialog.exec():
            self.colorEdit.setText(color_dialog.color.name())

    def get_values(self) -> tuple[str, str, str]:
        return (
            self.nameEdit.text().strip(),
            self.colorEdit.text().strip(),
            self.notesEdit.toPlainText().strip(),
        )

class TagManageDialog(MessageBoxBase):
    """Dialog to create, edit and remove tags."""

    def __init__(self, tag_service, parent=None):
        super().__init__(parent)
        self._parent=parent
        self.tag_changed=False
        self.setWindowTitle("Manage Tags")
        self.tag_service = tag_service
        self._tag_map: dict[str, int] = {}
        # self.resize(360, 240)

        self._layout = QVBoxLayout()
        self.new_tag_edit = LineEdit(self)
        self.new_tag_edit.setMinimumWidth(300)
        self.new_tag_edit.setPlaceholderText("Enter the tag and press Enter")
        self.new_tag_edit.returnPressed.connect(self.add_tag)
        self.tag_group = TagGroup(parent=self)
        self.tag_group.setMinimumHeight(100)
        self.tag_group.tagRemovedSignal.connect(self.remove_tag)
        self._layout.addWidget(self.new_tag_edit)
        self._layout.addWidget(self.tag_group)
        self.viewLayout.addLayout(self._layout)


        self._load_tags()

    def _load_tags(self):
        for tag in self.tag_service.get_tags():
            btn = self.tag_group.add_tag(tag.name, color=tag.color)
            btn.setToolTip(tag.notes)
            btn.installEventFilter(self)
            self._tag_map[tag.name] = tag.tag_id

    def add_tag(self):
        name = self.new_tag_edit.text().strip()
        if not name:
            return
        if self.tag_group.has_tag(name):
            MessageManager.send_info_message(f"{name} already exists!")
            return
        item = self.tag_service.create_tag(name)
        if item:
            btn = self.tag_group.add_tag(item.name, color=item.color)
            btn.setToolTip(item.notes)
            btn.installEventFilter(self)
            self._tag_map[item.name] = item.tag_id
        self.new_tag_edit.clear()

    def remove_tag(self, name: str):
        tag_id = self._tag_map.pop(name, None)
        if tag_id is not None:
            self.tag_service.remove_tag(tag_id)

    def eventFilter(self, obj, event):

        if isinstance(obj, TagPushButton) and event.type() == QEvent.ContextMenu:
            old_name = obj.text()
            tag_id = self._tag_map.get(old_name)
            dlg = TagEditDialog(old_name, obj.backgroundColor, obj.toolTip(), self._parent)
            if dlg.exec():
                new_name, color, notes = dlg.get_values()
                if not new_name:
                    return True
                if new_name != old_name and self.tag_group.has_tag(new_name):
                    MessageManager.send_info_message(f"{new_name} already exists!")
                    return True
                self.tag_changed=True
                self.tag_service.update_tag(tag_id, name=new_name, color=color, notes=notes)
                obj.setText(new_name)
                obj.setBackgroundColor(color)
                obj.setToolTip(notes)
                if new_name != old_name:
                    self.tag_group.tags[new_name] = self.tag_group.tags.pop(old_name)
                    self._tag_map[new_name] = self._tag_map.pop(old_name)
            return True
        return super().eventFilter(obj, event)
