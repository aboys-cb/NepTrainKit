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
import os
from .button import TagPushButton, TagGroup

from NepTrainKit.core import MessageManager

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


class DatasetSummaryMessageBox(MessageBoxBase):
    """Frameless dialog that presents dataset-wide summary statistics."""

    def __init__(self, parent=None, summary: dict | None = None):
        super().__init__(parent)
        self._summary: dict[str, Any] = summary or {}

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
            cfg_title = CaptionLabel("Config_type distribution (active structures):", self)
            layout.addWidget(cfg_title)
            cfg_grid = QGridLayout()
            cfg_grid.setContentsMargins(0, 0, 0, 0)
            cfg_grid.setSpacing(4)
            headers = ["Config_type", "Count", "Fraction", ""]
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
        """Render the summary into a simple HTML table layout."""
        counts = self._summary.get("counts", {})
        atoms = self._summary.get("atoms", {})
        elements = sorted(self._summary.get("elements", []) or [], key=lambda x: x.get("fraction", 0.0), reverse=True)
        cfg = self._summary.get("config_types", []) or []
        data_file = self._summary.get("data_file", "")
        model_file = self._summary.get("model_file", "")
        def _table(rows: list[dict], headers: list[str], cols: list[str]) -> str:
            if not rows:
                return "<p>No data.</p>"
            body = "\n".join(
                "<tr>" + "".join(f"<td>{item.get(k, '')}</td>" for k in cols) + "</tr>"
                for item in rows
            )
            head = "".join(f"<th>{h}</th>" for h in headers)
            return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        style = """
        <style>
        body{font-family:Arial, sans-serif; margin:16px;}
        h1{margin-bottom:8px;}
        table{border-collapse:collapse; width:100%; margin:8px 0;}
        th,td{border:1px solid #ccc; padding:6px 8px; text-align:left;}
        th{background:#f6f6f6;}
        </style>
        """
        counts_html = f"""
        <p>Data: {data_file}<br/>Model: {model_file}</p>
        <table>
          <tbody>
            <tr><th>Orig structures</th><td>{counts.get('orig_structures', 0)}</td></tr>
            <tr><th>Active structures</th><td>{counts.get('active_structures', 0)}</td></tr>
            <tr><th>Removed structures</th><td>{counts.get('removed_structures', 0)}</td></tr>
            <tr><th>Selected structures</th><td>{counts.get('selected_structures', 0)}</td></tr>
            <tr><th>Total atoms (active)</th><td>{atoms.get('total_atoms_active', 0)}</td></tr>
            <tr><th>Atoms per structure</th><td>min={atoms.get('min_atoms', 0)}, max={atoms.get('max_atoms', 0)}, mean={atoms.get('mean_atoms', 0.0):.1f}, median={atoms.get('median_atoms', 0.0):.1f}</td></tr>
          </tbody>
        </table>
        """
        elements_html = _table(
            [
                {
                    "Element": item.get("symbol", ""),
                    "Atoms": item.get("atoms", 0),
                    "Structures": item.get("structures", 0),
                    "Fraction (%)": f"{item.get('fraction', 0.0) * 100.0:.1f}",
                }
                for item in elements
            ],
            ["Element", "Atoms", "Structures", "Fraction (%)"],
            ["Element", "Atoms", "Structures", "Fraction (%)"],
        )
        cfg_html = _table(
            [
                {
                    "Config_type": item.get("name", ""),
                    "Count": item.get("count", 0),
                    "Fraction (%)": f"{item.get('fraction', 0.0) * 100.0:.1f}",
                }
                for item in cfg
            ],
            ["Config_type", "Count", "Fraction (%)"],
            ["Config_type", "Count", "Fraction (%)"],
        )
        return f"<!doctype html><html><head><meta charset='utf-8'><title>Dataset summary</title>{style}</head><body><h1>Dataset Summary</h1>{counts_html}<h2>Elements</h2>{elements_html}<h2>Config Types</h2>{cfg_html}</body></html>"

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
        self.maxNumLabel.setVisible(not r2_mode)
        self.intSpinBox.setVisible(not r2_mode)
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
        self.remove_tag=set()
        self.new_tag_info={}
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
            self.tag_group.add_tag(tag)
    def tag_removed(self,tag):
        if tag in self.new_tag_info.keys():
            self.new_tag_info.pop(tag)
        self.remove_tag.add(tag)
    def add_tag(self,tag,value):
        if self.tag_group.has_tag(tag):
            MessageManager.send_message_box(f"{tag} already exists, please delete it first")
            return
        self.new_tag_info[tag] = value
        self.tag_group.add_tag(tag)
    def validate(self):
        if len(self.new_tag_info)!=0 or len(self.remove_tag)!=0:
            title = 'Modify information confirmation'
            remove_info=";".join(self.remove_tag)
            add_info="\n".join([f"{k}={v}" for k,v in self.new_tag_info.items()])
            content = f"""You removed the following information from the structure: \n{remove_info}  \nadded the following information: \n{add_info}"""

            w = MessageBox(title, content, self)

            w.setClosableOnMaskClicked(True)


            if w.exec():

                return True
            else:
                return False
        return True

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
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        preset_row.addWidget(self.presetCombo, 1)
        preset_row.addWidget(self.importButton, 0)
        preset_row.addWidget(self.exportButton, 0)
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
