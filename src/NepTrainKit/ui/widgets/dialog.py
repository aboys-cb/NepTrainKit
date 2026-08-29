#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 22:45
# @Author  : Bing
# @email    : 1747193328@qq.com
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict

from PySide6.QtGui import QIcon, QDoubleValidator, QIntValidator, QColor
from PySide6.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QGridLayout,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QFormLayout,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QDialog,
)
from PySide6.QtCore import Signal, Qt, QUrl, QEvent, QPropertyAnimation, QEasingCurve, QTimer
from qfluentwidgets import (
    MessageBoxBase,
    SpinBox,
    CaptionLabel,
    DoubleSpinBox,
    CheckBox,
    ProgressBar,
    ComboBox,
    FluentStyleSheet,
    FluentTitleBar,
    TransparentToolButton,
    ColorDialog,
    TitleLabel,
    HyperlinkLabel,
    PushButton,
    LineEdit,
    EditableComboBox,
    PrimaryPushButton,
    Flyout,
    InfoBarIcon,
    MessageBox,
    TextEdit,
    FluentIcon,
    ToolTipFilter,
    ToolTipPosition,
)
import json
import html
import math
import os
import sys
import numpy as np
from .button import TagPushButton, TagGroup

if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM", "").split(":")[0].lower() == "offscreen":
    class FramelessDialog(QDialog):
        """Headless-safe stand-in for qframelesswindow's mac native dialog."""

        def setTitleBar(self, title_bar):
            self.titleBar = title_bar
            title_bar.setParent(self)
else:
    from qframelesswindow import FramelessDialog

from NepTrainKit.core import MessageManager
from NepTrainKit.config import Config
from NepTrainKit.core.types import (
    SearchType,
    CanvasMode,
    Brushes,
    Pens,
    DistributionGroupMode,
    DistributionScope,
    DistributionValueView,
    DistributionSelectMode,
    DistributionCurveStyle,
)
from NepTrainKit.core.io.base import DistributionRequest, NepPlotData
from NepTrainKit.ui.canvas.canvas_factory import create_result_canvas, resolve_canvas_host_widget
from NepTrainKit.ui.canvas.distribution_factory import create_distribution_plot_adapter

from NepTrainKit import module_path

from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.core.utils import get_xyz_nframe, read_nep_out_file, get_rmse
from .distribution import DistributionExplorerWidget, DistributionInspectorMessageBox
from .periodic_table import PeriodicTableDialog


class GetIntMessageBox(MessageBoxBase):
    """Custom message box"""

    def __init__(self, parent=None, tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.intSpinBox = SpinBox(self)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.intSpinBox)

        self.widget.setMinimumWidth(100)
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
    """Choose an export layout and format-specific DeepMD options."""

    def __init__(
        self,
        parent=None,
        default_format: str = "xyz",
        group_by_config_type: bool = True,
        mixed_atom_numb_pad: int = 0,
    ):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(self.tr("Choose export format"), self)
        self.titleLabel.setWordWrap(True)

        self.formatCombo = ComboBox(self)
        self.formatCombo.addItem(self.tr("XYZ (.xyz / extxyz)"), userData="xyz")
        self.formatCombo.addItem(self.tr("DeepMD NPY"), userData="deepmd/npy")
        self.formatCombo.addItem(
            self.tr("DeepMD NPY (Mixed)"), userData="deepmd/npy/mixed"
        )

        self.standardGroupingWidget = QWidget(self)
        grouping_layout = QHBoxLayout(self.standardGroupingWidget)
        grouping_layout.setContentsMargins(0, 0, 0, 0)
        grouping_layout.addWidget(CaptionLabel(self.tr("Subfolder grouping"), self))
        self.standardGroupingCombo = ComboBox(self.standardGroupingWidget)
        self.standardGroupingCombo.addItem(
            self.tr("By Config_type"), userData="config_type"
        )
        self.standardGroupingCombo.addItem(
            self.tr("By chemical formula"), userData="formula"
        )
        self.standardGroupingCombo.setCurrentIndex(0 if group_by_config_type else 1)
        grouping_layout.addWidget(self.standardGroupingCombo, 1)

        self.mixedPaddingWidget = QWidget(self)
        padding_layout = QGridLayout(self.mixedPaddingWidget)
        padding_layout.setContentsMargins(0, 0, 0, 0)
        padding_layout.addWidget(CaptionLabel(self.tr("Virtual atom padding"), self), 0, 0)
        self.mixedPaddingSpinBox = SpinBox(self.mixedPaddingWidget)
        self.mixedPaddingSpinBox.setRange(0, 1000000)
        self.mixedPaddingSpinBox.setValue(max(0, int(mixed_atom_numb_pad)))
        padding_layout.addWidget(self.mixedPaddingSpinBox, 0, 1)
        padding_hint = CaptionLabel(
            self.tr("0 groups exact atom counts; 8 rounds them up to multiples of 8."),
            self.mixedPaddingWidget,
        )
        padding_hint.setWordWrap(True)
        padding_layout.addWidget(padding_hint, 1, 0, 1, 2)

        default = (default_format or "xyz").strip().lower()
        if default in {"deepmd/npy/mixed", "npy/mixed", "mixed"}:
            self.formatCombo.setCurrentIndex(2)
        elif default in {"deepmd", "deepmd/npy", "npy", "dp"}:
            self.formatCombo.setCurrentIndex(1)
        else:
            self.formatCombo.setCurrentIndex(0)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.formatCombo)
        self.viewLayout.addWidget(self.standardGroupingWidget)
        self.viewLayout.addWidget(self.mixedPaddingWidget)
        self.formatCombo.currentIndexChanged.connect(self._update_option_visibility)
        self._update_option_visibility()

        self.widget.setMinimumWidth(380)

    def _update_option_visibility(self, *_args) -> None:
        """Show only the options belonging to the selected DeepMD layout."""
        selected = self.formatCombo.currentData()
        self.standardGroupingWidget.setVisible(selected == "deepmd/npy")
        self.mixedPaddingWidget.setVisible(selected == "deepmd/npy/mixed")

    def selected_format(self) -> str:
        """Return the selected export format identifier."""
        data = self.formatCombo.currentData()
        if isinstance(data, str) and data:
            return data
        text = self.formatCombo.currentText().lower()
        if "mixed" in text:
            return "deepmd/npy/mixed"
        return "deepmd/npy" if "deepmd" in text or "npy" in text else "xyz"

    def group_by_config_type(self) -> bool:
        """Return standard NPY grouping; ignored for XYZ and mixed output."""
        return self.standardGroupingCombo.currentData() == "config_type"

    def mixed_atom_numb_pad(self) -> int | None:
        """Return dpdata-compatible Mixed padding, or None when disabled."""
        value = int(self.mixedPaddingSpinBox.value())
        return value if value > 0 else None


class GetStrMessageBox(MessageBoxBase):
    """Custom message box"""

    def __init__(self, parent=None, tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.lineEdit = LineEdit(self)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.lineEdit)

        self.widget.setMinimumWidth(100)


class SparseMessageBox(MessageBoxBase):
    """Dialog for configuring sparsity-related parameters."""

    def __init__(self, parent=None, tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(4)
        self.intSpinBox = SpinBox(self)

        self.intSpinBox.setMaximum(9999999)
        self.intSpinBox.setMinimum(0)
        self.doubleSpinBox = DoubleSpinBox(self)
        self.doubleSpinBox.setDecimals(5)
        self.doubleSpinBox.setMinimum(0)
        self.doubleSpinBox.setMaximum(10)

        self.strategyCombo = ComboBox(self)
        self.strategyCombo.addItem(self.tr("Global FPS (compatible)"), userData="global")
        self.strategyCombo.addItem(
            self.tr("Element-set balanced FPS"),
            userData="element_set",
        )
        self.strategyHint = CaptionLabel("", self)
        self.strategyHint.setWordWrap(True)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Selection strategy"), self), 0, 0, 1, 1)
        self.frame_layout.addWidget(self.strategyCombo, 0, 1, 1, 2)
        self.frame_layout.addWidget(self.strategyHint, 1, 1, 1, 2)

        self.modeCombo = ComboBox(self)
        self.modeCombo.addItem(self.tr("Fixed count (FPS)"), userData="count")
        self.modeCombo.addItem(self.tr("R^2 stop (FPS)"), userData="r2")
        self.modeLabel = CaptionLabel(self.tr("Sampling mode"), self)
        self.frame_layout.addWidget(self.modeLabel, 2, 0, 1, 1)
        self.frame_layout.addWidget(self.modeCombo, 2, 1, 1, 2)

        self.maxNumLabel = CaptionLabel(self.tr("Sample limit"), self)
        self.frame_layout.addWidget(self.maxNumLabel, 3, 0, 1, 1)
        self.frame_layout.addWidget(self.intSpinBox, 3, 1, 1, 2)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Min distance"), self), 4, 0, 1, 1)

        self.frame_layout.addWidget(self.doubleSpinBox, 4, 1, 1, 2)

        self.r2Label = CaptionLabel(self.tr("R^2 threshold"), self)
        self.r2SpinBox = DoubleSpinBox(self)
        self.r2SpinBox.setDecimals(4)
        self.r2SpinBox.setRange(0.0, 1.0)
        self.r2SpinBox.setSingleStep(0.01)
        self.frame_layout.addWidget(self.r2Label, 5, 0, 1, 1)
        self.frame_layout.addWidget(self.r2SpinBox, 5, 1, 1, 2)

        self.descriptorCombo = ComboBox(self)
        self.descriptorCombo.addItem(self.tr("Reduced (PCA)"), userData="reduced")
        self.descriptorCombo.addItem(self.tr("Raw descriptor"), userData="raw")
        self.descriptorLabel = CaptionLabel(self.tr("Descriptor source"), self)
        self.frame_layout.addWidget(self.descriptorLabel, 6, 0, 1, 1)
        self.frame_layout.addWidget(self.descriptorCombo, 6, 1, 1, 2)

        self.advancedFrame = QFrame(self)
        self.advancedFrame.setVisible(False)
        self.advancedLayout = QGridLayout(self.advancedFrame)
        self.advancedLayout.setContentsMargins(0, 0, 0, 0)
        self.advancedLayout.setSpacing(4)

        self.trainingPathEdit = LineEdit(self)
        self.trainingPathEdit.setPlaceholderText(self.tr("Optional training dataset path (.xyz or folder)"))
        self.trainingPathEdit.setClearButtonEnabled(True)
        trainingPathWidget = QWidget(self)
        trainingPathLayout = QHBoxLayout(trainingPathWidget)
        trainingPathLayout.setContentsMargins(0, 0, 0, 0)
        trainingPathLayout.setSpacing(4)
        trainingPathLayout.addWidget(self.trainingPathEdit, 1)
        self.trainingBrowseButton = TransparentToolButton(FluentIcon.FOLDER_ADD, trainingPathWidget)
        trainingPathLayout.addWidget(self.trainingBrowseButton, 0)
        self.trainingBrowseButton.clicked.connect(self._pick_training_path)
        self.trainingBrowseButton.setToolTip(self.tr("Browse for an existing training dataset"))
        self.trainingBrowseButton.setAccessibleName(
            self.tr("Browse for an existing training dataset")
        )

        self.advancedLayout.addWidget(CaptionLabel(self.tr("Training dataset"), self), 1, 0)
        self.advancedLayout.addWidget(trainingPathWidget, 1, 1)

        # region option: use current selection as FPS region
        self.regionCheck = CheckBox(self.tr("Use current selection as region"), self)
        self.regionCheck.setToolTip(
            self.tr("When FPS sampling is performed in the selected region, the program will automatically deselect it so you can delete it directly.")
        )
        self.regionCheck.installEventFilter(ToolTipFilter(self.regionCheck, 300, ToolTipPosition.TOP))

        # training overlay option
        self.trainingOverlayCheck = CheckBox(self.tr("Show training overlay"), self)
        self.trainingOverlayCheck.setToolTip(
            self.tr("Display a scatter plot showing training data, loaded data, and selected structures in PCA space after sampling.")
        )
        self.trainingOverlayCheck.installEventFilter(ToolTipFilter(self.trainingOverlayCheck, 300, ToolTipPosition.TOP))

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)
        self.viewLayout.addWidget(self.advancedFrame)
        self.viewLayout.addWidget(self.regionCheck)
        self.viewLayout.addWidget(self.trainingOverlayCheck)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))

        self.widget.setMinimumWidth(420)
        self.advancedFrame.setVisible(True)
        self._balanced_strategy_active = False
        self.modeCombo.currentIndexChanged.connect(self._update_mode_visibility)
        self.strategyCombo.currentIndexChanged.connect(self._update_strategy_visibility)
        self._update_strategy_visibility()

    def _pick_training_path(self):
        """Prompt the user to choose a training dataset path."""
        path = call_path_dialog(
            self,
            self.tr("Select training dataset"),
            "select",
            file_filter="XYZ files (*.xyz);;All files (*.*)",
        )
        if not path:
            path = call_path_dialog(self, self.tr("Select training dataset folder"), "directory")
        if path:
            self.trainingPathEdit.setText(path)

    def _update_mode_visibility(self):
        """Toggle UI elements based on sampling mode selection."""
        balanced = self.strategyCombo.currentData() == "element_set"
        r2_mode = not balanced and self.modeCombo.currentData() == "r2"
        self.maxNumLabel.setVisible(True)
        self.intSpinBox.setVisible(True)
        self.r2Label.setVisible(r2_mode)
        self.r2SpinBox.setVisible(r2_mode)

    def _update_strategy_visibility(self):
        """Keep balanced FPS on the validated raw, fixed-count workflow."""
        balanced = self.strategyCombo.currentData() == "element_set"
        if balanced and not self._balanced_strategy_active:
            self._global_mode_index = self.modeCombo.currentIndex()
            self._global_descriptor_index = self.descriptorCombo.currentIndex()
        elif not balanced and self._balanced_strategy_active:
            self.modeCombo.setCurrentIndex(getattr(self, "_global_mode_index", 0))
            self.descriptorCombo.setCurrentIndex(
                getattr(self, "_global_descriptor_index", 0)
            )

        self._balanced_strategy_active = balanced
        if balanced:
            self.modeCombo.setCurrentIndex(self.modeCombo.findData("count"))
            self.descriptorCombo.setCurrentIndex(self.descriptorCombo.findData("raw"))
            self.strategyHint.setText(
                self.tr(
                    "Groups by element set, assigns sqrt-size quotas, and uses raw descriptors."
                )
            )
        else:
            self.strategyHint.setText(
                self.tr("Uses the existing global FPS behavior and descriptor options.")
            )
        self.modeCombo.setEnabled(not balanced)
        self.descriptorCombo.setEnabled(not balanced)
        self._update_mode_visibility()


class IndexSelectMessageBox(MessageBoxBase):
    """Dialog for selecting structures by index."""

    def __init__(self, parent=None, tip="Specify index or slice"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.indexEdit = LineEdit(self)
        self.checkBox = CheckBox(self.tr("Use original indices"), self)
        self.checkBox.setChecked(True)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.indexEdit)
        self.viewLayout.addWidget(self.checkBox)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
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

        self.frame_layout.addWidget(CaptionLabel(self.tr("X min"), self), 0, 0)
        self.frame_layout.addWidget(self.xMinSpin, 0, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("X max"), self), 0, 2)
        self.frame_layout.addWidget(self.xMaxSpin, 0, 3)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Y min"), self), 1, 0)
        self.frame_layout.addWidget(self.yMinSpin, 1, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Y max"), self), 1, 2)
        self.frame_layout.addWidget(self.yMaxSpin, 1, 3)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Logic"), self), 2, 0)
        self.frame_layout.addWidget(self.logicCombo, 2, 1, 1, 3)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
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
            self.aMinSpin,
            self.aMaxSpin,
            self.bMinSpin,
            self.bMaxSpin,
            self.cMinSpin,
            self.cMaxSpin,
            self.alphaMinSpin,
            self.alphaMaxSpin,
            self.betaMinSpin,
            self.betaMaxSpin,
            self.gammaMinSpin,
            self.gammaMaxSpin,
        ]
        for spin in spins:
            spin.setDecimals(4)
            spin.setRange(0, 1e6)

        # Lattice constants labels
        self.frame_layout.addWidget(CaptionLabel(self.tr("a min"), self), 0, 0)
        self.frame_layout.addWidget(self.aMinSpin, 0, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("a max"), self), 0, 2)
        self.frame_layout.addWidget(self.aMaxSpin, 0, 3)

        self.frame_layout.addWidget(CaptionLabel(self.tr("b min"), self), 1, 0)
        self.frame_layout.addWidget(self.bMinSpin, 1, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("b max"), self), 1, 2)
        self.frame_layout.addWidget(self.bMaxSpin, 1, 3)

        self.frame_layout.addWidget(CaptionLabel(self.tr("c min"), self), 2, 0)
        self.frame_layout.addWidget(self.cMinSpin, 2, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("c max"), self), 2, 2)
        self.frame_layout.addWidget(self.cMaxSpin, 2, 3)

        # Lattice angles labels
        self.frame_layout.addWidget(CaptionLabel(self.tr("α min"), self), 3, 0)
        self.frame_layout.addWidget(self.alphaMinSpin, 3, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("α max"), self), 3, 2)
        self.frame_layout.addWidget(self.alphaMaxSpin, 3, 3)

        self.frame_layout.addWidget(CaptionLabel(self.tr("β min"), self), 4, 0)
        self.frame_layout.addWidget(self.betaMinSpin, 4, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("β max"), self), 4, 2)
        self.frame_layout.addWidget(self.betaMaxSpin, 4, 3)

        self.frame_layout.addWidget(CaptionLabel(self.tr("γ min"), self), 5, 0)
        self.frame_layout.addWidget(self.gammaMinSpin, 5, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("γ max"), self), 5, 2)
        self.frame_layout.addWidget(self.gammaMaxSpin, 5, 3)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(400)


class ArrowMessageBox(MessageBoxBase):
    """Dialog for selecting arrow display options."""

    def __init__(self, parent=None, props=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(self.tr("Vector property"), self)
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

        self.showCheck = CheckBox(self.tr("Show arrows"), self)
        self.showCheck.setChecked(True)

        self.frame_layout.addWidget(CaptionLabel(self.tr("Property"), self), 0, 0)
        self.frame_layout.addWidget(self.propCombo, 0, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Scale"), self), 1, 0)
        self.frame_layout.addWidget(self.scaleSpin, 1, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Colormap"), self), 2, 0)
        self.frame_layout.addWidget(self.colorCombo, 2, 1)
        self.frame_layout.addWidget(self.showCheck, 3, 0, 1, 2)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(250)


class InputInfoMessageBox(MessageBoxBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(self.tr("New structure info"), self)
        self.titleLabel.setWordWrap(True)

        self._frame = QFrame(self)
        self.frame_layout = QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(2)

        self.keyEdit = LineEdit(self)
        self.valueEdit = LineEdit(self)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Key"), self), 1, 0)
        self.frame_layout.addWidget(self.keyEdit, 1, 1, 1, 3)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Value"), self), 2, 0)
        self.frame_layout.addWidget(self.valueEdit, 2, 1, 1, 3)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(100)

    def validate(self):
        if self.keyEdit.text().strip() != "":
            return True
        Flyout.create(
            icon=InfoBarIcon.INFORMATION,
            title=self.tr("Tip"),
            content=self.tr("A valid value must be entered"),
            target=self.keyEdit,
            parent=self,
            isClosable=True,
        )
        return False


class EditInfoMessageBox(MessageBoxBase):
    """Dialog for editing structure information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(self.tr("Edit info"), self)
        self.titleLabel.setWordWrap(True)
        self.new_tag_button = PrimaryPushButton(QIcon(":/images/src/images/copy_figure.svg"), self.tr("Add new tag"), self)
        self.new_tag_button.setMaximumWidth(200)
        self.new_tag_button.setObjectName("new_tag_button")
        self.new_tag_button.clicked.connect(self.new_tag)
        self.tag_group = TagGroup(parent=self)
        self.tag_group.tagRemovedSignal.connect(self.tag_removed)
        self.viewLayout.addWidget(self.new_tag_button)

        self.viewLayout.addWidget(self.tag_group)
        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
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
        key = box.keyEdit.text()
        value = box.valueEdit.text()

        if key.strip():
            self.add_tag(key.strip(), value)

    def init_tags(self, tags):
        for tag in tags:
            if tag == "species_id":
                continue
            btn = self.tag_group.add_tag(tag)
            btn.installEventFilter(self)
            self._display_to_original[tag] = tag

    def tag_removed(self, tag):
        if self._suppress_tag_removed:
            return
        if tag in self.new_tag_info.keys():
            self.new_tag_info.pop(tag)
        self.remove_tag.add(tag)

    def add_tag(self, tag, value):
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
        if len(self.new_tag_info) != 0 or len(self.remove_tag) != 0 or len(self.rename_tag_map) != 0:
            title = "Modify information confirmation"
            remove_info = ";".join(self.remove_tag)
            add_info = "\n".join([f"{k}={v}" for k, v in self.new_tag_info.items()])
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
        self.titleLabel = CaptionLabel(self.tr("Rename tag: {name}").format(name=old_name), self)
        self.titleLabel.setWordWrap(True)
        self.nameEdit = LineEdit(self)
        self.nameEdit.setText(old_name)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameEdit)
        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(320)

    def validate(self):
        if self.nameEdit.text().strip() != "":
            return True
        Flyout.create(
            icon=InfoBarIcon.INFORMATION,
            title=self.tr("Tip"),
            content=self.tr("A valid value must be entered"),
            target=self.nameEdit,
            parent=self,
            isClosable=True,
        )
        return False


@dataclass
class ShiftEnergyDialogValues:
    """Collected user inputs for energy baseline shifting."""

    group_patterns: list[str] = field(default_factory=list)
    alignment_mode: str = "DFT_TO_NEP"
    max_generations: int = 100000
    population_size: int = 40
    convergence_tol: float = 1e-8
    selected_preset_name: str = ""
    save_preset: bool = False
    preset_name: str = ""


class ShiftEnergyMessageBox(MessageBoxBase):
    """Dialog for energy baseline shift parameters."""

    def __init__(
        self,
        parent=None,
        tip="Use .* for one shared baseline; separate different Config_type baseline groups with semicolons.",
    ):
        super().__init__(parent)
        self._preset_placeholder = "None"
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.groupEdit = LineEdit(self)
        self.presetCombo = ComboBox(self)
        # self.presetCombo.setEnabled(False)
        self.importButton = TransparentToolButton(FluentIcon.FOLDER_ADD, self)
        self.exportButton = TransparentToolButton(FluentIcon.SAVE, self)
        self.deleteButton = TransparentToolButton(FluentIcon.DELETE, self)
        self.deleteButton.setToolTip(self.tr("Delete selected preset"))
        self.deleteButton.installEventFilter(ToolTipFilter(self.deleteButton, 300, ToolTipPosition.TOP))
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        preset_row.addWidget(self.presetCombo, 1)
        preset_row.addWidget(self.importButton, 0)
        preset_row.addWidget(self.exportButton, 0)
        preset_row.addWidget(self.deleteButton, 0)
        self.savePresetCheck = CheckBox(self.tr("Save baseline as preset"), self)
        self.presetNameEdit = LineEdit(self)
        self.presetNameEdit.setPlaceholderText(self.tr("Preset name"))
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
        self.modeCombo.addItem(self.tr("Reference group"), userData="REF_GROUP")
        self.modeCombo.addItem(self.tr("Zero baseline"), userData="ZERO_BASELINE")
        self.modeCombo.addItem(self.tr("DFT to NEP"), userData="DFT_TO_NEP")
        self._set_alignment_mode("DFT_TO_NEP")

        self.frame_layout.addWidget(CaptionLabel(self.tr("Max generations"), self), 0, 0)
        self.frame_layout.addWidget(self.genSpinBox, 0, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Population size"), self), 1, 0)
        self.frame_layout.addWidget(self.sizeSpinBox, 1, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("Convergence tolerance"), self), 2, 0)
        self.frame_layout.addWidget(self.tolSpinBox, 2, 1)
        self.frame_layout.addWidget(
            HyperlinkLabel(
                QUrl(
                    "https://github.com/brucefan1983/GPUMD/tree/master/tools/Analysis_and_Processing/energy-reference-aligner"
                ),
                self.tr("Alignment mode"),
                self,
            ),
            3,
            0,
        )
        self.frame_layout.addWidget(self.modeCombo, 3, 1)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(CaptionLabel(self.tr("Use existing preset (optional)"), self))
        self.viewLayout.addLayout(preset_row)
        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.setSpacing(4)
        save_row.addWidget(self.savePresetCheck)
        save_row.addWidget(self.presetNameEdit)
        self.viewLayout.addLayout(save_row)
        self.viewLayout.addWidget(self.groupEdit)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(250)

    def _set_alignment_mode(self, mode: str) -> None:
        for index in range(self.modeCombo.count()):
            if self.modeCombo.itemData(index) == mode:
                self.modeCombo.setCurrentIndex(index)
                return
        self.modeCombo.setCurrentIndex(2)

    def _alignment_mode(self) -> str:
        value = self.modeCombo.currentData()
        return str(value) if value else "DFT_TO_NEP"

    def set_defaults(
        self,
        suggested_patterns: list[str] | None,
        max_generations: int,
        population_size: int,
        convergence_tol: float,
    ) -> None:
        """Populate dialog inputs with default values."""
        patterns = [p.strip() for p in (suggested_patterns or []) if str(p).strip()]
        self.groupEdit.setText(";".join(patterns))
        self.genSpinBox.setValue(int(max_generations))
        self.sizeSpinBox.setValue(int(population_size))
        self.tolSpinBox.setValue(float(convergence_tol))

    def set_preset_names(self, names: list[str], placeholder: str = "None") -> None:
        """Refresh available preset names and keep the placeholder entry."""
        self._preset_placeholder = placeholder
        blocked = self.presetCombo.blockSignals(True)
        self.presetCombo.clear()
        self.presetCombo.addItem(placeholder)
        for name in names:
            text = (name or "").strip()
            if text and text != placeholder:
                self.presetCombo.addItem(text)
        self.presetCombo.setCurrentText(placeholder)
        self.presetCombo.blockSignals(blocked)

    def apply_preset_to_inputs(self, preset: Any, fallback_patterns: list[str] | None) -> None:
        """Apply preset values to editable widgets, or fallback defaults."""
        fallback = ";".join([p.strip() for p in (fallback_patterns or []) if str(p).strip()])
        if preset is None:
            self.groupEdit.setText(fallback)
            return
        patterns = list(getattr(preset, "group_patterns", []) or [])
        self.groupEdit.setText(";".join(patterns) if patterns else fallback)
        mode = getattr(preset, "alignment_mode", "")
        if mode:
            self._set_alignment_mode(str(mode))
        optimizer = dict(getattr(preset, "optimizer", {}) or {})
        try:
            if "max_generations" in optimizer:
                self.genSpinBox.setValue(int(optimizer["max_generations"]))
            if "population_size" in optimizer:
                self.sizeSpinBox.setValue(int(optimizer["population_size"]))
            if "convergence_tol" in optimizer:
                self.tolSpinBox.setValue(float(optimizer["convergence_tol"]))
        except Exception:
            pass

    def collect_values(self) -> ShiftEnergyDialogValues:
        """Read all user inputs and return a typed payload."""
        pattern_text = self.groupEdit.text().strip()
        group_patterns = [p.strip() for p in pattern_text.split(";") if p.strip()]
        selected_preset_name = self.presetCombo.currentText().strip()
        if selected_preset_name == self._preset_placeholder:
            selected_preset_name = ""
        return ShiftEnergyDialogValues(
            group_patterns=group_patterns,
            alignment_mode=self._alignment_mode(),
            max_generations=int(self.genSpinBox.value()),
            population_size=int(self.sizeSpinBox.value()),
            convergence_tol=float(self.tolSpinBox.value()),
            selected_preset_name=selected_preset_name,
            save_preset=bool(self.savePresetCheck.isChecked()),
            preset_name=self.presetNameEdit.text().strip(),
        )


class ProgressDialog(FramelessDialog):
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.setStyleSheet("ProgressDialog{background:white}")

        FluentStyleSheet.DIALOG.apply(self)

        self.setWindowTitle(title)
        self.setFixedSize(300, 100)
        self.__layout = QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.progressBar = ProgressBar(self)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.__layout.addWidget(self.progressBar)
        self.setLayout(self.__layout)
        self.__thread = BackgroundTask(self, show_tip=False)
        self.__thread.finished.connect(self.close)

        self.__thread.progressSignal.connect(self.progressBar.setValue)

    def closeEvent(self, event):
        if self.__thread.isRunning():
            self.__thread.stop_work()

    def run_task(self, task_function, *args, **kwargs):
        self.__thread.start_work(task_function, *args, **kwargs)


class DFTD3MessageBox(MessageBoxBase):
    """Dialog for DFTD3 parameters."""

    def __init__(self, parent=None, tip="DFTD3 correction"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.functionEdit = EditableComboBox(self)
        self.functionEdit.setPlaceholderText(self.tr("DFT D3 functional"))
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
            "wb97m",
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
        self.modeCombo.addItem(self.tr("Add DFT-D3"))
        self.modeCombo.addItem(self.tr("Subtract DFT-D3"))
        self.modeCombo.setCurrentIndex(0)

        self.frame_layout.addWidget(CaptionLabel(self.tr("D3 cutoff"), self), 0, 0)
        self.frame_layout.addWidget(self.d1SpinBox, 0, 1)
        self.frame_layout.addWidget(CaptionLabel(self.tr("D3 cutoff _cn"), self), 1, 0)
        self.frame_layout.addWidget(self.d1cnSpinBox, 1, 1)

        self.frame_layout.addWidget(CaptionLabel(self.tr("Alignment mode"), self), 3, 0)
        self.frame_layout.addWidget(self.modeCombo, 3, 1)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.functionEdit)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText(self.tr("OK"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(250)

    def validate(self):
        if self.modeCombo.currentIndex() != 0:
            if len(self.functionEdit.text()) == 0:
                self.functionEdit.setFocus()
                return False
        return True


class ProjectInfoMessageBox(MessageBoxBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._widget = QWidget(self)

        self.widget_layout = QGridLayout(self._widget)

        self.parent_combox = ComboBox(self._widget)
        self.project_name = LineEdit(self._widget)
        self.project_name.setPlaceholderText(self.tr("Project name"))

        self.project_note = TextEdit(self._widget)
        self.project_note.setMinimumSize(200, 100)
        self.project_note.setPlaceholderText(self.tr("Project notes"))
        self.widget_layout.addWidget(CaptionLabel(self.tr("Parent"), self), 0, 0)

        self.widget_layout.addWidget(self.parent_combox, 0, 1)

        self.widget_layout.addWidget(CaptionLabel(self.tr("Project name"), self), 1, 0)
        self.widget_layout.addWidget(self.project_name, 1, 1)
        self.widget_layout.addWidget(CaptionLabel(self.tr("Project notes"), self), 2, 0)
        self.widget_layout.addWidget(self.project_note, 2, 1)
        self.viewLayout.addWidget(self._widget)

    def validate(self):
        project_name = self.project_name.text().strip()
        if len(project_name) == 0:
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
        self.titleLabel = TitleLabel(self.tr("Create / edit model"), titleBar)

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
        self.model_name_edit.setPlaceholderText(self.tr("Model name"))

        info.addRow(CaptionLabel(self.tr("Parent"), self), self.parent_combox)
        info.addRow(CaptionLabel(self.tr("Type"), self), self.model_type_combox)
        info.addRow(CaptionLabel(self.tr("Name"), self), self.model_name_edit)

        rmseCard = QFrame(self._widget)
        rmse = QGridLayout(rmseCard)
        rmse.setContentsMargins(0, 0, 0, 0)
        rmse.setHorizontalSpacing(5)
        rmse.setVerticalSpacing(2)

        titleRmse = CaptionLabel(self.tr("RMSE (energy / force / virial)"), self)
        tf = titleRmse.font()
        tf.setBold(True)
        titleRmse.setFont(tf)

        self.energy_spinBox = LineEdit(rmseCard)
        self.force_spinBox = LineEdit(rmseCard)
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
        rmse.addWidget(CaptionLabel(self.tr("energy"), self), r, 0)
        rmse.addWidget(self.energy_spinBox, r, 1)
        rmse.addWidget(CaptionLabel("meV/atom", self), r, 2)
        r += 1
        rmse.addWidget(CaptionLabel(self.tr("force"), self), r, 0)
        rmse.addWidget(self.force_spinBox, r, 1)
        rmse.addWidget(CaptionLabel("meV/Å", self), r, 2)
        r += 1
        rmse.addWidget(CaptionLabel(self.tr("virial"), self), r, 0)
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
        path.setHorizontalSpacing(5)
        path.setVerticalSpacing(3)

        structureRow = QWidget(pathCard)
        h = QHBoxLayout(structureRow)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        self.train_path_edit = LineEdit(structureRow)
        self.train_path_edit.setPlaceholderText(self.tr("Model training path"))
        self.train_path_edit.editingFinished.connect(self.check_path)
        browse = TransparentToolButton(FluentIcon.FOLDER_ADD, structureRow)
        browse.setFixedHeight(self.train_path_edit.sizeHint().height())
        browse.clicked.connect(self._pick_file)
        h.addWidget(self.train_path_edit, 1)
        h.addWidget(browse, 0)

        path.addRow(CaptionLabel(self.tr("Path"), self), structureRow)

        root.addWidget(pathCard)

        tagsCard = QFrame(self._widget)
        tags = QFormLayout(tagsCard)
        tags.setLabelAlignment(Qt.AlignRight)
        tags.setHorizontalSpacing(0)
        tags.setVerticalSpacing(0)

        self.new_tag_edit = LineEdit(tagsCard)
        self.new_tag_edit.setPlaceholderText(self.tr("Enter the tag and press Enter"))
        self.new_tag_edit.returnPressed.connect(lambda: self.add_tag(self.new_tag_edit.text()))
        self.tag_group = TagGroup(parent=self)

        tags.addRow(CaptionLabel(self.tr("Tags"), self), self.new_tag_edit)
        tags.addRow(CaptionLabel(""), self.tag_group)  # 鐠?TagGroup 閻欘剙宕版稉鈧悰?
        root.addWidget(tagsCard)

        notesCard = QFrame(self._widget)
        notes = QFormLayout(notesCard)
        notes.setLabelAlignment(Qt.AlignRight)
        notes.setHorizontalSpacing(5)
        notes.setVerticalSpacing(0)

        self.model_note_edit = TextEdit(notesCard)
        self.model_note_edit.setPlaceholderText(self.tr("Model notes"))
        self.model_note_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # self.model_note_edit.setMinimumHeight(30)

        notes.addRow(CaptionLabel(self.tr("Notes"), self), self.model_note_edit)
        root.addWidget(notesCard)

        root.addStretch(1)

    def _pick_file(self):
        path = call_path_dialog(self, self.tr("Select the model folder path"), "directory")

        if path:
            self.train_path_edit.setText(path)
            self.check_path()

    def add_tag(self, tag):
        if self.tag_group.has_tag(tag):
            MessageManager.send_info_message(self.tr("{tag} already exists!").format(tag=tag))
            return

        self.tag_group.add_tag(tag)

    def check_path(self):
        _path = self.train_path_edit.text()
        path = Path(_path)
        if not path.exists():
            MessageManager.send_message_box(self.tr("{path} does not exist!").format(path=_path))
            return
        if self.model_type_combox.currentText() == "NEP":
            model_file = path.joinpath("nep.txt")
            if not model_file.exists():
                MessageManager.send_message_box(
                    self.tr("No 'nep.txt' found in the specified path. Its presence is not strictly required, but please make sure you know what you are doing.")
                )

            data_file = path.joinpath("train.xyz")
            if not data_file.exists():
                MessageManager.send_message_box(
                    self.tr("No 'train.xyz' training data file found in the specified path. This file is required to compute training error metrics; please make sure you know what you are doing.")
                )
                # data_size=0
                energy = 0
                force = 0
                virial = 0
            else:
                metric_specs = (
                    (
                        self.tr("energy"),
                        path.joinpath("energy_train.out"),
                        2,
                        lambda array: get_rmse(array[:, 0], array[:, 1]) * 1000,
                        self.energy_spinBox,
                    ),
                    (
                        self.tr("force"),
                        path.joinpath("force_train.out"),
                        6,
                        lambda array: get_rmse(array[:, :3], array[:, 3:6]) * 1000,
                        self.force_spinBox,
                    ),
                    (
                        self.tr("virial"),
                        path.joinpath("virial_train.out"),
                        12,
                        lambda array: get_rmse(array[:, :6], array[:, 6:12]) * 1000,
                        self.virial_spinBox,
                    ),
                )
                for metric, output_path, min_columns, calculate, target in metric_specs:
                    try:
                        values = np.atleast_2d(read_nep_out_file(output_path))
                        if values.shape[1] < min_columns:
                            raise ValueError(
                                self.tr("expected at least {count} columns").format(
                                    count=min_columns,
                                )
                            )
                        result = float(calculate(values))
                        if not np.isfinite(result):
                            raise ValueError(self.tr("result is not finite"))
                    except Exception as exc:  # noqa: BLE001 - keep manual values editable
                        MessageManager.send_message_box(
                            self.tr(
                                "Cannot calculate {metric} RMSE from {file}: {error}. "
                                "The current manual value is kept."
                            ).format(
                                metric=metric,
                                file=output_path.name,
                                error=exc,
                            )
                        )
                    else:
                        target.setText(str(round(result, 2)))

                return

            self.force_spinBox.setText(str(round(force, 2)))
            self.energy_spinBox.setText(str(round(energy, 2)))
            self.virial_spinBox.setText(str(round(virial, 2)))

    def get_dict(self):
        path = Path(self.train_path_edit.text())
        data_file = path.joinpath("train.xyz")
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
            parent_id=self.parent_combox.currentData(),
        )


class AdvancedModelSearchDialog(MessageBoxBase):
    searchRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Advanced search - models"))
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
        tLay = QHBoxLayout(titleBar)
        tLay.setContentsMargins(0, 0, 0, 0)
        self.titleLabel = TitleLabel(self.tr("Advanced model search"), titleBar)
        # f = self.titleLabel.font(); f.setPointSize(f.pointSize() + 3); f.setBold(True)
        # self.titleLabel.setFont(f)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        tLay.addWidget(self.titleLabel)
        root.addWidget(titleBar)

        formCard = QFrame(self)
        form = QFormLayout(formCard)
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(3)
        form.setVerticalSpacing(3)

        self.projectIdsEdit = LineEdit(formCard)
        self.projectIdsEdit.setPlaceholderText(self.tr("e.g. 1 or 1,3,5"))
        self.includeDescendantsChk = CheckBox(self.tr("Include sub-projects"), formCard)
        self.includeDescendantsChk.setChecked(True)

        # Parent id
        self.parentIdEdit = LineEdit(formCard)
        self.parentIdEdit.setPlaceholderText(self.tr("None or integer"))
        self.parentIdEdit.setValidator(QIntValidator())

        self.nameContainsEdit = LineEdit(formCard)
        self.nameContainsEdit.setPlaceholderText(self.tr("contains in name"))
        self.notesContainsEdit = LineEdit(formCard)
        self.notesContainsEdit.setPlaceholderText(self.tr("contains in notes"))

        self.modelTypeCombo = ComboBox(formCard)
        self.modelTypeCombo.addItem(self.tr("<Any>"), userData=None)
        self.modelTypeCombo.addItem("NEP", userData="NEP")
        self.modelTypeCombo.addItem("DeepMD", userData="DeepMD")
        self.modelTypeCombo.addItem(self.tr("Other"), userData="Other")

        self.tagsAllEdit = LineEdit(formCard)
        self.tagsAllEdit.setPlaceholderText(self.tr("tag1, tag2 (AND)"))
        self.tagsAnyEdit = LineEdit(formCard)
        self.tagsAnyEdit.setPlaceholderText(self.tr("tag1, tag2 (OR)"))
        self.tagsNoneEdit = LineEdit(formCard)
        self.tagsNoneEdit.setPlaceholderText(self.tr("tag1, tag2 (NOT)"))

        self.orderAscChk = CheckBox(self.tr("Order by created_at ascending"), formCard)
        self.orderAscChk.setChecked(True)
        self.limitEdit = LineEdit(formCard)
        self.limitEdit.setPlaceholderText(self.tr("e.g. 100"))
        self.limitEdit.setValidator(QIntValidator(0, 10**9))
        self.offsetEdit = LineEdit(formCard)
        self.offsetEdit.setPlaceholderText(self.tr("e.g. 0"))
        self.offsetEdit.setValidator(QIntValidator(0, 10**9))

        form.addRow(CaptionLabel(self.tr("Project ID(s):"), self), self.projectIdsEdit)
        form.addRow(CaptionLabel("", self), self.includeDescendantsChk)
        form.addRow(CaptionLabel(self.tr("Parent ID:"), self), self.parentIdEdit)
        form.addRow(CaptionLabel(self.tr("Model type:"), self), self.modelTypeCombo)
        form.addRow(CaptionLabel(self.tr("Name contains:"), self), self.nameContainsEdit)
        form.addRow(CaptionLabel(self.tr("Notes contains:"), self), self.notesContainsEdit)
        form.addRow(CaptionLabel(self.tr("Tags (ALL):"), self), self.tagsAllEdit)
        form.addRow(CaptionLabel(self.tr("Tags (ANY):"), self), self.tagsAnyEdit)
        form.addRow(CaptionLabel(self.tr("Tags (NOT):"), self), self.tagsNoneEdit)
        form.addRow(CaptionLabel(self.tr("Order:"), self), self.orderAscChk)
        form.addRow(CaptionLabel(self.tr("Limit:"), self), self.limitEdit)
        form.addRow(CaptionLabel(self.tr("Offset:"), self), self.offsetEdit)

        root.addWidget(formCard)

        self.buttonLayout.removeWidget(self.yesButton)
        self.buttonLayout.removeWidget(self.cancelButton)
        self.yesButton.hide()
        self.cancelButton.hide()
        self.searchBtn = PrimaryPushButton(self.tr("Search"), self)
        self.resetBtn = PrimaryPushButton(self.tr("Reset"), self)
        self.closeBtn = PrimaryPushButton(self.tr("Close"), self)
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
        model_type = self.modelTypeCombo.currentData()

        parent_text = self.parentIdEdit.text().strip()
        parent_id_val = int(parent_text) if parent_text.isdigit() else None

        params: Dict[str, Any] = dict(
            project_id=(project_ids[0] if len(project_ids) == 1 else (project_ids if project_ids else None)),
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
        self.setWindowTitle(self.tr("Edit tag"))
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

        form.addRow(self.tr("Name"), self.nameEdit)
        form.addRow(self.tr("Color"), colorWidget)
        form.addRow(self.tr("Notes"), self.notesEdit)
        layout.addLayout(form)

        self.colorBtn.clicked.connect(self._choose_color)

    def _choose_color(self):
        color_dialog = ColorDialog(QColor(self.colorEdit.text()), self.tr("Edit tag color"), self)
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
        self._parent = parent
        self.tag_changed = False
        self.setWindowTitle(self.tr("Manage tags"))
        self.tag_service = tag_service
        self._tag_map: dict[str, int] = {}
        # self.resize(360, 240)

        self._layout = QVBoxLayout()
        self.new_tag_edit = LineEdit(self)
        self.new_tag_edit.setMinimumWidth(300)
        self.new_tag_edit.setPlaceholderText(self.tr("Enter the tag and press Enter"))
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
                self.tag_changed = True
                self.tag_service.update_tag(tag_id, name=new_name, color=color, notes=notes)
                obj.setText(new_name)
                obj.setBackgroundColor(color)
                obj.setToolTip(notes)
                if new_name != old_name:
                    self.tag_group.tags[new_name] = self.tag_group.tags.pop(old_name)
                    self._tag_map[new_name] = self._tag_map.pop(old_name)
            return True
        return super().eventFilter(obj, event)


@dataclass
class _TrainingOverlayResultData:
    """Minimal result-data container used to reuse the existing result canvas."""

    datasets: list[Any]
    select_index: set[int] = field(default_factory=set)
    reject_index: set[int] = field(default_factory=set)


class TrainingOverlayDialog(FramelessDialog):
    """Non-modal dialog showing PCA scatter plot with training/loaded/selected structures."""

    def __init__(self, parent=None, pca_data=None, canvas_type: str | None = None):
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))
        self.setWindowTitle(self.tr("Training overlay"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)
        max_btn = getattr(self.titleBar, "maxBtn", None)
        if max_btn is not None:
            max_btn.hide()
            max_btn.setEnabled(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._fade_in_started = False
        self._fade_in_anim: QPropertyAnimation | None = None
        self._pca_data = pca_data or {}
        self._canvas_type = str(canvas_type or Config.get("widget", "canvas_type", CanvasMode.PYQTGRAPH.value)).strip()
        self._canvas_fallback_warned = False
        self._legend_labels: list = []
        self._overlay_result_data: _TrainingOverlayResultData | None = None
        self._canvas = None
        self._setup_ui()
        if pca_data:
            self._render_from_pca_data()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_interaction_later()
        if self._fade_in_started:
            return
        self._fade_in_started = True
        try:
            if self.isMaximized():
                self.setWindowOpacity(1.0)
                return
            self.setWindowOpacity(0.0)
            anim = QPropertyAnimation(self, b"windowOpacity")
            anim.setDuration(180)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.start()
            self._fade_in_anim = anim
        except Exception:
            self.setWindowOpacity(1.0)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if self._fade_in_anim is not None and self._fade_in_anim.state() == QPropertyAnimation.State.Running:
                self._fade_in_anim.stop()
                self.setWindowOpacity(1.0)
            if self.isMaximized():
                self.showNormal()
            self._refresh_interaction_later()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_interaction_later()

    def _refresh_interaction_later(self):
        """Re-enable interactive handlers after geometry/state changes."""
        QTimer.singleShot(0, self._ensure_viewbox_interaction)

    @staticmethod
    def compute_pca_data(training_path, result_data, selected_indices):
        """Pre-compute PCA data before showing dialog. Returns dict with pca results."""
        import numpy as np
        from NepTrainKit.core.io.sampler import pca

        if result_data is None:
            return None

        try:
            dataset = getattr(result_data, "descriptor", None)
            if dataset is None:
                return None

            desc_now_reduced = np.asarray(dataset.now_data, dtype=np.float32)
            n_current = desc_now_reduced.shape[0]

            current_coords = desc_now_reduced
            raw_all = getattr(result_data, "_descriptor_raw_all", None)
            if raw_all is not None and raw_all.size > 0:
                try:
                    data_obj = getattr(dataset, "data", None)
                    if data_obj is not None:
                        now_indices = getattr(data_obj, "now_indices", None)
                        if now_indices is not None:
                            raw_now = np.asarray(raw_all[now_indices], dtype=np.float32)
                            if raw_now.ndim == 2 and raw_now.shape[1] > 2:
                                current_coords = raw_now
                except Exception:
                    pass

            training_coords = None
            n_training = 0
            if training_path:
                try:
                    from NepTrainKit.core.io.importers import import_structures
                    from NepTrainKit.core.io.base import aggregate_per_atom_to_structure
                    from NepTrainKit.paths import as_path
                    from NepTrainKit.core.utils import read_nep_out_file

                    t_path = as_path(training_path)
                    t_structs = import_structures(t_path)
                    if t_structs:
                        t_counts = np.array([len(s) for s in t_structs], dtype=int)
                        stem = t_path.stem
                        if stem == "train":
                            t_desc_path = t_path.with_name("descriptor.out")
                        else:
                            t_desc_path = t_path.with_name(f"descriptor_{stem}.out")
                        t_desc = read_nep_out_file(t_desc_path, dtype=np.float32, ndmin=2)
                        if t_desc.size == 0:
                            nep_calc = getattr(result_data, "nep_calc", None)
                            if nep_calc:
                                t_desc = nep_calc.descriptors(t_structs, mean=True)
                        if t_desc.size != 0:
                            if t_desc.shape[0] == int(np.sum(t_counts)):
                                t_desc = aggregate_per_atom_to_structure(t_desc, t_counts, map_func=np.mean, axis=0)
                            training_coords = np.asarray(t_desc, dtype=np.float32)
                            n_training = training_coords.shape[0]
                except Exception:
                    pass

            coords_list = []
            if training_coords is not None and training_coords.size > 0:
                coords_list.append(training_coords)
            if current_coords.size > 0:
                coords_list.append(current_coords)

            if len(coords_list) == 0:
                return None

            combined = np.vstack(coords_list)
            reduced = pca(combined.astype(np.float32), n_components=2)

            offset = 0
            training_pca = np.array([])
            if n_training > 0:
                training_pca = reduced[offset : offset + n_training]
                offset += n_training
            current_pca = reduced[offset : offset + current_coords.shape[0]]

            selected_pca = np.array([])
            selected_current_indices = np.array([], dtype=np.int32)
            if len(selected_indices) > 0 and current_pca.size > 0:
                valid_mask = np.array([0 <= i < len(current_pca) for i in selected_indices], dtype=bool)
                if valid_mask.any():
                    selected_current_indices = np.asarray(selected_indices, dtype=np.int32)[valid_mask]
                    selected_pca = current_pca[selected_current_indices]

            return {
                "training_pca": training_pca,
                "current_pca": current_pca,
                "selected_pca": selected_pca,
                "selected_current_indices": selected_current_indices,
                "n_training": n_training,
                "n_current": n_current,
            }
        except Exception:
            return None

    def _setup_ui(self):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout
        from PySide6.QtGui import QPixmap, QPainter, QColor

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.setMenuBar(self.titleBar)

        self._plot_hint_label = CaptionLabel("", self)
        self._plot_hint_label.setWordWrap(True)
        self._plot_hint_label.setStyleSheet("padding: 2px 8px;")
        root_layout.addWidget(self._plot_hint_label)

        self._canvas, fallback = create_result_canvas(self._canvas_type, self)
        self._canvas.tool_bar = None
        canvas_host = resolve_canvas_host_widget(self._canvas)
        canvas_host.setMinimumSize(560, 430)
        root_layout.addWidget(canvas_host, 1)
        if fallback:
            self._plot_hint_label.setText(
                self.tr("Current canvas backend is vispy, but vispy canvas failed to initialize; fallback to pyqtgraph.")
            )
            self._canvas_fallback_warned = True

        # Bottom control bar: legend + export buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(10, 4, 10, 6)
        bottom_layout.setSpacing(8)

        # Legend
        for label, color_rgb in [
            (self.tr("Training"), (160, 160, 160)),
            (self.tr("Loaded"), (30, 120, 215)),
            (self.tr("Selected"), (220, 30, 30)),
        ]:
            pixmap = QPixmap(14, 14)
            pixmap.fill(QColor(255, 255, 255, 0))
            painter = QPainter(pixmap)
            painter.setBrush(QColor(*color_rgb))
            painter.setPen(QColor(*color_rgb).darker(120))
            painter.drawEllipse(1, 1, 12, 12)
            painter.end()

            icon_lbl = QLabel()
            icon_lbl.setPixmap(pixmap)

            text_lbl = QLabel(f"{label}: -")
            text_lbl.setStyleSheet("color:#444;font-size:12px;")

            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(4)
            item_layout.addWidget(icon_lbl)
            item_layout.addWidget(text_lbl)

            bottom_layout.addWidget(item_widget)
            self._legend_labels.append(text_lbl)

        bottom_layout.addStretch()

        # Export buttons
        self._reset_view_btn = PrimaryPushButton(self.tr("Reset view"), self)
        self._reset_view_btn.clicked.connect(self._on_reset_view)
        bottom_layout.addWidget(self._reset_view_btn)

        self._export_image_btn = PrimaryPushButton(self.tr("Export image"), self)
        self._export_image_btn.clicked.connect(self._on_export_image)
        bottom_layout.addWidget(self._export_image_btn)

        self._export_data_btn = PrimaryPushButton(self.tr("Export data"), self)
        self._export_data_btn.clicked.connect(self._on_export_data)
        bottom_layout.addWidget(self._export_data_btn)

        root_layout.addLayout(bottom_layout)

        # Store RawAxis class for use in _render_from_pca_data (pg is available here)
        class _RawAxis(pg.AxisItem):
            def __init__(self, orientation, parent=None, font_size=11):
                super().__init__(orientation, parent)
                font = self.label.font()
                font.setPointSize(font_size)
                self.label.setFont(font)
                self.enableAutoSIPrefix(False)

            def tickStrings(self, values, scale, spacing):
                return [f"{v:.6g}" for v in values]

        self._RawAxis = _RawAxis
        self.resize(760, 620)

    def _render_from_pca_data(self):
        """Render scatter plot from pre-computed PCA data."""
        import numpy as np

        training_pca = self._reshape_points(self._pca_data.get("training_pca", np.array([])))
        current_pca = self._reshape_points(self._pca_data.get("current_pca", np.array([])))
        selected_current_indices = np.asarray(
            self._pca_data.get("selected_current_indices", np.array([], dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1)

        if len(self._legend_labels) >= 3:
            self._legend_labels[0].setText(self.tr("Training: {count}").format(count=training_pca.shape[0]))
            self._legend_labels[1].setText(self.tr("Loaded: {count}").format(count=current_pca.shape[0]))
            self._legend_labels[2].setText(self.tr("Selected: {count}").format(count=selected_current_indices.size))

        result_data, loaded_ids, selected_ids = self._build_overlay_result_data(
            training_pca,
            current_pca,
            selected_current_indices,
        )
        self._overlay_result_data = result_data
        self._canvas.set_nep_result_data(result_data)
        self._canvas.init_axes(1)
        self._canvas.plot_nep_result()
        apply_groups = getattr(self._canvas, "apply_overlay_groups", None)
        if apply_groups is not None:
            apply_groups(loaded_ids, selected_ids)

        # Apply custom axis font and ensure mouse interaction (pyqtgraph only)
        self._apply_custom_axes()
        self._ensure_viewbox_interaction()

    def _apply_custom_axes(self):
        """Apply custom axis font (smaller, no SI scaling) to all plot axes.
        Only affects pyqtgraph backend; vispy backend ignores this.
        """
        # Only apply for pyqtgraph: vispy stores ViewBoxWidget in axes_list (no setAxisItems)
        if self._canvas_type != CanvasMode.PYQTGRAPH.value:
            return
        axes_list = getattr(self._canvas, "axes_list", None)
        if not axes_list:
            return
        RawAxis = self._RawAxis
        for plot in axes_list:
            old_bottom = plot.getAxis("bottom")
            old_left = plot.getAxis("left")
            bottom_label = str(getattr(old_bottom, "labelText", "") or "").strip()
            left_label = str(getattr(old_left, "labelText", "") or "").strip()
            bottom_axis = RawAxis("bottom")
            left_axis = RawAxis("left")
            plot.setAxisItems({"bottom": bottom_axis, "left": left_axis})
            plot.setLabel("bottom", bottom_label)
            plot.setLabel("left", left_label)
            plot.getAxis("left").setWidth(70)
            plot.getAxis("bottom").setHeight(50)

    def _ensure_viewbox_interaction(self):
        """Confirm ViewBox allows drag and wheel zoom."""
        axes_list = getattr(self._canvas, "axes_list", None)
        if axes_list:
            for axes in axes_list:
                set_mouse_enabled = getattr(axes, "setMouseEnabled", None)
                if callable(set_mouse_enabled):
                    set_mouse_enabled(True, True)

                get_view_box = getattr(axes, "getViewBox", None)
                if callable(get_view_box):
                    view_box = get_view_box()
                    if view_box is not None:
                        view_box.setMouseEnabled(True, True)
                        view_box.setMenuEnabled(False)

                view = getattr(axes, "view", None)
                camera = getattr(view, "camera", None)
                if camera is not None and hasattr(camera, "interactive"):
                    camera.interactive = True

        view_box = getattr(self._canvas, "viewBox", None)
        if view_box is not None:
            view_box.setMouseEnabled(True, True)
            view_box.setMenuEnabled(False)

    def _on_export_image(self):
        """Export canvas as image (unified pyqtgraph / vispy)."""
        path = call_path_dialog(
            self,
            "Export Image",
            "file",
            default_filename="pca_scatter.png",
            file_filter="PNG files (*.png);;All files (*.*)",
        )
        if not path:
            return
        try:
            if self._canvas_fallback_warned:
                # vispy backend: use built-in save()
                self._canvas.save(path)
            else:
                # pyqtgraph backend: use QWidget.grab()
                host = resolve_canvas_host_widget(self._canvas)
                host.grab().save(path)
            MessageManager.send_info_message(f"Image exported to: {path}")
        except Exception:
            MessageManager.send_warning_message("Failed to export image.")

    def _on_reset_view(self):
        """Reset scatter viewport to fit current data."""
        if self._canvas is None:
            return
        try:
            auto_range = getattr(self._canvas, "auto_range", None)
            if callable(auto_range):
                auto_range()
                return

            axes_list = getattr(self._canvas, "axes_list", None)
            if axes_list:
                for axes in axes_list:
                    get_view_box = getattr(axes, "getViewBox", None)
                    if callable(get_view_box):
                        view_box = get_view_box()
                        if view_box is not None:
                            view_box.autoRange()
        except Exception:
            MessageManager.send_warning_message("Failed to reset view.")

    def _on_export_data(self):
        """Export PCA data as CSV (PC1, PC2, Type)."""
        path = call_path_dialog(
            self,
            "Export PCA Data",
            "file",
            default_filename="pca_data.csv",
            file_filter="CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            rows = []
            training_pca = self._pca_data.get("training_pca")
            if training_pca is not None and training_pca.size > 0:
                for row in training_pca:
                    rows.append((float(row[0]), float(row[1]), "Training"))
            current_pca = self._pca_data.get("current_pca")
            if current_pca is not None and current_pca.size > 0:
                for row in current_pca:
                    rows.append((float(row[0]), float(row[1]), "Loaded"))
            selected_pca = self._pca_data.get("selected_pca")
            if selected_pca is not None and selected_pca.size > 0:
                for row in selected_pca:
                    rows.append((float(row[0]), float(row[1]), "Selected"))
            with open(path, "w", encoding="utf-8") as f:
                f.write("PC1,PC2,Type\n")
                for pc1, pc2, t in rows:
                    f.write(f"{pc1:.8g},{pc2:.8g},{t}\n")
            MessageManager.send_info_message(f"Data exported to: {path}")
        except Exception:
            MessageManager.send_warning_message("Failed to export data.")

    @staticmethod
    def _reshape_points(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return arr.reshape(-1, 2)

    def _build_overlay_result_data(self, training_pca, current_pca, selected_current_indices):
        import numpy as np

        point_blocks = [block for block in (training_pca, current_pca) if block.size > 0]
        if point_blocks:
            points = np.vstack(point_blocks).astype(np.float32, copy=False)
        else:
            points = np.empty((0, 2), dtype=np.float32)

        synthetic_ids = np.arange(points.shape[0], dtype=np.int32)
        if points.size == 0:
            plot_data = np.empty((0, 2), dtype=np.float32)
        else:
            plot_data = np.column_stack([points[:, 1], points[:, 0]]).astype(np.float32, copy=False)

        dataset = NepPlotData(
            plot_data,
            index_list=synthetic_ids,
            title="training_overlay",
        )
        dataset.display_title = "Training Overlay"
        dataset.x_label = "PC1"
        dataset.y_label = "PC2"
        dataset.parity_mode = False
        dataset.show_rmse = False
        dataset.base_brush = Brushes.TrainingOverlay
        dataset.base_pen = Pens.TrainingOverlay

        loaded_offset = int(training_pca.shape[0])
        loaded_ids = synthetic_ids[loaded_offset:].astype(np.int32)
        selected_current_indices = selected_current_indices[
            (selected_current_indices >= 0) & (selected_current_indices < int(current_pca.shape[0]))
        ]
        selected_ids = (loaded_offset + selected_current_indices).astype(np.int32)
        return _TrainingOverlayResultData(datasets=[dataset]), loaded_ids, selected_ids
