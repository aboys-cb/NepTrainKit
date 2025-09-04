#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 22:45
# @Author  : 兵
# @email    : 1747193328@qq.com
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QVBoxLayout, QFrame, QGridLayout, QPushButton, QLineEdit,QWidget
from PySide6.QtCore import Signal, Qt, QUrl
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
    TitleLabel, HyperlinkLabel, RadioButton, LineEdit, FlowLayout, EditableComboBox, PrimaryDropDownPushButton,
    PrimaryPushButton, Flyout, InfoBarIcon, MessageBox,TextEdit
)
from qframelesswindow import FramelessDialog
import json
import os
from .button import TagPushButton, TagGroup

from NepTrainKit.core import MessageManager

from NepTrainKit import module_path

from NepTrainKit.utils import LoadingThread


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
class SparseMessageBox(MessageBoxBase):
    """用于最远点取样的弹窗 """

    def __init__(self, parent=None,tip=""):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self._frame = QFrame(self)
        self.frame_layout=QGridLayout(self._frame)
        self.frame_layout.setContentsMargins(0,0,0,0)
        self.frame_layout.setSpacing(0)
        self.intSpinBox = SpinBox(self)

        self.intSpinBox.setMaximum(9999999)
        self.intSpinBox.setMinimum(0)
        self.doubleSpinBox = DoubleSpinBox(self)
        self.doubleSpinBox.setDecimals(3)
        self.doubleSpinBox.setMinimum(0)
        self.doubleSpinBox.setMaximum(10)

        self.frame_layout.addWidget(CaptionLabel("Max num", self),0,0,1,1)

        self.frame_layout.addWidget(self.intSpinBox,0,1,1,2)
        self.frame_layout.addWidget(CaptionLabel("Min distance", self),1,0,1,1)

        self.frame_layout.addWidget(self.doubleSpinBox,1,1,1,2)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self._frame )

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')

        self.widget.setMinimumWidth(200)


class IndexSelectMessageBox(MessageBoxBase):
    """Dialog for selecting structures by index."""

    def __init__(self, parent=None, tip="Specify index or slice"):
        super().__init__(parent)
        self.titleLabel = CaptionLabel(tip, self)
        self.titleLabel.setWordWrap(True)
        self.indexEdit = QLineEdit(self)
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
        self.groupEdit = QLineEdit(self)

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
        self.viewLayout.addWidget(self.groupEdit)
        self.viewLayout.addWidget(self._frame)

        self.yesButton.setText('Ok')
        self.cancelButton.setText('Cancel')
        self.widget.setMinimumWidth(250)




class ProgressDialog(FramelessDialog):
    """进度条弹窗"""
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


        with open(os.path.join(module_path, "Config/ptable.json"), "r", encoding="utf-8") as f:
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
            "NEP Only",
            "DFT-D3 only",
            "NEP with DFT-D3",
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
        self._widget = QWidget(self)

        self._widget_layout = QGridLayout(self._widget)



        self.viewLayout.addWidget(self._widget)