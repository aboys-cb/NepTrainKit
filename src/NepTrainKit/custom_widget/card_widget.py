#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/1/7 23:23
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QIcon, QDrag, QPixmap
from PySide6.QtWidgets import QWidget, QGridLayout

from qfluentwidgets import HeaderCardWidget, CheckBox, TransparentToolButton, ToolTipFilter, ToolTipPosition

from qfluentwidgets import FluentIcon as FIF

from NepTrainKit import utils
from NepTrainKit.core import MessageManager
from NepTrainKit.custom_widget import ProcessLabel
from ase.io import write as ase_write

class CheckableHeaderCardWidget(HeaderCardWidget):

    def __init__(self, parent=None):
        super(CheckableHeaderCardWidget, self).__init__(parent)
        self.state_checkbox=CheckBox()
        self.state_checkbox.setChecked(True)
        self.state_checkbox.stateChanged.connect(self.state_changed)
        self.state_checkbox.setToolTip("Enable or disable this card")


        self.headerLayout.insertWidget(0, self.state_checkbox, 0,Qt.AlignmentFlag.AlignLeft)


        self.headerLayout.setStretch(1, 3)
        self.headerLayout.setContentsMargins(10, 0, 3, 0)
        self.headerLayout.setSpacing(3)
        self.viewLayout.setContentsMargins(6, 0, 6, 0)
        self.headerLayout.setAlignment(self.headerLabel, Qt.AlignmentFlag.AlignLeft)
        self.check_state=True
    def state_changed(self, state):
        if state == 2:
            self.check_state = True
        else:
            self.check_state = False


class ShareCheckableHeaderCardWidget(CheckableHeaderCardWidget):
    exportSignal=Signal()
    def __init__(self, parent=None):
        super(ShareCheckableHeaderCardWidget, self).__init__(parent)
        self.export_button=TransparentToolButton(QIcon(":/images/src/images/export1.svg"),self)
        self.export_button.clicked.connect(self.exportSignal)
        self.export_button.setToolTip("Export data")
        self.export_button.installEventFilter(ToolTipFilter(self.export_button, 300, ToolTipPosition.TOP))

        self.close_button=TransparentToolButton(FIF.CLOSE,self)
        self.close_button.clicked.connect(self.close)
        self.close_button.setToolTip("Close card")
        self.close_button.installEventFilter(ToolTipFilter(self.close_button, 300, ToolTipPosition.TOP))


        self.headerLayout.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignRight)
        self.headerLayout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)

class MakeDataCardWidget(ShareCheckableHeaderCardWidget):
    """Base class for cards used in console workflow."""

    # group name used for card menu categorization
    group = None

    windowStateChangedSignal=Signal( )
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.window_state="expand"
        self.collapse_button=TransparentToolButton(QIcon(":/images/src/images/collapse.svg"),self)
        self.collapse_button.clicked.connect(self.collapse)
        self.collapse_button.setToolTip("Collapse or expand card")
        self.collapse_button.installEventFilter(ToolTipFilter(self.collapse_button, 300, ToolTipPosition.TOP))

        self.headerLayout.insertWidget(0, self.collapse_button, 0,Qt.AlignmentFlag.AlignLeft)
        self.windowStateChangedSignal.connect(self.update_window_state)

    def mouseMoveEvent(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        drag = QDrag(self)
        mime = QMimeData()
        drag.setMimeData(mime)

        # 显示拖拽时的控件预览
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())

        drag.exec(Qt.MoveAction)
    def collapse(self):

        if self.window_state == "collapse":
            self.window_state = "expand"
        else:

            self.window_state = "collapse"

        self.windowStateChangedSignal.emit( )
    def update_window_state(self):
        if self.window_state == "expand":
            self.collapse_button.setIcon(QIcon(":/images/src/images/collapse.svg"))
        else:
            self.collapse_button.setIcon(QIcon(":/images/src/images/expand.svg"))

    def from_dict(self, data):
        self.state_checkbox.setChecked(data['check_state'])

    def to_dict(self):

        return {
            'class': self.__class__.__name__,
            # 'name': self.card_name,
            'check_state': self.check_state,

        }
class MakeDataCard(MakeDataCardWidget):
    #通知下一个card执行
    separator=False
    card_name= "MakeDataCard"
    menu_icon=r":/images/src/images/logo.svg"
    runFinishedSignal=Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exportSignal.connect(self.export_data)
        self.dataset:list=None
        self.result_dataset=[]
        self.index=0
        # self.setFixedSize(400, 200)
        self.setting_widget = QWidget(self)
        self.viewLayout.setContentsMargins(3, 6, 3, 6)
        self.viewLayout.addWidget(self.setting_widget)
        self.settingLayout = QGridLayout(self.setting_widget)
        self.settingLayout.setContentsMargins(5, 0, 5,0)
        self.settingLayout.setSpacing(3)
        self.status_label = ProcessLabel(self)
        self.vBoxLayout.addWidget(self.status_label)
        self.windowStateChangedSignal.connect(self.show_setting)

    def show_setting(self ):
        if self.window_state == "expand":
            self.setting_widget.show( )

        else:
            self.setting_widget.hide( )

    def set_dataset(self,dataset):
        self.dataset = dataset
        self.result_dataset = []

        self.update_dataset_info()

    def write_result_dataset(self, file,**kwargs):
        ase_write(file,self.result_dataset,**kwargs)

    def export_data(self):

        if self.dataset is not None:

            path = utils.call_path_dialog(self, "Choose a file save location", "file",f"export_{self.getTitle().replace(' ', '_')}_structure.xyz")
            if not path:
                return
            thread=utils.LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.write_result_dataset, path)

    def process_structure(self, structure) :
        """
        params:
        structure:Atoms
        自定义对每个结构的处理 最后返回一个处理后的结构列表
        """
        raise NotImplementedError

    def closeEvent(self, event):

        if hasattr(self, "worker_thread"):

            if self.worker_thread.isRunning():

                self.worker_thread.terminate()
                self.runFinishedSignal.emit(self.index)

        self.deleteLater()
        super().closeEvent(event)

    def stop(self):
        if hasattr(self, "worker_thread"):
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.result_dataset = self.worker_thread.result_dataset
                self.update_dataset_info()
                del self.worker_thread

    def run(self):
        # 创建并启动线程

        if self.check_state:
            self.worker_thread = utils.DataProcessingThread(
                self.dataset,
                self.process_structure
            )
            self.status_label.set_colors(["#59745A" ])

            # 连接信号
            self.worker_thread.progressSignal.connect(self.update_progress)
            self.worker_thread.finishSignal.connect(self.on_processing_finished)
            self.worker_thread.errorSignal.connect(self.on_processing_error)

            self.worker_thread.start()
        else:
            self.result_dataset = self.dataset
            self.update_dataset_info()
            self.runFinishedSignal.emit(self.index)
        # self.worker_thread.wait()

    def update_progress(self, progress):
        self.status_label.setText(f"Processing {progress}%")
        self.status_label.set_progress(progress)

    def on_processing_finished(self):
        # self.status_label.setText("Processing finished")

        self.result_dataset = self.worker_thread.result_dataset
        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7" ])
        self.runFinishedSignal.emit(self.index)
        del self.worker_thread

    def on_processing_error(self, error):
        self.close_button.setEnabled(True)

        self.status_label.set_colors(["red" ])
        self.result_dataset = self.worker_thread.result_dataset
        del self.worker_thread
        self.update_dataset_info()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(f"Error occurred: {error}")



    def update_dataset_info(self ):
        text = f"Input structures: {len(self.dataset)} → Output: {len(self.result_dataset)}"
        self.status_label.setText(text)

class FilterDataCard(MakeDataCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Filter Data")

    def stop(self):

        if hasattr(self, "worker_thread"):

            if self.worker_thread.isRunning():
                self.worker_thread.terminate()


                self.result_dataset = []
                self.update_dataset_info()
                del self.worker_thread

    def update_progress(self, progress):
        self.status_label.setText(f"Processing {progress}%")
        self.status_label.set_progress(progress)

    def on_processing_finished(self):

        self.update_dataset_info()
        self.status_label.set_colors(["#a5d6a7" ])
        self.runFinishedSignal.emit(self.index)
        if hasattr(self, "worker_thread"):
            del self.worker_thread

    def on_processing_error(self, error):
        self.close_button.setEnabled(True)

        self.status_label.set_colors(["red" ])

        del self.worker_thread
        self.update_dataset_info()
        self.runFinishedSignal.emit(self.index)

        MessageManager.send_error_message(f"Error occurred: {error}")

    def update_dataset_info(self ):
        text = f"Input structures: {len(self.dataset)} → Selected: {len(self.result_dataset)}"
        self.status_label.setText(text)
