#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:38
# @email    : 1747193328@qq.com
import os.path
import sys
import traceback
from pathlib import Path

from loguru import logger

import numpy as np
from PySide6.QtCore import QUrl, QTimer, Qt, Signal, QThread
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QSplitter, QFrame, QSizePolicy
from qfluentwidgets import HyperlinkLabel, MessageBox, SpinBox, \
    StrongBodyLabel, getFont, ToolTipFilter, ToolTipPosition, TransparentToolButton, BodyLabel, \
    Action, StateToolTip,ComboBox

from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import LoadingThread
from NepTrainKit.config import Config

from NepTrainKit.core import MessageManager

from NepTrainKit.ui.widgets import ConfigTypeSearchLineEdit, ArrowMessageBox
from NepTrainKit.core.io import (ResultData, load_result_data, matches_result_loader)

from NepTrainKit.core.structure import table_info, atomic_numbers
from NepTrainKit.core.types import Brushes, CanvasMode, SearchType
from NepTrainKit.ui.views import (
    NepResultPlotWidget,
    NepDisplayGraphicsToolBar,
    StructureInfoWidget,
    StructureToolBar,
)




class ShowNepWidget(QWidget):
    """Visualise NEP result datasets and provide interactive structure tools.

    Parameters
    ----------
    parent : QWidget | None
        Optional owner widget that embeds this viewer.
    """
    updateBondInfoSignal=Signal(str)

    def __init__(self,parent=None):
        """Initialise plotting widgets, actions, and viewer state.

        Parameters
        ----------
        parent : QWidget | None
            Optional owner widget that embeds this viewer.
        """
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("ShowNepWidget")
        self.setAcceptDrops(True)
        self.nep_result_data:ResultData
        self.nep_result_data=None  # pyright:ignore
        self.init_action()
        self.init_ui()
        self.calculate_bond_thread:LoadingThread
        self.load_thread:QThread
        self.first_show=True



    def showEvent(self, event):
        """Attach export actions and optionally auto-load the latest dataset.

        Parameters
        ----------
        event : QShowEvent
            Show event forwarded by Qt.

        Returns
        -------
        None
            May trigger automatic loading when configured.
        """
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.addAction(self.export_selected_action)   # pyright:ignore

        # Refresh structure viewer style (background/lattice colors) from settings.
        if hasattr(self, "show_struct_widget") and hasattr(self.show_struct_widget, "apply_style_from_config"):
            try:
                self.show_struct_widget.apply_style_from_config()
            except Exception:
                logger.debug(traceback.format_exc())

        auto_load_config = Config.getboolean("widget","auto_load",False)
        if not auto_load_config:
            return
        if   self.first_show:
            self.first_show=False
            path = list(Path("./").glob("*.xyz"))

            if path :
                self.set_work_path(path[0].absolute().as_posix())

    def hideEvent(self, event):
        """Remove exported actions from the parent menus when hidden.

        Parameters
        ----------
        event : QHideEvent
            Hide event forwarded by Qt.

        Returns
        -------
        None
            Cleans up menu actions owned by the parent window.
        """
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.removeAction(self.export_selected_action)   # pyright:ignore

    def init_action(self):
        """Create reusable actions shared with the host application.

        Returns
        -------
        None
            Configures action callbacks for export operations.
        """
        self.export_selected_action=Action(QIcon(":/images/src/images/export1.svg"),"Export Selected Structures")
        self.export_selected_action.triggered.connect(self.export_selected_structures)

    def _on_search_mode_changed(self, index):
        """Sync the search mode combo-box with the search line-edit."""
        try:
            idx = int(index)
        except Exception:
            idx = int(getattr(self.search_mode_combo, "currentIndex", lambda: 0)())

        mapping = {
            0: SearchType.TAG,
            1: SearchType.FORMULA,
            2: SearchType.ELEMENTS,
        }
        self.search_lineEdit.set_search_type(mapping.get(idx, SearchType.TAG))

    def _on_nep_model_changed(self, index):
        """Handle NEP model file switch in the combo box."""
        # Ignore during initial setup or programmatic updates
        if getattr(self, '_updating_nep_combo', False):
            return
        
        # Ignore if still loading initial data
        if getattr(self, '_initial_loading', False):
            return
        
        if not hasattr(self, 'nep_result_data') or self.nep_result_data is None:
            return
        if not hasattr(self, '_available_nep_files') or not self._available_nep_files:
            return
        if index < 0 or index >= len(self._available_nep_files):
            return
        
        selected_nep_file = self._available_nep_files[index]
        
        # Check if the selected file is already loaded
        if hasattr(self.nep_result_data, 'nep_txt_path'):
            try:
                if selected_nep_file.samefile(self.nep_result_data.nep_txt_path):
                    return  # Already using this model
            except Exception:
                pass
        
        current_path = self.path_label.getUrl().toLocalFile()
        if not current_path or not os.path.exists(current_path):
            return
        
        # Reload data with the selected NEP file
        self._reload_with_nep_file(current_path, selected_nep_file)

    def _detect_nep_files(self, directory):
        """Detect all txt files containing 'nep' in the directory.
        
        Parameters
        ----------
        directory : str or Path
            Directory to search for NEP model files.
            
        Returns
        -------
        list[Path]
            List of Path objects for detected NEP files, sorted with 'nep.txt' first.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        # Find all txt files containing 'nep' in filename
        nep_files = []
        for txt_file in dir_path.glob("*.txt"):
            if "nep" in txt_file.stem.lower():
                nep_files.append(txt_file)
        
        # Sort: nep.txt first, then others alphabetically
        def sort_key(path):
            if path.name == "nep.txt":
                return (0, path.name)
            return (1, path.name)
        
        nep_files.sort(key=sort_key)
        return nep_files

    def _update_nep_model_combo(self, directory):
        """Update the NEP model combo box with available model files.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing NEP model files.
        """
        self.nep_model_combo.clear()
        self._available_nep_files = self._detect_nep_files(directory)
        
        if not self._available_nep_files:
            self.nep_model_combo.setEnabled(False)
            self.nep_model_combo.hide()
            return
        
        # Add detected files to combo box (only text, no userData)
        for nep_file in self._available_nep_files:
            self.nep_model_combo.addItem(nep_file.name)
        
        # Show and enable combo if multiple files found, otherwise hide it
        if len(self._available_nep_files) > 1:
            self.nep_model_combo.setEnabled(True)
            self.nep_model_combo.show()
        else:
            self.nep_model_combo.setEnabled(False)
            self.nep_model_combo.hide()

    def _reload_with_nep_file(self, xyz_path, nep_file):
        """Reload the dataset using a specific NEP model file.
        
        Parameters
        ----------
        xyz_path : str
            Path to the XYZ data file.
        nep_file : Path
            Path to the NEP model file to use.
        """
        if self.nep_result_data is None:
            return
        
        # Stop any ongoing loading
        self.stop_loading()
        
        # Store current selection state
        selected_indices = list(self.nep_result_data.select_index) if hasattr(self.nep_result_data, 'select_index') else []
        
        tip = StateToolTip("Switching NEP model", 'Please wait...', self)
        tip.show()
        
        try:
            # Get the xyz file path
            xyz_file = self.nep_result_data.data_xyz_path if hasattr(self.nep_result_data, 'data_xyz_path') else xyz_path
            xyz_path_obj = Path(xyz_file)
            file_name = xyz_path_obj.stem
            
            # Import classes needed to create data object directly
            from NepTrainKit.core.io.nep import NepTrainResultData
            from NepTrainKit.core.utils import is_charge_model
            
            # Determine the correct output directory for the selected NEP file
            nep_stem = nep_file.stem
            
            if nep_stem == "nep":
                # Standard nep.txt, output to current directory
                output_dir = xyz_path_obj.parent
            else:
                # Other NEP files, use result_XXX directory
                output_dir = xyz_path_obj.parent / f"result_{nep_stem}"
                output_dir.mkdir(exist_ok=True)
                logger.info(f"Output files will be saved to: {output_dir.name}/")
            
            # Build output paths for the selected NEP file
            energy_out_path = output_dir / f"energy_{file_name}.out"
            force_out_path = output_dir / f"force_{file_name}.out"
            stress_out_path = output_dir / f"stress_{file_name}.out"
            virial_out_path = output_dir / f"virial_{file_name}.out"
            
            if file_name == "train":
                descriptor_path = output_dir / "descriptor.out"
            else:
                descriptor_path = output_dir / f"descriptor_{file_name}.out"
            
            charge_out_path = output_dir / f"charge_{file_name}.out"
            bec_out_path = output_dir / f"bec_{file_name}.out"
            
            # Create new result data object directly (avoid auto-detection in from_path)
            self.nep_result_data = NepTrainResultData(
                nep_file,
                xyz_file,
                energy_out_path,
                force_out_path,
                stress_out_path,
                virial_out_path,
                descriptor_path,
                charge_out_path,
                bec_out_path,
                is_charge_model(nep_file),
                None,  # spin_force_out_path
            )
            
            # Reinitialize NEP calculator with the selected file
            try:
                from NepTrainKit.core.calculator import NepCalculator
                from NepTrainKit.core.types import NepBackend
                backend_config = Config.get("nep", "backend", NepBackend.AUTO.value)
                batch_size = Config.getint("nep", "gpu_batch_size", 1000) or 1000
                
                self.nep_result_data.nep_calc = NepCalculator(
                    model_file=nep_file.as_posix(),
                    backend=NepBackend(backend_config),
                    batch_size=batch_size
                )
                logger.info(f"NEP calculator initialized with {nep_file.name}")
            except Exception:
                logger.debug(f"Failed to initialize calculator: {traceback.format_exc()}")
            
            # Start loading in a new thread
            self.load_thread = QThread(self)
            tip.closedSignal.connect(self.stop_loading)
            self.nep_result_data.moveToThread(self.load_thread)
            self.load_thread.finished.connect(self.set_dataset)
            self.load_thread.finished.connect(lambda: tip.setState(True))
            self.load_thread.finished.connect(lambda: self._restore_selection(selected_indices))
            
            self.nep_result_data.loadFinishedSignal.connect(self.load_thread.quit)
            self.load_thread.started.connect(self.nep_result_data.load)
            self.load_thread.start()
            
        except Exception:
            logger.debug(traceback.format_exc())
            tip.setState(False)
            MessageManager.send_error_message(f"Failed to switch NEP model")
    
    def _restore_selection(self, indices):
        """Restore previously selected structure indices after reload.
        
        Parameters
        ----------
        indices : list
            List of structure indices to restore.
        """
        if indices and self.nep_result_data:
            try:
                self.nep_result_data.select(indices)
            except Exception:
                pass


    def init_ui(self):
        """Construct canvases, toolbars, and datasets controls for the viewer.

        Returns
        -------
        None
            Instantiates child widgets and connects inter-widget signals.
        """
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("show_nep_gridLayout")
        self.gridLayout.setContentsMargins(0,0,0,0)

        self.struct_widget = QWidget(self)
        self.struct_widget_layout = QGridLayout(self.struct_widget)
        canvas_type = Config.get("widget", "canvas_type",  str(CanvasMode.PYQTGRAPH.value))
        if canvas_type == CanvasMode.PYQTGRAPH:
            from NepTrainKit.ui.canvas.pyqtgraph import StructurePlotWidget
            self.show_struct_widget = StructurePlotWidget(self.struct_widget)

            self.struct_widget_layout.addWidget(self.show_struct_widget, 1, 0, 1, 1)

        else:

            from NepTrainKit.ui.canvas.vispy import StructurePlotWidget
            self.show_struct_widget = StructurePlotWidget( parent=self.struct_widget)

            self.struct_widget_layout.addWidget(self.show_struct_widget.native, 1, 0, 1, 1)
        self.structure_toolbar = StructureToolBar(self.struct_widget)
        self.structure_toolbar.showBondSignal.connect(self.show_struct_widget.set_show_bonds)
        self.structure_toolbar.orthoViewSignal.connect(self.show_struct_widget.set_projection)
        self.structure_toolbar.autoViewSignal.connect(self.show_struct_widget.set_auto_view)

        self.structure_toolbar.exportSignal.connect(self.export_single_struct)
        self.structure_toolbar.arrowSignal.connect(self.show_arrow_dialog)

        self.struct_info_widget = StructureInfoWidget(self.struct_widget)
        self.struct_index_widget = QWidget(self)
        self.struct_index_widget_layout = QHBoxLayout(self.struct_index_widget)
        self.struct_index_label = BodyLabel(self.struct_index_widget)
        self.struct_index_label.setText("Current structure (original file index):")

        self.struct_index_spinbox = SpinBox(self.struct_index_widget)

        self.struct_index_spinbox.upButton.clicked.disconnect(self.struct_index_spinbox.stepUp)
        self.struct_index_spinbox.downButton.clicked.disconnect(self.struct_index_spinbox.stepDown)
        self.struct_index_spinbox.downButton.clicked.connect(self.to_last_structure)
        self.struct_index_spinbox.upButton.clicked.connect(self.to_next_structure)
        self.struct_index_spinbox.setMinimum(0)
        self.struct_index_spinbox.setMaximum(0)
        self.play_timer=QTimer(self)
        self.play_timer.timeout.connect(self.play_show_structures)

        self.auto_switch_button = TransparentToolButton(QIcon(':/images/src/images/play.svg') ,self.struct_index_widget)
        self.auto_switch_button.clicked.connect(self.start_play)
        self.auto_switch_button.setCheckable(True)


        self.struct_index_widget_layout.addWidget(self.struct_index_label)
        self.struct_index_widget_layout.addWidget(self.struct_index_spinbox)

        self.struct_index_widget_layout.addWidget(self.auto_switch_button)
        self.struct_index_spinbox.valueChanged.connect(self.show_current_structure)

        self.bond_label=StrongBodyLabel(self.struct_widget)
        self.bond_label.setFont(getFont(20, QFont.Weight.DemiBold))
        self.bond_label.setWordWrap(True)
        # self.bond_label.setStyleSheet("QLabel { background-color: #f3f3f3; color: black; padding: 5px; }")
        self.bond_label.setToolTip('The Tip is the minimum distance between atoms in the current structure, in Å.')

        self.bond_label.installEventFilter(ToolTipFilter(self.bond_label, 300, ToolTipPosition.TOP))


        self.struct_widget_layout.addWidget(self.structure_toolbar, 0, 0, 1, 1)

        self.force_label = StrongBodyLabel(self.struct_widget)
        self.force_label.setWordWrap(True)
        self.force_label.setToolTip("Net force of the current structure (sum of all atomic forces).")

        # self.struct_widget_layout.addWidget(self.export_single_struct_button, 1, 0, 1, 1, alignment=Qt.AlignRight)
        self.struct_widget_layout.addWidget(self.struct_info_widget, 2, 0, 1, 1)
        self.struct_widget_layout.addWidget(self.bond_label,3, 0, 1, 1)
        self.struct_widget_layout.addWidget(self.force_label,4, 0, 1, 1)

        self.struct_widget_layout.addWidget(self.struct_index_widget, 5, 0, 1, 1)

        self.struct_widget_layout.setRowStretch(0, 3)
        self.struct_widget_layout.setRowStretch(1, 1)
        self.struct_widget_layout.setRowStretch(2, 0)
        self.struct_widget_layout.setSpacing(1)
        self.struct_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = QWidget(self)

        self.plot_widget_layout = QGridLayout(self.plot_widget)
        self.plot_widget_layout.setSpacing(1)
        self.plot_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_widget = NepResultPlotWidget(self  )
        self.graph_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.graph_widget.canvas.structureIndexChanged.connect(self.struct_index_spinbox.setValue)

        self.graph_toolbar = NepDisplayGraphicsToolBar(  self.plot_widget)
        self.graph_widget.set_tool_bar(self.graph_toolbar)
        frame = QFrame(self.plot_widget)
        frame_layout = QHBoxLayout(frame)
        self.search_lineEdit = ConfigTypeSearchLineEdit(self.plot_widget)
        self.search_lineEdit.searchSignal.connect(self.search_config_type)
        self.search_lineEdit.checkSignal.connect(self.checked_config_type)
        self.search_lineEdit.uncheckSignal.connect(self.uncheck_config_type)
        self.search_lineEdit.typeChangeSignal.connect(lambda search_type:self.search_lineEdit.setCompleterKeyWord(self.nep_result_data.structure.get_all_config(search_type)) if self.nep_result_data is not None else None)


        self.search_mode_combo = ComboBox(frame)
        self.search_mode_combo.addItem("tag")
        self.search_mode_combo.addItem("formula")
        self.search_mode_combo.addItem("elements")
        self.search_mode_combo.setToolTip("switch search mode")
        self.search_mode_combo.installEventFilter(ToolTipFilter(self.search_mode_combo, 300, ToolTipPosition.TOP))
        self.search_mode_combo.currentIndexChanged.connect(self._on_search_mode_changed)
        if hasattr(self.search_mode_combo, "activated"):
            self.search_mode_combo.activated.connect(self._on_search_mode_changed)
        frame_layout.addWidget(self.search_mode_combo)

        frame_layout.addWidget(self.search_lineEdit)
        self.path_label = HyperlinkLabel(self.plot_widget)
        self.path_label.setFixedHeight(30)

        self.dataset_info_label = BodyLabel(self.plot_widget)
        self.dataset_info_label.setFixedHeight(30)

        # Create a container for path label and NEP model selector
        self.path_container = QWidget(self.plot_widget)
        self.path_container_layout = QHBoxLayout(self.path_container)
        self.path_container_layout.setContentsMargins(0, 0, 0, 0)
        self.path_container_layout.setSpacing(5)
        
        self.nep_model_combo = ComboBox(self.path_container)
        self.nep_model_combo.setToolTip("Switch NEP model")
        self.nep_model_combo.setFixedWidth(120)
        self.nep_model_combo.installEventFilter(ToolTipFilter(self.nep_model_combo, 300, ToolTipPosition.TOP))
        self.nep_model_combo.currentIndexChanged.connect(self._on_nep_model_changed)
        
        self.path_container_layout.addWidget(self.path_label)
        self.path_container_layout.addWidget(self.nep_model_combo)
        self.path_container_layout.addStretch()

        self.plot_widget_layout.addWidget(self.graph_toolbar, 0, 0, 1, 2)

        self.plot_widget_layout.addWidget(frame, 1, 0, 1, 2)
        self.plot_widget_layout.addWidget(self.graph_widget, 2, 0, 1, 2)
        self.plot_widget_layout.addWidget(self.path_container, 3, 0, 1, 1)
        self.plot_widget_layout.addWidget(self.dataset_info_label , 3, 1, 1, 1)
        self.plot_widget_layout.setContentsMargins(0,0,0,0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.addWidget(self.struct_widget)
        self.splitter.setSizes([400,200])
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 2)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        self.updateBondInfoSignal.connect(self.bond_label.setText)

    def dragEnterEvent(self, event):
        """Accept drag events carrying file URLs for NEP datasets.

        Parameters
        ----------
        event : QDragEnterEvent
            Drag event forwarded by Qt.

        Returns
        -------
        None
            Updates the event acceptance state depending on payload.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle dropped files by loading the first NEP-compatible path.

        Parameters
        ----------
        event : QDropEvent
            Drop event containing file URLs.

        Returns
        -------
        None
            Updates the working dataset path when a file is provided.
        """
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()

            self.set_work_path(file_path)

    def open_file(self):
        """Prompt the user to select an XYZ result file to display.

        Returns
        -------
        None
            Updates the working dataset when a file is chosen.
        """
        path = call_path_dialog(self,"Please choose the XYZ file","select",file_filter="XYZ files (*.xyz)")
        if path:
            self.set_work_path(path)

    def export_file(self):
        """Export the entire NEP dataset to a directory asynchronously.

        Returns
        -------
        None
            Starts a background job that writes structures to disk.
        """
        if self.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        path=call_path_dialog(self,"Choose a file save location","directory")
        if path:
            thread=LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.nep_result_data.export_model_xyz, path)

    def export_selected_structures(self):
        """Export the currently selected subset of structures.

        Returns
        -------
        None
            Starts a background job to write selected atoms to disk.
        """
        if self.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        if len(self.nep_result_data.select_index)==0:
            MessageManager.send_info_message("Please select some structures first!")
            return
        path = call_path_dialog(self,"Please choose the XYZ file","file",file_filter="XYZ files (*.xyz)",default_filename="selected_structures.xyz")
        if path:
            thread=LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.nep_result_data.export_selected_xyz, path)

    def set_work_path(self, path:str):
        """Validate and load a NEP dataset from the specified path.

        Parameters
        ----------
        path : str
            File system path to a NEP dataset or result file.

        Returns
        -------
        None
            Starts loading after confirming overwrites.
        """
        if not matches_result_loader(path):
            MessageManager.send_info_message("unsupported file format")
            return


        url=self.path_label.getUrl().toString()
        old_path=url.replace("file://","")
        if sys.platform == "win32":
            old_path=old_path[1:]
        else:
            pass
        if os.path.exists(old_path):
            box=MessageBox("Ask","A working directory already exists. Loading a new directory will erase the previous results.\nDo you want to load the new working path?",self)
            box.exec_()
            if box.result()==0:
                return

        self.check_nep_result(path)

    def set_dataset(self,*args):
        """Bind the loaded NEP dataset to visual components.

        Parameters
        ----------
        *args : tuple
            Unused arguments required by the signal signature.

        Returns
        -------
        None
            Updates widget limits and triggers initial rendering.
        """
        if self.nep_result_data is None:
            return
        if not self.nep_result_data.load_flag :
            self.nep_result_data=None   # pyright:ignore
            return
        self.struct_index_spinbox.setMaximum(self.nep_result_data.num)
        self.graph_widget.set_dataset(self.nep_result_data)
        self.nep_result_data.updateInfoSignal.connect(self.update_dataset_info)
        self.nep_result_data.updateInfoSignal.emit()
        self.search_lineEdit.typeChangeSignal.emit(self.search_lineEdit.search_type)
        self.struct_index_spinbox.valueChanged.emit(0)

    def check_nep_result(self, path):
        """Load NEP metadata and start the background loading thread.

        Parameters
        ----------
        path : str
            Source file or directory containing NEP outputs.

        Returns
        -------
        None
            Schedules dataset loading on a worker thread.
        """
        
        # Set flag to prevent model change during initial load
        self._initial_loading = True

        file_name = os.path.basename(path)
        show_path = path if os.path.isdir(path) else os.path.dirname(path)
        
        try:
            self.nep_result_data = load_result_data(path)  # type: ignore
        except Exception:
            logger.debug(traceback.format_exc())
            self.nep_result_data = None   # pyright:ignore

        if self.nep_result_data is None:
            self._initial_loading = False
            return

        self.path_label.setText(f"Current file: {file_name}")
        self.path_label.setUrl(QUrl.fromLocalFile(show_path))
        
        # Detect and populate NEP model files for combo box
        self._available_nep_files = self._detect_nep_files(show_path)
        self._update_nep_model_combo(show_path)
        
        # Set the current model in combo box without triggering change event
        if self._available_nep_files:
            self._updating_nep_combo = True
            current_nep = self.nep_result_data.nep_txt_path
            
            # Try to match current nep file
            matched = False
            for idx, nep_file in enumerate(self._available_nep_files):
                try:
                    if nep_file.samefile(current_nep):
                        self.nep_model_combo.setCurrentIndex(idx)
                        matched = True
                        break
                except Exception:
                    pass
            
            # If no match found, use the first one (default)
            if not matched:
                self.nep_model_combo.setCurrentIndex(0)
            
            self._updating_nep_combo = False
        
        # self.graph_widget.set_dataset(self.dataset)
        self.load_thread=QThread(self)
        tip = StateToolTip("Loading", 'Please wait patiently~~', self )
        tip.show()
        tip.closedSignal.connect(self.stop_loading)
        self.nep_result_data.moveToThread(self.load_thread)
        self.load_thread.finished.connect(self.set_dataset)
        self.load_thread.finished.connect(lambda :tip.setState(True))
        self.load_thread.finished.connect(self._on_initial_load_complete)

        self.nep_result_data.loadFinishedSignal.connect(self.load_thread.quit)
        self.load_thread.started.connect(self.nep_result_data.load)
        self.load_thread.start()

        # self.nep_result_data.load()
    
    def _on_initial_load_complete(self):
        """Mark initial loading as complete, enable model switching."""
        self._initial_loading = False

    def stop_loading(self):
        """Stop ongoing background loading threads safely.

        Returns
        -------
        None
            Attempts to cancel the worker thread and reset state.
        """

        # Request cooperative cancel for structure IO and NEP calc
        if self.nep_result_data is not None:
            try:
                # propagate to both structure loader and calculator
                if hasattr(self.nep_result_data, "request_cancel"):
                    self.nep_result_data.request_cancel()
                else:
                    self.nep_result_data.nep_calc.cancel()
            except Exception:
                pass
        # Politely stop the worker thread's event loop
        try:
            if self.load_thread is not None and self.load_thread.isRunning():
                self.load_thread.quit()
                self.load_thread.wait()
        except Exception:
            pass
        #     self.nep_result_data.nep_calc_thread.stop()

    def to_last_structure(self):
        """Select the previous structure in the current result set.

        Returns
        -------
        Optional[int]
            Index of the new structure, or ``None`` if navigation failed.
        """

        if self.nep_result_data is None:
            return None
        current_index = self.struct_index_spinbox.value()
        if self.nep_result_data.select_index:

            sort_index = np.sort(np.array(list(self.nep_result_data.select_index)) )
        else:
            sort_index = np.sort(self.nep_result_data.structure.group_array.now_data, axis=0)
        index = np.searchsorted(sort_index, current_index, side='left')

        self.struct_index_spinbox.setValue(int(sort_index[index-1 if index>0 else index]))

    # @timeit

    def to_next_structure(self):
        """Advance to the next structure respecting current selections.

        Returns
        -------
        Optional[int]
            Index of the new structure, or ``None`` if navigation failed.
        """
        if self.nep_result_data is None:
            return None
        current_index=self.struct_index_spinbox.value()
        if self.nep_result_data.select_index:
            sort_index = np.sort(np.array(list(self.nep_result_data.select_index)) )

        else:
            sort_index = np.sort(self.nep_result_data.structure.group_array.now_data, axis=0)
        index = np.searchsorted(sort_index, current_index, side='right')
        if index>=sort_index.shape[0]:
            return False
        self.struct_index_spinbox.setValue(int(sort_index[index]))

        if index==sort_index.shape[0]-1:
            return True
        else:
            return False

    def start_play(self):
        """Toggle automatic iteration of structures in the viewer.

        Returns
        -------
        None
            Starts or stops the play timer based on the toggle state.
        """
        if self.auto_switch_button.isChecked():
            self.auto_switch_button.setIcon(QIcon(':/images/src/images/pause.svg'))
            self.play_timer.start(50)
        else:
            self.auto_switch_button.setIcon(QIcon(':/images/src/images/play.svg'))
            self.play_timer.stop()

    def play_show_structures(self):
        """Advance playback and stop when the end of the dataset is reached.

        Returns
        -------
        None
            Stops autoplay when there are no further structures.
        """
        if self.to_next_structure():
            self.auto_switch_button.click()

    def export_single_struct(self):
        """Export the currently displayed structure to an XYZ file.

        Returns
        -------
        None
            Writes the selected structure when a path is chosen.
        """
        if self.nep_result_data is None:
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        index=self.struct_index_spinbox.value()
        atoms=self.nep_result_data.get_atoms(index)
        path=call_path_dialog(self,"Choose a file save location","file",
                                    file_filter="XYZ files (*.xyz)",
                                    default_filename=f"structure_{index}.xyz")
        if path is not None:
            with open(path,"w",encoding="utf-8") as f:
                atoms.write(f)

    def show_arrow_dialog(self):
        """Configure vector arrow overlays for the current structure.

        Returns
        -------
        None
            Updates arrow display based on user selections.
        """
        structure = getattr(self.show_struct_widget, "structure", None)
        if structure is None:
            return
        props = [
            name for name, arr in structure.atomic_properties.items()
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3
        ]
        if not props:
            MessageManager.send_info_message("No vector data available")
            return
        box = ArrowMessageBox(self, props)
        cfg = getattr(self.show_struct_widget, "arrow_config", None)
        if cfg and cfg.get("prop_name") in props:
            box.propCombo.setCurrentText(cfg["prop_name"])
            box.scaleSpin.setValue(cfg["scale"])
            box.colorCombo.setCurrentText(cfg["cmap"])
            box.showCheck.setChecked(True)
        if not box.exec():
            return
        if box.showCheck.isChecked():
            prop = box.propCombo.currentText()
            scale = box.scaleSpin.value()
            cmap = box.colorCombo.currentText()
            self.show_struct_widget.show_arrow(prop, scale, cmap)
        else:
            self.show_struct_widget.clear_arrow()

    
    # @timeit

    def show_current_structure(self,current_index):
        """Render the requested structure index and refresh auxiliary views.

        Parameters
        ----------
        current_index : int
            Index within the loaded dataset to display.

        Returns
        -------
        None
            Updates the 3D view, bond statistics, and info panel.
        """

        try:
            atoms=self.nep_result_data.get_atoms(current_index)
        except Exception:
            logger.debug(traceback.format_exc())
            MessageManager.send_message_box("The index is invalid, perhaps the structure has been deleted")
            return

        self.graph_widget.canvas.plot_current_point(current_index)

        self.show_struct_widget.show_structure(atoms)
        self.update_structure_bond_info(atoms)
        self.struct_info_widget.show_structure_info(atoms)

        # Update net force label for the current structure
        force_text = "Net force: N/A"
        try:
            if getattr(atoms, "has_forces", False):
                forces = np.asarray(atoms.forces, dtype=np.float64)
                if forces.size != 0:
                    net = forces.sum(axis=0)
                    norm = float(np.linalg.norm(net))
                    force_text = (
                        f"Net force: ({net[0]:.3e}, {net[1]:.3e}, {net[2]:.3e}) | "
                        f"|ΣF| = {norm:.3e}"
                    )
        except Exception:
            logger.debug(traceback.format_exc())
        self.force_label.setText(force_text)

    def update_structure_bond_info(self,atoms):
        """Schedule bond statistics computation for the displayed structure.

        Parameters
        ----------
        atoms : Atoms
            Structure currently shown in the viewer.

        Returns
        -------
        None
            Starts background computation of bond distances.
        """
        self.calculate_bond_thread=LoadingThread(self,show_tip=False )
        self.calculate_bond_thread.start_work(self.calculate_bond_info,atoms)

    def calculate_bond_info(self,atoms):
        """Calculate bond lengths and highlight potentially unreasonable distances.

        Parameters
        ----------
        atoms : Atoms
            Structure currently shown in the viewer.

        Returns
        -------
        None
            Emits updated bond text and warning messages when needed.
        """
        distance_info = atoms.get_mini_distance_info()
        bond_text = ""
        radius_coefficient_config = Config.getfloat("widget","radius_coefficient",0.7)
        unreasonable = False

        for elems,bond_length in distance_info.items():
            elem0_info = table_info[str(atomic_numbers[elems[0]])]
            elem1_info = table_info[str(atomic_numbers[elems[1]])]

            if (elem0_info["radii"] + elem1_info["radii"]) * radius_coefficient_config > bond_length*100:
                bond_text += f"{elems[0]}-{elems[1]}:"

                bond_text += f'<font color="red">{bond_length:.2f}</font> Å | '
                unreasonable = True
            # else:
        self.updateBondInfoSignal.emit( bond_text )
        if unreasonable:
            MessageManager.send_info_message("The distance between atoms is too small, and the structure may be unreasonable.")

    def search_config_type(self,config:str,search_type:SearchType):
        """Highlight structures matching the provided configuration query.

        Parameters
        ----------
        config : str
            Configuration pattern or tag to search.
        search_type : SearchType
            Search strategy to apply.

        Returns
        -------
        None
            Updates scatter colours to indicate matching structures.
        """

        indexes= self.nep_result_data.structure.search_config(config,search_type)

        self.graph_widget.canvas.update_scatter_color(indexes,Brushes.Show)

    def checked_config_type(self, config:str,search_type:SearchType):
        """Select structures matching the given configuration criteria.

        Parameters
        ----------
        config : str
            Configuration pattern or tag to search.
        search_type : SearchType
            Search strategy to apply.

        Returns
        -------
        None
            Marks matching indices as selected.
        """

        indexes = self.nep_result_data.structure.search_config(config,search_type)
        self.graph_widget.canvas.select_index(indexes,  False)

    def uncheck_config_type(self, config:str,search_type:SearchType):
        """Deselect structures matching the given configuration criteria.

        Parameters
        ----------
        config : str
            Configuration pattern or tag to search.
        search_type : SearchType
            Search strategy to apply.

        Returns
        -------
        None
            Clears selection for the matching indices.
        """

        indexes = self.nep_result_data.structure.search_config(config,search_type)
        self.graph_widget.canvas.select_index(indexes,True )

    def update_dataset_info(self ):
        """Update the dataset status label with current selection metrics.

        Returns
        -------
        None
            Renders aggregated counts in the footer label.
        """
        info=f"Data: Orig: {self.nep_result_data.atoms_num_list.shape[0]} Now: {self.nep_result_data.structure.now_data.shape[0]} "\
        f"Rm: {self.nep_result_data.structure.remove_data.shape[0]} Sel: {len(self.nep_result_data.select_index)} Unsel: {self.nep_result_data.structure.now_data.shape[0]-len(self.nep_result_data.select_index)}"
        self.dataset_info_label.setText(info)

