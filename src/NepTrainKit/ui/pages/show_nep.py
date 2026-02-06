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

from NepTrainKit.ui.widgets import ConfigTypeSearchLineEdit, ArrowMessageBox, ExportFormatMessageBox
from NepTrainKit.core.io import (ResultData, load_result_data, matches_result_loader)

from NepTrainKit.core.structure import table_info, atomic_numbers
from NepTrainKit.core.types import Brushes, CanvasMode, SearchType
from NepTrainKit.paths import get_bundled_nep89_path
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
        # Cache for NEP result datasets keyed by NEP model path
        self._nep_result_cache: dict[Path, ResultData] = {}
        self._nep_cache_dir: Path | None = None
        self._initial_loading = False
        self._updating_nep_combo = False
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
        if hasattr(self._parent, "save_menu"):
            # Ensure we don't accumulate duplicates when the widget is shown repeatedly.
            for act in (
                self.export_all_action,
                self.export_selected_action,
                self.export_removed_action,
                self.export_current_action,
            ):
                try:
                    self._parent.save_menu.removeAction(act)  # pyright: ignore[attr-defined]
                except Exception:
                    pass
                self._parent.save_menu.addAction(act)  # pyright:ignore

        if hasattr(self._parent, "load_menu"):
            for act in (self.open_file_action, self.open_folder_action):
                try:
                    self._parent.load_menu.removeAction(act)  # pyright: ignore[attr-defined]
                except Exception:
                    pass
                self._parent.load_menu.addAction(act)  # pyright:ignore

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
        if hasattr(self._parent, "save_menu"):
            for act in (
                self.export_all_action,
                self.export_selected_action,
                self.export_removed_action,
                self.export_current_action,
            ):
                try:
                    self._parent.save_menu.removeAction(act)  # pyright:ignore
                except Exception:
                    pass

        if hasattr(self._parent, "load_menu"):
            for act in (self.open_file_action, self.open_folder_action):
                try:
                    self._parent.load_menu.removeAction(act)  # pyright:ignore
                except Exception:
                    pass

    def init_action(self):
        """Create reusable actions shared with the host application.

        Returns
        -------
        None
            Configures action callbacks for export operations.
        """
        self.open_file_action = Action(QIcon(':/images/src/images/open.svg'), "Open File…")
        self.open_file_action.triggered.connect(self.open_file)

        self.open_folder_action = Action(QIcon(':/images/src/images/open.svg'), "Open Folder…")
        self.open_folder_action.triggered.connect(self.open_folder)

        self.export_all_action = Action(QIcon(":/images/src/images/export1.svg"), "Export All…")
        self.export_all_action.triggered.connect(self.export_all_structures)

        self.export_selected_action = Action(QIcon(":/images/src/images/export1.svg"), "Export Selected (0)…")
        self.export_selected_action.triggered.connect(self.export_selected_structures)

        self.export_removed_action = Action(QIcon(":/images/src/images/export1.svg"), "Export Removed (0)…")
        self.export_removed_action.triggered.connect(self.export_removed_structures)

        self.export_current_action = Action(QIcon(":/images/src/images/export1.svg"), "Export Active (0)…")
        self.export_current_action.triggered.connect(self.export_active_structures)

        self._refresh_export_actions()

    def _is_busy(self) -> bool:
        """Return True when loading threads are running and exports should be disabled."""
        if getattr(self, "_initial_loading", False):
            return True
        try:
            if getattr(self, "load_thread", None) is not None and self.load_thread.isRunning():
                return True
        except Exception:
            pass
        return False

    def _dataset_ready(self) -> bool:
        """Return True when a dataset is loaded and usable for export."""
        data = getattr(self, "nep_result_data", None)
        return bool(data is not None and getattr(data, "load_flag", False))

    def _default_export_format(self) -> str:
        """Infer a sensible default export format from the current dataset path."""
        data_path = getattr(getattr(self, "nep_result_data", None), "data_xyz_path", None)
        try:
            candidate = data_path if isinstance(data_path, Path) else Path(str(data_path))
            if candidate.exists() and candidate.is_dir():
                return "deepmd/npy"
        except Exception:
            pass
        return "xyz"

    def _choose_export_format(self) -> str | None:
        """Ask the user to pick an export format; return None if cancelled."""
        remembered = Config.get("widget", "export_format", None)
        default_format = remembered or self._default_export_format()
        box = ExportFormatMessageBox(self, default_format=str(default_format))
        if not box.exec():
            return None
        fmt = box.selected_format()
        try:
            Config.set("widget", "export_format", fmt)
        except Exception:
            pass
        return fmt

    def _refresh_export_actions(self) -> None:
        """Refresh export action labels and enable states."""
        busy = self._is_busy()
        ready = self._dataset_ready()

        selected = 0
        removed = 0
        active = 0

        if ready:
            try:
                selected = len(self.nep_result_data.select_index)
            except Exception:
                selected = 0
            try:
                removed = int(self.nep_result_data.structure.remove_data.shape[0])
            except Exception:
                removed = 0
            try:
                active = int(self.nep_result_data.structure.now_data.shape[0])
            except Exception:
                active = 0

        self.export_selected_action.setText(f"Export Selected ({selected})…")
        self.export_removed_action.setText(f"Export Removed ({removed})…")
        self.export_current_action.setText(f"Export Active ({active})…")

        self.export_all_action.setEnabled(ready and not busy)
        self.export_selected_action.setEnabled(ready and (selected > 0) and not busy)
        self.export_removed_action.setEnabled(ready and (removed > 0) and not busy)
        self.export_current_action.setEnabled(ready and (active > 0) and not busy)

        # Keep open actions usable but avoid re-entrant loads while busy.
        self.open_file_action.setEnabled(not busy)
        self.open_folder_action.setEnabled(not busy)

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

        # Preserve current selection for both cached and reloaded paths.
        selected_indices: list[int] = []
        if getattr(self, "nep_result_data", None) is not None and hasattr(self.nep_result_data, "select_index"):
            try:
                selected_indices = list(self.nep_result_data.select_index)
            except Exception:
                selected_indices = []

        # Try use cached dataset first
        if hasattr(self, '_nep_result_cache') and hasattr(self, '_available_nep_files') and 0 <= index < len(self._available_nep_files):
            nep_file = self._available_nep_files[index]
            key = nep_file.resolve()
            cached = self._nep_result_cache.get(key)
            if cached is not None:
                # Reuse cached result without reloading
                self.stop_loading()
                self.nep_result_data = cached
                self.set_dataset()
                self._restore_selection(selected_indices)
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

        # Prefer the actual xyz path over the displayed directory link.
        xyz_path = getattr(self.nep_result_data, "data_xyz_path", None)
        if isinstance(xyz_path, Path):
            current_path = str(xyz_path)
        else:
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
        
        # Find all txt files containing 'nep' in filename under the current folder
        nep_files: list[Path] = []
        for txt_file in dir_path.glob("*.txt"):
            if "nep" in txt_file.stem.lower():
                nep_files.append(txt_file)
        
        # Sort: nep.txt first, then others alphabetically
        def sort_key(path):
            name = path.name.lower()
            if name == "nep.txt":
                return (0, name)
            return (1, name)
        
        nep_files.sort(key=sort_key)

        # Append bundled nep89 as an optional fallback choice (always last).
        try:
            nep89_path = get_bundled_nep89_path()
            if nep89_path.exists():
                already = False
                for existing in nep_files:
                    try:
                        if existing.samefile(nep89_path):
                            already = True
                            break
                    except Exception:
                        try:
                            if existing.resolve() == nep89_path.resolve():
                                already = True
                                break
                        except Exception:
                            continue
                if not already:
                    nep_files.append(nep89_path)
        except Exception:
            pass

        return nep_files

    def _update_nep_model_combo(self, directory):
        """Update the NEP model combo box with available model files.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing NEP model files.
        """
        self._updating_nep_combo = True
        try:
            self.nep_model_combo.blockSignals(True)
            self.nep_model_combo.clear()
            self._available_nep_files = self._detect_nep_files(directory)

            # Always show the combo; disable it when there's nothing to switch.
            # If no models were detected (e.g., invalid dir), fall back to bundled nep89.
            if not self._available_nep_files:
                try:
                    self._available_nep_files = [get_bundled_nep89_path()]
                except Exception:
                    self._available_nep_files = []

            # Add detected files to combo box (only text, no userData)
            for nep_file in self._available_nep_files:
                self.nep_model_combo.addItem(nep_file.name)

            self.nep_model_combo.show()
            # Enable combo only if multiple files found
            self.nep_model_combo.setEnabled(len(self._available_nep_files) > 1)
        finally:
            self.nep_model_combo.blockSignals(False)
            self._updating_nep_combo = False

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

        # Snapshot reusable structures before stopping threads (fast model switch).
        prefetched_structures = None
        try:
            if getattr(self.nep_result_data, "load_flag", False) and hasattr(self.nep_result_data, "structure"):
                prefetched_structures = list(self.nep_result_data.structure.all_data)
        except Exception:
            prefetched_structures = None

        # Stop any ongoing loading
        self.stop_loading()

        # Store current selection state
        selected_indices = list(self.nep_result_data.select_index) if hasattr(self.nep_result_data, 'select_index') else []
        reject_indices = list(getattr(self.nep_result_data, "reject_index", set()))
        
        tip = StateToolTip("Switching NEP model", 'Please wait...', self)
        tip.show()
        
        try:
            # Use the existing dataset class so the same loader type is preserved (NEP/DeepMD/etc).
            dataset_cls = type(self.nep_result_data)

            # Prefer the actual data path over UI labels.
            data_path = getattr(self.nep_result_data, "data_xyz_path", None)
            if isinstance(data_path, Path):
                data_path = str(data_path)
            if not data_path:
                data_path = xyz_path

            # Rebuild result data with the selected model but reuse structures to avoid re-reading.
            model_type = getattr(self.nep_result_data, "model_type", 0)
            try:
                self.nep_result_data = dataset_cls.from_path(
                    data_path,
                    model_type=model_type,
                    structures=prefetched_structures,
                    nep_txt_path=nep_file,
                )
            except TypeError:
                # Fallback for loaders that don't accept model_type.
                self.nep_result_data = dataset_cls.from_path(
                    data_path,
                    structures=prefetched_structures,
                    nep_txt_path=nep_file,
                )

            # Start loading in a new thread
            self.load_thread = QThread(self)
            tip.closedSignal.connect(self.stop_loading)
            self.nep_result_data.moveToThread(self.load_thread)
            self.load_thread.finished.connect(self.set_dataset)
            self.load_thread.finished.connect(lambda: tip.setState(True))
            self.load_thread.finished.connect(lambda: self._restore_reject(reject_indices))
            self.load_thread.finished.connect(lambda: self._restore_selection(selected_indices))

            self.nep_result_data.loadFinishedSignal.connect(self.load_thread.quit)
            self.load_thread.started.connect(self.nep_result_data.load)
            self.load_thread.start()
            self._refresh_export_actions()

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

    def _restore_reject(self, indices):
        """Restore previously rejected structure indices after reload."""
        if not indices or not self.nep_result_data:
            return
        try:
            if not hasattr(self.nep_result_data, "reject_index") or self.nep_result_data.reject_index is None:
                self.nep_result_data.reject_index = set()
            self.nep_result_data.reject_index.update(int(i) for i in indices)
        except Exception:
            return
        try:
            setter = getattr(self.graph_widget.canvas, "set_reject_highlight", None)
            if setter is not None:
                setter(list(indices), True)
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
        if hasattr(self.structure_toolbar, "rejectToggledSignal"):
            self.structure_toolbar.rejectToggledSignal.connect(self._toggle_reject_current)
        if hasattr(self.structure_toolbar, "dropRejectSignal"):
            self.structure_toolbar.dropRejectSignal.connect(self._drop_all_reject)

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
        self._refresh_export_actions()

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
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        for url in urls:
            try:
                path = url.toLocalFile()
            except Exception:
                continue
            if path and matches_result_loader(path):
                event.acceptProposedAction()
                return
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
        if not urls:
            return

        candidates: list[str] = []
        for url in urls:
            try:
                candidates.append(url.toLocalFile())
            except Exception:
                continue

        for path in candidates:
            if path and matches_result_loader(path):
                self.set_work_path(path)
                return

        MessageManager.send_info_message("unsupported file format")

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

    def open_folder(self):
        """Prompt the user to select a dataset folder (e.g., DeepMD/NPY directory)."""
        path = call_path_dialog(self, "Please choose the dataset folder", "directory")
        if path:
            self.set_work_path(path)

    def export_file(self):
        """Export the entire dataset (format chosen by the user)."""
        self.export_all_structures()

    def export_all_structures(self):
        """Export active + removed structures in either XYZ or deepmd/npy format."""
        if not self._dataset_ready():
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        path = call_path_dialog(self, "Choose a folder save location", "directory")
        if not path:
            return
        thread = LoadingThread(self, show_tip=True, title="Exporting data")
        if fmt == "deepmd/npy":
            thread.start_work(self.nep_result_data.export_model_npy, path)
        else:
            thread.start_work(self.nep_result_data.export_model_extxyz, path)

    def export_active_structures(self):
        """Export active (non-removed) structures in either XYZ or deepmd/npy format."""
        if not self._dataset_ready():
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        try:
            active = int(self.nep_result_data.structure.now_data.shape[0])
        except Exception:
            active = 0
        if active == 0:
            MessageManager.send_info_message("No active structures to export.")
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        thread = LoadingThread(self, show_tip=True, title="Exporting data")
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread.start_work(self.nep_result_data.export_active_npy, path)
            return

        path = call_path_dialog(
            self,
            "Choose a file save location",
            "file",
            file_filter="XYZ files (*.xyz)",
            default_filename="active_structures.xyz",
        )
        if not path:
            return
        thread.start_work(self.nep_result_data.export_active_xyz, path)

    def export_selected_structures(self):
        """Export the currently selected subset of structures.

        Returns
        -------
        None
            Starts a background job to write selected atoms to disk.
        """
        if not self._dataset_ready():
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        if len(self.nep_result_data.select_index) == 0:
            MessageManager.send_info_message("Please select some structures first!")
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread = LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self.nep_result_data.export_selected_npy, path)
        else:
            path = call_path_dialog(
                self,
                "Please choose the XYZ file",
                "file",
                file_filter="XYZ files (*.xyz)",
                default_filename="selected_structures.xyz",
            )
            if not path:
                return
            thread = LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self.nep_result_data.export_selected_xyz, path)

    def export_removed_structures(self):
        """Export removed structures in either XYZ or deepmd/npy format."""
        if not self._dataset_ready():
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        removed = int(self.nep_result_data.structure.remove_data.shape[0])
        if removed == 0:
            MessageManager.send_info_message("No removed structures to export.")
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread = LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self.nep_result_data.export_removed_npy, path)
        else:
            path = call_path_dialog(
                self,
                "Please choose the XYZ file",
                "file",
                file_filter="XYZ files (*.xyz)",
                default_filename="removed_structures.xyz",
            )
            if not path:
                return
            thread = LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self.nep_result_data.export_removed_xyz, path)

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
        if not hasattr(self.nep_result_data, "reject_index") or self.nep_result_data.reject_index is None:
            self.nep_result_data.reject_index = set()
        self.struct_index_spinbox.setMaximum(self.nep_result_data.num)
        self.graph_widget.set_dataset(self.nep_result_data)
        # Avoid duplicate signal connections for cached datasets
        if not getattr(self.nep_result_data, "_info_connected", False):
            self.nep_result_data.updateInfoSignal.connect(self.update_dataset_info)
            self.nep_result_data._info_connected = True
        self.nep_result_data.updateInfoSignal.emit()
        # Cache current dataset by its NEP model path for fast switching
        nep_path = getattr(self.nep_result_data, "nep_txt_path", None)
        if isinstance(nep_path, Path):
            try:
                self._nep_result_cache[nep_path.resolve()] = self.nep_result_data
            except Exception:
                pass
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
        self._refresh_export_actions()

        file_name = os.path.basename(path)
        show_path = path if os.path.isdir(path) else os.path.dirname(path)

        # Reset model cache when switching to a different working directory.
        try:
            resolved_dir = Path(show_path).resolve()
        except Exception:
            resolved_dir = None
        if resolved_dir is not None and resolved_dir != getattr(self, "_nep_cache_dir", None):
            self._nep_result_cache.clear()
            self._nep_cache_dir = resolved_dir
        
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
        model_dir = show_path
        try:
            nep_txt_path = getattr(self.nep_result_data, "nep_txt_path", None)
            if isinstance(nep_txt_path, Path):
                model_dir = str(nep_txt_path.parent)
        except Exception:
            model_dir = show_path
        self._update_nep_model_combo(model_dir)
        
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
        self._refresh_export_actions()

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
        self._refresh_export_actions()
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
        """Backward-compatible handler used by the structure toolbar export button."""
        self.export_current_structure()

    def _export_current_xyz(self, save_file_path: str, index: int) -> None:
        """Write a single structure to an XYZ file (runs in background thread)."""
        atoms = self.nep_result_data.get_atoms(index)
        with open(save_file_path, "w", encoding="utf-8") as handle:
            atoms.write(handle)
        MessageManager.send_info_message(f"File exported to: {save_file_path}")

    def export_current_structure(self):
        """Export the currently displayed structure in either XYZ or deepmd/npy."""
        if not self._dataset_ready():
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return
        index = int(self.struct_index_spinbox.value())
        fmt = self._choose_export_format()
        if fmt is None:
            return
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread = LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self.nep_result_data.export_current_npy, path, index)
            return

        path = call_path_dialog(
            self,
            "Choose a file save location",
            "file",
            file_filter="XYZ files (*.xyz)",
            default_filename=f"structure_{index}.xyz",
        )
        if not path:
            return
        thread = LoadingThread(self, show_tip=True, title="Exporting data")
        thread.start_work(self._export_current_xyz, path, index)

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

        # Sync reject toggle early so it updates even if rendering hits an exception.
        try:
            self._sync_reject_toolbar_state(int(current_index))
        except Exception:
            pass

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
        self._refresh_export_actions()

    def _active_reject_indices(self) -> set[int]:
        """Return rejected indices that are still active in the dataset."""
        if not self._dataset_ready():
            return set()
        reject = set(getattr(self.nep_result_data, "reject_index", set()))
        try:
            active = set(int(i) for i in self.nep_result_data.structure.group_array.now_data.tolist())
        except Exception:
            active = set()
        return reject & active

    def _sync_reject_toolbar_state(self, structure_index: int) -> None:
        """Update the structure toolbar reject toggle to match the current index."""
        if not self._dataset_ready():
            return
        reject = set(getattr(self.nep_result_data, "reject_index", set()))
        checked = int(structure_index) in reject
        try:
            if hasattr(self.structure_toolbar, "set_reject_checked"):
                self.structure_toolbar.set_reject_checked(checked)
        except Exception:
            pass

    def _toggle_reject_current(self, checked: bool) -> None:
        """Mark/unmark the current structure as rejected without changing navigation."""
        if not self._dataset_ready():
            return
        idx = int(self.struct_index_spinbox.value())
        if not hasattr(self.nep_result_data, "reject_index") or self.nep_result_data.reject_index is None:
            self.nep_result_data.reject_index = set()

        if checked:
            self.nep_result_data.reject_index.add(idx)
        else:
            try:
                self.nep_result_data.reject_index.discard(idx)
            except Exception:
                pass

        try:
            setter = getattr(self.graph_widget.canvas, "set_reject_highlight", None)
            if setter is not None:
                setter([idx], bool(checked))
        except Exception:
            pass
        self.update_dataset_info()

    def _drop_all_reject(self) -> None:
        """Delete all currently rejected active structures."""
        if not self._dataset_ready():
            MessageManager.send_info_message("NEP data has not been loaded yet!")
            return

        reject_active = self._active_reject_indices()
        if not reject_active:
            MessageManager.send_info_message("No bad structures tagged.")
            return

        n = len(reject_active)
        box = MessageBox(
            "Confirm",
            f"This will delete {n} structures marked as bad.\nDo you want to continue?",
            self,
        )
        box.exec_()
        if box.result() == 0:
            return

        try:
            self.nep_result_data.remove(list(reject_active))
        except Exception:
            logger.debug(traceback.format_exc())
            MessageManager.send_error_message("Failed to delete rejected structures.")
            return

        # Clear tags after delete (chosen default).
        try:
            self.nep_result_data.reject_index.clear()
        except Exception:
            self.nep_result_data.reject_index = set()

        # Full refresh (keeps UI/spinbox in sync).
        try:
            self.graph_widget.canvas.plot_nep_result()
        except Exception:
            pass
        self.update_dataset_info()

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
        rej = 0
        try:
            rej = len(self._active_reject_indices())
        except Exception:
            rej = 0
        info=f"Data: Orig: {self.nep_result_data.atoms_num_list.shape[0]} Now: {self.nep_result_data.structure.now_data.shape[0]} "\
        f"Rm: {self.nep_result_data.structure.remove_data.shape[0]} Sel: {len(self.nep_result_data.select_index)} Unsel: {self.nep_result_data.structure.now_data.shape[0]-len(self.nep_result_data.select_index)} Rej: {rej}"
        self.dataset_info_label.setText(info)
        self._refresh_export_actions()

