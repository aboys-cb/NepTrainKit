#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:38
# @email    : 1747193328@qq.com
import os.path
import sys
import traceback
from copy import copy
from pathlib import Path

from loguru import logger

import numpy as np
from PySide6.QtCore import QObject, QUrl, QTimer, Qt, QThread, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QSplitter, QFrame, QSizePolicy
from qfluentwidgets import FluentIcon, HyperlinkLabel, MessageBox, SpinBox, \
    ToolButton, ToolTipFilter, ToolTipPosition, TransparentToolButton, BodyLabel, \
    Action, StateToolTip, ComboBox, CaptionLabel, SimpleCardWidget

from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import BackgroundTask, run_in_thread
from NepTrainKit.config import Config

from NepTrainKit.core import MessageManager
from NepTrainKit.core.audit import analyze_structure_phase

from NepTrainKit.ui.widgets import (
    StructureFilterBar,
    ArrowMessageBox,
    ExportFormatMessageBox,
)
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.controllers import StructureFilterController
from NepTrainKit.core.io import (
    NepTrainResultData,
    ResultData,
    load_result_data,
    matches_result_loader,
)

from NepTrainKit.core.precision import get_export_significant_digits
from NepTrainKit.core.structure import write_structures_extxyz_atomic
from NepTrainKit.core.structure_inspection import inspect_structure
from NepTrainKit.core.types import Brushes, CanvasMode, SearchType
from NepTrainKit.paths import get_bundled_nep89_path
from NepTrainKit.ui.canvas.canvas_factory import (
    create_structure_plot,
    resolve_canvas_host_widget,
    supports_structure_arrows,
)
from NepTrainKit.ui.views import (
    NepResultPlotWidget,
    NepDisplayGraphicsToolBar,
    StructureInfoWidget,
    StructureToolBar,
)


_ARROW_DFT_FORCE = "__ntk_dft_force"
_ARROW_ML_FORCE = "__ntk_ml_force"
_ARROW_FORCE_ERROR = "__ntk_force_error"
_ARROW_DFT_MFORCE = "__ntk_dft_mforce"
_ARROW_ML_MFORCE = "__ntk_ml_mforce"
_ARROW_MFORCE_ERROR = "__ntk_mforce_error"

_ARROW_VECTOR_LABELS = {
    _ARROW_DFT_FORCE: "DFT force",
    _ARROW_ML_FORCE: "ML force",
    _ARROW_FORCE_ERROR: "Force error (ML - DFT)",
    _ARROW_DFT_MFORCE: "DFT mforce",
    _ARROW_ML_MFORCE: "ML mforce",
    _ARROW_MFORCE_ERROR: "MForce error (ML - DFT)",
}

_ARROW_VECTOR_SOURCES = (
    (("_force_vector_dataset", "force"), (_ARROW_DFT_FORCE, _ARROW_ML_FORCE, _ARROW_FORCE_ERROR)),
    (("_spin_force_vector_dataset", "_mforce_vector_dataset", "spin_force", "mforce"), (_ARROW_DFT_MFORCE, _ARROW_ML_MFORCE, _ARROW_MFORCE_ERROR)),
)

RESULT_DATA_FILE_FILTER = (
    "Supported data files (*.xyz *.extxyz *.traj *.dump *.lammpstrj *.lammpstraj "
    "OUTCAR OUTCAR* XDATCAR XDATCAR*);;"
    "Advanced / experimental structure files (*.out *.log *.data *.cfg input.data);;"
    "All files (*)"
)


def _set_loading_tip_content(tip, message: str) -> None:
    """Update a state tooltip and resize it for long runtime status text."""
    tip.setContent(message)
    content_label = getattr(tip, "contentLabel", None)
    title_label = getattr(tip, "titleLabel", None)
    close_button = getattr(tip, "closeButton", None)
    if content_label is None or title_label is None or close_button is None:
        return

    old_right = tip.x() + tip.width()
    parent = tip.parentWidget()
    parent_width = parent.width() if parent is not None else 0
    max_width = min(680, parent_width - 48) if parent_width > 328 else 680
    max_width = max(280, max_width)
    content_metrics = content_label.fontMetrics()
    title_width = title_label.fontMetrics().horizontalAdvance(title_label.text())
    text_width = content_metrics.horizontalAdvance(message)
    target_width = min(max_width, max(280, title_width + 56, text_width + 56))
    content_width = max(200, target_width - 56)
    should_wrap = text_width > content_width

    content_label.setWordWrap(should_wrap)
    content_label.setFixedWidth(content_width)
    content_height = (
        content_label.heightForWidth(content_width)
        if should_wrap
        else content_label.sizeHint().height()
    )
    content_height = max(content_metrics.height(), content_height)
    content_label.setFixedHeight(content_height)
    content_label.move(12, 27)

    target_height = max(51, 27 + content_height + 10)
    tip.setFixedSize(target_width, target_height)
    close_button.move(target_width - 24, 19)
    if tip.x() > 0:
        tip.move(max(0, old_right - target_width), tip.y())


class _LoadCompletionRelay(QObject):
    """Run result-load completion callbacks safely on the GUI thread."""

    def __init__(self, tip, result_data, callbacks, parent=None):
        super().__init__(parent)
        self._tip = tip
        self._result_data = result_data
        self._callbacks = tuple(callbacks)

    @Slot()
    def handle_finished(self) -> None:
        try:
            if bool(self._result_data.load_flag):
                self._tip.setState(True)
            else:
                # ``StateToolTip.setState(False)`` means "still running", not
                # "failed".  Close the spinner on failed/cancelled loads; the
                # loader has already published the actionable error message.
                self._tip.close()
            for callback in self._callbacks:
                callback()
        finally:
            self._callbacks = ()
            self._result_data = None
            self._tip = None
            self.deleteLater()

    @Slot(str)
    def update_content(self, message: str) -> None:
        """Translate late-bound worker status before showing it in the tooltip."""
        _set_loading_tip_content(self._tip, translate_runtime_message(message))



class ShowNepWidget(QWidget):
    """Visualise NEP result datasets and provide interactive structure tools.

    Parameters
    ----------
    parent : QWidget | None
        Optional owner widget that embeds this viewer.
    """
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
        self._search_job_id = 0
        self._completer_job_id = 0
        self._search_running = 0
        self._index_running = 0
        self._worker_threads: list[QThread] = []
        self._arrow_vector_lookup_cache = {}
        self._structure_mask_version_seen: int | None = None
        self._structure_canvas_fallback_warned = False
        self._pending_structure_index: int | None = None
        self._structure_update_scheduled = False
        self._structure_analysis_job_id = 0
        self._structure_analysis_cache: dict[tuple[int, int], tuple[object, object]] = {}
        self._phase_evidence_dataset_id: int | None = None
        self._phase_evidence_lookup: dict[int, object] = {}
        self.structure_filter_controller = StructureFilterController(self)
        self.init_action()
        self.init_ui()
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
        self.open_file_action = Action(QIcon(':/images/src/images/open.svg'), self.tr("Open File…"))
        self.open_file_action.triggered.connect(self.open_file)

        self.open_folder_action = Action(QIcon(':/images/src/images/open.svg'), self.tr("Open Folder…"))
        self.open_folder_action.triggered.connect(self.open_folder)

        self.export_all_action = Action(QIcon(":/images/src/images/export1.svg"), self.tr("Export All…"))
        self.export_all_action.triggered.connect(self.export_all_structures)

        self.export_selected_action = Action(
            QIcon(":/images/src/images/export1.svg"),
            self.tr("Export Selected ({selected})…").format(selected=0),
        )
        self.export_selected_action.triggered.connect(self.export_selected_structures)

        self.export_removed_action = Action(
            QIcon(":/images/src/images/export1.svg"),
            self.tr("Export Removed ({removed})…").format(removed=0),
        )
        self.export_removed_action.triggered.connect(self.export_removed_structures)

        self.export_current_action = Action(
            QIcon(":/images/src/images/export1.svg"),
            self.tr("Export Active ({active})…").format(active=0),
        )
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

        self.export_selected_action.setText(
            self.tr("Export Selected ({selected})…").format(selected=selected)
        )
        self.export_removed_action.setText(
            self.tr("Export Removed ({removed})…").format(removed=removed)
        )
        self.export_current_action.setText(
            self.tr("Export Active ({active})…").format(active=active)
        )

        self.export_all_action.setEnabled(ready and not busy)
        self.export_selected_action.setEnabled(ready and (selected > 0) and not busy)
        self.export_removed_action.setEnabled(ready and (removed > 0) and not busy)
        self.export_current_action.setEnabled(ready and (active > 0) and not busy)
        toolbar = getattr(self, "graph_toolbar", None)
        if toolbar is not None:
            toolbar.set_training_set_check_enabled(ready and not busy)

        # Keep open actions usable but avoid re-entrant loads while busy.
        self.open_file_action.setEnabled(not busy)
        self.open_folder_action.setEnabled(not busy)

    def _on_search_mode_changed(self, index):
        """Sync the search mode combo-box with the search line-edit."""
        if not hasattr(self, "search_lineEdit"):
            return
        try:
            idx = int(index)
        except Exception:
            idx = int(getattr(self.search_mode_combo, "currentIndex", lambda: 0)())

        combo = getattr(self, "search_mode_combo", None)
        search_type = combo.itemData(idx) if combo is not None and idx >= 0 else None
        search_type = search_type or SearchType.TAG
        use_filter = search_type in (SearchType.TAG, SearchType.FORMULA, SearchType.ELEMENTS, SearchType.EXPRESSION)
        if hasattr(self, "filter_edit_btn"):
            self.filter_edit_btn.setVisible(use_filter)
        self.search_lineEdit.set_search_type(search_type or SearchType.TAG)

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
        cache_outputs_override = getattr(
            self.nep_result_data,
            "_cache_outputs_override",
            None,
        )

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
        
        tip = StateToolTip(self.tr("Switching NEP model"), self.tr("Please wait..."), self)
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
            reload_kwargs = {
                "structures": prefetched_structures,
                "nep_txt_path": nep_file,
            }
            if (
                cache_outputs_override is not None
                and issubclass(dataset_cls, NepTrainResultData)
            ):
                reload_kwargs["cache_outputs"] = cache_outputs_override
            try:
                self.nep_result_data = dataset_cls.from_path(
                    data_path,
                    model_type=model_type,
                    **reload_kwargs,
                )
            except TypeError:
                # Fallback for loaders that don't accept model_type.
                self.nep_result_data = dataset_cls.from_path(
                    data_path,
                    **reload_kwargs,
                )
            self.nep_result_data.set_cache_outputs_override(
                cache_outputs_override
            )

            # Start loading in a new thread
            self.load_thread = QThread(self)
            tip.closedSignal.connect(self.stop_loading)
            self.nep_result_data.move_to_load_thread(self.load_thread)
            result_data = self.nep_result_data
            completion_relay = _LoadCompletionRelay(
                tip,
                result_data,
                (
                    self.set_dataset,
                    lambda: self._restore_reject(reject_indices),
                    lambda: self._restore_selection(selected_indices),
                ),
                self,
            )
            self.load_thread.finished.connect(completion_relay.handle_finished)

            self.nep_result_data.loadFinishedSignal.connect(self.load_thread.quit)
            self.nep_result_data.predictionStatusSignal.connect(
                completion_relay.update_content
            )
            self.load_thread.started.connect(self.nep_result_data.load)
            self.load_thread.start()
            self._refresh_export_actions()

        except Exception as error:
            logger.debug(traceback.format_exc())
            tip.setState(False)
            MessageManager.send_error_message(
                self.tr("Failed to switch NEP model") + f": {error}"
            )
    
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
        self.show_struct_widget, fallback = create_structure_plot(canvas_type, self.struct_widget)
        self.struct_widget_layout.addWidget(resolve_canvas_host_widget(self.show_struct_widget), 1, 0, 1, 1)
        if fallback and not self._structure_canvas_fallback_warned:
            MessageManager.send_warning_message(
                "Current canvas backend is vispy, but vispy structure canvas failed to initialize; fallback to pyqtgraph."
            )
            self._structure_canvas_fallback_warned = True
        self.structure_toolbar = StructureToolBar(self.struct_widget)
        self.structure_toolbar.showBondSignal.connect(self.show_struct_widget.set_show_bonds)
        self.structure_toolbar.orthoViewSignal.connect(self.show_struct_widget.set_projection)
        self.structure_toolbar.autoViewSignal.connect(self.show_struct_widget.set_auto_view)

        self.structure_toolbar.exportSignal.connect(self.export_single_struct)
        self.structure_toolbar.arrowSignal.connect(self.show_arrow_dialog)
        self._update_structure_arrow_availability()
        if hasattr(self.structure_toolbar, "rejectToggledSignal"):
            self.structure_toolbar.rejectToggledSignal.connect(self._toggle_reject_current)
        if hasattr(self.structure_toolbar, "dropRejectSignal"):
            self.structure_toolbar.dropRejectSignal.connect(self._drop_all_reject)

        self.struct_info_widget = StructureInfoWidget(self.struct_widget)
        self.struct_index_widget = SimpleCardWidget(self)
        self.struct_index_widget.setObjectName("structureNavigatorCard")
        self.struct_index_widget_layout = QHBoxLayout(self.struct_index_widget)
        self.struct_index_widget_layout.setContentsMargins(10, 5, 10, 5)
        self.struct_index_widget_layout.setSpacing(6)
        self.struct_index_label = CaptionLabel(self.struct_index_widget)
        self.struct_index_label.setText(self.tr("Original index"))

        self.struct_index_spinbox = SpinBox(self.struct_index_widget)
        self.struct_index_spinbox.setFixedWidth(150)
        self.struct_index_spinbox.setAlignment(Qt.AlignCenter)
        self.struct_index_spinbox.setAccessibleName(self.tr("Original structure index"))
        try:
            self.struct_index_spinbox.upButton.hide()
            self.struct_index_spinbox.downButton.hide()
        except AttributeError:
            pass
        self.struct_index_spinbox.setMinimum(0)
        self.struct_index_spinbox.setMaximum(0)
        self.struct_count_label = CaptionLabel(self.tr("/ 0 frames"), self.struct_index_widget)
        self.previous_structure_button = ToolButton(FluentIcon.LEFT_ARROW, self.struct_index_widget)
        self.previous_structure_button.setToolTip(self.tr("Previous structure"))
        self.previous_structure_button.setAccessibleName(self.tr("Previous structure"))
        self.previous_structure_button.clicked.connect(self.to_last_structure)
        self.next_structure_button = ToolButton(FluentIcon.RIGHT_ARROW, self.struct_index_widget)
        self.next_structure_button.setToolTip(self.tr("Next structure"))
        self.next_structure_button.setAccessibleName(self.tr("Next structure"))
        self.next_structure_button.clicked.connect(self.to_next_structure)
        self.play_timer=QTimer(self)
        self.play_timer.timeout.connect(self.play_show_structures)

        self.auto_switch_button = TransparentToolButton(FluentIcon.PLAY, self.struct_index_widget)
        self.auto_switch_button.setToolTip(self.tr("Play structures"))
        self.auto_switch_button.setAccessibleName(self.tr("Play structures"))
        self.auto_switch_button.clicked.connect(self.start_play)
        self.auto_switch_button.setCheckable(True)

        self.struct_index_widget_layout.addWidget(self.struct_index_label)
        self.struct_index_widget_layout.addStretch(1)
        self.struct_index_widget_layout.addWidget(self.previous_structure_button)
        self.struct_index_widget_layout.addWidget(self.struct_index_spinbox)
        self.struct_index_widget_layout.addWidget(self.struct_count_label)
        self.struct_index_widget_layout.addWidget(self.next_structure_button)
        self.struct_index_widget_layout.addWidget(self.auto_switch_button)
        self.struct_index_spinbox.valueChanged.connect(self.show_current_structure)
        self.struct_widget_layout.addWidget(self.structure_toolbar, 0, 0, 1, 1)
        self.struct_widget_layout.addWidget(self.struct_info_widget, 2, 0, 1, 1)
        self.struct_widget_layout.addWidget(self.struct_index_widget, 3, 0, 1, 1)

        self.struct_widget_layout.setRowStretch(1, 1)
        self.struct_widget_layout.setSpacing(6)
        self.struct_widget_layout.setContentsMargins(6, 6, 6, 6)

        self.plot_widget = QWidget(self)

        self.plot_widget_layout = QGridLayout(self.plot_widget)
        self.plot_widget_layout.setSpacing(1)
        self.plot_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_widget = NepResultPlotWidget(self  )
        self.graph_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.graph_widget.canvas.structureIndexChanged.connect(self.struct_index_spinbox.setValue)

        self.graph_toolbar = NepDisplayGraphicsToolBar(  self.plot_widget)
        self.graph_widget.set_tool_bar(self.graph_toolbar)
        self.graph_toolbar.trainingSetCheckSignal.connect(
            self.open_training_set_audit
        )
        frame = QFrame(self.plot_widget)
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.structure_filter_bar = StructureFilterBar(frame)
        self.structure_filter_bar.specChanged.connect(self._on_structure_filter_spec_changed)
        self.structure_filter_bar.previewRequested.connect(self._preview_structure_filter)
        self.structure_filter_bar.applyRequested.connect(self._apply_structure_filter_result)
        self.structure_filter_controller.stateChanged.connect(self._on_structure_filter_state_changed)
        self.structure_filter_controller.previewReady.connect(self._on_structure_filter_preview_ready)
        self.structure_filter_controller.previewFailed.connect(self._on_structure_filter_preview_failed)
        frame_layout.addWidget(self.structure_filter_bar, 1)
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
        self.nep_model_combo.setToolTip(self.tr("Switch NEP model"))
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
        for pane in (self.plot_widget, self.struct_widget):
            pane_policy = pane.sizePolicy()
            pane_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            pane.setSizePolicy(pane_policy)
        self.struct_widget.setMinimumWidth(460)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([690, 460])
        self.splitter.setStretchFactor(0, 60)
        self.splitter.setStretchFactor(1, 40)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        self._refresh_export_actions()

    def _update_structure_arrow_availability(self) -> None:
        """Enable or disable arrow controls based on canvas capabilities."""
        toolbar = getattr(self, "structure_toolbar", None)
        if toolbar is None or not hasattr(toolbar, "set_arrow_enabled"):
            return
        if supports_structure_arrows(getattr(self, "show_struct_widget", None)):
            toolbar.set_arrow_enabled(True)
            return
        toolbar.set_arrow_enabled(
            False,
            self.tr("Arrow overlay is available only for vispy structure canvas."),
        )

    def _get_completer_max_items(self) -> int:
        try:
            return int(Config.getint("widget", "completer_max_items", 50000))
        except Exception:
            return 50000

    def _update_search_status_label(self) -> None:
        label = getattr(self, "search_status_label", None)
        if label is None:
            return
        if getattr(self, "_search_running", 0) > 0:
            label.setText(self.tr("Searching…"))
            label.setVisible(True)
            return
        if getattr(self, "_index_running", 0) > 0:
            label.setText(self.tr("Indexing…"))
            label.setVisible(True)
            return
        label.setVisible(False)

    def _set_search_status(self, *, searching: bool | None = None, indexing: bool | None = None) -> None:
        if searching is not None:
            self._search_running = 1 if bool(searching) else 0
        if indexing is not None:
            self._index_running = 1 if bool(indexing) else 0
        self._update_search_status_label()

    def _begin_search(self) -> None:
        self._search_running = int(getattr(self, "_search_running", 0) or 0) + 1
        self._update_search_status_label()

    def _end_search(self) -> None:
        self._search_running = max(0, int(getattr(self, "_search_running", 0) or 0) - 1)
        self._update_search_status_label()

    def _begin_index(self) -> None:
        self._index_running = int(getattr(self, "_index_running", 0) or 0) + 1
        self._update_search_status_label()

    def _end_index(self) -> None:
        self._index_running = max(0, int(getattr(self, "_index_running", 0) or 0) - 1)
        self._update_search_status_label()

    def _track_worker_thread(self, thread: QThread) -> None:
        """Keep background threads alive until they finish."""
        self._worker_threads.append(thread)
        thread.finished.connect(self._on_worker_thread_finished)

    @Slot()
    def _on_worker_thread_finished(self) -> None:
        """Drop a completed worker reference from the GUI thread."""
        thread = self.sender()
        try:
            self._worker_threads.remove(thread)
        except ValueError:
            pass

    def _invalidate_background_jobs(self) -> None:
        """Invalidate in-flight background jobs and clear lightweight UI state."""
        self._search_job_id += 1
        self._completer_job_id += 1
        self._search_running = 0
        self._index_running = 0
        self._update_search_status_label()
        controller = getattr(self, "structure_filter_controller", None)
        if controller is not None:
            controller.invalidate_result()

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

        MessageManager.send_info_message(self.tr("unsupported file format"))

    def open_file(self):
        """Prompt the user to select an XYZ result file to display.

        Returns
        -------
        None
            Updates the working dataset when a file is chosen.
        """
        path = call_path_dialog(
            self,
            self.tr("Please choose the data file"),
            "select",
            file_filter=self.tr(
                "Supported data files (*.xyz *.extxyz *.traj *.dump *.lammpstrj *.lammpstraj "
                "OUTCAR OUTCAR* XDATCAR XDATCAR*);;"
                "Advanced / experimental structure files (*.out *.log *.data *.cfg input.data);;"
                "All files (*)"
            ),
        )
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
            MessageManager.send_info_message(self.tr("NEP data has not been loaded yet!"))
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        path = call_path_dialog(self, "Choose a folder save location", "directory")
        if not path:
            return
        thread = BackgroundTask(self, show_tip=True, title="Exporting data")
        if fmt == "deepmd/npy":
            thread.start_work(self.nep_result_data.export_model_npy, path)
        else:
            thread.start_work(self.nep_result_data.export_model_extxyz, path)

    def export_active_structures(self):
        """Export active (non-removed) structures in either XYZ or deepmd/npy format."""
        if not self._dataset_ready():
            MessageManager.send_info_message(self.tr("NEP data has not been loaded yet!"))
            return
        try:
            active = int(self.nep_result_data.structure.now_data.shape[0])
        except Exception:
            active = 0
        if active == 0:
            MessageManager.send_info_message(self.tr("No active structures to export."))
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        thread = BackgroundTask(self, show_tip=True, title="Exporting data")
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
            MessageManager.send_info_message(self.tr("NEP data has not been loaded yet!"))
            return
        if len(self.nep_result_data.select_index) == 0:
            MessageManager.send_info_message(self.tr("Please select some structures first!"))
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread = BackgroundTask(self, show_tip=True, title="Exporting data")
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
            thread = BackgroundTask(self, show_tip=True, title="Exporting data")
            thread.start_work(self.nep_result_data.export_selected_xyz, path)

    def export_removed_structures(self):
        """Export removed structures in either XYZ or deepmd/npy format."""
        if not self._dataset_ready():
            MessageManager.send_info_message(self.tr("NEP data has not been loaded yet!"))
            return
        removed = int(self.nep_result_data.structure.remove_data.shape[0])
        if removed == 0:
            MessageManager.send_info_message(self.tr("No removed structures to export."))
            return
        fmt = self._choose_export_format()
        if fmt is None:
            return
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread = BackgroundTask(self, show_tip=True, title="Exporting data")
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
            thread = BackgroundTask(self, show_tip=True, title="Exporting data")
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
            MessageManager.send_info_message(self.tr("unsupported file format"))
            return


        url=self.path_label.getUrl().toString()
        old_path=url.replace("file://","")
        if sys.platform == "win32":
            old_path=old_path[1:]
        else:
            pass
        if os.path.exists(old_path):
            box=MessageBox(
                self.tr("Confirm"),
                self.tr("A working directory already exists. Loading a new directory will erase the previous results.\nDo you want to load the new working path?"),
                self,
            )
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
        self._invalidate_background_jobs()
        if self.nep_result_data is None:
            return
        if not self.nep_result_data.load_flag :
            self.nep_result_data=None   # pyright:ignore
            return
        if not hasattr(self.nep_result_data, "reject_index") or self.nep_result_data.reject_index is None:
            self.nep_result_data.reject_index = set()
        structure_count = int(self.nep_result_data.structure.all_data.shape[0])
        self.struct_index_spinbox.setMaximum(max(0, structure_count - 1))
        self.struct_index_spinbox.lineEdit().setText(
            str(self.struct_index_spinbox.value())
        )
        self.struct_count_label.setText(
            self.tr("/ {count:,} frames").format(count=structure_count)
        )
        self._structure_analysis_job_id += 1
        self._structure_analysis_cache.clear()
        self._phase_evidence_dataset_id = None
        self._phase_evidence_lookup.clear()
        self.graph_widget.set_dataset(self.nep_result_data)
        self.structure_filter_controller.set_dataset(self.nep_result_data)
        self._update_structure_filter_suggestions()
        self.structure_filter_bar.set_selection_count(len(self.nep_result_data.select_index))
        self.structure_filter_bar.set_stale()
        self.graph_widget.canvas.clear_search_highlight()
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
        try:
            self._structure_mask_version_seen = int(self.nep_result_data.structure.data.version)
        except Exception:
            self._structure_mask_version_seen = None
        reset_camera_fit = getattr(self.show_struct_widget, "reset_camera_fit", None)
        if callable(reset_camera_fit):
            reset_camera_fit()
        self.struct_index_spinbox.valueChanged.emit(0)

    def _update_structure_filter_suggestions(self) -> None:
        """Reuse the loaded search caches in the composite filter editor."""
        data = getattr(self, "nep_result_data", None)
        if data is None:
            self.structure_filter_bar.set_suggestions({})
            return
        max_items = self._get_completer_max_items()
        suggestions = {}
        for search_type in (
            SearchType.TAG,
            SearchType.FORMULA,
            SearchType.ELEMENTS,
            SearchType.EXPRESSION,
        ):
            try:
                if data.has_completer_cache(search_type, max_items=max_items):
                    suggestions[search_type] = data.get_completer_cache(search_type, max_items=max_items)
            except Exception:
                logger.debug(traceback.format_exc())
        self.structure_filter_bar.set_suggestions(suggestions)

    def check_nep_result(
        self,
        path=None,
        *,
        structures=None,
        cache_outputs: bool | None = None,
        source_name: str | None = None,
    ):
        """Load NEP metadata and start the background loading thread.

        Parameters
        ----------
        path : str, optional
            Source file or directory containing NEP outputs. May be omitted
            when ``structures`` provides an in-memory dataset.
        structures : list[Structure], optional
            Pre-converted in-memory structures that bypass file parsing.
        cache_outputs : bool, optional
            Per-load cache policy. ``None`` follows the global setting.
        source_name : str, optional
            User-facing label for an in-memory dataset.

        Returns
        -------
        None
            Schedules dataset loading on a worker thread.
        """
        
        self._invalidate_background_jobs()
        # Set flag to prevent model change during initial load
        self._initial_loading = True
        self._refresh_export_actions()

        in_memory = structures is not None and path is None
        if path is None and structures is None:
            raise ValueError("path is required when structures are not provided")
        logical_path = "make_dataset.xyz" if path is None else os.fspath(path)
        file_name = source_name or os.path.basename(logical_path)
        show_path = (
            ""
            if in_memory
            else logical_path
            if os.path.isdir(logical_path)
            else os.path.dirname(logical_path)
        )

        # Reset model cache when switching to a different working directory.
        if in_memory:
            resolved_dir = None
        else:
            try:
                resolved_dir = Path(show_path).resolve()
            except Exception:
                resolved_dir = None
        if resolved_dir is not None and resolved_dir != getattr(self, "_nep_cache_dir", None):
            self._nep_result_cache.clear()
            self._nep_cache_dir = resolved_dir
        elif in_memory:
            self._nep_result_cache.clear()
            self._nep_cache_dir = None
        
        load_error = None
        try:
            if structures is None:
                self.nep_result_data = load_result_data(logical_path)  # type: ignore
            else:
                self.nep_result_data = NepTrainResultData.from_path(
                    logical_path,
                    structures=list(structures),
                    nep_txt_path=get_bundled_nep89_path(),
                    cache_outputs=cache_outputs,
                )
        except Exception as error:
            load_error = error
            logger.debug(traceback.format_exc())
            self.nep_result_data = None   # pyright:ignore
            MessageManager.send_error_message(
                f"Failed to load dataset: {error}. "
                "If official NEP .out files already exist, keep a complete set "
                "of energy, force, virial, and stress or mforce outputs in the "
                "dataset directory."
            )

        if self.nep_result_data is None:
            self._initial_loading = False
            if load_error is None:
                MessageManager.send_warning_message(
                    self.tr("unsupported file format")
                )
            return
        self.nep_result_data.set_cache_outputs_override(cache_outputs)

        if in_memory:
            self.path_label.setText(
                self.tr("Current dataset: {name}").format(name=file_name)
            )
            self.path_label.setUrl(QUrl())
        else:
            self.path_label.setText(
                self.tr("Current file: {file_name}").format(file_name=file_name)
            )
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
        tip = StateToolTip(self.tr("Loading"), self.tr("Please wait patiently..."), self)
        tip.show()
        tip.closedSignal.connect(self.stop_loading)
        self.nep_result_data.move_to_load_thread(self.load_thread)
        result_data = self.nep_result_data
        completion_relay = _LoadCompletionRelay(
            tip,
            result_data,
            (self.set_dataset, self._on_initial_load_complete),
            self,
        )
        self.load_thread.finished.connect(completion_relay.handle_finished)

        self.nep_result_data.loadFinishedSignal.connect(self.load_thread.quit)
        self.nep_result_data.predictionStatusSignal.connect(
            completion_relay.update_content
        )
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

        self._invalidate_background_jobs()
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

    def _on_search_type_changed(self, search_type: SearchType) -> None:
        """Apply a cached completer dictionary for the given search type without blocking UI."""
        if not hasattr(self, "search_lineEdit"):
            return
        data = getattr(self, "nep_result_data", None)
        if data is None:
            return
        max_items = self._get_completer_max_items()
        try:
            if hasattr(data, "has_completer_cache") and data.has_completer_cache(search_type, max_items=max_items):
                cache = data.get_completer_cache(search_type, max_items=max_items)
                self.search_lineEdit.setCompleterKeyWord(cache)
                self._index_running = 0
                self._update_search_status_label()
                return
        except Exception:
            pass

        # Cache not ready (e.g. legacy object). Build in a background thread.
        self._completer_job_id += 1
        job_id = self._completer_job_id
        dataset_id = id(data)
        self._begin_index()

        def _build_cache() -> bool:
            data.ensure_completer_cache(search_type, max_items=max_items)
            return True

        def _on_done(_result: object) -> None:
            try:
                if job_id != self._completer_job_id:
                    return
                current = getattr(self, "nep_result_data", None)
                if current is None or id(current) != dataset_id:
                    return
                try:
                    cache = current.get_completer_cache(self.search_lineEdit.search_type, max_items=max_items)
                    self.search_lineEdit.setCompleterKeyWord(cache)
                except Exception:
                    pass
            finally:
                self._end_index()

        def _on_err(msg: str) -> None:
            try:
                MessageManager.send_warning_message(
                    self.tr("Failed to build search completer cache: {msg}").format(msg=msg)
                )
            finally:
                self._end_index()

        thread = run_in_thread(self, _build_cache, on_finished=_on_done, on_error=_on_err)
        self._track_worker_thread(thread)

    def _run_async_search(self, config: str, search_type: SearchType, apply_result) -> None:
        """Run a structure search in background; apply_result runs on UI thread."""
        data = getattr(self, "nep_result_data", None)
        if data is None:
            return
        config = str(config or "").strip()
        if not config:
            # Avoid expensive "match everything" scans from accidental empty input.
            return
        self._search_job_id += 1
        job_id = self._search_job_id
        dataset_id = id(data)
        self._begin_search()

        def _compute():
            return data.search_config(config, search_type)

        def _on_done(indexes: object) -> None:
            try:
                if job_id != self._search_job_id:
                    return
                current = getattr(self, "nep_result_data", None)
                if current is None or id(current) != dataset_id:
                    return
                apply_result(list(indexes) if isinstance(indexes, (list, tuple, set)) else indexes)
            finally:
                self._end_search()

        def _on_err(msg: str) -> None:
            try:
                MessageManager.send_warning_message(
                    self.tr("Search failed: {msg}").format(msg=msg)
                )
            finally:
                self._end_search()

        thread = run_in_thread(self, _compute, on_finished=_on_done, on_error=_on_err)
        self._track_worker_thread(thread)

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
            self.auto_switch_button.setIcon(FluentIcon.PAUSE)
            self.auto_switch_button.setToolTip(self.tr("Pause structures"))
            self.auto_switch_button.setAccessibleName(self.tr("Pause structures"))
            self.play_timer.start(50)
        else:
            self.auto_switch_button.setIcon(FluentIcon.PLAY)
            self.auto_switch_button.setToolTip(self.tr("Play structures"))
            self.auto_switch_button.setAccessibleName(self.tr("Play structures"))
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
        atomic_float_digits = get_export_significant_digits()
        write_structures_extxyz_atomic(
            save_file_path,
            [atoms],
            atomic_float_digits=atomic_float_digits,
        )
        MessageManager.send_info_message(
            self.tr("File exported to: {save_file_path}").format(save_file_path=save_file_path)
        )

    def export_current_structure(self):
        """Export the currently displayed structure in either XYZ or deepmd/npy."""
        if not self._dataset_ready():
            MessageManager.send_info_message(self.tr("NEP data has not been loaded yet!"))
            return
        index = int(self.struct_index_spinbox.value())
        fmt = self._choose_export_format()
        if fmt is None:
            return
        if fmt == "deepmd/npy":
            path = call_path_dialog(self, "Choose a folder save location", "directory")
            if not path:
                return
            thread = BackgroundTask(self, show_tip=True, title="Exporting data")
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
        thread = BackgroundTask(self, show_tip=True, title="Exporting data")
        thread.start_work(self._export_current_xyz, path, index)

    def _get_arrow_source_dataset(self, candidate_names: tuple[str, ...]):
        data = getattr(self, "nep_result_data", None)
        if data is None:
            return None
        for name in candidate_names:
            try:
                dataset = getattr(data, name, None)
            except Exception:
                dataset = None
            if dataset is None:
                continue
            try:
                rows = np.asarray(dataset.all_data)
                if rows.ndim == 2 and int(getattr(dataset, "cols", 0) or 0) == 3 and rows.shape[1] >= 6:
                    return dataset
            except Exception:
                continue
        return None

    def _arrow_vector_lookup(self, dataset):
        cache = getattr(self, "_arrow_vector_lookup_cache", None)
        if cache is None:
            cache = {}
            self._arrow_vector_lookup_cache = cache
        rows = np.asarray(dataset.all_data)
        groups = np.asarray(dataset.group_array.all_data, dtype=np.int64).reshape(-1)
        key = (
            id(dataset),
            id(getattr(dataset.data, "all_data", None)),
            id(getattr(dataset.group_array, "all_data", None)),
            getattr(dataset.data, "version", 0),
            getattr(dataset.group_array, "version", 0),
            rows.shape,
            groups.shape,
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        lookup = {
            "rows": rows,
            "groups": groups,
            "sorted": bool(np.all(groups[:-1] <= groups[1:])) if groups.size > 1 else True,
        }
        dataset_id = id(dataset)
        for old_key in list(cache):
            if old_key[0] == dataset_id and old_key != key:
                cache.pop(old_key, None)
        cache[key] = lookup
        return lookup

    def _extract_arrow_vector_pair(self, dataset, structure_index: int, atom_count: int):
        try:
            lookup = self._arrow_vector_lookup(dataset)
            rows = lookup["rows"]
            groups = lookup["groups"]
        except Exception:
            return None
        if rows.ndim != 2 or rows.shape[0] != groups.size or rows.shape[1] < 6:
            return None
        if lookup.get("sorted", False):
            target = int(structure_index)
            left = int(np.searchsorted(groups, target, side="left"))
            right = int(np.searchsorted(groups, target, side="right"))
            selected = rows[left:right]
        else:
            selected = rows[groups == int(structure_index)]
        try:
            dft = np.asarray(selected[:, dataset.x_cols], dtype=np.float32).reshape(-1, 3)
            ml = np.asarray(selected[:, dataset.y_cols], dtype=np.float32).reshape(-1, 3)
        except Exception:
            return None
        if dft.shape != ml.shape or dft.ndim != 2 or dft.shape[1] != 3:
            return None
        return dft, ml

    @staticmethod
    def _expand_arrow_vector_pair(structure, dft: np.ndarray, ml: np.ndarray):
        atom_count = len(structure)
        if dft.shape == ml.shape == (atom_count, 3):
            return dft, ml
        props = getattr(structure, "atomic_properties", {}) or {}
        force_mag = props.get("force_mag")
        if force_mag is None:
            return None
        try:
            mag = np.asarray(force_mag, dtype=np.float32).reshape(atom_count, 3)
        except Exception:
            return None
        mask = ~np.all(mag == 0, axis=1)
        if int(np.sum(mask)) != int(dft.shape[0]):
            return None
        expanded_dft = np.zeros((atom_count, 3), dtype=np.float32)
        expanded_ml = np.zeros((atom_count, 3), dtype=np.float32)
        expanded_dft[mask] = dft
        expanded_ml[mask] = ml
        return expanded_dft, expanded_ml

    def _inject_ml_arrow_vectors(self, structure, structure_index: int) -> None:
        atom_count = len(structure)
        for candidate_names, prop_names in _ARROW_VECTOR_SOURCES:
            dataset = self._get_arrow_source_dataset(candidate_names)
            if dataset is None:
                continue
            pair = self._extract_arrow_vector_pair(dataset, int(structure_index), atom_count)
            if pair is None:
                continue
            dft, ml = pair
            expanded_pair = self._expand_arrow_vector_pair(structure, dft, ml)
            if expanded_pair is None:
                continue
            dft, ml = expanded_pair
            dft_prop, ml_prop, err_prop = prop_names
            structure.atomic_properties[dft_prop] = dft
            structure.atomic_properties[ml_prop] = ml
            structure.atomic_properties[err_prop] = ml - dft

    @staticmethod
    def _copy_structure_for_display(structure):
        display_structure = copy(structure)
        display_structure.atomic_properties = dict(getattr(structure, "atomic_properties", {}) or {})
        display_structure.properties = list(getattr(structure, "properties", []) or [])
        display_structure.additional_fields = dict(getattr(structure, "additional_fields", {}) or {})
        return display_structure

    @staticmethod
    def _arrow_display_names(props: list[str]) -> tuple[list[str], dict[str, str]]:
        labels: list[str] = []
        label_to_prop: dict[str, str] = {}
        for prop in props:
            label = _ARROW_VECTOR_LABELS.get(prop, prop)
            if label in label_to_prop:
                label = prop
            labels.append(label)
            label_to_prop[label] = prop
        return labels, label_to_prop

    def show_arrow_dialog(self):
        """Configure vector arrow overlays for the current structure.

        Returns
        -------
        None
            Updates arrow display based on user selections.
        """
        if not supports_structure_arrows(getattr(self, "show_struct_widget", None)):
            MessageManager.send_info_message(
                self.tr("Arrow overlay is unavailable for current structure canvas backend.")
            )
            return
        structure = getattr(self.show_struct_widget, "structure", None)
        if structure is None:
            return
        props = [
            name for name, arr in structure.atomic_properties.items()
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3
        ]
        if not props:
            MessageManager.send_info_message(self.tr("No vector data available"))
            return
        labels, label_to_prop = self._arrow_display_names(props)
        box = ArrowMessageBox(self, labels)
        cfg = getattr(self.show_struct_widget, "arrow_config", None)
        if cfg and cfg.get("prop_name") in props:
            box.propCombo.setCurrentText(_ARROW_VECTOR_LABELS.get(cfg["prop_name"], cfg["prop_name"]))
            box.scaleSpin.setValue(cfg["scale"])
            box.colorCombo.setCurrentText(cfg["cmap"])
            box.showCheck.setChecked(True)
        if not box.exec():
            return
        if box.showCheck.isChecked():
            label = box.propCombo.currentText()
            prop = label_to_prop.get(label, label)
            scale = box.scaleSpin.value()
            cmap = box.colorCombo.currentText()
            self.show_struct_widget.show_arrow(prop, scale, cmap, label)
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

        self.graph_widget.canvas.plot_current_point(current_index)
        self._pending_structure_index = int(current_index)
        if not self._structure_update_scheduled:
            self._structure_update_scheduled = True
            QTimer.singleShot(0, self._flush_current_structure_update)

    def _flush_current_structure_update(self):
        """Render the pending structure after the plot highlight has had a UI tick."""
        self._structure_update_scheduled = False
        current_index = self._pending_structure_index
        self._pending_structure_index = None
        if current_index is None:
            return

        try:
            atoms=self.nep_result_data.get_atoms(current_index)
        except Exception:
            logger.debug(traceback.format_exc())
            MessageManager.send_message_box("The index is invalid, perhaps the structure has been deleted")
            return

        display_atoms = self._copy_structure_for_display(atoms)
        self._inject_ml_arrow_vectors(display_atoms, int(current_index))
        self.show_struct_widget.show_structure(display_atoms)
        self.struct_info_widget.show_structure_info(atoms)
        if hasattr(self, "_structure_analysis_cache"):
            self._schedule_structure_analysis(atoms, int(current_index))
        self._refresh_export_actions()

    def set_phase_inventory(self, inventory, result_data=None) -> None:
        """Reuse completed Audit phase evidence in the per-frame inspector."""
        data = getattr(self, "nep_result_data", None)
        if data is None or (result_data is not None and result_data is not data):
            return
        lookup = {
            int(structure.source_index): structure
            for point in getattr(inventory, "composition_points", ())
            for structure in getattr(point, "structures", ())
        }
        self._phase_evidence_dataset_id = id(data)
        self._phase_evidence_lookup = lookup
        for key, (inspection, phase) in list(self._structure_analysis_cache.items()):
            if key[0] != id(data) or key[1] not in lookup:
                continue
            self._structure_analysis_cache[key] = (inspection, lookup[key[1]])
        try:
            current_index = int(self.struct_index_spinbox.value())
        except Exception:
            return
        phase = lookup.get(current_index)
        if phase is not None:
            self.struct_info_widget.show_phase_evidence(phase)

    def _phase_evidence_for_index(self, structure_index: int):
        data = getattr(self, "nep_result_data", None)
        if data is None or self._phase_evidence_dataset_id != id(data):
            return None
        return self._phase_evidence_lookup.get(int(structure_index))

    def _schedule_structure_analysis(self, atoms, structure_index: int) -> None:
        """Debounce expensive frame analysis and reject stale worker results."""
        data = getattr(self, "nep_result_data", None)
        if data is None:
            return
        dataset_id = id(data)
        cache_key = (dataset_id, int(structure_index))
        cached = self._structure_analysis_cache.get(cache_key)
        if cached is not None:
            self.struct_info_widget.show_analysis(*cached)
            return

        self._structure_analysis_job_id += 1
        job_id = self._structure_analysis_job_id
        QTimer.singleShot(
            100,
            lambda: self._start_structure_analysis(
                atoms,
                int(structure_index),
                dataset_id,
                job_id,
            ),
        )

    def _start_structure_analysis(
        self,
        atoms,
        structure_index: int,
        dataset_id: int,
        job_id: int,
    ) -> None:
        if job_id != self._structure_analysis_job_id:
            return
        data = getattr(self, "nep_result_data", None)
        if data is None or id(data) != dataset_id:
            return
        cached_phase = self._phase_evidence_for_index(structure_index)
        radius_coefficient = Config.getfloat("widget", "radius_coefficient", 0.7)

        def compute():
            inspection = inspect_structure(
                atoms,
                radius_coefficient=radius_coefficient,
            )
            phase = cached_phase
            if phase is None:
                try:
                    phase = analyze_structure_phase(
                        atoms,
                        source_index=structure_index,
                    )
                except Exception:
                    logger.debug(traceback.format_exc())
                    phase = None
            return inspection, phase

        def apply_result(payload) -> None:
            self._structure_analysis_cache[(dataset_id, structure_index)] = payload
            if job_id != self._structure_analysis_job_id:
                return
            current = getattr(self, "nep_result_data", None)
            if current is None or id(current) != dataset_id:
                return
            if int(self.struct_index_spinbox.value()) != structure_index:
                return
            self.struct_info_widget.show_analysis(*payload)

        def report_error(_message: str) -> None:
            if job_id != self._structure_analysis_job_id:
                return
            current = getattr(self, "nep_result_data", None)
            if current is None or id(current) != dataset_id:
                return
            if int(self.struct_index_spinbox.value()) == structure_index:
                self.struct_info_widget.show_analysis_unavailable()

        thread = run_in_thread(
            self,
            compute,
            on_finished=apply_result,
            on_error=report_error,
        )
        self._track_worker_thread(thread)

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
            MessageManager.send_info_message(self.tr("NEP data has not been loaded yet!"))
            return

        reject_active = self._active_reject_indices()
        if not reject_active:
            MessageManager.send_info_message(self.tr("No bad structures tagged."))
            return

        n = len(reject_active)
        box = MessageBox(
            self.tr("Confirm"),
            self.tr("This will delete {count} structures marked as bad.\nDo you want to continue?").format(count=n),
            self,
        )
        box.exec_()
        if box.result() == 0:
            return

        try:
            self.nep_result_data.remove(list(reject_active))
        except Exception:
            logger.debug(traceback.format_exc())
            MessageManager.send_error_message(self.tr("Failed to delete rejected structures."))
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

        if self.nep_result_data is None:
            return
        self._run_async_search(
            config,
            search_type,
            lambda indexes: self.graph_widget.canvas.set_search_highlight(indexes),
        )

    def _on_structure_filter_spec_changed(self, spec) -> None:
        """Invalidate the old preview whenever the typed query changes."""
        canvas = getattr(getattr(self, "graph_widget", None), "canvas", None)
        if canvas is not None:
            canvas.clear_search_highlight()
        if spec.is_empty():
            self.structure_filter_controller.clear()
            # A blank row is an editor draft, not a request to delete the
            # filter UI.  Only an actually empty condition list clears it.
            if not spec.conditions:
                self.structure_filter_bar.clear_state()
            else:
                self.structure_filter_bar.set_stale()
            return
        self.structure_filter_controller.set_spec(spec)
        self.structure_filter_bar.set_stale()

    def _preview_structure_filter(self) -> None:
        """Evaluate the current query asynchronously without changing selection."""
        if self.structure_filter_bar.spec.is_empty():
            return
        self.structure_filter_controller.preview()

    def _on_structure_filter_state_changed(self, state) -> None:
        self.structure_filter_bar.set_running(bool(state.running))
        if state.stale:
            self.structure_filter_bar.set_stale()

    def _on_structure_filter_preview_ready(self, result) -> None:
        canvas = getattr(getattr(self, "graph_widget", None), "canvas", None)
        if canvas is not None:
            canvas.set_search_highlight(result.indices)
        self.structure_filter_bar.set_result(
            len(result.indices),
            result.active_count,
            result.elapsed_ms,
        )

    def _on_structure_filter_preview_failed(self, error) -> None:
        self.structure_filter_bar.set_error(error)
        if not self.structure_filter_bar.editor_is_open:
            MessageManager.send_warning_message(
                self.tr("Filter failed: {message}").format(message=error.message)
            )

    def _apply_structure_filter_result(self, mode: str) -> None:
        """Apply cached matches as one undoable selection operation."""
        data = getattr(self, "nep_result_data", None)
        canvas = getattr(getattr(self, "graph_widget", None), "canvas", None)
        if data is None or canvas is None:
            return
        if mode == "clear":
            canvas.apply_selection_result((), "clear")
            self.structure_filter_bar.set_selection_count(len(data.select_index))
            return
        if not self.structure_filter_controller.result_is_current():
            self.structure_filter_bar.set_stale()
            MessageManager.send_info_message(self.tr("The filter result has expired. Preview it again before applying."))
            return
        result = self.structure_filter_controller.state.result
        if result is None:
            return
        canvas.apply_selection_result(result.indices, mode)
        self.structure_filter_bar.set_selection_count(len(data.select_index))

    def _on_filter_edit_clicked(self):
        """Open the composite filter editor (legacy action compatibility)."""
        self.structure_filter_bar.open_editor(None, add_if_empty=True)

    def _on_clear_all_selections(self):
        """Clear all structure selections."""
        data = getattr(self, "nep_result_data", None)
        if data is None:
            return
        try:
            all_indices = data.structure.group_array.now_data.tolist()
            if all_indices:
                self.graph_widget.canvas.select_index(all_indices, True)
        except Exception:
            pass

    def _on_tag_filter_search(self, filter_spec: dict, search_type: SearchType):
        """Highlight structures matching the tag filter spec."""
        if self.nep_result_data is None:
            return
        self._run_async_tags_search(
            filter_spec,
            search_type,
            lambda indexes: self.graph_widget.canvas.update_scatter_color(
                list(indexes) if isinstance(indexes, (list, tuple, set)) else [], Brushes.Show
            ),
        )

    def _on_tag_filter_check(self, filter_spec: dict, search_type: SearchType):
        """Select structures matching the tag filter spec."""
        if self.nep_result_data is None:
            return
        self._run_async_tags_search(
            filter_spec,
            search_type,
            lambda indexes: self.graph_widget.canvas.select_index(indexes, False),
        )

    def _on_tag_filter_uncheck(self, filter_spec: dict, search_type: SearchType):
        """Deselect structures matching the tag filter spec."""
        if self.nep_result_data is None:
            return
        self._run_async_tags_search(
            filter_spec,
            search_type,
            lambda indexes: self.graph_widget.canvas.select_index(indexes, True),
        )

    def _run_async_tags_search(self, filter_spec: dict, search_type: SearchType, apply_result) -> None:
        """Run a tag filter search in background; apply_result runs on UI thread."""
        data = getattr(self, "nep_result_data", None)
        if data is None:
            return
        self._search_job_id += 1
        job_id = self._search_job_id
        dataset_id = id(data)
        self._begin_search()

        def _compute():
            return data.search_config_tags(filter_spec, search_type)

        def _on_done(indexes: object) -> None:
            try:
                if job_id != self._search_job_id:
                    return
                current = getattr(self, "nep_result_data", None)
                if current is None or id(current) != dataset_id:
                    return
                apply_result(list(indexes) if isinstance(indexes, (list, tuple, set)) else indexes)
            finally:
                self._end_search()

        def _on_err(msg: str) -> None:
            try:
                MessageManager.send_warning_message(
                    self.tr("Search failed: {msg}").format(msg=msg)
                )
            finally:
                self._end_search()

        thread = run_in_thread(self, _compute, on_finished=_on_done, on_error=_on_err)
        self._track_worker_thread(thread)

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
        if self.nep_result_data is None:
            return
        if not str(config or "").strip():
            MessageManager.send_info_message(self.tr("Please enter a search query."))
            return
        self._run_async_search(
            config,
            search_type,
            lambda indexes: self.graph_widget.canvas.select_index(indexes, False),
        )

    def open_training_set_audit(self):
        """Open Training Set Audit for the currently loaded dataset."""
        if self.nep_result_data is None or not getattr(self.nep_result_data, "load_flag", False):
            MessageManager.send_info_message(
                self.tr("Please load a dataset before running Training Set Audit.")
            )
            return
        if hasattr(self._parent, "open_training_set_audit"):
            self._parent.open_training_set_audit(self.nep_result_data)
            return
        MessageManager.send_warning_message(
            self.tr("Training Set Audit page is not available.")
        )

    def open_training_set_distribution(self):
        """Open the unified audit page directly on numeric distributions."""
        if self.nep_result_data is None or not getattr(self.nep_result_data, "load_flag", False):
            MessageManager.send_info_message(
                self.tr("Please load a dataset before running Training Set Audit.")
            )
            return
        if hasattr(self._parent, "open_training_set_audit"):
            self._parent.open_training_set_audit(
                self.nep_result_data,
                initial_section="distribution",
            )
            return
        MessageManager.send_warning_message(
            self.tr("Training Set Audit page is not available.")
        )

    def run_distribution_analysis(self, request):
        """Run the existing distribution engine for the unified audit explorer."""
        if self.nep_result_data is None:
            return {}
        return self.graph_widget._run_distribution_analysis_task(  # noqa: SLF001
            self.nep_result_data,
            request,
        )

    def apply_distribution_selection(self, indices, mode) -> None:
        """Apply a distribution-bin selection through the Dataset Display canvas."""
        if self.nep_result_data is None:
            return
        self.graph_widget._apply_distribution_selection(  # noqa: SLF001
            self.nep_result_data,
            list(indices),
            str(mode),
        )

    def select_structure_indices(self, indices):
        """Replace the current selection with structure indices from Training Set Audit."""
        if self.nep_result_data is None:
            return
        current = list(getattr(self.nep_result_data, "select_index", set()))
        if current:
            self.graph_widget.canvas.select_index(current, True)
        clean = sorted({int(index) for index in indices if int(index) >= 0})
        if clean:
            self.graph_widget.canvas.select_index(clean, False)
        self._refresh_export_actions()

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
        if self.nep_result_data is None:
            return
        if not str(config or "").strip():
            MessageManager.send_info_message(self.tr("Please enter a search query."))
            return
        self._run_async_search(
            config,
            search_type,
            lambda indexes: self.graph_widget.canvas.select_index(indexes, True),
        )

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
        if hasattr(self, "structure_filter_bar"):
            self.structure_filter_bar.set_selection_count(len(self.nep_result_data.select_index))
        self._refresh_export_actions()
        # Active-mask changes invalidate cached filter indices.
        try:
            current_ver = int(self.nep_result_data.structure.data.version)
        except Exception:
            current_ver = None
        if current_ver is not None and current_ver != self._structure_mask_version_seen:
            self._structure_mask_version_seen = current_ver
            if hasattr(self, "structure_filter_controller"):
                self.structure_filter_controller.invalidate_result()
            if hasattr(self, "structure_filter_bar"):
                self.structure_filter_bar.set_stale()
            canvas = getattr(getattr(self, "graph_widget", None), "canvas", None)
            if canvas is not None:
                canvas.clear_search_highlight()
