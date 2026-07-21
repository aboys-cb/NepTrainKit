#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Application entry point for the NepTrainKit desktop client."""

import os
import sys
if sys.platform == "darwin":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
import warnings

from PySide6.QtCore import Qt, QFile, QTimer
from PySide6.QtGui import QIcon, QFont, QPalette, QColor
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout
from qfluentwidgets import (
    setTheme,
    Theme,
    FluentWindow,
    NavigationItemPosition,
    SplitToolButton,
    RoundMenu,
    FluentIcon,
    InfoBadge,
    InfoBadgePosition,
)
from ase.io import write as ase_write
from loguru import logger

from NepTrainKit.core.audit import (
    build_magnetic_inventory,
    build_phase_inventory,
    build_training_set_audit,
)
from NepTrainKit.ui.pages import *
from NepTrainKit.ui.messages import MessageManager
from NepTrainKit.ui.threads import run_in_thread
from NepTrainKit.ui.update import AutoUpdateNotifier, get_pending_update_version
from NepTrainKit.utils import timeit
from NepTrainKit.ui.updater import unzip
from NepTrainKit.paths import as_path
from NepTrainKit.i18n import install_translator

warnings.filterwarnings("ignore")

APP_ICON_RESOURCE = ':/images/src/images/logo.png'


def _application_icon() -> QIcon:
    """Return the shared application icon."""
    return QIcon(APP_ICON_RESOURCE)


def _set_macos_dock_icon(app: QApplication, icon: QIcon) -> None:
    """Set the macOS Dock icon when running from Python instead of an app bundle."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSImage  # type: ignore

        icon_path = Path(tempfile.gettempdir()) / "NepTrainKit-dock-icon.png"
        pixmap = icon.pixmap(512, 512)
        if pixmap.isNull() or not pixmap.save(str(icon_path), "PNG"):
            return
        image = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
        if image is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:
        logger.debug("Failed to set macOS Dock icon:\n{}", traceback.format_exc())




class NepTrainKitMainWindow(FluentWindow):
    """Main application window providing navigation between NepTrainKit pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setMicaEffectEnabled(False)
        self._update_badge = None
        self.init_ui()

    @timeit
    def init_ui(self) -> None:
        """Initialise interface elements and navigation."""
        MessageManager._createInstance(self)
        self.init_menu()
        self.init_widget()
        self.init_navigation()
        self.initWindow()
        self.stackedWidget.currentChanged.connect(self._refresh_page_actions)
        self._refresh_page_actions()

    def init_menu(self) -> None:
        """Create the toolbar housing common open/save actions."""
        self.menu_widget = QWidget(self)
        self.menu_widget.setStyleSheet("ButtonView{background: rgb(240, 244, 249)}")
        self.menu_gridLayout = QGridLayout(self.menu_widget)
        self.menu_gridLayout.setContentsMargins(3, 0, 3, 0)
        self.menu_gridLayout.setSpacing(1)

        self.open_dir_button = SplitToolButton(QIcon(':/images/src/images/open.svg'), self.menu_widget)
        self.open_dir_button.setAccessibleName(self.tr("Open"))
        self.open_dir_button.clicked.connect(self.open_file_dialog)
        self.load_menu = RoundMenu(parent=self)
        self.open_dir_button.setFlyout(self.load_menu)

        self.save_dir_button = SplitToolButton(QIcon(':/images/src/images/save.svg'), self.menu_widget)
        self.save_dir_button.setAccessibleName(self.tr("Save"))
        self.save_dir_button.clicked.connect(self.export_file_dialog)

        self.save_menu = RoundMenu(parent=self)
        self.save_dir_button.setFlyout(self.save_menu)

        self.menu_gridLayout.addWidget(self.open_dir_button, 0, 0)
        self.menu_gridLayout.addWidget(self.save_dir_button, 0, 1)
        self.titleBar.hBoxLayout.insertWidget(
            2,
            self.menu_widget,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignCenter,
        )

    def init_navigation(self) -> None:
        """Register the navigation items and default pages."""
        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(200)
        self.navigationInterface.addSeparator()

        self.addSubInterface(
            self.show_nep_interface,
            QIcon(':/images/src/images/show_nep.svg'),
            self.tr('NEP Dataset Display'),
        )
        self.addSubInterface(
            self.training_set_audit_interface,
            QIcon(':/images/src/images/summary.svg'),
            self.tr('Training Set Check'),
        )
        self.addSubInterface(
            self.make_data_interface,
            QIcon(':/images/src/images/make.svg'),
            self.tr('Make Dataset'),
        )
        self.addSubInterface(
            self.data_manager_interface,
            QIcon(':/images/src/images/dataset.svg'),
            self.tr('Data Management'),
        )
        self.addSubInterface(
            self.setting_interface,
            FluentIcon.SETTING,
            self.tr('Settings'),
            NavigationItemPosition.BOTTOM,
        )
        self.navigationInterface.activateWindow()

    def init_widget(self) -> None:
        """Instantiate the page widgets used by the navigation interface."""
        self.show_nep_interface = ShowNepWidget(self)
        self.training_set_audit_interface = TrainingSetAuditWidget(self)
        self.make_data_interface = MakeDataWidget(self)
        self.setting_interface = SettingsWidget(self)
        self.data_manager_interface = DataManagerWidget(self)
        self._audited_result_data = None
        self._audited_result_signature = None
        self._audited_result = None
        self._training_set_phase_thread = None
        self._training_set_phase_result = None
        self._training_set_phase_token = None
        self._make_dataset_handoff_thread = None
        self._make_dataset_handoff_dir = None
        self._make_dataset_handoff_pending_dir = None
        self._connect_training_set_audit_signals()
        self.make_data_interface.finalOutputRequestedSignal.connect(
            self.open_make_dataset_output
        )

    def _connect_training_set_audit_signals(self) -> None:
        """Wire Training Set Audit page actions back into the main window."""
        self.training_set_audit_interface.selectStructuresSignal.connect(
            self.handle_training_set_audit_selection
        )
        self.training_set_audit_interface.rerunAuditSignal.connect(
            lambda: self.open_training_set_audit(force=True)
        )
        self.training_set_audit_interface.requestDatasetOpenSignal.connect(
            self.open_dataset_for_training_set_audit
        )
        self.training_set_audit_interface.requestStructureEvidenceSignal.connect(
            self._request_training_set_structure_evidence
        )

    def _request_training_set_structure_evidence(self) -> None:
        """Run optional structure and magnetic evidence for the active audit."""
        data = getattr(self, "_audited_result_data", None)
        result = getattr(self, "_audited_result", None)
        if data is None or result is None:
            return
        if getattr(self.show_nep_interface, "nep_result_data", None) is not data:
            return
        self._start_training_set_phase_analysis(data, result)

    def initWindow(self) -> None:
        """Configure top-level window parameters such as size and title."""
        self.resize(1200, 700)
        self.setWindowIcon(_application_icon())
        self.setWindowTitle('NepTrainKit')
        desktop = QApplication.screens()[0].availableGeometry()
        width, height = desktop.width(), desktop.height()
        self.move(width // 2 - self.width() // 2, height // 2 - self.height() // 2)

    def open_file_dialog(self) -> None:
        """Delegate to the current widget's ``open_file`` handler when available."""
        widget = self.stackedWidget.currentWidget()
        handler = getattr(widget, "open_file", None)
        if callable(handler):
            handler()

    def export_file_dialog(self) -> None:
        """Delegate to the current widget's ``export_file`` handler when available."""
        widget = self.stackedWidget.currentWidget()
        handler = getattr(widget, "export_file", None)
        if callable(handler):
            handler()

    def _refresh_page_actions(self, *_args) -> None:
        """Enable global actions only when the active page implements them."""
        widget = self.stackedWidget.currentWidget()
        can_open = callable(getattr(widget, "open_file", None))
        can_save = callable(getattr(widget, "export_file", None))
        self.open_dir_button.setEnabled(can_open)
        self.save_dir_button.setEnabled(can_save)
        self.open_dir_button.setToolTip(
            self.tr("Open data for this page")
            if can_open
            else self.tr("Open is not available on this page")
        )
        self.save_dir_button.setToolTip(
            self.tr("Save data from this page")
            if can_save
            else self.tr("Save is not available on this page")
        )

    def open_dataset_for_training_set_audit(self) -> None:
        """Switch to Dataset Display and open a file for a future audit."""
        self.switchTo(self.show_nep_interface)
        self.show_nep_interface.open_file()

    def open_make_dataset_output(self, structures: list) -> None:
        """Persist a temporary handoff and open it in Dataset Display."""
        if not structures:
            MessageManager.send_info_message(
                self.tr("The workflow output is empty.")
            )
            return
        thread = getattr(self, "_make_dataset_handoff_thread", None)
        if thread is not None and thread.isRunning():
            MessageManager.send_info_message(
                self.tr("Dataset handoff is already in progress.")
            )
            return

        handoff_dir = tempfile.TemporaryDirectory(
            prefix="neptrainkit-make-dataset-"
        )
        path = Path(handoff_dir.name) / "make_dataset.xyz"
        self._make_dataset_handoff_pending_dir = handoff_dir
        MessageManager.send_info_message(
            self.tr("Preparing the workflow output for display...")
        )

        def _write_handoff() -> str:
            ase_write(path, structures, format="extxyz")
            return str(path)

        def _open_handoff(result_path: str) -> None:
            if self._make_dataset_handoff_pending_dir is not handoff_dir:
                handoff_dir.cleanup()
                return
            previous_dir = self._make_dataset_handoff_dir
            self._make_dataset_handoff_dir = handoff_dir
            self._make_dataset_handoff_pending_dir = None
            self.switchTo(self.show_nep_interface)
            self.show_nep_interface.check_nep_result(result_path)
            if previous_dir is not None:
                previous_dir.cleanup()

        def _handoff_failed(message: str) -> None:
            if self._make_dataset_handoff_pending_dir is handoff_dir:
                self._make_dataset_handoff_pending_dir = None
            handoff_dir.cleanup()
            MessageManager.send_error_message(
                self.tr("Failed to prepare workflow output: {message}").format(
                    message=message
                )
            )

        self._make_dataset_handoff_thread = run_in_thread(
            self,
            _write_handoff,
            on_finished=_open_handoff,
            on_error=_handoff_failed,
        )

    def _training_set_audit_signature(self, result_data) -> tuple[object, ...]:
        """Return a cheap snapshot for safe reuse of an unchanged audit run."""
        versions: list[object | None] = []
        for attribute in ("structure", "energy", "_force_vector_dataset", "virial"):
            try:
                dataset = getattr(result_data, attribute)
                versions.append(getattr(dataset.data, "version", None))
            except Exception:
                versions.append(None)
        indices: tuple[int, ...] = ()
        try:
            raw_indices = getattr(result_data.structure, "now_indices", ())
            indices = tuple(int(index) for index in raw_indices)
        except Exception:
            indices = ()
        file_signatures = []
        for attribute in ("data_xyz_path", "nep_txt_path"):
            try:
                target = Path(getattr(result_data, attribute))
                stat = target.stat()
                file_signatures.append((str(target), stat.st_size, stat.st_mtime_ns))
            except Exception:
                file_signatures.append(None)
        return tuple(versions), indices, tuple(file_signatures)

    def open_training_set_audit(
        self,
        result_data=None,
        *,
        initial_section: str = "summary",
        force: bool = False,
    ) -> None:
        """Build and show Training Set Audit for ``result_data`` or the current dataset."""
        data = result_data if result_data is not None else getattr(self.show_nep_interface, "nep_result_data", None)
        if data is None:
            MessageManager.send_info_message(
                self.tr("Please load a dataset before running Training Set Check.")
            )
            return
        dataset_id = str(getattr(data, "data_xyz_path", self.tr("current dataset")))
        signature = self._training_set_audit_signature(data)
        cached_result = getattr(self, "_audited_result", None)
        if (
            not force
            and cached_result is not None
            and getattr(self, "_audited_result_data", None) is data
            and getattr(self, "_audited_result_signature", None) == signature
        ):
            self.training_set_audit_interface.set_result(cached_result)
            self.training_set_audit_interface.set_distribution_context(
                data=data,
                run_analysis_callback=self.show_nep_interface.run_distribution_analysis,
                apply_selection_callback=self.show_nep_interface.apply_distribution_selection,
            )
            if initial_section == "distribution":
                self.training_set_audit_interface.show_distribution_explorer()
            self.stackedWidget.setCurrentWidget(self.training_set_audit_interface)
            return
        self.training_set_audit_interface.set_distribution_context(data=None)
        self.training_set_audit_interface.set_loading(dataset_id)
        self.stackedWidget.setCurrentWidget(self.training_set_audit_interface)

        def apply_result(result) -> None:
            self._training_set_audit_thread = None
            if getattr(self.show_nep_interface, "nep_result_data", None) is not data:
                return
            self._audited_result_data = data
            self._audited_result_signature = self._training_set_audit_signature(data)
            self._audited_result = result
            self.training_set_audit_interface.set_result(result)
            self.training_set_audit_interface.set_distribution_context(
                data=data,
                run_analysis_callback=self.show_nep_interface.run_distribution_analysis,
                apply_selection_callback=self.show_nep_interface.apply_distribution_selection,
            )
            if initial_section == "distribution":
                self.training_set_audit_interface.show_distribution_explorer()
            self.stackedWidget.setCurrentWidget(self.training_set_audit_interface)

        def report_error(message: str) -> None:
            self._training_set_audit_thread = None
            MessageManager.send_warning_message(
                self.tr("Training Set Check failed: {message}").format(message=message)
            )

        self._training_set_audit_thread = run_in_thread(
            self,
            build_training_set_audit,
            data,
            dataset_id=dataset_id,
            include_phase_inventory=False,
            include_magnetic_inventory=False,
            on_finished=apply_result,
            on_error=report_error,
        )

    def handle_training_set_audit_selection(self, indices) -> None:
        """Apply Training Set Audit indices only when the source dataset is still current."""
        current_data = getattr(self.show_nep_interface, "nep_result_data", None)
        current_signature = None if current_data is None else self._training_set_audit_signature(current_data)
        if (
            current_data is None
            or current_data is not self._audited_result_data
            or current_signature != self._audited_result_signature
        ):
            MessageManager.send_info_message(
                self.tr(
                    "Training Set Check results are stale. Please rerun the checks for the current dataset."
                )
            )
            return
        self.show_nep_interface.select_structure_indices(indices)
        self.stackedWidget.setCurrentWidget(self.show_nep_interface)

    def _start_training_set_phase_analysis(self, data, result) -> None:
        """Analyze every structure in the audited scope without blocking the page."""
        if getattr(result, "inventory", None) is None:
            return
        if (
            getattr(result, "phase_inventory", None) is not None
            and (
                getattr(result, "magnetic_inventory", None) is not None
                or result.overview_metrics.get("magnetic_inventory", {}).get("status")
                == "no-spin"
            )
        ):
            return
        if self._training_set_phase_result is result:
            return
        token = object()
        self._training_set_phase_token = token
        self._training_set_phase_result = result
        self.training_set_audit_interface.start_phase_analysis(
            result.inventory.structure_count
        )

        def compute():
            scope_indices = result.scope.indices if result.scope is not None else None
            geometry = data.structure.geometry_snapshot(scope_indices)
            phase_payload = build_phase_inventory(
                geometry,
                result.inventory,
                cache_owner=data.structure,
                progress=lambda completed, total: (
                    self.training_set_audit_interface.phaseAnalysisProgressSignal.emit(
                        completed, total * 2
                    )
                    if token is self._training_set_phase_token
                    else None
                ),
            )
            magnetic_payload = (
                build_magnetic_inventory(
                    geometry,
                    result.inventory,
                    getattr(data.structure, "all_data", ()),
                    cache_owner=data.structure,
                    progress=lambda completed, total: (
                        self.training_set_audit_interface.phaseAnalysisProgressSignal.emit(
                            total + completed, total * 2
                        )
                        if token is self._training_set_phase_token
                        else None
                    ),
                )
                if hasattr(geometry, "source_indices")
                else (None, False)
            )
            return phase_payload, magnetic_payload

        def apply_completed(payload) -> None:
            if token is not self._training_set_phase_token:
                return
            self._training_set_phase_thread = None
            self._training_set_phase_result = None
            (phase_inventory, phase_cache_hit), (
                magnetic_inventory,
                magnetic_cache_hit,
            ) = payload
            if (
                getattr(self.show_nep_interface, "nep_result_data", None) is not data
                or self._training_set_audit_signature(data)
                != self._audited_result_signature
                or self._audited_result is not result
            ):
                return
            phase_meta = dict(result.overview_metrics.get("phase_inventory", {}))
            phase_meta.update(
                {
                    "available": True,
                    "status": "complete",
                    "cache_hit": bool(phase_cache_hit),
                    "analyzed_structures": phase_inventory.analyzed_structure_count,
                }
            )
            overview = dict(result.overview_metrics)
            overview["phase_inventory"] = phase_meta
            overview["magnetic_inventory"] = {
                "available": (
                    magnetic_inventory is not None
                    and magnetic_inventory.analyzed_structure_count > 0
                ),
                "status": (
                    "complete"
                    if magnetic_inventory is not None
                    and magnetic_inventory.analyzed_structure_count > 0
                    else "no-spin"
                ),
                "cache_hit": bool(magnetic_cache_hit),
                "analyzed_structures": (
                    magnetic_inventory.analyzed_structure_count
                    if magnetic_inventory is not None else 0
                ),
                "missing_spin_structures": (
                    magnetic_inventory.missing_spin_count
                    if magnetic_inventory is not None else result.inventory.structure_count
                ),
            }
            updated_result = replace(
                result,
                overview_metrics=overview,
                phase_inventory=phase_inventory,
                magnetic_inventory=magnetic_inventory,
            )
            self._audited_result = updated_result
            self.training_set_audit_interface.finish_phase_analysis(updated_result)

        def report_error(message: str) -> None:
            if token is not self._training_set_phase_token:
                return
            self._training_set_phase_thread = None
            self._training_set_phase_result = None
            self.training_set_audit_interface.fail_phase_analysis(message)
            MessageManager.send_warning_message(
                self.tr("Full phase analysis failed: {message}").format(
                    message=message
                )
            )

        self._training_set_phase_thread = run_in_thread(
            self,
            compute,
            on_finished=apply_completed,
            on_error=report_error,
        )

    def _ensure_update_badge(self) -> None:
        """Create the persistent update badge on the Settings navigation item."""
        if self._update_badge is not None:
            return
        try:
            target = self.navigationInterface.widget(self.setting_interface.objectName())
        except Exception:
            return
        self._update_badge = InfoBadge.error(
            "New",
            parent=self.navigationInterface,
            target=target,
            position=InfoBadgePosition.NAVIGATION_ITEM,
        )
        self._update_badge.setCustomBackgroundColor("#E81123", "#E81123")
        self._update_badge.setFixedHeight(13)
        self._update_badge.adjustSize()
        self._update_badge.hide()

    def refresh_update_indicators(self) -> None:
        """Refresh update indicators in navigation and settings page."""
        self._ensure_update_badge()
        pending_version = get_pending_update_version()
        has_update = bool(pending_version)
        if self._update_badge is not None:
            self._update_badge.setVisible(has_update)
            if has_update and getattr(self._update_badge, "manager", None) is not None:
                self._update_badge.move(self._update_badge.manager.position())
        if hasattr(self.setting_interface, "refresh_update_hint"):
            self.setting_interface.refresh_update_hint()


def global_exception_handler(exc_type, exc_value, exc_traceback) -> None:
    """Log uncaught exceptions through ``loguru`` for post-mortem analysis."""
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(error_message)


def set_light_theme(app: QApplication) -> None:
    """Apply a light colour palette to ``app``."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Button, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(palette)
    app.setStyle("Fusion")


def configure_app(app: QApplication) -> None:
    """Apply the same theme, font, stylesheet, and translator used by the desktop app."""
    set_light_theme(app)
    app.setApplicationName("NepTrainKit")
    install_translator(app)
    icon = _application_icon()
    app.setWindowIcon(icon)
    _set_macos_dock_icon(app, icon)
    font = QFont("Arial", 12)
    app.setFont(font)

    theme_file = QFile(":/theme/src/qss/theme.qss")
    if theme_file.open(QFile.OpenModeFlag.ReadOnly):
        theme = theme_file.readAll().data().decode("utf-8")  # pyright: ignore[reportArgumentType]
        theme_file.close()
        app.setStyleSheet(theme)


def create_app(argv: list[str] | None = None) -> QApplication:
    """Create or configure the shared Qt application instance."""
    setTheme(Theme.LIGHT)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else argv)
    configure_app(app)
    return app


def create_main_window(*, show: bool = True) -> NepTrainKitMainWindow:
    """Create the main window used by both the app and documentation tools."""
    window = NepTrainKitMainWindow()
    if show:
        window.show()
    return window


def main() -> None:
    """Launch the NepTrainKit GUI application."""
    sys.excepthook = global_exception_handler

    update_zip = Path("update.zip")
    update_tar = Path("update.tar.gz")
    if update_zip.exists() or update_tar.exists():
        unzip()

    app = create_app(sys.argv)
    window = create_main_window(show=True)
    window.refresh_update_indicators()
    window.auto_update_notifier = AutoUpdateNotifier(window)
    QTimer.singleShot(3000, window.auto_update_notifier.start_if_due)

    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        resolved = as_path(dir_path).resolve()
        window.show_nep_interface.set_work_path(str(resolved))

    app.exec()


if __name__ == "__main__":
    main()
