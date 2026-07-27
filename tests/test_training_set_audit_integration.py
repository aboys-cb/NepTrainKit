#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

import NepTrainKit.main as main_module
import NepTrainKit.ui.pages.show_nep as show_nep_module
from NepTrainKit.config import Config
from NepTrainKit.ui.pages.settings import SettingsWidget
from NepTrainKit.ui.pages.show_nep import ShowNepWidget
from NepTrainKit.ui.views.toolbar import NepDisplayGraphicsToolBar
from NepTrainKit.ui.widgets.training_set_audit_window import (
    TrainingSetAuditHost,
    TrainingSetAuditWindow,
)
from NepTrainKit.core.audit.result import (
    AuditResult,
    AuditScope,
    AuditScopeKind,
    DatasetInventory,
    MagneticInventory,
    PhaseInventory,
)


class _Canvas:
    def __init__(self):
        self.calls = []

    def select_index(self, indices, reverse=False):
        self.calls.append((list(indices), reverse))


class _Menu:
    def __init__(self):
        self.actions = []

    def addAction(self, action):
        self.actions.append(action)

    def removeAction(self, action):
        if action in self.actions:
            self.actions.remove(action)


class TestTrainingSetAuditIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_show_nep_select_structure_indices_replaces_selection(self):
        widget = ShowNepWidget.__new__(ShowNepWidget)
        widget.nep_result_data = SimpleNamespace(select_index={4, 5})
        widget.graph_widget = SimpleNamespace(canvas=_Canvas())
        widget._refresh_export_actions = MagicMock()

        ShowNepWidget.select_structure_indices(widget, [1, 2, 3])

        assert widget.graph_widget.canvas.calls == [([4, 5], True), ([1, 2, 3], False)]

    def test_show_nep_open_training_set_audit_delegates_to_parent(self):
        parent = SimpleNamespace(open_training_set_audit=MagicMock())
        widget = ShowNepWidget.__new__(ShowNepWidget)
        widget._parent = parent
        widget.nep_result_data = SimpleNamespace(load_flag=True, structure=SimpleNamespace(now_data=np.array([1])))

        ShowNepWidget.open_training_set_audit(widget)

        parent.open_training_set_audit.assert_called_once_with(widget.nep_result_data)

    def test_show_nep_distribution_entry_opens_unified_audit_section(self):
        parent = SimpleNamespace(open_training_set_audit=MagicMock())
        widget = ShowNepWidget.__new__(ShowNepWidget)
        widget._parent = parent
        widget.nep_result_data = SimpleNamespace(load_flag=True)

        ShowNepWidget.open_training_set_distribution(widget)

        parent.open_training_set_audit.assert_called_once_with(
            widget.nep_result_data,
            initial_section="distribution",
        )

    def test_show_nep_audit_messages_use_show_nep_translation_context(self):
        widget = ShowNepWidget.__new__(ShowNepWidget)
        widget.nep_result_data = None
        widget._parent = SimpleNamespace()
        with (
            patch.object(
                ShowNepWidget,
                "tr",
                return_value="translated ShowNep audit message",
            ) as tr_mock,
            patch.object(
                show_nep_module.MessageManager, "send_info_message"
            ) as info_mock,
        ):
            ShowNepWidget.open_training_set_audit(widget)

        tr_mock.assert_called_once_with(
            "Please load a dataset before running Training Set Audit."
        )
        info_mock.assert_called_once_with("translated ShowNep audit message")

        widget.nep_result_data = SimpleNamespace(load_flag=True)
        with (
            patch.object(
                ShowNepWidget,
                "tr",
                return_value="translated unavailable audit page",
            ) as tr_mock,
            patch.object(
                show_nep_module.MessageManager, "send_warning_message"
            ) as warning_mock,
        ):
            ShowNepWidget.open_training_set_audit(widget)

        tr_mock.assert_called_once_with("Training Set Audit page is not available.")
        warning_mock.assert_called_once_with("translated unavailable audit page")

    def test_show_event_and_hide_event_keep_audit_out_of_save_menu(self):
        save_menu = _Menu()
        widget = ShowNepWidget.__new__(ShowNepWidget)
        widget._parent = SimpleNamespace(save_menu=save_menu)
        widget.export_all_action = object()
        widget.export_selected_action = object()
        widget.export_removed_action = object()
        widget.export_current_action = object()

        ShowNepWidget.showEvent(widget, None)
        self.assertEqual(
            save_menu.actions,
            [
                widget.export_all_action,
                widget.export_selected_action,
                widget.export_removed_action,
                widget.export_current_action,
            ],
        )

        ShowNepWidget.showEvent(widget, None)
        self.assertEqual(len(save_menu.actions), 4)

        ShowNepWidget.hideEvent(widget, None)
        self.assertEqual(save_menu.actions, [])

    def test_result_toolbar_exposes_training_set_check_instead_of_summary(self):
        toolbar = NepDisplayGraphicsToolBar()
        spy = QSignalSpy(toolbar.trainingSetCheckSignal)

        self.assertIn("training_set_check", toolbar._actions)
        self.assertNotIn("dataset_summary", toolbar._actions)
        self.assertNotIn("explore_distributions", toolbar._actions)
        actions = list(toolbar._actions.values())
        delete_index = next(
            index
            for index, action in enumerate(actions)
            if action.text() == "Delete selected items"
        )
        self.assertIs(actions[delete_index + 1], toolbar._actions["training_set_check"])
        toolbar_widgets = toolbar.bar._widgets
        check_index = next(
            index
            for index, widget in enumerate(toolbar_widgets)
            if callable(getattr(widget, "action", None))
            and widget.action() is toolbar._actions["training_set_check"]
        )
        self.assertEqual(
            type(toolbar_widgets[check_index - 1]).__name__,
            "CommandSeparator",
        )
        toolbar.set_training_set_check_enabled(False)
        self.assertFalse(toolbar._actions["training_set_check"].isEnabled())
        toolbar.set_training_set_check_enabled(True)
        toolbar._actions["training_set_check"].trigger()

        self.assertEqual(spy.count(), 1)

    def test_main_window_rejects_stale_training_set_audit_selection(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        audited_data = object()
        current_data = object()
        window._audited_result_data = audited_data
        window.switchTo = MagicMock()
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=current_data,
            select_structure_indices=MagicMock(),
        )
        with patch.object(main_module.MessageManager, "send_info_message") as info_mock:
            main_module.NepTrainKitMainWindow.handle_training_set_audit_selection(window, [1, 2])

        window.show_nep_interface.select_structure_indices.assert_not_called()
        window.switchTo.assert_not_called()
        info_mock.assert_called_once()

    def test_main_window_audit_messages_use_window_translation_context(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        window.show_nep_interface = SimpleNamespace(nep_result_data=None)
        with (
            patch.object(
                main_module.NepTrainKitMainWindow,
                "tr",
                return_value="translated main audit message",
            ) as tr_mock,
            patch.object(main_module.MessageManager, "send_info_message") as info_mock,
        ):
            main_module.NepTrainKitMainWindow.open_training_set_audit(window)

        tr_mock.assert_called_once_with(
            "Please load a dataset before running Training Set Audit."
        )
        info_mock.assert_called_once_with("translated main audit message")

        current_data = object()
        window._audited_result_data = object()
        window.show_nep_interface = SimpleNamespace(nep_result_data=current_data)
        with (
            patch.object(
                main_module.NepTrainKitMainWindow,
                "tr",
                return_value="translated stale audit message",
            ) as tr_mock,
            patch.object(main_module.MessageManager, "send_info_message") as info_mock,
        ):
            main_module.NepTrainKitMainWindow.handle_training_set_audit_selection(
                window, [1]
            )

        tr_mock.assert_called_once_with(
            "Training Set Audit results are stale. Please rerun the audit for the current dataset."
        )
        info_mock.assert_called_once_with("translated stale audit message")

    def test_main_window_runs_audit_off_the_ui_thread_and_applies_result_on_finish(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        data = SimpleNamespace(data_xyz_path="train.xyz")
        audit_result = object()
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=data,
            run_distribution_analysis=MagicMock(),
            apply_distribution_selection=MagicMock(),
        )
        window.training_set_audit_interface = SimpleNamespace(
            set_loading=MagicMock(),
            set_result=MagicMock(),
            set_distribution_context=MagicMock(),
            show_distribution_explorer=MagicMock(),
        )
        window._start_training_set_phase_analysis = MagicMock()
        window.switchTo = MagicMock()
        callbacks = {}
        scheduled = []

        def fake_run_in_thread(parent, func, *args, on_finished=None, on_error=None, **kwargs):
            callbacks["parent"] = parent
            callbacks["func"] = func
            callbacks["args"] = args
            callbacks["on_finished"] = on_finished
            callbacks["on_error"] = on_error
            return "audit-thread"

        with (
            patch.object(main_module, "run_in_thread", side_effect=fake_run_in_thread),
            patch.object(main_module, "build_training_set_audit", return_value=audit_result),
            patch.object(
                main_module.QTimer,
                "singleShot",
                side_effect=lambda delay, callback: scheduled.append((delay, callback)),
            ),
            patch.object(main_module.Config, "getboolean", return_value=True),
        ):
            main_module.NepTrainKitMainWindow.open_training_set_audit(window, data)
            window.training_set_audit_interface.set_result.assert_not_called()
            window.switchTo.assert_called_once_with(
                window.training_set_audit_interface
            )
            callbacks["on_finished"](audit_result)

        self.assertIs(callbacks["parent"], window)
        self.assertEqual(callbacks["args"], (data,))
        window.training_set_audit_interface.set_loading.assert_called_once_with(
            "train.xyz"
        )
        window.training_set_audit_interface.set_result.assert_called_once_with(audit_result)
        window.training_set_audit_interface.show_distribution_explorer.assert_not_called()
        self.assertEqual(window.switchTo.call_count, 2)
        window.switchTo.assert_called_with(
            window.training_set_audit_interface
        )
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 0)
        scheduled[0][1]()
        window._start_training_set_phase_analysis.assert_called_once_with(
            data, audit_result
        )

    def test_main_window_reuses_unchanged_training_set_audit_result(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        data = SimpleNamespace(data_xyz_path="train.xyz")
        cached_result = object()
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=data,
            run_distribution_analysis=MagicMock(),
            apply_distribution_selection=MagicMock(),
        )
        window.training_set_audit_interface = SimpleNamespace(
            set_loading=MagicMock(),
            set_result=MagicMock(),
            set_distribution_context=MagicMock(),
            show_distribution_explorer=MagicMock(),
        )
        window.switchTo = MagicMock()
        signature = main_module.NepTrainKitMainWindow._training_set_audit_signature(window, data)
        window._audited_result_data = data
        window._audited_result_signature = signature
        window._audited_result = cached_result
        window._start_training_set_phase_analysis = MagicMock()
        scheduled = []

        with (
            patch.object(main_module, "run_in_thread") as thread_mock,
            patch.object(
                main_module.QTimer,
                "singleShot",
                side_effect=lambda delay, callback: scheduled.append((delay, callback)),
            ),
            patch.object(main_module.Config, "getboolean", return_value=True),
        ):
            main_module.NepTrainKitMainWindow.open_training_set_audit(window, data)

        thread_mock.assert_not_called()
        window.training_set_audit_interface.set_loading.assert_not_called()
        window.training_set_audit_interface.set_result.assert_called_once_with(cached_result)
        window.switchTo.assert_called_once_with(
            window.training_set_audit_interface
        )
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 0)
        scheduled[0][1]()
        window._start_training_set_phase_analysis.assert_called_once_with(
            data, cached_result
        )

    def test_main_window_does_not_schedule_structure_evidence_when_disabled(self):
        window = main_module.NepTrainKitMainWindow.__new__(
            main_module.NepTrainKitMainWindow
        )
        window._request_training_set_structure_evidence = MagicMock()

        with (
            patch.object(main_module.Config, "getboolean", return_value=False),
            patch.object(main_module.QTimer, "singleShot") as timer_mock,
        ):
            window._schedule_training_set_structure_evidence()

        timer_mock.assert_not_called()

    def test_settings_default_and_persist_auto_structure_evidence(self):
        previous = Config.get(
            "training_set_audit", "auto_structure_evidence"
        )
        try:
            Config.delete("training_set_audit", "auto_structure_evidence")
            widget = SettingsWidget(None)

            self.assertTrue(widget.auto_structure_evidence_card.isChecked())
            widget.auto_structure_evidence_card.setValue(False)
            self.assertFalse(
                Config.getboolean(
                    "training_set_audit", "auto_structure_evidence", True
                )
            )
        finally:
            if previous is None:
                Config.delete("training_set_audit", "auto_structure_evidence")
            else:
                Config.set(
                    "training_set_audit", "auto_structure_evidence", previous
                )

    def test_main_window_successful_audit_selection_selects_and_navigates_to_dataset_display(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        shared_data = SimpleNamespace(
            structure=SimpleNamespace(
                data=SimpleNamespace(version=3),
                now_indices=np.array([0, 2, 4], dtype=int),
            )
        )
        window._audited_result_data = shared_data
        window._audited_result_signature = (
            main_module.NepTrainKitMainWindow._training_set_audit_signature(
                window,
                shared_data,
            )
        )
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=shared_data,
            select_structure_indices=MagicMock(),
        )
        window.switchTo = MagicMock()

        main_module.NepTrainKitMainWindow.handle_training_set_audit_selection(
            window, [4, 0]
        )

        window.show_nep_interface.select_structure_indices.assert_called_once_with([4, 0])
        window.switchTo.assert_called_once_with(
            window.show_nep_interface
        )

    def test_main_window_runs_complete_phase_analysis_off_ui_thread(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        structure = SimpleNamespace(
            data=SimpleNamespace(version=3),
            now_indices=np.array([0, 1], dtype=int),
            geometry_snapshot=MagicMock(return_value="geometry"),
        )
        data = SimpleNamespace(structure=structure)
        phase = PhaseInventory(
            schema_version="phase-inventory-v2",
            method_id="adaptive-cna-ordering-v1",
            reference_bank_id="aflow-l12-laves-v1",
            analysis_strategy="all-structures-v1",
            source_structure_count=2,
            analyzed_structure_count=2,
            analyzed_atom_count=32,
            composition_points=(),
        )
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 2},
            overview_metrics={"phase_inventory": {"available": False, "status": "pending"}},
            scope=AuditScope(AuditScopeKind.ACTIVE, (0, 1), 2),
            inventory=DatasetInventory(2, ("Ni",), ()),
        )
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=data,
            set_phase_inventory=MagicMock(),
        )
        window.training_set_audit_interface = SimpleNamespace(
            start_phase_analysis=MagicMock(),
            finish_phase_analysis=MagicMock(),
            fail_phase_analysis=MagicMock(),
        )
        window._audited_result_data = data
        window._audited_result = result
        window._training_set_phase_thread = None
        window._training_set_phase_result = None
        window._training_set_phase_token = None
        window._audited_result_signature = (
            main_module.NepTrainKitMainWindow._training_set_audit_signature(window, data)
        )
        callbacks = {}

        def fake_run_in_thread(parent, func, *args, on_finished=None, on_error=None, **kwargs):
            callbacks.update(func=func, on_finished=on_finished, on_error=on_error)
            return "phase-thread"

        with (
            patch.object(main_module, "run_in_thread", side_effect=fake_run_in_thread),
            patch.object(
                main_module,
                "build_phase_inventory",
                return_value=(phase, False),
            ) as phase_mock,
        ):
            main_module.NepTrainKitMainWindow._start_training_set_phase_analysis(
                window, data, result
            )
            payload = callbacks["func"]()
            callbacks["on_finished"](payload)

        window.show_nep_interface.set_phase_inventory.assert_called_once_with(
            phase,
            data,
        )

        structure.geometry_snapshot.assert_called_once_with((0, 1))
        phase_mock.assert_called_once()
        self.assertIs(window._audited_result.phase_inventory, phase)
        window.training_set_audit_interface.start_phase_analysis.assert_called_once_with(2)
        window.training_set_audit_interface.finish_phase_analysis.assert_called_once_with(
            window._audited_result
        )

    def test_main_window_reuses_persistent_phase_and_magnetic_evidence(self):
        window = main_module.NepTrainKitMainWindow.__new__(
            main_module.NepTrainKitMainWindow
        )
        structure = SimpleNamespace(
            data=SimpleNamespace(version=3),
            now_indices=np.array([0, 1], dtype=int),
            geometry_snapshot=MagicMock(return_value="geometry"),
        )
        data = SimpleNamespace(structure=structure)
        phase = PhaseInventory(
            schema_version="phase-inventory-v2",
            method_id="adaptive-cna-ordering-v1",
            reference_bank_id="aflow-l12-laves-v1",
            analysis_strategy="all-structures-v1",
            source_structure_count=2,
            analyzed_structure_count=2,
            analyzed_atom_count=32,
            composition_points=(),
        )
        magnetic = MagneticInventory(
            schema_version="magnetic-inventory-v3",
            method_id="spin-order-layer-afm-v3",
            analysis_strategy="all-spin-structures-v1",
            source_structure_count=2,
            analyzed_structure_count=2,
            missing_spin_count=0,
            composition_points=(),
        )
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 2},
            overview_metrics={
                "phase_inventory": {"available": False, "status": "pending"}
            },
            scope=AuditScope(AuditScopeKind.ACTIVE, (0, 1), 2),
            inventory=DatasetInventory(2, ("Ni",), ()),
        )
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=data,
            set_phase_inventory=MagicMock(),
        )
        window.training_set_audit_interface = SimpleNamespace(
            start_phase_analysis=MagicMock(),
            finish_phase_analysis=MagicMock(),
            fail_phase_analysis=MagicMock(),
            phaseAnalysisProgressSignal=SimpleNamespace(emit=MagicMock()),
        )
        window._audited_result_data = data
        window._audited_result = result
        window._training_set_phase_thread = None
        window._training_set_phase_result = None
        window._training_set_phase_token = None
        window._audited_result_signature = (
            main_module.NepTrainKitMainWindow._training_set_audit_signature(
                window, data
            )
        )
        callbacks = {}
        cache = SimpleNamespace(
            load_phase=MagicMock(return_value=phase),
            load_magnetic=MagicMock(return_value=magnetic),
            save_phase=MagicMock(),
            save_magnetic=MagicMock(),
        )

        def fake_run_in_thread(
            parent,
            func,
            *args,
            on_finished=None,
            on_error=None,
            **kwargs,
        ):
            callbacks.update(func=func, on_finished=on_finished, on_error=on_error)
            return "phase-thread"

        with (
            patch.object(main_module, "run_in_thread", side_effect=fake_run_in_thread),
            patch.object(
                main_module.TrainingSetEvidenceCache,
                "from_result_data",
                return_value=cache,
            ),
            patch.object(main_module, "build_phase_inventory") as phase_build,
            patch.object(main_module, "build_magnetic_inventory") as magnetic_build,
        ):
            main_module.NepTrainKitMainWindow._start_training_set_phase_analysis(
                window, data, result
            )
            payload = callbacks["func"]()
            callbacks["on_finished"](payload)

        phase_build.assert_not_called()
        magnetic_build.assert_not_called()
        cache.save_phase.assert_not_called()
        cache.save_magnetic.assert_not_called()
        self.assertTrue(
            window._audited_result.overview_metrics["phase_inventory"]["cache_hit"]
        )
        self.assertTrue(
            window._audited_result.overview_metrics["magnetic_inventory"]["cache_hit"]
        )

    def test_main_window_rejects_same_object_when_active_signature_changes(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        shared_data = SimpleNamespace(
            structure=SimpleNamespace(
                data=SimpleNamespace(version=2),
                now_indices=np.array([0, 2, 4], dtype=int),
            )
        )
        window._audited_result_data = shared_data
        window._audited_result_signature = (1, (0, 1, 2))
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=shared_data,
            select_structure_indices=MagicMock(),
        )
        with patch.object(main_module.MessageManager, "send_info_message") as info_mock:
            main_module.NepTrainKitMainWindow.handle_training_set_audit_selection(window, [1, 2])

        window.show_nep_interface.select_structure_indices.assert_not_called()
        info_mock.assert_called_once()

    def test_main_window_connects_rerun_signal_to_open_training_set_audit(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        window.training_set_audit_interface = main_module.TrainingSetAuditWidget()
        window._audit_window_owner = QWidget()
        window.training_set_audit_host = TrainingSetAuditHost()
        window.training_set_audit_window = TrainingSetAuditWindow(
            window._audit_window_owner
        )
        window.handle_training_set_audit_selection = MagicMock()
        window.open_training_set_audit = MagicMock()
        window.open_dataset_for_training_set_audit = MagicMock()
        window._request_training_set_structure_evidence = MagicMock()
        main_module.NepTrainKitMainWindow._connect_training_set_audit_signals(window)

        window.training_set_audit_interface.rerunAuditSignal.emit()
        window.training_set_audit_interface.requestStructureEvidenceSignal.emit()
        window.training_set_audit_interface.requestDatasetOpenSignal.emit()

        window.open_training_set_audit.assert_called_once_with(force=True)
        window._request_training_set_structure_evidence.assert_called_once_with()
        window.open_dataset_for_training_set_audit.assert_called_once_with()

    def test_training_set_audit_moves_between_host_and_owned_window(self):
        owner = QWidget()
        host = TrainingSetAuditHost(owner)
        audit = main_module.TrainingSetAuditWidget(host)
        host.attach(audit)
        self.assertEqual(
            audit.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        floating = TrainingSetAuditWindow(owner)
        window = main_module.NepTrainKitMainWindow.__new__(
            main_module.NepTrainKitMainWindow
        )
        window.training_set_audit_host = host
        window.training_set_audit_interface = audit
        window.training_set_audit_window = floating
        window.show_nep_interface = QWidget()
        window.switchTo = MagicMock()

        with patch.object(floating, "show_owned") as show_mock:
            main_module.NepTrainKitMainWindow.detach_training_set_audit(window)

        self.assertIsNone(host.content)
        self.assertIs(floating.content, audit)
        self.assertIs(audit.parentWidget(), floating)
        self.assertEqual(
            audit.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding,
        )
        self.assertFalse(host.placeholder.isHidden())
        self.assertIn("Return", audit.detach_button.toolTip())
        show_mock.assert_called_once_with()
        window.switchTo.assert_called_once_with(window.show_nep_interface)

        window.switchTo.reset_mock()
        with patch.object(floating, "remember_geometry") as remember_mock:
            main_module.NepTrainKitMainWindow.restore_training_set_audit(window)

        remember_mock.assert_called_once_with()
        self.assertIsNone(floating.content)
        self.assertIs(host.content, audit)
        self.assertIs(audit.parentWidget(), host)
        self.assertEqual(
            audit.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        self.assertTrue(host.placeholder.isHidden())
        self.assertIn("separate", audit.detach_button.toolTip())
        window.switchTo.assert_called_once_with(host)

    def test_owned_audit_window_close_requests_return_without_destroying_content(self):
        owner = QWidget()
        floating = TrainingSetAuditWindow(owner)
        content = QWidget()
        floating.attach(content)
        spy = QSignalSpy(floating.returnRequested)
        event = QCloseEvent()

        floating.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertEqual(spy.count(), 1)
        self.assertIs(floating.content, content)

    def test_audit_host_preserves_global_open_action_while_detached(self):
        host = TrainingSetAuditHost()
        audit = main_module.TrainingSetAuditWidget(host)
        host.attach(audit)
        spy = QSignalSpy(audit.requestDatasetOpenSignal)

        host.open_file()
        detached = host.take()
        host.open_file()

        self.assertIs(detached, audit)
        self.assertEqual(spy.count(), 2)

    def test_opening_audit_raises_existing_owned_window(self):
        window = main_module.NepTrainKitMainWindow.__new__(
            main_module.NepTrainKitMainWindow
        )
        window.training_set_audit_interface = object()
        window.training_set_audit_window = SimpleNamespace(
            is_detached=True,
            show_owned=MagicMock(),
        )
        window.switchTo = MagicMock()

        main_module.NepTrainKitMainWindow._show_training_set_audit_surface(window)

        window.training_set_audit_window.show_owned.assert_called_once_with()
        window.switchTo.assert_not_called()

    def test_audit_open_action_switches_to_display_before_opening(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        window.show_nep_interface = SimpleNamespace(open_file=MagicMock())
        window.switchTo = MagicMock()

        main_module.NepTrainKitMainWindow.open_dataset_for_training_set_audit(window)

        window.switchTo.assert_called_once_with(window.show_nep_interface)
        window.show_nep_interface.open_file.assert_called_once_with()

    def test_global_actions_follow_current_page_capabilities(self):
        class ActionProbe:
            def __init__(self):
                self.enabled = None
                self.tooltip = ""

            def setEnabled(self, enabled):
                self.enabled = enabled

            def setToolTip(self, tooltip):
                self.tooltip = tooltip

        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        window.open_dir_button = ActionProbe()
        window.save_dir_button = ActionProbe()
        current_page = SimpleNamespace(open_file=lambda: None)
        window.stackedWidget = SimpleNamespace(currentWidget=lambda: current_page)

        main_module.NepTrainKitMainWindow._refresh_page_actions(window)

        self.assertTrue(window.open_dir_button.enabled)
        self.assertFalse(window.save_dir_button.enabled)
        self.assertIn("not available", window.save_dir_button.tooltip)

    def test_make_dataset_output_handoff_opens_in_memory_dataset_directly(self):
        window = main_module.NepTrainKitMainWindow.__new__(
            main_module.NepTrainKitMainWindow
        )
        window._make_dataset_handoff_thread = None
        window._make_dataset_handoff_token = None
        window.show_nep_interface = SimpleNamespace(check_nep_result=MagicMock())
        window.switchTo = MagicMock()
        callbacks = {}

        def fake_run_in_thread(
            _parent, func, *args, on_finished=None, on_error=None, **kwargs
        ):
            del args, kwargs
            callbacks["func"] = func
            callbacks["finished"] = on_finished
            callbacks["error"] = on_error
            return SimpleNamespace(isRunning=lambda: False)

        with patch.object(main_module, "run_in_thread", side_effect=fake_run_in_thread):
            main_module.NepTrainKitMainWindow.open_make_dataset_output(
                window,
                [Atoms("Fe", positions=[[0.0, 0.0, 0.0]])],
            )

        converted = callbacks["func"]()
        callbacks["finished"](converted)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].elements.tolist(), ["Fe"])
        window.switchTo.assert_called_once_with(window.show_nep_interface)
        window.show_nep_interface.check_nep_result.assert_called_once_with(
            structures=converted,
            cache_outputs=False,
            source_name="Make Dataset output",
        )
        self.assertIsNone(window._make_dataset_handoff_thread)
        self.assertIsNone(window._make_dataset_handoff_token)

    def test_make_dataset_handoff_recovers_from_deleted_worker_wrapper(self):
        class DeletedThread:
            def isRunning(self):
                raise RuntimeError("Internal C++ object already deleted")

        window = main_module.NepTrainKitMainWindow.__new__(
            main_module.NepTrainKitMainWindow
        )
        window._make_dataset_handoff_thread = DeletedThread()
        window._make_dataset_handoff_token = None
        window.show_nep_interface = SimpleNamespace(check_nep_result=MagicMock())
        window.switchTo = MagicMock()
        replacement_thread = SimpleNamespace(isRunning=lambda: False)

        with patch.object(
            main_module, "run_in_thread", return_value=replacement_thread
        ) as run_mock:
            main_module.NepTrainKitMainWindow.open_make_dataset_output(
                window,
                [Atoms("Fe", positions=[[0.0, 0.0, 0.0]])],
            )

        run_mock.assert_called_once()
        self.assertIs(window._make_dataset_handoff_thread, replacement_thread)
        self.assertIsNotNone(window._make_dataset_handoff_token)

    def test_shipped_chinese_catalog_translates_audit_integration_contexts(self):
        catalog = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        translator = QTranslator(self._app)
        self.assertTrue(translator.load(str(catalog)))
        self._app.installTranslator(translator)
        try:
            main_context = {
                "Open data for this page": "打开当前页面的数据",
                "Open is not available on this page": "当前页面不支持打开操作",
                "Save data from this page": "保存当前页面的数据",
                "Save is not available on this page": "当前页面不支持保存操作",
                "Preparing the workflow output for display...": "正在准备工作流输出以供查看...",
                "Make Dataset": "构建数据集",
                "Training Set Audit": "训练集评估",
                "current dataset": "当前数据集",
                "Please load a dataset before running Training Set Audit.": "请先加载数据集，再运行训练集评估。",
                "Training Set Audit results are stale. Please rerun the audit for the current dataset.": "训练集评估结果已过期，请针对当前数据集重新评估。",
                "Full phase analysis failed: {message}": "完整相分析失败：{message}",
            }
            show_nep_context = {
                "Please load a dataset before running Training Set Audit.": "请先加载数据集，再运行训练集评估。",
                "Training Set Audit page is not available.": "训练集评估页面不可用。",
            }
            toolbar_context = {
                "Training Set Audit": "训练集评估",
            }
            settings_context = {
                "Automatically analyze structure evidence": "自动分析结构证据",
                "After the basic dataset audit appears, analyze phases and magnetic order in the background": "基础诊断显示后，在后台分析相结构与磁序",
            }
            for source, expected in main_context.items():
                self.assertEqual(
                    QCoreApplication.translate("NepTrainKitMainWindow", source),
                    expected,
                )
            for source, expected in show_nep_context.items():
                self.assertEqual(
                    QCoreApplication.translate("ShowNepWidget", source),
                    expected,
                )
            for source, expected in toolbar_context.items():
                self.assertEqual(
                    QCoreApplication.translate("NepDisplayGraphicsToolBar", source),
                    expected,
                )
            for source, expected in settings_context.items():
                self.assertEqual(
                    QCoreApplication.translate("SettingsWidget", source),
                    expected,
                )
        finally:
            self._app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()
