#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtWidgets import QApplication

import NepTrainKit.main as main_module
import NepTrainKit.ui.pages.show_nep as show_nep_module
from NepTrainKit.ui.pages.show_nep import ShowNepWidget
from NepTrainKit.core.audit.result import (
    AuditResult,
    AuditScope,
    AuditScopeKind,
    DatasetInventory,
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
            "Please load a dataset before running Training Set Check."
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

        tr_mock.assert_called_once_with("Training Set Check page is not available.")
        warning_mock.assert_called_once_with("translated unavailable audit page")

    def test_show_event_and_hide_event_manage_audit_action_lifecycle(self):
        save_menu = _Menu()
        widget = ShowNepWidget.__new__(ShowNepWidget)
        widget._parent = SimpleNamespace(save_menu=save_menu)
        widget.export_all_action = object()
        widget.export_selected_action = object()
        widget.export_removed_action = object()
        widget.export_current_action = object()
        widget.audit_current_dataset_action = object()

        ShowNepWidget.showEvent(widget, None)
        self.assertEqual(save_menu.actions[-1], widget.audit_current_dataset_action)
        self.assertEqual(save_menu.actions.count(widget.audit_current_dataset_action), 1)

        ShowNepWidget.showEvent(widget, None)
        self.assertEqual(save_menu.actions.count(widget.audit_current_dataset_action), 1)

        ShowNepWidget.hideEvent(widget, None)
        self.assertNotIn(widget.audit_current_dataset_action, save_menu.actions)

    def test_main_window_rejects_stale_training_set_audit_selection(self):
        window = main_module.NepTrainKitMainWindow.__new__(main_module.NepTrainKitMainWindow)
        audited_data = object()
        current_data = object()
        window._audited_result_data = audited_data
        window.stackedWidget = SimpleNamespace(setCurrentWidget=MagicMock())
        window.show_nep_interface = SimpleNamespace(
            nep_result_data=current_data,
            select_structure_indices=MagicMock(),
        )
        with patch.object(main_module.MessageManager, "send_info_message") as info_mock:
            main_module.NepTrainKitMainWindow.handle_training_set_audit_selection(window, [1, 2])

        window.show_nep_interface.select_structure_indices.assert_not_called()
        window.stackedWidget.setCurrentWidget.assert_not_called()
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
            "Please load a dataset before running Training Set Check."
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
            "Training Set Check results are stale. Please rerun the checks for the current dataset."
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
        window.stackedWidget = SimpleNamespace(setCurrentWidget=MagicMock())
        callbacks = {}

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
        ):
            main_module.NepTrainKitMainWindow.open_training_set_audit(window, data)

        self.assertIs(callbacks["parent"], window)
        self.assertEqual(callbacks["args"], (data,))
        window.training_set_audit_interface.set_loading.assert_called_once_with(
            "train.xyz"
        )
        window.stackedWidget.setCurrentWidget.assert_called_once_with(
            window.training_set_audit_interface
        )
        window.training_set_audit_interface.set_result.assert_not_called()
        callbacks["on_finished"](audit_result)
        window.training_set_audit_interface.set_result.assert_called_once_with(audit_result)
        window.training_set_audit_interface.show_distribution_explorer.assert_not_called()
        self.assertEqual(window.stackedWidget.setCurrentWidget.call_count, 2)
        window.stackedWidget.setCurrentWidget.assert_called_with(
            window.training_set_audit_interface
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
        window.stackedWidget = SimpleNamespace(setCurrentWidget=MagicMock())
        signature = main_module.NepTrainKitMainWindow._training_set_audit_signature(window, data)
        window._audited_result_data = data
        window._audited_result_signature = signature
        window._audited_result = cached_result

        with patch.object(main_module, "run_in_thread") as thread_mock:
            main_module.NepTrainKitMainWindow.open_training_set_audit(window, data)

        thread_mock.assert_not_called()
        window.training_set_audit_interface.set_loading.assert_not_called()
        window.training_set_audit_interface.set_result.assert_called_once_with(cached_result)
        window.stackedWidget.setCurrentWidget.assert_called_once_with(
            window.training_set_audit_interface
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
        window.stackedWidget = SimpleNamespace(setCurrentWidget=MagicMock())

        main_module.NepTrainKitMainWindow.handle_training_set_audit_selection(
            window, [4, 0]
        )

        window.show_nep_interface.select_structure_indices.assert_called_once_with([4, 0])
        window.stackedWidget.setCurrentWidget.assert_called_once_with(
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
        window.show_nep_interface = SimpleNamespace(nep_result_data=data)
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

        structure.geometry_snapshot.assert_called_once_with((0, 1))
        phase_mock.assert_called_once()
        self.assertIs(window._audited_result.phase_inventory, phase)
        window.training_set_audit_interface.start_phase_analysis.assert_called_once_with(2)
        window.training_set_audit_interface.finish_phase_analysis.assert_called_once_with(
            window._audited_result
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
        window.handle_training_set_audit_selection = MagicMock()
        window.open_training_set_audit = MagicMock()
        main_module.NepTrainKitMainWindow._connect_training_set_audit_signals(window)

        window.training_set_audit_interface.rerunAuditSignal.emit()

        window.open_training_set_audit.assert_called_once_with(force=True)

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
                "Training Set Check": "训练集评估",
                "current dataset": "当前数据集",
                "Please load a dataset before running Training Set Check.": "请先加载数据集，再运行训练集评估。",
                "Training Set Check results are stale. Please rerun the checks for the current dataset.": "训练集评估结果已过期，请针对当前数据集重新检查。",
                "Full phase analysis failed: {message}": "完整相分析失败：{message}",
            }
            show_nep_context = {
                "Check current dataset": "评估当前数据集",
                "Please load a dataset before running Training Set Check.": "请先加载数据集，再运行训练集评估。",
                "Training Set Check page is not available.": "训练集评估页面不可用。",
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
        finally:
            self._app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()
