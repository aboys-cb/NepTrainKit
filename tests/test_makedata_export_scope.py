#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QTranslator
from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.widgets.card_widget import MakeDataCard


def _card(name, *, enabled=True, output_count=1):
    card = SimpleNamespace(
        name=name,
        check_state=enabled,
        result_dataset=[object()] * output_count,
    )

    def write_result_dataset(file, *, append):
        file.write(f"{name}:{append}\n")

    card.write_result_dataset = MagicMock(side_effect=write_result_dataset)
    return card


class TestMakeDataExportScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_default_export_selects_only_final_enabled_output(self):
        first = _card("first")
        disabled = _card("disabled", enabled=False)
        final = _card("final")
        page = SimpleNamespace(
            workspace_card_widget=SimpleNamespace(cards=[first, disabled, final])
        )

        cards = MakeDataWidget._cards_for_export(page, include_all=False)

        self.assertEqual(cards, [final])

    def test_all_outputs_scope_skips_disabled_and_unavailable_cards(self):
        first = _card("first")
        unavailable = _card("unavailable", output_count=0)
        disabled = _card("disabled", enabled=False)
        final = _card("final")
        page = SimpleNamespace(
            workspace_card_widget=SimpleNamespace(
                cards=[first, unavailable, disabled, final]
            )
        )

        cards = MakeDataWidget._cards_for_export(page, include_all=True)

        self.assertEqual(cards, [first, final])

    def test_final_scope_uses_last_output_completed_in_current_run(self):
        completed = _card("completed")
        stale = _card("stale")
        page = SimpleNamespace(
            workspace_card_widget=SimpleNamespace(cards=[completed, stale]),
            _last_completed_card_index=0,
        )

        cards = MakeDataWidget._cards_for_export(page, include_all=False)

        self.assertEqual(cards, [completed])

    def test_writer_preserves_explicit_card_order_and_append_mode(self):
        first = _card("first")
        final = _card("final")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.xyz"

            MakeDataWidget._export_file(None, str(path), [first, final])

            self.assertEqual(path.read_text(), "first:False\nfinal:True\n")

    def test_writer_replaces_existing_file_only_after_success(self):
        first = _card("first")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.xyz"
            path.write_text("old dataset\n", encoding="utf-8")

            MakeDataWidget._export_file(None, str(path), [first])

            self.assertEqual(path.read_text(encoding="utf-8"), "first:False\n")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_writer_preserves_existing_file_when_card_export_fails(self):
        failing = _card("failing")

        def fail_after_partial_write(file, *, append):
            file.write("partial dataset\n")
            raise RuntimeError("synthetic export failure")

        failing.write_result_dataset.side_effect = fail_after_partial_write
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.xyz"
            path.write_text("old dataset\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "synthetic export failure"
            ):
                MakeDataWidget._export_file(None, str(path), [failing])

            self.assertEqual(path.read_text(encoding="utf-8"), "old dataset\n")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_chinese_catalog_names_both_export_scopes(self):
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
            widget = MakeDataWidget()
            self.assertEqual(
                widget.export_final_output_action.text(), "导出工作流最终输出"
            )
            self.assertEqual(
                widget.export_all_outputs_action.text(), "导出全部可用卡片输出"
            )
            self.assertEqual(
                widget.setting_group.view_output_button.text(), "查看输出"
            )
            self.assertEqual(
                widget.setting_group.view_output_action.text(), "查看勾选输出"
            )
        finally:
            self._app.removeTranslator(translator)

    def test_all_checked_outputs_are_emitted_for_dataset_display(self):
        widget = MakeDataWidget()
        first_structures = [object()]
        final_structures = [object(), object()]
        first = SimpleNamespace(result_dataset=first_structures)
        final = SimpleNamespace(result_dataset=final_structures)
        widget._cards_for_export = MagicMock(return_value=[first, final])
        requested = []
        widget.finalOutputRequestedSignal.connect(requested.append)

        widget.request_selected_outputs()

        widget._cards_for_export.assert_called_once_with(include_all=True)
        self.assertEqual(requested, [first_structures + final_structures])

    def test_card_output_action_emits_only_that_card_result(self):
        widget = MakeDataWidget()
        card = MakeDataCard(widget)
        structures = [object(), object()]
        card.result_dataset = structures
        card.set_output_available(True)
        widget._connect_card_output_actions(card)
        requested = []
        widget.finalOutputRequestedSignal.connect(requested.append)

        card.view_output_button.click()

        self.assertEqual(requested, [structures])

    @patch("NepTrainKit.ui.pages.makedata.call_path_dialog")
    @patch("NepTrainKit.ui.pages.makedata.MessageManager.send_info_message")
    def test_final_export_without_output_explains_next_step(
        self, info_message, path_dialog
    ):
        card = _card("final", output_count=0)
        page = SimpleNamespace(
            workspace_card_widget=SimpleNamespace(cards=[card]),
            tr=lambda text: text,
        )
        page._cards_for_export = lambda include_all: MakeDataWidget._cards_for_export(
            page, include_all
        )

        MakeDataWidget._start_export(page, include_all=False)

        path_dialog.assert_not_called()
        info_message.assert_called_once_with(
            "The final enabled card has no output. Run the workflow first."
        )


if __name__ == "__main__":
    unittest.main()
