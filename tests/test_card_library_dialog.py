#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QObject, Qt, QTranslator, Signal
from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.views.cards import ConsoleWidget
from NepTrainKit.ui.widgets.card_metadata import CardLibraryDialog


class TestCardLibraryDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_library_search_filters_card_metadata(self):
        dialog = CardLibraryDialog()
        self.assertGreater(dialog.card_list.count(), 1)
        class_name, _metadata = next(iter(dialog._metadata_by_class.items()))

        dialog.search_edit.setText(class_name)

        visible = [
            dialog.card_list.item(row)
            for row in range(dialog.card_list.count())
            if not dialog.card_list.item(row).isHidden()
        ]
        self.assertGreaterEqual(len(visible), 1)
        self.assertTrue(
            any(
                item.data(Qt.ItemDataRole.UserRole) == class_name
                for item in visible
            )
        )
        self.assertIn(str(len(visible)), dialog.result_count_label.text())

    def test_add_button_emits_selected_card_class(self):
        dialog = CardLibraryDialog()
        requested = []
        dialog.cardRequested.connect(requested.append)
        selected = dialog.card_list.currentItem()
        class_name = selected.data(Qt.ItemDataRole.UserRole)

        dialog.add_button.click()

        self.assertEqual(requested, [class_name])

    def test_empty_search_result_disables_add_action(self):
        dialog = CardLibraryDialog()

        dialog.search_edit.setText("card-name-that-does-not-exist")

        self.assertFalse(dialog.add_button.isEnabled())
        self.assertTrue(dialog.result_count_label.text().startswith("0"))

    def test_chinese_catalog_localizes_dynamic_card_metadata(self):
        translator = QTranslator(self._app)
        qm_path = (
            Path(__file__).parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        self.assertTrue(translator.load(str(qm_path)))
        self._app.installTranslator(translator)
        try:
            dialog = CardLibraryDialog()
            item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "CompositionGradientCard"
            )

            self.assertIn("[合金与组分]", item.text())
            self.assertIn("组分梯度", item.text())
            dialog.card_list.setCurrentItem(item)
            self.assertEqual(dialog.detail_title_label.text(), "组分梯度")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "按分层组分梯度分配原子种类。",
            )
            self.assertIn("作者", dialog.detail_contributors_label.text())

            dialog.search_edit.setText("组分梯度")
            self.assertFalse(item.isHidden())
        finally:
            self._app.removeTranslator(translator)

    def test_card_details_prioritize_user_facing_metadata(self):
        dialog = CardLibraryDialog()
        item = next(
            dialog.card_list.item(row)
            for row in range(dialog.card_list.count())
            if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
            == "CompositionGradientCard"
        )

        dialog.card_list.setCurrentItem(item)

        self.assertEqual(dialog.detail_title_label.text(), "Composition Gradient")
        self.assertEqual(dialog.detail_group_label.text(), "Alloy")
        self.assertTrue(dialog.detail_description_label.text())
        self.assertTrue(dialog.detail_technical_panel.isHidden())
        self.assertNotIn(
            "/Users/", dialog.detail_source_label.text()
        )

        dialog.detail_technical_button.click()

        self.assertFalse(dialog.detail_technical_panel.isHidden())
        self.assertEqual(
            dialog.detail_class_value.text(), "CompositionGradientCard"
        )
        self.assertEqual(
            dialog.detail_path_value.text(), "composition_gradient_card.py"
        )

    def test_console_forwards_library_add_request(self):
        class FakeLibraryDialog(QObject):
            cardRequested = Signal(str)

            def __init__(self, parent=None):
                super().__init__(parent)

            def exec(self):
                self.cardRequested.emit("CrystalPrototypeBuilderCard")

        console = ConsoleWidget()
        requested = []
        console.newCardSignal.connect(requested.append)
        with patch(
            "NepTrainKit.ui.views.cards.CardLibraryDialog", FakeLibraryDialog
        ):
            console.show_card_library()

        self.assertEqual(requested, ["CrystalPrototypeBuilderCard"])

    def test_console_exposes_labeled_card_search_button(self):
        console = ConsoleWidget()

        self.assertEqual(console.find_card_button.text(), "Find card")
        self.assertTrue(console.find_card_button.isEnabled())

    def test_console_card_menu_scrolls_instead_of_exceeding_screen_height(self):
        console = ConsoleWidget()

        self.assertEqual(console.menu.view.maxVisibleItems(), 10)
        self.assertGreater(len(console.menu.actions()), 10)

    def test_console_exposes_selected_output_handoff_action(self):
        console = ConsoleWidget()
        requests = []
        console.viewOutputSignal.connect(lambda: requests.append(True))

        self.assertFalse(console.view_output_button.isEnabled())
        self.assertEqual(console.view_output_button.text(), "View selected outputs")
        console.set_output_available(True)
        console.view_output_button.click()

        self.assertEqual(requests, [True])


if __name__ == "__main__":
    unittest.main()
