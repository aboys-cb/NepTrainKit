#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.pages.makedata import MakeDataWidget


class TestMakeDataSourceCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_run_source_card_without_input_dataset(self):
        widget = MakeDataWidget()
        widget.add_card("CrystalPrototypeBuilderCard")
        self.assertEqual(len(widget.workspace_card_widget.cards), 1)

        widget.run_card()

        card = widget.workspace_card_widget.cards[0]
        self.assertGreater(len(card.result_dataset), 0)

