#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from PySide6.QtCore import QEventLoop, QTimer
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

        card = widget.workspace_card_widget.cards[0]
        loop = QEventLoop()
        card.runFinishedSignal.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)

        widget.run_card()
        loop.exec()

        self.assertGreater(len(card.result_dataset), 0)

    def test_copy_single_card_json_can_be_pasted_back(self):
        widget = MakeDataWidget()
        widget.add_card("CrystalPrototypeBuilderCard")
        card = widget.workspace_card_widget.cards[0]

        payload = json.loads(card.to_json_text())

        self.assertEqual(payload["class"], "CrystalPrototypeBuilderCard")
        self.assertIn("params", payload)

        restored = MakeDataWidget()
        restored._add_card_configs(restored._normalise_card_config_payload(payload))

        self.assertEqual(len(restored.workspace_card_widget.cards), 1)
        self.assertEqual(restored.workspace_card_widget.cards[0].__class__.__name__, "CrystalPrototypeBuilderCard")

    def test_copy_workflow_json_can_be_pasted_back(self):
        widget = MakeDataWidget()
        widget.add_card("CrystalPrototypeBuilderCard")
        widget.add_card("SuperCellCard")

        payload = json.loads(widget.current_card_config_json())

        self.assertIn("software_version", payload)
        self.assertEqual([card["class"] for card in payload["cards"]], ["CrystalPrototypeBuilderCard", "SuperCellCard"])

        restored = MakeDataWidget()
        restored._add_card_configs(restored._normalise_card_config_payload(payload))

        self.assertEqual(
            [card.__class__.__name__ for card in restored.workspace_card_widget.cards],
            ["CrystalPrototypeBuilderCard", "SuperCellCard"],
        )
