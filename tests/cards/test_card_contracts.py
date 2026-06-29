from .card_test_base import *
from .card_test_base import _ExternalTestCard, _MetadataTestCard
from unittest.mock import patch
import time

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from NepTrainKit.core.cards.operation import StructureOperation
from NepTrainKit.ui.threads import DataProcessingThread
from NepTrainKit.ui.widgets import FilterDataCard


class TestCardContracts(BaseCardTest):
    def test_builtin_card_has_online_doc_url(self):
        card = StackingFaultCard()

        self.assertEqual(
            card.get_online_doc_url(),
            f"{DOCS_BASE_URL}module/make-dataset-cards/cards/stacking-fault-card.html",
        )
        self.assertFalse(card.doc_button.isHidden())

    def test_external_card_hides_online_doc_button(self):
        card = _ExternalTestCard()

        self.assertEqual(card.get_online_doc_url(), "")
        self.assertTrue(card.doc_button.isHidden())

    def test_card_contributor_metadata_includes_optional_email(self):
        metadata = CardManager.get_card_metadata("_MetadataTestCard")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.card_name, "Metadata Test Card")
        self.assertEqual(metadata.version, "0.1")
        self.assertEqual(metadata.contributors[0].email, "test@example.com")
        self.assertIn("Test Contributor", card_tooltip(metadata))
        self.assertIn("mailto:test@example.com", metadata_html(metadata))

        card = _MetadataTestCard()
        data = card.to_dict()
        self.assertEqual(data["metadata"]["contributors"], ["Test Contributor"])
        self.assertEqual(data["metadata"]["card_version"], "0.1")

    def test_builtin_cards_declare_contributor_metadata(self):
        chen_cards = {"OrganicMolConfigPBCCard", "LocalSolvationCard", "SolventBoxFillCard"}
        for class_name, metadata in CardManager.card_metadata_dict.items():
            if "_card" not in metadata.source_path:
                continue
            self.assertTrue(metadata.contributors, f"{class_name} should declare contributor metadata")
            contributor_names = {item.name for item in metadata.contributors}
            if class_name in chen_cards:
                self.assertIn("Chen Zherui", contributor_names)
            else:
                self.assertIn("NepTrainKit", contributor_names)

    def test_card_status_summary_uses_input_output_time_format(self):
        card = _ExternalTestCard()
        card.set_dataset([self.structure])
        card.result_dataset = [self.structure.copy(), self.structure.copy()]
        card._last_elapsed_seconds = 2.414

        card.update_dataset_info()

        self.assertEqual(card.status_label.text(), "Input: 1 -> Output: 2 | Time: 2.41 s")

    def test_filter_card_status_summary_uses_output_label(self):
        card = FilterDataCard()
        card.set_dataset([self.structure, self.structure.copy()])
        card.result_dataset = [self.structure.copy()]
        card._last_elapsed_seconds = 0.006

        card.update_dataset_info()

        self.assertEqual(card.status_label.text(), "Input: 2 -> Output: 1 | Time: 0.01 s")

    def test_card_stop_waits_for_worker_before_deleting_reference(self):
        class SlowOperation(StructureOperation):
            def run_structure(self, structure, params):
                time.sleep(0.05)
                return [structure.copy()]

        card = PerturbCard()
        card.set_dataset([self.structure.copy()])
        thread = DataProcessingThread([self.structure.copy() for _ in range(5)], SlowOperation(), None)
        card.worker_thread = thread
        thread.start()
        deadline = time.perf_counter() + 2.0
        while not thread.isRunning() and time.perf_counter() < deadline:
            self._app.processEvents()
            time.sleep(0.01)

        card.stop()

        self.assertFalse(thread.isRunning())
        self.assertFalse(hasattr(card, "worker_thread"))

    def test_card_drag_starts_only_after_drag_threshold(self):
        class FakeDrag:
            calls = 0

            def __init__(self, parent):
                FakeDrag.calls += 1

            def setMimeData(self, mime):
                pass

            def setPixmap(self, pixmap):
                pass

            def setHotSpot(self, pos):
                pass

            def exec(self, action):
                pass

        card = _ExternalTestCard()
        card.resize(420, 160)
        card.show()
        self._app.processEvents()

        press_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(10, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        small_move_event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(11, 11),
            QPointF(11, 11),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        large_move_event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(10 + QApplication.startDragDistance() + 1, 10),
            QPointF(10 + QApplication.startDragDistance() + 1, 10),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

        with patch("NepTrainKit.ui.widgets.card_widget.QDrag", FakeDrag):
            card.mousePressEvent(press_event)
            card.mouseMoveEvent(small_move_event)
            self.assertEqual(FakeDrag.calls, 0)

            card.mouseMoveEvent(large_move_event)
            self.assertEqual(FakeDrag.calls, 1)

    def test_operation_cards_write_only_params(self):
        for class_name, card_cls in CardManager.card_info_dict.items():
            if not hasattr(card_cls, "create_operation"):
                continue
            card = card_cls()
            if card.create_operation() is None:
                continue
            serialized = card.to_dict()
            self.assertEqual(
                set(serialized),
                BASE_CARD_KEYS,
                f"{class_name} should write only current params format",
            )

    def test_operation_card_params_roundtrip_through_current_schema(self):
        for class_name, card_cls in CardManager.card_info_dict.items():
            if not hasattr(card_cls, "create_operation"):
                continue
            card = card_cls()
            if card.create_operation() is None:
                continue

            restored = card_cls()
            restored.from_dict(card.to_dict())
            self.assertEqual(
                restored.get_params(),
                card.get_params(),
                f"{class_name} should preserve params through to_dict/from_dict",
            )

    def test_legacy_card_keys_still_load(self):
        strain = CellStrainCard()
        strain.from_dict(
            {
                "check_state": True,
                "organic": True,
                "engine_type": "biaxial",
                "x_range": [1.0, 2.0, 0.5],
                "y_range": [3.0, 4.0, 0.5],
                "z_range": [0.0, 0.0, 1.0],
            }
        )
        self.assertEqual(
            strain.get_params(),
            CellStrainParams(
                axes="biaxial",
                x_range=(1.0, 2.0, 0.5),
                y_range=(3.0, 4.0, 0.5),
                z_range=(0.0, 0.0, 1.0),
                identify_organic=True,
            ),
        )

        layer = LayerCopyCard()
        layer.from_dict(
            {
                "check_state": True,
                "preset_index": 0,
                "dz_expr": "A + z*0",
                "params": "A=1.5",
                "apply_mode": 2,
                "elements": "Si",
                "z_range": [0.0, 2.0],
                "wrap": True,
                "extend_cell_z": False,
                "extra_vacuum": [1.0],
                "layers": [2],
                "distance": [4.0],
            }
        )
        self.assertEqual(
            layer.get_params(),
            LayerCopyParams(
                preset_index=0,
                dz_expr="A + z*0",
                expression_params="A=1.5",
                apply_mode=2,
                elements="Si",
                z_range=(0.0, 2.0),
                wrap=True,
                extend_cell_z=False,
                extra_vacuum=1.0,
                layers=2,
                distance=4.0,
            ),
        )

        operation_params = LayerCopyParams(
            preset_index=1,
            dz_expr="z + 1",
            expression_params="",
            apply_mode=1,
            elements="",
            z_range=(-1.0, 1.0),
            wrap=False,
            extend_cell_z=True,
            extra_vacuum=0.5,
            layers=4,
            distance=2.5,
        )
        layer.from_dict(
            {
                "check_state": True,
                "operation_params": params_to_dict(operation_params),
            }
        )
        self.assertEqual(layer.get_params(), operation_params)
