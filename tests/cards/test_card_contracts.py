from .card_test_base import *
from .card_test_base import _ExternalTestCard, _MetadataTestCard
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
