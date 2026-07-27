from .card_test_base import *
from unittest.mock import patch

from NepTrainKit.core.audit.neighbor_scan import find_short_distance_structure_rows
from NepTrainKit.ui.views._card.fps_filter_card import FPSFilterDataCard
from NepTrainKit.ui.widgets.card_widget import FilterDataCard


class TestFilterCards(BaseCardTest):
    def test_fps_filter_operation_rejects_missing_model(self):
        params = FPSFilterParams(nep_path=str(self.test_dir / "data" / "missing_nep.txt"))

        with self.assertRaises(FileNotFoundError):
            FPSFilterOperation().run_dataset([self.structure.copy()], params)

    def test_fps_filter_card_roundtrip_preserves_backend_policy(self):
        card = FPSFilterDataCard()
        card.set_params(
            FPSFilterParams(
                nep_path="/tmp/nep.txt",
                n_samples=42,
                min_distance=0.125,
                backend="cuda",
                chunk_max_atoms=54321,
                strategy="element_set",
                existing_dataset_path="/tmp/train.xyz",
            )
        )
        restored = FPSFilterDataCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
        self.assertEqual(restored.strategy_combo.currentData(), "element_set")
        self.assertTrue(restored.advanced_button.isChecked())

    def test_fps_filter_legacy_roundtrip_keeps_global_strategy(self):
        restored = FPSFilterDataCard()
        restored.from_dict(
            {
                "class": "FPSFilterDataCard",
                "check_state": True,
                "nep_path": "/tmp/nep.txt",
                "num_condition": [12],
                "min_distance_condition": [0.02],
            }
        )

        self.assertEqual(restored.get_params().strategy, "global")
        self.assertEqual(restored.get_params().existing_dataset_path, "")

    def test_element_set_fps_covers_groups_and_records_report(self):
        dataset = [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[1.0, 0.0, 0.0]]),
            Atoms("H", positions=[[2.0, 0.0, 0.0]]),
            Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            Atoms("He", positions=[[1.0, 0.0, 0.0]]),
            Atoms("HHe", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ]
        descriptors = np.arange(len(dataset), dtype=float)[:, None]
        params = FPSFilterParams(
            nep_path=str(self.test_dir / "data" / "nep" / "nep.txt"),
            n_samples=5,
            min_distance=0.0,
            strategy="element_set",
        )
        operation = FPSFilterOperation()
        with patch("NepTrainKit.core.cards.filter.NepCalculator") as calculator_class:
            calculator_class.return_value.descriptors.return_value = descriptors
            selected = operation.run_dataset(dataset, params)

        self.assertEqual(len(selected), 5)
        self.assertEqual(
            {operation.element_set_key(structure) for structure in selected},
            {("H",), ("He",), ("H", "He")},
        )
        self.assertEqual(sum(item.selected_count for item in operation.last_group_report.values()), 5)

    def test_element_set_fps_rejects_budget_smaller_than_group_count(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            FPSFilterOperation.allocate_sqrt_quotas({("H",): 4, ("He",): 2}, 1)

    def test_element_set_fps_center_start_and_warm_start(self):
        points = np.asarray([[0.0], [4.0], [5.0], [6.0], [10.0]])

        self.assertEqual(
            FPSFilterOperation.centered_fps(points, n_samples=1, min_dist=0.0),
            [2],
        )
        self.assertEqual(
            FPSFilterOperation.centered_fps(
                points,
                n_samples=1,
                min_dist=0.0,
                selected_data=np.asarray([[0.0]]),
            ),
            [4],
        )

    def test_element_set_fps_operation_uses_matching_warm_start(self):
        dataset = [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[1.0, 0.0, 0.0]]),
            Atoms("H", positions=[[2.0, 0.0, 0.0]]),
        ]
        existing_path = self.test_dir / "data" / "nep" / "train.xyz"
        params = FPSFilterParams(
            nep_path=str(self.test_dir / "data" / "nep" / "nep.txt"),
            n_samples=1,
            min_distance=0.0,
            strategy="element_set",
            existing_dataset_path=str(existing_path),
        )
        operation = FPSFilterOperation()

        with (
            patch("NepTrainKit.core.cards.filter.import_structures", return_value=[dataset[0]]),
            patch("NepTrainKit.core.cards.filter.NepCalculator") as calculator_class,
        ):
            calculator_class.return_value.descriptors.side_effect = [
                np.asarray([[0.0], [3.0], [10.0]]),
                np.asarray([[0.0]]),
            ]
            selected = operation.run_dataset(dataset, params)

        self.assertEqual(selected, [dataset[2]])
        report = operation.last_group_report[("H",)]
        self.assertEqual(report.existing_count, 1)
        self.assertEqual(report.selected_count, 1)

    def test_global_fps_can_use_existing_training_set_as_warm_start(self):
        dataset = [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[1.0, 0.0, 0.0]]),
            Atoms("H", positions=[[2.0, 0.0, 0.0]]),
        ]
        existing_path = self.test_dir / "data" / "nep" / "train.xyz"
        params = FPSFilterParams(
            nep_path=str(self.test_dir / "data" / "nep" / "nep.txt"),
            n_samples=1,
            min_distance=0.0,
            strategy="global",
            existing_dataset_path=str(existing_path),
        )
        with (
            patch("NepTrainKit.core.cards.filter.import_structures", return_value=[dataset[0]]),
            patch("NepTrainKit.core.cards.filter.NepCalculator") as calculator_class,
        ):
            calculator_class.return_value.descriptors.side_effect = [
                np.asarray([[0.0], [3.0], [10.0]]),
                np.asarray([[0.0]]),
            ]
            selected = FPSFilterOperation().run_dataset(dataset, params)

        self.assertEqual(selected, [dataset[2]])

    def test_fps_filter_validates_parameters_and_descriptor_matrix(self):
        model_path = str(self.test_dir / "data" / "nep" / "nep.txt")
        dataset = [Atoms("H", positions=[[0, 0, 0]])]
        operation = FPSFilterOperation()

        with self.assertRaisesRegex(ValueError, "n_samples must be an integer"):
            operation.run_dataset(
                dataset,
                FPSFilterParams(nep_path=model_path, n_samples=1.5),
            )
        with self.assertRaisesRegex(ValueError, "min_distance must be >= 0"):
            operation.run_dataset(
                dataset,
                FPSFilterParams(nep_path=model_path, min_distance=-0.1),
            )
        with patch("NepTrainKit.core.cards.filter.NepCalculator") as calculator_class:
            calculator_class.return_value.descriptors.return_value = np.asarray(
                [[float("nan")]]
            )
            with self.assertRaisesRegex(ValueError, "contain NaN/Inf"):
                operation.run_dataset(
                    dataset,
                    FPSFilterParams(nep_path=model_path),
                )

    def test_fps_filter_preview_explains_budget_and_group_quotas(self):
        card = FPSFilterDataCard()
        dataset = [
            Atoms("H", positions=[[0, 0, 0]]),
            Atoms("H", positions=[[1, 0, 0]]),
            Atoms("He", positions=[[0, 0, 0]]),
        ]
        card.set_dataset(dataset)
        self.assertIn("keep at most 3", card.preview_label.text())
        self.assertIn("one global FPS budget", card.preview_label.text())

        card.strategy_combo.setCurrentIndex(
            card.strategy_combo.findData("element_set")
        )
        self.assertIn("H:2", card.preview_label.text())
        self.assertIn("He:1", card.preview_label.text())

        card.strategy_combo.setCurrentIndex(card.strategy_combo.findData("global"))
        card.advanced_button.setChecked(True)
        self.assertFalse(card.existing_dataset_widget.isHidden())

    def test_geometry_filter_operation_and_card_roundtrip(self):
        good = Atoms(
            symbols=["Si", "Si"],
            positions=[[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]],
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=True,
        )
        bad = good.copy()
        bad.positions[1] = [0.5, 0.0, 0.0]

        params = GeometryFilterParams(min_pair_distance=1.0, require_finite_cell=True)
        kept = GeometryFilterOperation().run_dataset([good, bad], params)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.allclose(kept[0].positions, good.positions))
        self.assertFalse(GeometryFilterOperation.has_pair_closer_than(good, 1.0))
        self.assertTrue(GeometryFilterOperation.has_pair_closer_than(bad, 1.0))

        card = GeometryFilterCard()
        self.assertIsInstance(card, FilterDataCard)
        card.min_pair_frame.set_input_value([1.4])
        card.min_vpa_frame.set_input_value([5.0])
        card.max_vpa_frame.set_input_value([80.0])
        card.require_cell_checkbox.setChecked(True)
        restored = GeometryFilterCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_geometry_filter_supports_nonperiodic_molecules_for_pair_only(self):
        molecule = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        )
        operation = GeometryFilterOperation()

        self.assertTrue(
            operation.keep_structure(
                molecule,
                GeometryFilterParams(min_pair_distance=0.5),
            )
        )
        self.assertFalse(
            operation.keep_structure(
                molecule,
                GeometryFilterParams(min_pair_distance=0.8),
            )
        )

    def test_geometry_filter_batch_scan_matches_single_structure_contract(self):
        triclinic_cell = np.array(
            [
                [4.0, 0.0, 0.0],
                [1.2, 3.7, 0.0],
                [0.4, 0.6, 4.2],
            ]
        )
        structures = [
            Atoms(
                "HH",
                positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            ),
            Atoms(
                "HH",
                positions=np.array([[0.01, 0.01, 0.01], [0.99, 0.99, 0.99]])
                @ triclinic_cell,
                cell=triclinic_cell,
                pbc=True,
            ),
            Atoms(
                "HH",
                positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                cell=[5.0, 5.0, 5.0],
                pbc=True,
            ),
        ]
        params = GeometryFilterParams(min_pair_distance=1.0)
        expected = [
            structure
            for structure in structures
            if GeometryFilterOperation.keep_structure(structure, params)
        ]

        actual = GeometryFilterOperation().run_dataset(structures, params)

        self.assertEqual(
            [any(structure is item for item in actual) for structure in structures],
            [any(structure is item for item in expected) for structure in structures],
        )

    def test_geometry_filter_batch_scan_runs_once_and_keeps_exact_cutoff(self):
        structures = [
            Atoms("HH", positions=[[0, 0, 0], [0.5, 0, 0]]),
            Atoms("HH", positions=[[0, 0, 0], [1.0, 0, 0]]),
            Atoms("HH", positions=[[0, 0, 0], [1.5, 0, 0]]),
        ]
        params = GeometryFilterParams(min_pair_distance=1.0)

        with patch(
            "NepTrainKit.core.cards.filter.find_short_distance_structure_rows",
            wraps=find_short_distance_structure_rows,
        ) as scan:
            kept = GeometryFilterOperation().run_dataset(structures, params)

        self.assertEqual(scan.call_count, 1)
        self.assertEqual(
            [any(structure is item for item in kept) for structure in structures],
            [False, True, True],
        )

    def test_geometry_filter_rejects_exact_overlap_and_nonfinite_positions(self):
        overlap = Atoms(
            "HH",
            positions=[[0, 0, 0], [0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        nonfinite = Atoms("H", positions=[[0, 0, 0]])
        nonfinite.positions[0, 0] = np.nan

        self.assertEqual(
            GeometryFilterOperation.shortest_pair_distance(overlap),
            0.0,
        )
        summary = GeometryFilterOperation.filter_summary(
            [overlap, nonfinite],
            GeometryFilterParams(min_pair_distance=0.1),
        )
        self.assertEqual(summary["kept_count"], 0)
        self.assertEqual(summary["reasons"]["pair_distance"], 1)
        self.assertEqual(summary["reasons"]["nonfinite_positions"], 1)

    def test_geometry_filter_rejects_invalid_threshold_ranges(self):
        params = GeometryFilterParams(
            min_volume_per_atom=10,
            max_volume_per_atom=5,
        )
        with self.assertRaisesRegex(ValueError, "minimum volume/atom"):
            GeometryFilterOperation().run_dataset([self.structure], params)

        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            GeometryFilterOperation().run_dataset(
                [self.structure],
                GeometryFilterParams(min_pair_distance=float("nan")),
            )

    def test_geometry_filter_preview_and_progressive_bulk_controls(self):
        good = Atoms(
            "Si2",
            positions=[[0, 0, 0], [2.35, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        overlap = good.copy()
        overlap.positions[1] = overlap.positions[0]
        card = GeometryFilterCard()
        card.set_dataset([good, overlap])

        self.assertTrue(card.min_vpa_frame.isHidden())
        self.assertIn("keep 2", card.preview_label.text())
        card.min_pair_frame.set_input_value([1.0])
        self.assertIn("keep 1", card.preview_label.text())
        self.assertIn("short pairs 1", card.preview_label.text())
        card.bulk_checkbox.setChecked(True)
        self.assertFalse(card.min_vpa_frame.isHidden())
