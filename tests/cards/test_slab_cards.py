"""Surface slab and random slab cards."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestSlabCards(BaseCardTest):
    def test_random_slab_card(self):
        card = RandomSlabCard()
        structure = self.structure.copy()
        card.plane_table.set_planes([(1, 0, 0)])
        card.layer_frame.set_input_value([1, 1, 1])
        card.vacuum_frame.set_input_value([10, 10, 1])

        results = card.process_structure(structure)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(len(atoms) >= len(structure) for atoms in results))

    def test_random_slab_plane_table_shows_all_short_rows_without_inner_scroll(self):
        card = RandomSlabCard()
        card.plane_table.set_planes([(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)])

        table = card.plane_table.table
        expected_minimum = table.horizontalHeader().height() + 4 * 30
        self.assertGreaterEqual(table.height(), expected_minimum)
        self.assertEqual([table.rowHeight(row) for row in range(4)], [30, 30, 30, 30])
        self.assertEqual(
            table.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

    def test_random_slab_operation_enumerates_layer_vacuum_product_and_roundtrips(self):
        structure = self.structure.copy()
        structure.info["Config_type"] = "bulk"
        params = RandomSlabParams(
            hkl_list=((1, 0, 0),),
            layer_range=(1, 2, 1),
            vacuum_range=(0.0, 2.0, 2.0),
        )

        results = RandomSlabOperation().run_structure(structure, params)

        self.assertEqual(len(results), 4)
        tags = [atoms.info.get("Config_type", "") for atoms in results]
        self.assertTrue(all("Slab(hkl=(1,0,0)" in tag for tag in tags))
        self.assertEqual(sum("R=1" in tag for tag in tags), 2)
        self.assertEqual(sum("R=2" in tag for tag in tags), 2)
        self.assertEqual(sum("vac=0" in tag for tag in tags), 2)
        self.assertEqual(sum("vac=2," in tag for tag in tags), 2)
        self.assertTrue(all(np.asarray(atoms.pbc, dtype=bool).all() for atoms in results))

        card = RandomSlabCard()
        card.set_params(params)
        restored = RandomSlabCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)

    def test_surface_slab_default_plan_is_exact_and_allocation_free(self):
        structure = Atoms(
            "Si8",
            positions=np.arange(24, dtype=float).reshape(8, 3) * 0.1,
            cell=np.asarray([[5.4, 0.0, 0.0], [0.7, 5.2, 0.0], [0.3, 0.4, 5.8]]),
            pbc=True,
        )
        operation = RandomSlabOperation()
        with patch("NepTrainKit.core.cards.defect.surface") as build_surface:
            plan = operation.plan(structure, RandomSlabParams())

        build_surface.assert_not_called()
        self.assertEqual(plan.hkl_list, ((1, 0, 0), (1, 1, 0), (1, 1, 1)))
        self.assertEqual(plan.repeats, (3, 4, 5, 6))
        self.assertEqual(plan.vacuums, (10.0,))
        self.assertEqual(plan.outputs, 12)
        self.assertEqual((plan.min_atoms_per_output, plan.max_atoms_per_output), (24, 48))
        self.assertEqual(plan.generated_atoms, 432)

    def test_surface_slab_reduces_and_deduplicates_hkl_without_merging_opposite_planes(self):
        canonical = RandomSlabOperation.canonical_hkl_list(
            ((0, 0, 1), (0, 0, 2), (0, 0, -1), (2, 2, 0), (1, 1, 0))
        )
        self.assertEqual(canonical, ((0, 0, 1), (0, 0, -1), (1, 1, 0)))

    def test_surface_slab_ranges_do_not_overshoot_stop(self):
        operation = RandomSlabOperation()
        self.assertEqual(
            operation._inclusive_integer_range(
                (1, 2, 3), code="test", label="repeat", minimum=1
            ),
            (1,),
        )
        self.assertEqual(
            operation._inclusive_float_range(
                (10.0, 11.0, 2.0), code="test", label="vacuum", minimum=0.0
            ),
            (10.0,),
        )

    def test_surface_slab_vacuum_is_per_side_and_normal_pbc_is_explicit(self):
        structure = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)
        params = RandomSlabParams(
            hkl_list=((0, 0, 1),),
            layer_range=(2, 2, 1),
            vacuum_range=(7.5, 7.5, 1.0),
            normal_pbc=False,
        )
        slab = RandomSlabOperation().run_structure(structure, params)[0]
        z_span = float(np.ptp(slab.positions[:, 2]))

        self.assertTrue(np.array_equal(np.asarray(slab.pbc, dtype=bool), [True, True, False]))
        self.assertAlmostEqual(float(slab.cell.lengths()[2]) - z_span, 15.0, places=10)
        self.assertIn("pz=0", slab.info["Config_type"])

    def test_surface_slab_repeats_site_arrays_and_preserves_structure_info(self):
        structure = Atoms("Fe2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
        structure.arrays["spin"] = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        structure.arrays["group"] = np.asarray([3, 4])
        structure.info.update({"Config_type": "bulk", "source": "fixture"})
        params = RandomSlabParams(
            hkl_list=((0, 0, 1),),
            layer_range=(2, 2, 1),
            vacuum_range=(5.0, 5.0, 1.0),
        )

        slab = RandomSlabOperation().run_structure(structure, params)[0]

        np.testing.assert_allclose(slab.arrays["spin"], np.tile(structure.arrays["spin"], (2, 1)))
        np.testing.assert_array_equal(slab.arrays["group"], np.tile(structure.arrays["group"], 2))
        self.assertEqual(slab.info["source"], "fixture")
        self.assertTrue(slab.info["Config_type"].startswith("bulk|Slab("))

    def test_surface_slab_rejects_invalid_input_and_budgets_before_building(self):
        structure = self.structure.copy()
        operation = RandomSlabOperation()
        cases = (
            (structure.copy(), RandomSlabParams(hkl_list=((0, 0, 0),))),
            (structure.copy(), RandomSlabParams(normal_pbc=False, vacuum_range=(0.0, 0.0, 1.0))),
            (structure.copy(), RandomSlabParams(max_outputs=1)),
            (structure.copy(), RandomSlabParams(max_generated_atoms=1)),
        )
        cases[0][0].pbc = (True, True, False)
        with patch("NepTrainKit.core.cards.defect.surface") as build_surface:
            for candidate, params in cases:
                with self.subTest(params=params), self.assertRaises(CardOperationError):
                    operation.run_structure(candidate, params)
        build_surface.assert_not_called()

    def test_surface_slab_build_failure_is_visible_not_partial(self):
        structure = self.structure.copy()
        params = RandomSlabParams(
            hkl_list=((1, 0, 0), (1, 1, 0)),
            layer_range=(1, 1, 1),
            vacuum_range=(10.0, 10.0, 1.0),
        )
        with patch("NepTrainKit.core.cards.defect.surface", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(CardOperationError, "boom"):
                RandomSlabOperation().run_structure(structure, params)

    def test_surface_slab_table_and_legacy_range_migration_are_user_visible(self):
        card = RandomSlabCard()
        card.plane_table.set_planes(((0, 0, 1), (0, 0, 2), (1, 1, 0)))
        card.plane_table._normalize_after_edit()
        self.assertEqual(card.plane_table.planes(), ((0, 0, 1), (1, 1, 0)))

        restored = RandomSlabCard()
        with patch("NepTrainKit.ui.views._card.random_slab_card.MessageManager.send_warning_message"):
            restored.from_dict(
                {
                    "class": "RandomSlabCard",
                    "check_state": True,
                    "params": {
                        "h_range": [0, 1, 2],
                        "k_range": [0, 0, 1],
                        "l_range": [1, 2, 1],
                        "layer_range": [2, 2, 1],
                        "vacuum_range": [10, 11, 2],
                    },
                }
            )
        self.assertEqual(restored.get_params().hkl_list, ((0, 0, 1),))
        self.assertEqual(restored.get_params().vacuum_range, (10.0, 11.0, 2.0))
        self.assertTrue(restored.legacy_notice.isVisibleTo(restored.legacy_notice.parentWidget()))
        self.assertIn("converted", restored.legacy_notice.text())

    def test_surface_slab_table_removes_focused_row_without_teardown_error(self):
        card = RandomSlabCard()
        card.plane_table.set_planes(((1, 1, 1),))
        card.plane_table._add_clicked()
        QApplication.processEvents()

        with patch("sys.excepthook") as excepthook:
            card.plane_table._remove_selected()
            QApplication.processEvents()

        excepthook.assert_not_called()
        self.assertEqual(card.plane_table.planes(), ((1, 1, 1),))

