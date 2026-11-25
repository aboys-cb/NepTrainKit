"""Cross-card defect operations."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestDefectOperations(BaseCardTest):
    def test_vacancy_rule_count_mode_distinguishes_fixed_and_range(self):
        item = VacancyRuleItem()
        item.show()
        QApplication.processEvents()
        self.assertEqual(item.fixed_count_frame.object_list[0].minimum(), 1)
        self.assertEqual(item.count_range_frame.object_list[0].minimum(), 0)
        self.assertTrue(item.fixed_count_frame.isVisible())
        self.assertFalse(item.count_range_frame.isVisible())
        item.element_edit.setText("Si")
        item.fixed_count_frame.set_input_value([2])
        fixed = item.to_rule()
        self.assertEqual(fixed["count"], [2, 2])
        self.assertEqual(fixed["count_mode"], "fixed")

        item.count_mode_combo.setCurrentText("Random range")
        QApplication.processEvents()
        self.assertFalse(item.fixed_count_frame.isVisible())
        self.assertTrue(item.count_range_frame.isVisible())
        self.assertTrue(all(spin.isVisible() for spin in item.count_range_frame.object_list))
        item.count_range_frame.set_input_value([1, 3])
        ranged = item.to_rule()
        self.assertEqual(ranged["count"], [1, 3])
        self.assertEqual(ranged["count_mode"], "random")

    def test_defect_surface_operations_are_ui_independent(self):
        structure = self.structure.copy()

        vac_results = RandomVacancyOperation().run_structure(
            structure,
            RandomVacancyParams(
                rules=[{"element": "Si", "count": [1, 1], "count_mode": "fixed"}],
                max_structures=2,
                use_seed=True,
                seed=3,
            ),
        )
        self.assertEqual(len(vac_results), 2)
        self.assertTrue(all(len(atoms) == len(structure) - 1 for atoms in vac_results))

        defect_results = VacancyDefectOperation().run_structure(
            structure,
            VacancyDefectParams(
                engine_type=1,
                use_num=False,
                concentration_condition=0.6,
                count_mode="random",
                max_structures=2,
                use_seed=True,
                seed=5,
            ),
        )
        self.assertEqual(len(defect_results), 2)
        self.assertTrue(all(len(atoms) < len(structure) for atoms in defect_results))

        fault_results = StackingFaultOperation().run_structure(
            structure,
            StackingFaultParams(hkl=(1, 1, 1), step=(0.1, 0.1, 0.1), layers=1),
        )
        self.assertEqual(len(fault_results), 1)
        self.assertGreater(
            np.abs(fault_results[0].get_positions() - structure.get_positions()).sum(),
            0.0,
        )
        self.assertIn("SF(", fault_results[0].info.get("Config_type", ""))

        slab_results = RandomSlabOperation().run_structure(
            structure,
            RandomSlabParams(
                hkl_list=((1, 0, 0),),
                layer_range=(1, 1, 1),
                vacuum_range=(10, 10, 1),
            ),
        )
        self.assertEqual(len(slab_results), 1)
        self.assertGreaterEqual(len(slab_results[0]), len(structure))
        self.assertIn("Slab(", slab_results[0].info.get("Config_type", ""))

        insert_results = InsertDefectOperation().run_structure(
            structure,
            InsertDefectParams(
                mode=0,
                species="H",
                insert_count=1,
                structure_count=1,
                min_distance=0.1,
                max_attempts=20,
                use_seed=True,
                seed=7,
            ),
        )
        self.assertEqual(len(insert_results), 1)
        self.assertEqual(len(insert_results[0]), len(structure) + 1)
        self.assertIn("Ins(int", insert_results[0].info.get("Config_type", ""))

