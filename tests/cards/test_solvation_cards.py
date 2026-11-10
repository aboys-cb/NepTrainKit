"""Solvation card coverage."""

from .card_test_base import *
import threading
import time
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

from ase.geometry import find_mic

from NepTrainKit.core.alloy import best_supercell_factors_max_atoms
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return bool(predicate())


class TestSolvationCards(BaseCardTest):
    def test_solvation_cards_roundtrip_all_flexible_solvent_controls(self):
        local_params = LocalSolvationParams(
            structures=2,
            solvent_count=3,
            sampling_mode="loose",
            center_mode="indices",
            center_indices="1,3",
            shell=(2.0, 3.5),
            min_distance=0.7,
            collision_scale=0.8,
            max_attempts=1234,
            strict_count=False,
            auto_box=True,
            box_size=80.0,
            box_padding=7.0,
            min_box=25.0,
            flex_solvent=True,
            flex_pool=7,
            flex_torsion_range=(-75.0, 95.0),
            flex_max_torsions=3,
            flex_gaussian_sigma=0.02,
            use_seed=True,
            seed=41,
        )
        local = LocalSolvationCard()
        local.set_params(local_params)
        restored_local = LocalSolvationCard()
        restored_local.from_dict(local.to_dict())
        self.assertEqual(restored_local.get_params(), local_params)

        box_params = SolventBoxFillParams(
            structures=2,
            count_mode="density",
            solvent_count=8,
            density=0.75,
            sampling_mode="loose",
            fill_packing=0.85,
            min_distance=0.65,
            collision_scale=0.78,
            max_attempts_per_solvent=321,
            strict_count=False,
            flex_solvent=True,
            flex_pool=9,
            flex_torsion_range=(-65.0, 85.0),
            flex_max_torsions=4,
            flex_gaussian_sigma=0.025,
            use_seed=True,
            seed=43,
        )
        box = SolventBoxFillCard()
        box.set_params(box_params)
        restored_box = SolventBoxFillCard()
        restored_box.from_dict(box.to_dict())
        self.assertEqual(restored_box.get_params(), box_params)

