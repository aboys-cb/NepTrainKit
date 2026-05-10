from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox

from .card_test_base import (
    BaseCardTest,
    CellScalingCard,
    MagneticMomentRotationCard,
    OrganicMolConfigPBCCard,
    PerturbCard,
    RandomSlabCard,
    StackingFaultCard,
    VacancyDefectCard,
    VibrationModePerturbCard,
)


def _double_spin(frame, index: int = 0) -> QDoubleSpinBox:
    spin = frame.object_list[index]
    assert isinstance(spin, QDoubleSpinBox)
    return spin


class TestUINumericPrecision(BaseCardTest):
    def test_atomic_perturb_distance_controls_keep_sub_milliangstrom_input(self):
        card = PerturbCard()

        assert _double_spin(card.scaling_condition_frame).decimals() >= 4
        card.scaling_condition_frame.set_input_value([0.0123])
        self.assertAlmostEqual(card.get_params().max_distance, 0.0123)

        row = card._add_element_row("H", 0.0456)
        assert _double_spin(row.distance_frame).decimals() >= 4
        self.assertAlmostEqual(row.get_value()[1], 0.0456)

    def test_fractional_controls_keep_fine_fraction_inputs(self):
        scaling = CellScalingCard()
        assert _double_spin(scaling.scaling_condition_frame).decimals() >= 6
        scaling.scaling_condition_frame.set_input_value([0.012345])
        self.assertAlmostEqual(scaling.get_params().max_scaling, 0.012345)

        vacancy = VacancyDefectCard()
        assert _double_spin(vacancy.concentration_condition_frame).decimals() >= 6
        vacancy.concentration_condition_frame.set_input_value([0.012345])
        self.assertAlmostEqual(vacancy.get_params().concentration_condition, 0.012345)

    def test_random_slab_vacuum_range_is_continuous(self):
        card = RandomSlabCard()
        for spin in card.vacuum_frame.object_list:
            assert isinstance(spin, QDoubleSpinBox)
            assert spin.decimals() >= 3

        card.vacuum_frame.set_input_value([8.5, 12.5, 0.5])
        self.assertEqual(card.get_params().vacuum_range, (8.5, 12.5, 0.5))

    def test_stacking_fault_displacement_range_is_continuous(self):
        card = StackingFaultCard()
        for spin in card.step_frame.object_list:
            assert isinstance(spin, QDoubleSpinBox)
            assert spin.decimals() >= 4

        card.step_frame.set_input_value([0.0, 0.25, 0.125])
        self.assertEqual(card.get_params().step, (0.0, 0.25, 0.125))

    def test_vibration_controls_keep_small_amplitudes(self):
        card = VibrationModePerturbCard()

        assert _double_spin(card.amplitude_frame).decimals() >= 4
        card.amplitude_frame.set_input_value([0.0123])
        self.assertAlmostEqual(card.get_params().amplitude, 0.0123)
        assert _double_spin(card.min_freq_frame).decimals() >= 3

    def test_organic_guard_controls_use_physical_precision(self):
        card = OrganicMolConfigPBCCard()

        expected = {
            "torsion_frame": 3,
            "sigma_frame": 4,
            "bond_detect_frame": 4,
            "bond_min_frame": 4,
            "bo_c_frame": 4,
            "bo_thr_frame": 6,
            "bond_max_frame": 4,
            "nonbond_min_frame": 4,
            "multbond_frame": 4,
            "box_frame": 3,
        }
        for attr, decimals in expected.items():
            assert _double_spin(getattr(card, attr)).decimals() >= decimals

    def test_rotation_axis_uses_vector_precision(self):
        card = MagneticMomentRotationCard()

        for spin in card.axis_frame.object_list:
            assert isinstance(spin, QDoubleSpinBox)
            assert spin.decimals() >= 6
