"""Magnetic moment rotation card: axis, magnitude bounds, seed provenance."""

import json
import tempfile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox

from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.magnetism import parse_magmom_map_any
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value

from .magnetism_test_base import *


class TestMagmomRotationCards(MagnetismCardTest):
    def test_rotation_reads_spin_without_legacy_initial_magmoms(self):
        structure = self._spin_chain()
        canonical = np.tile([0.0, 0.0, 2.0], (len(structure), 1))
        structure.set_array("spin", canonical)

        result = MagneticMomentRotationOperation().run_structure(
            structure,
            MagneticMomentRotationParams(
                max_angle=10.0,
                num_structures=1,
                lift_scalar=False,
                use_seed=True,
                seed=71,
            ),
        )[0]

        self.assertEqual(result.get_initial_magnetic_moments().shape, (4, 3))
        self.assertFalse(np.allclose(result.arrays["spin"], canonical))
        self.assertTrue(
            np.allclose(
                result.arrays["spin"],
                result.get_initial_magnetic_moments(),
                atol=1e-12,
            )
        )

    def test_magmom_rotation_lifts_scalar_to_vector(self):
        proto = CrystalPrototypeBuilderCard()
        set_combo_value(proto.structure_combo, "bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0].repeat((2, 2, 2))

        order = MagneticOrderCard()
        order.format_combo.setCurrentIndex(0)
        order.map_edit.setText("Fe:2.2")
        order.fm_checkbox.setChecked(True)
        order.afm_checkbox.setChecked(False)
        order.pm_checkbox.setChecked(False)
        fm = order.process_structure(base)[0]

        rot = MagneticMomentRotationCard()
        rot.elements_input.setText("Fe")
        rot.angle_frame.set_input_value([45.0])
        rot.lift_scalar_checkbox.setChecked(True)
        rot.axis_frame.set_input_value([0.0, 0.0, 1.0])
        rot.count_frame.set_input_value([1])

        rotated = rot.process_structure(fm)[0]
        m = np.array(rotated.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(m.ndim, 2)
        self.assertEqual(m.shape[1], 3)

    def test_magmom_rotation_seed_reproducibility_and_magnitude_bounds(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        params = MagneticMomentRotationParams(
            max_angle=10.0,
            num_structures=3,
            disturb_magnitude=True,
            magnitude_factor=(0.9, 1.1),
            use_seed=True,
            seed=21,
        )

        first = MagneticMomentRotationOperation().run_structure(structure, params)
        second = MagneticMomentRotationOperation().run_structure(structure, params)

        for a, b in zip(first, second):
            ma = np.array(a.get_initial_magnetic_moments(), dtype=float)
            mb = np.array(b.get_initial_magnetic_moments(), dtype=float)
            self.assertTrue(np.allclose(ma, mb, atol=1e-12))
            self.assertEqual(ma.shape, (4, 3))
            norms = np.linalg.norm(ma, axis=1)
            self.assertTrue(np.all(norms >= 1.8 - 1e-12))
            self.assertTrue(np.all(norms <= 2.2 + 1e-12))
            self.assertIn("SpinPert(", a.info.get("Config_type", ""))

    def test_magmom_rotation_defaults_cover_broad_training_perturbations(self):
        params = MagneticMomentRotationParams()
        self.assertEqual(params.max_angle, 30.0)
        self.assertEqual(tuple(params.magnitude_factor), (0.8, 1.2))

        card = MagneticMomentRotationCard()
        card_params = card.get_params()
        self.assertEqual(card_params.max_angle, params.max_angle)
        self.assertEqual(tuple(card_params.magnitude_factor), tuple(params.magnitude_factor))

    def test_random_spin_perturbation_samples_uniform_spherical_cap(self):
        vectors = np.tile(np.array([[0.0, 0.0, 2.0]]), (20000, 1))
        sampled, angles = MagneticMomentRotationOperation.sample_spherical_cap(
            vectors,
            30.0,
            np.random.default_rng(19),
        )

        np.testing.assert_allclose(np.linalg.norm(sampled, axis=1), 2.0, atol=1e-12)
        self.assertGreaterEqual(float(np.min(angles)), 0.0)
        self.assertLessEqual(float(np.max(angles)), 30.0 + 1e-12)
        expected_mean_cosine = 0.5 * (1.0 + np.cos(np.deg2rad(30.0)))
        actual_mean_cosine = float(np.mean(sampled[:, 2] / 2.0))
        self.assertAlmostEqual(actual_mean_cosine, expected_mean_cosine, delta=0.002)

    def test_magmom_rotation_card_roundtrips_selection_axis_and_magnitude_controls(self):
        params = MagneticMomentRotationParams(
            elements="Fe,Ni",
            max_angle=17.5,
            num_structures=6,
            lift_scalar=False,
            axis=(1.0, 1.0, 0.0),
            disturb_magnitude=False,
            magnitude_factor=(0.8, 1.2),
            use_seed=True,
            seed=67,
        )
        card = MagneticMomentRotationCard()
        card.set_params(params)
        normalized_params = card.get_params()
        restored = MagneticMomentRotationCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), normalized_params)
        self.assertTrue(card.axis_field.isHidden())
        self.assertTrue(card.magnitude_factor_field.isHidden())
        self.assertEqual(card.get_summary_text(), "≤ 17.5° · Fe,Ni · 6/input")

        preview = MagneticMomentRotationCard()
        preview.set_preview_input_count(2)
        self.assertIn("Planned (valid input): 2 × 5 = 10 outputs", preview.get_guidance_text())

    def test_magmom_rotation_fails_closed_for_inputs_that_cannot_change(self):
        operation = MagneticMomentRotationOperation()
        plain = self._spin_chain()
        with self.assertRaisesRegex(ValueError, "requires spin or initial magnetic moments"):
            operation.run_structure(plain, MagneticMomentRotationParams())

        scalar = self._spin_chain()
        scalar.set_initial_magnetic_moments([2.0] * len(scalar))
        with self.assertRaisesRegex(ValueError, "must be lifted to vectors"):
            operation.run_structure(
                scalar,
                MagneticMomentRotationParams(lift_scalar=False),
            )
        with self.assertRaisesRegex(ValueError, "non-zero"):
            operation.run_structure(
                scalar,
                MagneticMomentRotationParams(axis=(0.0, 0.0, 0.0)),
            )
        with self.assertRaisesRegex(ValueError, "No non-zero magnetic moments match"):
            operation.run_structure(
                scalar,
                MagneticMomentRotationParams(elements="Ni"),
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 180"):
            operation.run_structure(
                scalar,
                MagneticMomentRotationParams(max_angle=-1.0),
            )
        with self.assertRaisesRegex(ValueError, "Increase the perturbation angle"):
            operation.run_structure(
                scalar,
                MagneticMomentRotationParams(
                    max_angle=0.0,
                    disturb_magnitude=False,
                ),
            )
        with self.assertRaisesRegex(ValueError, "minimum must not exceed"):
            operation.run_structure(
                scalar,
                MagneticMomentRotationParams(magnitude_factor=(1.2, 0.8)),
            )
        rotation_only = operation.run_structure(
            scalar,
            MagneticMomentRotationParams(
                num_structures=1,
                disturb_magnitude=False,
                magnitude_factor=(1.2, 0.8),
                use_seed=True,
                seed=9,
            ),
        )
        self.assertEqual(len(rotation_only), 1)

    def test_magmom_rotation_uses_structure_seed_and_records_provenance(self):
        first_structure = self._spin_chain()
        first_structure.set_initial_magnetic_moments([2.0] * len(first_structure))
        second_structure = first_structure.copy()
        second_structure.positions[0, 0] += 0.2
        params = MagneticMomentRotationParams(
            max_angle=15.0,
            num_structures=2,
            disturb_magnitude=False,
            use_seed=True,
            seed=23,
        )

        first = MagneticMomentRotationOperation().run_structure(first_structure, params)
        repeated = MagneticMomentRotationOperation().run_structure(first_structure, params)
        second = MagneticMomentRotationOperation().run_structure(second_structure, params)

        for left, right in zip(first, repeated):
            np.testing.assert_allclose(left.arrays["spin"], right.arrays["spin"])
        self.assertFalse(np.allclose(first[0].arrays["spin"], second[0].arrays["spin"]))

        metadata = json.loads(first[0].info["spin_perturbation"])
        self.assertEqual(metadata["direction_distribution"], "uniform_spherical_cap")
        self.assertEqual(metadata["seed"], 23)
        self.assertEqual(metadata["sample_index"], 0)
        self.assertEqual(metadata["selected_count"], len(first_structure))
        self.assertFalse(metadata["disturb_magnitude"])
        self.assertGreater(metadata["realized_angle_max"], 0.0)
