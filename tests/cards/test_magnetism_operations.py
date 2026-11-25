"""Cross-card magnetism operations."""

import json
import tempfile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox

from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.magnetism import parse_magmom_map_any
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value

from .magnetism_test_base import *


class TestMagnetismOperations(MagnetismCardTest):
    def test_magnetic_operations_are_ui_independent(self):
        structure = self._spin_chain()

        set_result = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Map/default magnitude",
                format="Non-collinear (vector)",
                magmom_map="Fe:2.0",
                axis=[0.0, 0.0, 1.0],
            ),
        )[0]
        self.assertIn("MagSet(map,vec)", str(set_result.info.get("Config_type", "")))

        order_result = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(magmom_map="Fe:2.0", gen_fm=True, gen_afm=False),
        )[0]
        self.assertIn("MagFM", str(order_result.info.get("Config_type", "")))

        spiral_result = SpinSpiralOperation().run_structure(
            set_result,
            SpinSpiralParams(
                period_range=[4.0, 4.0, 1.0],
                phase_range=[0.0, 0.0, 15.0],
                chirality="Clockwise",
                max_outputs=1,
            ),
        )[0]
        self.assertIn("Helix(", str(spiral_result.info.get("Config_type", "")))

        folded = FoldedHelixOperation().run_structure(
            set_result,
            FoldedHelixParams(
                half_period_mode="Manual",
                half_period_layers=[2, 2, 1],
                angle_step_range=[30.0, 30.0, 15.0],
                phase_range=[0.0, 0.0, 15.0],
                max_outputs=1,
            ),
        )[0]
        self.assertIn("FoldedHelix(", str(folded.info.get("Config_type", "")))

        tilt_result = SmallAngleSpinTiltOperation().run_structure(
            set_result,
            SmallAngleSpinTiltParams(
                target_mode="Explicit indices (1-based)",
                target_indices="2",
                angle_list="5",
                include_reference=False,
            ),
        )[0]
        self.assertIn("SpinTilt(i=2,a=5", str(tilt_result.info.get("Config_type", "")))

        rotated = MagneticMomentRotationOperation().run_structure(
            set_result,
            MagneticMomentRotationParams(
                elements="Fe",
                max_angle=10.0,
                num_structures=1,
                use_seed=True,
                seed=7,
            ),
        )[0]
        moments = np.array(rotated.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(moments.shape, (4, 3))
        for output in (
            set_result,
            order_result,
            spiral_result,
            folded,
            tilt_result,
            rotated,
        ):
            self.assertIn("spin", output.arrays)
            self.assertEqual(np.asarray(output.arrays["spin"]).shape, (4, 3))
            initial = np.asarray(output.get_initial_magnetic_moments())
            if initial.ndim == 2:
                self.assertTrue(
                    np.allclose(output.arrays["spin"], initial, atol=1e-12)
                )

    def test_set_and_order_magnetic_maps_honor_defaults_directions_and_element_scope(self):
        structure = Atoms(
            ["Fe", "Ni", "Fe", "Ni"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            cell=[6.0, 6.0, 8.0],
            pbc=[False, False, True],
        )
        mapped = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Map/default magnitude",
                format="Non-collinear (vector)",
                axis=(0.0, 0.0, 1.0),
                magmom_map="Fe:[2,0,0]",
                use_element_dirs=True,
                default_moment=1.5,
                apply_elements="Fe,Ni",
            ),
        )[0]
        np.testing.assert_allclose(
            mapped.arrays["spin"],
            [
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.5],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.5],
            ],
            atol=1e-12,
        )

        scoped = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Map/default magnitude",
                format="Non-collinear (vector)",
                magmom_map="Fe:2",
                default_moment=9.0,
                apply_elements="Fe",
            ),
        )[0]
        np.testing.assert_allclose(
            np.linalg.norm(scoped.arrays["spin"], axis=1),
            [2.0, 0.0, 2.0, 0.0],
            atol=1e-12,
        )

        ordered = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(
                format="noncollinear",
                magmom_map="Fe:[0,2,0]",
                use_element_dirs=True,
                default_moment=1.0,
                apply_elements="Fe",
                gen_fm=True,
                gen_afm=False,
            ),
        )[0]
        np.testing.assert_allclose(
            ordered.arrays["spin"],
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            atol=1e-12,
        )

