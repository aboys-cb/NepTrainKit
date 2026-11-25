"""Group label card."""

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


class TestGroupLabelCards(BaseCardTest):
    def test_group_label_card_and_group_afm(self):
        proto = CrystalPrototypeBuilderCard()
        set_combo_value(proto.structure_combo, "bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0].repeat((2, 2, 2))

        gl = GroupLabelCard()
        set_combo_value(gl.plane_combo, "111")
        gl.group_a_edit.setText("A")
        gl.group_b_edit.setText("B")
        labeled = gl.process_structure(base)[0]
        self.assertIn("group", labeled.arrays)
        groups = set(str(g) for g in labeled.arrays["group"])
        self.assertTrue({"A", "B"}.issubset(groups))

        mag = MagneticOrderCard()
        mag.format_combo.setCurrentIndex(0)
        mag.map_edit.setText("Fe:2.2")
        mag.fm_checkbox.setChecked(False)
        mag.afm_checkbox.setChecked(True)
        mag.afm_mode_combo.setCurrentIndex(1)
        mag.group_a_edit.setText("A")
        mag.group_b_edit.setText("B")
        mag.pm_checkbox.setChecked(False)
        res = mag.process_structure(labeled)
        self.assertEqual(len(res), 1)
        afm = res[0]
        m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.any(m > 0) and np.any(m < 0))

    def test_group_label_default_handles_a_common_fcc_conventional_cell(self):
        structure = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams()
        )[0]
        card = GroupLabelCard()

        self.assertEqual(card.get_params().miller_index, "001")
        labeled = card.create_operation().run_structure(
            structure, card.get_params()
        )[0]

        self.assertEqual(
            set(str(value) for value in labeled.arrays["group"]),
            {"A", "B"},
        )

    def test_group_label_operation_is_ui_independent(self):
        base = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(
                lattice="bcc",
                element="Fe",
                a_range=(2.9, 2.9, 0.1),
                max_outputs=1,
            )
        )[0].repeat((2, 2, 2))

        labeled = GroupLabelOperation().run_structure(
            base,
            GroupLabelParams(
                miller_index="111",
                group_a="up",
                group_b="down",
            ),
        )[0]

        self.assertIn("group", labeled.arrays)
        self.assertEqual(set(str(value) for value in labeled.arrays["group"]), {"up", "down"})
        self.assertIn("Grp(hkl111,tol=0.05,up/down)", labeled.info.get("Config_type", ""))

    def test_group_label_alternates_detected_layers_after_expansion(self):
        base = Atoms(
            symbols=["Fe"] * 4,
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            cell=np.diag([4.0, 2.0, 2.0]),
            pbc=True,
        )
        params = GroupLabelParams(miller_index="100")
        labeled = GroupLabelOperation().run_structure(base, params)[0]
        self.assertEqual(list(labeled.arrays["group"]), ["A", "B", "A", "B"])

        expanded = base.repeat((2, 1, 1))
        repeated = GroupLabelOperation().run_structure(expanded, params)[0]
        self.assertEqual(
            list(repeated.arrays["group"]),
            ["A", "B", "A", "B", "A", "B", "A", "B"],
        )

    def test_group_label_uses_reciprocal_normal_in_nonorthogonal_cell(self):
        cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.0, 0.0], [0.4, 0.7, 5.0]])
        base = Atoms(
            symbols=["Fe"] * 4,
            scaled_positions=[[0.0, 0.2, 0.3], [0.25, 0.2, 0.3], [0.5, 0.2, 0.3], [0.75, 0.2, 0.3]],
            cell=cell,
            pbc=[True, True, False],
        )
        base.new_array("foo", np.arange(4))
        labeled = GroupLabelOperation().run_structure(
            base,
            GroupLabelParams(miller_index="100", layer_tolerance=0.05),
        )[0]
        self.assertEqual(list(labeled.arrays["group"]), ["A", "B", "A", "B"])
        np.testing.assert_allclose(labeled.cell.array, base.cell.array)
        np.testing.assert_allclose(labeled.positions, base.positions)
        np.testing.assert_array_equal(labeled.arrays["foo"], base.arrays["foo"])

    def test_group_label_merges_a_thermally_split_periodic_boundary_layer(self):
        base = Atoms(
            symbols=["Fe"] * 5,
            scaled_positions=[
                [0.005, 0.0, 0.0],
                [0.995, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [0.75, 0.0, 0.0],
            ],
            cell=np.diag([4.0, 2.0, 2.0]),
            pbc=True,
        )
        operation = GroupLabelOperation()
        layer_ids = operation.layer_ids(base, "100", 0.05)
        self.assertEqual(int(layer_ids.max()) + 1, 4)
        self.assertEqual(layer_ids[0], layer_ids[1])
        labeled = operation.run_structure(
            base,
            GroupLabelParams(miller_index="100", layer_tolerance=0.05),
        )[0]
        self.assertEqual(list(labeled.arrays["group"]), ["A", "A", "B", "A", "B"])

    def test_group_label_counts_periodic_110_layers_without_parallelepiped_duplicates(self):
        base = Atoms(
            symbols=["Fe"] * 4,
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([4.0, 4.0, 2.0]),
            pbc=True,
        )
        layer_ids = GroupLabelOperation().layer_ids(base, "110", 0.05)
        self.assertEqual(int(layer_ids.max()) + 1, 2)
        self.assertEqual(list(layer_ids), [0, 1, 1, 0])

    def test_group_label_rejects_invalid_settings_and_single_layer(self):
        base = self.structure.copy()
        operation = GroupLabelOperation()
        with self.assertRaisesRegex(ValueError, "Plane index"):
            operation.run_structure(base, GroupLabelParams(miller_index="112"))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            operation.run_structure(base, GroupLabelParams(layer_tolerance=0.0))
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            operation.run_structure(base, GroupLabelParams(group_a=""))
        with self.assertRaisesRegex(ValueError, "must be different"):
            operation.run_structure(base, GroupLabelParams(group_a="same", group_b="same"))

        single_layer = Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
        with self.assertRaisesRegex(ValueError, "at least two detected atomic layers"):
            operation.run_structure(single_layer, GroupLabelParams())

    def test_group_label_default_preserves_existing_groups(self):
        base = self.structure.copy()
        base.new_array(
            "group",
            np.asarray(["surface"] * len(base), dtype=object),
        )
        base.info["Config_type"] = "seed"
        params = GroupLabelParams()
        self.assertFalse(params.overwrite)
        output = GroupLabelOperation().run_structure(base, params)[0]
        self.assertIsNot(output, base)
        np.testing.assert_array_equal(output.arrays["group"], base.arrays["group"])
        self.assertEqual(output.info["Config_type"], "seed")

    def test_group_label_card_roundtrip(self):
        card = GroupLabelCard()
        set_combo_value(card.plane_combo, "110")
        card.tolerance_frame.set_input_value([0.08])
        card.group_a_edit.setText("alpha")
        card.group_b_edit.setText("beta")
        card.overwrite_checkbox.setChecked(False)

        restored = GroupLabelCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
        self.assertEqual(restored.get_params().miller_index, "110")
        self.assertAlmostEqual(restored.get_params().layer_tolerance, 0.08)

        legacy = GroupLabelCard()
        with patch(
            "NepTrainKit.ui.views._card.group_label_card.MessageManager.send_warning_message"
        ) as warning:
            legacy.from_dict(
                {
                    "class": "GroupLabelCard",
                    "check_state": True,
                    "mode": "k-vector layers (recommended)",
                    "kvec": "100",
                    "group_a": "old_a",
                    "group_b": "old_b",
                }
            )
        warning.assert_called_once()
        self.assertFalse(legacy.legacy_notice.isHidden())
        self.assertIn("real atomic layers", legacy.legacy_notice.text())
        self.assertEqual(legacy.get_params().miller_index, "100")
        self.assertTrue(legacy.get_params().overwrite)

    def test_group_label_preview_shows_layer_sequence_and_existing_labels(self):
        fcc = Atoms(
            symbols=["Ni"] * 4,
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.eye(3),
            pbc=True,
        )
        card = GroupLabelCard()
        card.set_preview_structure(fcc)
        card.set_preview_input_count(2)
        self.assertIn("2 layers", card.preview_label.text())
        self.assertIn("A(2)", card.preview_label.text())
        self.assertIn("B(2)", card.preview_label.text())
        self.assertIn("outputs 2", card.get_guidance_text())

        set_combo_value(card.plane_combo, "111")
        card._refresh_preview()
        self.assertIn("1 layer", card.preview_label.text())
        self.assertIn("A(4)", card.preview_label.text())
        self.assertIn("A=4", card.preview_label.text())
        self.assertIn("B=0", card.preview_label.text())
        self.assertIn("At least two layers", card.preview_label.text())
        self.assertIn("1 layer", card.get_summary_text())
        self.assertIn("outputs 2", card.get_guidance_text())
        self.assertIn("Only 1 layer", card.get_guidance_text())

        card.group_a_edit.clear()
        card._refresh_preview()
        self.assertEqual(card.get_summary_text(), "Complete the two group labels")
        self.assertIn("non-empty", card.get_guidance_text())
        card.group_a_edit.setText("A")
        card._refresh_preview()

        existing = fcc.copy()
        existing.new_array(
            "group",
            np.asarray(["surface", "surface", "bulk", "bulk"], dtype=object),
        )
        card.set_preview_structure(existing)
        self.assertIn("output will be unchanged", card.preview_label.text())
        self.assertIn("bulk=2", card.preview_label.text())
        self.assertIn("surface=2", card.preview_label.text())
        self.assertIn("Keep existing groups", card.get_summary_text())
        self.assertIn("preserved", card.get_guidance_text())

        card.overwrite_checkbox.setChecked(True)
        self.assertIn("will be overwritten", card.preview_label.text())

