from .card_test_base import *


class TestFilterCards(BaseCardTest):
    def test_fps_filter_operation_rejects_missing_model(self):
        params = FPSFilterParams(nep_path=str(self.test_dir / "data" / "missing_nep.txt"))

        with self.assertRaises(FileNotFoundError):
            FPSFilterOperation().run_dataset([self.structure.copy()], params)

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

        card = GeometryFilterCard()
        card.min_pair_frame.set_input_value([1.4])
        card.min_vpa_frame.set_input_value([5.0])
        card.max_vpa_frame.set_input_value([80.0])
        card.require_cell_checkbox.setChecked(True)
        restored = GeometryFilterCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())
