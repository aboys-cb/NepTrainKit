#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path

from NepTrainKit.config import Config
from NepTrainKit.core.io import load_result_data


class TestTaceResultData(unittest.TestCase):
    def setUp(self) -> None:
        Config()
        self.data_dir = Path(__file__).parent / "data" / "tace"
        # Ensure stable parser path for tests (avoid optional C++ fast parser differences).
        self._old_disable = os.environ.get("NEPKIT_DISABLE_FASTXYZ")
        os.environ["NEPKIT_DISABLE_FASTXYZ"] = "1"

    def tearDown(self) -> None:
        if self._old_disable is None:
            os.environ.pop("NEPKIT_DISABLE_FASTXYZ", None)
        else:
            os.environ["NEPKIT_DISABLE_FASTXYZ"] = self._old_disable

    def _load(self, name: str):
        xyz_path = self.data_dir / name
        dataset = load_result_data(str(xyz_path))
        self.assertIsNotNone(dataset)
        dataset.load()
        return dataset

    def test_mforce_dft_force_mag(self) -> None:
        dataset = self._load("predict_force_mag.xyz")
        titles = [d.title for d in dataset.datasets]
        self.assertIn("mforce", titles)
        self.assertNotIn("stress", titles)

    def test_mforce_dft_mforce(self) -> None:
        dataset = self._load("predict_mforce.xyz")
        titles = [d.title for d in dataset.datasets]
        self.assertIn("mforce", titles)
        self.assertNotIn("stress", titles)

    def test_mforce_missing_not_shown(self) -> None:
        dataset = self._load("predict_no_mforce.xyz")
        titles = [d.title for d in dataset.datasets]
        self.assertNotIn("mforce", titles)
        self.assertNotIn("stress", titles)

        vir = getattr(dataset, "virial", None)
        self.assertIsNotNone(vir)
        self.assertEqual(vir.now_data.shape[0], 2)
        self.assertEqual(vir.now_data.shape[1], 12)


if __name__ == "__main__":
    unittest.main()

