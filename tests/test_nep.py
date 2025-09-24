#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pathlib import Path
import os
import shutil
import tempfile
from NepTrainKit.core.io.nep import NepTrainResultData,NepPolarizabilityResultData,NepDipoleResultData
from numpy.testing import assert_allclose
from NepTrainKit.core import Structure
from NepTrainKit.core.types import ForcesMode
from NepTrainKit.config import  Config
from PySide6.QtWidgets import QApplication

app = QApplication()
Config()
Config.set("nep", "backend","cpu")

class TestNepTrainResultData( unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent
        self.data_dir = os.path.join(self.test_dir, "data/nep")
        self.train_path = os.path.join(self.data_dir, "train.xyz")
        self._tmp_dirs = []

    def tearDown(self):
        for tmp in self._tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_train(self):
        """测试结构加载功能"""
        result = NepTrainResultData.from_path(self.train_path)


        result.load()
        self.assertEqual(result.energy.num, 25)
        self.assertEqual(result.force.num, 6250)
        self.assertEqual(result.stress.num, 25)
        self.assertEqual(result.virial.num, 25)
        result.select([0,1,3])
        self.assertEqual(len(result.select_index),3)
        result.uncheck(0)
        self.assertEqual(len(result.select_index),2)

        self.assertEqual(result.select_index , {1,3})
        result.delete_selected()
        self.assertEqual(len(result.select_index) , 0)
        self.assertEqual(result.energy.num, 23)
        self.assertEqual(result.force.num, 5750)
        self.assertEqual(result.stress.num, 23)
        self.assertEqual(result.virial.num, 23)
        result.export_model_xyz(self.data_dir)
        export_good_model = Structure.read_multiple(
            os.path.join(self.data_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(
            os.path.join(self.data_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 23)
        self.assertEqual(len(export_remove_model), 2)
        os.remove(os.path.join(self.data_dir,"export_good_model.xyz"))
        os.remove(os.path.join(self.data_dir,"export_remove_model.xyz"))

    def test_load_train2(self):
        result = NepTrainResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.energy.num, 25)
        self.assertEqual(result.force.num, 6250)
        self.assertEqual(result.stress.num, 25)
        self.assertEqual(result.virial.num, 25)
        os.remove(os.path.join(self.data_dir,"energy_train.out"))
        os.remove(os.path.join(self.data_dir,"force_train.out"))
        os.remove(os.path.join(self.data_dir,"stress_train.out"))
        os.remove(os.path.join(self.data_dir,"virial_train.out"))
        os.remove(os.path.join(self.data_dir,"descriptor.out"))

    def _make_nep_workdir(self) -> str:
        tmp_dir = tempfile.mkdtemp(prefix="nep_test_")
        self._tmp_dirs.append(tmp_dir)
        for item in os.listdir(self.data_dir):
            src = os.path.join(self.data_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        return tmp_dir

    def test_sync_structures_updates_all_fields(self):
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()


        target_idx=10

        self.assertIsNotNone(target_idx)
        structure = result.structure.all_data[target_idx]
        structure.virial=np.array([1,2,3,1,2,3,1,2,3])
        structure.energy = structure.num_atoms * 2.5
        new_forces = np.full_like(structure.forces, 0.1234, dtype=np.float32)
        structure.forces = new_forces
        new_virial = np.linspace(1.0, 9.0, num=9, dtype=np.float32)
        structure.virial = new_virial

        result.sync_structures(fields=None, structure_indices=[target_idx])

        expected_energy = np.array([structure.per_atom_energy], dtype=np.float32)
        synced_energy = result.energy.now_data[target_idx, result.energy.x_cols]
        assert_allclose(synced_energy, expected_energy, atol=1e-6)

        force_rows = result.force.convert_index([target_idx])
        synced_force = result.force.now_data[force_rows, result.force.x_cols].reshape(-1, 3)
        assert_allclose(synced_force, new_forces, atol=1e-6)

        virial_rows = result.virial.convert_index([target_idx])
        expected_virial = structure.nep_virial.astype(np.float32)
        assert_allclose(
            result.virial.now_data[virial_rows, result.virial.x_cols],
            expected_virial.reshape(1, -1),
            atol=1e-6,
        )

        stress_rows = result.stress.convert_index([target_idx])
        atoms = result.atoms_num_list[target_idx]
        coeff = atoms / structure.volume
        expected_stress = (expected_virial * coeff * 160.21766208).astype(np.float32)
        assert_allclose(
            result.stress.now_data[stress_rows, result.stress.x_cols],
            expected_stress.reshape(1, -1),
            atol=1e-5,
        )

    def test_sync_structures_updates_all_when_indices_missing(self):
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()

        indices = list(map(int, result.structure.now_indices[:2]))
        expected = []
        for idx, per_atom in zip(indices, (3.0, 4.5)):
            structure = result.structure.all_data[idx]
            structure.energy = structure.num_atoms * per_atom
            expected.append(per_atom)

        result.sync_structures("energy")
        for idx, per_atom in zip(indices, expected):
            synced_energy = result.energy.now_data[idx, result.energy.x_cols]
            assert_allclose(synced_energy, np.array([per_atom], dtype=np.float32), atol=1e-6)

    def test_sync_structures_respects_force_mode_norm(self):
        tmp_dir = self._make_nep_workdir()
        prev_mode = Config.get("widget", "forces_data", ForcesMode.Raw)
        self.addCleanup(lambda: Config.set("widget", "forces_data", prev_mode if prev_mode is not None else ForcesMode.Raw))
        Config.set("widget", "forces_data", ForcesMode.Norm)

        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()

        target_idx = int(result.structure.now_indices[0])
        structure = result.structure.all_data[target_idx]
        new_forces = np.full_like(structure.forces, 0.25, dtype=np.float32)
        structure.forces = new_forces

        result.sync_structures("force", [target_idx])
        index=result.force.convert_index(target_idx)
        synced_force = result.force.now_data[index, result.force.x_cols]
 
        assert_allclose(synced_force, new_forces, atol=1e-6)

    def test_sync_structures_ignores_removed_and_unknown(self):
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()

        result.remove(0)
        snapshot = result.energy.now_data.copy()

        result.sync_structures("energy", [0])
        assert_allclose(result.energy.now_data, snapshot, atol=1e-12)

        result.sync_structures(["energy", "unknown"], [0, 1])
        assert_allclose(result.energy.now_data, snapshot, atol=1e-12)


    def test_inverse_select(self):
        result = NepTrainResultData.from_path(self.train_path)
        result.load()
        result.select([0,1,3])
        result.uncheck(0)
        result.inverse_select()
        self.assertEqual(len(result.select_index), result.num-2)
        self.assertNotIn(1, result.select_index)
        self.assertNotIn(3, result.select_index)


class TestNepPolarizabilityResultData( unittest.TestCase):
    def setUp(self):


        self.test_dir = Path(__file__).parent
        self.data_dir=os.path.join(self.test_dir,"data/polarizability")
        self.train_path=os.path.join(self.data_dir,"train.xyz")

    def tearDown(self):
        pass
    def test_load_train(self):


        """测试结构加载功能"""
        result = NepPolarizabilityResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.polarizability_diagonal.num, 5768)
        self.assertEqual(result.polarizability_no_diagonal.num, 5768)

        result.select([0,1,3])
        self.assertEqual(len(result.select_index),3)
        result.uncheck(0)
        self.assertEqual(len(result.select_index),2)

        self.assertEqual(result.select_index , {1,3})
        result.delete_selected()
        self.assertEqual(len(result.select_index) , 0)
        self.assertEqual(result.polarizability_diagonal.num, 5766)
        self.assertEqual(result.polarizability_no_diagonal.num, 5766)

        result.export_model_xyz(self.data_dir)
        export_good_model = Structure.read_multiple(os.path.join(self.data_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(os.path.join(self.data_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 5766)
        self.assertEqual(len(export_remove_model), 2)
        os.remove(os.path.join(self.data_dir,"export_good_model.xyz"))
        os.remove(os.path.join(self.data_dir,"export_remove_model.xyz"))

    def test_load_train2(self):


        result = NepPolarizabilityResultData.from_path(self.train_path)
        result.load()
        os.remove(os.path.join(self.data_dir,"polarizability_train.out"))
        os.remove(os.path.join(self.data_dir,"descriptor.out"))

    def test_inverse_select(self):

        result = NepPolarizabilityResultData.from_path(self.train_path)
        result.load()
        result.select([0,1,3])
        result.uncheck(0)
        result.inverse_select()
        self.assertEqual(len(result.select_index), result.num-2)
        self.assertNotIn(1, result.select_index)
        self.assertNotIn(3, result.select_index)

class TestNepDipoleResultData(unittest.TestCase):
    def setUp(self):


        self.test_dir = Path(__file__).parent
        self.data_dir=os.path.join(self.test_dir,"data/dipole")
        self.train_path=os.path.join(self.data_dir,"train.xyz")

    def tearDown(self):
        pass
    def test_load_train(self):
        """测试结构加载功能"""
        result = NepDipoleResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.dipole.num, 5768)

        result.select([0,1,3])
        self.assertEqual(len(result.select_index),3)
        result.uncheck(0)
        self.assertEqual(len(result.select_index),2)

        self.assertEqual(result.select_index , {1,3})
        result.delete_selected()
        self.assertEqual(len(result.select_index) , 0)
        self.assertEqual(result.dipole.num, 5766)


        result.export_model_xyz(self.data_dir)
        export_good_model = Structure.read_multiple(os.path.join(self.data_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(os.path.join(self.data_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 5766)
        self.assertEqual(len(export_remove_model), 2)
        os.remove(os.path.join(self.data_dir,"export_good_model.xyz"))
        os.remove(os.path.join(self.data_dir,"export_remove_model.xyz"))

    def test_load_train2(self):
        result = NepDipoleResultData.from_path(self.train_path)
        result.load()
        os.remove(os.path.join(self.data_dir,"dipole_train.out"))
        os.remove(os.path.join(self.data_dir,"descriptor.out"))

    def test_inverse_select(self):
        result = NepDipoleResultData.from_path(self.train_path)
        result.load()
        result.select([0,1,3])
        result.uncheck(0)
        result.inverse_select()
        self.assertEqual(len(result.select_index), result.num-2)
        self.assertNotIn(1, result.select_index)
        self.assertNotIn(3, result.select_index)


if __name__ == "__main__":
    unittest.main()
    app.exit()