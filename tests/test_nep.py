#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pathlib import Path
import os
import shutil
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# Keep test config/database writes inside the repository sandbox.
# This must run before importing NepTrainKit modules because NepTrainKit.config
# initialises the database connection at import time.
os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from NepTrainKit.core.io import NepTrainResultData,NepPolarizabilityResultData,NepDipoleResultData
from NepTrainKit.core.energy_shift import EnergyBaselinePreset
from numpy.testing import assert_allclose
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.types import ForcesMode
from NepTrainKit.config import  Config
from PySide6.QtWidgets import QApplication
from NepTrainKit.ui.widgets.dialog import ShiftEnergyDialogValues
import NepTrainKit.ui.views.nep as nep_view_module

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
        base_tmp = Path(__file__).resolve().parent / "_sandbox_tmp"
        base_tmp.mkdir(parents=True, exist_ok=True)
        tmp_path = base_tmp / f"nep_test_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True, exist_ok=False)
        tmp_dir = str(tmp_path)
        self._tmp_dirs.append(tmp_dir)
        for item in os.listdir(self.data_dir):
            src = os.path.join(self.data_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        return tmp_dir

    def test_export_active_xyz(self):
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()

        result.select([0, 1])
        result.delete_selected()

        export_path = os.path.join(tmp_dir, "active_structures.xyz")
        result.export_active_xyz(export_path)
        exported = Structure.read_multiple(export_path)
        self.assertEqual(len(exported), int(result.structure.now_data.shape[0]))

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

        expected_norm = np.linalg.norm(new_forces, axis=0, keepdims=True).astype(np.float32)
        assert_allclose(synced_force, expected_norm, atol=1e-6)

    def test_sync_structures_respects_force_mode_raw(self):
        tmp_dir = self._make_nep_workdir()
        prev_mode = Config.get("widget", "forces_data", ForcesMode.Raw)
        self.addCleanup(lambda: Config.set("widget", "forces_data", prev_mode if prev_mode is not None else ForcesMode.Raw))
        Config.set("widget", "forces_data", ForcesMode.Raw)

        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()

        target_idx = int(result.structure.now_indices[0])
        structure = result.structure.all_data[target_idx]
        new_forces = np.full_like(structure.forces, 0.25, dtype=np.float32)
        structure.forces = new_forces

        result.sync_structures("force", [target_idx])
        rows = result.force.convert_index([target_idx])
        synced_force = result.force.now_data[rows, result.force.x_cols].reshape(-1, 3)

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


class _ShiftDummyStructure:
    def __init__(self, config_types=None, total_structures: int = 3):
        self._config_types = list(config_types or ["A", "B"])
        self.now_data = [object() for _ in range(total_structures)]

    def get_all_config(self, _search_type):
        return list(self._config_types)


class _ShiftDummyData:
    def __init__(self):
        self.select_index = {0}
        self.structure = _ShiftDummyStructure()

    def iter_shift_energy_baseline(self, *_args, **_kwargs):
        if False:
            yield 1


class _FakeShiftDialog:
    def __init__(self, accepted: bool, values: ShiftEnergyDialogValues):
        self._accepted = accepted
        self._values = values

    def exec(self):
        return self._accepted

    def collect_values(self):
        return self._values


class TestNepResultPlotWidgetShiftEnergyBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    @staticmethod
    def _make_widget(data):
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._parent = None
        widget.canvas = SimpleNamespace(
            nep_result_data=data,
            plot_nep_result=MagicMock(),
        )
        return widget

    def test_shift_energy_baseline_cancel_does_not_run_task(self):
        data = _ShiftDummyData()
        values = ShiftEnergyDialogValues()
        dialog = _FakeShiftDialog(False, values)
        widget = self._make_widget(data)

        with patch.object(
            nep_view_module.NepResultPlotWidget,
            "_build_shift_energy_dialog",
            return_value=dialog,
        ), patch.object(nep_view_module.NepResultPlotWidget, "_run_shift_energy_task") as run_mock:
            widget.shift_energy_baseline()

        run_mock.assert_not_called()
        widget.canvas.plot_nep_result.assert_not_called()

    def test_shift_energy_baseline_passes_selected_preset_to_runner(self):
        data = _ShiftDummyData()
        values = ShiftEnergyDialogValues(
            group_patterns=["A.*"],
            alignment_mode="REF_GROUP",
            max_generations=123,
            population_size=7,
            convergence_tol=1e-5,
            selected_preset_name="preset_a",
        )
        dialog = _FakeShiftDialog(True, values)
        widget = self._make_widget(data)
        preset = EnergyBaselinePreset(metadata={"name": "preset_a"})
        captured: dict[str, object] = {}

        def _capture_run(_self, _data, _ref_index, run_values, selected_preset):
            captured["values"] = run_values
            captured["selected_preset"] = selected_preset
            return {"apply_stats": {"shifted_structures": 1, "total_structures": 3, "unmatched_config_types": []}}

        with patch.object(
            nep_view_module.NepResultPlotWidget,
            "_build_shift_energy_dialog",
            return_value=dialog,
        ), patch.object(
            nep_view_module,
            "load_energy_baseline_preset",
            return_value=preset,
        ) as load_mock, patch.object(
            nep_view_module.NepResultPlotWidget,
            "_run_shift_energy_task",
            autospec=True,
            side_effect=_capture_run,
        ):
            widget.shift_energy_baseline()

        load_mock.assert_called_once_with("preset_a")
        self.assertIs(captured["selected_preset"], preset)
        self.assertIs(captured["values"], values)
        widget.canvas.plot_nep_result.assert_called_once()

    def test_shift_energy_baseline_save_preset_after_run(self):
        data = _ShiftDummyData()
        values = ShiftEnergyDialogValues(
            group_patterns=["A.*"],
            save_preset=True,
            preset_name="custom_baseline",
        )
        dialog = _FakeShiftDialog(True, values)
        widget = self._make_widget(data)
        baseline = EnergyBaselinePreset(metadata={})

        with patch.object(
            nep_view_module.NepResultPlotWidget,
            "_build_shift_energy_dialog",
            return_value=dialog,
        ), patch.object(
            nep_view_module.NepResultPlotWidget,
            "_run_shift_energy_task",
            return_value={"baseline": baseline},
        ), patch.object(nep_view_module, "save_energy_baseline_preset") as save_mock:
            widget.shift_energy_baseline()

        save_mock.assert_called_once_with("custom_baseline", baseline)
        widget.canvas.plot_nep_result.assert_called_once()

    def test_shift_energy_baseline_invalid_selected_preset_aborts(self):
        data = _ShiftDummyData()
        values = ShiftEnergyDialogValues(selected_preset_name="missing_preset")
        dialog = _FakeShiftDialog(True, values)
        widget = self._make_widget(data)

        with patch.object(
            nep_view_module.NepResultPlotWidget,
            "_build_shift_energy_dialog",
            return_value=dialog,
        ), patch.object(
            nep_view_module,
            "load_energy_baseline_preset",
            return_value=None,
        ), patch.object(
            nep_view_module.NepResultPlotWidget,
            "_run_shift_energy_task",
        ) as run_mock, patch.object(
            nep_view_module.MessageManager,
            "send_warning_message",
        ) as warn_mock:
            widget.shift_energy_baseline()

        run_mock.assert_not_called()
        warn_mock.assert_called_once_with("Selected preset unavailable.")
        widget.canvas.plot_nep_result.assert_not_called()


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
