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
from NepTrainKit.core.io.base import DPPlotData, NepPlotData
from NepTrainKit.core.energy_shift import EnergyBaselinePreset
from NepTrainKit.core.precision import get_storage_float_dtype
from numpy.testing import assert_allclose
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.types import CanvasMode, ForcesMode
from NepTrainKit.config import  Config
from PySide6.QtWidgets import QApplication
from NepTrainKit.ui.widgets.dialog import ShiftEnergyDialogValues
import NepTrainKit.ui.widgets.dialog as dialog_module
import NepTrainKit.ui.views.nep as nep_view_module
import NepTrainKit.ui.pages.show_nep as show_nep_module
from NepTrainKit.ui.canvas.vispy.structure import StructurePlotWidget, StructureTurntableCamera

Config()
Config.set("nep", "backend","cpu")
Config.set("nep", "data_precision", "float32")

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
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)


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
        result.export_model_xyz(tmp_dir)
        export_good_model = Structure.read_multiple(
            os.path.join(tmp_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(
            os.path.join(tmp_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 23)
        self.assertEqual(len(export_remove_model), 2)

    def test_load_train2(self):
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()
        self.assertEqual(result.energy.num, 25)
        self.assertEqual(result.force.num, 6250)
        self.assertEqual(result.stress.num, 25)
        self.assertEqual(result.virial.num, 25)

    def _make_nep_workdir(self) -> str:
        tmp_path = Path(tempfile.mkdtemp(prefix=f"nep_test_{uuid.uuid4().hex}_"))
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

        expected_energy = np.array([structure.per_atom_energy], dtype=get_storage_float_dtype())
        synced_energy = result.energy.now_data[target_idx, result.energy.x_cols]
        assert_allclose(synced_energy, expected_energy, atol=1e-12)

        force_rows = result.force.convert_index([target_idx])
        synced_force = result.force.now_data[force_rows, result.force.x_cols].reshape(-1, 3)
        assert_allclose(synced_force, new_forces, atol=1e-6)

        virial_rows = result.virial.convert_index([target_idx])
        expected_virial = structure.nep_virial.astype(get_storage_float_dtype(), copy=False)
        assert_allclose(
            result.virial.now_data[virial_rows, result.virial.x_cols],
            expected_virial.reshape(1, -1),
            atol=1e-6,
        )

        stress_rows = result.stress.convert_index([target_idx])
        atoms = result.atoms_num_list[target_idx]
        coeff = atoms / structure.volume
        expected_stress = (expected_virial * coeff * 160.21766208).astype(get_storage_float_dtype(), copy=False)
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
            assert_allclose(synced_energy, np.array([per_atom], dtype=get_storage_float_dtype()), atol=1e-12)

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

        expected_norm = np.linalg.norm(new_forces, axis=0, keepdims=True).astype(get_storage_float_dtype(), copy=False)
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

    def test_undo_selection_history(self):
        tmp_dir = self._make_nep_workdir()
        local_train = os.path.join(tmp_dir, "train.xyz")
        result = NepTrainResultData.from_path(local_train)
        result.load()

        result.select([0, 1])
        result.uncheck(0)
        result.inverse_select()

        self.assertTrue(result.undo_selection())
        self.assertEqual(result.select_index, {1})
        self.assertTrue(result.undo_selection())
        self.assertEqual(result.select_index, {0, 1})
        self.assertTrue(result.undo_selection())
        self.assertEqual(result.select_index, set())
        self.assertFalse(result.undo_selection())

        result.select([0, 1])
        result.delete_selected()
        self.assertFalse(result.undo_selection())


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


class TestNepResultPlotWidgetCanvasFactory(unittest.TestCase):
    def _set_canvas_config(self, canvas_type, threshold=50):
        previous_canvas = Config.get("widget", "canvas_type", CanvasMode.PYQTGRAPH.value)
        previous_threshold = Config.get("widget", "auto_vispy_point_threshold", None)
        self.addCleanup(lambda: Config.set("widget", "canvas_type", previous_canvas))
        if previous_threshold is None:
            self.addCleanup(lambda: Config.delete("widget", "auto_vispy_point_threshold"))
        else:
            self.addCleanup(lambda: Config.set("widget", "auto_vispy_point_threshold", previous_threshold))
        Config.set("widget", "canvas_type", canvas_type)
        Config.set("widget", "auto_vispy_point_threshold", threshold)

    @staticmethod
    def _plot_dataset(points):
        return SimpleNamespace(datasets=[SimpleNamespace(x=np.zeros(points, dtype=np.float32))])

    def test_auto_canvas_prefers_pyqtgraph_below_threshold(self):
        self._set_canvas_config(CanvasMode.AUTO.value, threshold=50)
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._vispy_unavailable = False

        canvas_type = nep_view_module.NepResultPlotWidget._desired_canvas_type_for_dataset(
            widget, self._plot_dataset(49)
        )

        self.assertEqual(canvas_type, CanvasMode.PYQTGRAPH.value)

    def test_auto_canvas_prefers_vispy_at_threshold(self):
        self._set_canvas_config(CanvasMode.AUTO.value, threshold=50)
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._vispy_unavailable = False

        canvas_type = nep_view_module.NepResultPlotWidget._desired_canvas_type_for_dataset(
            widget, self._plot_dataset(50)
        )

        self.assertEqual(canvas_type, CanvasMode.VISPY.value)

    def test_manual_canvas_setting_overrides_auto_threshold(self):
        self._set_canvas_config(CanvasMode.PYQTGRAPH.value, threshold=50)
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._vispy_unavailable = False

        canvas_type = nep_view_module.NepResultPlotWidget._desired_canvas_type_for_dataset(
            widget, self._plot_dataset(5000)
        )

        self.assertEqual(canvas_type, CanvasMode.PYQTGRAPH.value)

    def test_manual_vispy_uses_pyqtgraph_when_vispy_unavailable(self):
        self._set_canvas_config(CanvasMode.VISPY.value, threshold=50)
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._vispy_unavailable = True

        canvas_type = nep_view_module.NepResultPlotWidget._desired_canvas_type_for_dataset(
            widget, self._plot_dataset(5000)
        )

        self.assertEqual(canvas_type, CanvasMode.PYQTGRAPH.value)

    def test_auto_canvas_uses_pyqtgraph_when_vispy_unavailable(self):
        self._set_canvas_config(CanvasMode.AUTO.value, threshold=50)
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._vispy_unavailable = True

        canvas_type = nep_view_module.NepResultPlotWidget._desired_canvas_type_for_dataset(
            widget, self._plot_dataset(5000)
        )

        self.assertEqual(canvas_type, CanvasMode.PYQTGRAPH.value)

    def test_set_dataset_switches_canvas_before_plotting(self):
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._canvas_type = CanvasMode.PYQTGRAPH.value
        widget.last_figure_num = 1
        new_canvas = SimpleNamespace(
            init_axes=MagicMock(),
            set_nep_result_data=MagicMock(),
            plot_nep_result=MagicMock(),
        )
        dataset = SimpleNamespace(datasets=[object(), object()])

        def switch_canvas(canvas_type):
            widget._canvas_type = canvas_type
            widget.canvas = new_canvas

        widget.swith_canvas = MagicMock(side_effect=switch_canvas)

        with patch.object(
            nep_view_module.NepResultPlotWidget,
            "_desired_canvas_type_for_dataset",
            return_value=CanvasMode.VISPY.value,
        ):
            nep_view_module.NepResultPlotWidget.set_dataset(widget, dataset)

        widget.swith_canvas.assert_called_once_with(CanvasMode.VISPY.value)
        new_canvas.init_axes.assert_called_once_with(2)
        new_canvas.set_nep_result_data.assert_called_once_with(dataset)
        new_canvas.plot_nep_result.assert_called_once()

    def test_swith_canvas_uses_factory_and_host_widget(self):
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._layout = MagicMock()
        widget._canvas_fallback_warned = False
        canvas_obj = object()
        host_widget = object()

        with patch.object(nep_view_module, "create_result_canvas", return_value=(canvas_obj, False)) as create_mock, patch.object(
            nep_view_module, "resolve_canvas_host_widget", return_value=host_widget
        ) as resolve_mock:
            nep_view_module.NepResultPlotWidget.swith_canvas(widget, "pyqtgraph")

        self.assertIs(widget.canvas, canvas_obj)
        create_mock.assert_called_once_with("pyqtgraph", widget)
        resolve_mock.assert_called_once_with(canvas_obj)
        widget._layout.addWidget.assert_called_once_with(host_widget)

    def test_swith_canvas_warns_only_once_for_fallback(self):
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._layout = MagicMock()
        widget._canvas_fallback_warned = False

        with patch.object(
            nep_view_module,
            "create_result_canvas",
            side_effect=[(object(), True), (object(), True)],
        ), patch.object(
            nep_view_module,
            "resolve_canvas_host_widget",
            side_effect=[object(), object()],
        ), patch.object(nep_view_module.MessageManager, "send_warning_message") as warn_mock:
            nep_view_module.NepResultPlotWidget.swith_canvas(widget, "vispy")
            nep_view_module.NepResultPlotWidget.swith_canvas(widget, "vispy")

        warn_mock.assert_called_once_with(
            "Current canvas backend is vispy, but vispy canvas failed to initialize; fallback to pyqtgraph."
        )
        self.assertTrue(widget._canvas_fallback_warned)


class TestTrainingOverlayDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_dialog_uses_result_canvas_and_single_axis(self):
        from PySide6.QtWidgets import QWidget

        fake_canvas = SimpleNamespace(
            tool_bar=None,
            set_nep_result_data=MagicMock(),
            init_axes=MagicMock(),
            plot_nep_result=MagicMock(),
            apply_overlay_groups=MagicMock(),
        )
        host_widget = QWidget()

        pca_data = {
            "training_pca": np.array([[0.0, 0.0]], dtype=np.float32),
            "current_pca": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            "selected_current_indices": np.array([1], dtype=np.int32),
        }

        with patch.object(dialog_module, "create_result_canvas", return_value=(fake_canvas, False)) as create_mock, patch.object(
            dialog_module,
            "resolve_canvas_host_widget",
            return_value=host_widget,
        ) as resolve_mock:
            dlg = dialog_module.TrainingOverlayDialog(parent=None, pca_data=pca_data, canvas_type="pyqtgraph")

        create_mock.assert_called_once_with("pyqtgraph", dlg)
        resolve_mock.assert_called_once_with(fake_canvas)
        fake_canvas.init_axes.assert_called_once_with(1)
        fake_canvas.plot_nep_result.assert_called_once()
        fake_canvas.apply_overlay_groups.assert_called_once()
        loaded_ids, selected_ids = fake_canvas.apply_overlay_groups.call_args.args
        self.assertListEqual(list(map(int, loaded_ids)), [1, 2])
        self.assertListEqual(list(map(int, selected_ids)), [2])

        result_data = fake_canvas.set_nep_result_data.call_args.args[0]
        self.assertEqual(len(result_data.datasets), 1)
        self.assertEqual(result_data.datasets[0].display_title, "Training Overlay")
        self.assertEqual(result_data.datasets[0].x_label, "PC1")
        self.assertEqual(result_data.datasets[0].y_label, "PC2")
        self.assertFalse(result_data.datasets[0].parity_mode)

    def test_dialog_shows_fallback_hint_for_vispy_failure(self):
        fake_canvas = SimpleNamespace(
            tool_bar=None,
            set_nep_result_data=MagicMock(),
            init_axes=MagicMock(),
            plot_nep_result=MagicMock(),
            apply_overlay_groups=MagicMock(),
        )
        from PySide6.QtWidgets import QWidget

        with patch.object(dialog_module, "create_result_canvas", return_value=(fake_canvas, True)), patch.object(
            dialog_module,
            "resolve_canvas_host_widget",
            return_value=QWidget(),
        ):
            dlg = dialog_module.TrainingOverlayDialog(parent=None, pca_data={}, canvas_type="vispy")

        self.assertIn("fallback to pyqtgraph", dlg._plot_hint_label.text().lower())


class _SparseBoxControl:
    def __init__(self, value, items=None):
        self._value = value
        self._items = list(items or [])

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def currentIndex(self):
        return self._value

    def setCurrentIndex(self, value):
        self._value = value

    def currentData(self):
        if self._items and isinstance(self._value, int) and 0 <= self._value < len(self._items):
            return self._items[self._value]
        return self._value

    def findData(self, value):
        try:
            return self._items.index(value)
        except ValueError:
            return -1

    def text(self):
        return self._value

    def setText(self, value):
        self._value = value

    def isChecked(self):
        return bool(self._value)


class _SparseBox:
    def __init__(
        self,
        accepted=True,
        training_path="train.xyz",
        show_overlay=True,
        selection_strategy="global",
    ):
        self._accepted = accepted
        self.intSpinBox = _SparseBoxControl(5)
        self.doubleSpinBox = _SparseBoxControl(0.1)
        self.regionCheck = _SparseBoxControl(False)
        self.descriptorCombo = _SparseBoxControl(0, ["reduced", "raw"])
        self.modeCombo = _SparseBoxControl(0, ["count", "r2"])
        self.strategyCombo = _SparseBoxControl(
            1 if selection_strategy == "element_set" else 0,
            ["global", "element_set"],
        )
        self.r2SpinBox = _SparseBoxControl(0.9)
        self.trainingPathEdit = _SparseBoxControl(training_path)
        self.trainingOverlayCheck = _SparseBoxControl(show_overlay)

    def exec(self):
        return self._accepted


class _OverlayDialogRecorder:
    last_kwargs = None
    shown = False

    def __init__(self, *args, **kwargs):
        type(self).last_kwargs = kwargs

    @staticmethod
    def compute_pca_data(*_args, **_kwargs):
        return {
            "training_pca": np.array([[0.0, 0.0]], dtype=np.float32),
            "current_pca": np.array([[1.0, 1.0]], dtype=np.float32),
            "selected_current_indices": np.array([0], dtype=np.int32),
        }

    def show(self):
        type(self).shown = True


class TestSparseMessageBoxStrategy(unittest.TestCase):
    def test_balanced_strategy_locks_validated_descriptor_and_mode(self):
        from PySide6.QtWidgets import QWidget

        self._app = QApplication.instance() or QApplication([])
        parent = QWidget()
        box = dialog_module.SparseMessageBox(parent)
        box.modeCombo.setCurrentIndex(box.modeCombo.findData("r2"))
        box.descriptorCombo.setCurrentIndex(box.descriptorCombo.findData("reduced"))

        box.strategyCombo.setCurrentIndex(box.strategyCombo.findData("element_set"))

        self.assertEqual(box.modeCombo.currentData(), "count")
        self.assertEqual(box.descriptorCombo.currentData(), "raw")
        self.assertFalse(box.modeCombo.isEnabled())
        self.assertFalse(box.descriptorCombo.isEnabled())
        self.assertTrue(box.r2SpinBox.isHidden())

        box.strategyCombo.setCurrentIndex(box.strategyCombo.findData("global"))
        self.assertEqual(box.modeCombo.currentData(), "r2")
        self.assertEqual(box.descriptorCombo.currentData(), "reduced")
        self.assertTrue(box.modeCombo.isEnabled())
        self.assertTrue(box.descriptorCombo.isEnabled())
        self.assertFalse(box.r2SpinBox.isHidden())


class TestNepResultPlotWidgetSparseOverlay(unittest.TestCase):
    def setUp(self):
        self._prev_canvas = Config.get("widget", "canvas_type", "pyqtgraph")
        self._prev_training_path = Config.get("widget", "sparse_training_path", "")
        self._prev_strategy = Config.get("widget", "sparse_selection_strategy", "global")
        Config.set("widget", "sparse_selection_strategy", "global")

    def tearDown(self):
        Config.set("widget", "canvas_type", self._prev_canvas)
        Config.set("widget", "sparse_training_path", self._prev_training_path)
        Config.set("widget", "sparse_selection_strategy", self._prev_strategy)
        _OverlayDialogRecorder.last_kwargs = None
        _OverlayDialogRecorder.shown = False

    def test_sparse_point_passes_canvas_type_to_overlay_dialog(self):
        Config.set("widget", "canvas_type", "vispy")
        Config.set("widget", "sparse_training_path", "train.xyz")
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._parent = None
        data = SimpleNamespace(
            sparse_point_selection=MagicMock(return_value=([3], False)),
        )
        widget.canvas = SimpleNamespace(
            nep_result_data=data,
            select_index=MagicMock(),
        )

        with patch.object(nep_view_module, "SparseMessageBox", return_value=_SparseBox()), patch.object(
            nep_view_module,
            "TrainingOverlayDialog",
            _OverlayDialogRecorder,
        ):
            nep_view_module.NepResultPlotWidget.sparse_point(widget)

        data.sparse_point_selection.assert_called_once()
        self.assertEqual(
            data.sparse_point_selection.call_args.kwargs["selection_strategy"],
            "global",
        )
        widget.canvas.select_index.assert_called_once_with([3], False)
        self.assertTrue(_OverlayDialogRecorder.shown)
        self.assertEqual(_OverlayDialogRecorder.last_kwargs["canvas_type"], "vispy")

    def test_sparse_point_skips_overlay_without_training_path(self):
        Config.set("widget", "sparse_training_path", "")
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._parent = None
        data = SimpleNamespace(
            sparse_point_selection=MagicMock(return_value=([3], False)),
        )
        widget.canvas = SimpleNamespace(
            nep_result_data=data,
            select_index=MagicMock(),
        )

        with patch.object(nep_view_module, "SparseMessageBox", return_value=_SparseBox(training_path="", show_overlay=True)), patch.object(
            nep_view_module,
            "TrainingOverlayDialog",
            _OverlayDialogRecorder,
        ):
            nep_view_module.NepResultPlotWidget.sparse_point(widget)

        self.assertIsNone(_OverlayDialogRecorder.last_kwargs)

    def test_sparse_point_passes_element_set_strategy_with_raw_fixed_count(self):
        Config.set("widget", "sparse_selection_strategy", "element_set")
        widget = nep_view_module.NepResultPlotWidget.__new__(nep_view_module.NepResultPlotWidget)
        widget._parent = None
        data = SimpleNamespace(
            sparse_point_selection=MagicMock(return_value=([1, 4], False)),
            _last_sparse_group_report={("H",): {}, ("He",): {}},
        )
        widget.canvas = SimpleNamespace(
            nep_result_data=data,
            select_index=MagicMock(),
        )

        with patch.object(
            nep_view_module,
            "SparseMessageBox",
            return_value=_SparseBox(
                training_path="",
                show_overlay=False,
                selection_strategy="element_set",
            ),
        ), patch.object(nep_view_module.MessageManager, "send_info_message"):
            nep_view_module.NepResultPlotWidget.sparse_point(widget)

        kwargs = data.sparse_point_selection.call_args.kwargs
        self.assertEqual(kwargs["selection_strategy"], "element_set")
        self.assertEqual(kwargs["descriptor_source"], "raw")
        self.assertEqual(kwargs["sampling_mode"], "count")
        widget.canvas.select_index.assert_called_once_with([1, 4], False)


class TestShowNepWidgetArrowCapability(unittest.TestCase):
    class _FakeStructure:
        def __init__(self, index=0, natoms=1):
            self.index = index
            self.atomic_properties = {}
            self.has_forces = False
            self._natoms = natoms

        def __len__(self):
            return self._natoms

        def copy(self):
            clone = TestShowNepWidgetArrowCapability._FakeStructure(self.index, self._natoms)
            clone.atomic_properties = {key: value.copy() for key, value in self.atomic_properties.items()}
            clone.has_forces = self.has_forces
            return clone

    def test_update_structure_arrow_availability_disables_when_unsupported(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget.show_struct_widget = object()
        widget.structure_toolbar = SimpleNamespace(set_arrow_enabled=MagicMock())
        with patch.object(show_nep_module, "supports_structure_arrows", return_value=False):
            show_nep_module.ShowNepWidget._update_structure_arrow_availability(widget)
        widget.structure_toolbar.set_arrow_enabled.assert_called_once_with(
            False,
            "Arrow overlay is available only for vispy structure canvas.",
        )

    def test_show_arrow_dialog_guard_when_backend_not_supported(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget.show_struct_widget = object()
        with patch.object(show_nep_module, "supports_structure_arrows", return_value=False), patch.object(
            show_nep_module.MessageManager, "send_info_message"
        ) as info_mock:
            show_nep_module.ShowNepWidget.show_arrow_dialog(widget)
        info_mock.assert_called_once_with("Arrow overlay is unavailable for current structure canvas backend.")

    def test_ml_arrow_vectors_use_nep_plot_column_order(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        rows = np.array(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
                [7.0, 8.0, 9.0, 70.0, 80.0, 90.0],
            ],
            dtype=np.float32,
        )
        widget.nep_result_data = SimpleNamespace(_force_vector_dataset=NepPlotData(rows, group_list=np.array([2, 1]), title="force"))
        structure = self._FakeStructure(natoms=2)

        show_nep_module.ShowNepWidget._inject_ml_arrow_vectors(widget, structure, 0)

        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_ML_FORCE], rows[:2, :3])
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_DFT_FORCE], rows[:2, 3:])
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_FORCE_ERROR], rows[:2, :3] - rows[:2, 3:])

    def test_ml_arrow_vectors_use_deepmd_plot_column_order(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        rows = np.array(
            [
                [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
                [40.0, 50.0, 60.0, 4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        widget.nep_result_data = SimpleNamespace(_force_vector_dataset=DPPlotData(rows, group_list=np.array([2]), title="force"))
        structure = self._FakeStructure(natoms=2)

        show_nep_module.ShowNepWidget._inject_ml_arrow_vectors(widget, structure, 0)

        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_DFT_FORCE], rows[:, :3])
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_ML_FORCE], rows[:, 3:])
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_FORCE_ERROR], rows[:, 3:] - rows[:, :3])

    def test_deepmd_mforce_arrow_error_does_not_require_existing_mforce_property(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        rows = np.array(
            [
                [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
                [40.0, 50.0, 60.0, 4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        widget.nep_result_data = SimpleNamespace(mforce=DPPlotData(rows, group_list=np.array([2]), title="mforce"))
        structure = self._FakeStructure(natoms=2)
        structure.atomic_properties["force"] = np.zeros((2, 3), dtype=np.float32)

        show_nep_module.ShowNepWidget._inject_ml_arrow_vectors(widget, structure, 0)

        self.assertNotIn("mforce", structure.atomic_properties)
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_DFT_MFORCE], rows[:, :3])
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_ML_MFORCE], rows[:, 3:])
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_MFORCE_ERROR], rows[:, 3:] - rows[:, :3])

    def test_sparse_mforce_arrow_error_uses_force_mag_mask(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        rows = np.array(
            [
                [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
                [40.0, 50.0, 60.0, 4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        widget.nep_result_data = SimpleNamespace(mforce=DPPlotData(rows, group_list=np.array([2]), title="mforce"))
        structure = self._FakeStructure(natoms=3)
        structure.atomic_properties["force_mag"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        show_nep_module.ShowNepWidget._inject_ml_arrow_vectors(widget, structure, 0)

        expected_error = np.zeros((3, 3), dtype=np.float32)
        expected_error[1:] = rows[:, 3:] - rows[:, :3]
        np.testing.assert_allclose(structure.atomic_properties[show_nep_module._ARROW_MFORCE_ERROR], expected_error)

    def test_arrow_display_names_hide_internal_ml_force_keys(self):
        labels, label_to_prop = show_nep_module.ShowNepWidget._arrow_display_names(
            [show_nep_module._ARROW_ML_FORCE, show_nep_module._ARROW_FORCE_ERROR, "spin"]
        )

        self.assertEqual(labels, ["ML force", "Force error (ML - DFT)", "spin"])
        self.assertEqual(label_to_prop["ML force"], show_nep_module._ARROW_ML_FORCE)
        self.assertEqual(label_to_prop["spin"], "spin")

    def test_existing_atomic_vector_properties_remain_arrow_options(self):
        structure = self._FakeStructure(natoms=2)
        structure.atomic_properties["spin"] = np.ones((2, 3), dtype=np.float32)
        structure.atomic_properties[show_nep_module._ARROW_ML_FORCE] = np.zeros((2, 3), dtype=np.float32)

        props = [
            name for name, arr in structure.atomic_properties.items()
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3
        ]
        labels, label_to_prop = show_nep_module.ShowNepWidget._arrow_display_names(props)

        self.assertIn("spin", labels)
        self.assertIn("ML force", labels)
        self.assertEqual(label_to_prop["spin"], "spin")

    def test_display_structure_copy_reuses_atomic_arrays(self):
        source = self._FakeStructure(index=0, natoms=2)
        spin = np.ones((2, 3), dtype=np.float32)
        source.atomic_properties["spin"] = spin

        display = show_nep_module.ShowNepWidget._copy_structure_for_display(source)

        self.assertIsNot(display, source)
        self.assertIsNot(display.atomic_properties, source.atomic_properties)
        self.assertIs(display.atomic_properties["spin"], spin)
        display.atomic_properties[show_nep_module._ARROW_ML_FORCE] = np.zeros((2, 3), dtype=np.float32)
        self.assertNotIn(show_nep_module._ARROW_ML_FORCE, source.atomic_properties)

    def test_show_current_structure_defers_and_coalesces_structure_render(self):
        app = QApplication.instance() or QApplication([])
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._pending_structure_index = None
        widget._structure_update_scheduled = False
        widget._sync_reject_toolbar_state = MagicMock()
        source_atoms = self._FakeStructure(index=3)
        widget.nep_result_data = SimpleNamespace(get_atoms=MagicMock(return_value=source_atoms))
        widget.graph_widget = SimpleNamespace(canvas=SimpleNamespace(plot_current_point=MagicMock()))
        widget.show_struct_widget = SimpleNamespace(show_structure=MagicMock())
        widget.update_structure_bond_info = MagicMock()
        widget.struct_info_widget = SimpleNamespace(show_structure_info=MagicMock())
        widget.force_label = SimpleNamespace(setText=MagicMock())
        widget._refresh_export_actions = MagicMock()

        show_nep_module.ShowNepWidget.show_current_structure(widget, 1)
        show_nep_module.ShowNepWidget.show_current_structure(widget, 2)
        show_nep_module.ShowNepWidget.show_current_structure(widget, 3)

        self.assertEqual(widget.graph_widget.canvas.plot_current_point.call_count, 3)
        widget.show_struct_widget.show_structure.assert_not_called()

        app.processEvents()

        widget.nep_result_data.get_atoms.assert_called_once_with(3)
        widget.show_struct_widget.show_structure.assert_called_once()
        rendered_atoms = widget.show_struct_widget.show_structure.call_args.args[0]
        self.assertEqual(rendered_atoms.index, 3)
        self.assertIsNot(rendered_atoms, source_atoms)

    def test_ml_arrow_vectors_do_not_mutate_export_source_structure(self):
        app = QApplication.instance() or QApplication([])
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        widget._pending_structure_index = None
        widget._structure_update_scheduled = False
        widget._sync_reject_toolbar_state = MagicMock()
        source_atoms = self._FakeStructure(index=0, natoms=2)
        rows = np.array(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            ],
            dtype=np.float32,
        )
        widget.nep_result_data = SimpleNamespace(
            get_atoms=MagicMock(return_value=source_atoms),
            _force_vector_dataset=NepPlotData(rows, group_list=np.array([2]), title="force"),
        )
        widget.graph_widget = SimpleNamespace(canvas=SimpleNamespace(plot_current_point=MagicMock()))
        widget.show_struct_widget = SimpleNamespace(show_structure=MagicMock())
        widget.update_structure_bond_info = MagicMock()
        widget.struct_info_widget = SimpleNamespace(show_structure_info=MagicMock())
        widget.force_label = SimpleNamespace(setText=MagicMock())
        widget._refresh_export_actions = MagicMock()

        show_nep_module.ShowNepWidget.show_current_structure(widget, 0)
        app.processEvents()

        rendered_atoms = widget.show_struct_widget.show_structure.call_args.args[0]
        self.assertIn(show_nep_module._ARROW_ML_FORCE, rendered_atoms.atomic_properties)
        self.assertNotIn(show_nep_module._ARROW_ML_FORCE, source_atoms.atomic_properties)
        self.assertNotIn(show_nep_module._ARROW_DFT_FORCE, source_atoms.atomic_properties)
        self.assertNotIn(show_nep_module._ARROW_FORCE_ERROR, source_atoms.atomic_properties)

    def test_arrow_vector_lookup_keeps_force_and_mforce_cached(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        force = NepPlotData(np.zeros((2, 6), dtype=np.float32), group_list=np.array([2]), title="force")
        mforce = NepPlotData(np.ones((2, 6), dtype=np.float32), group_list=np.array([2]), title="mforce")

        show_nep_module.ShowNepWidget._arrow_vector_lookup(widget, force)
        show_nep_module.ShowNepWidget._arrow_vector_lookup(widget, mforce)

        self.assertEqual(len(widget._arrow_vector_lookup_cache), 2)

    def test_arrow_vector_pair_uses_cached_sorted_lookup(self):
        widget = show_nep_module.ShowNepWidget.__new__(show_nep_module.ShowNepWidget)
        widget._arrow_vector_lookup_cache = {}
        rows = np.array(
            [
                [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
                [4.0, 5.0, 6.0, 14.0, 15.0, 16.0],
                [7.0, 8.0, 9.0, 17.0, 18.0, 19.0],
            ],
            dtype=np.float32,
        )
        dataset = NepPlotData(rows, group_list=np.array([2, 1]), title="force")

        dft, ml = show_nep_module.ShowNepWidget._extract_arrow_vector_pair(widget, dataset, 0, 2)

        np.testing.assert_allclose(ml, rows[:2, :3])
        np.testing.assert_allclose(dft, rows[:2, 3:])
        self.assertEqual(len(widget._arrow_vector_lookup_cache), 1)


class TestStructurePlotWidgetInteraction(unittest.TestCase):
    class _Projection:
        def __init__(self, screen_positions):
            self.screen_positions = np.asarray(screen_positions, dtype=np.float64)

        def map(self, positions):
            positions = np.asarray(positions)
            return np.column_stack(
                [
                    self.screen_positions[: positions.shape[0]],
                    np.zeros(positions.shape[0], dtype=np.float64),
                    np.ones(positions.shape[0], dtype=np.float64),
                ]
            )

    def test_structure_canvas_uses_custom_turntable_camera(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=False, size=(200, 150))
        try:
            self.assertIsInstance(canvas.view.camera, StructureTurntableCamera)
            self.assertEqual(canvas.view.camera.translate_speed, 1.3)
            canvas.set_projection(True)
            self.assertIsInstance(canvas.view.camera, StructureTurntableCamera)
            self.assertEqual(canvas.view.camera.translate_speed, 1.3)
            canvas.set_projection(False)
            self.assertIsInstance(canvas.view.camera, StructureTurntableCamera)
            self.assertEqual(canvas.view.camera.translate_speed, 1.3)
        finally:
            canvas.close()
            app.processEvents()

    def test_nearest_atom_for_double_click_uses_screen_distance(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=False, size=(200, 150))
        canvas.structure = SimpleNamespace(positions=np.zeros((3, 3), dtype=np.float32))
        projection = self._Projection(
            [
                [10.0, 10.0],
                [40.0, 30.0],
                [100.0, 100.0],
            ]
        )
        try:
            with patch.object(canvas.view.scene, "node_transform", return_value=projection):
                atom_index = canvas._nearest_atom_index_at_canvas_pos(np.array([42.0, 31.0]))
                blank_index = canvas._nearest_atom_index_at_canvas_pos(np.array([70.0, 70.0]), max_distance=5.0)

            self.assertEqual(atom_index, 1)
            self.assertIsNone(blank_index)
        finally:
            canvas.close()
            app.processEvents()

    def test_left_double_click_sets_rotation_center_without_moving_view(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=False, size=(200, 150))
        props = [
            {"name": "species", "type": "S", "cols": 1},
            {"name": "pos", "type": "R", "cols": 3},
        ]
        structure = Structure(
            np.eye(3),
            {
                "pos": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
                "species": np.array(["H", "H"]),
            },
            props,
            {},
        )
        projection = self._Projection(
            [
                [10.0, 10.0],
                [30.0, 30.0],
            ]
        )
        event = SimpleNamespace(button=1, pos=np.array([31.0, 29.0]), handled=False)
        original_center = tuple(canvas.view.camera.center)

        try:
            canvas.show_structure(structure)
            with patch.object(canvas.view.scene, "node_transform", return_value=projection), patch.object(
                canvas, "update"
            ) as update_mock:
                canvas._on_mouse_double_click(event)

            self.assertEqual(tuple(canvas.view.camera.center), original_center)
            self.assertEqual(canvas.view.camera._rotation_center, (1.0, 2.0, 3.0))
            self.assertIsNotNone(canvas._rotation_center_marker)
            self.assertTrue(event.handled)
            self.assertGreaterEqual(update_mock.call_count, 1)
        finally:
            canvas.close()
            app.processEvents()

    def test_show_structure_resets_to_default_rotation_center_marker(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=False, size=(200, 150))
        props = [
            {"name": "species", "type": "S", "cols": 1},
            {"name": "pos", "type": "R", "cols": 3},
        ]
        first = Structure(
            np.eye(3),
            {"pos": np.zeros((1, 3), dtype=np.float32), "species": np.array(["H"])},
            props,
            {},
        )
        second = Structure(
            np.eye(3),
            {"pos": np.ones((1, 3), dtype=np.float32), "species": np.array(["H"])},
            props,
            {},
        )
        try:
            canvas.show_structure(first)
            canvas.view.camera.set_rotation_center((1.0, 2.0, 3.0))
            canvas._set_rotation_center_marker((1.0, 2.0, 3.0))
            canvas.show_structure(second)

            self.assertIsNone(canvas.view.camera._rotation_center)
            self.assertIsNotNone(canvas._rotation_center_marker)
        finally:
            canvas.close()
            app.processEvents()

    def test_show_structure_displays_default_rotation_center_marker(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=False, size=(200, 150))
        props = [
            {"name": "species", "type": "S", "cols": 1},
            {"name": "pos", "type": "R", "cols": 3},
        ]
        structure = Structure(
            np.eye(3),
            {"pos": np.zeros((1, 3), dtype=np.float32), "species": np.array(["H"])},
            props,
            {},
        )
        try:
            canvas.show_structure(structure)

            self.assertIsNone(canvas.view.camera._rotation_center)
            self.assertIsNotNone(canvas._rotation_center_marker)
        finally:
            canvas.close()
            app.processEvents()

    def test_rotation_center_keeps_pivot_screen_position_stable(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=True, size=(500, 400))
        props = [
            {"name": "species", "type": "S", "cols": 1},
            {"name": "pos", "type": "R", "cols": 3},
        ]
        structure = Structure(
            np.eye(3) * 10,
            {"pos": np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32), "species": np.array(["H", "H"])},
            props,
            {},
        )
        try:
            canvas.show_structure(structure)
            canvas.show()
            app.processEvents()
            camera = canvas.view.camera
            camera.center = (0, 0, 0)
            camera.distance = 50
            camera.scale_factor = 10
            camera.azimuth = 30
            camera.elevation = 30
            app.processEvents()

            pivot = np.array([2.0, 0.0, 0.0])
            before = camera._project_scene_point(pivot)
            camera.set_rotation_center(pivot)
            press = SimpleNamespace(pos=np.array([100.0, 100.0]))
            mouse = SimpleNamespace(press_event=press, pos=np.array([120.0, 100.0]), modifiers=[])
            camera._event_value = None
            camera._update_rotation(SimpleNamespace(mouse_event=mouse))
            app.processEvents()
            after = camera._project_scene_point(pivot)

            self.assertLess(float(np.linalg.norm(after - before)), 0.5)
        finally:
            canvas.close()
            app.processEvents()

    def test_right_drag_pan_tracks_screen_delta_after_zoom(self):
        app = QApplication.instance() or QApplication([])
        canvas = StructurePlotWidget(show=True, size=(500, 400))
        try:
            canvas.show()
            app.processEvents()
            camera = canvas.view.camera
            camera.center = (0, 0, 0)
            camera.distance = 5
            camera.scale_factor = 1
            camera.azimuth = 20
            camera.elevation = 30
            app.processEvents()
            point = np.array([0.0, 0.0, 0.0])
            before = camera._project_scene_point(point)

            camera._event_value = None
            camera._translate_from_mouse_delta(np.array([100.0, 100.0]), np.array([120.0, 100.0]))
            app.processEvents()
            after = camera._project_scene_point(point)

            delta = after - before
            self.assertGreater(float(delta[0]), 15.0)
            self.assertLess(float(delta[0]), 30.0)
            self.assertAlmostEqual(float(delta[1]), 0.0, delta=1.0)
        finally:
            canvas.close()
            app.processEvents()


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
