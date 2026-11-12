"""ShowNEP load ownership, export snapshots, edits and navigation boundaries."""

import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.config import Config
from NepTrainKit.core.io import NepTrainResultData
from NepTrainKit.core.io.base import NepPlotData, ResultData, StructureData, StructureSyncRule
from NepTrainKit.core.io.importers import ase_atoms_to_structure
from NepTrainKit.core.search import StructureFilterEngine
from NepTrainKit.core.types import FilterField, SearchType, StructureFilterCondition, StructureFilterSpec, TextMatchMode
from NepTrainKit.ui.controllers.structure_filter_controller import StructureFilterController
from NepTrainKit.ui.pages import show_nep


class LoadHarness(show_nep.ShowNepWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.nep_result_data = object()
        self.load_thread = self._load_relay = self._loading_data = self._pending_load = None
        self._initial_loading = self._load_cancel_requested = False

    def _stop_playback(self):
        pass

    def _invalidate_background_jobs(self):
        pass

    def _refresh_export_actions(self):
        pass


class GatedResult(QObject):
    loadFinishedSignal = Signal()
    predictionStatusSignal = Signal(str)

    def __init__(self, *, succeeds=True, released=False):
        super().__init__()
        self.load_flag = False
        self.succeeds = succeeds
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        if released:
            self.release.set()

    move_to_load_thread = ResultData.move_to_load_thread
    _restore_load_thread_affinity = ResultData._restore_load_thread_affinity

    def request_cancel(self):
        self.cancelled.set()

    @Slot()
    def load(self):
        self.started.set()
        self.release.wait(3)
        self.load_flag = self.succeeds and not self.cancelled.is_set()
        self._restore_load_thread_affinity()
        self.loadFinishedSignal.emit()


class LoadTip(QObject):
    closedSignal = Signal()

    def __init__(self):
        super().__init__()
        self.closed = False
        self.success = False

    def close(self):
        self.closed = True
        self.closedSignal.emit()

    def setState(self, value):
        self.success = value


def pump_until(predicate):
    deadline = time.monotonic() + 4
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.001)
    assert predicate()


def test_cancel_is_nonblocking_and_only_latest_load_publishes():
    viewer = LoadHarness()
    old = viewer.nep_result_data
    first, skipped, latest = GatedResult(), GatedResult(released=True), GatedResult(released=True)
    tips = [LoadTip() for _ in range(3)]
    published = []
    timer = QTimer()
    ticks = []
    timer.timeout.connect(lambda: ticks.append(1))
    try:
        viewer._start_result_load(first, tips[0], (lambda: published.append(first),))
        pump_until(first.started.is_set)
        viewer._queue_after_load(lambda: viewer._start_result_load(skipped, tips[1], ()))
        viewer._queue_after_load(lambda: viewer._start_result_load(latest, tips[2], (lambda: published.append(latest),)))
        assert first.cancelled.is_set()
        assert not first.release.is_set()
        assert viewer.nep_result_data is old
        timer.start(1)
        pump_until(lambda: bool(ticks))
        assert not latest.started.is_set()
        first.release.set()
        pump_until(lambda: viewer._load_relay is None)
        assert not skipped.started.is_set()
        assert published == [latest]
        assert viewer.nep_result_data is latest
        assert tips[0].closed and tips[2].success
        assert latest.thread() is viewer.thread()
    finally:
        timer.stop()
        first.release.set()
        pump_until(lambda: viewer._load_relay is None)
        viewer.deleteLater()


@pytest.mark.parametrize("outcome", ["failure", "cancel", "closing"])
def test_failed_or_cancelled_load_preserves_displayed_dataset(outcome):
    viewer = LoadHarness()
    old = viewer.nep_result_data
    result = GatedResult(succeeds=outcome != "failure")
    tip = LoadTip()
    published = Mock()
    try:
        viewer._start_result_load(result, tip, (published,))
        pump_until(result.started.is_set)
        pending = Mock()
        if outcome == "cancel":
            tip.close()
        elif outcome == "closing":
            viewer._thread_shutdown = SimpleNamespace(closing=True)
            viewer._pending_load = pending
        result.release.set()
        pump_until(lambda: viewer._load_relay is None)
        assert viewer.nep_result_data is old
        assert tip.closed
        published.assert_not_called()
        pending.assert_not_called()
        assert not viewer._is_busy()
    finally:
        result.release.set()
        pump_until(lambda: viewer._load_relay is None)
        viewer.deleteLater()


def test_obsolete_completion_cannot_publish_and_always_releases_references():
    callback, finish = Mock(), Mock()
    relay = show_nep._LoadCompletionRelay(LoadTip(), SimpleNamespace(load_flag=True), (callback,),
        is_current=lambda: False, on_finished=finish)
    relay.handle_finished()
    callback.assert_not_called()
    finish.assert_called_once()
    assert relay._result_data is None


def test_cancelled_switch_restores_loaded_model_label(tmp_path):
    from qfluentwidgets import ComboBox
    viewer = LoadHarness()
    viewer.nep_model_combo = ComboBox(viewer)
    viewer._available_nep_files = [tmp_path / "A.txt", tmp_path / "B.txt"]
    viewer.nep_model_combo.addItems(["A", "B"])
    viewer.nep_model_combo.setCurrentIndex(1)
    viewer.nep_result_data = SimpleNamespace(nep_txt_path=viewer._available_nep_files[0])
    changed = Mock()
    viewer.nep_model_combo.currentIndexChanged.connect(changed)
    try:
        viewer._select_loaded_model()
        assert viewer.nep_model_combo.currentIndex() == 0
        changed.assert_not_called()
    finally:
        viewer.deleteLater()


@pytest.fixture
def small_data(tmp_path):
    data = ResultData(tmp_path / "nep.txt", tmp_path / "train.xyz", tmp_path / "descriptor.out")
    frames = [ase_atoms_to_structure(Atoms(symbol, positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True))
              for symbol in ("H", "He")]
    for i, frame in enumerate(frames):
        frame.energy = i + 1.
        frame.forces = np.zeros((1, 3))
    data._atoms_dataset = StructureData(frames)
    data.select_index = {0}
    return data


@pytest.mark.parametrize("scope,npy", [(scope, npy) for scope in ("selected", "active", "removed", "all", "current")
                                     for npy in (False, True) if scope != "current" or npy])
def test_export_snapshot_freezes_frames_masks_labels_and_type_map(small_data, tmp_path, scope, npy):
    data = small_data
    data.structure.remove([1])
    writer = data.prepare_export(scope, npy=npy, index=0)
    data.select_index = {1}
    data.structure.all_data[0].energy = 100.
    data.structure.all_data[0].positions[:] = 2.
    data.structure.remove([0])
    data.structure.data.revoke()
    data.structure.group_array.revoke()
    if npy:
        with patch("NepTrainKit.core.io.base.save_npy_structure") as save:
            writer(tmp_path)
        expected = {"all": [[1.], [2.]], "removed": [[2.]]}.get(scope, [[1.]])
        assert [[frame.energy for frame in call.args[1]] for call in save.call_args_list] == expected
        for call in save.call_args_list:
            assert set(call.kwargs["type_map"]) == {"H", "He"}
            assert np.all(call.args[1][0].positions == 0.)
    else:
        path = tmp_path if scope == "all" else tmp_path / "frames.xyz"
        writer(path)
        target = path / "export_good_model.xyz" if scope == "all" else path
        frames = read(target, index=":")
        assert frames[0].get_chemical_formula() == ("He" if scope == "removed" else "H")
        assert frames[0].get_potential_energy() == (2. if scope == "removed" else 1.)


def test_current_xyz_callback_keeps_original_dataset_and_frame(small_data, tmp_path):
    viewer = SimpleNamespace(nep_result_data=small_data, _dataset_ready=lambda: True,
        _choose_export_format=lambda: ("xyz", True, None),
        struct_index_spinbox=SimpleNamespace(value=lambda: 0), tr=lambda text: text)
    viewer._export_current_xyz = show_nep.ShowNepWidget._export_current_xyz.__get__(viewer)
    worker = Mock()
    path = tmp_path / "current.xyz"
    with patch.object(show_nep, "BackgroundTask", return_value=worker), \
         patch.object(show_nep, "call_path_dialog", return_value=str(path)):
        show_nep.ShowNepWidget.export_current_structure(viewer)
    viewer.nep_result_data = None
    small_data.structure.all_data[0].energy = 100.
    func, *args = worker.start_work.call_args.args
    func(*args)
    assert read(path).get_potential_energy() == 1.


@pytest.fixture
def loaded_data(tmp_path):
    work = tmp_path / "dataset"
    shutil.copytree(Path(__file__).parent / "data/nep", work)
    Config.set("nep", "backend", "cpu")
    data = NepTrainResultData.from_path(work / "train.xyz")
    data.load()
    assert data.load_flag
    yield data
    if data.nep_calc is not None:
        data.nep_calc.close()


def test_metadata_edit_updates_references_queries_and_content_revision(loaded_data):
    data = loaded_data
    frame = data.structure.all_data[0]
    spec = StructureFilterSpec(conditions=(StructureFilterCondition(condition_id="tag",
        field=FilterField.CONFIG_TYPE, text_values=(frame.tag,), match_mode=TextMatchMode.EXACT),))
    controller = StructureFilterController()
    controller.set_dataset(data)
    controller.set_spec(spec)
    controller.state.result = StructureFilterEngine.evaluate(data, spec)
    data.structure.ensure_completer_cache()
    prediction = data.energy.y.copy()
    version = data.structure.data.version
    data.select([0])
    data.update_structure_metadata([], {"Config_type": "edited", "energy": str(len(frame) * 96.25)})
    assert data.structure.data.version > version
    assert data._content_revision == 1
    assert not controller.result_is_current()
    assert "edited" in data.structure.get_completer_cache(SearchType.TAG)
    assert data.energy.x[0] == pytest.approx(96.25)
    np.testing.assert_equal(data.energy.y, prediction)
    assert 0 not in StructureFilterEngine.evaluate(data, spec).indices


def test_metadata_content_refresh_discards_model_and_analysis_caches(loaded_data):
    data = loaded_data
    canvas = Mock()
    viewer = SimpleNamespace(nep_result_data=data, _dataset_ready=lambda: True,
        _structure_content_version_seen=0, _structure_mask_version_seen=data.structure.data.version,
        _nep_result_cache={"other-model": object()}, _structure_analysis_job_id=3,
        _structure_analysis_cache={"old": object()}, _arrow_vector_lookup_cache={"old": object()},
        _phase_evidence_dataset_id=id(data), _phase_evidence_lookup={0: object()},
        graph_widget=SimpleNamespace(canvas=canvas), _update_structure_filter_suggestions=Mock(),
        _active_reject_indices=lambda: [], dataset_info_label=Mock(), empty_hint=Mock(),
        _refresh_export_actions=Mock(), structure_filter_controller=Mock(), tr=lambda text: text)
    data.select([0])
    data.update_structure_metadata([], {"Config_type": "new-tag"})
    for _ in range(2):
        show_nep.ShowNepWidget.update_dataset_info(viewer)
    canvas.plot_nep_result.assert_called_once_with(preserve_selection=True)
    assert not viewer._nep_result_cache
    assert not viewer._structure_analysis_cache
    assert not viewer._arrow_vector_lookup_cache
    assert not viewer._phase_evidence_lookup
    assert viewer._structure_analysis_job_id == 4
    viewer._update_structure_filter_suggestions.assert_called_once()
    viewer.structure_filter_controller.invalidate_result.assert_called_once()


def test_model_reload_keeps_edited_references_even_with_existing_official_outputs(loaded_data):
    data = loaded_data
    data.select([0])
    data.update_structure_metadata([], {"virial": "[100, 0, 0, 0, 200, 0, 0, 0, 300]"})
    expected = data.virial.all_data[0, data.virial.x_cols].copy()
    viewer = SimpleNamespace(nep_result_data=data, _start_result_load=Mock(),
        set_dataset=Mock(), _restore_reject=Mock(), _restore_selection=Mock(), tr=lambda text: text)
    with patch.object(show_nep, "StateToolTip"):
        show_nep.ShowNepWidget._reload_with_nep_file(viewer, str(data.data_xyz_path), data.nep_txt_path)
    reloaded, _, callbacks = viewer._start_result_load.call_args.args
    try:
        reloaded.load()
        assert reloaded.load_flag
        assert reloaded.nep_calc is None
        assert not np.allclose(reloaded.virial.all_data[0, reloaded.virial.x_cols], expected)
        for callback in callbacks:
            callback()
        np.testing.assert_allclose(reloaded.virial.all_data[0, reloaded.virial.x_cols], expected)
        assert "virial" in reloaded._modified_reference_fields
    finally:
        if reloaded.nep_calc is not None:
            reloaded.nep_calc.close()


def test_force_metadata_rename_updates_norm_plot_and_raw_arrows(loaded_data):
    data = loaded_data
    frame = data.structure.all_data[0]
    raw = data._force_vector_dataset
    # Match the normal force-norm display, whose values do not share raw storage.
    data._force_dataset = NepPlotData(np.zeros((data.structure.num, 6)), title="force")
    frame.atomic_properties["replacement_force"] = np.full_like(frame.forces, .25)
    frame.properties.append({"name": "replacement_force", "type": "R", "count": 3})
    data.select([0])
    data.update_structure_metadata([], {}, {"replacement_force": frame.force_label})
    np.testing.assert_allclose(data.force.all_data[0, data.force.x_cols], np.sqrt(len(frame)) * .25)
    np.testing.assert_allclose(raw.all_data[raw.convert_index([0]), raw.x_cols], .25)


@pytest.mark.parametrize("field", ["energy", "virial", "force"])
def test_removing_reference_labels_clears_plotted_values(loaded_data, field):
    data = loaded_data
    frame = data.structure.all_data[0]
    data.select([0])
    data.update_structure_metadata([frame.force_label if field == "force" else field], {})
    dataset = getattr(data, field)
    rows = dataset.convert_index([0])
    assert np.isnan(dataset.all_data[rows, dataset.x_cols]).all()
    if field == "virial":
        assert np.isnan(data.stress.all_data[0, data.stress.x_cols]).all()
    if field == "force":
        raw = data._force_vector_dataset
        assert np.isnan(raw.all_data[raw.convert_index([0]), raw.x_cols]).all()


def test_reference_sync_invalidates_max_error_cache():
    energy = NepPlotData(np.array([[0., 10.], [0., 1.]]), title="energy")
    assert energy.get_max_error_index(1) == [0]
    rule = StructureSyncRule("energy", "x_cols", lambda *args: (np.array([0, 1]), np.array([[0.], [20.]])))
    assert rule.apply(SimpleNamespace(energy=energy))
    assert energy.get_max_error_index(1) == [1]


@pytest.mark.parametrize("active,busy,selection,current,expected", [
    ([], False, set(), 0, None), ([0], True, set(), 0, None),
    ([0], False, set(), 0, 0), ([0, 2], False, {1, 2}, 2, 2),
    ([0, 2], False, set(), 0, 2),
])
def test_navigation_empty_loading_removed_selection_and_end(active, busy, selection, current, expected):
    spin = SimpleNamespace(value=lambda: current, setValue=Mock())
    viewer = SimpleNamespace(nep_result_data=SimpleNamespace(select_index=selection,
        structure=SimpleNamespace(group_array=SimpleNamespace(now_data=np.array(active)))),
        _dataset_ready=lambda: True, _is_busy=lambda: busy, struct_index_spinbox=spin,
        _stop_playback=Mock())
    for name in ("_navigation_indices", "to_next_structure"):
        setattr(viewer, name, getattr(show_nep.ShowNepWidget, name).__get__(viewer))
    show_nep.ShowNepWidget.to_last_structure(viewer)
    spin.setValue.reset_mock()
    show_nep.ShowNepWidget.play_show_structures(viewer)
    if expected is None or expected == current:
        spin.setValue.assert_not_called()
    else:
        spin.setValue.assert_called_once_with(expected)
    viewer._stop_playback.assert_called_once()


def test_start_play_with_no_available_frames_stops_timer():
    button = Mock()
    button.isChecked.return_value = True
    viewer = SimpleNamespace(auto_switch_button=button, play_timer=Mock(),
        _navigation_indices=lambda: np.empty(0, dtype=np.int64), tr=lambda text: text)
    show_nep.ShowNepWidget.start_play(viewer)
    viewer.play_timer.start.assert_not_called()
    viewer.play_timer.stop.assert_called_once()
    button.setChecked.assert_called_once_with(False)
