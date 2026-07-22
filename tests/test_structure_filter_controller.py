from types import SimpleNamespace
from unittest.mock import MagicMock

from NepTrainKit.core.search import StructureFilterValidationError
from NepTrainKit.core.types import (
    FilterField,
    StructureFilterCondition,
    StructureFilterResult,
    StructureFilterSpec,
)
from NepTrainKit.ui.controllers import structure_filter_controller as controller_module
from NepTrainKit.ui.controllers.structure_filter_controller import StructureFilterController
from NepTrainKit.ui.pages.show_nep import ShowNepWidget


def _dataset(version=1):
    return SimpleNamespace(structure=SimpleNamespace(data=SimpleNamespace(version=version)))


def _spec(value):
    return StructureFilterSpec(
        conditions=(
            StructureFilterCondition(
                condition_id=value,
                field=FilterField.CONFIG_TYPE,
                text_values=(value,),
            ),
        )
    )


def _result(spec, version=1):
    return StructureFilterResult(
        indices=(1,),
        active_count=3,
        dataset_version=version,
        elapsed_ms=1.0,
        spec=spec,
    )


def test_superseded_background_result_is_ignored(monkeypatch):
    pending = []

    def fake_run_in_thread(parent, func, *, on_finished, on_error):
        pending.append((func, on_finished, on_error))
        return object()

    monkeypatch.setattr(controller_module, "run_in_thread", fake_run_in_thread)
    monkeypatch.setattr(
        controller_module.StructureFilterEngine,
        "evaluate",
        lambda dataset, spec: _result(spec),
    )

    controller = StructureFilterController()
    controller.set_dataset(_dataset())
    first = _spec("first")
    second = _spec("second")
    ready = []
    controller.previewReady.connect(ready.append)

    controller.set_spec(first)
    controller.preview()
    controller.set_spec(second)
    controller.preview()

    pending[0][1](pending[0][0]())
    assert ready == []
    pending[1][1](pending[1][0]())
    assert [result.spec for result in ready] == [second]


def test_validation_error_keeps_previous_result_but_marks_it_stale(monkeypatch):
    callbacks = []

    def fake_run_in_thread(parent, func, *, on_finished, on_error):
        callbacks.append((func, on_finished))
        return object()

    monkeypatch.setattr(controller_module, "run_in_thread", fake_run_in_thread)
    controller = StructureFilterController()
    controller.set_dataset(_dataset())
    good = _spec("good")
    bad = _spec("bad")

    controller.set_spec(good)
    monkeypatch.setattr(
        controller_module.StructureFilterEngine,
        "evaluate",
        lambda dataset, spec: _result(spec),
    )
    controller.preview()
    callbacks.pop(0)[1](_result(good))

    controller.set_spec(bad)
    error = StructureFilterValidationError("invalid_regex", "Invalid regex", "bad")
    monkeypatch.setattr(
        controller_module.StructureFilterEngine,
        "evaluate",
        lambda dataset, spec: (_ for _ in ()).throw(error),
    )
    controller.preview()
    func, done = callbacks.pop(0)
    done(func())

    assert controller.state.result.spec == good
    assert controller.state.error is error
    assert controller.state.stale
    assert not controller.result_is_current()


def test_dataset_version_change_invalidates_cached_result(monkeypatch):
    callbacks = []

    def fake_run_in_thread(parent, func, *, on_finished, on_error):
        callbacks.append((func, on_finished))
        return object()

    monkeypatch.setattr(controller_module, "run_in_thread", fake_run_in_thread)
    dataset = _dataset()
    spec = _spec("surface")
    controller = StructureFilterController()
    controller.set_dataset(dataset)
    controller.set_spec(spec)
    monkeypatch.setattr(
        controller_module.StructureFilterEngine,
        "evaluate",
        lambda current, current_spec: _result(current_spec, current.structure.data.version),
    )

    controller.preview()
    func, done = callbacks.pop(0)
    done(func())
    assert controller.result_is_current()

    dataset.structure.data.version += 1
    assert not controller.result_is_current()


def test_removing_the_last_condition_clears_preview_state_and_highlight():
    canvas = SimpleNamespace(clear_search_highlight=MagicMock())
    page = SimpleNamespace(
        structure_filter_controller=SimpleNamespace(
            clear=MagicMock(),
            set_spec=MagicMock(),
        ),
        structure_filter_bar=SimpleNamespace(
            clear_state=MagicMock(),
            set_stale=MagicMock(),
        ),
        graph_widget=SimpleNamespace(canvas=canvas),
    )

    ShowNepWidget._on_structure_filter_spec_changed(page, StructureFilterSpec())

    page.structure_filter_controller.clear.assert_called_once_with()
    page.structure_filter_controller.set_spec.assert_not_called()
    page.structure_filter_bar.clear_state.assert_called_once_with()
    page.structure_filter_bar.set_stale.assert_not_called()
    canvas.clear_search_highlight.assert_called_once_with()
