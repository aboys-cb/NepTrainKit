"""Asynchronous state controller for the Show NEP structure filter."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from NepTrainKit.core.search import StructureFilterEngine, StructureFilterValidationError
from NepTrainKit.core.types import StructureFilterResult, StructureFilterSpec
from NepTrainKit.ui.threads import run_in_thread


@dataclass
class StructureFilterState:
    spec: StructureFilterSpec = StructureFilterSpec()
    result: StructureFilterResult | None = None
    error: StructureFilterValidationError | None = None
    running: bool = False
    stale: bool = False


class StructureFilterController(QObject):
    """Own the active query and reject superseded background results."""

    stateChanged = Signal(object)
    previewReady = Signal(object)
    previewFailed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = StructureFilterState()
        self._dataset = None
        self._job_id = 0

    @property
    def dataset(self):
        return self._dataset

    def set_dataset(self, dataset) -> None:
        self._job_id += 1
        self._dataset = dataset
        self.state.result = None
        self.state.error = None
        self.state.running = False
        self.state.stale = False
        self.stateChanged.emit(self.state)

    def set_spec(self, spec: StructureFilterSpec) -> None:
        if spec == self.state.spec:
            return
        self._job_id += 1
        self.state.spec = spec
        self.state.error = None
        self.state.running = False
        self.state.stale = self.state.result is not None
        self.stateChanged.emit(self.state)

    def invalidate_result(self) -> None:
        self._job_id += 1
        self.state.running = False
        if self.state.result is not None:
            self.state.stale = True
        self.stateChanged.emit(self.state)

    def clear(self) -> None:
        self._job_id += 1
        self.state = StructureFilterState()
        self.stateChanged.emit(self.state)

    def result_is_current(self) -> bool:
        result = self.state.result
        if result is None or self.state.stale or self._dataset is None:
            return False
        return (
            result.spec == self.state.spec
            and result.dataset_version == StructureFilterEngine.dataset_version(self._dataset)
        )

    def preview(self) -> None:
        dataset = self._dataset
        spec = self.state.spec
        if dataset is None:
            error = StructureFilterValidationError("invalid_dataset", "No dataset is loaded.")
            self.state.error = error
            self.previewFailed.emit(error)
            self.stateChanged.emit(self.state)
            return

        self._job_id += 1
        job_id = self._job_id
        dataset_id = id(dataset)
        self.state.running = True
        self.state.error = None
        self.stateChanged.emit(self.state)

        def _compute():
            try:
                return StructureFilterEngine.evaluate(dataset, spec)
            except StructureFilterValidationError as exc:
                return exc

        def _done(value) -> None:
            if job_id != self._job_id or self._dataset is None or id(self._dataset) != dataset_id:
                return
            self.state.running = False
            if isinstance(value, StructureFilterValidationError):
                self.state.error = value
                self.state.stale = self.state.result is not None
                self.previewFailed.emit(value)
            else:
                self.state.result = value
                self.state.error = None
                self.state.stale = False
                self.previewReady.emit(value)
            self.stateChanged.emit(self.state)

        def _failed(message: str) -> None:
            if job_id != self._job_id:
                return
            error = StructureFilterValidationError("search_failed", message)
            self.state.running = False
            self.state.error = error
            self.state.stale = self.state.result is not None
            self.previewFailed.emit(error)
            self.stateChanged.emit(self.state)

        run_in_thread(self, _compute, on_finished=_done, on_error=_failed)
