"""Lightweight result data used to preview generated card output."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import QCoreApplication

from NepTrainKit.core import MessageManager
from NepTrainKit.core.precision import get_storage_float_dtype

from .base import NepPlotData, ResultData


def _tr(text: str) -> str:
    return QCoreApplication.translate("StructurePreviewResultData", text)


class StructurePreviewResultData(ResultData):
    """Load structures for inspection without running an NEP prediction."""

    is_structure_preview = True

    def __init__(self, data_path: str | Path):
        path = Path(data_path)
        super().__init__(
            nep_txt_path=path,
            data_xyz_path=path,
            descriptor_path=path.with_suffix(".preview-descriptor"),
        )
        self._preview_dataset = NepPlotData([], title="structure")
        self._descriptor_dataset = NepPlotData([], title="descriptor")
        self._descriptor_raw_all = np.array([], dtype=np.float32)

    @property
    def datasets(self) -> list[NepPlotData]:
        return [self._preview_dataset]

    def load(self) -> None:
        """Parse structures and build one cheap structure-size overview plot."""
        try:
            self.load_structures()
            if self.cancel_event.is_set() or self._atoms_dataset.num == 0:
                return

            count = int(self.atoms_num_list.shape[0])
            indices = np.arange(count, dtype=get_storage_float_dtype())
            atom_counts = self.atoms_num_list.astype(
                get_storage_float_dtype(), copy=False
            )
            values = np.column_stack((atom_counts, indices))
            dataset = NepPlotData(values, title="structure")
            dataset.display_title = _tr("Structure overview")
            dataset.x_label = _tr("Structure index")
            dataset.y_label = _tr("Atoms")
            dataset.parity_mode = False
            dataset.show_rmse = False
            self._preview_dataset = dataset
            self.load_flag = True
        except Exception as error:
            logger.error(traceback.format_exc())
            MessageManager.send_error_message(
                _tr("Failed to load generated structures: {message}").format(
                    message=error
                )
            )
        finally:
            self.loadFinishedSignal.emit()

    def _load_dataset(self) -> None:
        """The preview dataset is built directly in :meth:`load`."""

