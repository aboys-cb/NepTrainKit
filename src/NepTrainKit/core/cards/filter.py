"""UI-independent dataset filtering operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io import farthest_point_sampling
from NepTrainKit.core.types import NepBackend

from .operation import DatasetOperation


@dataclass(frozen=True)
class FPSFilterParams:
    """Parameters for descriptor-space farthest point sampling."""

    nep_path: str
    n_samples: int = 100
    min_distance: float = 0.01
    backend: str = "auto"
    batch_size: int = 1000


class FPSFilterOperation(DatasetOperation):
    """Select representative structures using NEP descriptors and FPS."""

    def run_dataset(self, dataset, params: FPSFilterParams) -> list:
        nep_path = Path(params.nep_path)
        if not nep_path.exists():
            raise FileNotFoundError(f"NEP file does not exist: {nep_path}")

        nep_calc = NepCalculator(
            model_file=str(nep_path),
            backend=NepBackend(params.backend),
            batch_size=int(params.batch_size),
        )
        desc_array = nep_calc.get_structures_descriptor(dataset)
        remaining_indices = farthest_point_sampling(
            desc_array,
            n_samples=int(params.n_samples),
            min_dist=float(params.min_distance),
        )
        return [dataset[i] for i in remaining_indices]
