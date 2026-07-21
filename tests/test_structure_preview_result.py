from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from NepTrainKit.core.io import StructurePreviewResultData


def test_structure_preview_loads_without_nep_prediction(tmp_path: Path):
    path = tmp_path / "generated.xyz"
    ase_write(
        path,
        [
            Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=[8, 8, 8]),
            Atoms(
                "Fe2",
                positions=[[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]],
                cell=[8, 8, 8],
            ),
        ],
        format="extxyz",
    )
    result = StructurePreviewResultData(path)

    result.load()

    assert result.load_flag
    assert result.num == 2
    assert not hasattr(result, "nep_calc")
    assert len(result.datasets) == 1
    assert result.datasets[0].x.tolist() == [0.0, 1.0]
    assert result.datasets[0].y.tolist() == [1.0, 2.0]
    assert result.datasets[0].parity_mode is False
