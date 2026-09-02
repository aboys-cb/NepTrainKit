from types import SimpleNamespace

import numpy as np
from ase import Atoms

from NepTrainKit.core.geometry_cache import structure_cell_array, structure_pbc_flags


class _CellWithExplicitStorage:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __array__(self, *args, **kwargs):
        raise AssertionError("wrapped cell array protocol must not be invoked")


def test_structure_cell_array_supports_plain_and_wrapped_cells():
    expected = np.arange(9, dtype=np.float64).reshape(3, 3)
    plain = structure_cell_array(
        SimpleNamespace(cell=expected[:, ::-1]),
        dtype=np.float32,
    )
    wrapped = structure_cell_array(
        SimpleNamespace(cell=_CellWithExplicitStorage(expected)),
        dtype=np.float32,
    )

    assert plain.dtype == np.float32
    assert plain.flags.c_contiguous
    np.testing.assert_array_equal(plain, expected[:, ::-1])
    assert wrapped.dtype == np.float32
    assert wrapped.flags.c_contiguous
    np.testing.assert_array_equal(wrapped, expected)


def test_structure_pbc_flags_uses_ase_state_without_additional_fields():
    structure = Atoms("Fe", cell=np.eye(3), pbc=(True, False, True))

    np.testing.assert_array_equal(
        structure_pbc_flags(structure),
        np.asarray([1, 0, 1], dtype=np.uint8),
    )
