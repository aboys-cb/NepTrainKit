from types import SimpleNamespace

import numpy as np

from NepTrainKit.core.geometry_cache import structure_cell_array


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
