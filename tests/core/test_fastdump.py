"""Direct coverage for the native LAMMPS atom-block parser."""
from __future__ import annotations

import numpy as np
import pytest

fastdump = pytest.importorskip("NepTrainKit._native._dump")


def _parse(block: bytes, n_atoms: int, tokens: list[int], kinds: list[int], n_tokens: int | None = None):
    width = max(tokens, default=0) + 1 if n_tokens is None else n_tokens
    return fastdump.parse_block(memoryview(block), n_atoms, width, tokens, kinds)


def test_parse_block_returns_typed_columns_and_interned_labels():
    block = b"1 Fe 0.5 1.5 2.5 1 2 3\n2 Ni 3.5 4.5 5.5 -1 -2 -3\n"

    columns, rows, problem_row, problem_width, bad_row, at_eof = _parse(
        block, 2, [0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 0, 0, 0, 0, 0, 0]
    )

    assert (rows, problem_row, problem_width, bad_row, at_eof) == (2, -1, 0, -1, False)
    ids, labels = columns[1]
    assert labels == ["Fe", "Ni"]
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(columns[2], [0.5, 3.5])
    np.testing.assert_allclose(columns[7], [3.0, -3.0])


def test_parse_block_accepts_crlf_and_blank_lines():
    block = b"1 Fe 0 0 0\r\n\r\n2 Ni 1 1 1\r\n"

    columns, rows, problem_row, _width, bad_row, _eof = _parse(block, 2, [0, 2], [1, 0], n_tokens=5)

    assert (rows, problem_row, bad_row) == (2, -1, -1)
    np.testing.assert_allclose(columns[1], [0.0, 1.0])


def test_parse_block_reports_a_partial_trailing_row():
    block = b"1 Fe 0 0 0\n2 Ni 1 1\n"

    columns, rows, problem_row, problem_width, bad_row, at_eof = _parse(
        block, 2, [0, 1, 2, 3, 4], [1, 2, 0, 0, 0]
    )

    assert (rows, problem_row, problem_width, bad_row, at_eof) == (1, 1, 4, -1, True)


def test_parse_block_reports_bad_numeric_tokens():
    block = b"1 Fe 0 0 0\n2 Ni 1 1 bad\n"

    _columns, rows, problem_row, _width, bad_row, _eof = _parse(
        block, 2, [0, 1, 2, 3, 4], [1, 2, 0, 0, 0]
    )

    assert (rows, problem_row, bad_row) == (2, -1, 1)


def test_parse_block_reports_extra_rows_beyond_the_header():
    block = b"1 Fe 0 0 0\n2 Ni 1 1 1\n"

    _columns, rows, problem_row, problem_width, _bad, at_eof = _parse(
        block, 1, [0, 1, 2, 3, 4], [1, 2, 0, 0, 0]
    )

    assert (rows, problem_row, problem_width, at_eof) == (1, 1, -1, False)


def test_parse_block_handles_an_empty_buffer():
    columns, rows, problem_row, _width, bad_row, at_eof = _parse(b"", 0, [0], [1])

    assert (rows, problem_row, bad_row, at_eof) == (0, -1, -1, False)
    assert columns[0].shape == (0,)


def test_parse_blocks_matches_single_frame_parsing():
    first = b"1 Fe 0 0 0 1 2 3\n2 Ni 1 1 1 -1 -2 -3\n"
    second = b"1 Ni 2 2 2 4 5 6\n2 Fe 3 3 3 -4 -5 -6\n"
    block = first + second
    tokens = [0, 1, 2, 3, 4, 5, 6, 7]
    kinds = [1, 2, 0, 0, 0, 0, 0, 0]

    single = [
        fastdump.parse_block(memoryview(first), 2, 8, tokens, kinds),
        fastdump.parse_block(memoryview(second), 2, 8, tokens, kinds),
    ]
    batched = fastdump.parse_blocks(
        block, [0, len(first)], [len(first), len(block)], [2, 2], 8, tokens, kinds, 4
    )

    assert len(batched) == 2
    for expected, actual in zip(single, batched):
        assert expected[1:] == actual[1:]
        for left, right in zip(expected[0], actual[0]):
            if isinstance(left, tuple):
                np.testing.assert_array_equal(left[0], right[0])
                assert left[1] == right[1]
            else:
                np.testing.assert_allclose(left, right)


def test_parse_blocks_keeps_per_frame_label_tables():
    first = b"1 Fe 0 0 0\n2 Ni 1 1 1\n"
    second = b"1 Ni 0 0 0\n2 Fe 1 1 1\n"
    block = first + second

    results = fastdump.parse_blocks(
        block, [0, len(first)], [len(first), len(block)], [2, 2], 5, [0, 1, 2, 3, 4],
        [1, 2, 0, 0, 0], 2
    )

    assert results[0][0][1][1] == ["Fe", "Ni"]
    assert results[1][0][1][1] == ["Ni", "Fe"]
    np.testing.assert_array_equal(results[0][0][1][0], [0, 1])
    np.testing.assert_array_equal(results[1][0][1][0], [0, 1])
