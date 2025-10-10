#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Utilities shared by NEP file importers."""

import re
import traceback
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
from loguru import logger

from NepTrainKit.core import MessageManager


def get_rmse(array1: npt.NDArray[np.floating], array2: npt.NDArray[np.floating]) -> float:
    """Return the root mean squared error between two arrays."""
    return float(np.sqrt(((array1 - array2) ** 2).mean()))


def read_nep_in(file_name: str | Path) -> dict[str, str]:
    """Parse ``nep.in`` key-value pairs into a dictionary."""
    file_path = Path(file_name)
    if not file_path.exists():
        return {}
    run_in: dict[str, str] = {}
    try:
        content = file_path.read_text(encoding="utf8")
        groups = re.findall(r"^([A-Za-z_]+)\s+([^#\n]*)", content, re.MULTILINE)
        for key, value in groups:
            run_in[key.strip()] = value.strip()
    except Exception:  # noqa: BLE001
        logger.debug(traceback.format_exc())
        MessageManager.send_warning_message("read nep.in file error")
        run_in = {}
    return run_in


def check_fullbatch(run_in: dict[str, str], structure_num: int) -> bool:
    """Return ``True`` when configuration implies full-batch prediction."""
    if run_in.get("prediction") == "1":
        return True
    return int(run_in.get("batch", 1000)) >= structure_num


def read_nep_out_file(file_path: Path | str, **kwargs) -> npt.NDArray[np.float32]:
    """Load ``nep`` output data if the file exists, otherwise return empty array."""
    path = Path(file_path)
    if path.exists():
        data = np.loadtxt(path, **kwargs)
        logger.info("Reading file: {}, shape: {}", path, data.shape)
        return data
    return np.array([])

def split_by_natoms(array, natoms_list:list[int]) -> list[npt.NDArray]:
    """Split a flat array into sub-arrays according to the number of atoms in each structure."""
    if array.size == 0:
        return array
    counts = np.asarray(list(natoms_list), dtype=int)
    split_indices = np.cumsum(counts)[:-1]
    split_arrays = np.split(array, split_indices)
    return split_arrays
def aggregate_per_atom_to_structure(
    array: npt.NDArray[np.float32],
    atoms_num_list: Iterable[int],
    map_func=np.linalg.norm,
    axis: int = 0,
) -> npt.NDArray[np.float32]:
    """Aggregate per-atom data into per-structure values based on atom counts."""
    split_arrays = split_by_natoms(array, atoms_num_list)
    func = partial(map_func, axis=axis)
    return np.array(list(map(func, split_arrays)))


def get_nep_type(file_path: Path | str) -> int:
    """Return the NEP type identifier encoded within ``nep.txt``."""
    nep_type_to_model_type = {
        "nep3": 0,
        "nep3_zbl": 0,
        "nep3_dipole": 1,
        "nep3_polarizability": 2,
        "nep4": 0,
        "nep4_zbl": 0,
        "nep4_dipole": 1,
        "nep4_polarizability": 2,
        "nep4_zbl_temperature": 3,
        "nep4_temperature": 3,
        "nep5": 0,
        "nep5_zbl": 0,
    }
    path = Path(file_path)
    try:
        first_line = path.read_text().splitlines()[0]
        nep_type = first_line.split()[0]
        return nep_type_to_model_type.get(nep_type, 0)
    except (IndexError, FileNotFoundError):
        return 0
    except Exception as error:  # noqa: BLE001
        logger.warning(f"An error occurred while parsing {path}: {error}")
        return 0


def get_xyz_nframe(path: Path | str) -> int:
    """Return the frame count of an ``.xyz`` file."""
    file_path = Path(path)
    if not file_path.exists():
        return 0
    content = file_path.read_text(encoding="utf8")
    nums = re.findall(r"^(\d+)$", content, re.MULTILINE)
    return len(nums)

def concat_nep_dft_array(nep_array: npt.NDArray[np.float32],dft_array:npt.NDArray[np.float32],deepmd=False) -> npt.NDArray[np.float32]:
    """Concatenate ``nep_array`` with ``dft_array``."""

    if nep_array.size == 0:
        np.nan_to_num(dft_array, copy=False, nan=0.0)
        nep_dft_array = np.column_stack([dft_array, dft_array])
    else:
        mask = np.isnan(dft_array)
        if mask.any():
            MessageManager.send_warning_message("use nep3 calculator to calculate result replace the original result")

        dft_array[mask] = nep_array[mask]
        if deepmd:
            nep_dft_array = np.column_stack([dft_array, nep_array])
        else:
            nep_dft_array = np.column_stack([nep_array, dft_array])

    return nep_dft_array