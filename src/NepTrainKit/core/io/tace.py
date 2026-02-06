#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse-only loader for TACE prediction EXTXYZ files.

TACE writes predictions back into an EXTXYZ file using fields prefixed with
``TACE_``. This loader compares those predictions against the un-prefixed
reference (DFT) fields and exposes plot datasets compatible with the existing UI.

Notes
-----
- Descriptors are still handled by :class:`~NepTrainKit.core.io.base.ResultData`
  via nep89 (or a user-selected NEP model) for sampling/filtering only.
- This loader does not run any TACE calculations and does not generate NEP-style
  ``*.out`` cache files for energy/force/virial.
- Virial is shown; stress is intentionally not produced or displayed.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from NepTrainKit.config import Config
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.types import ForcesMode, parse_forces_mode
from NepTrainKit.core.utils import aggregate_per_atom_to_structure, concat_nep_dft_array
from NepTrainKit.paths import as_path, get_bundled_nep89_path

from .base import NepPlotData, ResultData


class TaceResultData(ResultData):
    """Result loader for TACE EXTXYZ prediction files."""

    _energy_dataset: NepPlotData
    _force_dataset: NepPlotData
    _virial_dataset: NepPlotData
    _mforce_dataset: NepPlotData | None

    def __init__(
        self,
        nep_txt_path: Path | str,
        data_xyz_path: Path | str,
        descriptor_path: Path | str,
        *,
        import_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(Path(nep_txt_path), Path(data_xyz_path), Path(descriptor_path), import_options=import_options)
        self._mforce_dataset = None

    @property
    def datasets(self) -> list[NepPlotData]:
        items: list[NepPlotData] = [self.energy, self.force, self.virial]
        if self._mforce_dataset is not None:
            items.append(self.mforce)
        items.append(self.descriptor)
        return items

    @property
    def energy(self) -> NepPlotData:
        return self._energy_dataset

    @property
    def force(self) -> NepPlotData:
        return self._force_dataset

    @property
    def virial(self) -> NepPlotData:
        return self._virial_dataset

    @property
    def mforce(self) -> NepPlotData:
        if self._mforce_dataset is None:
            return NepPlotData([], title="mforce")
        return self._mforce_dataset

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        structures: list[Structure] | None = None,
        nep_txt_path: Path | str | None = None,
        import_options: dict[str, Any] | None = None,
    ) -> "TaceResultData":
        dataset_path = as_path(path)
        nep_path = Path(nep_txt_path) if nep_txt_path is not None else get_bundled_nep89_path()
        descriptor_path = dataset_path.with_name(f"descriptor_tace_{dataset_path.stem}.out")
        inst = cls(nep_path, dataset_path, descriptor_path, import_options=import_options)
        if structures is not None:
            try:
                inst.set_structures(structures)
            except Exception:
                pass
        return inst

    @staticmethod
    def _safe_float_array(value: Any) -> npt.NDArray[np.float32]:
        if value is None:
            return np.array([], dtype=np.float32)
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float32).ravel()
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value, dtype=np.float32).ravel()
            except Exception:
                return np.array([], dtype=np.float32)
        text = str(value).strip().replace(",", " ")
        if not text:
            return np.array([], dtype=np.float32)
        parts = [p for p in text.split() if p]
        try:
            return np.asarray(parts, dtype=np.float32).ravel()
        except Exception:
            return np.array([], dtype=np.float32)

    @staticmethod
    def _extract_dft_mforce(structure: Structure) -> npt.NDArray[np.float32] | None:
        """Return DFT magnetic force array from either ``force_mag`` or ``mforce``."""
        props = getattr(structure, "atomic_properties", {}) or {}
        if "force_mag" in props:
            return np.asarray(props["force_mag"], dtype=np.float32).reshape(-1, 3)
        if "mforce" in props:
            return np.asarray(props["mforce"], dtype=np.float32).reshape(-1, 3)
        return None

    @staticmethod
    def _extract_tace_mforce(structure: Structure) -> npt.NDArray[np.float32] | None:
        props = getattr(structure, "atomic_properties", {}) or {}
        key = "TACE_noncollinear_magnetic_forces"
        if key in props:
            return np.asarray(props[key], dtype=np.float32).reshape(-1, 3)
        return None

    def _load_dataset(self) -> None:
        """Build plot datasets from fields inside the EXTXYZ structures."""
        try:
            structures = self.structure.now_data.tolist()
        except Exception:
            structures = []

        if not structures:
            self._energy_dataset = NepPlotData([], title="energy")
            self._force_dataset = NepPlotData([], title="force")
            self._virial_dataset = NepPlotData([], title="virial")
            self._mforce_dataset = None
            return

        # ----- energy -----
        ref_energy = np.array(
            [s.energy if getattr(s, "has_energy", False) else np.nan for s in structures],
            dtype=np.float32,
        )
        pred_energy = np.array(
            [float(getattr(s, "additional_fields", {}).get("TACE_energy", np.nan)) for s in structures],
            dtype=np.float32,
        )
        energy_array = concat_nep_dft_array(pred_energy, ref_energy, quantity="energies")
        energy_array = (energy_array / self.atoms_num_list.reshape(-1, 1)).astype(np.float32, copy=False)
        self._energy_dataset = NepPlotData(energy_array, title="energy")

        # ----- force -----
        def _force_or_nan(structure: Structure, key: str) -> npt.NDArray[np.float32]:
            props = getattr(structure, "atomic_properties", {}) or {}
            if key in props:
                return np.asarray(props[key], dtype=np.float32).reshape(-1, 3)
            return np.full((len(structure), 3), np.nan, dtype=np.float32)

        ref_forces = np.vstack(
            [
                s.forces.astype(np.float32, copy=False).reshape(-1, 3)
                if getattr(s, "has_forces", False)
                else np.full((len(s), 3), np.nan, dtype=np.float32)
                for s in structures
            ],
            dtype=np.float32,
        )
        pred_forces = np.vstack([_force_or_nan(s, "TACE_forces") for s in structures], dtype=np.float32)
        force_array = concat_nep_dft_array(pred_forces, ref_forces, quantity="forces")

        default_forces = parse_forces_mode(Config.get("widget", "forces_data", ForcesMode.Raw))
        if force_array.size != 0 and default_forces == ForcesMode.Norm:
            force_array = aggregate_per_atom_to_structure(force_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)
            self._force_dataset = NepPlotData(force_array, title="force")
        else:
            self._force_dataset = NepPlotData(force_array, group_list=self.atoms_num_list, title="force")

        # ----- virial (no stress) -----
        ref_v6 = np.vstack(
            [s.nep_virial.astype(np.float32, copy=False).reshape(1, 6) if getattr(s, "has_virial", False) else np.full((1, 6), np.nan, dtype=np.float32) for s in structures],
            dtype=np.float32,
        )
        pred_v6_list: list[npt.NDArray[np.float32]] = []
        for s in structures:
            raw = getattr(s, "additional_fields", {}).get("TACE_virials", None)
            arr = self._safe_float_array(raw)
            if arr.size == 9:
                v6 = arr[[0, 4, 8, 1, 5, 6]]
                v6 = v6 / float(len(s) or 1)
            elif arr.size == 6:
                v6 = arr / float(len(s) or 1)
            else:
                v6 = np.full((6,), np.nan, dtype=np.float32)
            pred_v6_list.append(np.asarray(v6, dtype=np.float32).reshape(1, 6))
        pred_v6 = np.vstack(pred_v6_list, dtype=np.float32)
        virial_array = concat_nep_dft_array(pred_v6, ref_v6, quantity="virials")
        self._virial_dataset = NepPlotData(virial_array, title="virial")

        # ----- optional magnetic force (mforce) -----
        has_mforce_all = True
        for s in structures:
            if self._extract_tace_mforce(s) is None:
                has_mforce_all = False
                break
            if self._extract_dft_mforce(s) is None:
                has_mforce_all = False
                break

        if not has_mforce_all:
            self._mforce_dataset = None
            return

        try:
            ref_mf = np.vstack([self._extract_dft_mforce(s) for s in structures], dtype=np.float32)  # type: ignore[arg-type]
            pred_mf = np.vstack([self._extract_tace_mforce(s) for s in structures], dtype=np.float32)  # type: ignore[arg-type]
            mforce_array = concat_nep_dft_array(pred_mf, ref_mf, quantity="magnetic forces")
            if mforce_array.size != 0 and default_forces == ForcesMode.Norm:
                mforce_array = aggregate_per_atom_to_structure(mforce_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)
                self._mforce_dataset = NepPlotData(mforce_array, title="mforce")
            else:
                self._mforce_dataset = NepPlotData(mforce_array, group_list=self.atoms_num_list, title="mforce")
        except Exception:
            logger.debug(traceback.format_exc())
            self._mforce_dataset = None
