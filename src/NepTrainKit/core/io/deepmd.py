#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
from PySide6.QtCore import QObject, Signal
from loguru import logger

from .base import NepPlotData, StructureData
from NepTrainKit.core.structure import Structure


class DeepmdResultData(QObject):
    updateInfoSignal = Signal()
    loadFinishedSignal = Signal()

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.select_index = set()
        self.load_flag = False

    @classmethod
    def from_path(cls, path):
        return cls(Path(path))

    def load_structures(self):
        type_map_file = self.data_dir / "type_map.raw"
        if type_map_file.exists():
            type_map = [t.strip() for t in open(type_map_file).read().split()]
        else:
            type_map = []
        coords = np.load(self.data_dir / "coord.npy")
        boxes = np.load(self.data_dir / "box.npy")
        types = np.load(self.data_dir / "type.npy")
        energies = np.load(self.data_dir / "energy.npy") if (self.data_dir / "energy.npy").exists() else None
        forces = np.load(self.data_dir / "force.npy") if (self.data_dir / "force.npy").exists() else None
        virials = np.load(self.data_dir / "virial.npy") if (self.data_dir / "virial.npy").exists() else None
        structures = []
        for i in range(coords.shape[0]):
            species_index = types[i]
            if type_map:
                species = [type_map[j] for j in species_index]
            else:
                species = list(species_index)
            s = Structure.from_deepmd(
                boxes[i],
                species,
                coords[i],
                energy=energies[i] if energies is not None else None,
                forces=forces[i] if forces is not None else None,
                virial=virials[i] if virials is not None else None,
            )
            structures.append(s)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(s) for s in structures])

    def load(self):
        try:
            self.load_structures()
            self._load_dataset()
            self.load_flag = True
        except Exception:
            logger.error("load dataset error", exc_info=True)
        self.loadFinishedSignal.emit()

    @property
    def structure(self):
        return self._atoms_dataset

    @property
    def num(self):
        return self._atoms_dataset.num

    def is_select(self, i):
        return i in self.select_index

    def select(self, indices):
        idx = np.asarray(indices, dtype=int) if not isinstance(indices, int) else np.array([indices])
        idx = np.unique(idx)
        idx = idx[(idx >= 0) & (idx < len(self.structure.all_data)) & (self.structure.data._active_mask[idx])]
        self.select_index.update(idx)
        self.updateInfoSignal.emit()

    def uncheck(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            self.select_index.discard(i)
        self.updateInfoSignal.emit()

    def remove(self, i):
        self.structure.remove(i)
        for dataset in self.dataset:
            dataset.remove(i)
        self.updateInfoSignal.emit()

    @property
    def is_revoke(self):
        return self.structure.remove_data.size != 0

    def revoke(self):
        self.structure.revoke()
        for dataset in self.dataset:
            dataset.revoke()
        self.updateInfoSignal.emit()

    def delete_selected(self):
        self.remove(list(self.select_index))
        self.select_index.clear()
        self.updateInfoSignal.emit()

    def _load_dataset(self):
        if (self.data_dir / "energy.npy").exists():
            energy = np.load(self.data_dir / "energy.npy")
            energy_array = np.column_stack([energy, energy]).astype(np.float32)
        else:
            energy_array = np.array([])
        if (self.data_dir / "force.npy").exists():
            force = np.load(self.data_dir / "force.npy")
            f = force.reshape(force.shape[0] * force.shape[1], 3)
            force_array = np.column_stack([f, f]).astype(np.float32)
            group = np.repeat(np.arange(force.shape[0]), force.shape[1])
        else:
            force_array = np.array([])
            group = 1
        if (self.data_dir / "virial.npy").exists():
            virial = np.load(self.data_dir / "virial.npy")
            v = virial.reshape(virial.shape[0], -1)
            virial_array = np.column_stack([v, v]).astype(np.float32)
            volume = np.array([s.volume for s in self.structure.now_data])
            coefficient = (self.atoms_num_list / volume)[:, np.newaxis]
            stress = virial * coefficient[:, :, None] * 160.21766208
            stress_array = np.column_stack([stress.reshape(virial.shape[0], -1), stress.reshape(virial.shape[0], -1)]).astype(np.float32)
        else:
            virial_array = np.array([])
            stress_array = np.array([])
        self._energy_dataset = NepPlotData(energy_array, title="energy")
        self._force_dataset = NepPlotData(force_array, group_list=group, title="force")
        self._virial_dataset = NepPlotData(virial_array, title="virial")
        self._stress_dataset = NepPlotData(stress_array, title="stress")
        self._descriptor_dataset = NepPlotData([], title="descriptor")

    @property
    def dataset(self):
        return [self.energy, self.force, self.stress, self.virial, self.descriptor]

    @property
    def energy(self):
        return self._energy_dataset

    @property
    def force(self):
        return self._force_dataset

    @property
    def stress(self):
        return self._stress_dataset

    @property
    def virial(self):
        return self._virial_dataset

    @property
    def descriptor(self):
        return self._descriptor_dataset
