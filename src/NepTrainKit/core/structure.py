#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""High-level data structures and helpers for NEP datasets."""
# @Time    : 2024/11/21 14:45
# @email    : 1747193328@qq.com
from __future__ import annotations
import json
import re
from copy import deepcopy
from pathlib import Path
from functools import cached_property
from typing import Any,IO

import numpy as np
import numpy.typing as npt
from ase import neighborlist

from loguru import logger
from scipy.sparse.csgraph import connected_components
from collections import defaultdict, Counter
from NepTrainKit.utils import timeit
from NepTrainKit.config import Config
from NepTrainKit.paths import PathLike, as_path, ensure_directory

from NepTrainKit import module_path
ptable_path = module_path / "Config/ptable.json"
with ptable_path.open("r", encoding="utf-8") as f:
    table_info = json.loads(f.read())


atomic_numbers={elem_info["symbol"]:elem_info["number"] for elem_info in table_info.values()}

class Structure:
    """Container for EXTXYZ frames (lattice, positions, species, and fields).

    Notes
    -----
    - Coordinates are stored in Cartesian Angstroms under ``pos``.
    - Frame-level attributes like ``energy``, ``pbc``, and ``virial`` live in ``additional_fields``.
    """

    def __init__(self,
                 lattice: list[float]|npt.NDArray[np.float32],
                 atomic_properties:dict[str,npt.NDArray[np.float32]],
                 properties:list[dict[str,str]],
                 additional_fields:dict[str,Any]  ):
        """Initialize a Structure instance.

        Parameters
        ----------
        lattice : list[float] or numpy.ndarray
            Lattice matrix in Angstroms with shape (3, 3).
        atomic_properties : dict[str, numpy.ndarray]
            Per-atom arrays. Must include "pos" of shape (N, 3).
        properties : list[dict[str, str]]
            EXTXYZ Properties descriptors for per-atom arrays.
        additional_fields : dict[str, Any]
            Frame-level attributes (e.g., "energy", "pbc", "virial").
        """
        super().__init__()
        self.properties = properties
        self.lattice = np.array(lattice,dtype=np.float32).reshape((3,3))  # Optional: Lattice vectors
        self.atomic_properties = atomic_properties
        self.additional_fields = additional_fields
        if "Config_type" not in self.additional_fields.keys():
            self.additional_fields["Config_type"] = Config.get("widget", "default_config_type", "neptrainkit")
        if "force" in self.atomic_properties.keys():
            self.force_label="force"
        else:
            self.force_label = "forces"
        self.formula
    @property
    def tag(self)->str:
        """Alias for the ``Config_type`` additional field."""
        return self.additional_fields.get("Config_type", "")

    @tag.setter
    def tag(self, value):
        self.additional_fields["Config_type"] = value
    def get_prop_key(self,additional_fields=True,atomic_properties=True)->list[str]:
        """List all property keys available on the structure.

        Parameters
        ----------
        additional_fields : bool, default=True
            Include keys from ``additional_fields``.
        atomic_properties : bool, default=True
            Include keys from ``atomic_properties``.

        Returns
        -------
        list of str
            Combined keys such as ["pos", "energy", ...].
        """
        keys=[]
        if additional_fields:
            keys.extend(self.additional_fields.keys())
        if atomic_properties:
            keys.extend(self.atomic_properties.keys())
        return keys
    def remove_atomic_properties(self,key:str):
        if key in self.atomic_properties:
            self.atomic_properties.pop(key)
            for prop in self.properties:
                if prop["name"]==key:
                    self.properties.remove(prop)
                    break

    def __len__(self):
        return len(self.elements)

    @classmethod
    def read_xyz(cls, filename:str) -> Structure:
        """Read a single EXTXYZ structure from a file path.

        Parameters
        ----------
        filename : str
            Path to an .xyz file containing a single frame.

        Returns
        -------
        Structure
            Parsed structure instance.
        """
        with open(filename, 'r',encoding="utf8") as f:
            structure = cls.parse_xyz(f.read())
        return structure

    @staticmethod
    def iter_read_multiple(filename: str, cancel_event=None):
        
        """Iterate frames in a multi-structure EXTXYZ file.

        Parameters
        ----------
        filename : str
            Path to a multi-frame .xyz file.
        cancel_event : threading.Event or None, optional
            If provided and is_set(), stop early.

        Yields
        ------
        Structure
            Parsed structures one by one.
        """
        with open(filename, "r",encoding="utf8") as file:
            while True:
                if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                    return
                num_atoms_line = file.readline()
                if not num_atoms_line:
                    break
                num_atoms_line = num_atoms_line.strip()
                if not num_atoms_line:
                    continue

                num_atoms = int(num_atoms_line)

                global_line = file.readline()
                if not global_line:
                    break
                structure_lines = [num_atoms_line, global_line.rstrip()]
                for _ in range(num_atoms):
                    if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                        return
                    line = file.readline()
                    if not line:
                        break
                    structure_lines.append(line.rstrip())

                yield Structure.parse_xyz(structure_lines)




    @property
    def cell(self):
        return self.lattice

    @property
    def volume(self):
        return np.abs(np.linalg.det(self.lattice))

    @property
    def abc(self):
        """Return lattice vector lengths (a, b, c)."""
        return np.linalg.norm(self.lattice, axis=1)

    @property
    def angles(self):
        """Return lattice angles (alpha, beta, gamma) in degrees."""
        a_vec, b_vec, c_vec = self.lattice

        def _angle(v1, v2):
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_ang = np.clip(cos_ang, -1.0, 1.0)
            return np.degrees(np.arccos(cos_ang))

        alpha = _angle(b_vec, c_vec)
        beta = _angle(a_vec, c_vec)
        gamma = _angle(a_vec, b_vec)
        return np.array([alpha, beta, gamma], dtype=np.float32)

    @property
    def numbers(self):
        return [atomic_numbers[element] for element in self.elements ]
    @property
    def spin_num(self)->int:
        if  "force_mag" not in self.atomic_properties :
            return 0
        mag=self.atomic_properties["force_mag"]
        count = np.sum(~np.all(mag == 0, axis=1))
        return count
    @cached_property
    def formula(self):
        # diffs = np.diff(self.numbers)
        # change_points = np.where(diffs != 0)[0] + 1
        # segments = np.split(self.elements, change_points)
        # result = [f"{segment[0]}{len(segment)}" for segment in segments]
        # return "".join(result)
        # formula = ""
        # elems={}
        # for element in self.elements:
        #     if element in elems.keys():
        #         elems[element]+=1
        #     else:
        #         elems[element]=1
        # for element,count in elems.items():
        #     formula+=element+str(count)
        # return formula
        return self.__get_formula(sub=False)



    @cached_property
    def html_formula(self)->str:

        return self.__get_formula(sub=True)

    def __get_formula(self, sub=False)->str:
        # priority = {'C': 0, 'H': 1}

        def priority(sym):
            if sym == 'C':
                return (0, 0)
            if sym == 'H':
                return (0, 1)
            z = atomic_numbers.get(sym, 999)
            return (1, z) if z != 999 else (2, sym)

        cnt = Counter(self.elements)
        count2 = dict(sorted(cnt.items(), key=lambda kv: priority(kv[0])))
        if sub:
            formula = ''.join(symb + ("<sub>" + str(n) + "</sub>" if n > 1 else '')
                              for symb, n in count2.items())
        else:

            formula = ''.join(symb + (str(n) if n > 1 else '')
                              for symb, n in count2.items())
        return formula
    @property
    def per_atom_energy(self):
        """Energy per atom (same units as energy)."""
        return self.energy/self.num_atoms
    @property
    def energy(self):
        """Total energy stored in additional_fields['energy']."""
        return self.additional_fields["energy"]
    @energy.setter
    def energy(self,new_energy:float):
        self.additional_fields["energy"] = new_energy
    @property
    def forces(self):
        """Per-atom forces array with shape (N, 3)."""
        return self.atomic_properties[self.force_label]
    @forces.setter
    def forces(self,arr:npt.NDArray[np.float32]):
        has_forces=[i["name"]==self.force_label for i in self.properties]
        if not any(has_forces):
            self.properties.append({'name': self.force_label, 'type': 'R', 'count': 3})

        self.atomic_properties[self.force_label] = arr
    @property
    def has_virial(self):
        """Whether virial or stress data is available."""
        return "virial" in self.additional_fields or "stress" in self.additional_fields

    @property
    def virial(self):
        """Return virial; convert stress to virial when only stress is present.

        Returns
        -------
        numpy.ndarray
            Virial vector/matrix flattened. If only stress is present, uses
            virial = -stress * volume.
        """
        try:
            vir =self.additional_fields["virial"]
        except:
            try:
                vir = self.additional_fields["stress"]  * self.volume * -1
            except:
                raise ValueError("No virial or stress data")
        return vir
    @virial.setter
    def virial(self,new_virial:npt.NDArray[np.float32]):
        self.additional_fields["virial"] = new_virial

    @property
    def nep_virial(self):

        """Virial in NEP 6-component order per atom: [xx, yy, zz, xy, yz, zx]."""
        vir=self.virial
        return vir[[0,4,8,1,5,6]]/self.num_atoms

    @property
    def nep_dipole(self):
        """Dipole moment per atom parsed from additional_fields['dipole']."""
        dipole=np.array(self.dipole.split(" "),dtype=np.float32)
        return dipole/self.num_atoms

    @property
    def nep_polarizability(self):
        """Polarizability in NEP 6-component order per atom from additional_fields['pol']."""
        vir = np.array(self.pol.split(" "), dtype=np.float32)
        return vir[[0,4,8,1,5,6]] / self.num_atoms

    def get_chemical_symbols(self):
        return self.elements

    @property
    def elements(self):
        """Per-atom chemical symbols (shape (N,))."""
        return self.atomic_properties['species']

    @property
    def positions(self):
        """Cartesian positions in Angstroms with shape (N, 3)."""
        return self.atomic_properties['pos']

    @property
    def num_atoms(self):
        """Number of atoms in the structure."""
        return len(self.elements)

    def copy(self):
        """Deep copy the structure and its arrays."""
        return deepcopy(self)

    def set_lattice(self, new_lattice: npt.NDArray[np.float32],in_place=False):
        """Scale positions to a new lattice and update pos.

        Parameters
        ----------
        new_lattice : numpy.ndarray
            New lattice matrix in Angstroms with shape (3, 3).
        in_place : bool, default=False
            If True, modify this object; otherwise return a copy.

        Returns
        -------
        Structure
            Updated structure (self if in_place=True).
        """
        target = self if in_place else self.copy()
        old_lattice = target.lattice
        old_positions = target.positions

        M = np.linalg.solve(old_lattice, new_lattice)
        new_positions = old_positions @ M

        target.lattice = new_lattice
        target.atomic_properties['pos'] = new_positions

        return target

    def supercell(self, scale_factor, order="atom-major", tol=1e-5)->Structure:
        
        scale_factor = np.asarray(scale_factor, dtype=np.float32)
        if scale_factor.size == 1:
            scale_factor = np.full(3, scale_factor)
        if scale_factor.size != 3:
            raise ValueError("scale_factor must be an array-like of length 3")
        if scale_factor.min() < 1:
            raise ValueError("scale_factor must be >= 1")

        scale_factor = np.asarray(scale_factor, dtype=np.int64)

        new_lattice = self.lattice * scale_factor[:, None]

        inv_orig_lattice = np.linalg.inv(self.lattice)
        frac_pos = self.positions @ inv_orig_lattice
        frac_pos = frac_pos % 1.0

        n_a, n_b, n_c = scale_factor
        offsets_a = np.arange(n_a)[:, None] * np.array([1, 0, 0])
        offsets_b = np.arange(n_b)[:, None] * np.array([0, 1, 0])
        offsets_c = np.arange(n_c)[:, None] * np.array([0, 0, 1])

        full_offsets = (offsets_a[:, None, None] +
                        offsets_b[None, :, None] +
                        offsets_c[None, None, :]).reshape(-1, 3)

        expanded_frac = frac_pos[:, None, :] + full_offsets[None, :, :]
        expanded_frac = expanded_frac.reshape(-1, 3) / scale_factor

        new_positions = expanded_frac @ new_lattice

        if order == "cell-major":
            new_elements = np.tile(self.elements, int(np.prod(scale_factor)))
        elif order == "atom-major":
            new_elements = np.repeat(self.elements, np.prod(scale_factor))
        else:
            raise ValueError( )

        atomic_properties = {}
        atomic_properties['pos'] = new_positions.astype(np.float32)
        atomic_properties['species'] = new_elements

        properties=[{'name': 'species', 'type': 'S', 'count': 1}, {'name': 'pos', 'type': 'R', 'count': 3}]
        additional_fields={}
        additional_fields['pbc'] = self.additional_fields.get('pbc', "T T T")
        additional_fields["Config_type"] =self.additional_fields.get('Config_type', "")+f" super cell({scale_factor})"

        return Structure(new_lattice, atomic_properties, properties, additional_fields)
    def adjust_reasonable(self, coefficient=0.7)->bool:
        distance_info = self.get_mini_distance_info()
        for elems, bond_length in distance_info.items():
            elem0_info = table_info[str(atomic_numbers[elems[0]])]
            elem1_info = table_info[str(atomic_numbers[elems[1]])]

            if (elem0_info["radii"] + elem1_info["radii"]) * coefficient > bond_length * 100:
                return False
        return True





    def __getstate__(self):

        state = self.__dict__.copy()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, item):

        if item in self.additional_fields.keys():
            return self.additional_fields[item]
        elif item in self.atomic_properties.keys():
            return self.atomic_properties[item]
        else:
            raise AttributeError


    @classmethod
    def parse_xyz(cls, lines:list[str]|str)->Structure:
        
        if isinstance(lines, str):
            lines = lines.strip().split('\n')
        # Parse the second line (global properties)
        global_properties = lines[1].strip()
        lattice, properties, additional_fields = cls._parse_global_properties(global_properties)
        array = np.array([line.split() for line in lines[2:]],dtype=object )

        atomic_properties = {}
        index = 0

        for prop in properties:

            _info = array[:, index:index + prop["count"]]
            #
            # _info =[row[index:index + prop["count"]] for row in array]

            if prop["type"] == "S":
                pass
                _info=_info.astype( np.str_)

            elif prop["type"] == "R":
                _info=_info.astype( np.float32)

            elif prop["type"] == "L":
                _info=_info.astype( np.bool_)
            elif  prop["type"] == "I":
                _info=_info.astype( np.int32)

            else:
                pass
            if prop["count"] == 1:
                _info = _info.flatten()
            else:
                _info = _info.reshape((-1, prop["count"]))

            atomic_properties[prop["name"]] = _info
            index += prop["count"]
        del array

        # return
        return cls(lattice, atomic_properties, properties, additional_fields)

    @classmethod
    def _parse_global_properties(cls, line:str)->tuple[list[float],list[dict[str,Any]],dict[str,Any]]:
        
        pattern = r'(\w+)=\s*"([^"]+)"|(\w+)=([\S]+)'
        matches = re.findall(pattern, line)
        properties = []
        lattice = None
        additional_fields = {}

        for match in matches:
            key = match[0] or match[2]
            # key=key.capitalize()
            value = match[1] or match[3]

            if key.capitalize()  == "Lattice":
                lattice = list(map(float, value.split()))
            elif key.capitalize()  == "Properties":
                # Parse Properties details
                properties = cls._parse_properties(value)
            else:

                if '"' in value:

                    value = value.strip('"')
                else:
                    try:
                        value = float(value)
                    except Exception as e:
                        value = value
                if key == "config_type" or key == "Config_type":
                    key = "Config_type"
                    value=str(value)
                if key.lower() in ("energy", "pbc","virial","stress"):
                    key=key.lower()
                if key =="virial" or key =="stress":
                    value= np.array(str(value).split(" "), dtype=np.float32)   # pyright:ignore
                additional_fields[key] = value
                # print(additional_fields)
        return lattice, properties, additional_fields

    @staticmethod
    def _parse_properties(properties_str)->list[dict[str,Any]]:
        """
        Parse `Properties` attribute string to extract atom-specific fields.
        """
        tokens = properties_str.split(":")
        parsed_properties = []
        i = 0
        while i < len(tokens):
            name = tokens[i]
            dtype = tokens[i + 1]
            count = int(tokens[i + 2]) if i + 2 < len(tokens) else 1
            parsed_properties.append({"name": name, "type": dtype, "count": count})
            i += 3
        return parsed_properties

    @staticmethod
    @timeit
    def read_multiple(filename:str ):
        """Read a multi-structure XYZ file and return a list of Structure objects."""



        # data_to_process = []
        structures = []

        with open(filename, "r",encoding="utf8") as file:
            while True:
                num_atoms_line = file.readline()
                if not num_atoms_line:
                    break
                num_atoms_line = num_atoms_line.strip()
                if not num_atoms_line:
                    continue
                num_atoms = int(num_atoms_line)
                structure_lines = [num_atoms_line, file.readline().rstrip()]  # global properties
                for _ in range(num_atoms):
                    line = file.readline()
                    if not line:
                        break
                    structure_lines.append(line.rstrip())

                structure = Structure.parse_xyz(structure_lines)
                structures.append(structure)
                del structure_lines

        return structures

    def write(self, file:IO):
        """Write the structure as an EXTXYZ frame to a file-like object.

        Parameters
        ----------
        file : IO
            Open text stream supporting write().
        """


        # Write number of atoms
        file.write(f"{self.num_atoms}\n")

        # Write global properties
        global_line = []
        if self.lattice.size!=0:
            global_line.append(f'Lattice="' + ' '.join(f"{x}" for x in self.cell.flatten()) + '"')

        props = ":".join(f"{p['name']}:{p['type']}:{p['count']}" for p in self.properties)
        global_line.append(f"Properties={props}")
        for key, value in self.additional_fields.items():

            if isinstance(value, (float, int,np.number)):
                global_line.append(f"{key}={value}")
            elif isinstance(value, np.ndarray):
                value_str = " ".join(map(str, value.flatten()))
                global_line.append(f'{key}="{value_str}"')
            elif isinstance(value, (list,set,tuple)):
                value_str = " ".join(map(str, value ))
                global_line.append(f'{key}="{value_str}"')
            else:
                global_line.append(f'{key}="{value}"')
        file.write(" ".join(global_line) + "\n")

        for row in range(self.num_atoms):
            line = ""
            for prop  in self.properties :
                if prop["count"] == 1:
                    values=[self.atomic_properties[prop["name"]][row]]
                else:
                    values= self.atomic_properties[prop["name"]][row, :]

                if prop["type"] == 'S':
                    line += " ".join([f"{x }" for x in values]) + " "

                elif prop["type"] == 'R':
                    line += " ".join([f"{x:.10g}" for x in values]) + " "
                else:
                    line += " ".join([f"{x}" for x in values]) + " "
            file.write(line.strip() + "\n")

    def get_all_distances(self):
        """Compute all-pairs distances using periodic minimum image.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (N, N) with distances.
        """
        return  calculate_pairwise_distances(self.cell, self.positions, False)

    def get_mini_distance_info(self):
        
        """Minimum interatomic distance for each element pair.

        Returns
        -------
        dict[tuple[str, str], float]
            Map from element pair (A, B) with A <= B to the minimal distance
            across the structure.
        """
        cell = np.asarray(self.cell, dtype=np.float32).reshape(3, 3)
        inv_cell = np.linalg.inv(cell)
        coords = np.asarray(self.positions, dtype=np.float32).reshape(-1, 3)
        frac = coords @ inv_cell

        symbols = np.asarray([str(s) for s in self.elements])
        uniq_elems, codes = np.unique(symbols, return_inverse=True)
        E = uniq_elems.shape[0]
        indices_by_code: list[np.ndarray] = [np.where(codes == c)[0] for c in range(E)]

        min_mat = np.full((E, E), np.inf, dtype=np.float32)

        metric = cell @ cell.T  # 3x3

        N = frac.shape[0]
        if N == 0:
            return {}
        target_bytes = 80 * 1024 * 1024
        per_pair_bytes = 4.0 * (3.0 + 1.0)  # df(3) + d(1)
        block_size = int(max(128, min(1024, target_bytes / (per_pair_bytes * N))))

        for i0 in range(0, N, block_size):
            i1 = min(N, i0 + block_size)
            df = frac[i0:i1, None, :] - frac[None, :, :]  # (bs, N, 3)
            df -= np.round(df)
            tmp = df @ metric  # (bs,N,3)
            d2 = np.einsum('ijk,ijk->ij', df, tmp, dtype=np.float32)
            for row, ii in enumerate(range(i0, i1)):
                d2[row, ii] = np.inf
            for row, ii in enumerate(range(i0, i1)):
                ci = int(codes[ii])
                drow = d2[row]
                for cj in range(E):
                    idxs = indices_by_code[cj]
                    if idxs.size == 0:
                        continue
                    m = float(np.min(drow[idxs]))
                    if not np.isfinite(m):
                        continue
                    a, b = (ci, cj) if ci <= cj else (cj, ci)
                    if m < min_mat[a, b]:
                        min_mat[a, b] = m

        out: dict[tuple[str, str], float] = {}
        for a in range(E):
            for b in range(a, E):
                val = float(min_mat[a, b])
                if np.isfinite(val):
                    out[(str(uniq_elems[a]), str(uniq_elems[b]))] = np.sqrt(val, dtype=np.float32).item()
        return out

    def get_bond_pairs(self):
        
        """Likely bonded pairs using a covalent-radii heuristic.

        Returns
        -------
        list[tuple[int, int]]
            Upper-triangular pairs where distance < 1.15 * (r_i + r_j).
        """
        i, j = np.triu_indices(len(self), k=1)
        pos = np.array(self.positions)
        diff = pos[i] - pos[j]
        upper_distances = np.linalg.norm(diff, axis=1)
        covalent_radii = np.array([table_info[str(n)]["radii"] / 100 for n in self.numbers])
        radius_sum = covalent_radii[i] + covalent_radii[j]
        bond_mask = (upper_distances < radius_sum * 1.15)
        bond_pairs = [(i[k], j[k]) for k in np.where(bond_mask)[0]]
        return bond_pairs

    def get_bad_bond_pairs(self, coefficient=0.8):
        
        """Pairs that violate a short-bond threshold.

        Parameters
        ----------
        coefficient : float, default=0.8
            Threshold relative to the sum of covalent radii.

        Returns
        -------
        list[tuple[int, int]]
            Upper-triangular pairs shorter than the threshold.
        """
        i, j = np.triu_indices(len(self), k=1)
        distances = self.get_all_distances()
        upper_distances = distances[i, j]
        covalent_radii = np.array([table_info[str(n)]["radii"] / 100 for n in self.numbers])
        radius_sum = covalent_radii[i] + covalent_radii[j]
        bond_mask = (upper_distances < radius_sum * coefficient)
        bad_bond_pairs = [(i[k], j[k]) for k in np.where(bond_mask)[0]]
        return bad_bond_pairs



def calculate_pairwise_distances(lattice_params:npt.NDArray[np.float32],
                                 atom_coords:npt.NDArray[np.float32],
                                 fractional=True,
                                 block_size:int=2048
                                 )->npt.NDArray[np.float32]:
    """All-pairs distances under periodic minimum-image convention.

    Parameters
    ----------
    lattice_params : numpy.ndarray
        Lattice matrix with shape (3, 3).
    atom_coords : numpy.ndarray
        Coordinates with shape (N, 3).
    fractional : bool, default=True
        If True, ``atom_coords`` are fractional; otherwise Cartesian.
    block_size : int, default=2048
        Row-block size to balance memory and speed for large N.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape (N, N).
    """


    # Convert to fractional coordinates for minimum image without enumerating 27 images
    cell = np.asarray(lattice_params, dtype=np.float32).reshape(3, 3)
    coords = np.asarray(atom_coords, dtype=np.float32).reshape(-1, 3)
    if fractional:
        frac = coords.astype(np.float32)
    else:
        # frac = cart @ inv(cell)
        inv_cell = np.linalg.inv(cell)
        frac = coords @ inv_cell

    N = frac.shape[0]
    dmat = np.empty((N, N), dtype=np.float32)
    # Process in row blocks to reduce peak memory from O(N^2*27)
    for i0 in range(0, N, max(1, int(block_size))):
        i1 = min(N, i0 + int(block_size))
        # fractional differences with MIC wrapping into [-0.5, 0.5)
        df = frac[i0:i1, None, :] - frac[None, :, :]
        df -= np.round(df)
        # back to Cartesian and compute norms
        dr = df @ cell
        d = np.sqrt(np.sum(dr * dr, axis=2), dtype=np.float32)
        dmat[i0:i1, :] = d
    np.fill_diagonal(dmat, 0.0)
    return dmat



def is_organic_cluster(symbols:list[str]) -> bool:
    """Check whether the structure represents an organic molecular cluster."""
    has_carbon = 'C' in symbols
    organic_elements = {'H', 'O', 'N', 'S', 'P'}
    has_organic_elements = any(symbol in organic_elements for symbol in symbols)
    return has_carbon and has_organic_elements



def get_clusters(structure):
    """Connected-atom clusters under ASE natural cutoffs.

    Parameters
    ----------
    structure : ase.Atoms
        ASE atoms object used for neighbor analysis.

    Returns
    -------
    tuple[list[list[int]], list[bool]]
        Cluster index lists and a boolean list marking organic clusters.
    """
    cutoff = neighborlist.natural_cutoffs(structure)
    nl = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(structure)
    matrix = nl.get_connectivity_matrix()
    n_components, component_list = connected_components(matrix)

    component_array = np.array(component_list)
    all_symbols = [atom.symbol for atom in structure]

    clusters = []
    is_organic_list = []
    for i in range(n_components):
        cluster_indices = np.where(component_array == i)[0].tolist()
        cluster_symbols = [all_symbols[j] for j in cluster_indices]
        clusters.append(cluster_indices)
        is_organic_list.append(is_organic_cluster(cluster_symbols))

    return clusters, is_organic_list


def unwrap_molecule(structure, cluster_indices):
    """Unwrap atoms in a molecular cluster back into the primary simulation cell."""
    pos = structure.positions[cluster_indices]
    cell = structure.cell
    ref_pos = pos[0]

    delta = pos - ref_pos

    inv_cell = np.linalg.inv(cell.T)
    frac_delta = np.dot(delta, inv_cell)
    frac_delta -= np.round(frac_delta)
    mic_delta = np.dot(frac_delta, cell.T)

    unwrapped_pos = ref_pos + mic_delta
    return unwrapped_pos


def process_organic_clusters(structure, new_structure, clusters, is_organic_list):

    """Recenter and unwrap organic molecular clusters.

    Parameters
    ----------
    structure : ase.Atoms
        Original ASE atoms with the reference cell.
    new_structure : ase.Atoms
        Target ASE atoms whose positions will be updated.
    clusters : list[list[int]]
        Atom-index clusters from get_clusters().
    is_organic_list : list[bool]
        Flags indicating organic clusters.
    """
    for cluster_indices, is_organic in zip(clusters, is_organic_list):
        if is_organic:
            unwrapped_pos = unwrap_molecule(structure, cluster_indices)

            center_unwrapped = np.mean(unwrapped_pos, axis=0)

            scaled_center = np.dot(center_unwrapped, np.linalg.inv(structure.cell)) % 1.0
            center_original = np.dot(scaled_center, structure.cell)

            delta_pos = unwrapped_pos - center_unwrapped

            center_new = np.dot(scaled_center, new_structure.cell)

            pos_new = center_new + delta_pos

            new_structure.positions[cluster_indices] = pos_new
    new_structure.wrap()




def _load_npy_structure(folder: PathLike, cancel_event=None):
    
    """Load a DeepMD-style dataset from numpy files into Structure objects.

    Parameters
    ----------
    folder : PathLike
        Root folder containing type.raw, optional type_map.raw, and set.* subfolders.
    cancel_event : threading.Event or None, optional
        If provided and is_set(), stop early.

    Returns
    -------
    list[Structure]
        Loaded structures labeled by Config_type from the folder name.
    """
    structures: list[Structure] = []
    folder_path = as_path(folder)
    type_map_path = folder_path / 'type_map.raw'
    type_path = folder_path / 'type.raw'

    if not type_path.exists():
        return structures

    type_ = np.loadtxt(type_path, dtype=int, ndmin=1)
    if type_map_path.exists():
        type_map = np.loadtxt(type_map_path, dtype=str, ndmin=1)
    else:
        type_map = np.array([f'Type_{i + 1}' for i in range(np.max(type_) + 1)], dtype=str, ndmin=1)

    elem_list = type_map[type_]
    atoms_num = len(elem_list)
    nopbc = (folder_path / 'nopbc').is_file()
    sets = sorted(folder_path.glob('set.*'))
    dataset_dict: dict[str, list[np.ndarray]] = {}
    for set_path in sets:
        if cancel_event is not None and getattr(cancel_event, 'is_set', None) and cancel_event.is_set():
            return structures
        for data_path in set_path.iterdir():
            if cancel_event is not None and getattr(cancel_event, 'is_set', None) and cancel_event.is_set():
                return structures
            key = data_path.stem
            data = np.load(data_path)
            if data.ndim == 1:
                data = data.reshape(data.shape[0], -1)
            dataset_dict.setdefault(key, []).append(data)

    config_type = folder_path.name
    for key in list(dataset_dict.keys()):
        dataset_dict[key] = np.concatenate(dataset_dict[key], axis=0)
    logger.debug(f"load {dataset_dict['box'].shape[0]} structures from {folder_path}" )
    for index in range(dataset_dict['box'].shape[0]):
        if cancel_event is not None and getattr(cancel_event, 'is_set', None) and cancel_event.is_set():
            break
        box = dataset_dict['box'][index].reshape(3, 3)
        coords = dataset_dict['coord'][index].reshape(-1, 3)
        properties = [
            {'name': 'species', 'type': 'S', 'count': 1},
            {'name': 'pos', 'type': 'R', 'count': 3},
        ]
        info = {'species': elem_list, 'pos': coords}
        additional_fields = {'Config_type': config_type, 'pbc': 'F F F' if nopbc else 'T T T',"type_map": type_map.tolist()}
        for key, value in dataset_dict.items():
            if key in {'box', 'coord'}:
                continue

            prop = value[index]
            count = prop.shape[0]
            if count > atoms_num and key != 'virial':
                col = count // atoms_num
                info[key] = prop.reshape((-1, col))
                properties.append({'name': key, 'type': 'R', 'count': col})
            else:
                if count == 1:
                    additional_fields[key] = prop[0]
                else:
                    additional_fields[key] = prop.flatten()
        structure = Structure(
            lattice=box,
            atomic_properties=info,
            properties=properties,
            additional_fields=additional_fields,
        )
        structures.append(structure)
    return structures

def load_npy_structure(folders: PathLike, cancel_event=None):
    """Recursively load DeepMD datasets beneath ``folders``."""
    folder_path = as_path(folders)
    if (folder_path / 'type.raw').exists():
        return _load_npy_structure(folder_path, cancel_event=cancel_event)
    structures: list[Structure] = []
    if folder_path.is_dir():
        for child in folder_path.iterdir():
            if cancel_event is not None and getattr(cancel_event, 'is_set', None) and cancel_event.is_set():
                break
            structures.extend(load_npy_structure(child, cancel_event=cancel_event))
    return structures

def get_type_map(structures: list[Structure]) -> list[str]:
    global_type_map = []
    for structure in structures:
        type_map = structure.additional_fields.get("type_map", structure.atomic_properties["species"])
        global_type_map.extend(map(str, type_map))
    return list(dict.fromkeys(global_type_map))

@timeit
def save_npy_structure(folder: PathLike, structures: list[Structure],type_map:list[str]|None=None):


    """Save structures to a DeepMD-style .npy dataset layout.

    Parameters
    ----------
    folder : PathLike
        Target root folder. One subfolder per Config_type is created.
    structures : list[Structure]
        Structures to persist. Per-atom arrays are saved under set.000 and
        per-frame values under the config folder.
    type_map: list[str]
    """
    target_root = as_path(folder)
    ensure_directory(target_root)

    dataset_dict = defaultdict(lambda: defaultdict(list))
    if type_map is None:
        type_map = get_type_map(structures)

    for structure in structures:
        # config_type=structure.tag
        config_type=structure.formula
        species=structure.atomic_properties["species"]
        type_data = np.array([type_map.index(item) for item in species]).flatten()
        sort_index = np.argsort(type_data)
        dataset_dict[config_type]["type"] = type_data[sort_index]
        dataset_dict[config_type]["box"].append(structure.lattice.flatten())
        dataset_dict[config_type]["coord"].append(structure.atomic_properties["pos"][sort_index].flatten())


        for prop_info  in structure.properties:
            name=prop_info["name"]
            if name not in [  "species", "pos"]:
                dataset_dict[config_type][name].append(structure.atomic_properties[name][sort_index].flatten())
        if "virial" in structure.additional_fields:
            virial = structure.additional_fields["virial"]
            dataset_dict[config_type]["virial"].append(virial)
        if "energy" in structure.additional_fields:
            dataset_dict[config_type]["energy"].append(structure.energy)


    for config, data in dataset_dict.items():
        save_path = ensure_directory(target_root / config / 'set.000')

        np.savetxt(target_root / config / 'type_map.raw', type_map, fmt='%s')

        np.savetxt(target_root / config / 'type.raw', data['type'], fmt='%d')
        for key, value in data.items():
            if key == 'type':
                continue
            np.save(save_path / f'{key}.npy', np.vstack(value))




