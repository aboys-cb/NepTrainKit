#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/21 14:45
# @Author  : 兵
# @email    : 1747193328@qq.com
import glob
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import neighborlist
from ase.geometry import find_mic
from loguru import logger
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from NepTrainKit import utils, module_path



with open(os.path.join(module_path, "Config/ptable.json"), "r", encoding="utf-8") as f:
    table_info = json.loads(f.read())


atomic_numbers={elem_info["symbol"]:elem_info["number"] for elem_info in table_info.values()}

class Structure:
    """
    extxyz格式的结构类
    原子坐标是笛卡尔坐标
    """
    def __init__(self, lattice, structure_info, properties, additional_fields):
        super().__init__()
        self.properties = properties
        self.lattice = np.array(lattice,dtype=np.float32).reshape((3,3))  # Optional: Lattice vectors
        self.structure_info = structure_info
        self.additional_fields = additional_fields
        if "Config_type" not in self.additional_fields.keys():
            self.additional_fields["Config_type"] = ""
        if "force" in self.structure_info.keys():
            self.force_label="force"
        else:
            self.force_label = "forces"

    @property
    def tag(self):
        """Alias for the ``Config_type`` additional field."""
        return self.additional_fields.get("Config_type", "")

    @tag.setter
    def tag(self, value):
        self.additional_fields["Config_type"] = value

    def __len__(self):
        return len(self.elements)

    @classmethod
    def read_xyz(cls, filename):
        with open(filename, 'r') as f:
            structure = cls.parse_xyz(f.read())
        return structure


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
        if  "force_mag" not in self.structure_info :
            return 0
        mag=self.structure_info["force_mag"]
        count = np.sum(~np.all(mag == 0, axis=1))
        return count
    @property
    def formula(self):
        #这种形式会导致化学式过长 比如有机分子环境下
        # diffs = np.diff(self.numbers)
        # # 找到变化的地方，包含第一个元素的起始位置
        # change_points = np.where(diffs != 0)[0] + 1
        # # 在变化的位置之间进行分段
        # segments = np.split(self.elements, change_points)
        # # 格式化每个段落
        # result = [f"{segment[0]}{len(segment)}" for segment in segments]
        # return "".join(result)
        # 改成下面的形式
        formula = ""
        elems={}
        for element in self.elements:
            if element in elems.keys():
                elems[element]+=1
            else:
                elems[element]=1
        for element,count in elems.items():
            formula+=element+str(count)
        return formula

    @property
    def html_formula(self):
        formula = ""
        elems = {}
        for element in self.elements:
            if element in elems.keys():
                elems[element] += 1
            else:
                elems[element] = 1
        for element, count in elems.items():
            formula += element +"<sub>" + str(count) + "</sub>"
        return formula

    @property
    def per_atom_energy(self):
        return self.energy/self.num_atoms
    @property
    def energy(self):
        return self.additional_fields["energy"]
    @energy.setter
    def energy(self,new_energy):
        self.additional_fields["energy"] = new_energy
    @property
    def forces(self):
        return self.structure_info[self.force_label]
    @forces.setter
    def forces(self,arr):
        has_forces=[i["name"]==self.force_label for i in self.properties]
        if not any(has_forces):
            self.properties.append({'name': self.force_label, 'type': 'R', 'count': 3})

        self.structure_info[self.force_label] = arr

    @property
    def virial(self):
        try:
            vir =self.additional_fields["virial"]
        except:
            # 检查下有没有压强
            try:
                vir = self.additional_fields["stress"]  * self.volume * -1
            except:
                raise ValueError("No virial or stress data")
        return vir
    @virial.setter
    def virial(self,new_virial):
        self.additional_fields["virial"] = new_virial

    @property
    def nep_virial(self):

        vir=self.virial
        return vir[[0,4,8,1,5,6]]/self.num_atoms

    @property
    def nep_dipole(self):
        dipole=np.array(self.dipole.split(" "),dtype=np.float32)
        return dipole/self.num_atoms

    @property
    def nep_polarizability(self):
        vir = np.array(self.pol.split(" "), dtype=np.float32)
        return vir[[0,4,8,1,5,6]] / self.num_atoms

    def get_chemical_symbols(self):
        return self.elements

    @property
    def elements(self):
        return self.structure_info['species']

    @property
    def positions(self):
        return self.structure_info['pos']

    @property
    def num_atoms(self):
        return len(self.elements)

    def copy(self):
        return deepcopy(self)

    def set_lattice(self, new_lattice: np.ndarray,in_place=False):
        """
        根据新晶格缩放原子位置，支持原地修改或返回新对象。

        :param new_lattice: 新晶格矩阵（3x3 numpy 数组）
        :param in_place: 是否修改当前对象（默认 False，返回新对象）
        :return: 更新后的 Structure 对象（若 in_place=True，则返回 self）
        """
        target = self if in_place else self.copy()
        old_lattice = target.lattice
        old_positions = target.positions

        # 计算变换矩阵（参考 ASE）
        M = np.linalg.solve(old_lattice, new_lattice)
        new_positions = old_positions @ M

        # 更新晶格和坐标
        target.lattice = new_lattice
        target.structure_info['pos'] = new_positions

        return target

    def supercell(self, scale_factor, order="atom-major", tol=1e-5):
        """
        按指定比例因子扩展晶胞，参考 ASE 的高效实现。
        保持晶格角度不变，并支持按元素排序。

        :param scale_factor: 扩展比例因子（标量或长度为3的数组，对应 a、b、c 方向）
        :param order: 原子排序方式，"cell-major"（默认）或 "atom-major"
        :param tol: 数值容差，用于边界检查
        :return: 扩展后的新 Structure 对象
        :raises ValueError: 如果 scale_factor 无效
        """
        # 输入验证和标准化
        scale_factor = np.asarray(scale_factor, dtype=np.float32)
        if scale_factor.size == 1:
            scale_factor = np.full(3, scale_factor)
        if scale_factor.size != 3:
            raise ValueError("scale_factor 必须是标量或长度为3的数组")
        if scale_factor.min() < 1:
            raise ValueError("scale_factor 必须大于等于 1")

        # 输入验证（同上）
        scale_factor = np.asarray(scale_factor, dtype=np.int64)  # 限制为整数扩展

        # 计算新晶格（各方向独立扩展）
        new_lattice = self.lattice * scale_factor[:, None]

        # 转换原始坐标到分数坐标并包裹
        inv_orig_lattice = np.linalg.inv(self.lattice)
        frac_pos = self.positions @ inv_orig_lattice
        frac_pos = frac_pos % 1.0  # 严格包裹

        # 生成扩展网格（明确轴向顺序）
        n_a, n_b, n_c = scale_factor
        # 生成平移向量时确保a方向优先
        offsets_a = np.arange(n_a)[:, None] * np.array([1, 0, 0])  # a方向偏移
        offsets_b = np.arange(n_b)[:, None] * np.array([0, 1, 0])  # b方向偏移
        offsets_c = np.arange(n_c)[:, None] * np.array([0, 0, 1])  # c方向偏移

        # 计算所有偏移组合（a方向最外层循环）
        full_offsets = (offsets_a[:, None, None] +
                        offsets_b[None, :, None] +
                        offsets_c[None, None, :]).reshape(-1, 3)

        # 扩展分数坐标
        expanded_frac = frac_pos[:, None, :] + full_offsets[None, :, :]
        expanded_frac = expanded_frac.reshape(-1, 3) / scale_factor  # 归一化到新分数坐标

        # 转换到新笛卡尔坐标
        new_positions = expanded_frac @ new_lattice

        # 元素扩展（保持a方向优先顺序）
        if order == "cell-major":
            new_elements = np.tile(self.elements, np.prod(scale_factor))
        elif order == "atom-major":
            new_elements = np.repeat(self.elements, np.prod(scale_factor))
        else:
            raise ValueError( )

        # 更新结构信息
        structure_info = {}
        structure_info['pos'] = new_positions.astype(np.float32)
        structure_info['species'] = new_elements

        properties=[{'name': 'species', 'type': 'S', 'count': 1}, {'name': 'pos', 'type': 'R', 'count': 3}]
        # 设置周期性边界条件（假设与原始一致）
        additional_fields={}
        additional_fields['pbc'] = self.additional_fields.get('pbc', "T T T")
        additional_fields["Config_type"] =self.additional_fields.get('Config_type', "")+f" super cell({scale_factor})"

        return Structure(new_lattice, structure_info, properties, additional_fields)
    def adjust_reasonable(self, coefficient=0.7):
        """
        根据传入系数 对比共价半径和实际键长，
        如果实际键长小于coefficient*共价半径之和，判定为不合理结构 返回False
        否则返回 True
        :param coefficient: 系数
        :return:

        """
        distance_info = self.get_mini_distance_info()
        for elems, bond_length in distance_info.items():
            elem0_info = table_info[str(atomic_numbers[elems[0]])]
            elem1_info = table_info[str(atomic_numbers[elems[1]])]

            # 相邻原子距离小于共价半径之和×系数就选中
            if (elem0_info["radii"] + elem1_info["radii"]) * coefficient > bond_length * 100:
                return False
        return True





    # 在序列化时使用 __getstate__ 进行处理
    def __getstate__(self):
        # 返回对象的状态字典，这里可以控制哪些属性需要序列化

        state = self.__dict__.copy()

        return state

    # 反序列化时使用 __setstate__
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, item):

        if item in self.additional_fields.keys():
            return self.additional_fields[item]
        elif item in self.structure_info.keys():
            return self.structure_info[item]
        else:
            raise AttributeError


    @classmethod
    def parse_xyz(cls, lines):
        """
        Parse a single structure from a list of lines.
        """
        if isinstance(lines, str):
            lines = lines.strip().split('\n')
        # Parse the second line (global properties)
        global_properties = lines[1].strip()
        lattice, properties, additional_fields = cls._parse_global_properties(global_properties)
        array = np.array([line.split() for line in lines[2:]],dtype=object )

        structure_info = {}
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
                _info=_info.astype( np.int8)

            else:
                pass
            if prop["count"] == 1:
                _info = _info.flatten()
            else:
                _info = _info.reshape((-1, prop["count"]))

            structure_info[prop["name"]] = _info
            index += prop["count"]
        del array

        # return
        return cls(lattice, structure_info, properties, additional_fields)

    @classmethod
    def _parse_global_properties(cls, line):
        """
        Parse global properties from the second line of an XYZ block.
        """
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

                    value = value.strip('"')  # 去掉引号
                else:
                    try:
                        value = float(value)
                    except Exception as e:
                        value = value
                if key == "config_type" or key == "Config_type":
                    # 这里是为了后面的Config搜索做统一
                    key = "Config_type"
                    value=str(value)
                if key.lower() in ("energy", "pbc","virial","stress"):
                    key=key.lower()
                if key =="virial" or key =="stress":
                    value= np.array(value.split(" "), dtype=np.float32)
                additional_fields[key] = value
                # print(additional_fields)
        return lattice, properties, additional_fields

    @staticmethod
    def _parse_properties(properties_str):
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
    @utils.timeit
    def read_multiple(filename ):
        """
        Read a multi-structure XYZ file and return a list of Structure objects.
        """


        # data_to_process = []
        structures = []

        with open(filename, "r") as file:
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

    def write(self, file):
        """
        Write the current structure to an XYZ file.
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

            if isinstance(value, (float, int)):
                global_line.append(f"{key}={value}")
            elif isinstance(value, np.ndarray):
                value_str = " ".join(map(str, value.flatten()))
                global_line.append(f'{key}="{value_str}"')


            else:
                global_line.append(f'{key}="{value}"')
        file.write(" ".join(global_line) + "\n")

        for row in range(self.num_atoms):
            line = ""
            for prop  in self.properties :
                if prop["count"] == 1:
                    values=[self.structure_info[prop["name"]][row]]
                else:
                    values=self.structure_info[prop["name"]][row,:]

                if prop["type"] == 'S':  # 字符串类型
                    line += " ".join([f"{x }" for x in values]) + " "

                elif prop["type"] == 'R':  # 浮点数类型
                    line += " ".join([f"{x:.10g}" for x in values]) + " "
                else:
                    line += " ".join([f"{x}" for x in values]) + " "
            file.write(line.strip() + "\n")

    def get_all_distances(self):
        return  calculate_pairwise_distances(self.cell, self.positions,False)

    def get_mini_distance_info(self):
        """
        返回原子对之间的最小距离
        """
        dist_matrix = calculate_pairwise_distances(self.cell, self.positions,False)
        symbols=self.elements
        # 提取上三角矩阵（排除对角线）
        i, j = np.triu_indices(len(self), k=1)
        # 用字典来存储每种元素对的最小键长
        bond_lengths = {}
        # 遍历所有原子对，计算每一对元素的最小键长
        for idx in range(len(i)):
            atom_i, atom_j = symbols[i[idx]], symbols[j[idx]]
            # if atom_i==atom_j:
            #     continue
            # 获取当前键长
            bond_length = dist_matrix[i[idx], j[idx]]
            # if bond_length>5:
            #     continue
            # 确保元素对按字母顺序排列，避免 Cs-Ag 和 Ag-Cs 视为不同
            element_pair = tuple(sorted([atom_i, atom_j]))
            # 如果该元素对尚未存在于字典中，初始化其最小键长
            if element_pair not in bond_lengths:
                bond_lengths[element_pair] = bond_length
            else:
                # 更新最小键长
                bond_lengths[element_pair] = min(bond_lengths[element_pair], bond_length)

        return bond_lengths
    def get_bond_pairs(self):
        """
        返回在范围内的所有键长
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
        """
        根据键长阈值判断
        返回所有的非物理键长
        """
        i, j = np.triu_indices(len(self), k=1)
        distances = self.get_all_distances()
        upper_distances = distances[i, j]
        covalent_radii = np.array([table_info[str(n)]["radii"] / 100 for n in self.numbers])
        radius_sum = covalent_radii[i] + covalent_radii[j]
        bond_mask = (upper_distances < radius_sum * coefficient)

        bad_bond_pairs = [(i[k], j[k]) for k in np.where(bond_mask)[0]]
        return bad_bond_pairs

def calculate_pairwise_distances(lattice_params:np.ndarray, atom_coords:np.ndarray, fractional=True):
    """
    计算晶体中所有原子对之间的距离，考虑周期性边界条件

    参数:
    lattice_params: 晶格参数，3x3 numpy array 表示晶格向量 (a, b, c)
    atom_coords: 原子坐标，Nx3 numpy array
    fractional: 是否为分数坐标 (True) 或笛卡尔坐标 (False)

    返回:
    distances: NxN numpy array，所有原子对之间的距离
    """


    if fractional:
        atom_coords = np.dot(atom_coords, lattice_params)

    diff = atom_coords[np.newaxis, :, :] - atom_coords[:, np.newaxis, :]
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype=np.int8).T.reshape(-1, 3)
    lattice_shifts = np.dot(shifts, lattice_params)
    all_diffs = diff[:, :, np.newaxis, :] + lattice_shifts[np.newaxis, np.newaxis, :, :]
    all_distances = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
    distances = np.min(all_distances, axis=-1)
    np.fill_diagonal(distances, 0)
    return distances



# 判断团簇是否为有机分子
def is_organic_cluster(symbols):
    has_carbon = 'C' in symbols
    organic_elements = {'H', 'O', 'N', 'S', 'P'}
    has_organic_elements = any(symbol in organic_elements for symbol in symbols)
    return has_carbon and has_organic_elements

# # 识别结构中的团簇


def get_clusters(structure):
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
# 解包跨越边界的分子


def unwrap_molecule(structure, cluster_indices):
    pos = structure.positions[cluster_indices]
    cell = structure.cell
    ref_pos = pos[0]

    # 所有位置相对 ref_pos 的位移
    delta = pos - ref_pos

    # 求解最小镜像向量
    inv_cell = np.linalg.inv(cell.T)  # 提前计算逆矩阵
    frac_delta = np.dot(delta, inv_cell)
    frac_delta -= np.round(frac_delta)
    mic_delta = np.dot(frac_delta, cell.T)

    unwrapped_pos = ref_pos + mic_delta
    return unwrapped_pos


# 封装循环部分：处理有机分子团簇
def process_organic_clusters(structure, new_structure, clusters, is_organic_list):
    """处理有机分子团簇并更新原子位置"""

    for cluster_indices, is_organic in zip(clusters, is_organic_list):
        if is_organic:
            # 解包分子
            unwrapped_pos = unwrap_molecule(structure, cluster_indices)

            # 计算解包后质心
            center_unwrapped = np.mean(unwrapped_pos, axis=0)

            # 将质心转换到分数坐标并映射回晶胞内
            scaled_center = np.dot(center_unwrapped, np.linalg.inv(structure.cell)) % 1.0
            center_original = np.dot(scaled_center, structure.cell)

            # 计算原子相对于质心的位移
            delta_pos = unwrapped_pos - center_unwrapped

            # 在新晶胞中计算新质心
            center_new = np.dot(scaled_center, new_structure.cell)

            # 新位置 = 新质心 + 原始位移
            pos_new = center_new + delta_pos

            # 更新原子位置
            new_structure.positions[cluster_indices] = pos_new
    new_structure.wrap()



def _load_npy_structure(folder):
    structures = []
    type_map_path = os.path.join(folder, "type_map.raw")
    type_path = os.path.join(folder, "type.raw")

    if  not os.path.exists(type_path):
        return structures


    # Load once and reuse
    type_ = np.loadtxt(type_path, dtype=int,ndmin=1)

    if   os.path.exists(type_map_path) :
        type_map = np.loadtxt(type_map_path, dtype=str, ndmin=1)
    else:
        type_map=np.array([f"Type_{i+1}" for i in np.unique(type_)], dtype=str, ndmin=1)
    # Use np.array and list comprehension for faster element mapping


    elem_list = type_map[type_]

    atoms_num = len(elem_list)
    nopbc = os.path.isfile(os.path.join(folder, "nopbc"))

    sets = sorted(glob.glob(os.path.join(folder, "set.*")))
    dataset_dict = {}

    # Load data from files and use np.concatenate instead of vstack
    for _set in sets:
        for data_path in Path(_set).iterdir():
            key = data_path.stem
            data = np.load(data_path)

            if key in dataset_dict:
                dataset_dict[key].append(data)  # Collect in lists for later concatenation
            else:
                dataset_dict[key] = [data]

    config_type = os.path.basename(folder)

    # Efficient concatenation outside the loop
    for key in dataset_dict.keys():
        dataset_dict[key] = np.concatenate(dataset_dict[key], axis=0)
    logger.debug(f"load {dataset_dict['box'].shape[0]} structures from {folder}" )
    for index in range(dataset_dict["box"].shape[0]):
        box = dataset_dict["box"][index].reshape(3, 3)
        coords = dataset_dict["coord"][index].reshape(-1, 3)

        properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ]
        info = {
            "species": elem_list,
            "pos": coords,
        }
        additional_fields = {"Config_type": config_type}
        additional_fields["pbc"] = "F F F" if nopbc else "T T T"

        # Optimize adding properties and additional fields
        for key in dataset_dict.keys():
            if key not in ["box", "coord"]:
                prop = dataset_dict[key][index]
                count = prop.shape[0]

                if count > atoms_num and key!="virial":
                    col = count // atoms_num

                    info[key] = prop.reshape((-1, col))
                    properties.append({"name": key, "type": "R", "count": col})
                else:
                    if count == 1:
                        additional_fields[key] = prop[0]
                    else:

                        additional_fields[key] = prop.flatten()

        structure = Structure(lattice=box, structure_info=info, properties=properties,
                              additional_fields=additional_fields)
        structures.append(structure)

    return structures


def load_npy_structure(folders):
    if os.path.exists(os.path.join(folders, "type.raw")):
        return _load_npy_structure(folders)
    else:
        structures = []
        if os.path.isdir(folders):
            for folder in Path(folders).iterdir():
                structures.extend(load_npy_structure(folder))

        return structures

@utils.timeit
def save_npy_structure(folder, structures):
    """
    保存结构信息到指定的文件夹，根据Config_type将数据组织
    :param folder: 保存的目标文件夹
    :param structures: 包含结构信息的Structure列表
    """


    # 确保文件夹存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 创建用于保存数据的字典

    dataset_dict = defaultdict(lambda: defaultdict(list))

    # 遍历所有结构并收集数据
    for structure in structures:
        # 从结构对象中提取数据
        config_type=structure.tag
        dataset_dict[config_type]["box"].append(structure.lattice.flatten())  # 确保box是3x3矩阵展平为1D数组
        dataset_dict[config_type]["coord"].append(structure.structure_info["pos"].flatten())  # 确保coords是1D数组
        dataset_dict[config_type]["species"].append(structure.structure_info["species"])

        # 保存每个额外字段（如果有）
        for prop_info  in structure.properties:
            name=prop_info["name"]
            if name not in [  "species", "pos"]:
                dataset_dict[config_type][name].append(structure.structure_info[name].flatten())
        if "virial" in structure.additional_fields:
            virial = structure.additional_fields["virial"]
            dataset_dict[config_type]["virial"].append(virial)
        if "energy" in structure.additional_fields:
            dataset_dict[config_type]["energy"].append(structure.energy)


    for config ,data in dataset_dict.items():
        save_path = os.path.join(folder, config,"set.000")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        species=data["species"][0]
        unique_species = list(set([species   for species in species]))
        np.savetxt(os.path.join(folder,config,  "type_map.raw"), unique_species, fmt="%s")
        type_data = np.array([unique_species.index(species)   for species in species]).flatten()
        np.savetxt(os.path.join(folder, config, "type.raw"), type_data, fmt="%d")
        # 保存额外字段（如果有）
        for key, value in data.items():
            if key =="species":
                continue


            np.save(os.path.join(save_path, f"{key}.npy"), np.vstack(value ))


