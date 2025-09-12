/*
    NepTrainKit CPU bindings for NEP
    Copyright (C) 2025 NepTrainKit contributors

    This file is part of NepTrainKit and integrates code from NEP_CPU
    (https://github.com/brucefan1983/NEP_CPU) by Zheyong Fan, Junjie Wang,
    Eric Lindgren and contributors, which is licensed under the GNU
    General Public License, version 3 (or, at your option, any later version).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nep.h"
#include "nep.cpp"
#ifdef _WIN32
#include <windows.h>
#endif
#include <tuple>
#include <atomic>

namespace py = pybind11;

// 计算列的平均值
std::vector<double> calculate_column_averages(const std::vector<std::vector<double>>& arr) {
    std::vector<double> averages;

    if (arr.empty()) return averages;

    size_t num_columns = arr[0].size();

    // 计算每列的平均值
    for (size_t col = 0; col < num_columns; ++col) {
        double sum = 0;
        size_t row_count = arr.size();
        for (size_t row = 0; row < row_count; ++row) {
            sum += arr[row][col];
        }
        averages.push_back(sum / row_count);
    }

    return averages;
}

// 计算行的平均值
std::vector<double> calculate_row_averages(const std::vector<std::vector<double>>& arr) {
    std::vector<double> averages;

    if (arr.empty()) return averages;

    // 遍历每一行
    for (const auto& row : arr) {
        double sum = 0;
        size_t num_elements = row.size();

        // 遍历当前行的每个元素，累加
        for (size_t i = 0; i < num_elements; ++i) {
            sum += row[i];
        }

        // 计算该行的平均值并保存
        averages.push_back(sum / num_elements);
    }

    return averages;
}

// 重塑数组（将一维数组重塑为二维）
void reshape(const std::vector<double>& input, int rows, int cols, std::vector<std::vector<double>>& result) {
    if (input.size() != rows * cols) {
        throw std::invalid_argument("The number of elements does not match the new shape.");
    }

    result.resize(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = input[i * cols + j];
        }
    }
}

// 矩阵转置
void transpose(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output) {
    int rows = input.size();
    int cols = input[0].size();

    // 初始化转置矩阵
    output.resize(cols, std::vector<double>(rows));

    // 执行转置操作
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c][r] = input[r][c];
        }
    }
}

// 转换函数：UTF-8 到系统编码
std::string convert_path(const std::string& utf8_path) {
#ifdef _WIN32
    // Windows：将 UTF-8 转换为 ANSI（例如 GBK）
    int wstr_size = MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    std::wstring wstr(wstr_size, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, &wstr[0], wstr_size);

    int ansi_size = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string ansi_path(ansi_size, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &ansi_path[0], ansi_size, nullptr, nullptr);
    return ansi_path;
#else
    // Linux/macOS：直接返回 UTF-8
    return utf8_path;
#endif
}


class CpuNep : public NEP3 {
public:
    CpuNep(const std::string& potential_filename)  {


    std::string utf8_path  = convert_path(potential_filename);


    init_from_file(utf8_path, false);
    }

private:
    std::atomic<bool> canceled_{false};

    inline void check_canceled() const {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
    }

public:
    void cancel() { canceled_.store(true, std::memory_order_relaxed); }
    void reset_cancel() { canceled_.store(false, std::memory_order_relaxed); }
    bool is_canceled() const { return canceled_.load(std::memory_order_relaxed); }




std::tuple<std::vector<std::vector<double>>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
calculate(const std::vector<std::vector<int>>& type,
          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {
    py::gil_scoped_release _gil_release;

    size_t type_size = type.size();
    std::vector<std::vector<double>> potentials(type_size);  // 预分配空间
    std::vector<std::vector<double>> forces(type_size);      // 预分配空间
    std::vector<std::vector<double>> virials(type_size);     // 预分配空间
    // OpenMP 并行化报错

    for (int i = 0; i < type_size; ++i) {
        check_canceled();

        potentials[i].resize(type[i].size());
        forces[i].resize(type[i].size() * 3);  // 假设 force 是 3D 向量
        virials[i].resize(type[i].size() * 9);  // 假设 virial 是 3x3 矩阵

        // 调用计算函数
        compute(type[i], box[i], position[i],
                potentials[i], forces[i], virials[i]);

    }

    return std::make_tuple(potentials, forces, virials);
}

std::tuple<std::vector<std::vector<double>>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
calculate_dftd3(
  const std::string& functional,
  const double D3_cutoff,
  const double D3_cutoff_cn,
const std::vector<std::vector<int>>& type,
          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {
    py::gil_scoped_release _gil_release;

    size_t type_size = type.size();
    std::vector<std::vector<double>> potentials(type_size);  // 预分配空间
    std::vector<std::vector<double>> forces(type_size);      // 预分配空间
    std::vector<std::vector<double>> virials(type_size);     // 预分配空间
    // OpenMP 并行化报错

    for (int i = 0; i < type_size; ++i) {
        check_canceled();

        potentials[i].resize(type[i].size());
        forces[i].resize(type[i].size() * 3);  // 假设 force 是 3D 向量
        virials[i].resize(type[i].size() * 9);  // 假设 virial 是 3x3 矩阵

        // 调用计算函数
        compute_dftd3(functional,D3_cutoff,D3_cutoff_cn,type[i], box[i], position[i],
                potentials[i], forces[i], virials[i]);

    }

    return std::make_tuple(potentials, forces, virials);
}



std::tuple<std::vector<std::vector<double>>,
           std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
calculate_with_dftd3(
  const std::string& functional,
  const double D3_cutoff,
  const double D3_cutoff_cn,
const std::vector<std::vector<int>>& type,

          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {
    py::gil_scoped_release _gil_release;

    size_t type_size = type.size();
    std::vector<std::vector<double>> potentials(type_size);  // 预分配空间
    std::vector<std::vector<double>> forces(type_size);      // 预分配空间
    std::vector<std::vector<double>> virials(type_size);     // 预分配空间
    // OpenMP 并行化报错

    for (int i = 0; i < type_size; ++i) {
        check_canceled();

        potentials[i].resize(type[i].size());
        forces[i].resize(type[i].size() * 3);  // 假设 force 是 3D 向量
        virials[i].resize(type[i].size() * 9);  // 假设 virial 是 3x3 矩阵

        // 调用计算函数
        compute_with_dftd3(functional,D3_cutoff,D3_cutoff_cn,type[i], box[i], position[i],
                potentials[i], forces[i], virials[i]);

    }

    return std::make_tuple(potentials, forces, virials);
}


    // 获取 descriptor
    std::vector<double> get_descriptor(const std::vector<int>& type,
                                       const std::vector<double>& box,
                                       const std::vector<double>& position) {
        py::gil_scoped_release _gil_release;

        std::vector<double> descriptor(type.size() * annmb.dim);
        find_descriptor(type, box, position, descriptor);
        return descriptor;
    }

    // 获取元素列表
    std::vector<std::string> get_element_list() {
        return element_list;
    }

    // 获取所有结构的 descriptor
    std::vector<std::vector<double>> get_structures_descriptor(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {
    py::gil_scoped_release _gil_release;

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_descriptors(type_size, std::vector<double>(annmb.dim));

        for (int i = 0; i < type_size; ++i) {
            check_canceled();
            std::vector<double> struct_des(type[i].size() * annmb.dim);
            find_descriptor(type[i], box[i], position[i], struct_des);
//
            // 重塑 descriptor 以适应矩阵
            std::vector<std::vector<double>> struct_des_reshaped;
            reshape(struct_des, annmb.dim, type[i].size(), struct_des_reshaped);

            // 计算行平均
            all_descriptors[i] = calculate_row_averages(struct_des_reshaped);
        }

        return all_descriptors;
    }
    // 获取所有结构的 polarizability
    std::vector<std::vector<double>> get_structures_polarizability(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {
    py::gil_scoped_release _gil_release;

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_polarizability(type_size, std::vector<double>(6));

        for (int i = 0; i < type_size; ++i) {
            check_canceled();
            std::vector<double> struct_pol(6);
            find_polarizability(type[i], box[i], position[i], struct_pol);

            all_polarizability[i] = struct_pol;
        }

        return all_polarizability;
    }

        // 获取所有结构的 polarizability
    std::vector<std::vector<double>> get_structures_dipole(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {
    py::gil_scoped_release _gil_release;

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_dipole(type_size, std::vector<double>(3));

        for (int i = 0; i < type_size; ++i) {
            check_canceled();
            std::vector<double> struct_dipole(3);
            find_dipole(type[i], box[i], position[i], struct_dipole);

            all_dipole[i] = struct_dipole;
        }

        return all_dipole;
    }
};

// pybind11 模块绑定
PYBIND11_MODULE(nep_cpu, m) {
    m.doc() = "A pybind11 module for NEP";

    py::class_<CpuNep>(m, "CpuNep")
        .def(py::init<const std::string&>(), py::arg("potential_filename"))
        .def("calculate", &CpuNep::calculate)
        .def("calculate_with_dftd3", &CpuNep::calculate_with_dftd3)
        .def("calculate_dftd3", &CpuNep::calculate_dftd3)

        .def("cancel", &CpuNep::cancel)
        .def("reset_cancel", &CpuNep::reset_cancel)
        .def("is_canceled", &CpuNep::is_canceled)

        .def("get_descriptor", &CpuNep::get_descriptor)

        .def("get_element_list", &CpuNep::get_element_list)
        .def("get_structures_polarizability", &CpuNep::get_structures_polarizability)
        .def("get_structures_dipole", &CpuNep::get_structures_dipole)

        .def("get_structures_descriptor", &CpuNep::get_structures_descriptor);

}
