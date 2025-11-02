// SPDX-License-Identifier: GPL-3.0-or-later
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
#include <pybind11/numpy.h>
#include <Python.h>
#include "nep.h"
#include "nep.cpp"
#ifdef _WIN32
#include <windows.h>
#endif
#include <tuple>
#include <atomic>
#include <utility>

namespace py = pybind11;

// Release the GIL only if currently held by this thread.
struct ScopedReleaseIfHeld {
    PyThreadState* state{nullptr};
    ScopedReleaseIfHeld() {
        if (PyGILState_Check()) {
            state = PyEval_SaveThread();
        }
    }
    ~ScopedReleaseIfHeld() {
        if (state) {
            PyEval_RestoreThread(state);
        }
    }
    ScopedReleaseIfHeld(const ScopedReleaseIfHeld&) = delete;
    ScopedReleaseIfHeld& operator=(const ScopedReleaseIfHeld&) = delete;
};
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




std::tuple<pybind11::array, pybind11::array, pybind11::array>
calculate(const std::vector<std::vector<int>>& type,
          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {
    const size_t nframes = type.size();
    size_t total_atoms = 0;
    for (const auto& t : type) total_atoms += t.size();

    double* pot_buf = nullptr;
    double* frc_buf = nullptr;
    double* vir_buf = nullptr;
    size_t cursor = 0;

    {
        // Heavy compute without holding the GIL; build contiguous buffers
        ScopedReleaseIfHeld _gil_release;
        pot_buf = new double[total_atoms];
        frc_buf = new double[total_atoms * 3];
        vir_buf = new double[total_atoms * 9];

        for (size_t i = 0; i < nframes; ++i) {
            check_canceled();
            const size_t Ni = type[i].size();
            std::vector<double> p(Ni);
            std::vector<double> f(Ni * 3);
            std::vector<double> v(Ni * 9);
            compute(type[i], box[i], position[i], p, f, v);
            for (size_t m = 0; m < Ni; ++m) {
                pot_buf[cursor + m] = p[m];
                // force layout from NEP3: [fx[N], fy[N], fz[N]]
                frc_buf[(cursor + m) * 3 + 0] = f[m + 0 * Ni];
                frc_buf[(cursor + m) * 3 + 1] = f[m + 1 * Ni];
                frc_buf[(cursor + m) * 3 + 2] = f[m + 2 * Ni];
                // virial layout from NEP3: [xx[N], xy[N], xz[N], yx[N], yy[N], yz[N], zx[N], zy[N], zz[N]]
                double* row = vir_buf + (cursor + m) * 9;
                row[0] = v[m + 0 * Ni]; // xx
                row[1] = v[m + 1 * Ni]; // xy
                row[2] = v[m + 2 * Ni]; // xz
                row[3] = v[m + 3 * Ni]; // yx
                row[4] = v[m + 4 * Ni]; // yy
                row[5] = v[m + 5 * Ni]; // yz
                row[6] = v[m + 6 * Ni]; // zx
                row[7] = v[m + 7 * Ni]; // zy
                row[8] = v[m + 8 * Ni]; // zz
            }
            cursor += Ni;
        }
    }

    auto c1 = pybind11::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c2 = pybind11::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c3 = pybind11::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    std::vector<std::ptrdiff_t> shp_p{static_cast<pybind11::ssize_t>(cursor)};
    std::vector<std::ptrdiff_t> shp_f{static_cast<pybind11::ssize_t>(cursor), 3};
    std::vector<std::ptrdiff_t> shp_v{static_cast<pybind11::ssize_t>(cursor), 9};
    pybind11::array ap(pybind11::dtype::of<double>(), shp_p,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(pot_buf), c1);
    pybind11::array af(pybind11::dtype::of<double>(), shp_f,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(frc_buf), c2);
    pybind11::array av(pybind11::dtype::of<double>(), shp_v,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(vir_buf), c3);
    return std::make_tuple(ap, af, av);
}

std::tuple<pybind11::array, pybind11::array, pybind11::array, pybind11::array>
calculate(const std::vector<std::vector<int>>& type,
          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position,
          const std::vector<std::vector<double>>& spin
          ) {
    const size_t nframes = type.size();
    size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
    double* pot_buf = nullptr; double* frc_buf = nullptr; double* mf_buf = nullptr; double* vir_buf = nullptr;
    size_t cursor = 0;
    {
        ScopedReleaseIfHeld _gil_release;
        pot_buf = new double[total_atoms];
        frc_buf = new double[total_atoms * 3];
        mf_buf  = new double[total_atoms * 3];
        vir_buf = new double[total_atoms * 9];
        for (size_t i = 0; i < nframes; ++i) {
            check_canceled();
            const size_t Ni = type[i].size();
            std::vector<double> p(Ni), f(Ni * 3), v(Ni * 9), mf(Ni * 3);
            compute(type[i], box[i], position[i], spin[i], p, f, v, mf);
            for (size_t m = 0; m < Ni; ++m) {
                pot_buf[cursor + m] = p[m];
                // forces
                frc_buf[(cursor + m) * 3 + 0] = f[m + 0 * Ni];
                frc_buf[(cursor + m) * 3 + 1] = f[m + 1 * Ni];
                frc_buf[(cursor + m) * 3 + 2] = f[m + 2 * Ni];
                // magnetic forces
                mf_buf[(cursor + m) * 3 + 0] = mf[m + 0 * Ni];
                mf_buf[(cursor + m) * 3 + 1] = mf[m + 1 * Ni];
                mf_buf[(cursor + m) * 3 + 2] = mf[m + 2 * Ni];
                // virials (9 components in NEP3 order)
                double* row = vir_buf + (cursor + m) * 9;
                row[0] = v[m + 0 * Ni]; // xx
                row[1] = v[m + 1 * Ni]; // xy
                row[2] = v[m + 2 * Ni]; // xz
                row[3] = v[m + 3 * Ni]; // yx
                row[4] = v[m + 4 * Ni]; // yy
                row[5] = v[m + 5 * Ni]; // yz
                row[6] = v[m + 6 * Ni]; // zx
                row[7] = v[m + 7 * Ni]; // zy
                row[8] = v[m + 8 * Ni]; // zz
            }
            cursor += Ni;
        }
    }
    auto c1 = pybind11::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c2 = pybind11::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c3 = pybind11::capsule(mf_buf,  [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c4 = pybind11::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    std::vector<std::ptrdiff_t> shp_p{static_cast<pybind11::ssize_t>(cursor)};
    std::vector<std::ptrdiff_t> shp_f{static_cast<pybind11::ssize_t>(cursor), 3};
    std::vector<std::ptrdiff_t> shp_v{static_cast<pybind11::ssize_t>(cursor), 9};
    pybind11::array ap(pybind11::dtype::of<double>(), shp_p,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(pot_buf), c1);
    pybind11::array af(pybind11::dtype::of<double>(), shp_f,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(frc_buf), c2);
    pybind11::array am(pybind11::dtype::of<double>(), shp_f,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(mf_buf),  c3);
    pybind11::array av(pybind11::dtype::of<double>(), shp_v,
                       std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                       static_cast<void*>(vir_buf), c4);
    return std::make_tuple(ap, af, am, av);
}


std::tuple<pybind11::array, pybind11::array, pybind11::array>
calculate_dftd3(
  const std::string& functional,
  const double D3_cutoff,
  const double D3_cutoff_cn,
const std::vector<std::vector<int>>& type,
          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {
    const size_t nframes = type.size();
    size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
    double* pot_buf = nullptr; double* frc_buf = nullptr; double* vir_buf = nullptr; size_t cursor = 0;
    {
        ScopedReleaseIfHeld _gil_release;
        pot_buf = new double[total_atoms];
        frc_buf = new double[total_atoms * 3];
        vir_buf = new double[total_atoms * 9];
        for (size_t i = 0; i < nframes; ++i) {
            check_canceled();
            const size_t Ni = type[i].size();
            std::vector<double> p(Ni), f(Ni * 3), v(Ni * 9);
            compute_dftd3(functional, D3_cutoff, D3_cutoff_cn, type[i], box[i], position[i], p, f, v);
            for (size_t m = 0; m < Ni; ++m) {
                pot_buf[cursor + m] = p[m];
                frc_buf[(cursor + m) * 3 + 0] = f[m + 0 * Ni];
                frc_buf[(cursor + m) * 3 + 1] = f[m + 1 * Ni];
                frc_buf[(cursor + m) * 3 + 2] = f[m + 2 * Ni];
                double* row = vir_buf + (cursor + m) * 9;
                row[0] = v[m + 0 * Ni];
                row[1] = v[m + 1 * Ni];
                row[2] = v[m + 2 * Ni];
                row[3] = v[m + 3 * Ni];
                row[4] = v[m + 4 * Ni];
                row[5] = v[m + 5 * Ni];
                row[6] = v[m + 6 * Ni];
                row[7] = v[m + 7 * Ni];
                row[8] = v[m + 8 * Ni];
            }
            cursor += Ni;
        }
    }
    auto c1 = pybind11::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c2 = pybind11::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    auto c3 = pybind11::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
    std::vector<std::ptrdiff_t> shp_p{static_cast<pybind11::ssize_t>(cursor)};
    std::vector<std::ptrdiff_t> shp_f{static_cast<pybind11::ssize_t>(cursor), 3};
    std::vector<std::ptrdiff_t> shp_v{static_cast<pybind11::ssize_t>(cursor), 9};
    pybind11::array ap(pybind11::dtype::of<double>(), shp_p, std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(double))}, static_cast<void*>(pot_buf), c1);
    pybind11::array af(pybind11::dtype::of<double>(), shp_f, std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))}, static_cast<void*>(frc_buf), c2);
    pybind11::array av(pybind11::dtype::of<double>(), shp_v, std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))}, static_cast<void*>(vir_buf), c3);
    return std::make_tuple(ap, af, av);
}



std::tuple<pybind11::array, pybind11::array, pybind11::array>
calculate_with_dftd3(
  const std::string& functional,
  const double D3_cutoff,
  const double D3_cutoff_cn,
const std::vector<std::vector<int>>& type,

          const std::vector<std::vector<double>>& box,
          const std::vector<std::vector<double>>& position) {
    const size_t nframes = type.size();
    size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
    float* pot_buf = nullptr; float* frc_buf = nullptr; float* vir_buf = nullptr; size_t cursor = 0;
    {
        ScopedReleaseIfHeld _gil_release;
        pot_buf = new float[total_atoms];
        frc_buf = new float[total_atoms * 3];
        vir_buf = new float[total_atoms * 9];
        for (size_t i = 0; i < nframes; ++i) {
            check_canceled();
            const size_t Ni = type[i].size();
            std::vector<double> p(Ni), f(Ni * 3), v(Ni * 9);
            compute_with_dftd3(functional, D3_cutoff, D3_cutoff_cn, type[i], box[i], position[i], p, f, v);
            for (size_t m = 0; m < Ni; ++m) {
                pot_buf[cursor + m] = static_cast<float>(p[m]);
                frc_buf[(cursor + m) * 3 + 0] = static_cast<float>(f[m + 0 * Ni]);
                frc_buf[(cursor + m) * 3 + 1] = static_cast<float>(f[m + 1 * Ni]);
                frc_buf[(cursor + m) * 3 + 2] = static_cast<float>(f[m + 2 * Ni]);
                float* row = vir_buf + (cursor + m) * 9;
                row[0] = static_cast<float>(v[m + 0 * Ni]);
                row[1] = static_cast<float>(v[m + 1 * Ni]);
                row[2] = static_cast<float>(v[m + 2 * Ni]);
                row[3] = static_cast<float>(v[m + 3 * Ni]);
                row[4] = static_cast<float>(v[m + 4 * Ni]);
                row[5] = static_cast<float>(v[m + 5 * Ni]);
                row[6] = static_cast<float>(v[m + 6 * Ni]);
                row[7] = static_cast<float>(v[m + 7 * Ni]);
                row[8] = static_cast<float>(v[m + 8 * Ni]);
            }
            cursor += Ni;
        }
    }
    auto c1 = pybind11::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    auto c2 = pybind11::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    auto c3 = pybind11::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shp_p{static_cast<pybind11::ssize_t>(cursor)};
    std::vector<std::ptrdiff_t> shp_f{static_cast<pybind11::ssize_t>(cursor), 3};
    std::vector<std::ptrdiff_t> shp_v{static_cast<pybind11::ssize_t>(cursor), 9};
    pybind11::array ap(pybind11::dtype::of<float>(), shp_p, std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(float))}, static_cast<void*>(pot_buf), c1);
    pybind11::array af(pybind11::dtype::of<float>(), shp_f, std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(float)), static_cast<pybind11::ssize_t>(sizeof(float))}, static_cast<void*>(frc_buf), c2);
    pybind11::array av(pybind11::dtype::of<float>(), shp_v, std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(float)), static_cast<pybind11::ssize_t>(sizeof(float))}, static_cast<void*>(vir_buf), c3);
    return std::make_tuple(ap, af, av);
}


    // 获取 descriptor: return flat py::array length N*dim ordered as d0[N], d1[N], ...
    pybind11::array get_descriptor(const std::vector<int>& type,
                                   const std::vector<double>& box,
                                   const std::vector<double>& position) {
        std::vector<double> descriptor;
        {
            ScopedReleaseIfHeld _gil_release;
            descriptor.resize(type.size() * static_cast<size_t>(annmb.dim));
            find_descriptor(type, box, position, descriptor);
        }
        const size_t total = descriptor.size();
        float* buf = new float[total];
        for (size_t i = 0; i < total; ++i) buf[i] = static_cast<float>(descriptor[i]);
        auto c = pybind11::capsule(buf, [](void* p){ delete[] reinterpret_cast<float*>(p); });
        std::vector<std::ptrdiff_t> shp{static_cast<pybind11::ssize_t>(total)};
        return pybind11::array(pybind11::dtype::of<float>(), shp,
                               std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(float))},
                               static_cast<void*>(buf), c);
    }

    // 获取元素列表
    std::vector<std::string> get_element_list() {
        return element_list;
    }

    // 获取所有结构的 descriptor: return (total_atoms, dim) float32
    pybind11::array get_structures_descriptor(
            const std::vector<std::vector<int>>& type,
            const std::vector<std::vector<double>>& box,
            const std::vector<std::vector<double>>& position) {
        const size_t nframes = type.size();
        size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
        const int dim = annmb.dim;
        float* buf = nullptr;
        {
            ScopedReleaseIfHeld _gil_release;
            buf = new float[total_atoms * static_cast<size_t>(dim)];
            size_t cursor = 0;
            for (size_t i = 0; i < nframes; ++i) {
                check_canceled();
                const size_t Ni = type[i].size();
                std::vector<double> struct_des(Ni * static_cast<size_t>(dim));
                find_descriptor(type[i], box[i], position[i], struct_des);
                // struct_des order: d0[N], d1[N] ...
                for (size_t m = 0; m < Ni; ++m) {
                    float* row = buf + (cursor + m) * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        row[d] = static_cast<float>(struct_des[m + static_cast<size_t>(d) * Ni]);
                    }
                }
                cursor += Ni;
            }
        }
        auto cap = pybind11::capsule(buf, [](void* p){ delete[] reinterpret_cast<float*>(p); });
        std::vector<std::ptrdiff_t> shp{static_cast<pybind11::ssize_t>(total_atoms), static_cast<pybind11::ssize_t>(dim)};
        return pybind11::array(pybind11::dtype::of<float>(), shp,
                               std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(dim * sizeof(float)), static_cast<pybind11::ssize_t>(sizeof(float))},
                               static_cast<void*>(buf), cap);
    }
    // 获取所有结构的 polarizability
    pybind11::array get_structures_polarizability(const std::vector<std::vector<int>>& type,
                                     const std::vector<std::vector<double>>& box,
                                     const std::vector<std::vector<double>>& position) {
        const size_t nframes = type.size();
        float* buf = new float[nframes * 6];
        {
            ScopedReleaseIfHeld _gil_release;
            for (size_t i = 0; i < nframes; ++i) {
                check_canceled();
                std::vector<double> struct_pol(6);
                find_polarizability(type[i], box[i], position[i], struct_pol);
                float* row = buf + i * 6;
                for (int k = 0; k < 6; ++k) row[k] = static_cast<float>(struct_pol[k]);
            }
        }
        auto cap = pybind11::capsule(buf, [](void* p){ delete[] reinterpret_cast<float*>(p); });
        std::vector<std::ptrdiff_t> shp{static_cast<pybind11::ssize_t>(nframes), 6};
        return pybind11::array(pybind11::dtype::of<float>(), shp,
                               std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(6*sizeof(float)), static_cast<pybind11::ssize_t>(sizeof(float))},
                               static_cast<void*>(buf), cap);
    }

        // 获取所有结构的 polarizability
    pybind11::array get_structures_dipole(const std::vector<std::vector<int>>& type,
                                     const std::vector<std::vector<double>>& box,
                                     const std::vector<std::vector<double>>& position) {
        const size_t nframes = type.size();
        float* buf = new float[nframes * 3];
        {
            ScopedReleaseIfHeld _gil_release;
            for (size_t i = 0; i < nframes; ++i) {
                check_canceled();
                std::vector<double> struct_dip(3);
                find_dipole(type[i], box[i], position[i], struct_dip);
                float* row = buf + i * 3;
                for (int k = 0; k < 3; ++k) row[k] = static_cast<float>(struct_dip[k]);
            }
        }
        auto cap = pybind11::capsule(buf, [](void* p){ delete[] reinterpret_cast<float*>(p); });
        std::vector<std::ptrdiff_t> shp{static_cast<pybind11::ssize_t>(nframes), 3};
        return pybind11::array(pybind11::dtype::of<float>(), shp,
                               std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(float)), static_cast<pybind11::ssize_t>(sizeof(float))},
                               static_cast<void*>(buf), cap);
    }
};

// pybind11 模块绑定
PYBIND11_MODULE(nep_cpu, m) {
    m.doc() = "A pybind11 module for NEP";

    py::class_<CpuNep>(m, "CpuNep")
        .def(py::init<const std::string&>(), py::arg("potential_filename"))
        .def("calculate", py::overload_cast<const std::vector<std::vector<int>>&, const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&>(&CpuNep::calculate))
        .def("calculate", py::overload_cast<const std::vector<std::vector<int>>&, const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&>(&CpuNep::calculate))

        .def("calculate_with_dftd3", &CpuNep::calculate_with_dftd3)
        .def("calculate_dftd3", &CpuNep::calculate_dftd3)

        .def("cancel", &CpuNep::cancel)
        .def("reset_cancel", &CpuNep::reset_cancel)
        .def("is_canceled", &CpuNep::is_canceled)

        .def("get_descriptor", &CpuNep::get_descriptor)

        .def("get_element_list", &CpuNep::get_element_list)
        .def("get_structures_polarizability", &CpuNep::get_structures_polarizability)
        .def("get_structures_dipole", &CpuNep::get_structures_dipole)

        .def("get_structures_descriptor", &CpuNep::get_structures_descriptor,
             py::arg("type"), py::arg("box"), py::arg("position"));

}
