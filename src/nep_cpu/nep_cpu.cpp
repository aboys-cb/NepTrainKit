// SPDX-License-Identifier: GPL-3.0-or-later
/*
    NepTrainKit CPU bindings for NEP
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
#include <vector>
#include <string>
#include <stdexcept>

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

// 转换函数：UTF-8 到系统编码
std::string convert_path(const std::string& utf8_path) {
#ifdef _WIN32
    int wstr_size = MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    std::wstring wstr(static_cast<size_t>(wstr_size), 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, &wstr[0], wstr_size);

    int ansi_size = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string ansi_path(static_cast<size_t>(ansi_size), 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &ansi_path[0], ansi_size, nullptr, nullptr);
    return ansi_path;
#else
    return utf8_path;
#endif
}


class CpuNep : public NEP3 {
public:
    explicit CpuNep(const std::string& potential_filename)  {
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

    double* pot_buf = nullptr;
    double* frc_buf = nullptr;
    double* vir_buf = nullptr;
    size_t cursor = 0;
    {
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
            compute_dftd3(functional,D3_cutoff,D3_cutoff_cn,type[i], box[i], position[i],
                    p, f, v);
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
    double* pot_buf = nullptr;
    double* frc_buf = nullptr;
    double* vir_buf = nullptr;
    size_t cursor = 0;

    {
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

            compute_with_dftd3(functional,D3_cutoff,D3_cutoff_cn,type[i], box[i], position[i],
                    p, f, v);
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

    // 获取 descriptor
    std::vector<double> get_descriptor(const std::vector<int>& type,
                                       const std::vector<double>& box,
                                       const std::vector<double>& position) {
        ScopedReleaseIfHeld _gil_release;

        std::vector<double> descriptor(static_cast<size_t>(type.size()) * static_cast<size_t>(annmb.dim));
        find_descriptor(type, box, position, descriptor);
        return descriptor;
    }

    // 获取元素列表
    std::vector<std::string> get_element_list() {
        return element_list;
    }

    // 获取所有结构的 descriptor
    std::vector<std::vector<double>> get_structures_descriptor(
            const std::vector<std::vector<int>>& type,
            const std::vector<std::vector<double>>& box,
            const std::vector<std::vector<double>>& position) {
        ScopedReleaseIfHeld _gil_release;

        const size_t type_size = type.size();
        size_t total_atoms = 0;
        for (const auto& t : type) {
            total_atoms += t.size();
        }
        std::vector<std::vector<double>> all_descriptors;
        all_descriptors.reserve(total_atoms);

        for (size_t i = 0; i < type_size; ++i) {
            check_canceled();
            std::vector<double> struct_des(type[i].size() * annmb.dim);
            find_descriptor(type[i], box[i], position[i], struct_des);

            const size_t atom_count = type[i].size();
            for (size_t atom_idx = 0; atom_idx < atom_count; ++atom_idx) {
                std::vector<double> atom_descriptor(static_cast<size_t>(annmb.dim));
                for (int dim_idx = 0; dim_idx < annmb.dim; ++dim_idx) {
                    const size_t offset = static_cast<size_t>(dim_idx) * atom_count + atom_idx;
                    atom_descriptor[static_cast<size_t>(dim_idx)] = struct_des[offset];
                }
                all_descriptors.emplace_back(std::move(atom_descriptor));
            }
        }

        return all_descriptors;
    }
    // 获取所有结构的 polarizability
    std::vector<std::vector<double>> get_structures_polarizability(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {
    ScopedReleaseIfHeld _gil_release;

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_polarizability(type_size, std::vector<double>(6));

        for (int i = 0; i < static_cast<int>(type_size); ++i) {
            check_canceled();
            std::vector<double> struct_pol(6);
            find_polarizability(type[i], box[i], position[i], struct_pol);

            all_polarizability[static_cast<size_t>(i)] = struct_pol;
        }

        return all_polarizability;
    }

        // 获取所有结构的 dipole
    std::vector<std::vector<double>> get_structures_dipole(const std::vector<std::vector<int>>& type,
                                                     const std::vector<std::vector<double>>& box,
                                                     const std::vector<std::vector<double>>& position) {
    ScopedReleaseIfHeld _gil_release;

        size_t type_size = type.size();
        std::vector<std::vector<double>> all_dipole(type_size, std::vector<double>(3));

        for (int i = 0; i < static_cast<int>(type_size); ++i) {
            check_canceled();
            std::vector<double> struct_dipole(3);
            find_dipole(type[i], box[i], position[i], struct_dipole);

            all_dipole[static_cast<size_t>(i)] = struct_dipole;
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

        .def("get_structures_descriptor", &CpuNep::get_structures_descriptor,
             py::arg("type"), py::arg("box"), py::arg("position"));

}
