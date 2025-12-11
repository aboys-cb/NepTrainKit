// SPDX-License-Identifier: GPL-3.0-or-later
/*
    NepTrainKit CPU bindings for qNEP (charge-enabled NEP4)
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <atomic>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include "qnep.h"

#ifdef _WIN32
#include <windows.h>
#endif

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

// Convert UTF-8 paths on Windows to the local ANSI code page so C++ IO can open files.
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

class CpuQNep : public QNEP {
public:
    explicit CpuQNep(const std::string& potential_filename) {
        std::string utf8_path = convert_path(potential_filename);
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

    std::tuple<
        pybind11::array,
        pybind11::array,
        pybind11::array,
        pybind11::array,
        pybind11::array>
    calculate(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position) {

        const size_t nframes = type.size();
        size_t total_atoms = 0;
        for (const auto& t : type) total_atoms += t.size();

        double* pot_buf = nullptr;
        double* frc_buf = nullptr;
        double* vir_buf = nullptr;
        double* chg_buf = nullptr;
        double* bec_buf = nullptr;
        size_t cursor = 0;

        {
            ScopedReleaseIfHeld _gil_release;

            pot_buf = new double[total_atoms];
            frc_buf = new double[total_atoms * 3];
            vir_buf = new double[total_atoms * 9];
            chg_buf = new double[total_atoms];
            bec_buf = new double[total_atoms * 9];

            for (size_t i = 0; i < nframes; ++i) {
                check_canceled();

                const size_t n_atoms = type[i].size();
                std::vector<double> potentials(n_atoms);
                std::vector<double> forces(n_atoms * 3);
                std::vector<double> virials(n_atoms * 9);
                std::vector<double> charges(n_atoms);
                std::vector<double> becs(n_atoms * 9);

                compute(type[i], box[i], position[i],
                        potentials, forces, virials,
                        charges, becs);

                for (size_t m = 0; m < n_atoms; ++m) {
                    pot_buf[cursor + m] = potentials[m];
                    frc_buf[(cursor + m) * 3 + 0] = forces[m + 0 * n_atoms];
                    frc_buf[(cursor + m) * 3 + 1] = forces[m + 1 * n_atoms];
                    frc_buf[(cursor + m) * 3 + 2] = forces[m + 2 * n_atoms];
                    double* row_v = vir_buf + (cursor + m) * 9;
                    row_v[0] = virials[m + 0 * n_atoms];
                    row_v[1] = virials[m + 1 * n_atoms];
                    row_v[2] = virials[m + 2 * n_atoms];
                    row_v[3] = virials[m + 3 * n_atoms];
                    row_v[4] = virials[m + 4 * n_atoms];
                    row_v[5] = virials[m + 5 * n_atoms];
                    row_v[6] = virials[m + 6 * n_atoms];
                    row_v[7] = virials[m + 7 * n_atoms];
                    row_v[8] = virials[m + 8 * n_atoms];
                    chg_buf[cursor + m] = charges[m];
                    double* row_b = bec_buf + (cursor + m) * 9;
                    row_b[0] = becs[m + 0 * n_atoms];
                    row_b[1] = becs[m + 1 * n_atoms];
                    row_b[2] = becs[m + 2 * n_atoms];
                    row_b[3] = becs[m + 3 * n_atoms];
                    row_b[4] = becs[m + 4 * n_atoms];
                    row_b[5] = becs[m + 5 * n_atoms];
                    row_b[6] = becs[m + 6 * n_atoms];
                    row_b[7] = becs[m + 7 * n_atoms];
                    row_b[8] = becs[m + 8 * n_atoms];
                }
                cursor += n_atoms;
            }
        }

        auto c1 = pybind11::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c2 = pybind11::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c3 = pybind11::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c4 = pybind11::capsule(chg_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c5 = pybind11::capsule(bec_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });

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
        pybind11::array ac(pybind11::dtype::of<double>(), shp_p,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(chg_buf), c4);
        pybind11::array ab(pybind11::dtype::of<double>(), shp_v,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(bec_buf), c5);
        return std::make_tuple(ap, af, av, ac, ab);
    }

    // Return descriptor for one structure
    std::vector<double> get_descriptor(const std::vector<int>& type,
                                       const std::vector<double>& box,
                                       const std::vector<double>& position) {
        ScopedReleaseIfHeld _gil_release;
        std::vector<double> descriptor(static_cast<size_t>(type.size() * annmb.dim));
        find_descriptor(type, box, position, descriptor);
        return descriptor;
    }

    std::vector<std::string> get_element_list() {
        return element_list;
    }

    // Return descriptors for all structures, flattened per atom.
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
            std::vector<double> struct_des(type[i].size() * static_cast<size_t>(annmb.dim));
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
};

PYBIND11_MODULE(qnep_cpu, m) {
    m.doc() = "qNEP CPU bindings with charge/BEC outputs";

    py::class_<CpuQNep>(m, "CpuQNep")
        .def(py::init<const std::string&>(), py::arg("potential_filename"))
        .def("calculate", &CpuQNep::calculate)
        .def("cancel", &CpuQNep::cancel)
        .def("reset_cancel", &CpuQNep::reset_cancel)
        .def("is_canceled", &CpuQNep::is_canceled)
        .def("get_descriptor", &CpuQNep::get_descriptor)
        .def("get_element_list", &CpuQNep::get_element_list)
        .def("get_structures_descriptor", &CpuQNep::get_structures_descriptor,
             py::arg("type"), py::arg("box"), py::arg("position"));
}
