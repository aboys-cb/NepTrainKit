// SPDX-License-Identifier: GPL-3.0-or-later
/*
    NepTrainKit GPU bindings for NEP (descriptor I/O and utilities)
    Copyright (C) 2025 NepTrainKit contributors

    This file adapts and interfaces with GPUMD
    (https://github.com/brucefan1983/GPUMD) by Zheyong Fan and the
    GPUMD development team, licensed under the GNU General Public License
    version 3 (or later). Portions of logic and data structures are derived
    from GPUMD source files.

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

#include <tuple>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <ctime>
#include <atomic>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif
#include <cstddef>

#include "nep_parameters.cuh"
#include "structure.cuh"
#include "dataset.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "tnep.cuh"
#include "utilities/error.cuh"
#include "nep_desc.cuh"

namespace py = pybind11;

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

static std::string convert_path(const std::string& utf8_path) {
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

static inline float get_area(const float* a, const float* b) {
    float s1 = a[1] * b[2] - a[2] * b[1];
    float s2 = a[2] * b[0] - a[0] * b[2];
    float s3 = a[0] * b[1] - a[1] * b[0];
    return std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

static inline float get_det9(const float* box) {
    return box[0] * (box[4] * box[8] - box[5] * box[7]) +
           box[1] * (box[5] * box[6] - box[3] * box[8]) +
           box[2] * (box[3] * box[7] - box[4] * box[6]);
}

static void fill_box_and_cells_from_original(const Parameters& para, Structure& s) {
    float a[3] = {s.box_original[0], s.box_original[3], s.box_original[6]};
    float b[3] = {s.box_original[1], s.box_original[4], s.box_original[7]};
    float c[3] = {s.box_original[2], s.box_original[5], s.box_original[8]};
    float det = get_det9(s.box_original);
    s.volume = std::abs(det);

    s.num_cell[0] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(b, c))));
    s.num_cell[1] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(c, a))));
    s.num_cell[2] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(a, b))));

    s.box[0] = s.box_original[0] * s.num_cell[0];
    s.box[3] = s.box_original[3] * s.num_cell[0];
    s.box[6] = s.box_original[6] * s.num_cell[0];
    s.box[1] = s.box_original[1] * s.num_cell[1];
    s.box[4] = s.box_original[4] * s.num_cell[1];
    s.box[7] = s.box_original[7] * s.num_cell[1];
    s.box[2] = s.box_original[2] * s.num_cell[2];
    s.box[5] = s.box_original[5] * s.num_cell[2];
    s.box[8] = s.box_original[8] * s.num_cell[2];

    s.box[9]  = s.box[4] * s.box[8] - s.box[5] * s.box[7];
    s.box[10] = s.box[2] * s.box[7] - s.box[1] * s.box[8];
    s.box[11] = s.box[1] * s.box[5] - s.box[2] * s.box[4];
    s.box[12] = s.box[5] * s.box[6] - s.box[3] * s.box[8];
    s.box[13] = s.box[0] * s.box[8] - s.box[2] * s.box[6];
    s.box[14] = s.box[2] * s.box[3] - s.box[0] * s.box[5];
    s.box[15] = s.box[3] * s.box[7] - s.box[4] * s.box[6];
    s.box[16] = s.box[1] * s.box[6] - s.box[0] * s.box[7];
    s.box[17] = s.box[0] * s.box[4] - s.box[1] * s.box[3];

    det *= s.num_cell[0] * s.num_cell[1] * s.num_cell[2];
    for (int n = 9; n < 18; ++n) {
        s.box[n] /= det;
    }
}

class GpuNep {
private:
    NepParameters para;
    std::vector<float> elite;
    std::unique_ptr<Potential> potential;
    std::atomic<bool> canceled_{false};

    inline void check_canceled() const {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
    }

public:
    GpuNep(const std::string& potential_filename)   {
        cudaError_t err = cudaFree(0);
        bool ok_ = (err == cudaSuccess);
        std::string error_msg_ = ok_ ? "" : cudaGetErrorString(err);
        if (!ok_) {
            throw std::runtime_error("GpuNep: " + error_msg_);
        }

        std::string path = convert_path(potential_filename);
        para.load_from_nep_txt(path, elite);
        para.prediction = 1; // prediction mode
        para.output_descriptor = 0;
        if (para.charge_mode) {
            // enable BEC prediction when using a charge-capable model
            para.has_bec = true;
        }
    }

    void cancel() { canceled_.store(true, std::memory_order_relaxed); }
    void reset_cancel() { canceled_.store(false, std::memory_order_relaxed); }
    bool is_canceled() const { return canceled_.load(std::memory_order_relaxed); }

    std::vector<std::string> get_element_list() const {
        return para.elements;
    }
    void set_batch_size(int bs) {
        if (bs < 1) return;
        para.batch_size = bs;
    }

    std::vector<std::vector<double>> calculate_descriptors(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    pybind11::array calculate_descriptors_scaled(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    std::vector<std::vector<double>> calculate_descriptors_avg(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    std::vector<std::vector<double>> get_structures_dipole(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    std::vector<std::vector<double>> get_structures_polarizability(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);


std::vector<Structure> create_structures(const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position){
        const size_t batch = type.size();
        if (box.size() != batch || position.size() != batch) {
            throw std::runtime_error("Input lists must have the same outer length.");
        }
        std::vector<Structure> structures(batch);
        for (size_t i = 0; i < batch; ++i) {
            const auto& t = type[i];
            const auto& b = box[i];
            const auto& p = position[i];
            const int Na = static_cast<int>(t.size());
            if (b.size() != 9) {
                throw std::runtime_error("Each box must have 9 components: ax,bx,cx, ay,by,cy, az,bz,cz.");
            }
            if (p.size() != static_cast<size_t>(Na) * 3) {
                throw std::runtime_error("Each position must have 3*N components arranged as x[N],y[N],z[N].");
            }

            int tmin = 1e9, tmax = -1e9;
            for (int n = 0; n < Na; ++n) { if (t[n] < tmin) tmin = t[n]; if (t[n] > tmax) tmax = t[n]; }
            if (tmin < 0 || tmax >= para.num_types) {
                throw std::runtime_error("type index out of range for this model");
            }

            Structure s;
            s.num_atom = Na;
            s.has_virial = 0;
            s.has_atomic_virial = 0;
            s.atomic_virial_diag_only = 1;
            s.has_temperature = 0;
            s.has_bec = para.has_bec ? 1 : 0;
            s.weight = 1.0f;
            s.energy_weight = 1.0f;
            for (int k = 0; k < 6; ++k) s.virial[k] = -1e6f;
            for (int k = 0; k < 9; ++k) s.box_original[k] = static_cast<float>(b[k]);
            s.bec.resize(Na * 9);

            for (int k = 0; k < Na*9; ++k) s.bec[k] = 0.0;

            s.type.resize(Na);
            s.x.resize(Na);
            s.y.resize(Na);
            s.z.resize(Na);
            s.fx.resize(Na);
            s.fy.resize(Na);
            s.fz.resize(Na);
            for (int n = 0; n < Na; ++n) {
                s.type[n] = t[n];
                s.x[n] = static_cast<float>(p[n]);
                s.y[n] = static_cast<float>(p[n + Na]);
                s.z[n] = static_cast<float>(p[n + Na * 2]);
                s.fx[n] = 0.0f;
                s.fy[n] = 0.0f;
                s.fz[n] = 0.0f;
            }

            fill_box_and_cells_from_original(para, s);
            structures[i] = std::move(s);
        }
        return structures;



        }

 
        
        


    std::tuple<pybind11::array, // potentials [total_atoms]
               pybind11::array, // forces [total_atoms,3]
               pybind11::array> // virials [total_atoms,9]
    calculate(const std::vector<std::vector<int>>& type,
              const std::vector<std::vector<double>>& box,
              const std::vector<std::vector<double>>& position)

    {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
 
 

        int devCount = 0;
        auto devErr = gpuGetDeviceCount(&devCount);
        if (devErr != gpuSuccess || devCount <= 0) {
            throw std::runtime_error("CUDA device not available");
        }
        std::vector<Structure> structures = create_structures(type, box, position);
        const int structure_num = static_cast<int>(structures.size());

        std::vector<std::vector<double>> potentials(structure_num);
        std::vector<std::vector<double>> forces(structure_num);
        std::vector<std::vector<double>> virials(structure_num);
        for (int i = 0; i < structure_num; ++i) {
            const int Na = static_cast<int>(type[i].size());
            potentials[i].resize(Na);
            forces[i].resize(Na * 3);
            virials[i].resize(Na * 9);
        }
        const int bs = para.batch_size > 0 ? para.batch_size : structure_num;

        std::vector<Dataset> dataset_vec(1);
        {
            ScopedReleaseIfHeld _gil_release;
            for (int start = 0; start < structure_num; start += bs) {
                if (canceled_.load(std::memory_order_relaxed)) {
                    throw std::runtime_error("Canceled by user");
                }
                int end = std::min(start + bs, structure_num);
                dataset_vec[0].construct(para, structures, start, end, 0 /*device id*/);
                if (para.train_mode == 1 || para.train_mode == 2) {
                    potential.reset(new TNEP(para,
                                               dataset_vec[0].N,
                                               dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                               dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                               para.version,
                                               1));
                } else {
                  if (para.charge_mode) {
                    potential.reset(new NEP_Charge(para,
                                                   dataset_vec[0].N,
                                                   dataset_vec[0].Nc,
                                                   dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                                   dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                                   para.version,
                                                   1));
                  } else {
                    potential.reset(new NEP(para,
                                            dataset_vec[0].N,
                                            dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                            dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                            para.version,
                                            1));
                  }
                }
                potential->find_force(para, elite.data(), dataset_vec, false, true, 1);
                auto err_sync = gpuDeviceSynchronize();
                if (err_sync != gpuSuccess) {
                    throw std::runtime_error(std::string("CUDA sync failed: ") + gpuGetErrorString(err_sync));
                }
                dataset_vec[0].energy.copy_to_host(dataset_vec[0].energy_cpu.data());
                dataset_vec[0].force.copy_to_host(dataset_vec[0].force_cpu.data());
                dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
                const int Nslice = dataset_vec[0].N;
                for (int gi = start; gi < end; ++gi) {
                    int li = gi - start;
                    const int Na = dataset_vec[0].Na_cpu[li];
                    const int offset = dataset_vec[0].Na_sum_cpu[li];
                    for (int m = 0; m < Na; ++m) {
                        potentials[gi][m] = static_cast<double>(dataset_vec[0].energy_cpu[offset + m]);
                        double fx = static_cast<double>(dataset_vec[0].force_cpu[offset + m]);
                        double fy = static_cast<double>(dataset_vec[0].force_cpu[offset + m + Nslice]);
                        double fz = static_cast<double>(dataset_vec[0].force_cpu[offset + m + Nslice * 2]);
                        forces[gi][m] = fx;
                        forces[gi][m + Na] = fy;
                        forces[gi][m + Na * 2] = fz;
                        double v_xx = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 0]);
                        double v_yy = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 1]);
                        double v_zz = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 2]);
                        double v_xy = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 3]);
                        double v_yz = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 4]);
                        double v_zx = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 5]);
                        virials[gi][0 * Na + m] = v_xx;
                        virials[gi][1 * Na + m] = v_xy;
                        virials[gi][2 * Na + m] = v_zx;
                        virials[gi][3 * Na + m] = v_xy;
                        virials[gi][4 * Na + m] = v_yy;
                        virials[gi][5 * Na + m] = v_yz;
                        virials[gi][6 * Na + m] = v_zx;
                        virials[gi][7 * Na + m] = v_yz;
                        virials[gi][8 * Na + m] = v_zz;
                    }
                }

            }
        }

        size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
        double* pot_buf = new double[total_atoms];
        double* frc_buf = new double[total_atoms * 3];
        double* vir_buf = new double[total_atoms * 9];
        size_t cursor = 0;
        for (size_t i = 0; i < type.size(); ++i) {
            const size_t Na = type[i].size();
            const auto& p = potentials[i];
            const auto& f = forces[i];
            const auto& v = virials[i];
            for (size_t m = 0; m < Na; ++m) {
                pot_buf[cursor + m] = p[m];
                frc_buf[(cursor + m) * 3 + 0] = f[m + 0 * Na];
                frc_buf[(cursor + m) * 3 + 1] = f[m + 1 * Na];
                frc_buf[(cursor + m) * 3 + 2] = f[m + 2 * Na];
                double* row = vir_buf + (cursor + m) * 9;
                row[0] = v[m + 0 * Na];
                row[1] = v[m + 1 * Na];
                row[2] = v[m + 2 * Na];
                row[3] = v[m + 3 * Na];
                row[4] = v[m + 4 * Na];
                row[5] = v[m + 5 * Na];
                row[6] = v[m + 6 * Na];
                row[7] = v[m + 7 * Na];
                row[8] = v[m + 8 * Na];
            }
            cursor += Na;
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

    std::tuple<pybind11::array, // potentials [total_atoms]
               pybind11::array, // forces [total_atoms,3]
               pybind11::array, // virials [total_atoms,9]
               pybind11::array, // charges [total_atoms]
               pybind11::array  // bec [total_atoms,9]
               >
    calculate_qnep(const std::vector<std::vector<int>>& type,
                   const std::vector<std::vector<double>>& box,
                   const std::vector<std::vector<double>>& position)
    {
        size_t total_atoms_input = 0;
        for (const auto& t : type) total_atoms_input += t.size();
        if (para.charge_mode == 0) {
            throw std::runtime_error("Charge model not enabled in this NEP.");
        }
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }

        int devCount = 0;
        auto devErr = gpuGetDeviceCount(&devCount);
        if (devErr != gpuSuccess || devCount <= 0) {
            throw std::runtime_error("CUDA device not available");
        }
        std::vector<Structure> structures = create_structures(type, box, position);
        const int structure_num = static_cast<int>(structures.size());
        int max_atoms_in_frame = 0;
        size_t atoms_seen = 0;
        for (const auto& s : structures) {
            atoms_seen += static_cast<size_t>(s.num_atom);
            if (s.num_atom > max_atoms_in_frame) {
                max_atoms_in_frame = s.num_atom;
            }
        }

        std::vector<std::vector<double>> potentials(structure_num);
        std::vector<std::vector<double>> forces(structure_num);
        std::vector<std::vector<double>> virials(structure_num);
        std::vector<std::vector<double>> charges(structure_num);
        std::vector<std::vector<double>> becs(structure_num);
        for (int i = 0; i < structure_num; ++i) {
            const int Na = static_cast<int>(type[i].size());
            potentials[i].resize(Na);
            forces[i].resize(Na * 3);
            virials[i].resize(Na * 9);
            charges[i].resize(Na);
            becs[i].resize(Na * 9);
        }
        const int bs = para.batch_size > 0 ? para.batch_size : structure_num;

        std::vector<Dataset> dataset_vec(1);
        {
            ScopedReleaseIfHeld _gil_release;
            for (int start = 0; start < structure_num; start += bs) {
                if (canceled_.load(std::memory_order_relaxed)) {
                    throw std::runtime_error("Canceled by user");
                }
                int end = std::min(start + bs, structure_num);
                dataset_vec[0].construct(para, structures, start, end, 0 /*device id*/);
                potential.reset(new NEP_Charge(para,
                                               dataset_vec[0].N,
                                               dataset_vec[0].Nc,
                                               dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                               dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                               para.version,
                                               1));
                potential->find_force(para, elite.data(), dataset_vec, false, true, 1);
                auto err_sync = gpuDeviceSynchronize();
                if (err_sync != gpuSuccess) {
                    throw std::runtime_error(std::string("CUDA sync failed: ") + gpuGetErrorString(err_sync));
                }
                dataset_vec[0].energy.copy_to_host(dataset_vec[0].energy_cpu.data());
                dataset_vec[0].force.copy_to_host(dataset_vec[0].force_cpu.data());
                dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
                dataset_vec[0].charge.copy_to_host(dataset_vec[0].charge_cpu.data());
                dataset_vec[0].bec.copy_to_host(dataset_vec[0].bec_cpu.data());
                const int Nslice = dataset_vec[0].N;
                for (int gi = start; gi < end; ++gi) {
                    int li = gi - start;
                    const int Na = dataset_vec[0].Na_cpu[li];
                    const int offset = dataset_vec[0].Na_sum_cpu[li];
                    for (int m = 0; m < Na; ++m) {
                        potentials[gi][m] = static_cast<double>(dataset_vec[0].energy_cpu[offset + m]);
                        double fx = static_cast<double>(dataset_vec[0].force_cpu[offset + m]);
                        double fy = static_cast<double>(dataset_vec[0].force_cpu[offset + m + Nslice]);
                        double fz = static_cast<double>(dataset_vec[0].force_cpu[offset + m + Nslice * 2]);
                        forces[gi][m] = fx;
                        forces[gi][m + Na] = fy;
                        forces[gi][m + Na * 2] = fz;
                        double v_xx = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 0]);
                        double v_yy = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 1]);
                        double v_zz = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 2]);
                        double v_xy = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 3]);
                        double v_yz = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 4]);
                        double v_zx = static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 5]);
                        virials[gi][0 * Na + m] = v_xx;
                        virials[gi][1 * Na + m] = v_xy;
                        virials[gi][2 * Na + m] = v_zx;
                        virials[gi][3 * Na + m] = v_xy;
                        virials[gi][4 * Na + m] = v_yy;
                        virials[gi][5 * Na + m] = v_yz;
                        virials[gi][6 * Na + m] = v_zx;
                        virials[gi][7 * Na + m] = v_yz;
                        virials[gi][8 * Na + m] = v_zz;
                        charges[gi][m] = static_cast<double>(dataset_vec[0].charge_cpu[offset + m]);
                        for (int d = 0; d < 9; ++d) {
                            becs[gi][d * Na + m] = static_cast<double>(dataset_vec[0].bec_cpu[offset + m + Nslice * d]);
                        }
                    }
                }
            }
        }

        size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
        double* pot_buf = new double[total_atoms];
        double* frc_buf = new double[total_atoms * 3];
        double* vir_buf = new double[total_atoms * 9];
        double* chg_buf = new double[total_atoms];
        double* bec_buf = new double[total_atoms * 9];
        size_t cursor = 0;
        for (size_t i = 0; i < type.size(); ++i) {
            const size_t Na = type[i].size();
            const auto& p = potentials[i];
            const auto& f = forces[i];
            const auto& v = virials[i];
            const auto& q = charges[i];
            const auto& b = becs[i];
            for (size_t m = 0; m < Na; ++m) {
                pot_buf[cursor + m] = p[m];
                chg_buf[cursor + m] = q[m];
                frc_buf[(cursor + m) * 3 + 0] = f[m + 0 * Na];
                frc_buf[(cursor + m) * 3 + 1] = f[m + 1 * Na];
                frc_buf[(cursor + m) * 3 + 2] = f[m + 2 * Na];
                double* row_v = vir_buf + (cursor + m) * 9;
                row_v[0] = v[m + 0 * Na];
                row_v[1] = v[m + 1 * Na];
                row_v[2] = v[m + 2 * Na];
                row_v[3] = v[m + 3 * Na];
                row_v[4] = v[m + 4 * Na];
                row_v[5] = v[m + 5 * Na];
                row_v[6] = v[m + 6 * Na];
                row_v[7] = v[m + 7 * Na];
                row_v[8] = v[m + 8 * Na];
                double* row_b = bec_buf + (cursor + m) * 9;
                row_b[0] = b[m + 0 * Na];
                row_b[1] = b[m + 1 * Na];
                row_b[2] = b[m + 2 * Na];
                row_b[3] = b[m + 3 * Na];
                row_b[4] = b[m + 4 * Na];
                row_b[5] = b[m + 5 * Na];
                row_b[6] = b[m + 6 * Na];
                row_b[7] = b[m + 7 * Na];
                row_b[8] = b[m + 8 * Na];
            }
            cursor += Na;
        }
        auto c1 = pybind11::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c2 = pybind11::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c3 = pybind11::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c4 = pybind11::capsule(chg_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        auto c5 = pybind11::capsule(bec_buf, [](void* f){ delete[] reinterpret_cast<double*>(f); });
        std::vector<std::ptrdiff_t> shp_p{static_cast<pybind11::ssize_t>(cursor)};
        std::vector<std::ptrdiff_t> shp_f{static_cast<pybind11::ssize_t>(cursor), 3};
        std::vector<std::ptrdiff_t> shp_v{static_cast<pybind11::ssize_t>(cursor), 9};
        std::vector<std::ptrdiff_t> shp_c{static_cast<pybind11::ssize_t>(cursor)};
        std::vector<std::ptrdiff_t> shp_b{static_cast<pybind11::ssize_t>(cursor), 9};
        pybind11::array ap(pybind11::dtype::of<double>(), shp_p,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(pot_buf), c1);
        pybind11::array af(pybind11::dtype::of<double>(), shp_f,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(3*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(frc_buf), c2);
        pybind11::array av(pybind11::dtype::of<double>(), shp_v,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(vir_buf), c3);
        pybind11::array aq(pybind11::dtype::of<double>(), shp_c,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(chg_buf), c4);
        pybind11::array ab(pybind11::dtype::of<double>(), shp_b,
                           std::vector<std::ptrdiff_t>{static_cast<pybind11::ssize_t>(9*sizeof(double)), static_cast<pybind11::ssize_t>(sizeof(double))},
                           static_cast<void*>(bec_buf), c5);
        return std::make_tuple(ap, af, av, aq, ab);
    }
         
};

// pybind11 module bindings for NepTrainKit.nep_gpu
// ---- Implementation of GpuNep::calculate_descriptors ----
std::vector<std::vector<double>> GpuNep::calculate_descriptors(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position)
{
    ScopedReleaseIfHeld _gil_release;
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }

    int devCount = 0; auto devErr = gpuGetDeviceCount(&devCount);
    if (devErr != gpuSuccess || devCount <= 0) {
        throw std::runtime_error("CUDA device not available");
    }
    std::vector<Structure> structures = create_structures(type, box, position);
    const int structure_num = static_cast<int>(structures.size());

    std::vector<std::vector<double>> descriptors(structure_num);
    for (int i = 0; i < structure_num; ++i) {
        const int Na = static_cast<int>(type[i].size());
        descriptors[i].resize(static_cast<size_t>(Na) * static_cast<size_t>(para.dim));
    }

    const int bs = para.batch_size > 0 ? para.batch_size : structure_num;



    std::vector<Dataset> dataset_vec(1);
    std::vector<float> desc_host;
    for (int start = 0; start < structure_num; start += bs) {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
        int end = std::min(start + bs, structure_num);
        dataset_vec[0].construct(para, structures, start, end, 0);
        NEP_Descriptors desc_engine(para,
        dataset_vec[0].N,
        dataset_vec[0].N * dataset_vec[0].max_NN_radial,
        dataset_vec[0].N * dataset_vec[0].max_NN_angular,
        para.version
         );

        desc_engine.update_parameters_from_host(elite.data());

        desc_engine.compute_descriptors(para, dataset_vec[0]);
        desc_engine.copy_descriptors_to_host(desc_host);
    const int Nslice = dataset_vec[0].N;
    const int dim = para.dim;
    int num_L = para.L_max;
    if (para.L_max_4body == 2) num_L += 1;
    if (para.L_max_5body == 1) num_L += 1;
    const int dim_desc = (para.n_max_radial + 1) + (para.n_max_angular + 1) * num_L; // filled by kernels

        #pragma omp parallel for schedule(static)
        for (int gi = start; gi < end; ++gi) {
            const int li = gi - start;
            const int Na = dataset_vec[0].Na_cpu[li];
            const int offset = dataset_vec[0].Na_sum_cpu[li];
            double* out = descriptors[gi].data();
            for (int m = 0; m < Na; ++m) {
                double* row = out + static_cast<size_t>(m) * dim;
                // fill only the descriptor dims computed by kernels
                #pragma omp simd
                for (int d = 0; d < dim_desc; ++d) {
                    row[d] = static_cast<double>(desc_host[offset + m + static_cast<size_t>(d) * Nslice]);
                }
                // zero the rest (e.g., temperature dimension in train_mode==3)
                for (int d = dim_desc; d < dim; ++d) row[d] = 0.0;
            }
        }
    }
    return descriptors;
}

// Scaled per-atom descriptors using para.q_scaler_cpu (if present)
py::array GpuNep::calculate_descriptors_scaled(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position)
{
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }

    // 1) Compute raw per-frame descriptors on GPU, returns [frames][Na*dim]
    // Note: calculate_descriptors already releases the GIL internally.
    auto raw = calculate_descriptors(type, box, position);

    // 2) Prepare a contiguous float32 buffer [total_atoms, dim]
    const int dim = para.dim;
    const bool have_scaler = static_cast<int>(para.q_scaler_cpu.size()) == dim;
    size_t total_atoms = 0;
    for (const auto& t : type) total_atoms += t.size();

    int num_L = para.L_max;
    if (para.L_max_4body == 2) num_L += 1;
    if (para.L_max_5body == 1) num_L += 1;
    const int dim_desc = (para.n_max_radial + 1) + (para.n_max_angular + 1) * num_L;

    // Allocate plain heap memory so we can hand ownership to NumPy safely
    const size_t total_elems = total_atoms * static_cast<size_t>(dim);
    float* data = nullptr;
    {
        // Release GIL only for CPU-side packing/scaling
        ScopedReleaseIfHeld _gil_release;
        try {
            data = new float[total_elems];
        } catch (const std::bad_alloc&) {
            throw std::runtime_error("Out of host memory allocating descriptor array");
        }

        // 3) Pack and scale into the contiguous buffer without touching Python APIs
        size_t cursor = 0; // current atom index across frames
        for (size_t frame_idx = 0; frame_idx < raw.size(); ++frame_idx) {
            const auto& frame = raw[frame_idx];
            const size_t Na = type[frame_idx].size();
            for (size_t atom_idx = 0; atom_idx < Na; ++atom_idx, ++cursor) {
                const double* src = frame.data() + atom_idx * static_cast<size_t>(dim);
                float* row = data + cursor * static_cast<size_t>(dim);
                // valid descriptor region
                for (int d = 0; d < dim_desc; ++d) {
                    float v = static_cast<float>(src[d]);
                    if (have_scaler) v *= static_cast<float>(para.q_scaler_cpu[d]);
                    row[d] = v;
                }
                // pad remaining dims with zero
                for (int d = dim_desc; d < dim; ++d) row[d] = 0.0f;
            }
        }
    }

    // 4) Wrap in a NumPy array, transferring ownership via capsule
    auto free_when_done = py::capsule(data, [](void* f) {
        delete[] reinterpret_cast<float*>(f);
    });

    // shape = [total_atoms, dim], strides in bytes
    std::vector<std::ptrdiff_t> shape{static_cast<std::ptrdiff_t>(total_atoms), static_cast<std::ptrdiff_t>(dim)};
    std::vector<std::ptrdiff_t> strides{static_cast<std::ptrdiff_t>(dim * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))};
    return py::array(py::buffer_info(
        data,                            // ptr
        sizeof(float),                   // itemsize
        py::format_descriptor<float>::format(), // format
        2,                               // ndim
        shape,                           // shape
        strides                          // strides
    ), free_when_done);
}


// ---- Structure dipole (3 comps) ----
std::vector<std::vector<double>> GpuNep::get_structures_dipole(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position)
{
    ScopedReleaseIfHeld _gil_release;
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }
    if (para.train_mode != 1) {
        throw std::runtime_error("Model is not a dipole NEP (train_mode!=1)");
    }

    int devCount = 0; auto devErr = gpuGetDeviceCount(&devCount);
    if (devErr != gpuSuccess || devCount <= 0) {
        throw std::runtime_error("CUDA device not available");
    }

    std::vector<Structure> structures = create_structures(type, box, position);
    const int structure_num = static_cast<int>(structures.size());
    std::vector<std::vector<double>> dipoles(structure_num, std::vector<double>(3, 0.0));
    const int bs = para.batch_size > 0 ? para.batch_size : structure_num;

    std::vector<Dataset> dataset_vec(1);
    for (int start = 0; start < structure_num; start += bs) {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
        int end = std::min(start + bs, structure_num);
        dataset_vec[0].construct(para, structures, start, end, 0);
        // For dipole/polarizability, use TNEP path
        potential.reset(new TNEP(para,
                                 dataset_vec[0].N,
                                 dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                 dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                 para.version,
                                 1));
            potential->find_force(para, elite.data(), dataset_vec, false, true, 1);
            auto err_sync = gpuDeviceSynchronize();
            if (err_sync != gpuSuccess) {
                throw std::runtime_error(std::string("CUDA sync failed: ") + gpuGetErrorString(err_sync));
            }
        dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
        const int Nslice = dataset_vec[0].N;
        for (int gi = start; gi < end; ++gi) {
            int li = gi - start;
            const int Na = dataset_vec[0].Na_cpu[li];
            const int offset = dataset_vec[0].Na_sum_cpu[li];
            double dx = 0.0, dy = 0.0, dz = 0.0;
            for (int m = 0; m < Na; ++m) {
                dx += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 0]);
                dy += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 1]);
                dz += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 2]);
            }
            dipoles[gi][0] = dx;
            dipoles[gi][1] = dy;
            dipoles[gi][2] = dz;
        }
    }
    return dipoles;
}

// ---- Structure polarizability (6 comps) ----
std::vector<std::vector<double>> GpuNep::get_structures_polarizability(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position)
{
    ScopedReleaseIfHeld _gil_release;
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }
    if (para.train_mode != 2) {
        throw std::runtime_error("Model is not a polarizability NEP (train_mode!=2)");
    }

    int devCount = 0; auto devErr = gpuGetDeviceCount(&devCount);
    if (devErr != gpuSuccess || devCount <= 0) {
        throw std::runtime_error("CUDA device not available");
    }

    std::vector<Structure> structures = create_structures(type, box, position);
    const int structure_num = static_cast<int>(structures.size());
    std::vector<std::vector<double>> pols(structure_num, std::vector<double>(6, 0.0));
    const int bs = para.batch_size > 0 ? para.batch_size : structure_num;

    std::vector<Dataset> dataset_vec(1);
    for (int start = 0; start < structure_num; start += bs) {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
        int end = std::min(start + bs, structure_num);
        dataset_vec[0].construct(para, structures, start, end, 0);
        potential.reset(new TNEP(para,
                                 dataset_vec[0].N,
                                 dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                 dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                 para.version,
                                 1));
        potential->find_force(para, elite.data(), dataset_vec, false, true, 1);
        auto err_sync = gpuDeviceSynchronize();
        if (err_sync != gpuSuccess) {
            throw std::runtime_error(std::string("CUDA sync failed: ") + gpuGetErrorString(err_sync));
        }
        dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
        const int Nslice = dataset_vec[0].N;
        for (int gi = start; gi < end; ++gi) {
            int li = gi - start;
            const int Na = dataset_vec[0].Na_cpu[li];
            const int offset = dataset_vec[0].Na_sum_cpu[li];
            double xx=0.0, yy=0.0, zz=0.0, xy=0.0, yz=0.0, zx=0.0;
            for (int m = 0; m < Na; ++m) {
                xx += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 0]);
                yy += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 1]);
                zz += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 2]);
                xy += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 3]);
                yz += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 4]);
                zx += static_cast<double>(dataset_vec[0].virial_cpu[offset + m + Nslice * 5]);
            }
            pols[gi][0] = xx;
            pols[gi][1] = yy;
            pols[gi][2] = zz;
            pols[gi][3] = xy;
            pols[gi][4] = yz;
            pols[gi][5] = zx;
        }
    }
    return pols;


}

PYBIND11_MODULE(nep_gpu, m) {
    m.doc() = "GPU-accelerated NEP bindings";
    auto cls = py::class_<GpuNep>(m, "GpuNep")
        .def(py::init<const std::string&>())

        .def("get_element_list", &GpuNep::get_element_list)
        .def("set_batch_size", &GpuNep::set_batch_size)
        .def("calculate", &GpuNep::calculate)
        .def("calculate_qnep", &GpuNep::calculate_qnep)
        .def("cancel", &GpuNep::cancel)
        .def("reset_cancel", &GpuNep::reset_cancel)
        .def("is_canceled", &GpuNep::is_canceled)
        .def("get_descriptor", &GpuNep::calculate_descriptors)

        .def("get_structures_descriptor", &GpuNep::calculate_descriptors_scaled,
             py::arg("type"), py::arg("box"), py::arg("position"))
        .def("get_structures_dipole", &GpuNep::get_structures_dipole)
        .def("get_structures_polarizability", &GpuNep::get_structures_polarizability);

    // expose charge-capable alias
    m.attr("GpuQNep") = cls;

    m.def("_version_tag", [](){ return std::string("nep_gpu_ext_desc_1"); });
}
