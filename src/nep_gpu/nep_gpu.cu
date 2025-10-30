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
#include <cstddef>  // 引入 std::ptrdiff_t
// GPUMD NEP headers (resolved via include_dirs set in setup.py)
// Relax access locally to read descriptor buffers (no IO) without touching core


#include "nep_parameters.cuh"
#include "structure.cuh"
#include "dataset.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "tnep.cuh"
#include "utilities/error.cuh"
#include "nep_desc.cuh"
// Spin-enabled NEP path
#include "nep_spin.cuh"


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
static std::string convert_path(const std::string& utf8_path) {
#ifdef _WIN32
    int wstr_size = MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    std::wstring wstr(wstr_size, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, &wstr[0], wstr_size);

    int ansi_size = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string ansi_path(ansi_size, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &ansi_path[0], ansi_size, nullptr, nullptr);
    return ansi_path;
#else
    return utf8_path;
#endif
}

// Helpers copied from structure.cu (kept here to avoid exposing non-exported statics)
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

    // number of replicated cells along each direction (same as structure.cu)
    s.num_cell[0] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(b, c))));
    s.num_cell[1] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(c, a))));
    s.num_cell[2] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(a, b))));

    // expanded box
    s.box[0] = s.box_original[0] * s.num_cell[0];
    s.box[3] = s.box_original[3] * s.num_cell[0];
    s.box[6] = s.box_original[6] * s.num_cell[0];
    s.box[1] = s.box_original[1] * s.num_cell[1];
    s.box[4] = s.box_original[4] * s.num_cell[1];
    s.box[7] = s.box_original[7] * s.num_cell[1];
    s.box[2] = s.box_original[2] * s.num_cell[2];
    s.box[5] = s.box_original[5] * s.num_cell[2];
    s.box[8] = s.box_original[8] * s.num_cell[2];

    // inverse of expanded box (cofactor divided by det)
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

                // 1. 先检测 CUDA
        cudaError_t err = cudaFree(0);
        bool ok_ = (err == cudaSuccess);
        std::string error_msg_ = ok_ ? "" : cudaGetErrorString(err);



        // 3. 如果后面有 **必须 CUDA** 的步骤，再判断
        if (!ok_) {
            // 可选：直接抛异常，让 Python 立刻知道
            throw std::runtime_error("GpuNep: " + error_msg_);
        }


        std::string path = convert_path(potential_filename);
        // std::printf("[nep_gpu] GpuNep init: potential='%s'\n", path.c_str());
        

        para.load_from_nep_txt(path, elite);
        // std::printf("[nep_gpu] loaded nep.txt: version=%d train_mode=%d charge_mode=%d num_types=%d dim=%d rc_r=%.3f rc_a=%.3f zbl=%d flex_zbl=%d\n",
                    //  para.version, para.train_mode, para.charge_mode, para.num_types, para.dim,
                    //  para.rc_radial, para.rc_angular, (int)para.enable_zbl, (int)para.flexible_zbl);
        // if (!para.elements.empty()) {
            // std::printf("[nep_gpu] elements:");
            // for (auto &e : para.elements) std::printf(" %s", e.c_str());
            // std::printf("\n");
        // }
        para.prediction = 1; // prediction mode
        // do not output descriptor files from within bindings
        para.output_descriptor = 0;
//         std::printf("[nep_gpu] init done.\n");
    }

    void cancel() { canceled_.store(true, std::memory_order_relaxed); }
    void reset_cancel() { canceled_.store(false, std::memory_order_relaxed); }
    bool is_canceled() const { return canceled_.load(std::memory_order_relaxed); }

    std::vector<std::string> get_element_list() const {
        return para.elements;
    }
    void set_batch_size(int bs) {
        if (bs < 1) {
            // std::printf("[nep_gpu] set_batch_size ignored (bs<1).\n");
            return;
        }
        para.batch_size = bs;
        // std::printf("[nep_gpu] set_batch_size = %d\n", para.batch_size);
    }

    // Compute per-atom descriptors for each frame.
    // Returns: contiguous array [total_atoms, dim] (float32)
    pybind11::array calculate_descriptors(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    // Per-atom descriptors scaled by q_scaler (matches vendor descriptor output scaling)
    // Return a contiguous NumPy array [total_atoms, dim] to avoid Python list conversion overhead.
    pybind11::array calculate_descriptors_scaled(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    // Per-atom descriptors (scaled); each inner vector has length dim
    std::vector<std::vector<double>> calculate_descriptors_avg(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    // Structure-level dipole (3 comps per frame) for dipole models (train_mode==1)
    pybind11::array get_structures_dipole(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    // Structure-level polarizability (6 comps per frame) for polarizability models (train_mode==2)
    pybind11::array get_structures_polarizability(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    // Spin-enabled compute: per-atom potential/force/virial on real atoms, plus magnetic force
    // spin component arrays are component-major per frame: [sx[N], sy[N], sz[N]]
    std::tuple<pybind11::array, // potentials [total_real_atoms]
               pybind11::array, // forces [total_real_atoms, 3]
               pybind11::array, // virials [total_real_atoms, 9]
               pybind11::array  // magnetic forces [total_real_atoms, 3]
              >
    calculate_spin(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position,
        const std::vector<std::vector<double>>& spin);

    // Spin-enabled per-atom descriptors on real atoms; returns [total_real_atoms, dim] (float32)
    pybind11::array calculate_descriptors_spin(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position,
        const std::vector<std::vector<double>>& spin);

    // Spin-enabled scaled descriptors as contiguous array [total_real_atoms, dim]
    pybind11::array calculate_descriptors_scaled_spin(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position,
        const std::vector<std::vector<double>>& spin);

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
//             std::printf("[nep_gpu]  frame %zu: Na=%d\n", i, Na);
            if (b.size() != 9) {
                throw std::runtime_error("Each box must have 9 components: ax,bx,cx, ay,by,cy, az,bz,cz.");
            }
            if (p.size() != static_cast<size_t>(Na) * 3) {
                throw std::runtime_error("Each position must have 3*N components arranged as x[N],y[N],z[N].");
            }

            // Validate type range against model's num_types
            int tmin = 1e9, tmax = -1e9;
            for (int n = 0; n < Na; ++n) { if (t[n] < tmin) tmin = t[n]; if (t[n] > tmax) tmax = t[n]; }
            if (tmin < 0 || tmax >= para.num_types) {
//                 std::printf("[nep_gpu][FATAL] type index out of range: min=%d max=%d (num_types=%d). Types must be 0..num_types-1 in nep.txt order.\n",
//                            tmin, tmax, para.num_types);
                throw std::runtime_error("type index out of range for this model");
            }

            Structure s;
            s.num_atom = Na;
//             s.has_force = 0;
//             s.has_energy = 0;
            s.has_virial = 0;
            s.has_atomic_virial = 0;
            s.atomic_virial_diag_only = 1;
            s.has_temperature = 0;
            s.has_bec=0;
            s.weight = 1.0f;
            s.energy_weight = 1.0f;
            for (int k = 0; k < 6; ++k) s.virial[k] = -1e6f;
            for (int k = 0; k < 9; ++k) s.box_original[k] = static_cast<float>(b[k]);
            s.bec.resize(Na * 9);

            for (int k = 0; k < Na*9; ++k) s.bec[k] = 0.0;

            // coordinates in split arrays
            s.type.resize(Na);
            s.x.resize(Na);
            s.y.resize(Na);
            s.z.resize(Na);
            // ensure reference force arrays exist even if has_force==0
            s.fx.resize(Na);
            s.fy.resize(Na);
            s.fz.resize(Na);
            for (int n = 0; n < Na; ++n) {
                s.type[n] = t[n];
                s.x[n] = static_cast<float>(p[n]);
                s.y[n] = static_cast<float>(p[n + Na]);
                s.z[n] = static_cast<float>(p[n + Na * 2]);
                // fill dummy force refs to avoid copy_structures reading empty vectors
                s.fx[n] = 0.0f;
                s.fy[n] = 0.0f;
                s.fz[n] = 0.0f;
            }

            // derive expanded box and inverse + num_cell
            fill_box_and_cells_from_original(para, s);
//             std::printf("[nep_gpu]   num_cell=(%d,%d,%d) volume=%.6f\n", s.num_cell[0], s.num_cell[1], s.num_cell[2], s.volume);

            structures[i] = std::move(s);
        }
        return structures;



        }

// Build structures with spin vectors attached (per frame component-major spin arrays)
std::vector<Structure> create_structures(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position,
        const std::vector<std::vector<double>>& spin)
{
    const size_t batch = type.size();
    if (box.size() != batch || position.size() != batch || spin.size() != batch) {
        throw std::runtime_error("Input lists must have the same outer length.");
    }
    std::vector<Structure> structures(batch);
    for (size_t i = 0; i < batch; ++i) {
        const auto& t = type[i];
        const auto& b = box[i];
        const auto& p = position[i];
        const auto& s = spin[i];
        const int Na = static_cast<int>(t.size());
        if (b.size() != 9) {
            throw std::runtime_error("Each box must have 9 components: ax,bx,cx, ay,by,cy, az,bz,cz.");
        }
        if (p.size() != static_cast<size_t>(Na) * 3) {
            throw std::runtime_error("Each position must have 3*N components arranged as x[N],y[N],z[N].");
        }
        if (s.size() != static_cast<size_t>(Na) * 3) {
            throw std::runtime_error("Each spin must have 3*N components arranged as sx[N],sy[N],sz[N].");
        }

        // Validate type range against model's num_types
        int tmin = 1e9, tmax = -1e9;
        for (int n = 0; n < Na; ++n) { if (t[n] < tmin) tmin = t[n]; if (t[n] > tmax) tmax = t[n]; }
        if (tmin < 0 || tmax >= para.num_types) {
            throw std::runtime_error("type index out of range for this model");
        }

        Structure st;
        st.num_atom = Na;
        st.has_virial = 0;
        st.has_atomic_virial = 0;
        st.atomic_virial_diag_only = 1;
        st.has_temperature = 0;
        st.has_bec = 0;
        st.has_spin = 1;
        st.weight = 1.0f;
        st.energy_weight = 1.0f;
        for (int k = 0; k < 6; ++k) st.virial[k] = -1e6f;
        for (int k = 0; k < 9; ++k) st.box_original[k] = static_cast<float>(b[k]);

        // split component storage
        st.type.resize(Na);
        st.x.resize(Na); st.y.resize(Na); st.z.resize(Na);
        st.fx.resize(Na); st.fy.resize(Na); st.fz.resize(Na);
        st.bec.resize(Na * 9);
        st.spinx.resize(Na); st.spiny.resize(Na); st.spinz.resize(Na);

        for (int n = 0; n < Na; ++n) {
            st.type[n] = t[n];
            st.x[n] = static_cast<float>(p[n]);
            st.y[n] = static_cast<float>(p[n + Na]);
            st.z[n] = static_cast<float>(p[n + 2 * Na]);
            st.fx[n] = 0.0f; st.fy[n] = 0.0f; st.fz[n] = 0.0f;
            st.spinx[n] = static_cast<float>(s[n]);
            st.spiny[n] = static_cast<float>(s[n + Na]);
            st.spinz[n] = static_cast<float>(s[n + 2 * Na]);
        }

        // derive expanded box and inverse + num_cell
        fill_box_and_cells_from_original(para, st);
        structures[i] = std::move(st);
    }
    return structures;
}

// ---- Spin compute: energy/force/virial + magnetic force ----
// moved out-of-class definitions to avoid in-class redeclaration

 
        
        


    std::tuple<pybind11::array, // potentials [total_atoms]
               pybind11::array, // forces [total_atoms,3]
               pybind11::array  // virials [total_atoms,9]
              >
    calculate(const std::vector<std::vector<int>>& type,
              const std::vector<std::vector<double>>& box,
              const std::vector<std::vector<double>>& position)

    {
        // std::printf("[nep_gpu] calculate() enter\n");

        // Release the Python GIL during heavy GPU/CPU work to allow concurrency
        ScopedReleaseIfHeld _gil_release;
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
 
 

        // Early device check to avoid crashing inside construct on some systems
        int devCount = 0;
        auto devErr = gpuGetDeviceCount(&devCount);
        if (devErr != gpuSuccess || devCount <= 0) {
            // std::printf("[nep_gpu][FATAL] No CUDA device available or runtime error. devCount=%d\n", devCount);
            throw std::runtime_error("CUDA device not available");
        }
        // std::printf("[nep_gpu] CUDA device count = %d\n", devCount);
        // build structures for all inputs
        std::vector<Structure> structures = create_structures(type, box, position);
        const int structure_num = static_cast<int>(structures.size());

        // Prepare outputs as contiguous buffers [total_atoms], [total_atoms,3], [total_atoms,9]
        size_t total_atoms = 0; for (const auto& t : type) total_atoms += t.size();
        float* pot_buf = new float[total_atoms];
        float* frc_buf = new float[total_atoms * 3];
        float* vir_buf = new float[total_atoms * 9];
        size_t cursor = 0; // atom index across frames
        const int bs = para.batch_size > 0 ? para.batch_size : structure_num;
        const int Nc_max = std::min(bs, structure_num);




        // Pass 2: run each slice using the reusable Potential

        std::vector<Dataset> dataset_vec(1);
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
            CHECK(gpuDeviceSynchronize());
            dataset_vec[0].energy.copy_to_host(dataset_vec[0].energy_cpu.data());
            dataset_vec[0].force.copy_to_host(dataset_vec[0].force_cpu.data());
            dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
            const int Nslice = dataset_vec[0].N;
            for (int gi = start; gi < end; ++gi) {
                int li = gi - start;
                const int Na = dataset_vec[0].Na_cpu[li];
                const int offset = dataset_vec[0].Na_sum_cpu[li];
                for (int m = 0; m < Na; ++m) {
                    pot_buf[cursor + m] = dataset_vec[0].energy_cpu[offset + m];
                    float fx = dataset_vec[0].force_cpu[offset + m];
                    float fy = dataset_vec[0].force_cpu[offset + m + Nslice];
                    float fz = dataset_vec[0].force_cpu[offset + m + Nslice * 2];
                    frc_buf[(cursor + m) * 3 + 0] = fx;
                    frc_buf[(cursor + m) * 3 + 1] = fy;
                    frc_buf[(cursor + m) * 3 + 2] = fz;
                    float v_xx = dataset_vec[0].virial_cpu[offset + m + Nslice * 0];
                    float v_yy = dataset_vec[0].virial_cpu[offset + m + Nslice * 1];
                    float v_zz = dataset_vec[0].virial_cpu[offset + m + Nslice * 2];
                    float v_xy = dataset_vec[0].virial_cpu[offset + m + Nslice * 3];
                    float v_yz = dataset_vec[0].virial_cpu[offset + m + Nslice * 4];
                    float v_zx = dataset_vec[0].virial_cpu[offset + m + Nslice * 5];
                    float* row = vir_buf + (cursor + m) * 9;
                    row[0] = v_xx; row[1] = v_xy; row[2] = v_zx;
                    row[3] = v_xy; row[4] = v_yy; row[5] = v_yz;
                    row[6] = v_zx; row[7] = v_yz; row[8] = v_zz;
                }
                cursor += static_cast<size_t>(Na);
            }

        }
        
        
        
        // Wrap arrays for Python
        auto cap1 = py::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
        auto cap2 = py::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
        auto cap3 = py::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
        std::vector<std::ptrdiff_t> shape_p{static_cast<py::ssize_t>(cursor)};
        std::vector<std::ptrdiff_t> shape_f{static_cast<py::ssize_t>(cursor), 3};
        std::vector<std::ptrdiff_t> shape_v{static_cast<py::ssize_t>(cursor), 9};
        py::array ap(py::dtype::of<float>(), shape_p, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(pot_buf), cap1);
        py::array af(py::dtype::of<float>(), shape_f, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(3*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(frc_buf), cap2);
        py::array av(py::dtype::of<float>(), shape_v, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(9*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(vir_buf), cap3);
        return std::make_tuple(ap, af, av);
}
         
};

// pybind11 module bindings for NepTrainKit.nep_gpu
// ---- Implementation of GpuNep::calculate_descriptors (py::array) ----
py::array GpuNep::calculate_descriptors(
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

    size_t total_atoms = 0;
    for (const auto& t : type) total_atoms += t.size();
    const int dim = para.dim;

    // host buffer
    float* data = nullptr;
    try {
        data = new float[total_atoms * static_cast<size_t>(dim)];
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("Out of host memory allocating descriptor array");
    }

    const int bs = para.batch_size > 0 ? para.batch_size : structure_num;
    std::vector<Dataset> dataset_vec(1);
    std::vector<float> desc_host;
    size_t base_cursor = 0; // cumulative atom index across frames
    for (int start = 0; start < structure_num; start += bs) {
        if (canceled_.load(std::memory_order_relaxed)) {
            delete[] data;
            throw std::runtime_error("Canceled by user");
        }
        int end = std::min(start + bs, structure_num);
        dataset_vec[0].construct(para, structures, start, end, 0);
        NEP_Descriptors desc_engine(para,
            dataset_vec[0].N,
            dataset_vec[0].N * dataset_vec[0].max_NN_radial,
            dataset_vec[0].N * dataset_vec[0].max_NN_angular,
            para.version);

        desc_engine.update_parameters_from_host(elite.data());
        desc_engine.compute_descriptors(para, dataset_vec[0]);
        desc_engine.copy_descriptors_to_host(desc_host);

        const int Nslice = dataset_vec[0].N;
        int num_L = para.L_max;
        if (para.L_max_4body == 2) num_L += 1;
        if (para.L_max_5body == 1) num_L += 1;
        const int dim_desc = (para.n_max_radial + 1) + (para.n_max_angular + 1) * num_L; // filled by kernels

        for (int gi = start; gi < end; ++gi) {
            const int li = gi - start;
            const int Na = dataset_vec[0].Na_cpu[li];
            const int offset = dataset_vec[0].Na_sum_cpu[li];
            for (int m = 0; m < Na; ++m) {
                float* row = data + (base_cursor + static_cast<size_t>(m)) * dim;
                // valid dims
                for (int d = 0; d < dim_desc; ++d) {
                    row[d] = desc_host[offset + m + static_cast<size_t>(d) * Nslice];
                }
                // zero pad
                for (int d = dim_desc; d < dim; ++d) row[d] = 0.0f;
            }
            base_cursor += static_cast<size_t>(Na);
        }
    }

    // Wrap into NumPy array
    auto capsule = py::capsule(data, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shape{static_cast<std::ptrdiff_t>(total_atoms), static_cast<std::ptrdiff_t>(dim)};
    std::vector<std::ptrdiff_t> strides{static_cast<std::ptrdiff_t>(dim * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))};
    return py::array(py::dtype::of<float>(), shape, strides, static_cast<void*>(data), capsule);
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

    // 1) Compute raw descriptors as NumPy array [total_atoms, dim] (float32)
    // Note: calculate_descriptors already releases the GIL internally
    py::array raw = calculate_descriptors(type, box, position);

    // 2) Prepare output array with same shape/dtype
    auto buf = raw.request();
    if (buf.ndim != 2) throw std::runtime_error("Unexpected descriptor array ndim");
    const int dim = para.dim;
    const bool have_scaler = static_cast<int>(para.q_scaler_cpu.size()) == dim;
    int num_L = para.L_max;
    if (para.L_max_4body == 2) num_L += 1;
    if (para.L_max_5body == 1) num_L += 1;
    const int dim_desc = (para.n_max_radial + 1) + (para.n_max_angular + 1) * num_L;

    // Allocate new buffer and scale
    const size_t total_elems = static_cast<size_t>(buf.shape[0]) * static_cast<size_t>(buf.shape[1]);
    float* out = nullptr;
    try { out = new float[total_elems]; } catch (...) { throw std::runtime_error("Out of host memory allocating descriptor array"); }
    {
        ScopedReleaseIfHeld _gil_release;
        const float* src = static_cast<const float*>(buf.ptr);
        const size_t dim_sz = static_cast<size_t>(dim);
        for (size_t i = 0; i < static_cast<size_t>(buf.shape[0]); ++i) {
            float* row_o = out + i * dim_sz;
            const float* row_i = src + i * dim_sz;
            for (int d = 0; d < dim_desc; ++d) {
                float v = row_i[d];
                if (have_scaler) v *= static_cast<float>(para.q_scaler_cpu[d]);
                row_o[d] = v;
            }
            for (int d = dim_desc; d < dim; ++d) row_o[d] = 0.0f;
        }
    }
    auto capsule = py::capsule(out, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shape{buf.shape[0], buf.shape[1]};
    std::vector<std::ptrdiff_t> strides{static_cast<py::ssize_t>(buf.shape[1] * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))};
    return py::array(py::dtype::of<float>(), shape, strides, static_cast<void*>(out), capsule);
}


// ---- Structure dipole (3 comps) ----
py::array GpuNep::get_structures_dipole(
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
    // contiguous [structure_num, 3]
    float* buf = new float[static_cast<size_t>(structure_num) * 3];
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
        CHECK(gpuDeviceSynchronize());
        dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
        const int Nslice = dataset_vec[0].N;
        for (int gi = start; gi < end; ++gi) {
            int li = gi - start;
            const int Na = dataset_vec[0].Na_cpu[li];
            const int offset = dataset_vec[0].Na_sum_cpu[li];
            float dx = 0.0f, dy = 0.0f, dz = 0.0f;
            for (int m = 0; m < Na; ++m) {
                dx += dataset_vec[0].virial_cpu[offset + m + Nslice * 0];
                dy += dataset_vec[0].virial_cpu[offset + m + Nslice * 1];
                dz += dataset_vec[0].virial_cpu[offset + m + Nslice * 2];
            }
            float* row = buf + static_cast<size_t>(gi) * 3;
            row[0] = dx; row[1] = dy; row[2] = dz;
        }
    }
    auto cap = py::capsule(buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shape{structure_num, 3};
    std::vector<std::ptrdiff_t> strides{static_cast<py::ssize_t>(3*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))};
    return py::array(py::dtype::of<float>(), shape, strides, static_cast<void*>(buf), cap);
}

// ---- Structure polarizability (6 comps) ----
py::array GpuNep::get_structures_polarizability(
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
    float* buf = new float[static_cast<size_t>(structure_num) * 6];
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
        CHECK(gpuDeviceSynchronize());
        dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
        const int Nslice = dataset_vec[0].N;
        for (int gi = start; gi < end; ++gi) {
            int li = gi - start;
            const int Na = dataset_vec[0].Na_cpu[li];
            const int offset = dataset_vec[0].Na_sum_cpu[li];
            float xx=0.0f, yy=0.0f, zz=0.0f, xy=0.0f, yz=0.0f, zx=0.0f;
            for (int m = 0; m < Na; ++m) {
                xx += dataset_vec[0].virial_cpu[offset + m + Nslice * 0];
                yy += dataset_vec[0].virial_cpu[offset + m + Nslice * 1];
                zz += dataset_vec[0].virial_cpu[offset + m + Nslice * 2];
                xy += dataset_vec[0].virial_cpu[offset + m + Nslice * 3];
                yz += dataset_vec[0].virial_cpu[offset + m + Nslice * 4];
                zx += dataset_vec[0].virial_cpu[offset + m + Nslice * 5];
            }
            float* row = buf + static_cast<size_t>(gi) * 6;
            row[0] = xx; row[1] = yy; row[2] = zz;
            row[3] = xy; row[4] = yz; row[5] = zx;
        }
    }
    auto cap2 = py::capsule(buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shape2{structure_num, 6};
    std::vector<std::ptrdiff_t> strides2{static_cast<py::ssize_t>(6*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))};
    return py::array(py::dtype::of<float>(), shape2, strides2, static_cast<void*>(buf), cap2);


}

// ---- Spin compute (out-of-class definitions) ----
std::tuple<py::array, // potentials [total_real_atoms]
           py::array, // forces [total_real_atoms, 3]
           py::array, // virials [total_real_atoms, 9]
           py::array  // magnetic forces [total_real_atoms, 3]
          >
GpuNep::calculate_spin(
    const std::vector<std::vector<int>>& type,
    const std::vector<std::vector<double>>& box,
    const std::vector<std::vector<double>>& position,
    const std::vector<std::vector<double>>& spin)
{
    ScopedReleaseIfHeld _gil_release;
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }
    if (para.spin_mode != 1) {
        throw std::runtime_error("Model is not a spin NEP (header should contain 'spin')");
    }
    int devCount = 0; auto devErr = gpuGetDeviceCount(&devCount);
    if (devErr != gpuSuccess || devCount <= 0) {
        throw std::runtime_error("CUDA device not available");
    }

    std::vector<Structure> structures = create_structures(type, box, position, spin);
    const int structure_num = static_cast<int>(structures.size());
    const int bs = para.batch_size > 0 ? para.batch_size : structure_num;
    // total real atoms across frames
    size_t total_real_atoms = 0;
    {
        // Rough pass: sum lengths from input; actual real atoms equal to per-frame Na_real
        for (const auto& t : type) total_real_atoms += t.size();
    }
    float* pot_buf = new float[total_real_atoms];
    float* frc_buf = new float[total_real_atoms * 3];
    float* vir_buf = new float[total_real_atoms * 9];
    float* mf_buf  = new float[total_real_atoms * 3];
    size_t cursor = 0;

    std::vector<Dataset> dataset_vec(1);
    for (int start = 0; start < structure_num; start += bs) {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
        int end = std::min(start + bs, structure_num);
        dataset_vec[0].construct(para, structures, start, end, 0);

        // Use spin pipeline
        NEP_Spin spin_engine(
            para,
            dataset_vec[0].N,
            dataset_vec[0].N * dataset_vec[0].max_NN_radial,
            dataset_vec[0].N * dataset_vec[0].max_NN_angular,
            para.version,
            1);
        spin_engine.find_force(para, elite.data(), dataset_vec, false, true, 1);
        CHECK(gpuDeviceSynchronize());
        // copy results to host
        dataset_vec[0].energy.copy_to_host(dataset_vec[0].energy_cpu.data());
        dataset_vec[0].force.copy_to_host(dataset_vec[0].force_cpu.data());
        dataset_vec[0].virial.copy_to_host(dataset_vec[0].virial_cpu.data());
        std::vector<float> fm_host(dataset_vec[0].N_real * 3, 0.0f);
        dataset_vec[0].fm_pred.copy_to_host(fm_host.data());

        const int Nslice = dataset_vec[0].N;
        const int Nreal_slice = dataset_vec[0].N_real;
        for (int gi = start; gi < end; ++gi) {
            int li = gi - start;
            const int Na_total = dataset_vec[0].Na_cpu[li];
            const int Na_real  = dataset_vec[0].Na_real_cpu[li];
            const int off_total = dataset_vec[0].Na_sum_cpu[li];
            const int off_real  = dataset_vec[0].Na_real_sum_cpu[li];
            for (int m = 0; m < Na_real; ++m) {
                int idx = off_total + m; // real atoms first in each config
                // potential
                pot_buf[cursor + m] = dataset_vec[0].energy_cpu[idx];
                // force
                frc_buf[(cursor + m) * 3 + 0] = dataset_vec[0].force_cpu[idx];
                frc_buf[(cursor + m) * 3 + 1] = dataset_vec[0].force_cpu[idx + Nslice];
                frc_buf[(cursor + m) * 3 + 2] = dataset_vec[0].force_cpu[idx + 2 * Nslice];
                // virial: expand 6 -> 9 in row-major [xx,xy,xz,yx,yy,yz,zx,zy,zz]
                float vxx = dataset_vec[0].virial_cpu[idx + 0 * Nslice];
                float vyy = dataset_vec[0].virial_cpu[idx + 1 * Nslice];
                float vzz = dataset_vec[0].virial_cpu[idx + 2 * Nslice];
                float vxy = dataset_vec[0].virial_cpu[idx + 3 * Nslice];
                float vyz = dataset_vec[0].virial_cpu[idx + 4 * Nslice];
                float vzx = dataset_vec[0].virial_cpu[idx + 5 * Nslice];
                float* row = vir_buf + (cursor + m) * 9;
                row[0] = vxx; row[1] = vxy; row[2] = vzx;
                row[3] = vxy; row[4] = vyy; row[5] = vyz;
                row[6] = vzx; row[7] = vyz; row[8] = vzz;
                // magnetic force from slice-contiguous fm_host (component-major)
                mf_buf[(cursor + m) * 3 + 0] = fm_host[off_real + m];
                mf_buf[(cursor + m) * 3 + 1] = fm_host[off_real + m + Nreal_slice];
                mf_buf[(cursor + m) * 3 + 2] = fm_host[off_real + m + 2 * Nreal_slice];
            }
            cursor += static_cast<size_t>(Na_real);
        }
    }
    auto c1 = py::capsule(pot_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    auto c2 = py::capsule(frc_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    auto c3 = py::capsule(vir_buf, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    auto c4 = py::capsule(mf_buf,  [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shp_p{static_cast<py::ssize_t>(cursor)};
    std::vector<std::ptrdiff_t> shp_f{static_cast<py::ssize_t>(cursor), 3};
    std::vector<std::ptrdiff_t> shp_v{static_cast<py::ssize_t>(cursor), 9};
    py::array ap(py::dtype::of<float>(), shp_p, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(pot_buf), c1);
    py::array af(py::dtype::of<float>(), shp_f, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(3*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(frc_buf), c2);
    py::array av(py::dtype::of<float>(), shp_v, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(9*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(vir_buf), c3);
    py::array am(py::dtype::of<float>(), shp_f, std::vector<std::ptrdiff_t>{static_cast<py::ssize_t>(3*sizeof(float)), static_cast<py::ssize_t>(sizeof(float))}, static_cast<void*>(mf_buf),  c4);
    return {ap, af, av, am};
}

// ---- Spin descriptors (per real atom) ----
py::array GpuNep::calculate_descriptors_spin(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position,
        const std::vector<std::vector<double>>& spin)
{
    ScopedReleaseIfHeld _gil_release;
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }
    if (para.spin_mode != 1) {
        throw std::runtime_error("Model is not a spin NEP (header should contain 'spin')");
    }
    int devCount = 0; auto devErr = gpuGetDeviceCount(&devCount);
    if (devErr != gpuSuccess || devCount <= 0) {
        throw std::runtime_error("CUDA device not available");
    }
    std::vector<Structure> structures = create_structures(type, box, position, spin);
    const int structure_num = static_cast<int>(structures.size());
    // Pre-allocate contiguous buffer using upper bound (sum of Na across frames)
    size_t total_upper = 0; for (const auto& t : type) total_upper += t.size();
    const int dim = para.dim;
    float* data = new float[total_upper * static_cast<size_t>(dim)];
    size_t cursor = 0;
    const int bs = para.batch_size > 0 ? para.batch_size : structure_num;

    std::vector<Dataset> dataset_vec(1);
    for (int start = 0; start < structure_num; start += bs) {
        if (canceled_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Canceled by user");
        }
        int end = std::min(start + bs, structure_num);
        dataset_vec[0].construct(para, structures, start, end, 0);
        // run spin NEP to fill descriptors (heavy but consistent)
        NEP_Spin spin_engine(
            para,
            dataset_vec[0].N,
            dataset_vec[0].N * dataset_vec[0].max_NN_radial,
            dataset_vec[0].N * dataset_vec[0].max_NN_angular,
            para.version,
            1);
        spin_engine.find_force(para, elite.data(), dataset_vec, false, true, 1);
        CHECK(gpuDeviceSynchronize());
        // copy descriptors (N * dim)
        auto& dvec = spin_engine.get_descriptors(0);
        std::vector<float> desc_host;
        desc_host.resize(dvec.size());
        dvec.copy_to_host(desc_host.data());

        const int Nslice = dataset_vec[0].N;
        const int Nreal_slice = dataset_vec[0].N_real;
        int num_L = para.L_max; if (para.L_max_4body == 2) num_L += 1; if (para.L_max_5body == 1) num_L += 1;
        const int dim_desc = (para.n_max_radial + 1) + (para.n_max_angular + 1) * num_L;

        for (int gi = start; gi < end; ++gi) {
            const int li = gi - start;
            const int Na_real = dataset_vec[0].Na_real_cpu[li];
            const int off_total = dataset_vec[0].Na_sum_cpu[li];
            for (int m = 0; m < Na_real; ++m) {
                float* row = data + (cursor + m) * static_cast<size_t>(dim);
                for (int d = 0; d < dim_desc; ++d) {
                    row[d] = desc_host[off_total + m + static_cast<size_t>(d) * Nslice];
                }
                for (int d = dim_desc; d < dim; ++d) row[d] = 0.0f;
            }
            cursor += static_cast<size_t>(Na_real);
        }
    }
    auto cap = py::capsule(data, [](void* f){ delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shape{static_cast<py::ssize_t>(cursor), static_cast<py::ssize_t>(dim)};
    std::vector<std::ptrdiff_t> strides{static_cast<py::ssize_t>(dim * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))};
    return py::array(py::dtype::of<float>(), shape, strides, static_cast<void*>(data), cap);
}

pybind11::array GpuNep::calculate_descriptors_scaled_spin(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position,
        const std::vector<std::vector<double>>& spin)
{
    if (canceled_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Canceled by user");
    }
    // compute raw as contiguous array
    py::array raw = calculate_descriptors_spin(type, box, position, spin);
    auto buf = raw.request();
    const int dim = para.dim;
    const bool have_scaler = static_cast<int>(para.q_scaler_cpu.size()) == dim;
    size_t total_real_atoms = static_cast<size_t>(buf.shape[0]);
    int num_L = para.L_max; if (para.L_max_4body == 2) num_L += 1; if (para.L_max_5body == 1) num_L += 1;
    const int dim_desc = (para.n_max_radial + 1) + (para.n_max_angular + 1) * num_L;

    float* data = nullptr;
    {
        ScopedReleaseIfHeld _gil_release;
        data = new float[total_real_atoms * static_cast<size_t>(dim)];
        const float* src = static_cast<const float*>(buf.ptr);
        for (size_t i = 0; i < total_real_atoms; ++i) {
            float* row = data + i * static_cast<size_t>(dim);
            const float* rsrc = src + i * static_cast<size_t>(dim);
            for (int d = 0; d < dim_desc; ++d) {
                float v = rsrc[d];
                if (have_scaler) v *= static_cast<float>(para.q_scaler_cpu[d]);
                row[d] = v;
            }
            for (int d = dim_desc; d < dim; ++d) row[d] = 0.0f;
        }
    }

    auto free_when_done = py::capsule(data, [](void* f) { delete[] reinterpret_cast<float*>(f); });
    std::vector<std::ptrdiff_t> shape{static_cast<std::ptrdiff_t>(total_real_atoms), static_cast<std::ptrdiff_t>(dim)};
    std::vector<std::ptrdiff_t> strides{static_cast<std::ptrdiff_t>(dim * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))};
    return py::array(py::dtype::of<float>(), shape, strides, static_cast<void*>(data), free_when_done);
}

PYBIND11_MODULE(nep_gpu, m) {
    m.doc() = "GPU-accelerated NEP bindings";
    py::class_<GpuNep>(m, "GpuNep")
        .def(py::init<const std::string&>())

        .def("get_element_list", &GpuNep::get_element_list)
        .def("set_batch_size", &GpuNep::set_batch_size)
        .def("calculate", &GpuNep::calculate)
        .def("cancel", &GpuNep::cancel)
        .def("reset_cancel", &GpuNep::reset_cancel)
        .def("is_canceled", &GpuNep::is_canceled)
        .def("get_descriptor", &GpuNep::calculate_descriptors)

        .def("get_structures_descriptor", &GpuNep::calculate_descriptors_scaled,
             py::arg("type"), py::arg("box"), py::arg("position"))
        .def("get_structures_dipole", &GpuNep::get_structures_dipole)
        .def("get_structures_polarizability", &GpuNep::get_structures_polarizability)
        // Spin-enabled overload of calculate (same API name as CPU)
        .def("calculate", &GpuNep::calculate_spin,
             py::arg("type"), py::arg("box"), py::arg("position"), py::arg("spin"))
        // Spin-enabled overload for get_structures_descriptor (same API name as CPU)
        .def("get_structures_descriptor", &GpuNep::calculate_descriptors_scaled_spin,
             py::arg("type"), py::arg("box"), py::arg("position"), py::arg("spin"));

    m.def("_version_tag", [](){ return std::string("nep_gpu_ext_desc_1"); });
}
