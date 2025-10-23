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


namespace py = pybind11;




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
    // Returns: vector over frames; each inner vector is row-major [Na * dim].
    std::vector<std::vector<double>> calculate_descriptors(
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
    std::vector<std::vector<double>> get_structures_dipole(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position);

    // Structure-level polarizability (6 comps per frame) for polarizability models (train_mode==2)
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

 
        
        


    std::tuple<std::vector<std::vector<double>>, // potentials
               std::vector<std::vector<double>>, // forces
               std::vector<std::vector<double>>> // virials (9 per atom)
    calculate(const std::vector<std::vector<int>>& type,
              const std::vector<std::vector<double>>& box,
              const std::vector<std::vector<double>>& position)

    {
        // std::printf("[nep_gpu] calculate() enter\n");

        // Release the Python GIL during heavy GPU/CPU work to allow concurrency
        py::gil_scoped_release _gil_release;
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

        // Prepare outputs and process in slices to control GPU memory
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
        
        
        
        // std::printf("[nep_gpu] calculate() done\n");
        return std::make_tuple(potentials, forces, virials);
}
         
};

// pybind11 module bindings for NepTrainKit.nep_gpu
// ---- Implementation of GpuNep::calculate_descriptors ----
std::vector<std::vector<double>> GpuNep::calculate_descriptors(
        const std::vector<std::vector<int>>& type,
        const std::vector<std::vector<double>>& box,
        const std::vector<std::vector<double>>& position)
{
    py::gil_scoped_release _gil_release;
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
        py::gil_scoped_release _gil_release;
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
    py::gil_scoped_release _gil_release;
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
        CHECK(gpuDeviceSynchronize());
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
    py::gil_scoped_release _gil_release;
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
        CHECK(gpuDeviceSynchronize());
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
        .def("get_structures_polarizability", &GpuNep::get_structures_polarizability);

    m.def("_version_tag", [](){ return std::string("nep_gpu_ext_desc_1"); });
}
