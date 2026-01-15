// SPDX-License-Identifier: GPL-3.0-or-later
/*
    Minimal descriptor interfaces built from GPUMD NEP kernels
    Copyright (C) 2025 NepTrainKit contributors

    This file declares interfaces for kernels derived from GPUMD
    (https://github.com/brucefan1983/GPUMD) by Zheyong Fan and the
    GPUMD development team, which is licensed under the GNU General
    Public License version 3 (or later).

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

// Minimal descriptor computer built from NEP kernels without touching vendor files.
#pragma once

#include "parameters.cuh"
#include "dataset.cuh"
#include "nep.cuh" // for NEP::ParaMB, NEP::ANN and NEP_Data
#include "nep_spin.cuh" // for NEP_Spin::ParaMB/ANN and spin descriptor dispatch
#include "utilities/gpu_macro.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_spin_utilities.cuh"
#include "utilities/common.cuh"
#include "mic.cuh"

// Kernels (implemented in nep_desc.cu)
__global__ void gpu_find_neighbor_list_desc(
  const NEP::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const float g_rc_radial,
  const float g_rc_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular);

__global__ void find_descriptors_radial_desc(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors);

__global__ void find_descriptors_angular_desc(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz);

class NEP_Descriptors {
public:
  NEP_Descriptors(Parameters& para,
                  int N,
                  int N_times_max_NN_radial,
                  int N_times_max_NN_angular,
                  int version)
  {
    // Setup paramb similar to NEP::NEP
    paramb.version = version;
    paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
    paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
    paramb.num_types = para.num_types;
    paramb.n_max_radial = para.n_max_radial;
    paramb.n_max_angular = para.n_max_angular;
    paramb.L_max = para.L_max;
    paramb.num_L = paramb.L_max;
    if (para.L_max_4body == 2) paramb.num_L += 1;
    if (para.L_max_5body == 1) paramb.num_L += 1;
    paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;

    paramb.basis_size_radial = para.basis_size_radial;
    paramb.basis_size_angular = para.basis_size_angular;
    paramb.num_types_sq = para.num_types * para.num_types;
    paramb.num_c_radial = paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

    // Parameters keeps rc_* sized to NUM_ELEMENTS; only the first num_types
    // entries are meaningful.
    for (int n = 0; n < NUM_ELEMENTS; ++n) {
      paramb.rc_radial[n] = para.rc_radial[n];
      paramb.rc_angular[n] = para.rc_angular[n];
    }

    ann.dim = para.dim;
    ann.num_neurons1 = para.num_neurons1;
    ann.num_para = para.number_of_variables;

    // Precompute spin descriptor layout for *_spin models.
    spin_enabled = (para.spin_mode > 0);
    if (spin_enabled) {
      // Base descriptor dim computed by the radial+angular kernels.
      spin_offset = (para.n_max_radial + 1) + (para.n_max_angular + 1) * paramb.num_L;

      // Spin descriptor dim matches Parameters::calculate_parameters().
      const int nspin = (para.n_max_radial + 1);
      const int kmax_ex = nep_spin_clamp_kmax(para.spin_kmax_ex);
      const int kmax_dmi = nep_spin_clamp_kmax(para.spin_kmax_dmi);
      const int kmax_ani = nep_spin_clamp_kmax(para.spin_kmax_ani);
      const int kmax_sia = nep_spin_clamp_kmax(para.spin_kmax_sia);
      const int ex_blocks = nep_spin_blocks_from_kmax(kmax_ex);
      const int dmi_blocks = nep_spin_blocks_from_kmax(kmax_dmi);
      const int ani_blocks = nep_spin_blocks_from_kmax(kmax_ani);
      const int sia_blocks = nep_spin_blocks_from_kmax(kmax_sia);
      const int pair_blocks = ex_blocks + dmi_blocks + ani_blocks + sia_blocks;
      spin_dim = nspin * pair_blocks + nep_spin_clamp_pmax(para.spin_pmax);

      if (spin_offset + spin_dim > para.dim) {
        PRINT_INPUT_ERROR("Spin descriptor layout exceeds para.dim (nep.txt may be inconsistent).");
      }

      // Fill the NEP_Spin::ParaMB fields needed by the spin descriptor kernels.
      spin_paramb.version = version;
      spin_paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
      spin_paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
      spin_paramb.num_types = para.num_types;
      for (int t = 0; t < spin_paramb.num_types; ++t) {
        spin_paramb.rc_radial[t] = para.rc_radial[t];
        spin_paramb.rc_angular[t] = para.rc_angular[t];
      }
      spin_paramb.n_max_radial = para.n_max_radial;
      spin_paramb.n_max_angular = para.n_max_angular;
      spin_paramb.L_max = para.L_max;
      spin_paramb.spin_kmax_ex = para.spin_kmax_ex;
      spin_paramb.spin_kmax_dmi = para.spin_kmax_dmi;
      spin_paramb.spin_kmax_ani = para.spin_kmax_ani;
      spin_paramb.spin_kmax_sia = para.spin_kmax_sia;
      spin_paramb.spin_pmax = para.spin_pmax;
      spin_paramb.spin_ex_phi_mode = para.spin_ex_phi_mode;
      spin_paramb.spin_onsite_basis_mode = para.spin_onsite_basis_mode;
      spin_paramb.spin_mref = para.spin_mref;
      spin_paramb.num_L = paramb.num_L;
      spin_paramb.dim_angular = paramb.dim_angular;
      spin_paramb.basis_size_radial = para.basis_size_radial;
      spin_paramb.basis_size_angular = para.basis_size_angular;
      spin_paramb.num_types_sq = para.num_types * para.num_types;
      spin_paramb.num_c_radial = spin_paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

      // c_spin layout (matches NEP_Spin ctor).
      spin_paramb.spin_blocks = ex_blocks + dmi_blocks + ani_blocks + sia_blocks;
      spin_paramb.c_spin_block_stride =
        spin_paramb.num_types_sq * nspin * (para.basis_size_radial + 1);
      const int num_c_angular =
        spin_paramb.num_types_sq * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
      if (para.spin_mode == 2) {
        if (spin_paramb.spin_blocks > 0) {
          spin_paramb.num_c_spin = spin_paramb.c_spin_block_stride;
          spin_paramb.c_spin_offset = spin_paramb.num_c_radial + num_c_angular;
        }
      } else if (para.spin_mode == 3) {
        if (spin_paramb.spin_blocks > 0) {
          spin_paramb.num_c_spin = spin_paramb.c_spin_block_stride * spin_paramb.spin_blocks;
          spin_paramb.c_spin_offset = spin_paramb.num_c_radial + num_c_angular;
        }
      } else {
        // spin_mode == 1: reuse lattice c coefficients (num_c_spin = 0).
        spin_paramb.num_c_spin = 0;
        spin_paramb.c_spin_offset = 0;
      }
      for (int n = 0; n < static_cast<int>(para.atomic_numbers.size()); ++n) {
        spin_paramb.atomic_numbers[n] = para.atomic_numbers[n] - 1;
      }
    }

    // Allocate buffers on device 0 only (single-GPU path)
    data.NN_radial.resize(N);
    data.NN_angular.resize(N);
    data.NL_radial.resize(N_times_max_NN_radial);
    data.NL_angular.resize(N_times_max_NN_angular);
    data.x12_radial.resize(N_times_max_NN_radial);
    data.y12_radial.resize(N_times_max_NN_radial);
    data.z12_radial.resize(N_times_max_NN_radial);
    data.x12_angular.resize(N_times_max_NN_angular);
    data.y12_angular.resize(N_times_max_NN_angular);
    data.z12_angular.resize(N_times_max_NN_angular);
    data.descriptors.resize(N * ann.dim);
    data.sum_fxyz.resize(N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
    // Fp/parameters not needed for pure descriptor compute
  }

  void update_parameters_from_host(const float* host_parameters) {
    // Copy host parameters to device and map ANN pointers to device buffer
    data.parameters.resize(ann.num_para);
    data.parameters.copy_from_host(host_parameters);
    float* pointer = data.parameters.data();
    for (int t = 0; t < paramb.num_types; ++t) {
      if (t > 0 && paramb.version == 3) {
        pointer -= (ann.dim + 2) * ann.num_neurons1;
      }
      ann.w0[t] = pointer;                  pointer += ann.num_neurons1 * ann.dim;
      ann.b0[t] = pointer;                  pointer += ann.num_neurons1;
      ann.w1[t] = pointer;                  pointer += ann.num_neurons1;
    }
    ann.b1 = pointer;                        pointer += 1;
    ann.c = pointer;

    if (spin_enabled) {
      // Same parameter layout as NEP_Spin::update_potential().
      spin_ann.dim = ann.dim;
      spin_ann.num_neurons1 = ann.num_neurons1;
      spin_ann.num_para = ann.num_para;

      float* sp = data.parameters.data();
      for (int t = 0; t < spin_paramb.num_types; ++t) {
        if (t > 0 && spin_paramb.version == 3) {
          sp -= (spin_ann.dim + 2) * spin_ann.num_neurons1;
        }
        spin_ann.w0[t] = sp;                 sp += spin_ann.num_neurons1 * spin_ann.dim;
        spin_ann.b0[t] = sp;                 sp += spin_ann.num_neurons1;
        spin_ann.w1[t] = sp;                 sp += spin_ann.num_neurons1;
      }
      spin_ann.b1 = sp;                      sp += 1;
      spin_ann.c = sp;
    }
  }

  void compute_descriptors(Parameters& para, Dataset& dset) {
    const int N = dset.N;
    const int block_size = 32;
    const int grid_size = (N - 1) / block_size + 1;

    gpu_find_neighbor_list_desc<<<dset.Nc, 256>>>(
      paramb,
      dset.N,
      dset.Na.data(),
      dset.Na_sum.data(),
      dset.type.data(),
      para.rc_radial_max,
      para.rc_angular_max,
      dset.box.data(),
      dset.box_original.data(),
      dset.num_cell.data(),
      dset.r.data(),
      dset.r.data() + dset.N,
      dset.r.data() + dset.N * 2,
      data.NN_radial.data(),
      data.NL_radial.data(),
      data.NN_angular.data(),
      data.NL_angular.data(),
      data.x12_radial.data(),
      data.y12_radial.data(),
      data.z12_radial.data(),
      data.x12_angular.data(),
      data.y12_angular.data(),
      data.z12_angular.data());
    GPU_CHECK_KERNEL

    find_descriptors_radial_desc<<<grid_size, block_size>>>(
      dset.N,
      data.NN_radial.data(),
      data.NL_radial.data(),
      paramb,
      ann,
      dset.type.data(),
      data.x12_radial.data(),
      data.y12_radial.data(),
      data.z12_radial.data(),
      data.descriptors.data());
    GPU_CHECK_KERNEL

    find_descriptors_angular_desc<<<grid_size, block_size>>>(
      dset.N,
      data.NN_angular.data(),
      data.NL_angular.data(),
      paramb,
      ann,
      dset.type.data(),
      data.x12_angular.data(),
      data.y12_angular.data(),
      data.z12_angular.data(),
      data.descriptors.data(),
      data.sum_fxyz.data());
    GPU_CHECK_KERNEL

    // Spin descriptor block (written starting at spin_offset).
    if (spin_enabled && spin_dim > 0) {
      const bool has_spin = (dset.spin.size() == static_cast<size_t>(N) * 3);
      if (!has_spin) {
        // Zero only the spin portion for robustness (e.g., when users forget to pass spins).
        CHECK(gpuMemset(
          data.descriptors.data() + static_cast<size_t>(spin_offset) * static_cast<size_t>(N),
          0,
          static_cast<size_t>(spin_dim) * static_cast<size_t>(N) * sizeof(float)));
      } else {
        launch_find_descriptors_radial_spin_spherical_full(
          dim3(grid_size),
          dim3(block_size),
          N,
          data.NN_radial.data(),
          data.NL_radial.data(),
          spin_paramb,
          spin_ann,
          dset.type.data(),
          data.x12_radial.data(),
          data.y12_radial.data(),
          data.z12_radial.data(),
          dset.spin.data(),
          data.descriptors.data(),
          spin_offset,
          0);
        GPU_CHECK_KERNEL
      }
    }
  }

  void copy_descriptors_to_host(std::vector<float>& out) {
    out.resize(data.descriptors.size());
    data.descriptors.copy_to_host(out.data());
  }

  int descriptor_dim() const { return ann.dim; }

private:
  NEP::ParaMB paramb{};
  NEP::ANN ann{};
  NEP_Data data{};

  // Optional NEP-Spin descriptor support.
  bool spin_enabled = false;
  int spin_offset = 0;
  int spin_dim = 0;
  NEP_Spin::ParaMB spin_paramb{};
  NEP_Spin::ANN spin_ann{};
};
