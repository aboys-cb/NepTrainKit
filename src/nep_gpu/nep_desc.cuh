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
#include "utilities/gpu_macro.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/common.cuh"
#include "mic.cuh"

// Kernels (implemented in nep_desc.cu)
__global__ void gpu_find_neighbor_list_desc(
  const NEP::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const bool use_typewise_cutoff,
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
    paramb.rc_radial = para.rc_radial;
    paramb.rcinv_radial = 1.0f / paramb.rc_radial;
    paramb.rc_angular = para.rc_angular;
    paramb.rcinv_angular = 1.0f / paramb.rc_angular;
    paramb.use_typewise_cutoff = para.use_typewise_cutoff;
    paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
    paramb.typewise_cutoff_radial_factor = para.typewise_cutoff_radial_factor;
    paramb.typewise_cutoff_angular_factor = para.typewise_cutoff_angular_factor;
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

    for (int n = 0; n < (int)para.atomic_numbers.size(); ++n) {
      paramb.atomic_numbers[n] = para.atomic_numbers[n] - 1; // starting from 0
    }

    ann.dim = para.dim;
    ann.num_neurons1 = para.num_neurons1;
    ann.num_para = para.number_of_variables;

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
      para.use_typewise_cutoff,
      dset.type.data(),
      para.rc_radial,
      para.rc_angular,
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
};
