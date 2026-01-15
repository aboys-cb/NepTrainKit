/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <cuda_runtime.h>
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
class Parameters;
class Dataset;

struct NEP_Spin_Data {
  GPU_Vector<int> NN_radial;  // radial neighbor number
  GPU_Vector<int> NL_radial;  // radial neighbor list
  GPU_Vector<int> NN_angular; // angular neighbor number
  GPU_Vector<int> NL_angular; // angular neighbor list
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;    // angular helper
  GPU_Vector<float> parameters;  // parameters to be optimized
};

class NEP_Spin : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial[NUM_ELEMENTS] = {0.0f};  // radial cutoff
    float rc_angular[NUM_ELEMENTS] = {0.0f}; // angular cutoff
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;  // n = 0..n_max_radial
    int n_max_angular = 0; // n = 0..n_max_angular
    int L_max = 0;         // l = 1..L_max
    int dim_angular;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_c_spin = 0;          // total number of spin radial c coefficients (spin_mode 2/3), else 0
    int spin_blocks = 0;         // number of enabled spin blocks (sum over ex/dmi/ani/sia)
    int c_spin_offset = 0;       // starting index of c_spin within ANN.c
    int c_spin_block_stride = 0; // number of c coefficients per spin block
    int version = 4;
    int atomic_numbers[NUM_ELEMENTS];
    float mforce_sign = -1.0f; // magnetic force sign: -dE/ds by default
    int spin_kmax_ex = 2;       // exchange Chebyshev order (k=0..spin_kmax_ex)
    int spin_kmax_dmi = 0;      // DMI Chebyshev order (k=0..spin_kmax_dmi), -1 disables
    int spin_kmax_ani = 0;      // ANI Chebyshev order (k=0..spin_kmax_ani), -1 disables
    int spin_kmax_sia = 0;      // SIA Chebyshev order (k=0..spin_kmax_sia), -1 disables
    int spin_pmax = 2;          // on-site longitudinal order (p=1..spin_pmax)
    int spin_ex_phi_mode = 0;       // exchange amplitude mode (see Parameters::spin_ex_phi_mode)
    int spin_onsite_basis_mode = 0; // on-site basis mode (see Parameters::spin_onsite_basis_mode)
    float spin_mref = 1.0f;         // reference magnitude for on-site Chebyshev mapping
  };

  struct ANN {
    int dim = 0;             // descriptor dimension
    int num_neurons1 = 0;    // hidden neurons
    int num_para = 0;        // number of parameters
    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;  // bias for output
    const float* c;   // descriptor mixing coefficients
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  NEP_Spin(
    Parameters& para,
    int N,
    int N_times_max_NN_radial,
    int N_times_max_NN_angular,
    int version,
    int deviceCount);

  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    bool calculate_neighbor,
    int deviceCount) override;

private:
  ParaMB paramb;
  ANN annmb[16];
  NEP_Spin_Data nep_data[16];
  ZBL zbl;
  void update_potential(Parameters& para, float* parameters, ANN& ann);
};

// ----------------------------------------------------------------------
// Spherical spin-feature kernels (host dispatch APIs)
// ----------------------------------------------------------------------
// Declarations live here (instead of a separate nep_spin_spherical.cuh) to keep the
// interface surface aligned with the "one-module header" style used in GPUMD.

// Host-side dispatch to a kmax_pair-specialized full descriptor kernel (k=0..8),
// to avoid paying the register cost of K=8 when kmax_pair is small.
void launch_find_descriptors_radial_spin_spherical_full(
  const dim3 grid,
  const dim3 block,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  float* g_descriptors,
  int spin_offset,
  cudaStream_t stream = 0);

// Host-side dispatch to a kmax_pair-specialized full force kernel (k=0..8).
void launch_find_force_radial_spin_spherical_full(
  const dim3 grid,
  const dim3 block,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  int spin_offset,
  cudaStream_t stream = 0);


// Host-side dispatch to a kmax_pair-specialized full mforce kernel (k=0..8).
void launch_find_mforce_radial_spin_spherical_full(
  const dim3 grid,
  const dim3 block,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  float* g_mx,
  float* g_my,
  float* g_mz,
  int spin_offset,
  cudaStream_t stream = 0);
