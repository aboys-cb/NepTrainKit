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

#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

class Parameters;
class Dataset;

// Independent spin-only NEP implementation.
// NEP_Spin encapsulates the full NEP pipeline but assumes spin mode is active,
// and only handles magnetic use cases (real+pseudo atoms, magnetic force, etc.).
class NEP_Spin : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 2.5f;
    float typewise_cutoff_angular_factor = 2.0f;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial = 0.0f;
    float rc_angular = 0.0f;
    float rcinv_radial = 0.0f;
    float rcinv_angular = 0.0f;
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular = 0;
    int num_L = 0;
    int num_types = 0;      // total types (real + virtual)
    int num_types_real = 0; // real-only types
    int num_types_sq = 0;
    int num_c_radial = 0;
    int debug_disable_type_fold_for_ann = 0;
    int version = 4;
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;
    const float* c;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types = 0; // real types only for flexible ZBL
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct Data {
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
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> parameters;  // parameters to be optimized
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

  // Optional: expose descriptors for debugging
  inline GPU_Vector<float>& get_descriptors(int device_id) { return data_[device_id].descriptors; }

private:
  void update_potential(Parameters& para, float* parameters, ANN& ann);

  ParaMB paramb_;
  ANN annmb_[16];
  Data data_[16];
  ZBL zbl_;
};

