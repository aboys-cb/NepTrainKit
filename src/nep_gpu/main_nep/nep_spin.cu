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

/*----------------------------------------------------------------------------80
Spin-only NEP implementation consolidated as a single compilation unit.
Function ordering follows src/main_nep/nep.cu for consistency.
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "nep_spin.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef NEP_SPIN_PROFILE
#include <cuda_runtime.h>
struct ProfTimer {
  cudaEvent_t a, b;
  ProfTimer() { cudaEventCreate(&a); cudaEventCreate(&b); }
  ~ProfTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start() { cudaEventRecord(a); }
  float stop() {
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
  }
};
#endif

// -------------------------------
// Neighbor list (spin-specific)
// -------------------------------

// Neighbor list construction and spin-specific injections
// (Section NEIGHBORS begins)

static __global__ void gpu_find_neighbor_list(
  const NEP_Spin::ParaMB paramb,
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
  const int max_NN_radial_cap,
  const int max_NN_angular_cap,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular,
  const int* __restrict__ g_is_pseudo,
  const int skip_pseudo_centers)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    if (skip_pseudo_centers && g_is_pseudo != nullptr && g_is_pseudo[n1]) {
      NN_radial[n1] = 0;
      NN_angular[n1] = 0;
      continue;
    }
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      // base list keeps only real->real pairs; skip any pseudo neighbor
      if (g_is_pseudo != nullptr && g_is_pseudo[n2]) {
        continue;
      }
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_radial = g_rc_radial;
            float rc_angular = g_rc_angular;
            if (use_typewise_cutoff) {
              int z1 = paramb.atomic_numbers[t1];
              int z2 = paramb.atomic_numbers[t2];
              rc_radial = min(
                (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor,
                rc_radial);
              rc_angular = min(
                (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor,
                rc_angular);
            }
            if (distance_square < rc_radial * rc_radial) {
              if (count_radial < max_NN_radial_cap) {
                NL_radial[count_radial * N + n1] = n2;
                x12_radial[count_radial * N + n1] = x12;
                y12_radial[count_radial * N + n1] = y12;
                z12_radial[count_radial * N + n1] = z12;
                count_radial++;
              }
            }
            if (distance_square < rc_angular * rc_angular) {
              if (count_angular < max_NN_angular_cap) {
                NL_angular[count_angular * N + n1] = n2;
                x12_angular[count_angular * N + n1] = x12;
                y12_angular[count_angular * N + n1] = y12;
                z12_angular[count_angular * N + n1] = z12;
                count_angular++;
              }
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

static __global__ void inject_self_neighbor(
  const int N,
  const int N_real,
  const int Nc,
  const int* __restrict__ Na,
  const int* __restrict__ Na_sum,
  const float* __restrict__ g_box,
  const int* __restrict__ host2real,
  const int* __restrict__ host2pseudo,
  const float* __restrict__ rx,
  const float* __restrict__ ry,
  const float* __restrict__ rz,
  int* NN,
  int* NL,
  const int max_NN,
  float* x12,
  float* y12,
  float* z12)
{
  int bid = blockIdx.x;
  int N1 = Na_sum[bid];
  int N2 = N1 + Na[bid];
  const float* __restrict__ box = g_box + 18 * bid;
  for (int i = threadIdx.x; i < N_real; i += blockDim.x) {
    int nr = host2real[i];
    int np = host2pseudo[i];
    if (nr < 0 || np < 0) continue;
    if (nr < N1 || nr >= N2) continue; // center must be in this block
    // inject pseudo as a neighbor to its real center (with de-dup)
    int count = NN[nr];
    bool found = false;
    for (int k = 0; k < count; ++k) {
      int idx = k * N + nr;
      if (NL[idx] == np) { found = true; break; }
    }
    if (!found && count < max_NN) {
      float dx = rx[np] - rx[nr];
      float dy = ry[np] - ry[nr];
      float dz = rz[np] - rz[nr];
      dev_apply_mic(box, dx, dy, dz);
      int w = count * N + nr;
      NL[w] = np;
      x12[w] = dx;
      y12[w] = dy;
      z12[w] = dz;
      NN[nr] = count + 1;
    }
  }
}

static __global__ void inject_mirror_neighbors(
  const int N,
  const int Nc,
  const int* __restrict__ Na,
  const int* __restrict__ Na_sum,
  const float* __restrict__ g_box,
  const int* __restrict__ g_is_pseudo,
  const int* __restrict__ real2pseudo,
  const int* __restrict__ pseudo2real,
  const float* __restrict__ rx,
  const float* __restrict__ ry,
  const float* __restrict__ rz,
  int* NN,
  int* NL,
  const int max_NN,
  float* x12,
  float* y12,
  float* z12,
  const float rc2,
  int* overflow)
{
  int bid = blockIdx.x;
  int N1 = Na_sum[bid];
  int N2 = N1 + Na[bid];
  const float* __restrict__ box = g_box + 18 * bid;
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    if (g_is_pseudo && g_is_pseudo[n1]) continue;
    int count = NN[n1];
    int nn0 = count; // iterate over original neighbors only
    for (int i = 0; i < nn0; ++i) {
      int n2 = NL[i * N + n1];
      int m = pseudo2real[n2];
      if (m < 0) continue;
      int p = real2pseudo[n1];
      if (p < 0 || p == n2) continue;
      // de-dup check before computing MIC
      bool exists = false;
      for (int t = 0; t < count; ++t) {
        if (NL[t * N + n1] == p) { exists = true; break; }
      }
      if (exists) continue;
      float dx = rx[p] - rx[n1];
      float dy = ry[p] - ry[n1];
      float dz = rz[p] - rz[n1];
      dev_apply_mic(box, dx, dy, dz);
      float r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < rc2) {
        if (count < max_NN) {
          NL[count * N + n1] = p;
          x12[count * N + n1] = dx;
          y12[count * N + n1] = dy;
          z12[count * N + n1] = dz;
          count++;
        } else {
          overflow[n1] = 1;
        }
      }
    }
    NN[n1] = count;
  }
}

// Extend neighbor list with pseudo neighbors (PT-style):
// base NN/NL contain only real->real; we append (1) self pseudo and (2) each neighbor's pseudo.
static __global__ void extend_with_pseudo_neighbors(
  const int N,
  const int Nc,
  const int* __restrict__ Na,
  const int* __restrict__ Na_sum,
  const int* __restrict__ g_is_pseudo,
  const int* __restrict__ real2pseudo,
  const float* __restrict__ rx,
  const float* __restrict__ ry,
  const float* __restrict__ rz,
  int* NN,
  int* NL,
  const int max_NN,
  float* x12,
  float* y12,
  float* z12,
  const float* __restrict__ g_box,
  int* overflow)
{
  int bid = blockIdx.x;
  int N1 = Na_sum[bid];
  int N2 = N1 + Na[bid];
  const float* __restrict__ box = g_box + 18 * bid;
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    if (g_is_pseudo && g_is_pseudo[n1]) continue; // real centers only
    int count_base = NN[n1];
    int write_count = count_base;

    // 1) append self pseudo if exists
    int p1 = real2pseudo[n1];
    if (p1 >= 0) {
      if (write_count < max_NN) {
        float dx = rx[p1] - rx[n1];
        float dy = ry[p1] - ry[n1];
        float dz = rz[p1] - rz[n1];
        // conservative MIC for self-pseudo only
        dev_apply_mic(box, dx, dy, dz);
        int w = write_count * N + n1;
        NL[w] = p1;
        x12[w] = dx;
        y12[w] = dy;
        z12[w] = dz;
        write_count++;
      } else if (overflow) {
        overflow[n1] = 1;
      }
    }

    // 2) append each neighbor's pseudo (reuse base MIC via x12 + spin shift)
    for (int i = 0; i < count_base; ++i) {
      int idx = i * N + n1;
      int n2 = NL[idx];
      int p2 = real2pseudo[n2];
      if (p2 < 0) continue;
      if (write_count < max_NN) {
        float dx = x12[idx] + (rx[p2] - rx[n2]);
        float dy = y12[idx] + (ry[p2] - ry[n2]);
        float dz = z12[idx] + (rz[p2] - rz[n2]);
        // Apply MIC after adding spin shift to keep consistency with base pair MIC
        dev_apply_mic(box, dx, dy, dz);
        int w = write_count * N + n1;
        NL[w] = p2;
        x12[w] = dx;
        y12[w] = dy;
        z12[w] = dz;
        write_count++;
      } else if (overflow) {
        overflow[n1] = 1;
      }
    }

    NN[n1] = write_count;
  }
}

// (Section NEIGHBORS ends)

// -------------------------------
// Descriptor construction
// -------------------------------

static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  const int* __restrict__ g_is_pseudo,
  const int skip_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    if (skip_pseudo && g_is_pseudo != nullptr && g_is_pseudo[n1]) {
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        g_descriptors[n1 + n * N] = 0.0f;
      }
      return;
    }
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);

      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

// Real-center version: compute descriptors only for centers listed in g_centers (global indices)
static __global__ void find_descriptors_radial_idx(
  const int N,
  const int N_centers,
  const int* __restrict__ g_centers,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_centers) {
    int n1 = g_centers[i];
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz,
  const int* __restrict__ g_is_pseudo,
  const int skip_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    if (skip_pseudo && g_is_pseudo != nullptr && g_is_pseudo[n1]) {
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        for (int l = 0; l < paramb.num_L; ++l) {
          int ln = l * (paramb.n_max_angular + 1) + n;
          g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = 0.0f;
        }
        for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = 0.0f;
        }
      }
      return;
    }

    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

// Real-center version: compute angular descriptors only for centers in g_centers
static __global__ void find_descriptors_angular_idx(
  const int N,
  const int N_centers,
  const int* __restrict__ g_centers,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_centers) {
    int n1 = g_centers[i];
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

// (Section DESCRIPTORS ends)

// -------------------------------
// ANN application and force kernels (part 1)
// -------------------------------


static void __global__ find_max_min(
  const int N,
  const float* __restrict__ g_q,
  float* __restrict__ g_q_scaler,
  const int* __restrict__ g_is_pseudo,
  const int skip_pseudo)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  __shared__ int s_cnt[1024];
  s_max[tid] = -1000000.0f;
  s_min[tid] = +1000000.0f;
  s_cnt[tid] = 0;
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      bool include = true;
      if (skip_pseudo && g_is_pseudo != nullptr) {
        include = (g_is_pseudo[n] == 0);
      }
      if (include) {
        const int m = n + N * bid;
        float q = g_q[m];
        if (q > s_max[tid]) s_max[tid] = q;
        if (q < s_min[tid]) s_min[tid] = q;
        s_cnt[tid] += 1;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) s_max[tid] = s_max[tid + offset];
      if (s_min[tid] > s_min[tid + offset]) s_min[tid] = s_min[tid + offset];
      s_cnt[tid] += s_cnt[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (s_cnt[0] > 0) {
      g_q_scaler[bid] = min(g_q_scaler[bid], 1.0f / (s_max[0] - s_min[0]));
    }
  }
}

// Real-center version of find_max_min: only over indices in g_centers
static void __global__ find_max_min_idx(
  const int N_centers,
  const int* __restrict__ g_centers,
  const int N_total,
  const float* __restrict__ g_q,
  float* __restrict__ g_q_scaler)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f;
  s_min[tid] = +1000000.0f;
  const int stride = 1024;
  const int number_of_rounds = (N_centers - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int i = round * stride + tid;
    if (i < N_centers) {
      int n = g_centers[i];
      const int m = n + N_total * bid;
      float q = g_q[m];
      if (q > s_max[tid]) s_max[tid] = q;
      if (q < s_min[tid]) s_min[tid] = q;
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) s_max[tid] = s_max[tid + offset];
      if (s_min[tid] > s_min[tid + offset]) s_min[tid] = s_min[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler[bid] = min(g_q_scaler[bid], 1.0f / (s_max[0] - s_min[0]));
  }
}

static __global__ void apply_ann(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp,
  const int* __restrict__ g_is_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  int type = g_type[n1];
  if (n1 < N) {
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    bool is_pseudo = (g_is_pseudo != nullptr) ? (g_is_pseudo[n1] != 0) : false;
    int type_eff = type;
    if (!paramb.debug_disable_type_fold_for_ann && paramb.num_types_real > 0 && type >= paramb.num_types_real) {
      type_eff = type - paramb.num_types_real;
    }
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type_eff],
      annmb.b0[type_eff],
      annmb.w1[type_eff],
      annmb.b1,
      q,
      F,
      Fp);
    if (is_pseudo) {
      g_pe[n1] = 0.0f;
    } else {
      g_pe[n1] = F;
    }
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

// Real-center version
static __global__ void apply_ann_idx(
  const int N,
  const int N_centers,
  const int* __restrict__ g_centers,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_centers) {
    int n1 = g_centers[i];
    int type = g_type[n1];
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    int type_eff = type;
    if (!paramb.debug_disable_type_fold_for_ann && paramb.num_types_real > 0 && type >= paramb.num_types_real) {
      type_eff = type - paramb.num_types_real;
    }
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type_eff],
      annmb.b0[type_eff],
      annmb.w1[type_eff],
      annmb.b1,
      q,
      F,
      Fp);
    g_pe[n1] = F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void apply_ann_temperature(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  float* __restrict__ g_q_scaler,
  const float* __restrict__ g_temperature,
  float* g_pe,
  float* g_Fp,
  const int* __restrict__ g_is_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  int type = g_type[n1];
  float temperature = g_temperature[n1];
  if (n1 < N) {
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim - 1; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    g_q_scaler[annmb.dim - 1] = 0.001; // temperature dimension scaler
    q[annmb.dim - 1] = temperature * g_q_scaler[annmb.dim - 1];
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    bool is_pseudo = (g_is_pseudo != nullptr) ? (g_is_pseudo[n1] != 0) : false;
    int type_eff = type;
    if (!paramb.debug_disable_type_fold_for_ann && paramb.num_types_real > 0 && type >= paramb.num_types_real) {
      type_eff = type - paramb.num_types_real;
    }
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type_eff],
      annmb.b0[type_eff],
      annmb.w1[type_eff],
      annmb.b1,
      q,
      F,
      Fp);
    if (is_pseudo) {
      g_pe[n1] = 0.0f;
    } else {
      g_pe[n1] = F;
    }
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

// Real-center version with temperature
static __global__ void apply_ann_temperature_idx(
  const int N,
  const int N_centers,
  const int* __restrict__ g_centers,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  float* __restrict__ g_q_scaler,
  const float* __restrict__ g_temperature,
  float* g_pe,
  float* g_Fp)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_centers) {
    int n1 = g_centers[i];
    int type = g_type[n1];
    float temperature = g_temperature[n1];
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim - 1; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    g_q_scaler[annmb.dim - 1] = 0.001; // keep behavior
    q[annmb.dim - 1] = temperature * g_q_scaler[annmb.dim - 1];
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    int type_eff = type;
    if (!paramb.debug_disable_type_fold_for_ann && paramb.num_types_real > 0 && type >= paramb.num_types_real) {
      type_eff = type - paramb.num_types_real;
    }
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type_eff],
      annmb.b0[type_eff],
      annmb.w1[type_eff],
      annmb.b1,
      q,
      F,
      Fp);
    g_pe[n1] = F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

// Zero energy on pseudo atoms to keep per-structure energy sums correct
static __global__ void zero_pseudo_energy(
  int N_real,
  const int* __restrict__ g_host2pseudo,
  float* __restrict__ g_pe)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_real) {
    int np = g_host2pseudo[i];
    if (np >= 0) g_pe[np] = 0.0f;
  }
}

static __global__ void zero_force(
  const int N, float* g_fx, float* g_fy, float* g_fz, float* g_vxx, float* g_vyy, float* g_vzz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    g_vxx[n1] = 0.0f;
    g_vyy[n1] = 0.0f;
    g_vzz[n1] = 0.0f;
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  const int* __restrict__ g_is_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    if (g_is_pseudo != nullptr && g_is_pseudo[n1]) {
      return;
    }
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] = s_virial_xy;
    g_virial[n1 + N * 4] = s_virial_yz;
    g_virial[n1 + N * 5] = s_virial_zx;
  }
}

// (Section ANN-FORCE part1 ends)

// -------------------------------
// ANN application and force kernels (part 2)
// -------------------------------

static __global__ void find_force_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  const int* __restrict__ g_is_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    if (g_is_pseudo != nullptr && g_is_pseudo[n1]) {
      return;
    }
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int neighbor_number = g_NN[n1];
    int type1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int type2 = g_type[n2];
      float rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[type1]] + COVALENT_RADIUS[paramb.atomic_numbers[type2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += type1 * paramb.num_types + type2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ZBL zbl,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe,
  const int* __restrict__ g_is_pseudo)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    // Do not compute ZBL when the center is a pseudo atom; also skip any
    // ZBL contribution that involves a pseudo neighbor. Pseudo atoms are
    // auxiliary for spin coupling and should not participate in ZBL.
    if (g_is_pseudo != nullptr && g_is_pseudo[n1]) {
      return;
    }
    float s_pe = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1]; // starting from 1
    float pow_zi = pow(float(zi), 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      // Skip pairs that involve a pseudo neighbor to avoid spurious ZBL
      if (g_is_pseudo != nullptr && g_is_pseudo[n2]) {
        continue;
      }
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2]; // starting from 1
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = rc_outer * 0.5f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
      s_pe += f * 0.5f;
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
    g_pe[n1] += s_pe;
  }
}

// (Section ANN-FORCE part2 ends)

// -------------------------------
// Spin-mode helpers
// -------------------------------

static __global__ void update_pseudo_positions(
  int N_real,
  const int* __restrict__ g_host2pseudo,
  const int* __restrict__ g_host2real,
  const float* __restrict__ g_alpha,
  const float* __restrict__ g_spin,
  float* __restrict__ g_rx,
  float* __restrict__ g_ry,
  float* __restrict__ g_rz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_real) {
    int n_real = g_host2real[i];
    int n_pseudo = g_host2pseudo[i];
    if (n_pseudo < 0) return;
    float a = g_alpha[i];
    if (a == 0.0f) return;
    float sx = g_spin[i];
    float sy = g_spin[i + N_real];
    float sz = g_spin[i + 2 * N_real];
    g_rx[n_pseudo] = g_rx[n_real] + a * sx;
    g_ry[n_pseudo] = g_ry[n_real] + a * sy;
    g_rz[n_pseudo] = g_rz[n_real] + a * sz;
  }
}


// Combined post-processing for spin after force evaluation.
// Order within each i (real atom):
//   1) accumulate pseudo force to its real counterpart (mechanical force)
//   2) derive magnetic force Fm = alpha * F_pseudo
//   3) add virial correction using dr = -alpha * s and F_pseudo
// Note: ensure step (3) happens after (1), as requested.
static __global__ void postprocess_spin_after_forces(
  int N,
  int N_real,
  const int* __restrict__ g_host2real,
  const int* __restrict__ g_host2pseudo,
  const float* __restrict__ g_alpha,
  const float* __restrict__ g_spin,
  float* __restrict__ g_fx,
  float* __restrict__ g_fy,
  float* __restrict__ g_fz,
  float* __restrict__ g_fm_x,
  float* __restrict__ g_fm_y,
  float* __restrict__ g_fm_z,
  float* __restrict__ g_virial)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N_real) {
    int n_real = g_host2real[i];
    int n_pseudo = g_host2pseudo[i];

    // default outputs for magnetic force
    float fm_x = 0.0f, fm_y = 0.0f, fm_z = 0.0f;

    if (n_real >= 0 && n_pseudo >= 0) {
      float a = g_alpha[i];

      // 1) accumulate pseudo force to real
      g_fx[n_real] += g_fx[n_pseudo];
      g_fy[n_real] += g_fy[n_pseudo];
      g_fz[n_real] += g_fz[n_pseudo];

      // 2) derive magnetic force from pseudo force
      if (a != 0.0f) {
        fm_x = a * g_fx[n_pseudo];
        fm_y = a * g_fy[n_pseudo];
        fm_z = a * g_fz[n_pseudo];
      }

      // 3) add virial correction using dr = -a * s and F_pseudo
      if (a != 0.0f) {
        float drx = -a * g_spin[i];
        float dry = -a * g_spin[i + N_real];
        float drz = -a * g_spin[i + 2 * N_real];
        float Fpx = g_fx[n_pseudo];
        float Fpy = g_fy[n_pseudo];
        float Fpz = g_fz[n_pseudo];
        atomicAdd(&g_virial[n_real + 0 * N], drx * Fpx);
        atomicAdd(&g_virial[n_real + 1 * N], dry * Fpy);
        atomicAdd(&g_virial[n_real + 2 * N], drz * Fpz);
        atomicAdd(&g_virial[n_real + 3 * N], drx * Fpy);
        atomicAdd(&g_virial[n_real + 4 * N], dry * Fpz);
        atomicAdd(&g_virial[n_real + 5 * N], drz * Fpx);
      }
    }

    // write magnetic force prediction (zero if no pseudo mapping)
    g_fm_x[i] = fm_x;
    g_fm_y[i] = fm_y;
    g_fm_z[i] = fm_z;
  }
}

// (Section HELPERS ends)

// -------------------------------
// NEP_Spin methods (constructor + update_potential)
// -------------------------------

NEP_Spin::NEP_Spin(
  Parameters& para,
  int N,
  int N_times_max_NN_radial,
  int N_times_max_NN_angular,
  int version,
  int deviceCount)
{
  paramb_.version = version;
  paramb_.rc_radial = para.rc_radial;
  paramb_.rcinv_radial = 1.0f / paramb_.rc_radial;
  paramb_.rc_angular = para.rc_angular;
  paramb_.rcinv_angular = 1.0f / paramb_.rc_angular;
  paramb_.use_typewise_cutoff = para.use_typewise_cutoff;
  paramb_.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb_.typewise_cutoff_radial_factor = para.typewise_cutoff_radial_factor;
  paramb_.typewise_cutoff_angular_factor = para.typewise_cutoff_angular_factor;
  paramb_.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  // spin-only: total types = real + virtual
  paramb_.num_types_real = para.num_types;
  paramb_.num_types = para.num_types * 2;
  paramb_.debug_disable_type_fold_for_ann = para.debug_disable_type_fold_for_ann;

  paramb_.n_max_radial = para.n_max_radial;
  paramb_.n_max_angular = para.n_max_angular;
  paramb_.L_max = para.L_max;
  paramb_.num_L = paramb_.L_max;
  if (para.L_max_4body == 2) paramb_.num_L += 1;
  if (para.L_max_5body == 1) paramb_.num_L += 1;
  paramb_.dim_angular = (para.n_max_angular + 1) * paramb_.num_L;

  paramb_.basis_size_radial = para.basis_size_radial;
  paramb_.basis_size_angular = para.basis_size_angular;
  paramb_.num_types_sq = paramb_.num_types * paramb_.num_types;
  paramb_.num_c_radial = paramb_.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

  zbl_.enabled = para.enable_zbl;
  zbl_.flexibled = para.flexible_zbl;
  zbl_.rc_inner = para.zbl_rc_inner;
  zbl_.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < (int)para.atomic_numbers.size(); ++n) {
    zbl_.atomic_numbers[n] = para.atomic_numbers[n];        // start from 1 (real types)
    paramb_.atomic_numbers[n] = para.atomic_numbers[n] - 1; // start from 0
  }
  // mirror real atomic numbers into virtual type slots
  for (int n = 0; n < para.num_types; ++n) {
    int v = n + para.num_types;
    paramb_.atomic_numbers[v] = paramb_.atomic_numbers[n];
  }
  if (zbl_.flexibled) {
    zbl_.num_types = para.num_types; // table defined on real types
    int num_type_zbl = (para.num_types * (para.num_types + 1)) / 2;
    for (int n = 0; n < num_type_zbl * 10; ++n) {
      zbl_.para[n] = para.zbl_para[n];
    }
  }

  for (int device_id = 0; device_id < deviceCount; device_id++) {
    gpuSetDevice(device_id);
    annmb_[device_id].dim = para.dim;
    annmb_[device_id].num_neurons1 = para.num_neurons1;
    annmb_[device_id].num_para = para.number_of_variables;

    data_[device_id].NN_radial.resize(N);
    data_[device_id].NN_angular.resize(N);
    data_[device_id].NL_radial.resize(N_times_max_NN_radial);
    data_[device_id].NL_angular.resize(N_times_max_NN_angular);
    data_[device_id].x12_radial.resize(N_times_max_NN_radial);
    data_[device_id].y12_radial.resize(N_times_max_NN_radial);
    data_[device_id].z12_radial.resize(N_times_max_NN_radial);
    data_[device_id].x12_angular.resize(N_times_max_NN_angular);
    data_[device_id].y12_angular.resize(N_times_max_NN_angular);
    data_[device_id].z12_angular.resize(N_times_max_NN_angular);
    data_[device_id].descriptors.resize(N * annmb_[device_id].dim);
    data_[device_id].Fp.resize(N * annmb_[device_id].dim);
    data_[device_id].sum_fxyz.resize(
      N * (paramb_.n_max_angular + 1) * ((paramb_.L_max + 1) * (paramb_.L_max + 1) - 1));
    data_[device_id].parameters.resize(annmb_[device_id].num_para);
  }
}

void NEP_Spin::update_potential(Parameters& /*para*/, float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb_.num_types_real; ++t) {
    if (t > 0 && paramb_.version == 3) {
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
  // alias virtual types to real counterparts
  for (int t = paramb_.num_types_real; t < paramb_.num_types; ++t) {
    int real_t = t - paramb_.num_types_real;
    ann.w0[t] = ann.w0[real_t];
    ann.b0[t] = ann.b0[real_t];
    ann.w1[t] = ann.w1[real_t];
  }
}
// -------------------------------
// NEP_Spin methods (find_force)
// -------------------------------

void NEP_Spin::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  bool calculate_neighbor,
  int device_in_this_iter)
{
  // copy parameters and update per-device ANN views
  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    data_[device_id].parameters.copy_from_host(parameters + device_id * para.number_of_variables);
    update_potential(para, data_[device_id].parameters.data(), annmb_[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    const int block_size = 32;
    const int grid_size = (dataset[device_id].N - 1) / block_size + 1;
    
#ifdef NEP_SPIN_PROFILE
    float ms_desc = 0.0f, ms_ann = 0.0f, ms_force = 0.0f, ms_post = 0.0f, ms_neighbor = 0.0f;
    ProfTimer timer;
#endif

    // update pseudo coordinates if recalc neighbors requested
    if (calculate_neighbor && dataset[device_id].N_real > 0) {
      int grid_up = (dataset[device_id].N_real - 1) / 256 + 1;
      update_pseudo_positions<<<grid_up, 256>>>(
        dataset[device_id].N_real,
        dataset[device_id].host2pseudo.data(),
        dataset[device_id].host2real.data(),
        dataset[device_id].alpha.data(),
        dataset[device_id].spin.data(),
        dataset[device_id].r.data(),
        dataset[device_id].r.data() + dataset[device_id].N,
        dataset[device_id].r.data() + dataset[device_id].N * 2);
      GPU_CHECK_KERNEL
    }

    if (calculate_neighbor) {
      #ifdef NEP_SPIN_PROFILE
      timer.start();
      #endif
      // neighbor list with spin awareness
      gpu_find_neighbor_list<<<dataset[device_id].Nc, 256>>>(
        paramb_,
        dataset[device_id].N,
        dataset[device_id].Na.data(),
        dataset[device_id].Na_sum.data(),
        para.use_typewise_cutoff,
        dataset[device_id].type.data(),
        para.rc_radial,
        para.rc_angular,
        dataset[device_id].box.data(),
        dataset[device_id].box_original.data(),
        dataset[device_id].num_cell.data(),
        dataset[device_id].r.data(),
        dataset[device_id].r.data() + dataset[device_id].N,
        dataset[device_id].r.data() + dataset[device_id].N * 2,
        dataset[device_id].max_NN_radial,
        dataset[device_id].max_NN_angular,
        data_[device_id].NN_radial.data(),
        data_[device_id].NL_radial.data(),
        data_[device_id].NN_angular.data(),
        data_[device_id].NL_angular.data(),
        data_[device_id].x12_radial.data(),
        data_[device_id].y12_radial.data(),
        data_[device_id].z12_radial.data(),
        data_[device_id].x12_angular.data(),
        data_[device_id].y12_angular.data(),
        data_[device_id].z12_angular.data(),
        dataset[device_id].is_pseudo.data(),
        1);
      GPU_CHECK_KERNEL

      // PT-style neighbor extension: build real->real base; then append pseudo neighbors in one pass
      if (dataset[device_id].N_real > 0) {
        int gridN = dataset[device_id].Nc;
        // build real_index -> pseudo_index mapping on host (size N), -1 if no pseudo
        GPU_Vector<int> real2pseudo;
        real2pseudo.resize(dataset[device_id].N);
        {
          std::vector<int> h_real2pseudo(dataset[device_id].N, -1);
          std::vector<int> h_host2real(dataset[device_id].N_real);
          std::vector<int> h_host2pseudo(dataset[device_id].N_real);
          dataset[device_id].host2real.copy_to_host(h_host2real.data());
          dataset[device_id].host2pseudo.copy_to_host(h_host2pseudo.data());
          for (int i = 0; i < dataset[device_id].N_real; ++i) {
            int nr = h_host2real[i];
            int np = h_host2pseudo[i];
            if (nr >= 0) h_real2pseudo[nr] = np;
          }
          real2pseudo.copy_from_host(h_real2pseudo.data());
        }

        GPU_Vector<int> overflow_extend_radial, overflow_extend_angular;
        overflow_extend_radial.resize(dataset[device_id].N);
        overflow_extend_angular.resize(dataset[device_id].N);

        // extend radial list
        extend_with_pseudo_neighbors<<<gridN, 256>>>(
          dataset[device_id].N,
          dataset[device_id].Nc,
          dataset[device_id].Na.data(),
          dataset[device_id].Na_sum.data(),
          dataset[device_id].is_pseudo.data(),
          real2pseudo.data(),
          dataset[device_id].r.data(),
          dataset[device_id].r.data() + dataset[device_id].N,
          dataset[device_id].r.data() + dataset[device_id].N * 2,
          data_[device_id].NN_radial.data(),
          data_[device_id].NL_radial.data(),
          dataset[device_id].max_NN_radial,
          data_[device_id].x12_radial.data(),
          data_[device_id].y12_radial.data(),
          data_[device_id].z12_radial.data(),
          dataset[device_id].box.data(),
          overflow_extend_radial.data());
        GPU_CHECK_KERNEL

        // extend angular list
        extend_with_pseudo_neighbors<<<gridN, 256>>>(
          dataset[device_id].N,
          dataset[device_id].Nc,
          dataset[device_id].Na.data(),
          dataset[device_id].Na_sum.data(),
          dataset[device_id].is_pseudo.data(),
          real2pseudo.data(),
          dataset[device_id].r.data(),
          dataset[device_id].r.data() + dataset[device_id].N,
          dataset[device_id].r.data() + dataset[device_id].N * 2,
          data_[device_id].NN_angular.data(),
          data_[device_id].NL_angular.data(),
          dataset[device_id].max_NN_angular,
          data_[device_id].x12_angular.data(),
          data_[device_id].y12_angular.data(),
          data_[device_id].z12_angular.data(),
          dataset[device_id].box.data(),
          overflow_extend_angular.data());
        GPU_CHECK_KERNEL
      }
      #ifdef NEP_SPIN_PROFILE
      ms_neighbor += timer.stop();
      #endif
    }

    // descriptors: compute only on real centers
    if (dataset[device_id].N_real > 0) {
      #ifdef NEP_SPIN_PROFILE
      timer.start();
      #endif
      int grid_real = (dataset[device_id].N_real - 1) / block_size + 1;
      find_descriptors_radial<<<grid_size, block_size>>>(
        dataset[device_id].N,
        data_[device_id].NN_radial.data(),
        data_[device_id].NL_radial.data(),
        paramb_,
        annmb_[device_id],
        dataset[device_id].type.data(),
        data_[device_id].x12_radial.data(),
        data_[device_id].y12_radial.data(),
        data_[device_id].z12_radial.data(),
        data_[device_id].descriptors.data(),
        dataset[device_id].is_pseudo.data(),
        1);
      GPU_CHECK_KERNEL

      find_descriptors_angular<<<grid_size, block_size>>>(
        dataset[device_id].N,
        data_[device_id].NN_angular.data(),
        data_[device_id].NL_angular.data(),
        paramb_,
        annmb_[device_id],
        dataset[device_id].type.data(),
        data_[device_id].x12_angular.data(),
        data_[device_id].y12_angular.data(),
        data_[device_id].z12_angular.data(),
        data_[device_id].descriptors.data(),
        data_[device_id].sum_fxyz.data(),
        dataset[device_id].is_pseudo.data(),
        1);
      GPU_CHECK_KERNEL
      #ifdef NEP_SPIN_PROFILE
      ms_desc += timer.stop();
      #endif
    }

    // update q scaler (real centers only)
    if (calculate_q_scaler && dataset[device_id].N_real > 0) {
      find_max_min<<<annmb_[device_id].dim, 1024>>>(
        dataset[device_id].N,
        data_[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].is_pseudo.data(),
        1);
      GPU_CHECK_KERNEL
    }

    // zero accumulators
    zero_force<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].virial.data() + dataset[device_id].N,
      dataset[device_id].virial.data() + dataset[device_id].N * 2);
    GPU_CHECK_KERNEL

    // apply ANN only on real centers; keep pseudo energy = 0
    if (dataset[device_id].N_real > 0) {
      #ifdef NEP_SPIN_PROFILE
      timer.start();
      #endif
      int grid_real = (dataset[device_id].N_real - 1) / block_size + 1;
      if (para.train_mode == 3) {
        apply_ann_temperature<<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb_,
          annmb_[device_id],
          dataset[device_id].type.data(),
          data_[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].temperature_ref_gpu.data(),
          dataset[device_id].energy.data(),
          data_[device_id].Fp.data(),
          dataset[device_id].is_pseudo.data());
        GPU_CHECK_KERNEL
      } else {
        apply_ann<<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb_,
          annmb_[device_id],
          dataset[device_id].type.data(),
          data_[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].energy.data(),
          data_[device_id].Fp.data(),
          dataset[device_id].is_pseudo.data());
        GPU_CHECK_KERNEL
      }
      // ensure pseudo energies are zero for per-structure sums
      int grid_zero = (dataset[device_id].N_real - 1) / 256 + 1;
      zero_pseudo_energy<<<grid_zero, 256>>>(
        dataset[device_id].N_real,
        dataset[device_id].host2pseudo.data(),
        dataset[device_id].energy.data());
      GPU_CHECK_KERNEL
      #ifdef NEP_SPIN_PROFILE
      ms_ann += timer.stop();
      #endif
    }

    // forces from descriptors
    find_force_radial<<<grid_size, block_size>>>(
      dataset[device_id].N,
      data_[device_id].NN_radial.data(),
      data_[device_id].NL_radial.data(),
      paramb_,
      annmb_[device_id],
      dataset[device_id].type.data(),
      data_[device_id].x12_radial.data(),
      data_[device_id].y12_radial.data(),
      data_[device_id].z12_radial.data(),
      data_[device_id].Fp.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].is_pseudo.data());
    GPU_CHECK_KERNEL

    find_force_angular<<<grid_size, block_size>>>(
      dataset[device_id].N,
      data_[device_id].NN_angular.data(),
      data_[device_id].NL_angular.data(),
      paramb_,
      annmb_[device_id],
      dataset[device_id].type.data(),
      data_[device_id].x12_angular.data(),
      data_[device_id].y12_angular.data(),
      data_[device_id].z12_angular.data(),
      data_[device_id].Fp.data(),
      data_[device_id].sum_fxyz.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].is_pseudo.data());
    GPU_CHECK_KERNEL

    if (zbl_.enabled) {
      find_force_ZBL<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb_,
        zbl_,
        data_[device_id].NN_angular.data(),
        data_[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        data_[device_id].x12_angular.data(),
        data_[device_id].y12_angular.data(),
        data_[device_id].z12_angular.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data(),
        dataset[device_id].energy.data(),
        dataset[device_id].is_pseudo.data());
      GPU_CHECK_KERNEL
    }

    // spin post-processing: single pass for accumulate -> magnetic force -> virial correction
    if (dataset[device_id].N_real > 0) {
      #ifdef NEP_SPIN_PROFILE
      timer.start();
      #endif
      int grid_post = (dataset[device_id].N_real - 1) / 256 + 1;
      postprocess_spin_after_forces<<<grid_post, 256>>>(
        dataset[device_id].N,
        dataset[device_id].N_real,
        dataset[device_id].host2real.data(),
        dataset[device_id].host2pseudo.data(),
        dataset[device_id].alpha.data(),
        dataset[device_id].spin.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].fm_pred.data(),
        dataset[device_id].fm_pred.data() + dataset[device_id].N_real,
        dataset[device_id].fm_pred.data() + dataset[device_id].N_real * 2,
        dataset[device_id].virial.data());
      GPU_CHECK_KERNEL
      #ifdef NEP_SPIN_PROFILE
      ms_post += timer.stop();
      #endif
    }

#ifdef NEP_SPIN_PROFILE
    // force kernels time accumulates both radial+angular(+ZBL) implicitly; approximate with neighbor omitted time
    // We report per-device timings once per call.
    printf("[NEP_Spin Profile][dev %d] neighbor=%.3f ms, desc=%.3f ms, ann=%.3f ms, post=%.3f ms\n",
      device_id, ms_neighbor, ms_desc, ms_ann, ms_post);
#endif
  }
}
