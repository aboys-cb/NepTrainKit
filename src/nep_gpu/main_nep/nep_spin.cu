/*
  NEP_Spin minimal skeleton; kernels and logic appended in follow-up patches.
*/

#include "dataset.cuh"
#include "nep_spin.cuh"
#include "mic.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_spin_utilities.cuh"
#include "utilities/kernel_timing.cuh"
#include <vector>
#include <cstdio>
#include <cmath>
 
 
 

// compute per-dimension scaling from descriptor min/max
static void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f; // a small number
  s_min[tid] = +1000000.0f; // a large number
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      if (q > s_max[tid]) {
        s_max[tid] = q;
      }
      if (q < s_min[tid]) {
        s_min[tid] = q;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) {
        s_max[tid] = s_max[tid + offset];
      }
      if (s_min[tid] > s_min[tid + offset]) {
        s_min[tid] = s_min[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    float range = s_max[0] - s_min[0];
    if (!(range > 1.0e-12f)) {
      // Descriptor range is (near-)zero; avoid leaving scaler at its huge initialization value.
      range = 1.0f;
    }
    g_q_scaler[bid] = min(g_q_scaler[bid], 1.0f / range);
  }
}


 


/*
// ----------------------------------------------------------------------
// Spherical spin-feature kernels (implementation)
// ----------------------------------------------------------------------
  Spherical spin-feature kernels for NEP-Spin model
  Implements: D^ex, D^dmi, D^ani, D^sia and their derivatives (force/mforce)

  Optimizations applied:
    1. Kernel Splitting: Separated Ex, DMI, ANI, SIA, and Onsite calculations into distinct kernels to reduce register pressure.
    2. Read-Only Cache: Used __ldg() for read-only global memory accesses (e.g., annmb.c, neighbors).
    3. Template Specialization: Hard-coded loops for low Chebyshev order values (KMAX_PAIR).
*/


// ===================================================================
// Kernel 1: Exchange Interaction (Ex)
// ===================================================================
template <int KMAX_PAIR>
__global__ void find_descriptors_radial_spin_spherical_ex_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    float* __restrict__ g_descriptors,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 >= N) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);
    const int nspin = paramb.n_max_radial + 1;

    // KMAX clamping
    const int kmax_ex = (paramb.spin_kmax_ex < -1) ? -1 : (paramb.spin_kmax_ex > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_ex);
    if (kmax_ex < 0) return; // No Ex interaction

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    // Initialize accumulators
    extern __shared__ float s_mem[];
    float* q_ex = s_mem + threadIdx.x * (KMAX_PAIR + 1) * MAX_NUM_N;
    for (int i = 0; i < (KMAX_PAIR + 1) * MAX_NUM_N; ++i) q_ex[i] = 0.0f;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                const float sdot = nep_spin_dot3(si, sj);
                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                float Tk[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_Tk<KMAX_PAIR>(c, kmax_ex, Tk);

                const float phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);
                float ex_invariant[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_ex_invariant<KMAX_PAIR>(phi, Tk, kmax_ex, ex_invariant);

                // Accumulate
                for (int n = 0; n <= paramb.n_max_radial; ++n) {
                    float gn_ex[KMAX_PAIR + 1] = {0.0f};
                    for (int kb = 0; kb <= bs_loc; ++kb) {
                        float fn_val = fn12[kb];
                        #pragma unroll
                        for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                            if (kk <= kmax_ex) {
                                int c_idx = nep_spin_get_c_index(
                                  mode,
                                  paramb.c_spin_offset,
                                  paramb.c_spin_block_stride,
                                  paramb.num_types_sq,
                                  paramb.num_types,
                                  bs,
                                  kk,
                                  n,
                                  kb,
                                  t1,
                                  t2);
                                gn_ex[kk] += fn_val * __ldg(&annmb.c[c_idx]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                        if (kk <= kmax_ex) {
                            q_ex[kk * MAX_NUM_N + n] += gn_ex[kk] * ex_invariant[kk];
                        }
                    }
                }
            }
        }
    }

    // Write Ex descriptors
    for (int kk = 0; kk <= kmax_ex; ++kk) {
        int off = spin_offset + kk * nspin;
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
            g_descriptors[n1 + (off + n) * N] = q_ex[kk * MAX_NUM_N + n];
        }
    }
}

// ===================================================================
// Kernel 2: DMI Interaction
// ===================================================================
template <int KMAX_PAIR>
__global__ void find_descriptors_radial_spin_spherical_dmi_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    float* __restrict__ g_descriptors,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 >= N) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);
    const int nspin = paramb.n_max_radial + 1;

    const int kmax_dmi = (paramb.spin_kmax_dmi < -1) ? -1 : (paramb.spin_kmax_dmi > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_dmi);
    if (kmax_dmi < 0) return;

    // Offsets for DMI (after Ex)
    const int kmax_ex = (paramb.spin_kmax_ex < -1) ? -1 : (paramb.spin_kmax_ex > 8 ? 8 : paramb.spin_kmax_ex);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_block0 = ex_blocks;

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    extern __shared__ float s_mem[];
    float* q_dmi = s_mem + threadIdx.x * (KMAX_PAIR + 1) * MAX_NUM_N;
    for (int i = 0; i < (KMAX_PAIR + 1) * MAX_NUM_N; ++i) q_dmi[i] = 0.0f;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                const float sdot = nep_spin_dot3(si, sj);
                float sixsj[3];
                nep_spin_cross3(si, sj, sixsj);
                const float dmi_val = nep_spin_dot3(sixsj, rhat);

                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                float Tk[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_Tk<KMAX_PAIR>(c, kmax_dmi, Tk);

                for (int n = 0; n <= paramb.n_max_radial; ++n) {
                    float gn_dmi[KMAX_PAIR + 1] = {0.0f};
                    for (int kb = 0; kb <= bs_loc; ++kb) {
                        float fn_val = fn12[kb];
                        #pragma unroll
                        for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                            if (kk <= kmax_dmi) {
                                int c_idx_dmi = nep_spin_get_c_index(
                                  mode,
                                  paramb.c_spin_offset,
                                  paramb.c_spin_block_stride,
                                  paramb.num_types_sq,
                                  paramb.num_types,
                                  bs,
                                  dmi_block0 + kk,
                                  n,
                                  kb,
                                  t1,
                                  t2);
                                gn_dmi[kk] += fn_val * __ldg(&annmb.c[c_idx_dmi]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                        if (kk <= kmax_dmi) {
                            q_dmi[kk * MAX_NUM_N + n] += gn_dmi[kk] * (dmi_val * Tk[kk]);
                        }
                    }
                }
            }
        }
    }

    // Write DMI descriptors
    for (int kk = 0; kk <= kmax_dmi; ++kk) {
        int off = spin_offset + (dmi_block0 + kk) * nspin;
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
            g_descriptors[n1 + (off + n) * N] = q_dmi[kk * MAX_NUM_N + n];
        }
    }
}

// ===================================================================
// Kernel 3: ANI Interaction
// ===================================================================
template <int KMAX_PAIR>
__global__ void find_descriptors_radial_spin_spherical_ani_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    float* __restrict__ g_descriptors,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 >= N) return;

    const int kmax_ani = (paramb.spin_kmax_ani < -1) ? -1 : (paramb.spin_kmax_ani > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_ani);
    if (kmax_ani < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);
    const int nspin = paramb.n_max_radial + 1;

    // Calculate offsets
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_block0 = ex_blocks + dmi_blocks;

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    extern __shared__ float s_mem[];
    float* q_ani = s_mem + threadIdx.x * (KMAX_PAIR + 1) * MAX_NUM_N;
    for (int i = 0; i < (KMAX_PAIR + 1) * MAX_NUM_N; ++i) q_ani[i] = 0.0f;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
                float sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
                float ani_scalar = si_r * sj_r;

                const float sdot = nep_spin_dot3(si, sj);
                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                float Tk[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_Tk<KMAX_PAIR>(c, kmax_ani, Tk);

                for (int n = 0; n <= paramb.n_max_radial; ++n) {
                    float gn_ani[KMAX_PAIR + 1] = {0.0f};
                    for (int kb = 0; kb <= bs_loc; ++kb) {
                        float fn_val = fn12[kb];
                        #pragma unroll
                        for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                            if (kk <= kmax_ani) {
                                int c_idx = nep_spin_get_c_index(
                                  mode,
                                  paramb.c_spin_offset,
                                  paramb.c_spin_block_stride,
                                  paramb.num_types_sq,
                                  paramb.num_types,
                                  bs,
                                  ani_block0 + kk,
                                  n,
                                  kb,
                                  t1,
                                  t2);
                                gn_ani[kk] += fn_val * __ldg(&annmb.c[c_idx]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                        if (kk <= kmax_ani) {
                            q_ani[kk * MAX_NUM_N + n] += gn_ani[kk] * (ani_scalar * Tk[kk]);
                        }
                    }
                }
            }
        }
    }

    // Write ANI descriptors
    for (int kk = 0; kk <= kmax_ani; ++kk) {
        int off = spin_offset + (ani_block0 + kk) * nspin;
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
            g_descriptors[n1 + (off + n) * N] = q_ani[kk * MAX_NUM_N + n];
        }
    }
}

// ===================================================================
// Kernel 4: SIA Interaction
// ===================================================================
template <int KMAX_PAIR>
__global__ void find_descriptors_radial_spin_spherical_sia_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    float* __restrict__ g_descriptors,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 >= N) return;

    const int kmax_sia = (paramb.spin_kmax_sia < -1) ? -1 : (paramb.spin_kmax_sia > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_sia);
    if (kmax_sia < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);
    const int nspin = paramb.n_max_radial + 1;

    // Calculate offsets
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int kmax_ani = clamp(paramb.spin_kmax_ani);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_blocks = (kmax_ani >= 0) ? (kmax_ani + 1) : 0;
    const int sia_block0 = ex_blocks + dmi_blocks + ani_blocks;

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    extern __shared__ float s_mem[];
    float* q_sia = s_mem + threadIdx.x * (KMAX_PAIR + 1) * MAX_NUM_N;
    for (int i = 0; i < (KMAX_PAIR + 1) * MAX_NUM_N; ++i) q_sia[i] = 0.0f;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

            float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
            float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
            float sia_scalar = si_r * si_r;

            float Tk[KMAX_PAIR + 1] = {0.0f};
            Tk[0] = 1.0f;
            if (neighbor_has_spin) {
                const float sdot = nep_spin_dot3(si, sj);
                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));
                nep_spin_fill_Tk<KMAX_PAIR>(c, kmax_sia, Tk);
            }

            for (int n = 0; n <= paramb.n_max_radial; ++n) {
                float gn_sia[KMAX_PAIR + 1] = {0.0f};
                for (int kb = 0; kb <= bs_loc; ++kb) {
                    float fn_val = fn12[kb];
                    #pragma unroll
                    for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                        if (kk <= kmax_sia) {
                            int c_idx = nep_spin_get_c_index(
                              mode,
                              paramb.c_spin_offset,
                              paramb.c_spin_block_stride,
                              paramb.num_types_sq,
                              paramb.num_types,
                              bs,
                              sia_block0 + kk,
                              n,
                              kb,
                              t1,
                              t2);
                            gn_sia[kk] += fn_val * __ldg(&annmb.c[c_idx]);
                        }
                    }
                }
                #pragma unroll
                for (int kk = 0; kk <= KMAX_PAIR; ++kk) {
                    if (kk <= kmax_sia) {
                        if (kk == 0 || neighbor_has_spin) {
                            q_sia[kk * MAX_NUM_N + n] += gn_sia[kk] * (sia_scalar * Tk[kk]);
                        }
                    }
                }
            }
        }
    }

    // Write SIA descriptors
    for (int kk = 0; kk <= kmax_sia; ++kk) {
        int off = spin_offset + (sia_block0 + kk) * nspin;
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
            g_descriptors[n1 + (off + n) * N] = q_sia[kk * MAX_NUM_N + n];
        }
    }
}

// ===================================================================
// Kernel 5: Onsite Interaction
// ===================================================================
__global__ void find_descriptors_radial_spin_spherical_onsite_k(
    const int N,
    const NEP_Spin::ParaMB paramb,
    const float* __restrict__ g_spin,
    float* __restrict__ g_descriptors,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 >= N) return;

    int spin_pmax = paramb.spin_pmax;
    if (spin_pmax <= 0) return;
    if (spin_pmax > 8) spin_pmax = 8;

    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int kmax_ani = clamp(paramb.spin_kmax_ani);
    const int kmax_sia = clamp(paramb.spin_kmax_sia);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_blocks = (kmax_ani >= 0) ? (kmax_ani + 1) : 0;
    const int sia_blocks = (kmax_sia >= 0) ? (kmax_sia + 1) : 0;
    const int pair_blocks = ex_blocks + dmi_blocks + ani_blocks + sia_blocks;

    const int nspin = paramb.n_max_radial + 1;
    const int onsite_offset = spin_offset + nspin * pair_blocks;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    if (si2 <= kSpinZeroEpsSph) {
        for (int p = 1; p <= spin_pmax; ++p) {
            g_descriptors[n1 + (onsite_offset + (p - 1)) * N] = 0.0f;
        }
        return;
    }

    int basis_mode = paramb.spin_onsite_basis_mode;
    if (basis_mode == 0) {
        float m2 = si2;
        float m2p = m2;
        for (int p = 1; p <= spin_pmax; ++p) {
            g_descriptors[n1 + (onsite_offset + (p - 1)) * N] = m2p;
            m2p *= m2;
        }
    } else {
        float y = si2;
        float yref = paramb.spin_mref;
        if (basis_mode == 2) {
            y = sqrtf(si2);
        } else {
            yref = paramb.spin_mref * paramb.spin_mref;
        }
        if (yref <= 0.0f) yref = 1.0f;
        float x = (y - yref) / (y + yref + 1.0e-12f);
        x = fminf(1.0f, fmaxf(-1.0f, x));

        float Tp[9];
        Tp[0] = 1.0f;
        if (spin_pmax >= 1) Tp[1] = x;
        for (int p = 2; p <= spin_pmax; ++p) {
            Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
        }
        for (int p = 1; p <= spin_pmax; ++p) {
            g_descriptors[n1 + (onsite_offset + (p - 1)) * N] = Tp[p];
        }
    }
}

// ===================================================================
// Launch Function
// ===================================================================
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
    cudaStream_t stream)
{
    auto clamp_kmax = [](int kmax) {
        if (kmax < -1) return -1;
        if (kmax > 8) return 8;
        return kmax;
    };

    auto get_smem = [&](int k) {
        return block.x * (k + 1) * MAX_NUM_N * sizeof(float);
    };
    
    // Dispatch Ex
    int kmax_ex = clamp_kmax(paramb.spin_kmax_ex);
    if (kmax_ex >= 0) {
        switch (kmax_ex) {
            case 0: find_descriptors_radial_spin_spherical_ex_k<0><<<grid, block, get_smem(0), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 1: find_descriptors_radial_spin_spherical_ex_k<1><<<grid, block, get_smem(1), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 2: find_descriptors_radial_spin_spherical_ex_k<2><<<grid, block, get_smem(2), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 3: find_descriptors_radial_spin_spherical_ex_k<3><<<grid, block, get_smem(3), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 4: find_descriptors_radial_spin_spherical_ex_k<4><<<grid, block, get_smem(4), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            default: find_descriptors_radial_spin_spherical_ex_k<8><<<grid, block, get_smem(8), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
        }
    }

    // Dispatch DMI
    int kmax_dmi = clamp_kmax(paramb.spin_kmax_dmi);
    if (kmax_dmi >= 0) {
        switch (kmax_dmi) {
            case 0: find_descriptors_radial_spin_spherical_dmi_k<0><<<grid, block, get_smem(0), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 1: find_descriptors_radial_spin_spherical_dmi_k<1><<<grid, block, get_smem(1), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 2: find_descriptors_radial_spin_spherical_dmi_k<2><<<grid, block, get_smem(2), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 3: find_descriptors_radial_spin_spherical_dmi_k<3><<<grid, block, get_smem(3), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 4: find_descriptors_radial_spin_spherical_dmi_k<4><<<grid, block, get_smem(4), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            default: find_descriptors_radial_spin_spherical_dmi_k<8><<<grid, block, get_smem(8), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
        }
    }

    // Dispatch ANI
    int kmax_ani = clamp_kmax(paramb.spin_kmax_ani);
    if (kmax_ani >= 0) {
        switch (kmax_ani) {
            case 0: find_descriptors_radial_spin_spherical_ani_k<0><<<grid, block, get_smem(0), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 1: find_descriptors_radial_spin_spherical_ani_k<1><<<grid, block, get_smem(1), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 2: find_descriptors_radial_spin_spherical_ani_k<2><<<grid, block, get_smem(2), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 3: find_descriptors_radial_spin_spherical_ani_k<3><<<grid, block, get_smem(3), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 4: find_descriptors_radial_spin_spherical_ani_k<4><<<grid, block, get_smem(4), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            default: find_descriptors_radial_spin_spherical_ani_k<8><<<grid, block, get_smem(8), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
        }
    }

    // Dispatch SIA
    int kmax_sia = clamp_kmax(paramb.spin_kmax_sia);
    if (kmax_sia >= 0) {
        switch (kmax_sia) {
            case 0: find_descriptors_radial_spin_spherical_sia_k<0><<<grid, block, get_smem(0), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 1: find_descriptors_radial_spin_spherical_sia_k<1><<<grid, block, get_smem(1), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 2: find_descriptors_radial_spin_spherical_sia_k<2><<<grid, block, get_smem(2), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 3: find_descriptors_radial_spin_spherical_sia_k<3><<<grid, block, get_smem(3), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            case 4: find_descriptors_radial_spin_spherical_sia_k<4><<<grid, block, get_smem(4), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
            default: find_descriptors_radial_spin_spherical_sia_k<8><<<grid, block, get_smem(8), stream>>>(N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_descriptors, spin_offset); break;
        }
    }

    // Dispatch Onsite
    if (paramb.spin_pmax > 0) {
        find_descriptors_radial_spin_spherical_onsite_k<<<grid, block, 0, stream>>>(N, paramb, g_spin, g_descriptors, spin_offset);
    }
}
// Force Kernels (Ex, DMI, ANI, SIA)
// ===================================================================

template <int KMAX_PAIR>
__global__ void find_force_radial_spin_spherical_ex_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_fx,
    float* __restrict__ g_fy,
    float* __restrict__ g_fz,
    float* __restrict__ g_virial,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_ex = (paramb.spin_kmax_ex < -1) ? -1 : (paramb.spin_kmax_ex > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_ex);
    
    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_ex >= 0) {
        for (int k = 0; k <= kmax_ex; ++k) {
            int off = spin_offset + k * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_ex < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float fx_n1 = 0.0f, fy_n1 = 0.0f, fz_n1 = 0.0f;
    float vir[6] = {0.0f};

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12, dfc12;
            find_fc_and_fcp(rc, rcinv, d12, fc12, dfc12);
            float fn12[MAX_NUM_N], dfn12[MAX_NUM_N];
            find_fn_and_fnp(bs_loc, rcinv, d12, fc12, dfc12, fn12, dfn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                float sdot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
                float sj_norm = sqrtf(sj2);
                float denom = si_norm * sj_norm;
                float c = sdot / (denom + 1.0e-12f);
                c = fminf(1.0f, fmaxf(-1.0f, c));

                float Tk[KMAX_PAIR + 1];
                Tk[0] = 1.0f;
                if (KMAX_PAIR >= 1) Tk[1] = c;
                #pragma unroll
                for (int kk = 2; kk <= KMAX_PAIR; ++kk) {
                    if (kk <= kmax_ex) Tk[kk] = 2.0f * c * Tk[kk - 1] - Tk[kk - 2];
                }

                float phi = denom;
                if (paramb.spin_ex_phi_mode == 1) phi = si_norm;
                else if (paramb.spin_ex_phi_mode == 2) phi = sj_norm;
                else if (paramb.spin_ex_phi_mode == 3) phi = 1.0f;

                float force_mag = 0.0f;
                for (int n = 0; n < nspin; ++n) {
                    for (int k = 0; k <= kmax_ex; ++k) {
                         float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                         float dC_dr = 0.0f;
                         for (int kb = 0; kb <= bs_loc; ++kb) {
                             int c_idx = nep_spin_get_c_index(
                               mode,
                               paramb.c_spin_offset,
                               paramb.c_spin_block_stride,
                               paramb.num_types_sq,
                               paramb.num_types,
                               bs,
                               k,
                               n,
                               kb,
                               t1,
                               t2);
                             dC_dr += dfn12[kb] * __ldg(&annmb.c[c_idx]);
                         }
                         force_mag += fp_val * dC_dr * (phi * Tk[k]); 
                    }
                }
                
                float fvec[3] = {force_mag * rhat[0], force_mag * rhat[1], force_mag * rhat[2]};
                
                fx_n1 += fvec[0];
                fy_n1 += fvec[1];
                fz_n1 += fvec[2];
                
                atomicAdd(&g_fx[n2], -fvec[0]);
                atomicAdd(&g_fy[n2], -fvec[1]);
                atomicAdd(&g_fz[n2], -fvec[2]);
                
                vir[0] -= fvec[0] * r12[0];
                vir[1] -= fvec[1] * r12[1];
                vir[2] -= fvec[2] * r12[2];
                vir[3] -= r12[0] * fvec[1];
                vir[4] -= r12[1] * fvec[2];
                vir[5] -= r12[2] * fvec[0];
            }
        }
    }
    
    atomicAdd(&g_fx[n1], fx_n1);
    atomicAdd(&g_fy[n1], fy_n1);
    atomicAdd(&g_fz[n1], fz_n1);
    
    for (int d = 0; d < 6; ++d) atomicAdd(&g_virial[n1 + d * N], vir[d]);
}

template <int KMAX_PAIR>
__global__ void find_force_radial_spin_spherical_dmi_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_fx,
    float* __restrict__ g_fy,
    float* __restrict__ g_fz,
    float* __restrict__ g_virial,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_dmi = (paramb.spin_kmax_dmi < -1) ? -1 : (paramb.spin_kmax_dmi > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_dmi);
    
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_block0 = ex_blocks;

    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_dmi >= 0) {
        for (int k = 0; k <= kmax_dmi; ++k) {
            int off = spin_offset + (dmi_block0 + k) * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_dmi < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float fx_n1 = 0.0f, fy_n1 = 0.0f, fz_n1 = 0.0f;
    float vir[6] = {0.0f};

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12, dfc12;
            find_fc_and_fcp(rc, rcinv, d12, fc12, dfc12);
            float fn12[MAX_NUM_N], dfn12[MAX_NUM_N];
            find_fn_and_fnp(bs_loc, rcinv, d12, fc12, dfc12, fn12, dfn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                float sdot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
                float sixsj[3] = {si[1]*sj[2] - si[2]*sj[1], si[2]*sj[0] - si[0]*sj[2], si[0]*sj[1] - si[1]*sj[0]};
                float dmi_val = sixsj[0]*rhat[0] + sixsj[1]*rhat[1] + sixsj[2]*rhat[2];

                float sj_norm = sqrtf(sj2);
                float c = sdot / (si_norm * sj_norm + 1.0e-12f);
                c = fminf(1.0f, fmaxf(-1.0f, c));

                float Tk[KMAX_PAIR + 1];
                Tk[0] = 1.0f;
                if (KMAX_PAIR >= 1) Tk[1] = c;
                #pragma unroll
                for (int kk = 2; kk <= KMAX_PAIR; ++kk) {
                    if (kk <= kmax_dmi) Tk[kk] = 2.0f * c * Tk[kk - 1] - Tk[kk - 2];
                }

                float fvec[3] = {0.0f};
                
                for (int n = 0; n < nspin; ++n) {
                    for (int k = 0; k <= kmax_dmi; ++k) {
                         float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                         
                         float dC_dr = 0.0f;
                         float C_val = 0.0f;
                         for (int kb = 0; kb <= bs_loc; ++kb) {
                             int c_idx = nep_spin_get_c_index(
                               mode,
                               paramb.c_spin_offset,
                               paramb.c_spin_block_stride,
                               paramb.num_types_sq,
                               paramb.num_types,
                               bs,
                               dmi_block0 + k,
                               n,
                               kb,
                               t1,
                               t2);
                             float coeff = __ldg(&annmb.c[c_idx]);
                             dC_dr += dfn12[kb] * coeff;
                             C_val += fn12[kb] * coeff;
                         }
                         
                         float term1 = (dC_dr * dmi_val - C_val * dmi_val / d12) * Tk[k];
                         float term2 = (C_val * Tk[k] / d12);
                         
                         fvec[0] += fp_val * (term1 * rhat[0] + term2 * sixsj[0]);
                         fvec[1] += fp_val * (term1 * rhat[1] + term2 * sixsj[1]);
                         fvec[2] += fp_val * (term1 * rhat[2] + term2 * sixsj[2]);
                    }
                }
                
                fx_n1 += fvec[0];
                fy_n1 += fvec[1];
                fz_n1 += fvec[2];
                
                atomicAdd(&g_fx[n2], -fvec[0]);
                atomicAdd(&g_fy[n2], -fvec[1]);
                atomicAdd(&g_fz[n2], -fvec[2]);
                
                vir[0] -= fvec[0] * r12[0];
                vir[1] -= fvec[1] * r12[1];
                vir[2] -= fvec[2] * r12[2];
                vir[3] -= r12[0] * fvec[1];
                vir[4] -= r12[1] * fvec[2];
                vir[5] -= r12[2] * fvec[0];
            }
        }
    }
    
    atomicAdd(&g_fx[n1], fx_n1);
    atomicAdd(&g_fy[n1], fy_n1);
    atomicAdd(&g_fz[n1], fz_n1);
    
    for (int d = 0; d < 6; ++d) atomicAdd(&g_virial[n1 + d * N], vir[d]);
}

template <int KMAX_PAIR>
__global__ void find_force_radial_spin_spherical_ani_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_fx,
    float* __restrict__ g_fy,
    float* __restrict__ g_fz,
    float* __restrict__ g_virial,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_ani = (paramb.spin_kmax_ani < -1) ? -1 : (paramb.spin_kmax_ani > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_ani);
    
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_block0 = ex_blocks + dmi_blocks;

    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_ani >= 0) {
        for (int k = 0; k <= kmax_ani; ++k) {
            int off = spin_offset + (ani_block0 + k) * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_ani < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float fx_n1 = 0.0f, fy_n1 = 0.0f, fz_n1 = 0.0f;
    float vir[6] = {0.0f};

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12, dfc12;
            find_fc_and_fcp(rc, rcinv, d12, fc12, dfc12);
            float fn12[MAX_NUM_N], dfn12[MAX_NUM_N];
            find_fn_and_fnp(bs_loc, rcinv, d12, fc12, dfc12, fn12, dfn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                float sdot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
                float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
                float sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
                float ani_scalar = si_r * sj_r;

                float sj_norm = sqrtf(sj2);
                float c = sdot / (si_norm * sj_norm + 1.0e-12f);
                c = fminf(1.0f, fmaxf(-1.0f, c));

                float Tk[KMAX_PAIR + 1];
                Tk[0] = 1.0f;
                if (KMAX_PAIR >= 1) Tk[1] = c;
                #pragma unroll
                for (int kk = 2; kk <= KMAX_PAIR; ++kk) {
                    if (kk <= kmax_ani) Tk[kk] = 2.0f * c * Tk[kk - 1] - Tk[kk - 2];
                }

                float fvec[3] = {0.0f};
                
                for (int n = 0; n < nspin; ++n) {
                    for (int k = 0; k <= kmax_ani; ++k) {
                         float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                         
                         float dC_dr = 0.0f;
                         float C_val = 0.0f;
                         for (int kb = 0; kb <= bs_loc; ++kb) {
                             int c_idx = nep_spin_get_c_index(
                               mode,
                               paramb.c_spin_offset,
                               paramb.c_spin_block_stride,
                               paramb.num_types_sq,
                               paramb.num_types,
                               bs,
                               ani_block0 + k,
                               n,
                               kb,
                               t1,
                               t2);
                             float coeff = __ldg(&annmb.c[c_idx]);
                             dC_dr += dfn12[kb] * coeff;
                             C_val += fn12[kb] * coeff;
                         }
                         
                         float term1 = (dC_dr * ani_scalar - 2.0f * C_val * ani_scalar / d12) * Tk[k];
                         float term2 = (C_val * Tk[k] / d12);
                         
                         fvec[0] += fp_val * (term1 * rhat[0] + term2 * (sj_r * si[0] + si_r * sj[0]));
                         fvec[1] += fp_val * (term1 * rhat[1] + term2 * (sj_r * si[1] + si_r * sj[1]));
                         fvec[2] += fp_val * (term1 * rhat[2] + term2 * (sj_r * si[2] + si_r * sj[2]));
                    }
                }
                
                fx_n1 += fvec[0];
                fy_n1 += fvec[1];
                fz_n1 += fvec[2];
                
                atomicAdd(&g_fx[n2], -fvec[0]);
                atomicAdd(&g_fy[n2], -fvec[1]);
                atomicAdd(&g_fz[n2], -fvec[2]);
                
                vir[0] -= fvec[0] * r12[0];
                vir[1] -= fvec[1] * r12[1];
                vir[2] -= fvec[2] * r12[2];
                vir[3] -= r12[0] * fvec[1];
                vir[4] -= r12[1] * fvec[2];
                vir[5] -= r12[2] * fvec[0];
            }
        }
    }
    
    atomicAdd(&g_fx[n1], fx_n1);
    atomicAdd(&g_fy[n1], fy_n1);
    atomicAdd(&g_fz[n1], fz_n1);
    
    for (int d = 0; d < 6; ++d) atomicAdd(&g_virial[n1 + d * N], vir[d]);
}

template <int KMAX_PAIR>
__global__ void find_force_radial_spin_spherical_sia_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_fx,
    float* __restrict__ g_fy,
    float* __restrict__ g_fz,
    float* __restrict__ g_virial,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_sia = (paramb.spin_kmax_sia < -1) ? -1 : (paramb.spin_kmax_sia > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_sia);
    
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int kmax_ani = clamp(paramb.spin_kmax_ani);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_blocks = (kmax_ani >= 0) ? (kmax_ani + 1) : 0;
    const int sia_block0 = ex_blocks + dmi_blocks + ani_blocks;

    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_sia >= 0) {
        for (int k = 0; k <= kmax_sia; ++k) {
            int off = spin_offset + (sia_block0 + k) * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_sia < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float fx_n1 = 0.0f, fy_n1 = 0.0f, fz_n1 = 0.0f;
    float vir[6] = {0.0f};

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12, dfc12;
            find_fc_and_fcp(rc, rcinv, d12, fc12, dfc12);
            float fn12[MAX_NUM_N], dfn12[MAX_NUM_N];
            find_fn_and_fnp(bs_loc, rcinv, d12, fc12, dfc12, fn12, dfn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

            float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
            float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
            float sia_scalar = si_r * si_r;

            float Tk[KMAX_PAIR + 1];
            Tk[0] = 1.0f;
            if (neighbor_has_spin) {
                float sdot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
                float sj_norm = sqrtf(sj2);
                float c = sdot / (si_norm * sj_norm + 1.0e-12f);
                c = fminf(1.0f, fmaxf(-1.0f, c));
                if (KMAX_PAIR >= 1) Tk[1] = c;
                #pragma unroll
                for (int kk = 2; kk <= KMAX_PAIR; ++kk) {
                    if (kk <= kmax_sia) Tk[kk] = 2.0f * c * Tk[kk - 1] - Tk[kk - 2];
                }
            } else {
                 for (int kk=1; kk<=KMAX_PAIR; ++kk) Tk[kk] = 0.0f;
            }

            float fvec[3] = {0.0f};
            
            for (int n = 0; n < nspin; ++n) {
                for (int k = 0; k <= kmax_sia; ++k) {
                     if (k > 0 && !neighbor_has_spin) continue;

                     float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                     
                     float dC_dr = 0.0f;
                     float C_val = 0.0f;
                     for (int kb = 0; kb <= bs_loc; ++kb) {
                         int c_idx = nep_spin_get_c_index(
                           mode,
                           paramb.c_spin_offset,
                           paramb.c_spin_block_stride,
                           paramb.num_types_sq,
                           paramb.num_types,
                           bs,
                           sia_block0 + k,
                           n,
                           kb,
                           t1,
                           t2);
                         float coeff = __ldg(&annmb.c[c_idx]);
                         dC_dr += dfn12[kb] * coeff;
                         C_val += fn12[kb] * coeff;
                     }
                     
                     float term1 = (dC_dr * sia_scalar - 2.0f * C_val * sia_scalar / d12) * Tk[k];
                     float term2 = (2.0f * C_val * Tk[k] * si_r / d12);
                     
                     fvec[0] += fp_val * (term1 * rhat[0] + term2 * si[0]);
                     fvec[1] += fp_val * (term1 * rhat[1] + term2 * si[1]);
                     fvec[2] += fp_val * (term1 * rhat[2] + term2 * si[2]);
                }
            }
            
            fx_n1 += fvec[0];
            fy_n1 += fvec[1];
            fz_n1 += fvec[2];
            
            atomicAdd(&g_fx[n2], -fvec[0]);
            atomicAdd(&g_fy[n2], -fvec[1]);
            atomicAdd(&g_fz[n2], -fvec[2]);
            
            vir[0] -= fvec[0] * r12[0];
            vir[1] -= fvec[1] * r12[1];
            vir[2] -= fvec[2] * r12[2];
            vir[3] -= r12[0] * fvec[1];
            vir[4] -= r12[1] * fvec[2];
            vir[5] -= r12[2] * fvec[0];
        }
    }
    
    atomicAdd(&g_fx[n1], fx_n1);
    atomicAdd(&g_fy[n1], fy_n1);
    atomicAdd(&g_fz[n1], fz_n1);
    
    for (int d = 0; d < 6; ++d) atomicAdd(&g_virial[n1 + d * N], vir[d]);
}

// ===================================================================
// Magnetic Force Kernels (Ex, DMI, ANI, SIA)
// ===================================================================

template <int KMAX_PAIR>
__global__ void find_mforce_radial_spin_spherical_ex_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_mx,
    float* __restrict__ g_my,
    float* __restrict__ g_mz,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_ex = (paramb.spin_kmax_ex < -1) ? -1 : (paramb.spin_kmax_ex > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_ex);
    
    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_ex >= 0) {
        for (int k = 0; k <= kmax_ex; ++k) {
            int off = spin_offset + k * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_ex < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float mx_n1 = 0.0f, my_n1 = 0.0f, mz_n1 = 0.0f;
    const float msign = paramb.mforce_sign;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                const float sdot = nep_spin_dot3(si, sj);
                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                float Tk[KMAX_PAIR + 1] = {0.0f};
                float Uk[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_Tk_and_dTk<KMAX_PAIR>(c, kmax_ex, Tk, Uk);

                float mvec_i[3] = {0.0f};
                float mvec_j[3] = {0.0f};

                for (int n = 0; n < nspin; ++n) {
                    for (int k = 0; k <= kmax_ex; ++k) {
                         float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                         float C_val = 0.0f;
                         for (int kb = 0; kb <= bs_loc; ++kb) {
                             int c_idx = nep_spin_get_c_index(
                               mode,
                               paramb.c_spin_offset,
                               paramb.c_spin_block_stride,
                               paramb.num_types_sq,
                               paramb.num_types,
                               bs,
                               k,
                               n,
                               kb,
                               t1,
                               t2);
                             C_val += fn12[kb] * __ldg(&annmb.c[c_idx]);
                         }
                         
                         float term_i_si, term_i_sj, term_j_si, term_j_sj;
                         float inv_si = 1.0f / (si_norm + 1e-12f);
                         float inv_sj = 1.0f / (sj_norm + 1e-12f);

                         if (paramb.spin_ex_phi_mode == 0) {
                             float ratio = sj_norm * inv_si;
                             term_i_si = (Tk[k] - c * Uk[k]) * ratio;
                             term_i_sj = Uk[k];
                             float ratio_j = si_norm * inv_sj;
                             term_j_sj = (Tk[k] - c * Uk[k]) * ratio_j;
                             term_j_si = Uk[k];
                         } else if (paramb.spin_ex_phi_mode == 1) {
                             term_i_si = (Tk[k] - c * Uk[k]) * inv_si;
                             term_i_sj = Uk[k] * inv_sj;
                             term_j_si = Uk[k] * inv_sj;
                             term_j_sj = -c * Uk[k] * si_norm * inv_sj * inv_sj;
                         } else if (paramb.spin_ex_phi_mode == 2) {
                             term_i_si = -c * Uk[k] * sj_norm * inv_si * inv_si;
                             term_i_sj = Uk[k] * inv_si;
                             term_j_sj = (Tk[k] - c * Uk[k]) * inv_sj;
                             term_j_si = Uk[k] * inv_si;
                         } else {
                             float inv_denom = 1.0f / (denom + 1e-12f);
                             float ratio_i = sj_norm * inv_si;
                             term_i_si = -c * Uk[k] * ratio_i * inv_denom;
                             term_i_sj = Uk[k] * inv_denom;
                             float ratio_j = si_norm * inv_sj;
                             term_j_sj = -c * Uk[k] * ratio_j * inv_denom;
                             term_j_si = Uk[k] * inv_denom;
                         }
                         
                         float pre = fp_val * C_val;
                         mvec_i[0] += pre * (term_i_si * si[0] + term_i_sj * sj[0]);
                         mvec_i[1] += pre * (term_i_si * si[1] + term_i_sj * sj[1]);
                         mvec_i[2] += pre * (term_i_si * si[2] + term_i_sj * sj[2]);
                         
                         mvec_j[0] += pre * (term_j_si * si[0] + term_j_sj * sj[0]);
                         mvec_j[1] += pre * (term_j_si * si[1] + term_j_sj * sj[1]);
                         mvec_j[2] += pre * (term_j_si * si[2] + term_j_sj * sj[2]);
                    }
                }
                
                mx_n1 += msign * mvec_i[0];
                my_n1 += msign * mvec_i[1];
                mz_n1 += msign * mvec_i[2];
                
                atomicAdd(&g_mx[n2], msign * mvec_j[0]);
                atomicAdd(&g_my[n2], msign * mvec_j[1]);
                atomicAdd(&g_mz[n2], msign * mvec_j[2]);
            }
        }
    }
    
    atomicAdd(&g_mx[n1], mx_n1);
    atomicAdd(&g_my[n1], my_n1);
    atomicAdd(&g_mz[n1], mz_n1);
}



template <int KMAX_PAIR>
__global__ void find_mforce_radial_spin_spherical_dmi_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_mx,
    float* __restrict__ g_my,
    float* __restrict__ g_mz,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_dmi = (paramb.spin_kmax_dmi < -1) ? -1 : (paramb.spin_kmax_dmi > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_dmi);
    
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_block0 = ex_blocks;

    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_dmi >= 0) {
        for (int k = 0; k <= kmax_dmi; ++k) {
            int off = spin_offset + (dmi_block0 + k) * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_dmi < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float mx_n1 = 0.0f, my_n1 = 0.0f, mz_n1 = 0.0f;
    const float msign = paramb.mforce_sign;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                const float sdot = nep_spin_dot3(si, sj);
                float sixsj[3];
                nep_spin_cross3(si, sj, sixsj);
                const float dmi_val = nep_spin_dot3(sixsj, rhat);

                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                float Tk[KMAX_PAIR + 1] = {0.0f};
                float Uk[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_Tk_and_dTk<KMAX_PAIR>(c, kmax_dmi, Tk, Uk);

                float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
                float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
                nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);

                // d(dmi_val)/dsi = sj x rhat
                float dDMI_dsi[3] = {
                    sj[1]*rhat[2] - sj[2]*rhat[1],
                    sj[2]*rhat[0] - sj[0]*rhat[2],
                    sj[0]*rhat[1] - sj[1]*rhat[0]
                };
                // d(dmi_val)/dsj = rhat x si
                float dDMI_dsj[3] = {
                    rhat[1]*si[2] - rhat[2]*si[1],
                    rhat[2]*si[0] - rhat[0]*si[2],
                    rhat[0]*si[1] - rhat[1]*si[0]
                };

                float mvec_i[3] = {0.0f};
                float mvec_j[3] = {0.0f};

                for (int n = 0; n < nspin; ++n) {
                    for (int k = 0; k <= kmax_dmi; ++k) {
                         float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                         float C_val = 0.0f;
                         for (int kb = 0; kb <= bs_loc; ++kb) {
                             int c_idx = nep_spin_get_c_index(
                               mode,
                               paramb.c_spin_offset,
                               paramb.c_spin_block_stride,
                               paramb.num_types_sq,
                               paramb.num_types,
                               bs,
                               dmi_block0 + k,
                               n,
                               kb,
                               t1,
                               t2);
                             C_val += fn12[kb] * __ldg(&annmb.c[c_idx]);
                         }
                         
                         float term1_i = C_val * Tk[k];
                         float term2_i = C_val * dmi_val * Uk[k];
                          
                         for(int d=0; d<3; ++d) {
                             mvec_i[d] += fp_val * (term1_i * dDMI_dsi[d] + term2_i * dc_dsi[d]);
                             mvec_j[d] += fp_val * (term1_i * dDMI_dsj[d] + term2_i * dc_dsj[d]);
                         }
                    }
                }
                
                mx_n1 += msign * mvec_i[0];
                my_n1 += msign * mvec_i[1];
                mz_n1 += msign * mvec_i[2];
                
                atomicAdd(&g_mx[n2], msign * mvec_j[0]);
                atomicAdd(&g_my[n2], msign * mvec_j[1]);
                atomicAdd(&g_mz[n2], msign * mvec_j[2]);
            }
        }
    }
    
    atomicAdd(&g_mx[n1], mx_n1);
    atomicAdd(&g_my[n1], my_n1);
    atomicAdd(&g_mz[n1], mz_n1);
}

template <int KMAX_PAIR>
__global__ void find_mforce_radial_spin_spherical_ani_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_mx,
    float* __restrict__ g_my,
    float* __restrict__ g_mz,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_ani = (paramb.spin_kmax_ani < -1) ? -1 : (paramb.spin_kmax_ani > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_ani);
    
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_block0 = ex_blocks + dmi_blocks;

    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_ani >= 0) {
        for (int k = 0; k <= kmax_ani; ++k) {
            int off = spin_offset + (ani_block0 + k) * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_ani < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float mx_n1 = 0.0f, my_n1 = 0.0f, mz_n1 = 0.0f;
    const float msign = paramb.mforce_sign;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            
            if (sj2 > kSpinZeroEpsSph) {
                float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
                const float sdot = nep_spin_dot3(si, sj);
                const float si_r = nep_spin_dot3(si, rhat);
                const float sj_r = nep_spin_dot3(sj, rhat);
                float ani_scalar = si_r * sj_r;

                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                float Tk[KMAX_PAIR + 1] = {0.0f};
                float Uk[KMAX_PAIR + 1] = {0.0f};
                nep_spin_fill_Tk_and_dTk<KMAX_PAIR>(c, kmax_ani, Tk, Uk);

                float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
                float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
                nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);

                float mvec_i[3] = {0.0f};
                float mvec_j[3] = {0.0f};

                for (int n = 0; n < nspin; ++n) {
                    for (int k = 0; k <= kmax_ani; ++k) {
                         float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                         float C_val = 0.0f;
                         for (int kb = 0; kb <= bs_loc; ++kb) {
                             int c_idx = nep_spin_get_c_index(
                               mode,
                               paramb.c_spin_offset,
                               paramb.c_spin_block_stride,
                               paramb.num_types_sq,
                               paramb.num_types,
                               bs,
                               ani_block0 + k,
                               n,
                               kb,
                               t1,
                               t2);
                             C_val += fn12[kb] * __ldg(&annmb.c[c_idx]);
                         }
                         
                         float term1_i = C_val * ani_scalar * Uk[k];
                         float term2_i = C_val * Tk[k];
                          
                         for(int d=0; d<3; ++d) {
                             mvec_i[d] += fp_val * (term1_i * dc_dsi[d] + term2_i * rhat[d] * sj_r);
                             mvec_j[d] += fp_val * (term1_i * dc_dsj[d] + term2_i * rhat[d] * si_r);
                         }
                    }
                }
                
                mx_n1 += msign * mvec_i[0];
                my_n1 += msign * mvec_i[1];
                mz_n1 += msign * mvec_i[2];
                
                atomicAdd(&g_mx[n2], msign * mvec_j[0]);
                atomicAdd(&g_my[n2], msign * mvec_j[1]);
                atomicAdd(&g_mz[n2], msign * mvec_j[2]);
            }
        }
    }
    
    atomicAdd(&g_mx[n1], mx_n1);
    atomicAdd(&g_my[n1], my_n1);
    atomicAdd(&g_mz[n1], mz_n1);
}

template <int KMAX_PAIR>
__global__ void find_mforce_radial_spin_spherical_sia_k(
    const int N,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    const NEP_Spin::ParaMB paramb,
    const NEP_Spin::ANN annmb,
    const int* __restrict__ g_type,
    const float* __restrict__ g_x12,
    const float* __restrict__ g_y12,
    const float* __restrict__ g_z12,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_mx,
    float* __restrict__ g_my,
    float* __restrict__ g_mz,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int nspin = paramb.n_max_radial + 1;
    const int kmax_sia = (paramb.spin_kmax_sia < -1) ? -1 : (paramb.spin_kmax_sia > KMAX_PAIR ? KMAX_PAIR : paramb.spin_kmax_sia);
    
    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int kmax_ani = clamp(paramb.spin_kmax_ani);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_blocks = (kmax_ani >= 0) ? (kmax_ani + 1) : 0;
    const int sia_block0 = ex_blocks + dmi_blocks + ani_blocks;

    extern __shared__ float s_mem[];
    float* s_Fp = s_mem; 
    
    if (n1 < N && kmax_sia >= 0) {
        for (int k = 0; k <= kmax_sia; ++k) {
            int off = spin_offset + (sia_block0 + k) * nspin;
            for (int n = 0; n < nspin; ++n) {
                s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n] = g_Fp[n1 + (off + n) * N];
            }
        }
    }
    __syncthreads();

    if (n1 >= N || kmax_sia < 0) return;

    const int neighbor_number = __ldg(&g_NN[n1]);
    const int t1 = __ldg(&g_type[n1]);

    const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);

    const int bs = paramb.basis_size_radial;
    int bs_loc = (bs >= MAX_NUM_N) ? MAX_NUM_N - 1 : bs;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    
    float mx_n1 = 0.0f, my_n1 = 0.0f, mz_n1 = 0.0f;
    const float msign = paramb.mforce_sign;

    if (si2 > kSpinZeroEpsSph) {
        float si_norm = sqrtf(si2);
        
        for (int i1 = 0; i1 < neighbor_number; ++i1) {
            const int index = n1 + i1 * N;
            const int n2 = __ldg(&g_NL[index]);
            const int t2 = __ldg(&g_type[n2]);
            
            float r12[3] = {__ldg(&g_x12[index]), __ldg(&g_y12[index]), __ldg(&g_z12[index])};
            float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            if (d12 <= 0.0f) continue;

            float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rcinv = 1.0f / rc;
            
            float fc12;
            find_fc(rc, rcinv, d12, fc12);
            float fn12[MAX_NUM_N];
            find_fn(bs_loc, rcinv, d12, fc12, fn12);

            float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
            float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
            bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

            float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};
            float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
            float sia_scalar = si_r * si_r;

            float Tk[KMAX_PAIR + 1] = {0.0f};
            float Uk[KMAX_PAIR + 1] = {0.0f};
            float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
            float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
            Tk[0] = 1.0f;

            if (neighbor_has_spin) {
                const float sdot = nep_spin_dot3(si, sj);
                const float sj_norm = sqrtf(sj2);
                const float denom = si_norm * sj_norm;
                const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

                nep_spin_fill_Tk_and_dTk<KMAX_PAIR>(c, kmax_sia, Tk, Uk);
                nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
            }

            float mvec_i[3] = {0.0f};
            float mvec_j[3] = {0.0f};

            for (int n = 0; n < nspin; ++n) {
                for (int k = 0; k <= kmax_sia; ++k) {
                     if (k > 0 && !neighbor_has_spin) continue;

                     float fp_val = s_Fp[threadIdx.x * (KMAX_PAIR + 1) * nspin + k * nspin + n];
                     float C_val = 0.0f;
                     for (int kb = 0; kb <= bs_loc; ++kb) {
                         int c_idx = nep_spin_get_c_index(
                           mode,
                           paramb.c_spin_offset,
                           paramb.c_spin_block_stride,
                           paramb.num_types_sq,
                           paramb.num_types,
                           bs,
                           sia_block0 + k,
                           n,
                           kb,
                           t1,
                           t2);
                         C_val += fn12[kb] * __ldg(&annmb.c[c_idx]);
                     }
                     
                     float term1_i = C_val * sia_scalar * Uk[k];
                     float term2_i = C_val * Tk[k] * 2.0f * si_r; // d(sia_scalar)/dsi = 2 * si_r * rhat
                     
                     for(int d=0; d<3; ++d) {
                         mvec_i[d] += fp_val * (term1_i * dc_dsi[d] + term2_i * rhat[d]);
                         mvec_j[d] += fp_val * (term1_i * dc_dsj[d]);
                     }
                }
            }
            
            mx_n1 += msign * mvec_i[0];
            my_n1 += msign * mvec_i[1];
            mz_n1 += msign * mvec_i[2];
            
            atomicAdd(&g_mx[n2], msign * mvec_j[0]);
            atomicAdd(&g_my[n2], msign * mvec_j[1]);
            atomicAdd(&g_mz[n2], msign * mvec_j[2]);
        }
    }
    
    atomicAdd(&g_mx[n1], mx_n1);
    atomicAdd(&g_my[n1], my_n1);
    atomicAdd(&g_mz[n1], mz_n1);
}


// ===================================================================
// Onsite mforce kernel (spin_pmax terms)
// ===================================================================
__global__ void find_mforce_radial_spin_spherical_onsite_k(
    const int N,
    const NEP_Spin::ParaMB paramb,
    const float* __restrict__ g_spin,
    const float* __restrict__ g_Fp,
    float* __restrict__ g_mx,
    float* __restrict__ g_my,
    float* __restrict__ g_mz,
    const int spin_offset)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 >= N) return;

    int spin_pmax = paramb.spin_pmax;
    if (spin_pmax <= 0) return;
    if (spin_pmax > 8) spin_pmax = 8;

    auto clamp = [](int k) { return (k < -1) ? -1 : (k > 8 ? 8 : k); };
    const int kmax_ex = clamp(paramb.spin_kmax_ex);
    const int kmax_dmi = clamp(paramb.spin_kmax_dmi);
    const int kmax_ani = clamp(paramb.spin_kmax_ani);
    const int kmax_sia = clamp(paramb.spin_kmax_sia);
    const int ex_blocks = (kmax_ex >= 0) ? (kmax_ex + 1) : 0;
    const int dmi_blocks = (kmax_dmi >= 0) ? (kmax_dmi + 1) : 0;
    const int ani_blocks = (kmax_ani >= 0) ? (kmax_ani + 1) : 0;
    const int sia_blocks = (kmax_sia >= 0) ? (kmax_sia + 1) : 0;
    const int pair_blocks = ex_blocks + dmi_blocks + ani_blocks + sia_blocks;

    const int nspin = paramb.n_max_radial + 1;
    const int onsite_offset = spin_offset + nspin * pair_blocks;

    float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
    const float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) return;

    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    const float msign = paramb.mforce_sign;
    const int basis_mode = paramb.spin_onsite_basis_mode;

    if (basis_mode == 0) {
        const float m2 = si2;
        float m2pow = 1.0f; // m2^(p-1)
        for (int p = 1; p <= spin_pmax; ++p) {
            const float Fp_p = __ldg(&g_Fp[n1 + (onsite_offset + (p - 1)) * N]);
            const float coeff = msign * Fp_p * (2.0f * p) * m2pow;
            mx += coeff * si[0];
            my += coeff * si[1];
            mz += coeff * si[2];
            m2pow *= m2;
        }
    } else {
        float y = si2;
        float yref = paramb.spin_mref * paramb.spin_mref;
        const float si_norm = sqrtf(si2);
        const float inv_si_norm = 1.0f / (si_norm + 1.0e-12f);
        if (basis_mode == 2) {
            y = si_norm;
            yref = paramb.spin_mref;
        }
        if (yref <= 0.0f) yref = 1.0f;

        const float denom = y + yref;
        const float inv_denom = 1.0f / (denom + 1.0e-12f);
        float x = (y - yref) * inv_denom;
        x = fminf(1.0f, fmaxf(-1.0f, x));
        const float dx_dy = (2.0f * yref) * inv_denom * inv_denom;

        float Tp[9] = {0.0f};
        float dTp[9] = {0.0f};
        Tp[0] = 1.0f;
        dTp[0] = 0.0f;
        if (spin_pmax >= 1) {
            Tp[1] = x;
            dTp[1] = 1.0f;
        }
        for (int p = 2; p <= spin_pmax; ++p) {
            Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
            dTp[p] = 2.0f * Tp[p - 1] + 2.0f * x * dTp[p - 1] - dTp[p - 2];
        }

        float dy_dsi[3];
        if (basis_mode == 2) {
            dy_dsi[0] = inv_si_norm * si[0];
            dy_dsi[1] = inv_si_norm * si[1];
            dy_dsi[2] = inv_si_norm * si[2];
        } else {
            dy_dsi[0] = 2.0f * si[0];
            dy_dsi[1] = 2.0f * si[1];
            dy_dsi[2] = 2.0f * si[2];
        }

        for (int p = 1; p <= spin_pmax; ++p) {
            const float Fp_p = __ldg(&g_Fp[n1 + (onsite_offset + (p - 1)) * N]);
            const float coeff = msign * Fp_p * dTp[p] * dx_dy;
            mx += coeff * dy_dsi[0];
            my += coeff * dy_dsi[1];
            mz += coeff * dy_dsi[2];
        }
    }

    atomicAdd(&g_mx[n1], mx);
    atomicAdd(&g_my[n1], my);
    atomicAdd(&g_mz[n1], mz);
}

// ===================================================================
// Launch Functions
// ===================================================================

void launch_find_force_radial_spin_spherical_full(
    const dim3 grid, const dim3 block, const int N, const int* g_NN, const int* g_NL,
    const NEP_Spin::ParaMB paramb, const NEP_Spin::ANN annmb, const int* g_type,
    const float* g_x12, const float* g_y12, const float* g_z12, const float* g_spin,
    const float* g_Fp, float* g_fx, float* g_fy, float* g_fz, float* g_virial,
    int spin_offset, cudaStream_t stream)
{
    auto clamp_kmax = [](int kmax) { return (kmax < -1) ? -1 : ((kmax > 8) ? 8 : kmax); };
    auto get_smem = [&](int k) { return block.x * (k + 1) * (paramb.n_max_radial + 1) * sizeof(float); };

    int kmax_ex = clamp_kmax(paramb.spin_kmax_ex);
    if (kmax_ex >= 0) {
        find_force_radial_spin_spherical_ex_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_fx, g_fy, g_fz, g_virial, spin_offset);
    }
    int kmax_dmi = clamp_kmax(paramb.spin_kmax_dmi);
    if (kmax_dmi >= 0) {
        find_force_radial_spin_spherical_dmi_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_fx, g_fy, g_fz, g_virial, spin_offset);
    }
    int kmax_ani = clamp_kmax(paramb.spin_kmax_ani);
    if (kmax_ani >= 0) {
        find_force_radial_spin_spherical_ani_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_fx, g_fy, g_fz, g_virial, spin_offset);
    }
    int kmax_sia = clamp_kmax(paramb.spin_kmax_sia);
    if (kmax_sia >= 0) {
        find_force_radial_spin_spherical_sia_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_fx, g_fy, g_fz, g_virial, spin_offset);
    }
}

void launch_find_mforce_radial_spin_spherical_full(
    const dim3 grid, const dim3 block, const int N, const int* g_NN, const int* g_NL,
    const NEP_Spin::ParaMB paramb, const NEP_Spin::ANN annmb, const int* g_type,
    const float* g_x12, const float* g_y12, const float* g_z12, const float* g_spin,
    const float* g_Fp, float* g_mx, float* g_my, float* g_mz,
    int spin_offset, cudaStream_t stream)
{
    auto clamp_kmax = [](int kmax) { return (kmax < -1) ? -1 : ((kmax > 8) ? 8 : kmax); };
    auto get_smem = [&](int k) { return block.x * (k + 1) * (paramb.n_max_radial + 1) * sizeof(float); };

    int kmax_ex = clamp_kmax(paramb.spin_kmax_ex);
    if (kmax_ex >= 0) {
        find_mforce_radial_spin_spherical_ex_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_mx, g_my, g_mz, spin_offset);
    }
    int kmax_dmi = clamp_kmax(paramb.spin_kmax_dmi);
    if (kmax_dmi >= 0) {
        find_mforce_radial_spin_spherical_dmi_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_mx, g_my, g_mz, spin_offset);
    }
    int kmax_ani = clamp_kmax(paramb.spin_kmax_ani);
    if (kmax_ani >= 0) {
        find_mforce_radial_spin_spherical_ani_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_mx, g_my, g_mz, spin_offset);
    }
    int kmax_sia = clamp_kmax(paramb.spin_kmax_sia);
    if (kmax_sia >= 0) {
        find_mforce_radial_spin_spherical_sia_k<8><<<grid, block, get_smem(8), stream>>>(
            N, g_NN, g_NL, paramb, annmb, g_type, g_x12, g_y12, g_z12, g_spin, g_Fp, g_mx, g_my, g_mz, spin_offset);
    }

    if (paramb.spin_pmax > 0) {
        find_mforce_radial_spin_spherical_onsite_k<<<grid, block, 0, stream>>>(
            N, paramb, g_spin, g_Fp, g_mx, g_my, g_mz, spin_offset);
    }
}

// Note: we intentionally only expose the "full" launchers. For exchange-only models,
// set `spin_kmax_dmi/spin_kmax_ani/spin_kmax_sia = -1` and the full launchers will only
// launch the exchange kernels.



// baseline radial descriptors
static __global__ void find_descriptors_radial_spinbase(
  const int N,
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    int bs = paramb.basis_size_radial;
    if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
    if (neighbor_number > 0) for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(bs, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= bs; ++k) {
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

// baseline angular descriptors
static __global__ void find_descriptors_angular_spinbase(
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
  float* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};
    int bs = paramb.basis_size_angular;
    if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      if (neighbor_number > 0) for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(bs, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= bs; ++k) {
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

// Build neighbor list (radial + angular), adapted from NEP
static __global__ void gpu_find_neighbor_list_spin(
  const NEP_Spin::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  const int cap_radial,
  const int cap_angular,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
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
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) { continue; }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            float d2 = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_r = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rc_a = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
            if (d2 < rc_r * rc_r) {
              if (count_radial < cap_radial) {
                NL_radial[count_radial * N + n1] = n2;
                x12_radial[count_radial * N + n1] = x12;
                y12_radial[count_radial * N + n1] = y12;
                z12_radial[count_radial * N + n1] = z12;
                count_radial++;
              }
            }
            if (d2 < rc_a * rc_a) {
              if (count_angular < cap_angular) {
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
    // Clamp neighbor counts to allocated per-atom capacity to avoid OOB reads later
    // when looping over neighbors using g_NN as the bound.
    NN_radial[n1] = min(count_radial, cap_radial);
    NN_angular[n1] = min(count_angular, cap_angular);
  }
}

NEP_Spin::NEP_Spin(
  Parameters& para,
  int N,
  int N_times_max_NN_radial,
  int N_times_max_NN_angular,
  int version,
  int deviceCount)
{
  // allocate arrays per device using neighbor list capacity from dataset
  paramb.version = version;
  paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  paramb.num_types = para.num_types;
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial[t] = para.rc_radial[t];
    paramb.rc_angular[t] = para.rc_angular[t];
  }
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.spin_kmax_ex = para.spin_kmax_ex;
  paramb.spin_kmax_dmi = para.spin_kmax_dmi;
  paramb.spin_kmax_ani = para.spin_kmax_ani;
  paramb.spin_kmax_sia = para.spin_kmax_sia;
  paramb.spin_pmax = para.spin_pmax;
  paramb.spin_ex_phi_mode = para.spin_ex_phi_mode;
  paramb.spin_onsite_basis_mode = para.spin_onsite_basis_mode;
  paramb.spin_mref = para.spin_mref;
  paramb.num_L = paramb.L_max;
  if (para.L_max_4body == 2) { paramb.num_L += 1; }
  if (para.L_max_5body == 1) { paramb.num_L += 1; }
  paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;
  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
  paramb.num_c_radial = paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

  // spin_blocks and c_spin layout (for spin_mode 2/3)
  auto blocks_from_kmax = [](int kmax) { return (kmax >= 0) ? (kmax + 1) : 0; };
  int ex_blocks = blocks_from_kmax(para.spin_kmax_ex);
  int dmi_blocks = blocks_from_kmax(para.spin_kmax_dmi);
  int ani_blocks = blocks_from_kmax(para.spin_kmax_ani);
  int sia_blocks = blocks_from_kmax(para.spin_kmax_sia);
  paramb.spin_blocks = ex_blocks + dmi_blocks + ani_blocks + sia_blocks;

  int nspin = para.n_max_radial + 1;
  paramb.c_spin_block_stride =
    paramb.num_types_sq * nspin * (para.basis_size_radial + 1);

  int num_c_angular =
    paramb.num_types_sq * (para.n_max_angular + 1) * (para.basis_size_angular + 1);

  if (para.spin_mode == 2) {
    if (paramb.spin_blocks > 0) {
      paramb.num_c_spin = paramb.c_spin_block_stride;
      paramb.c_spin_offset = paramb.num_c_radial + num_c_angular;
    } else {
      paramb.num_c_spin = 0;
      paramb.c_spin_offset = 0;
    }
  } else if (para.spin_mode == 3) {
    if (paramb.spin_blocks > 0) {
      paramb.num_c_spin = paramb.c_spin_block_stride * paramb.spin_blocks;
      paramb.c_spin_offset = paramb.num_c_radial + num_c_angular;
    } else {
      paramb.num_c_spin = 0;
      paramb.c_spin_offset = 0;
    }
  } else {
    paramb.num_c_spin = 0;
    paramb.c_spin_offset = 0;
  }
  for (int n = 0; n < (int)para.atomic_numbers.size(); ++n) {
    paramb.atomic_numbers[n] = para.atomic_numbers[n] - 1;
  }

  for (int device_id = 0; device_id < deviceCount; device_id++) {
    gpuSetDevice(device_id);
    annmb[device_id].dim = para.dim;
    annmb[device_id].num_neurons1 = para.num_neurons1;
    annmb[device_id].num_para = para.number_of_variables;

    nep_data[device_id].NN_radial.resize(N);
    nep_data[device_id].NN_angular.resize(N);
    nep_data[device_id].NL_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].NL_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].x12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].y12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].z12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].x12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].y12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].z12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    nep_data[device_id].Fp.resize(N * annmb[device_id].dim);
    nep_data[device_id].sum_fxyz.resize(
      N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
    nep_data[device_id].parameters.resize(annmb[device_id].num_para);
  }
}

void NEP_Spin::update_potential(Parameters& para, float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { pointer -= (ann.dim + 2) * ann.num_neurons1; }
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
}

 
// zero spin descriptor block when spin data is absent
static __global__ void
zero_spin_descriptor_block(const int N, const int spin_dim, float* g_descriptors, int spin_offset)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    for (int d = 0; d < spin_dim; ++d) {
      g_descriptors[n1 + (spin_offset + d) * N] = 0.0f;
    }
  }
}

// Apply ANN
static __global__ void apply_ann_spin(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];
    float q[MAX_DIM];
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    float F = 0.0f;
    float Fp[MAX_DIM];
    for (int d = 0; d < annmb.dim; ++d) {
      Fp[d] = 0.0f;
    }
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type],
      annmb.b0[type],
      annmb.w1[type],
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

// zero mechanical forces
static __global__ void zero_force_spin(
  const int N, float* g_fx, float* g_fy, float* g_fz, float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    // zero all 6 virial components for this atom
    for (int d = 0; d < 6; ++d) {
      g_virial[n1 + d * N] = 0.0f;
    }
  }
}

// baseline radial forces
static __global__ void find_force_radial_spinbase(
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
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;
    float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
    if (neighbor_number > 0) for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      int bs = paramb.basis_size_radial;
      if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) { f12[d] += tmp12 * r12[d]; }
      }

      fi_acc_x += f12[0];
      fi_acc_y += f12[1];
      fi_acc_z += f12[2];
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
    if (neighbor_number > 0) {
      atomicAdd(&g_fx[n1], fi_acc_x);
      atomicAdd(&g_fy[n1], fi_acc_y);
      atomicAdd(&g_fz[n1], fi_acc_z);
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

// baseline angular forces
static __global__ void find_force_angular_spinbase(
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
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;
    float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
    float Fp_loc[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz_loc[NUM_OF_ABC * MAX_NUM_N] = {0.0f};
    int dim_ang = paramb.dim_angular;
    if (dim_ang > MAX_DIM_ANGULAR) dim_ang = MAX_DIM_ANGULAR;
    for (int d = 0; d < dim_ang; ++d) {
      Fp_loc[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    int abc_count = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    if (abc_count > NUM_OF_ABC) abc_count = NUM_OF_ABC;
    int nmax = paramb.n_max_angular + 1;
    if (nmax > MAX_NUM_N) nmax = MAX_NUM_N;
    for (int n = 0; n < nmax; ++n) {
      for (int abc = 0; abc < abc_count; ++abc) {
        sum_fxyz_loc[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int neighbor_number = g_NN[n1];
    if (neighbor_number > 0) for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};

      int bs = paramb.basis_size_angular;
      if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n < nmax; ++n) {
        float gn12 = 0.0f, gnp12 = 0.0f;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.num_L, n, nmax, d12, r12, gn12, gnp12, Fp_loc, sum_fxyz_loc, f12);
      }

      fi_acc_x += f12[0];
      fi_acc_y += f12[1];
      fi_acc_z += f12[2];
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
    if (neighbor_number > 0) {
      atomicAdd(&g_fx[n1], fi_acc_x);
      atomicAdd(&g_fy[n1], fi_acc_y);
      atomicAdd(&g_fz[n1], fi_acc_z);
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

 

// zero magnetic force
static __global__ void zero_mforce_spin(const int N, float* g_mx, float* g_my, float* g_mz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) { g_mx[n1] = 0.0f; g_my[n1] = 0.0f; g_mz[n1] = 0.0f; }
}

static KernelTiming g_kernel_timing_spin[16];
static long long g_kernel_timing_spin_call = 0;

 
void NEP_Spin::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  bool calculate_neighbor,
  int device_in_this_iter)
{
  const long long call_id = g_kernel_timing_spin_call++;
  const bool do_profile =
    (para.kernel_timing != 0) &&
    (call_id >= (long long)para.kernel_timing_skip) &&
    (((call_id - (long long)para.kernel_timing_skip) % (long long)para.kernel_timing_every) == 0);

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    update_potential(para, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    const int block_size = 32;
    const int grid_size = (dataset[device_id].N - 1) / block_size + 1;
    const int block_size_spin = 32;
    const int grid_size_spin = (dataset[device_id].N - 1) / block_size_spin + 1;
    KernelTiming& kt = g_kernel_timing_spin[device_id];

    if (calculate_neighbor) {
      int cap_radial_per_atom = (int)(nep_data[device_id].NL_radial.size() / dataset[device_id].N);
      int cap_angular_per_atom = (int)(nep_data[device_id].NL_angular.size() / dataset[device_id].N);
      if (cap_radial_per_atom < 1) cap_radial_per_atom = 1;
      if (cap_angular_per_atom < 1) cap_angular_per_atom = 1;
      if (do_profile) {
        int tok = kt.begin("gpu_find_neighbor_list_spin");
        gpu_find_neighbor_list_spin<<<dataset[device_id].Nc, 256>>>(
          paramb,
          dataset[device_id].N,
          dataset[device_id].Na.data(),
          dataset[device_id].Na_sum.data(),
          dataset[device_id].type.data(),
          dataset[device_id].box.data(),
          dataset[device_id].box_original.data(),
          dataset[device_id].num_cell.data(),
          dataset[device_id].r.data(),
          dataset[device_id].r.data() + dataset[device_id].N,
          dataset[device_id].r.data() + dataset[device_id].N * 2,
          cap_radial_per_atom,
          cap_angular_per_atom,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data());
        kt.end(tok);
      } else {
        gpu_find_neighbor_list_spin<<<dataset[device_id].Nc, 256>>>(
          paramb,
          dataset[device_id].N,
          dataset[device_id].Na.data(),
          dataset[device_id].Na_sum.data(),
          dataset[device_id].type.data(),
          dataset[device_id].box.data(),
          dataset[device_id].box_original.data(),
          dataset[device_id].num_cell.data(),
          dataset[device_id].r.data(),
          dataset[device_id].r.data() + dataset[device_id].N,
          dataset[device_id].r.data() + dataset[device_id].N * 2,
          cap_radial_per_atom,
          cap_angular_per_atom,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data());
      }
      GPU_CHECK_KERNEL
 
    }

    // descriptors baseline
    if (do_profile) {
      int tok = kt.begin("find_descriptors_radial_spinbase");
      find_descriptors_radial_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data());
      kt.end(tok);
    } else {
      find_descriptors_radial_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("find_descriptors_angular_spinbase");
      find_descriptors_angular_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
      kt.end(tok);
    } else {
      find_descriptors_angular_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
    }
    GPU_CHECK_KERNEL

    // Spin descriptor block starts after baseline [radial | angular].
    // Base dimension: dim_base = (n_max_r + 1) + dim_angular.
    // Added spin dims = (pair_blocks * (n_max_r + 1) + spin_pmax), where pair_blocks is derived
    // from per-block kmax (ex/dmi/ani/sia).
    const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
    bool has_spin = (dataset[device_id].spin.size() == dataset[device_id].N * 3);
    // Compute blocks for spin features and assert the spin block fits into dim
    int nspin = paramb.n_max_radial + 1;

    auto clamp_kmax = [](int kmax) {
      if (kmax < -1) return -1;
      if (kmax > 8) return 8;
      return kmax;
    };
    auto clamp_pmax = [](int pmax) {
      if (pmax < 0) return 0;
      if (pmax > 8) return 8;
      return pmax;
    };
    auto blocks_from_kmax = [](int kmax) { return (kmax >= 0) ? (kmax + 1) : 0; };

    int kmax_ex = clamp_kmax(paramb.spin_kmax_ex);
    int kmax_dmi = clamp_kmax(paramb.spin_kmax_dmi);
    int kmax_ani = clamp_kmax(paramb.spin_kmax_ani);
    int kmax_sia = clamp_kmax(paramb.spin_kmax_sia);
    int spin_pmax = clamp_pmax(paramb.spin_pmax);

    int ex_blocks = blocks_from_kmax(kmax_ex);
    int dmi_blocks = blocks_from_kmax(kmax_dmi);
    int ani_blocks = blocks_from_kmax(kmax_ani);
    int sia_blocks = blocks_from_kmax(kmax_sia);
    int pair_blocks = ex_blocks + dmi_blocks + ani_blocks + sia_blocks;

    int spin_dim = nspin * pair_blocks + spin_pmax;
    int end_idx = spin_offset + spin_dim;
    if (end_idx > annmb[device_id].dim) {
      printf(
        "[spin][error] device=%d spin block OOB: end=%d > dim=%d (offset=%d nspin=%d pair_blocks=%d spin_pmax=%d)\n",
        device_id,
        end_idx,
        annmb[device_id].dim,
        spin_offset,
        nspin,
        pair_blocks,
        spin_pmax);
      fflush(stdout);
      exit(1);
    }
    // printf("[DEBUG][NEP_Spin] device=%d has_spin=%d spin_size=%zu N=%d dim=%d\n",
    //        device_id,
    //        has_spin ? 1 : 0,
    //        dataset[device_id].spin.size(),
    //        dataset[device_id].N,
    //        annmb[device_id].dim);
    if (has_spin) {
      // spherical invariants
      if (do_profile) {
        int tok = kt.begin("find_descriptors_radial_spin_spherical_full");
        launch_find_descriptors_radial_spin_spherical_full(
          grid_size_spin,
          block_size_spin,
          dataset[device_id].N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data(),
          spin_offset);
        kt.end(tok);
      } else {
        launch_find_descriptors_radial_spin_spherical_full(
          grid_size_spin,
          block_size_spin,
          dataset[device_id].N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data(),
          spin_offset);
      }
      GPU_CHECK_KERNEL
    } else {
      // zero the spin descriptor block to keep ANN input consistent
      if (do_profile) {
        int tok = kt.begin("zero_spin_descriptor_block");
        zero_spin_descriptor_block<<<grid_size_spin, block_size_spin>>>(
          dataset[device_id].N, spin_dim, nep_data[device_id].descriptors.data(), spin_offset);
        kt.end(tok);
      } else {
        zero_spin_descriptor_block<<<grid_size_spin, block_size_spin>>>(
          dataset[device_id].N, spin_dim, nep_data[device_id].descriptors.data(), spin_offset);
      }
      GPU_CHECK_KERNEL
    }

    // In prediction mode, optionally dump descriptors like NEP/NEP_Charge
    if (para.prediction == 1 && para.output_descriptor >= 1) {
      FILE* fid_descriptor = my_fopen("descriptor.out", "a");
      std::vector<float> descriptor_cpu(nep_data[device_id].descriptors.size());
      nep_data[device_id].descriptors.copy_to_host(descriptor_cpu.data());
      for (int nc = 0; nc < dataset[device_id].Nc; ++nc) {
        float q_structure[MAX_DIM] = {0.0f};
        for (int na = 0; na < dataset[device_id].Na_cpu[nc]; ++na) {
          int n = dataset[device_id].Na_sum_cpu[nc] + na;
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            float q = descriptor_cpu[n + d * dataset[device_id].N] * para.q_scaler_cpu[d];
            q_structure[d] += q;
            if (para.output_descriptor == 2) {
              fprintf(fid_descriptor, "%g ", q);
            }
          }
          if (para.output_descriptor == 2) {
            fprintf(fid_descriptor, "\n");
          }
        }
        if (para.output_descriptor == 1) {
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            fprintf(fid_descriptor, "%g ", q_structure[d] / dataset[device_id].Na_cpu[nc]);
          }
        }
        if (para.output_descriptor == 1) {
          fprintf(fid_descriptor, "\n");
        }
      }
      fclose(fid_descriptor);
    }

    if (calculate_q_scaler) {
      if (do_profile) {
        int tok = kt.begin("find_max_min");
        find_max_min<<<annmb[device_id].dim, 1024>>>(
          dataset[device_id].N,
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data());
        kt.end(tok);
      } else {
        find_max_min<<<annmb[device_id].dim, 1024>>>(
          dataset[device_id].N,
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data());
      }
      GPU_CHECK_KERNEL
    }

    if (do_profile) {
      int tok = kt.begin("zero_force_spin");
      zero_force_spin<<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
      kt.end(tok);
    } else {
      zero_force_spin<<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("apply_ann_spin");
      apply_ann_spin<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].energy.data(),
        nep_data[device_id].Fp.data());
      kt.end(tok);
    } else {
      apply_ann_spin<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].energy.data(),
        nep_data[device_id].Fp.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("find_force_radial_spinbase");
      find_force_radial_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
      kt.end(tok);
    } else {
      find_force_radial_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("find_force_angular_spinbase");
      find_force_angular_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].Fp.data(),
        nep_data[device_id].sum_fxyz.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
      kt.end(tok);
    } else {
      find_force_angular_spinbase<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].Fp.data(),
        nep_data[device_id].sum_fxyz.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
    }
    GPU_CHECK_KERNEL

    // spin-radial mechanical force contribution (only if spin exists)
    // Spin-dependent mechanical forces (w.r.t. positions) contribute to virial;
    // these kernels update the stress.
    if (has_spin) {
      auto launch_spin_force_radial_kernel = [&]() {
        // Maintain only one implementation. When non-ex blocks are disabled (kmax_* < 0),
        // the full launcher will only launch the exchange kernel.
        launch_find_force_radial_spin_spherical_full(
          grid_size_spin,
          block_size_spin,
          dataset[device_id].N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + dataset[device_id].N,
          dataset[device_id].force.data() + dataset[device_id].N * 2,
          dataset[device_id].virial.data(),
           spin_offset);
      };
      if (do_profile) {
        int tok = kt.begin("find_force_radial_spin_spherical_full");
        launch_spin_force_radial_kernel();
        kt.end(tok);
      } else {
        launch_spin_force_radial_kernel();
      }
      GPU_CHECK_KERNEL
    }

    // compute magnetic force
    // Magnetic force is -âˆ‚E/âˆ‚s and must NOT contribute to virial.
    // Only write the three mforce components.
    const bool need_mforce = has_spin;
    if (need_mforce && dataset[device_id].mforce.size() != (size_t)dataset[device_id].N * 3) {
      dataset[device_id].mforce.resize(dataset[device_id].N * 3);
    }
    if (need_mforce) {
      if (do_profile) {
        int tok0 = kt.begin("zero_mforce_spin");
        zero_mforce_spin<<<grid_size_spin, block_size_spin>>>(
          dataset[device_id].N,
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + dataset[device_id].N,
          dataset[device_id].mforce.data() + dataset[device_id].N * 2);
        kt.end(tok0);
      } else {
        zero_mforce_spin<<<grid_size_spin, block_size_spin>>>(
          dataset[device_id].N,
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + dataset[device_id].N,
          dataset[device_id].mforce.data() + dataset[device_id].N * 2);
      }
      GPU_CHECK_KERNEL

      if (do_profile) {
        int tok1 = kt.begin("find_mforce_radial_spin_spherical_full");
        launch_find_mforce_radial_spin_spherical_full(
          grid_size_spin,
          block_size_spin,
          dataset[device_id].N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + dataset[device_id].N,
          dataset[device_id].mforce.data() + dataset[device_id].N * 2,
          spin_offset);
        kt.end(tok1);
      } else {
        launch_find_mforce_radial_spin_spherical_full(
          grid_size_spin,
          block_size_spin,
          dataset[device_id].N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + dataset[device_id].N,
          dataset[device_id].mforce.data() + dataset[device_id].N * 2,
          spin_offset);

      }
      GPU_CHECK_KERNEL
    }

    if (do_profile) {
      kt.flush();
    }
  }

  if (do_profile) {
    KernelTiming merged;
    for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
      merged.merge_from(g_kernel_timing_spin[device_id]);
    }
    printf("[kernel_timing] NEP_Spin::find_force call=%lld devices=%d\n", call_id, device_in_this_iter);
    merged.print_top("NEP_Spin GPU kernels", para.kernel_timing_topk);
    for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
      g_kernel_timing_spin[device_id].reset();
    }
  }
}
