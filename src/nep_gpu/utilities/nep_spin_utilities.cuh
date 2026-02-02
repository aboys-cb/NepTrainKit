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

// Spin-specific helpers shared by training-side (src/main_nep)
// and MD-side (src/force) NEP_Spin implementations.

// Allow limited host compilation of this header (e.g. IDE indexing),
// while keeping the CUDA attributes when compiled by nvcc.
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

// Small threshold on |s|^2 to treat a spin vector as effectively zero.
static constexpr float kSpinZeroEpsSph = 1.0e-8f;

// Helper for selecting which c coefficients to use for spin descriptors.
enum SpinCMode { SPIN_C_SHARED_LATTICE, SPIN_C_SINGLE, SPIN_C_PER_BLOCK };

static __host__ __device__ __forceinline__ SpinCMode nep_spin_get_c_mode(
  const int num_c_spin, const int c_spin_block_stride)
{
  if (num_c_spin == 0) return SPIN_C_SHARED_LATTICE; // spin_mode=1: reuse lattice c_radial
  if (num_c_spin == c_spin_block_stride) return SPIN_C_SINGLE; // spin_mode=2: one shared c_spin
  return SPIN_C_PER_BLOCK; // spin_mode=3: per-block c_spin^(b)
}

// Compute the index into ANN::c for a spin-radial coefficient c(block,n,k,t1,t2).
// The "block" is the concatenated spin-block index (ex/dmi/ani/sia), matching
// the descriptor layout used by both training and MD paths.
static __host__ __device__ __forceinline__ int nep_spin_get_c_index(
  const SpinCMode mode,
  const int c_spin_offset,
  const int c_spin_block_stride,
  const int num_types_sq,
  const int num_types,
  const int basis_size,
  const int block_idx,
  const int n,
  const int k,
  const int t1_loc,
  const int t2_loc)
{
  const int base = (n * (basis_size + 1) + k) * num_types_sq + t1_loc * num_types + t2_loc;
  if (mode == SPIN_C_SHARED_LATTICE) return base;
  if (mode == SPIN_C_SINGLE) return c_spin_offset + base;
  return c_spin_offset + block_idx * c_spin_block_stride + base;
}

// -----------------------------------------------------------------------------
// Shared spherical-spin math helpers (descriptor-side)
// -----------------------------------------------------------------------------

static __host__ __device__ __forceinline__ float nep_spin_clamp_unit(const float x)
{
  return (x > 1.0f) ? 1.0f : ((x < -1.0f) ? -1.0f : x);
}

static __host__ __device__ __forceinline__ float nep_spin_dot3(const float a[3], const float b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static __host__ __device__ __forceinline__ void nep_spin_cross3(
  const float a[3], const float b[3], float out[3])
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

// Compute Chebyshev T_k(c) for k=0..KMAX, using kmax as an enable mask.
// Values for k>kmax are set to 0 to avoid accidental use.
template <int KMAX>
static __host__ __device__ __forceinline__ void nep_spin_fill_Tk(
  const float c, const int kmax, float* Tk)
{
  Tk[0] = 1.0f;
  if (KMAX >= 1) {
    Tk[1] = (kmax >= 1) ? c : 0.0f;
  }
  #pragma unroll
  for (int kk = 2; kk <= KMAX; ++kk) {
    const float val = 2.0f * c * Tk[kk - 1] - Tk[kk - 2];
    Tk[kk] = (kk <= kmax) ? val : 0.0f;
  }
}

template <int KMAX>
static __host__ __device__ __forceinline__ void nep_spin_fill_Tk_and_dTk(
  const float c, const int kmax, float* Tk, float* dTk)
{
  Tk[0] = 1.0f;
  dTk[0] = 0.0f;
  if (KMAX >= 1) {
    Tk[1] = (kmax >= 1) ? c : 0.0f;
    dTk[1] = (kmax >= 1) ? 1.0f : 0.0f;
  }
  #pragma unroll
  for (int kk = 2; kk <= KMAX; ++kk) {
    if (kk <= kmax) {
      Tk[kk] = 2.0f * c * Tk[kk - 1] - Tk[kk - 2];
      dTk[kk] = 2.0f * Tk[kk - 1] + 2.0f * c * dTk[kk - 1] - dTk[kk - 2];
    } else {
      Tk[kk] = 0.0f;
      dTk[kk] = 0.0f;
    }
  }
}

static __host__ __device__ __forceinline__ void nep_spin_fill_dc_dsi_dsj(
  const float si[3],
  const float sj[3],
  const float sdot,
  const float si_norm,
  const float sj_norm,
  float dc_dsi[3],
  float dc_dsj[3])
{
  const float denom = si_norm * sj_norm;
  const float inv_denom = 1.0f / (denom + 1.0e-12f);
  const float inv_a2 = 1.0f / (si_norm * si_norm + 1.0e-12f);
  const float inv_b2 = 1.0f / (sj_norm * sj_norm + 1.0e-12f);
  dc_dsi[0] = inv_denom * (sj[0] - sdot * inv_a2 * si[0]);
  dc_dsi[1] = inv_denom * (sj[1] - sdot * inv_a2 * si[1]);
  dc_dsi[2] = inv_denom * (sj[2] - sdot * inv_a2 * si[2]);
  dc_dsj[0] = inv_denom * (si[0] - sdot * inv_b2 * sj[0]);
  dc_dsj[1] = inv_denom * (si[1] - sdot * inv_b2 * sj[1]);
  dc_dsj[2] = inv_denom * (si[2] - sdot * inv_b2 * sj[2]);
}

static __host__ __device__ __forceinline__ void nep_spin_ex_phi_and_grads(
  const int phi_mode,
  const float si[3],
  const float sj[3],
  const float si_norm,
  const float sj_norm,
  const float inv_si_norm,
  const float inv_sj_norm,
  float& phi,
  float dphi_dsi[3],
  float dphi_dsj[3])
{
  dphi_dsi[0] = dphi_dsi[1] = dphi_dsi[2] = 0.0f;
  dphi_dsj[0] = dphi_dsj[1] = dphi_dsj[2] = 0.0f;

  if (phi_mode == 0) {
    phi = si_norm * sj_norm;
    const float scale_i = sj_norm * inv_si_norm;
    const float scale_j = si_norm * inv_sj_norm;
    dphi_dsi[0] = scale_i * si[0];
    dphi_dsi[1] = scale_i * si[1];
    dphi_dsi[2] = scale_i * si[2];
    dphi_dsj[0] = scale_j * sj[0];
    dphi_dsj[1] = scale_j * sj[1];
    dphi_dsj[2] = scale_j * sj[2];
    return;
  }
  if (phi_mode == 1) {
    phi = si_norm;
    dphi_dsi[0] = inv_si_norm * si[0];
    dphi_dsi[1] = inv_si_norm * si[1];
    dphi_dsi[2] = inv_si_norm * si[2];
    return;
  }
  if (phi_mode == 2) {
    phi = sj_norm;
    dphi_dsj[0] = inv_sj_norm * sj[0];
    dphi_dsj[1] = inv_sj_norm * sj[1];
    dphi_dsj[2] = inv_sj_norm * sj[2];
    return;
  }
  phi = 1.0f; // phi_mode == 3
}

static __host__ __device__ __forceinline__ float nep_spin_ex_phi(
  const int phi_mode, const float si_norm, const float sj_norm, const float denom)
{
  if (phi_mode == 1) return si_norm;
  if (phi_mode == 2) return sj_norm;
  if (phi_mode == 3) return 1.0f;
  return denom; // phi_mode == 0
}

template <int KMAX>
static __host__ __device__ __forceinline__ void nep_spin_fill_ex_invariant(
  const float phi, const float* Tk, const int kmax_ex, float* ex_invariant)
{
  #pragma unroll
  for (int kk = 0; kk <= KMAX; ++kk) {
    ex_invariant[kk] = (kk <= kmax_ex) ? (phi * Tk[kk]) : 0.0f;
  }
}

// -----------------------------------------------------------------------------
// Shared parameter helpers (kmax/pmax clamping and block layout)
// -----------------------------------------------------------------------------

static __host__ __device__ __forceinline__ int nep_spin_clamp_kmax(const int kmax, const int kmax_max = 8)
{
  if (kmax < -1) return -1;
  if (kmax > kmax_max) return kmax_max;
  return kmax;
}

static __host__ __device__ __forceinline__ int nep_spin_clamp_pmax(const int pmax, const int pmax_max = 8)
{
  if (pmax < 0) return 0;
  if (pmax > pmax_max) return pmax_max;
  return pmax;
}

static __host__ __device__ __forceinline__ int nep_spin_blocks_from_kmax(const int kmax)
{
  return (kmax >= 0) ? (kmax + 1) : 0;
}

struct NepSpinPairBlocks {
  int kmax_ex = -1;
  int kmax_dmi = -1;
  int kmax_ani = -1;
  int kmax_sia = -1;
  int kmax_pair = 0;

  int ex_blocks = 0;
  int dmi_blocks = 0;
  int ani_blocks = 0;
  int sia_blocks = 0;

  int dmi_block0 = 0;
  int ani_block0 = 0;
  int sia_block0 = 0;

  int pair_blocks = 0;
};

template <typename ParaMB>
static __host__ __device__ __forceinline__ NepSpinPairBlocks nep_spin_get_pair_blocks(const ParaMB& paramb)
{
  NepSpinPairBlocks out;
  out.kmax_ex = nep_spin_clamp_kmax(paramb.spin_kmax_ex);
  out.kmax_dmi = nep_spin_clamp_kmax(paramb.spin_kmax_dmi);
  out.kmax_ani = nep_spin_clamp_kmax(paramb.spin_kmax_ani);
  out.kmax_sia = nep_spin_clamp_kmax(paramb.spin_kmax_sia);

  out.ex_blocks = nep_spin_blocks_from_kmax(out.kmax_ex);
  out.dmi_blocks = nep_spin_blocks_from_kmax(out.kmax_dmi);
  out.ani_blocks = nep_spin_blocks_from_kmax(out.kmax_ani);
  out.sia_blocks = nep_spin_blocks_from_kmax(out.kmax_sia);

  out.dmi_block0 = out.ex_blocks;
  out.ani_block0 = out.dmi_block0 + out.dmi_blocks;
  out.sia_block0 = out.ani_block0 + out.ani_blocks;
  out.pair_blocks = out.ex_blocks + out.dmi_blocks + out.ani_blocks + out.sia_blocks;

  int kmax_pair = out.kmax_ex;
  if (out.kmax_dmi > kmax_pair) kmax_pair = out.kmax_dmi;
  if (out.kmax_ani > kmax_pair) kmax_pair = out.kmax_ani;
  if (out.kmax_sia > kmax_pair) kmax_pair = out.kmax_sia;
  if (kmax_pair < 0) kmax_pair = 0;
  out.kmax_pair = kmax_pair;

  return out;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_nspin(const ParaMB& paramb)
{
  const int nmax = (paramb.spin_n_max >= 0) ? paramb.spin_n_max : paramb.n_max_radial;
  return nmax + 1;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_spin_dim(const ParaMB& paramb)
{
  const NepSpinPairBlocks blocks = nep_spin_get_pair_blocks(paramb);
  const int nspin = nep_spin_nspin(paramb);
  const int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  return nspin * blocks.pair_blocks + spin_pmax;
}
