/*
    Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren
    This file is part of NEP_CPU.
    NEP_CPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    NEP_CPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with NEP_CPU.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
A CPU implementation of the neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep.h"
#include "dftd3para.h"
#include "nep_utilities.h"
#include "neighbor_nep.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include("lmptype.h")
#include "lmptype.h"
#endif
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{

constexpr double kSpinZeroEpsSph = 1.0e-8;
constexpr int kSpinMaxPair = 8;
#ifdef NEIGHMASK
constexpr int kLammpsNeighMask = NEIGHMASK;
#else
// Fallback for standalone builds (and for LAMMPS builds where lmptype.h isn't reachable).
// NOTE: In LAMMPS (at least 22 Jul 2025 - Update 2), NEIGHMASK is 0x1FFFFFFF.
constexpr int kLammpsNeighMask = 0x1FFFFFFF;
#endif

inline int lammps_unpack_neigh_index(const int packed)
{
  return static_cast<int>(
    static_cast<std::uint32_t>(packed) & static_cast<std::uint32_t>(kLammpsNeighMask));
}

enum SpinCMode { SPIN_C_SHARED_LATTICE, SPIN_C_SINGLE, SPIN_C_PER_BLOCK };

inline SpinCMode nep_spin_get_c_mode(const int num_c_spin, const int c_spin_block_stride)
{
  if (num_c_spin == 0) return SPIN_C_SHARED_LATTICE;
  if (num_c_spin == c_spin_block_stride) return SPIN_C_SINGLE;
  return SPIN_C_PER_BLOCK;
}

inline int nep_spin_get_c_index(
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

inline double nep_spin_clamp_unit(const double x)
{
  return (x > 1.0) ? 1.0 : ((x < -1.0) ? -1.0 : x);
}

inline double nep_spin_dot3(const double a[3], const double b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void nep_spin_cross3(const double a[3], const double b[3], double out[3])
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

inline void nep_spin_fill_Tk(const double c, const int kmax, double* Tk)
{
  Tk[0] = 1.0;
  if (kSpinMaxPair >= 1) {
    Tk[1] = (kmax >= 1) ? c : 0.0;
  }
  for (int kk = 2; kk <= kSpinMaxPair; ++kk) {
    const double val = 2.0 * c * Tk[kk - 1] - Tk[kk - 2];
    Tk[kk] = (kk <= kmax) ? val : 0.0;
  }
}

inline void nep_spin_fill_Tk_and_dTk(const double c, const int kmax, double* Tk, double* dTk)
{
  Tk[0] = 1.0;
  dTk[0] = 0.0;
  if (kSpinMaxPair >= 1) {
    Tk[1] = (kmax >= 1) ? c : 0.0;
    dTk[1] = (kmax >= 1) ? 1.0 : 0.0;
  }
  for (int kk = 2; kk <= kSpinMaxPair; ++kk) {
    if (kk <= kmax) {
      Tk[kk] = 2.0 * c * Tk[kk - 1] - Tk[kk - 2];
      dTk[kk] = 2.0 * Tk[kk - 1] + 2.0 * c * dTk[kk - 1] - dTk[kk - 2];
    } else {
      Tk[kk] = 0.0;
      dTk[kk] = 0.0;
    }
  }
}

inline void nep_spin_fill_dc_dsi_dsj(
  const double si[3],
  const double sj[3],
  const double sdot,
  const double si_norm,
  const double sj_norm,
  double dc_dsi[3],
  double dc_dsj[3])
{
  const double denom = si_norm * sj_norm;
  const double inv_denom = 1.0 / (denom + 1.0e-12);
  const double inv_a2 = 1.0 / (si_norm * si_norm + 1.0e-12);
  const double inv_b2 = 1.0 / (sj_norm * sj_norm + 1.0e-12);
  dc_dsi[0] = inv_denom * (sj[0] - sdot * inv_a2 * si[0]);
  dc_dsi[1] = inv_denom * (sj[1] - sdot * inv_a2 * si[1]);
  dc_dsi[2] = inv_denom * (sj[2] - sdot * inv_a2 * si[2]);
  dc_dsj[0] = inv_denom * (si[0] - sdot * inv_b2 * sj[0]);
  dc_dsj[1] = inv_denom * (si[1] - sdot * inv_b2 * sj[1]);
  dc_dsj[2] = inv_denom * (si[2] - sdot * inv_b2 * sj[2]);
}

inline double nep_spin_ex_phi(const int phi_mode, const double si_norm, const double sj_norm, const double denom)
{
  if (phi_mode == 1) return si_norm;
  if (phi_mode == 2) return sj_norm;
  if (phi_mode == 3) return 1.0;
  return denom;
}

inline int nep_spin_clamp_kmax(const int kmax)
{
  if (kmax < -1) return -1;
  if (kmax > kSpinMaxPair) return kSpinMaxPair;
  return kmax;
}

inline int nep_spin_clamp_pmax(const int pmax)
{
  if (pmax < 0) return 0;
  if (pmax > kSpinMaxPair) return kSpinMaxPair;
  return pmax;
}

inline int nep_spin_blocks_from_kmax(const int kmax) { return (kmax >= 0) ? (kmax + 1) : 0; }

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

inline NepSpinPairBlocks nep_spin_get_pair_blocks(const NEP::ParaMB& paramb)
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

void find_descriptor_small_box(
  const bool calculating_potential,
  const bool calculating_descriptor,
  const bool calculating_latent_space,
  const bool calculating_polarizability,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12_radial,
  const double* g_y12_radial,
  const double* g_z12_radial,
  const double* g_x12_angular,
  const double* g_y12_angular,
  const double* g_z12_angular,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_radial,
  const double* g_gn_angular,
#endif
  double* g_Fp,
  double* g_sum_fxyz,
  double* g_potential,
  double* g_descriptor,
  double* g_latent_space,
  double* g_virial,
  bool calculating_B_projection,
  double* g_B_projection)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      double fc12;
      int t2 = g_type[n2];
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
#endif
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
        int index_left, index_right;
        double weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * paramb.num_types + g_type[n2];
        double gn12 =
          g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
#else
        int t2 = g_type[n2];
        double fc12;
        double rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5;
        double rcinv = 1.0 / rc;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
#endif
      }
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    if (calculating_descriptor) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_descriptor[d * N + n1] = q[d] * paramb.q_scaler[d];
      }
    }

    if (
      calculating_potential || calculating_latent_space || calculating_polarizability ||
      calculating_B_projection) {
      for (int d = 0; d < annmb.dim; ++d) {
        q[d] = q[d] * paramb.q_scaler[d];
      }

      double F = 0.0, Fp[MAX_DIM] = {0.0}, latent_space[MAX_NEURON] = {0.0};

      if (calculating_polarizability) {
        apply_ann_one_layer(
          annmb.dim, annmb.num_neurons1, annmb.w0_pol[t1], annmb.b0_pol[t1], annmb.w1_pol[t1],
          annmb.b1_pol, q, F, Fp, latent_space, false, nullptr);
        g_virial[n1] = F;
        g_virial[n1 + N * 4] = F;
        g_virial[n1 + N * 8] = F;

        for (int d = 0; d < annmb.dim; ++d) {
          Fp[d] = 0.0;
        }
        for (int d = 0; d < annmb.num_neurons1; ++d) {
          latent_space[d] = 0.0;
        }
      }

      if (paramb.version == 5) {
        apply_ann_one_layer_nep5(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F,
          Fp, latent_space);
      } else {
        apply_ann_one_layer(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F,
          Fp, latent_space, calculating_B_projection,
          g_B_projection + n1 * (annmb.num_neurons1 * (annmb.dim + 2)));
      }

      if (calculating_latent_space) {
        for (int n = 0; n < annmb.num_neurons1; ++n) {
          g_latent_space[n * N + n1] = latent_space[n];
        }
      }

      if (calculating_potential) {
        g_potential[n1] += F;
      }

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
      }
    }
  }
}

void find_force_radial_small_box(
  const bool is_dipole,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gnp_radial,
#endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 =
          g_gnp_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#else
      double fc12, fcp12;
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#endif

      if (g_fx) {
        g_fx[n1] += f12[0];
        g_fx[n2] -= f12[0];
      }

      if (g_fy) {
        g_fy[n1] += f12[1];
        g_fy[n2] -= f12[1];
      }

      if (g_fz) {
        g_fz[n1] += f12[2];
        g_fz[n2] -= f12[2];
      }

      if (!is_dipole) {
        g_virial[n2 + 0 * N] -= r12[0] * f12[0];
        g_virial[n2 + 1 * N] -= r12[0] * f12[1];
        g_virial[n2 + 2 * N] -= r12[0] * f12[2];
        g_virial[n2 + 3 * N] -= r12[1] * f12[0];
        g_virial[n2 + 4 * N] -= r12[1] * f12[1];
        g_virial[n2 + 5 * N] -= r12[1] * f12[2];
        g_virial[n2 + 6 * N] -= r12[2] * f12[0];
        g_virial[n2 + 7 * N] -= r12[2] * f12[1];
        g_virial[n2 + 8 * N] -= r12[2] * f12[2];
      } else {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        g_virial[n2 + 0 * N] -= r12_square * f12[0];
        g_virial[n2 + 1 * N] -= r12_square * f12[1];
        g_virial[n2 + 2 * N] -= r12_square * f12[2];
      }
    }
  }
}

void find_force_angular_small_box(
  const bool is_dipole,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_sum_fxyz,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_angular,
  const double* g_gnp_angular,
#endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        int index_left_all =
          (index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        int index_right_all =
          (index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        double gn12 =
          g_gn_angular[index_left_all] * weight_left + g_gn_angular[index_right_all] * weight_right;
        double gnp12 = g_gnp_angular[index_left_all] * weight_left +
                       g_gnp_angular[index_right_all] * weight_right;
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }
#else
      int t2 = g_type[n2];
      double fc12, fcp12;
      double rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }
#endif

      if (g_fx) {
        g_fx[n1] += f12[0];
        g_fx[n2] -= f12[0];
      }

      if (g_fy) {
        g_fy[n1] += f12[1];
        g_fy[n2] -= f12[1];
      }

      if (g_fz) {
        g_fz[n1] += f12[2];
        g_fz[n2] -= f12[2];
      }

      if (!is_dipole) {
        g_virial[n2 + 0 * N] -= r12[0] * f12[0];
        g_virial[n2 + 1 * N] -= r12[0] * f12[1];
        g_virial[n2 + 2 * N] -= r12[0] * f12[2];
        g_virial[n2 + 3 * N] -= r12[1] * f12[0];
        g_virial[n2 + 4 * N] -= r12[1] * f12[1];
        g_virial[n2 + 5 * N] -= r12[1] * f12[2];
        g_virial[n2 + 6 * N] -= r12[2] * f12[0];
        g_virial[n2 + 7 * N] -= r12[2] * f12[1];
        g_virial[n2 + 8 * N] -= r12[2] * f12[2];
      } else {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        g_virial[n2 + 0 * N] -= r12_square * f12[0];
        g_virial[n2 + 1 * N] -= r12_square * f12[1];
        g_virial[n2 + 2 * N] -= r12_square * f12[2];
      }
    }
  }
}

void find_force_ZBL_small_box(
  const int N,
  NEP::ParaMB& paramb,
  const NEP::ZBL& zbl,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int type1 = g_type[n1];
    int zi = paramb.atomic_numbers[type1] + 1;
    double pow_zi = pow(double(zi), 0.23);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f, fp;
      int type2 = g_type[n2];
      int zj = paramb.atomic_numbers[type2] + 1;
      double a_inv = (pow_zi + pow(double(zj), 0.23)) * 2.134563;
      double zizj = K_C_SP * zi * zj;
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
        double ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        double rc_inner = zbl.rc_inner;
        double rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = std::min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
      g_pe[n1] += f * 0.5;
    }
  }
}

void find_descriptor_small_box(
  const bool calculating_potential,
  const bool calculating_descriptor,
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12_radial,
  const double* g_y12_radial,
  const double* g_z12_radial,
  const double* g_x12_angular,
  const double* g_y12_angular,
  const double* g_z12_angular,
  double* g_Fp,
  double* g_sum_fxyz,
  double* g_charge,
  double* g_charge_derivative,
  double* g_potential,
  double* g_descriptor)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

      double fc12;
      int t2 = g_type[n2];
      double rc = paramb.rc_radial_max;
      double rcinv = 1.0 / rc;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        int t2 = g_type[n2];
        double fc12;
        double rc = paramb.rc_angular_max;
        double rcinv = 1.0 / rc;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    if (calculating_descriptor) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_descriptor[d * N + n1] = q[d] * paramb.q_scaler[d];
      }
    }

    if (calculating_potential) {
      for (int d = 0; d < annmb.dim; ++d) {
        q[d] = q[d] * paramb.q_scaler[d];
      }

      double F = 0.0, Fp[MAX_DIM] = {0.0};
      double charge = 0.0;
      double charge_derivative[MAX_DIM] = {0.0};

      apply_ann_one_layer_charge(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        charge,
        charge_derivative);

      if (calculating_potential) {
        g_potential[n1] += F;
        g_charge[n1] = charge;
      }

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
        g_charge_derivative[d * N + n1] = charge_derivative[d] * paramb.q_scaler[d];
      }
    }
  }
}

void find_descriptor_spin_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12_radial,
  const double* g_y12_radial,
  const double* g_z12_radial,
  const double* g_x12_angular,
  const double* g_y12_angular,
  const double* g_z12_angular,
  const double* g_spin,
  double* g_Fp,
  double* g_sum_fxyz,
  double* g_potential,
  double* g_descriptor)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  const int spin_dim = nspin * spin_blocks.pair_blocks + spin_pmax;
  const int spin_end = spin_offset + spin_dim;
  if (spin_end > annmb.dim) {
    std::cout << "Spin descriptor block exceeds ANN dim.\n";
    exit(1);
  }

  const SpinCMode spin_c_mode = nep_spin_get_c_mode(
    static_cast<int>(paramb.num_c_spin),
    static_cast<int>(paramb.c_spin_block_stride));

  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

      double fc12;
      int t2 = g_type[n2];
      double rc = paramb.rc_radial[t1];
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      int bs = paramb.basis_size_radial;
      if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
      find_fn(bs, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        double r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        int t2 = g_type[n2];
        double fc12;
        double rc = paramb.rc_angular[t1];
        if (paramb.use_typewise_cutoff) {
          rc = std::min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        double rcinv = 1.0 / rc;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        int bs = paramb.basis_size_angular;
        if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
        find_fn(bs, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int d = spin_offset; d < spin_end; ++d) {
      q[d] = 0.0;
    }

    if (spin_blocks.pair_blocks > 0 || spin_pmax > 0) {
      double si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
      double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];

      if (si2 > kSpinZeroEpsSph) {
        double si_norm = sqrt(si2);
        for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
          int index = i1 * N + n1;
          int n2 = g_NL_radial[index];
          int t2 = g_type[n2];
          double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
          double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
          if (d12 <= 0.0) {
            continue;
          }
          double rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

          double rc = paramb.rc_radial[t1];
          if (paramb.use_typewise_cutoff) {
            rc = std::min(
              (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
                paramb.typewise_cutoff_radial_factor,
              rc);
          }
          double rcinv = 1.0 / rc;
          double fc12;
          find_fc(rc, rcinv, d12, fc12);
          double fn12[MAX_NUM_N];
          int bs = paramb.basis_size_radial;
          if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
          find_fn(bs, rcinv, d12, fc12, fn12);

          double sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
          double sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
          const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
          double sj_norm = neighbor_has_spin ? sqrt(sj2) : 0.0;

          if (spin_blocks.kmax_ex >= 0 && neighbor_has_spin) {
            const double sdot = nep_spin_dot3(si, sj);
            const double denom = si_norm * sj_norm;
            const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
            double Tk[9] = {0.0};
            nep_spin_fill_Tk(c, spin_blocks.kmax_ex, Tk);
            const double phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);

            for (int n = 0; n < nspin; ++n) {
              double gn_ex[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_ex; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_ex[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_ex; ++kk) {
                int off = spin_offset + kk * nspin;
                q[off + n] += gn_ex[kk] * (phi * Tk[kk]);
              }
            }
          }

          if (spin_blocks.kmax_dmi >= 0 && neighbor_has_spin) {
            const double sdot = nep_spin_dot3(si, sj);
            double sixsj[3];
            nep_spin_cross3(si, sj, sixsj);
            const double dmi_val = nep_spin_dot3(sixsj, rhat);
            const double denom = si_norm * sj_norm;
            const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
            double Tk[9] = {0.0};
            nep_spin_fill_Tk(c, spin_blocks.kmax_dmi, Tk);

            for (int n = 0; n < nspin; ++n) {
              double gn_dmi[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_dmi; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    spin_blocks.dmi_block0 + kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_dmi[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_dmi; ++kk) {
                int off = spin_offset + (spin_blocks.dmi_block0 + kk) * nspin;
                q[off + n] += gn_dmi[kk] * (dmi_val * Tk[kk]);
              }
            }
          }

          if (spin_blocks.kmax_ani >= 0 && neighbor_has_spin) {
            const double si_r = nep_spin_dot3(si, rhat);
            const double sj_r = nep_spin_dot3(sj, rhat);
            const double ani_scalar = si_r * sj_r;
            const double sdot = nep_spin_dot3(si, sj);
            const double denom = si_norm * sj_norm;
            const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
            double Tk[9] = {0.0};
            nep_spin_fill_Tk(c, spin_blocks.kmax_ani, Tk);

            for (int n = 0; n < nspin; ++n) {
              double gn_ani[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_ani; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    spin_blocks.ani_block0 + kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_ani[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_ani; ++kk) {
                int off = spin_offset + (spin_blocks.ani_block0 + kk) * nspin;
                q[off + n] += gn_ani[kk] * (ani_scalar * Tk[kk]);
              }
            }
          }

          if (spin_blocks.kmax_sia >= 0) {
            const double si_r = nep_spin_dot3(si, rhat);
            const double sia_scalar = si_r * si_r;
            double Tk[9] = {0.0};
            Tk[0] = 1.0;
            if (neighbor_has_spin) {
              const double sdot = nep_spin_dot3(si, sj);
              const double denom = si_norm * sj_norm;
              const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
              nep_spin_fill_Tk(c, spin_blocks.kmax_sia, Tk);
            } else {
              for (int kk = 1; kk <= kSpinMaxPair; ++kk) {
                Tk[kk] = 0.0;
              }
            }

            for (int n = 0; n < nspin; ++n) {
              double gn_sia[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_sia; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    spin_blocks.sia_block0 + kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_sia[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_sia; ++kk) {
                if (kk == 0 || neighbor_has_spin) {
                  int off = spin_offset + (spin_blocks.sia_block0 + kk) * nspin;
                  q[off + n] += gn_sia[kk] * (sia_scalar * Tk[kk]);
                }
              }
            }
          }
        }
      }

      if (spin_pmax > 0) {
        const int onsite_offset = spin_offset + nspin * spin_blocks.pair_blocks;
        double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
        if (si2 <= kSpinZeroEpsSph) {
          for (int p = 1; p <= spin_pmax; ++p) {
            q[onsite_offset + (p - 1)] = 0.0;
          }
        } else if (paramb.spin_onsite_basis_mode == 0) {
          double m2 = si2;
          double m2p = m2;
          for (int p = 1; p <= spin_pmax; ++p) {
            q[onsite_offset + (p - 1)] = m2p;
            m2p *= m2;
          }
        } else {
          double y = si2;
          double yref = paramb.spin_mref;
          if (paramb.spin_onsite_basis_mode == 2) {
            y = sqrt(si2);
          } else {
            yref = paramb.spin_mref * paramb.spin_mref;
          }
          if (yref <= 0.0) yref = 1.0;
          double x = (y - yref) / (y + yref + 1.0e-12);
          x = std::max(-1.0, std::min(1.0, x));

          double Tp[9] = {0.0};
          Tp[0] = 1.0;
          if (spin_pmax >= 1) Tp[1] = x;
          for (int p = 2; p <= spin_pmax; ++p) {
            Tp[p] = 2.0 * x * Tp[p - 1] - Tp[p - 2];
          }
          for (int p = 1; p <= spin_pmax; ++p) {
            q[onsite_offset + (p - 1)] = Tp[p];
          }
        }
      }
    }

    if (g_descriptor) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_descriptor[d * N + n1] = q[d] * paramb.q_scaler[d];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    double F = 0.0;
    double Fp[MAX_DIM] = {0.0};
    double latent_space[MAX_NEURON] = {0.0};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F,
      Fp, latent_space, false, nullptr);

    g_potential[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

void find_force_radial_spinbase_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
      double fc12, fcp12;
      double rc = paramb.rc_radial[t1];
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      int bs = paramb.basis_size_radial;
      if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
      find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];

      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_force_angular_spinbase_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};
      int t2 = g_type[n2];
      double fc12, fcp12;
      double rc = paramb.rc_angular[t1];
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      int bs = paramb.basis_size_angular;
      if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
      find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }

      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_mforce_radial_spin_spherical_onsite_small_box(
  NEP::ParaMB& paramb,
  const int N,
  const double* g_spin,
  const double* g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  if (spin_pmax <= 0) return;
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const int onsite_offset = spin_offset + nspin * spin_blocks.pair_blocks;
  const double msign = paramb.mforce_sign;

  for (int n1 = 0; n1 < N; ++n1) {
    double si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
    const double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) {
      continue;
    }

    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;

    if (paramb.spin_onsite_basis_mode == 0) {
      const double m2 = si2;
      double m2pow = 1.0;
      for (int p = 1; p <= spin_pmax; ++p) {
        const double Fp_p = g_Fp[(onsite_offset + (p - 1)) * N + n1];
        const double coeff = msign * Fp_p * (2.0 * p) * m2pow;
        mx += coeff * si[0];
        my += coeff * si[1];
        mz += coeff * si[2];
        m2pow *= m2;
      }
    } else {
      double y = si2;
      double yref = paramb.spin_mref * paramb.spin_mref;
      const double si_norm = sqrt(si2);
      const double inv_si_norm = 1.0 / (si_norm + 1.0e-12);
      if (paramb.spin_onsite_basis_mode == 2) {
        y = si_norm;
        yref = paramb.spin_mref;
      }
      if (yref <= 0.0) yref = 1.0;

      const double denom = y + yref;
      const double inv_denom = 1.0 / (denom + 1.0e-12);
      double x = (y - yref) * inv_denom;
      x = std::max(-1.0, std::min(1.0, x));
      const double dx_dy = (2.0 * yref) * inv_denom * inv_denom;

      double Tp[9] = {0.0};
      double dTp[9] = {0.0};
      Tp[0] = 1.0;
      dTp[0] = 0.0;
      if (spin_pmax >= 1) {
        Tp[1] = x;
        dTp[1] = 1.0;
      }
      for (int p = 2; p <= spin_pmax; ++p) {
        Tp[p] = 2.0 * x * Tp[p - 1] - Tp[p - 2];
        dTp[p] = 2.0 * Tp[p - 1] + 2.0 * x * dTp[p - 1] - dTp[p - 2];
      }

      double dy_dsi[3];
      if (paramb.spin_onsite_basis_mode == 2) {
        dy_dsi[0] = inv_si_norm * si[0];
        dy_dsi[1] = inv_si_norm * si[1];
        dy_dsi[2] = inv_si_norm * si[2];
      } else {
        dy_dsi[0] = 2.0 * si[0];
        dy_dsi[1] = 2.0 * si[1];
        dy_dsi[2] = 2.0 * si[2];
      }

      for (int p = 1; p <= spin_pmax; ++p) {
        const double Fp_p = g_Fp[(onsite_offset + (p - 1)) * N + n1];
        const double coeff = msign * Fp_p * dTp[p] * dx_dy;
        mx += coeff * dy_dsi[0];
        my += coeff * dy_dsi[1];
        mz += coeff * dy_dsi[2];
      }
    }

    g_mx[n1] += mx;
    g_my[n1] += my;
    g_mz[n1] += mz;
  }
}

void find_force_radial_spin_spherical_fused_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_spin,
  const double* g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  if (spin_blocks.pair_blocks == 0) {
    return;
  }
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const SpinCMode spin_c_mode = nep_spin_get_c_mode(
    static_cast<int>(paramb.num_c_spin),
    static_cast<int>(paramb.c_spin_block_stride));
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  for (int n1 = 0; n1 < N; ++n1) {
    double si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
    double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) {
      continue;
    }
    const double si_norm = sqrt(si2);
    const int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (d12 <= 0.0) {
        continue;
      }
      double rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

      double rc = paramb.rc_radial[t1];
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;

      double fc12, dfc12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, dfc12);
      double fn12[MAX_NUM_N];
      double dfn12[MAX_NUM_N];
      find_fn_and_fnp(bs, rcinv, d12, fc12, dfc12, fn12, dfn12);

      double sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
      double sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
      const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
      double sj_norm = neighbor_has_spin ? sqrt(sj2) : 0.0;
      double sdot = neighbor_has_spin ? nep_spin_dot3(si, sj) : 0.0;
      double denom = neighbor_has_spin ? (si_norm * sj_norm) : 0.0;
      double c = neighbor_has_spin ? nep_spin_clamp_unit(sdot / (denom + 1.0e-12)) : 0.0;

      double fvec[3] = {0.0, 0.0, 0.0};

      if (spin_blocks.kmax_ex >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        nep_spin_fill_Tk(c, spin_blocks.kmax_ex, Tk);
        const double phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);
        double force_mag = 0.0;
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ex; ++k) {
            const double fp_val = g_Fp[(spin_offset + k * nspin + n) * N + n1];
            double dC_dr = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                k,
                n,
                kb,
                t1,
                t2);
              dC_dr += dfn12[kb] * annmb.c[c_idx];
            }
            force_mag += fp_val * dC_dr * (phi * Tk[k]);
          }
        }
        fvec[0] += force_mag * rhat[0];
        fvec[1] += force_mag * rhat[1];
        fvec[2] += force_mag * rhat[2];
      }

      if (spin_blocks.kmax_dmi >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        nep_spin_fill_Tk(c, spin_blocks.kmax_dmi, Tk);
        double sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        const double dmi_val = nep_spin_dot3(sixsj, rhat);
        double fvec_dmi[3] = {0.0, 0.0, 0.0};
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_dmi; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.dmi_block0 + k) * nspin + n) * N + n1];
            double dC_dr = 0.0;
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.dmi_block0 + k,
                n,
                kb,
                t1,
                t2);
              const double coeff = annmb.c[c_idx];
              dC_dr += dfn12[kb] * coeff;
              C_val += fn12[kb] * coeff;
            }
            const double term1 = (dC_dr * dmi_val - C_val * dmi_val / d12) * Tk[k];
            const double term2 = (C_val * Tk[k] / d12);
            fvec_dmi[0] += fp_val * (term1 * rhat[0] + term2 * sixsj[0]);
            fvec_dmi[1] += fp_val * (term1 * rhat[1] + term2 * sixsj[1]);
            fvec_dmi[2] += fp_val * (term1 * rhat[2] + term2 * sixsj[2]);
          }
        }
        fvec[0] += fvec_dmi[0];
        fvec[1] += fvec_dmi[1];
        fvec[2] += fvec_dmi[2];
      }

      if (spin_blocks.kmax_ani >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        nep_spin_fill_Tk(c, spin_blocks.kmax_ani, Tk);
        const double si_r = nep_spin_dot3(si, rhat);
        const double sj_r = nep_spin_dot3(sj, rhat);
        const double ani_scalar = si_r * sj_r;
        double fvec_ani[3] = {0.0, 0.0, 0.0};
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ani; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.ani_block0 + k) * nspin + n) * N + n1];
            double dC_dr = 0.0;
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.ani_block0 + k,
                n,
                kb,
                t1,
                t2);
              const double coeff = annmb.c[c_idx];
              dC_dr += dfn12[kb] * coeff;
              C_val += fn12[kb] * coeff;
            }
            const double term1 = (dC_dr * ani_scalar - 2.0 * C_val * ani_scalar / d12) * Tk[k];
            const double term2 = (C_val * Tk[k] / d12);
            fvec_ani[0] += fp_val * (term1 * rhat[0] + term2 * (sj_r * si[0] + si_r * sj[0]));
            fvec_ani[1] += fp_val * (term1 * rhat[1] + term2 * (sj_r * si[1] + si_r * sj[1]));
            fvec_ani[2] += fp_val * (term1 * rhat[2] + term2 * (sj_r * si[2] + si_r * sj[2]));
          }
        }
        fvec[0] += fvec_ani[0];
        fvec[1] += fvec_ani[1];
        fvec[2] += fvec_ani[2];
      }

      if (spin_blocks.kmax_sia >= 0) {
        double Tk[9] = {0.0};
        Tk[0] = 1.0;
        if (neighbor_has_spin) {
          nep_spin_fill_Tk(c, spin_blocks.kmax_sia, Tk);
        } else {
          for (int kk = 1; kk <= kSpinMaxPair; ++kk) {
            Tk[kk] = 0.0;
          }
        }
        const double si_r = nep_spin_dot3(si, rhat);
        const double sia_scalar = si_r * si_r;
        double fvec_sia[3] = {0.0, 0.0, 0.0};
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_sia; ++k) {
            if (k > 0 && !neighbor_has_spin) {
              continue;
            }
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.sia_block0 + k) * nspin + n) * N + n1];
            double dC_dr = 0.0;
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.sia_block0 + k,
                n,
                kb,
                t1,
                t2);
              const double coeff = annmb.c[c_idx];
              dC_dr += dfn12[kb] * coeff;
              C_val += fn12[kb] * coeff;
            }
            const double term1 = (dC_dr * sia_scalar - 2.0 * C_val * sia_scalar / d12) * Tk[k];
            const double term2 = (2.0 * C_val * Tk[k] * si_r / d12);
            fvec_sia[0] += fp_val * (term1 * rhat[0] + term2 * si[0]);
            fvec_sia[1] += fp_val * (term1 * rhat[1] + term2 * si[1]);
            fvec_sia[2] += fp_val * (term1 * rhat[2] + term2 * si[2]);
          }
        }
        fvec[0] += fvec_sia[0];
        fvec[1] += fvec_sia[1];
        fvec[2] += fvec_sia[2];
      }

      g_fx[n1] += fvec[0];
      g_fy[n1] += fvec[1];
      g_fz[n1] += fvec[2];
      g_fx[n2] -= fvec[0];
      g_fy[n2] -= fvec[1];
      g_fz[n2] -= fvec[2];

      g_virial[n2 + 0 * N] -= r12[0] * fvec[0];
      g_virial[n2 + 1 * N] -= r12[0] * fvec[1];
      g_virial[n2 + 2 * N] -= r12[0] * fvec[2];
      g_virial[n2 + 3 * N] -= r12[1] * fvec[0];
      g_virial[n2 + 4 * N] -= r12[1] * fvec[1];
      g_virial[n2 + 5 * N] -= r12[1] * fvec[2];
      g_virial[n2 + 6 * N] -= r12[2] * fvec[0];
      g_virial[n2 + 7 * N] -= r12[2] * fvec[1];
      g_virial[n2 + 8 * N] -= r12[2] * fvec[2];
    }
  }
}

void find_mforce_radial_spin_spherical_fused_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_spin,
  const double* g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  if (spin_blocks.pair_blocks == 0) {
    return;
  }
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const SpinCMode spin_c_mode = nep_spin_get_c_mode(
    static_cast<int>(paramb.num_c_spin),
    static_cast<int>(paramb.c_spin_block_stride));
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  const double msign = paramb.mforce_sign;

  for (int n1 = 0; n1 < N; ++n1) {
    double si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
    double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) {
      continue;
    }
    const double si_norm = sqrt(si2);
    const double inv_si = 1.0 / (si_norm + 1.0e-12);
    const int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (d12 <= 0.0) {
        continue;
      }
      double rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

      double rc = paramb.rc_radial[t1];
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;

      double fc12;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(bs, rcinv, d12, fc12, fn12);

      double sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
      double sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
      const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
      const double sj_norm = neighbor_has_spin ? sqrt(sj2) : 0.0;
      const double inv_sj = neighbor_has_spin ? (1.0 / (sj_norm + 1.0e-12)) : 0.0;
      const double sdot = neighbor_has_spin ? nep_spin_dot3(si, sj) : 0.0;
      const double denom = neighbor_has_spin ? (si_norm * sj_norm) : 0.0;
      const double c =
        neighbor_has_spin ? nep_spin_clamp_unit(sdot / (denom + 1.0e-12)) : 0.0;

      double mvec_i[3] = {0.0, 0.0, 0.0};
      double mvec_j[3] = {0.0, 0.0, 0.0};

      if (spin_blocks.kmax_ex >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_ex, Tk, Uk);
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ex; ++k) {
            const double fp_val = g_Fp[(spin_offset + k * nspin + n) * N + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            double term_i_si = 0.0;
            double term_i_sj = 0.0;
            double term_j_si = 0.0;
            double term_j_sj = 0.0;

            if (paramb.spin_ex_phi_mode == 0) {
              const double ratio = sj_norm * inv_si;
              term_i_si = (Tk[k] - c * Uk[k]) * ratio;
              term_i_sj = Uk[k];
              const double ratio_j = si_norm * inv_sj;
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
              const double inv_denom = 1.0 / (denom + 1.0e-12);
              const double ratio_i = sj_norm * inv_si;
              term_i_si = -c * Uk[k] * ratio_i * inv_denom;
              term_i_sj = Uk[k] * inv_denom;
              const double ratio_j = si_norm * inv_sj;
              term_j_sj = -c * Uk[k] * ratio_j * inv_denom;
              term_j_si = Uk[k] * inv_denom;
            }

            const double pre = fp_val * C_val;
            mvec_i[0] += pre * (term_i_si * si[0] + term_i_sj * sj[0]);
            mvec_i[1] += pre * (term_i_si * si[1] + term_i_sj * sj[1]);
            mvec_i[2] += pre * (term_i_si * si[2] + term_i_sj * sj[2]);
            mvec_j[0] += pre * (term_j_si * si[0] + term_j_sj * sj[0]);
            mvec_j[1] += pre * (term_j_si * si[1] + term_j_sj * sj[1]);
            mvec_j[2] += pre * (term_j_si * si[2] + term_j_sj * sj[2]);
          }
        }
      }

      if (spin_blocks.kmax_dmi >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_dmi, Tk, Uk);
        double dc_dsi[3] = {0.0, 0.0, 0.0};
        double dc_dsj[3] = {0.0, 0.0, 0.0};
        nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
        double sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        const double dmi_val = nep_spin_dot3(sixsj, rhat);
        double dDMI_dsi[3] = {
          sj[1] * rhat[2] - sj[2] * rhat[1],
          sj[2] * rhat[0] - sj[0] * rhat[2],
          sj[0] * rhat[1] - sj[1] * rhat[0]};
        double dDMI_dsj[3] = {
          rhat[1] * si[2] - rhat[2] * si[1],
          rhat[2] * si[0] - rhat[0] * si[2],
          rhat[0] * si[1] - rhat[1] * si[0]};

        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_dmi; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.dmi_block0 + k) * nspin + n) * N + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.dmi_block0 + k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            const double term1 = C_val * Tk[k];
            const double term2 = C_val * dmi_val * Uk[k];
            for (int d = 0; d < 3; ++d) {
              mvec_i[d] += fp_val * (term1 * dDMI_dsi[d] + term2 * dc_dsi[d]);
              mvec_j[d] += fp_val * (term1 * dDMI_dsj[d] + term2 * dc_dsj[d]);
            }
          }
        }
      }

      if (spin_blocks.kmax_ani >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_ani, Tk, Uk);
        double dc_dsi[3] = {0.0, 0.0, 0.0};
        double dc_dsj[3] = {0.0, 0.0, 0.0};
        nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
        const double si_r = nep_spin_dot3(si, rhat);
        const double sj_r = nep_spin_dot3(sj, rhat);
        const double ani_scalar = si_r * sj_r;

        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ani; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.ani_block0 + k) * nspin + n) * N + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.ani_block0 + k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            const double term1 = C_val * ani_scalar * Uk[k];
            const double term2 = C_val * Tk[k];
            for (int d = 0; d < 3; ++d) {
              mvec_i[d] += fp_val * (term1 * dc_dsi[d] + term2 * rhat[d] * sj_r);
              mvec_j[d] += fp_val * (term1 * dc_dsj[d] + term2 * rhat[d] * si_r);
            }
          }
        }
      }

      if (spin_blocks.kmax_sia >= 0) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        double dc_dsi[3] = {0.0, 0.0, 0.0};
        double dc_dsj[3] = {0.0, 0.0, 0.0};
        Tk[0] = 1.0;
        if (neighbor_has_spin) {
          nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_sia, Tk, Uk);
          nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
        }
        const double si_r = nep_spin_dot3(si, rhat);
        const double sia_scalar = si_r * si_r;

        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_sia; ++k) {
            if (k > 0 && !neighbor_has_spin) {
              continue;
            }
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.sia_block0 + k) * nspin + n) * N + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.sia_block0 + k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            const double term1 = C_val * sia_scalar * Uk[k];
            const double term2 = C_val * Tk[k] * 2.0 * si_r;
            for (int d = 0; d < 3; ++d) {
              mvec_i[d] += fp_val * (term1 * dc_dsi[d] + term2 * rhat[d]);
              mvec_j[d] += fp_val * (term1 * dc_dsj[d]);
            }
          }
        }
      }

      g_mx[n1] += msign * mvec_i[0];
      g_my[n1] += msign * mvec_i[1];
      g_mz[n1] += msign * mvec_i[2];
      g_mx[n2] += msign * mvec_j[0];
      g_my[n2] += msign * mvec_j[1];
      g_mz[n2] += msign * mvec_j[2];
    }
  }
}

void zero_total_charge(const int N, double* g_charge)
{
  double mean_charge = 0.0;
  for (int n = 0; n < N; ++n) {
    mean_charge += g_charge[n];
  }
  mean_charge /= N;
  for (int n = 0; n < N; ++n) {
    g_charge[n] -= mean_charge;
  }
}

void find_force_radial_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_charge_derivative,
  const double* g_D_real, 
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
      double fc12, fcp12;
      double rc = paramb.rc_radial_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        double tmp12 = (g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1]) * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      if (g_fx) {
        g_fx[n1] += f12[0];
        g_fx[n2] -= f12[0];
      }

      if (g_fy) {
        g_fy[n1] += f12[1];
        g_fy[n2] -= f12[1];
      }

      if (g_fz) {
        g_fz[n1] += f12[2];
        g_fz[n2] -= f12[2];
      }

      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_force_angular_small_box(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_Fp,
  const double* g_charge_derivative,
  const double* g_D_real, 
  const double* g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1] 
        + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};
      int t2 = g_type[n2];
      double fc12, fcp12;
      double rc = paramb.rc_angular_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }

      if (g_fx) {
        g_fx[n1] += f12[0];
        g_fx[n2] -= f12[0];
      }

      if (g_fy) {
        g_fy[n1] += f12[1];
        g_fy[n2] -= f12[1];
      }

      if (g_fz) {
        g_fz[n1] += f12[2];
        g_fz[n2] -= f12[2];
      }

      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_bec_diagonal(const int N, const double* g_q, double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    g_bec[n1 + N * 0] = g_q[n1];
    g_bec[n1 + N * 1] = 0.0;
    g_bec[n1 + N * 2] = 0.0;
    g_bec[n1 + N * 3] = 0.0;
    g_bec[n1 + N * 4] = g_q[n1];
    g_bec[n1 + N * 5] = 0.0;
    g_bec[n1 + N * 6] = 0.0;
    g_bec[n1 + N * 7] = 0.0;
    g_bec[n1 + N * 8] = g_q[n1];
  }
}

void find_bec_radial_small_box(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_charge_derivative,
  double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double fc12, fcp12;
      double rc = paramb.rc_radial_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      double f12[3] = {0.0};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const double tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      double bec_xx = 0.5* (r12[0] * f12[0]);
      double bec_xy = 0.5* (r12[0] * f12[1]);
      double bec_xz = 0.5* (r12[0] * f12[2]);
      double bec_yx = 0.5* (r12[1] * f12[0]);
      double bec_yy = 0.5* (r12[1] * f12[1]);
      double bec_yz = 0.5* (r12[1] * f12[2]);
      double bec_zx = 0.5* (r12[2] * f12[0]);
      double bec_zy = 0.5* (r12[2] * f12[1]);
      double bec_zz = 0.5* (r12[2] * f12[2]);

      g_bec[n1] += bec_xx;
      g_bec[n1 + N] += bec_xy;
      g_bec[n1 + N * 2] += bec_xz;
      g_bec[n1 + N * 3] += bec_yx;
      g_bec[n1 + N * 4] += bec_yy;
      g_bec[n1 + N * 5] += bec_yz;
      g_bec[n1 + N * 6] += bec_zx;
      g_bec[n1 + N * 7] += bec_zy;
      g_bec[n1 + N * 8] += bec_zz;

      g_bec[n2] -= bec_xx;
      g_bec[n2 + N] -= bec_xy;
      g_bec[n2 + N * 2] -= bec_xz;
      g_bec[n2 + N * 3] -= bec_yx;
      g_bec[n2 + N * 4] -= bec_yy;
      g_bec[n2 + N * 5] -= bec_yz;
      g_bec[n2 + N * 6] -= bec_zx;
      g_bec[n2 + N * 7] -= bec_zy;
      g_bec[n2 + N * 8] -= bec_zz;
    }
  }
}

void find_bec_angular_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  const double* g_charge_derivative,
  const double* g_sum_fxyz,
  double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};
      double fc12, fcp12;
      int t2 = g_type[n2];
      double rc = paramb.rc_angular_max;
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
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

      double bec_xx = 0.5* (r12[0] * f12[0]);
      double bec_xy = 0.5* (r12[0] * f12[1]);
      double bec_xz = 0.5* (r12[0] * f12[2]);
      double bec_yx = 0.5* (r12[1] * f12[0]);
      double bec_yy = 0.5* (r12[1] * f12[1]);
      double bec_yz = 0.5* (r12[1] * f12[2]);
      double bec_zx = 0.5* (r12[2] * f12[0]);
      double bec_zy = 0.5* (r12[2] * f12[1]);
      double bec_zz = 0.5* (r12[2] * f12[2]);

      g_bec[n1] += bec_xx;
      g_bec[n1 + N] += bec_xy;
      g_bec[n1 + N * 2] += bec_xz;
      g_bec[n1 + N * 3] += bec_yx;
      g_bec[n1 + N * 4] += bec_yy;
      g_bec[n1 + N * 5] += bec_yz;
      g_bec[n1 + N * 6] += bec_zx;
      g_bec[n1 + N * 7] += bec_zy;
      g_bec[n1 + N * 8] += bec_zz;

      g_bec[n2] -= bec_xx;
      g_bec[n2 + N] -= bec_xy;
      g_bec[n2 + N * 2] -= bec_xz;
      g_bec[n2 + N * 3] -= bec_yx;
      g_bec[n2 + N * 4] -= bec_yy;
      g_bec[n2 + N * 5] -= bec_yz;
      g_bec[n2 + N * 6] -= bec_zx;
      g_bec[n2 + N * 7] -= bec_zy;
      g_bec[n2 + N * 8] -= bec_zz;
    }
  }
}

void scale_bec(const int N, const double* sqrt_epsilon_inf, double* g_bec)
{
  for (int n1 = 0; n1 < N; ++n1) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

void find_force_charge_real_space_only_small_box(
  const int N,
  const NEP::Charge_Para charge_para,
  const int* g_NN,
  const int* g_NL,
  const double* g_charge,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  double* g_D_real)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double s_fx = 0.0;
    double s_fy = 0.0;
    double s_fz = 0.0;
    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;
    double q1 = g_charge[n1];
    double s_pe = 0; // no self energy
    double D_real = 0; // no self energy

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double q2 = g_charge[n2];
      double qq = q1 * q2;
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;

      double erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * (erfc_r + charge_para.A * d12 + charge_para.B);
      s_pe += 0.5 * qq * (erfc_r + charge_para.A * d12 + charge_para.B);
      double f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 = -0.5 * K_C_SP * qq * (f2 * d12inv * d12inv - charge_para.A * d12inv);
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      double f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_sxy;
    g_virial[n1 + 2 * N] += s_sxz;
    g_virial[n1 + 3 * N] += s_syx;
    g_virial[n1 + 4 * N] += s_syy;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_szx;
    g_virial[n1 + 7 * N] += s_szy;
    g_virial[n1 + 8 * N] += s_szz;
    g_D_real[n1] = K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

void find_force_charge_real_space_small_box(
  const int N,
  const NEP::Charge_Para charge_para,
  const int* g_NN,
  const int* g_NL,
  const double* g_charge,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  double* g_D_real)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double s_fx = 0.0;
    double s_fy = 0.0;
    double s_fz = 0.0;
    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;
    double q1 = g_charge[n1];
    double s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5 * q1 * q1; // self energy part
    double D_real = -q1 * charge_para.two_alpha_over_sqrt_pi; // self energy part

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double q2 = g_charge[n2];
      double qq = q1 * q2;
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;

      double erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      s_pe += 0.5 * qq * erfc_r;
      double f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 *= -0.5 * K_C_SP * qq * d12inv * d12inv;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      double f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_sxy;
    g_virial[n1 + 2 * N] += s_sxz;
    g_virial[n1 + 3 * N] += s_syx;
    g_virial[n1 + 4 * N] += s_syy;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_szx;
    g_virial[n1 + 7 * N] += s_szy;
    g_virial[n1 + 8 * N] += s_szz;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

void find_dftd3_coordination_number(
  NEP::DFTD3& dftd3,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int z1 = dftd3.atomic_number[g_type[n1]];
    double R_cov_1 = dftd3para::Bohr * dftd3para::covalent_radius[z1];
    double cn_temp = 0.0;
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int z2 = dftd3.atomic_number[g_type[n2]];
      double R_cov_2 = dftd3para::Bohr * dftd3para::covalent_radius[z2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      cn_temp += 1.0 / (exp(-16.0 * ((R_cov_1 + R_cov_2) / d12 - 1.0)) + 1.0);
    }
    dftd3.cn[n1] = cn_temp;
  }
}

void add_dftd3_force(
  NEP::DFTD3& dftd3,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_potential,
  double* g_force,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int z1 = dftd3.atomic_number[g_type[n1]];
    int num_cn_1 = dftd3para::num_cn[z1];
    double dc6_sum = 0.0;
    double dc8_sum = 0.0;
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int z2 = dftd3.atomic_number[g_type[n2]];
      int z_small = z1, z_large = z2;
      if (z1 > z2) {
        z_small = z2;
        z_large = z1;
      }
      int z12 = z_small * dftd3para::max_elem - (z_small * (z_small - 1)) / 2 + (z_large - z_small);
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double d12_4 = d12_2 * d12_2;
      double d12_6 = d12_4 * d12_2;
      double d12_8 = d12_6 * d12_2;
      double c6 = 0.0;
      double dc6 = 0.0;
      int num_cn_2 = dftd3para::num_cn[z2];
      if (num_cn_1 == 1 && num_cn_2 == 1) {
        c6 = dftd3para::c6_ref[z12 * dftd3para::max_cn2];
      } else {
        double W = 0.0;
        double dW = 0.0;
        double Z = 0.0;
        double dZ = 0.0;
        for (int i = 0; i < num_cn_1; ++i) {
          for (int j = 0; j < num_cn_2; ++j) {
            double diff_i = dftd3.cn[n1] - dftd3para::cn_ref[z1 * dftd3para::max_cn + i];
            double diff_j = dftd3.cn[n2] - dftd3para::cn_ref[z2 * dftd3para::max_cn + j];
            double L_ij = exp(-4.0 * (diff_i * diff_i + diff_j * diff_j));
            W += L_ij;
            dW += L_ij * (-8.0 * diff_i);
            double c6_ref_ij =
              (z1 < z2) ? dftd3para::c6_ref[z12 * dftd3para::max_cn2 + i * dftd3para::max_cn + j]
                        : dftd3para::c6_ref[z12 * dftd3para::max_cn2 + j * dftd3para::max_cn + i];
            Z += c6_ref_ij * L_ij;
            dZ += c6_ref_ij * L_ij * (-8.0 * diff_i);
          }
        }
        if (W < 1.0e-30) {
          int i = num_cn_1 - 1;
          int j = num_cn_2 - 1;
          c6 = (z1 < z2) ? dftd3para::c6_ref[z12 * dftd3para::max_cn2 + i * dftd3para::max_cn + j]
                         : dftd3para::c6_ref[z12 * dftd3para::max_cn2 + j * dftd3para::max_cn + i];
        } else {
          W = 1.0 / W;
          c6 = Z * W;
          dc6 = dZ * W - c6 * dW * W;
        }
      }

      c6 *= dftd3para::HartreeBohr6;
      dc6 *= dftd3para::HartreeBohr6;
      double c8_over_c6 = 3.0 * dftd3para::r2r4[z1] * dftd3para::r2r4[z2] * dftd3para::Bohr2;
      double c8 = c6 * c8_over_c6;
      double damp = dftd3.a1 * sqrt(c8_over_c6) + dftd3.a2;
      double damp_2 = damp * damp;
      double damp_4 = damp_2 * damp_2;
      double damp_6 = 1.0 / (d12_6 + damp_4 * damp_2);
      double damp_8 = 1.0 / (d12_8 + damp_4 * damp_4);
      g_potential[n1] -= (dftd3.s6 * c6 * damp_6 + dftd3.s8 * c8 * damp_8) * 0.5;
      double f2 = dftd3.s6 * c6 * 3.0 * d12_4 * (damp_6 * damp_6) +
                  dftd3.s8 * c8 * 4.0 * d12_6 * (damp_8 * damp_8);
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1 + 0 * N] += f12[0];
      g_force[n1 + 1 * N] += f12[1];
      g_force[n1 + 2 * N] += f12[2];
      g_force[n2 + 0 * N] -= f12[0];
      g_force[n2 + 1 * N] -= f12[1];
      g_force[n2 + 2 * N] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
      dc6_sum += dc6 * dftd3.s6 * damp_6;
      dc8_sum += dc6 * c8_over_c6 * dftd3.s8 * damp_8;
    }
    dftd3.dc6_sum[n1] = dc6_sum;
    dftd3.dc8_sum[n1] = dc8_sum;
  }
}

void add_dftd3_force_extra(
  const NEP::DFTD3& dftd3,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x12,
  const double* g_y12,
  const double* g_z12,
  double* g_force,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int z1 = dftd3.atomic_number[g_type[n1]];
    double R_cov_1 = dftd3para::Bohr * dftd3para::covalent_radius[z1];
    double dc6_sum = dftd3.dc6_sum[n1];
    double dc8_sum = dftd3.dc8_sum[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int z2 = dftd3.atomic_number[g_type[n2]];
      double R_cov_2 = dftd3para::Bohr * dftd3para::covalent_radius[z2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double d12 = sqrt(d12_2);
      double cn_exp_factor = exp(-16.0 * ((R_cov_1 + R_cov_2) / d12 - 1.0));
      double f2 = cn_exp_factor * 16.0 * (R_cov_1 + R_cov_2) * (dc6_sum + dc8_sum); // not 8.0
      f2 /= (cn_exp_factor + 1.0) * (cn_exp_factor + 1.0) * d12 * d12_2;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1 + 0 * N] += f12[0];
      g_force[n1 + 1 * N] += f12[1];
      g_force[n1 + 2 * N] += f12[2];
      g_force[n2 + 0 * N] -= f12[0];
      g_force[n2 + 1 * N] -= f12[1];
      g_force[n2 + 2 * N] -= f12[2];
      g_virial[n2 + 0 * N] -= r12[0] * f12[0];
      g_virial[n2 + 1 * N] -= r12[0] * f12[1];
      g_virial[n2 + 2 * N] -= r12[0] * f12[2];
      g_virial[n2 + 3 * N] -= r12[1] * f12[0];
      g_virial[n2 + 4 * N] -= r12[1] * f12[1];
      g_virial[n2 + 5 * N] -= r12[1] * f12[2];
      g_virial[n2 + 6 * N] -= r12[2] * f12[0];
      g_virial[n2 + 7 * N] -= r12[2] * f12[1];
      g_virial[n2 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

void find_descriptor_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_radial,
  const double* g_gn_angular,
#endif
  double* g_Fp,
  double* g_sum_fxyz,
  double& g_total_potential,
  double* g_potential)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      double fc12;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
#endif
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
        int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
        double rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5;
        double rcinv = 1.0 / rc;

        double r12[3] = {
          g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

        double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        if (d12sq >= rc * rc) {
          continue;
        }
        double d12 = sqrt(d12sq);


#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
        int index_left, index_right;
        double weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * paramb.num_types + t2;
        double gn12 =
          g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
#else
        double fc12;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
#endif
      }
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1] = s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    double F = 0.0, Fp[MAX_DIM] = {0.0}, latent_space[MAX_NEURON] = {0.0};

    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
        latent_space);
    } else {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp,
        latent_space, false, nullptr);
    }

    g_total_potential += F; // always calculate this
    if (g_potential) {      // only calculate when required
      g_potential[n1] += F;
    }

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * nlocal + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

void find_force_radial_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double* g_Fp,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gnp_radial,
#endif
  double** g_force,
  double g_total_virial[6],
  double** g_virial)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 =
          g_gnp_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        double tmp12 = g_Fp[n1 + n * nlocal] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#else
      double fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        double tmp12 = g_Fp[n1 + n * nlocal] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#endif

      g_force[n1][0] += f12[0];
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];

      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial) {                       // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        g_virial[n2][6] -= r12[1] * f12[0]; // yx
        g_virial[n2][7] -= r12[2] * f12[0]; // zx
        g_virial[n2][8] -= r12[2] * f12[1]; // zy
      }
    }
  }
}

void find_force_angular_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double* g_Fp,
  double* g_sum_fxyz,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  const double* g_gn_angular,
  const double* g_gnp_angular,
#endif
  double** g_force,
  double g_total_virial[6],
  double** g_virial)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * nlocal + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * nlocal + n1];
    }

    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      double f12[3] = {0.0};

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        int index_left_all =
          (index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        int index_right_all =
          (index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        double gn12 =
          g_gn_angular[index_left_all] * weight_left + g_gn_angular[index_right_all] * weight_right;
        double gnp12 = g_gnp_angular[index_left_all] * weight_left +
                       g_gnp_angular[index_right_all] * weight_right;
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }
#else
      double fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }
#endif

      g_force[n1][0] += f12[0];
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];
      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial) {                       // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        g_virial[n2][6] -= r12[1] * f12[0]; // yx
        g_virial[n2][7] -= r12[2] * f12[0]; // zx
        g_virial[n2][8] -= r12[2] * f12[1]; // zy
      }
    }
  }
}

void find_force_ZBL_for_lammps(
  NEP::ParaMB& paramb,
  const NEP::ZBL& zbl,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double** g_force,
  double g_total_virial[6],
  double** g_virial,
  double& g_total_potential,
  double* g_potential)
{
  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int type1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    int zi = paramb.atomic_numbers[type1] + 1;
    double pow_zi = pow(double(zi), 0.23);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double max_rc_outer = 2.5;
      if (d12sq >= max_rc_outer * max_rc_outer) {
        continue;
      }
      double d12 = sqrt(d12sq);

      double d12inv = 1.0 / d12;
      double f, fp;
      int type2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      int zj = paramb.atomic_numbers[type2] + 1;
      double a_inv = (pow_zi + pow(double(zj), 0.23)) * 2.134563;
      double zizj = K_C_SP * zi * zj;
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
        double ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        double rc_inner = zbl.rc_inner;
        double rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = std::min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_force[n1][0] += f12[0]; // accumulation here
      g_force[n1][1] += f12[1];
      g_force[n1][2] += f12[2];
      g_force[n2][0] -= f12[0];
      g_force[n2][1] -= f12[1];
      g_force[n2][2] -= f12[2];
      // always calculate the total virial:
      g_total_virial[0] -= r12[0] * f12[0]; // xx
      g_total_virial[1] -= r12[1] * f12[1]; // yy
      g_total_virial[2] -= r12[2] * f12[2]; // zz
      g_total_virial[3] -= r12[0] * f12[1]; // xy
      g_total_virial[4] -= r12[0] * f12[2]; // xz
      g_total_virial[5] -= r12[1] * f12[2]; // yz
      if (g_virial) {                       // only calculate the per-atom virial when required
        g_virial[n2][0] -= r12[0] * f12[0]; // xx
        g_virial[n2][1] -= r12[1] * f12[1]; // yy
        g_virial[n2][2] -= r12[2] * f12[2]; // zz
        g_virial[n2][3] -= r12[0] * f12[1]; // xy
        g_virial[n2][4] -= r12[0] * f12[2]; // xz
        g_virial[n2][5] -= r12[1] * f12[2]; // yz
        g_virial[n2][6] -= r12[1] * f12[0]; // yx
        g_virial[n2][7] -= r12[2] * f12[0]; // zx
        g_virial[n2][8] -= r12[2] * f12[1]; // zy
      }
      g_total_potential += f * 0.5; // always calculate this
      if (g_potential) {            // only calculate when required
        g_potential[n1] += f * 0.5;
      }
    }
  }
}
void find_descriptor_spin_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double** g_spin,
  double* g_Fp,
  double* g_sum_fxyz,
  double& g_total_potential,
  double* g_potential)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  const int spin_dim = nspin * spin_blocks.pair_blocks + spin_pmax;
  const int spin_end = spin_offset + spin_dim;
  if (spin_end > annmb.dim) {
    std::cout << "Spin descriptor block exceeds ANN dim.\n";
    exit(1);
  }

  const SpinCMode spin_c_mode = nep_spin_get_c_mode(
    static_cast<int>(paramb.num_c_spin),
    static_cast<int>(paramb.c_spin_block_stride));

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    double q[MAX_DIM] = {0.0};

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      double rcinv = 1.0 / rc;
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);

      double fc12;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      int bs = paramb.basis_size_radial;
      if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
      find_fn(bs, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
        int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
        double rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5;
        double rcinv = 1.0 / rc;

        double r12[3] = {
          g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

        double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        if (d12sq >= rc * rc) {
          continue;
        }
        double d12 = sqrt(d12sq);

        double fc12;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        int bs = paramb.basis_size_angular;
        if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
        find_fn(bs, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= bs; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1] = s[abc];
      }
    }

    for (int d = spin_offset; d < spin_end; ++d) {
      q[d] = 0.0;
    }

    if (spin_blocks.pair_blocks > 0 || spin_pmax > 0) {
      double si_mag = g_spin[n1][3];
      double si[3] = {g_spin[n1][0] * si_mag, g_spin[n1][1] * si_mag, g_spin[n1][2] * si_mag};
      double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];

      if (si2 > kSpinZeroEpsSph) {
        double si_norm = sqrt(si2);
        for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
          int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
          int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
          double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
          double rcinv = 1.0 / rc;
          double r12[3] = {
            g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

          double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          if (d12sq >= rc * rc) {
            continue;
          }
          double d12 = sqrt(d12sq);
          if (d12 <= 0.0) {
            continue;
          }
          double rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

          double fc12;
          find_fc(rc, rcinv, d12, fc12);
          double fn12[MAX_NUM_N];
          int bs = paramb.basis_size_radial;
          if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
          find_fn(bs, rcinv, d12, fc12, fn12);

          double sj_mag = g_spin[n2][3];
          double sj[3] = {g_spin[n2][0] * sj_mag, g_spin[n2][1] * sj_mag, g_spin[n2][2] * sj_mag};
          double sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
          const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
          double sj_norm = neighbor_has_spin ? sqrt(sj2) : 0.0;

          if (spin_blocks.kmax_ex >= 0 && neighbor_has_spin) {
            const double sdot = nep_spin_dot3(si, sj);
            const double denom = si_norm * sj_norm;
            const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
            double Tk[9] = {0.0};
            nep_spin_fill_Tk(c, spin_blocks.kmax_ex, Tk);
            const double phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);

            for (int n = 0; n < nspin; ++n) {
              double gn_ex[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_ex; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_ex[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_ex; ++kk) {
                int off = spin_offset + kk * nspin;
                q[off + n] += gn_ex[kk] * (phi * Tk[kk]);
              }
            }
          }

          if (spin_blocks.kmax_dmi >= 0 && neighbor_has_spin) {
            const double sdot = nep_spin_dot3(si, sj);
            double sixsj[3];
            nep_spin_cross3(si, sj, sixsj);
            const double dmi_val = nep_spin_dot3(sixsj, rhat);
            const double denom = si_norm * sj_norm;
            const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
            double Tk[9] = {0.0};
            nep_spin_fill_Tk(c, spin_blocks.kmax_dmi, Tk);

            for (int n = 0; n < nspin; ++n) {
              double gn_dmi[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_dmi; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    spin_blocks.dmi_block0 + kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_dmi[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_dmi; ++kk) {
                int off = spin_offset + (spin_blocks.dmi_block0 + kk) * nspin;
                q[off + n] += gn_dmi[kk] * (dmi_val * Tk[kk]);
              }
            }
          }

          if (spin_blocks.kmax_ani >= 0 && neighbor_has_spin) {
            const double si_r = nep_spin_dot3(si, rhat);
            const double sj_r = nep_spin_dot3(sj, rhat);
            const double ani_scalar = si_r * sj_r;
            const double sdot = nep_spin_dot3(si, sj);
            const double denom = si_norm * sj_norm;
            const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
            double Tk[9] = {0.0};
            nep_spin_fill_Tk(c, spin_blocks.kmax_ani, Tk);

            for (int n = 0; n < nspin; ++n) {
              double gn_ani[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_ani; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    spin_blocks.ani_block0 + kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_ani[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_ani; ++kk) {
                int off = spin_offset + (spin_blocks.ani_block0 + kk) * nspin;
                q[off + n] += gn_ani[kk] * (ani_scalar * Tk[kk]);
              }
            }
          }

          if (spin_blocks.kmax_sia >= 0) {
            const double si_r = nep_spin_dot3(si, rhat);
            const double sia_scalar = si_r * si_r;
            double Tk[9] = {0.0};
            Tk[0] = 1.0;
            if (neighbor_has_spin) {
              const double sdot = nep_spin_dot3(si, sj);
              const double denom = si_norm * sj_norm;
              const double c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12));
              nep_spin_fill_Tk(c, spin_blocks.kmax_sia, Tk);
            } else {
              for (int kk = 1; kk <= kSpinMaxPair; ++kk) {
                Tk[kk] = 0.0;
              }
            }

            for (int n = 0; n < nspin; ++n) {
              double gn_sia[9] = {0.0};
              for (int kb = 0; kb <= bs; ++kb) {
                double fn_val = fn12[kb];
                for (int kk = 0; kk <= spin_blocks.kmax_sia; ++kk) {
                  int c_idx = nep_spin_get_c_index(
                    spin_c_mode,
                    static_cast<int>(paramb.c_spin_offset),
                    static_cast<int>(paramb.c_spin_block_stride),
                    static_cast<int>(paramb.num_types_sq),
                    static_cast<int>(paramb.num_types),
                    bs,
                    spin_blocks.sia_block0 + kk,
                    n,
                    kb,
                    t1,
                    t2);
                  gn_sia[kk] += fn_val * annmb.c[c_idx];
                }
              }
              for (int kk = 0; kk <= spin_blocks.kmax_sia; ++kk) {
                if (kk == 0 || neighbor_has_spin) {
                  int off = spin_offset + (spin_blocks.sia_block0 + kk) * nspin;
                  q[off + n] += gn_sia[kk] * (sia_scalar * Tk[kk]);
                }
              }
            }
          }
        }
      }

      if (spin_pmax > 0) {
        const int onsite_offset = spin_offset + nspin * spin_blocks.pair_blocks;
        double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
        if (si2 <= kSpinZeroEpsSph) {
          for (int p = 1; p <= spin_pmax; ++p) {
            q[onsite_offset + (p - 1)] = 0.0;
          }
        } else if (paramb.spin_onsite_basis_mode == 0) {
          double m2 = si2;
          double m2p = m2;
          for (int p = 1; p <= spin_pmax; ++p) {
            q[onsite_offset + (p - 1)] = m2p;
            m2p *= m2;
          }
        } else {
          double y = si2;
          double yref = paramb.spin_mref;
          if (paramb.spin_onsite_basis_mode == 2) {
            y = sqrt(si2);
          } else {
            yref = paramb.spin_mref * paramb.spin_mref;
          }
          if (yref <= 0.0) yref = 1.0;
          double x = (y - yref) / (y + yref + 1.0e-12);
          x = std::max(-1.0, std::min(1.0, x));

          double Tp[9] = {0.0};
          Tp[0] = 1.0;
          if (spin_pmax >= 1) Tp[1] = x;
          for (int p = 2; p <= spin_pmax; ++p) {
            Tp[p] = 2.0 * x * Tp[p - 1] - Tp[p - 2];
          }
          for (int p = 1; p <= spin_pmax; ++p) {
            q[onsite_offset + (p - 1)] = Tp[p];
          }
        }
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    double F = 0.0;
    double Fp[MAX_DIM] = {0.0};
    double latent_space[MAX_NEURON] = {0.0};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F,
      Fp, latent_space, false, nullptr);

    g_total_potential += F; // always calculate this
    if (g_potential) {      // only calculate when required
      g_potential[n1] += F;
    }

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * nlocal + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

void find_force_radial_spin_spherical_fused_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double** g_spin,
  double* g_Fp,
  double** g_force,
  double g_total_virial[6],
  double** g_virial)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  if (spin_blocks.pair_blocks == 0) {
    return;
  }
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const SpinCMode spin_c_mode = nep_spin_get_c_mode(
    static_cast<int>(paramb.num_c_spin),
    static_cast<int>(paramb.c_spin_block_stride));
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    double si_mag = g_spin[n1][3];
    double si[3] = {g_spin[n1][0] * si_mag, g_spin[n1][1] * si_mag, g_spin[n1][2] * si_mag};
    double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) {
      continue;
    }
    const double si_norm = sqrt(si2);

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};
      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      if (d12 <= 0.0) {
        continue;
      }
      double rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

      double rcinv = 1.0 / rc;
      double fc12, dfc12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, dfc12);
      double fn12[MAX_NUM_N];
      double dfn12[MAX_NUM_N];
      find_fn_and_fnp(bs, rcinv, d12, fc12, dfc12, fn12, dfn12);

      double sj_mag = g_spin[n2][3];
      double sj[3] = {g_spin[n2][0] * sj_mag, g_spin[n2][1] * sj_mag, g_spin[n2][2] * sj_mag};
      double sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
      const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
      double sj_norm = neighbor_has_spin ? sqrt(sj2) : 0.0;
      double sdot = neighbor_has_spin ? nep_spin_dot3(si, sj) : 0.0;
      double denom = neighbor_has_spin ? (si_norm * sj_norm) : 0.0;
      double c = neighbor_has_spin ? nep_spin_clamp_unit(sdot / (denom + 1.0e-12)) : 0.0;

      double fvec[3] = {0.0, 0.0, 0.0};

      if (spin_blocks.kmax_ex >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        nep_spin_fill_Tk(c, spin_blocks.kmax_ex, Tk);
        const double phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);
        double force_mag = 0.0;
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ex; ++k) {
            const double fp_val = g_Fp[(spin_offset + k * nspin + n) * nlocal + n1];
            double dC_dr = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                k,
                n,
                kb,
                t1,
                t2);
              dC_dr += dfn12[kb] * annmb.c[c_idx];
            }
            force_mag += fp_val * dC_dr * (phi * Tk[k]);
          }
        }
        fvec[0] += force_mag * rhat[0];
        fvec[1] += force_mag * rhat[1];
        fvec[2] += force_mag * rhat[2];
      }

      if (spin_blocks.kmax_dmi >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        nep_spin_fill_Tk(c, spin_blocks.kmax_dmi, Tk);
        double sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        const double dmi_val = nep_spin_dot3(sixsj, rhat);
        double fvec_dmi[3] = {0.0, 0.0, 0.0};
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_dmi; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.dmi_block0 + k) * nspin + n) * nlocal + n1];
            double dC_dr = 0.0;
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.dmi_block0 + k,
                n,
                kb,
                t1,
                t2);
              const double coeff = annmb.c[c_idx];
              dC_dr += dfn12[kb] * coeff;
              C_val += fn12[kb] * coeff;
            }
            const double term1 = (dC_dr * dmi_val - C_val * dmi_val / d12) * Tk[k];
            const double term2 = (C_val * Tk[k] / d12);
            fvec_dmi[0] += fp_val * (term1 * rhat[0] + term2 * sixsj[0]);
            fvec_dmi[1] += fp_val * (term1 * rhat[1] + term2 * sixsj[1]);
            fvec_dmi[2] += fp_val * (term1 * rhat[2] + term2 * sixsj[2]);
          }
        }
        fvec[0] += fvec_dmi[0];
        fvec[1] += fvec_dmi[1];
        fvec[2] += fvec_dmi[2];
      }

      if (spin_blocks.kmax_ani >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        nep_spin_fill_Tk(c, spin_blocks.kmax_ani, Tk);
        const double si_r = nep_spin_dot3(si, rhat);
        const double sj_r = nep_spin_dot3(sj, rhat);
        const double ani_scalar = si_r * sj_r;
        double fvec_ani[3] = {0.0, 0.0, 0.0};
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ani; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.ani_block0 + k) * nspin + n) * nlocal + n1];
            double dC_dr = 0.0;
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.ani_block0 + k,
                n,
                kb,
                t1,
                t2);
              const double coeff = annmb.c[c_idx];
              dC_dr += dfn12[kb] * coeff;
              C_val += fn12[kb] * coeff;
            }
            const double term1 = (dC_dr * ani_scalar - 2.0 * C_val * ani_scalar / d12) * Tk[k];
            const double term2 = (C_val * Tk[k] / d12);
            fvec_ani[0] += fp_val * (term1 * rhat[0] + term2 * (sj_r * si[0] + si_r * sj[0]));
            fvec_ani[1] += fp_val * (term1 * rhat[1] + term2 * (sj_r * si[1] + si_r * sj[1]));
            fvec_ani[2] += fp_val * (term1 * rhat[2] + term2 * (sj_r * si[2] + si_r * sj[2]));
          }
        }
        fvec[0] += fvec_ani[0];
        fvec[1] += fvec_ani[1];
        fvec[2] += fvec_ani[2];
      }

      if (spin_blocks.kmax_sia >= 0) {
        double Tk[9] = {0.0};
        Tk[0] = 1.0;
        if (neighbor_has_spin) {
          nep_spin_fill_Tk(c, spin_blocks.kmax_sia, Tk);
        } else {
          for (int kk = 1; kk <= kSpinMaxPair; ++kk) {
            Tk[kk] = 0.0;
          }
        }
        const double si_r = nep_spin_dot3(si, rhat);
        const double sia_scalar = si_r * si_r;
        double fvec_sia[3] = {0.0, 0.0, 0.0};
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_sia; ++k) {
            if (k > 0 && !neighbor_has_spin) {
              continue;
            }
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.sia_block0 + k) * nspin + n) * nlocal + n1];
            double dC_dr = 0.0;
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.sia_block0 + k,
                n,
                kb,
                t1,
                t2);
              const double coeff = annmb.c[c_idx];
              dC_dr += dfn12[kb] * coeff;
              C_val += fn12[kb] * coeff;
            }
            const double term1 = (dC_dr * sia_scalar - 2.0 * C_val * sia_scalar / d12) * Tk[k];
            const double term2 = (2.0 * C_val * Tk[k] * si_r / d12);
            fvec_sia[0] += fp_val * (term1 * rhat[0] + term2 * si[0]);
            fvec_sia[1] += fp_val * (term1 * rhat[1] + term2 * si[1]);
            fvec_sia[2] += fp_val * (term1 * rhat[2] + term2 * si[2]);
          }
        }
        fvec[0] += fvec_sia[0];
        fvec[1] += fvec_sia[1];
        fvec[2] += fvec_sia[2];
      }

      g_force[n1][0] += fvec[0];
      g_force[n1][1] += fvec[1];
      g_force[n1][2] += fvec[2];
      g_force[n2][0] -= fvec[0];
      g_force[n2][1] -= fvec[1];
      g_force[n2][2] -= fvec[2];

      g_total_virial[0] -= r12[0] * fvec[0]; // xx
      g_total_virial[1] -= r12[1] * fvec[1]; // yy
      g_total_virial[2] -= r12[2] * fvec[2]; // zz
      g_total_virial[3] -= r12[0] * fvec[1]; // xy
      g_total_virial[4] -= r12[0] * fvec[2]; // xz
      g_total_virial[5] -= r12[1] * fvec[2]; // yz
      if (g_virial) {
        g_virial[n2][0] -= r12[0] * fvec[0]; // xx
        g_virial[n2][1] -= r12[1] * fvec[1]; // yy
        g_virial[n2][2] -= r12[2] * fvec[2]; // zz
        g_virial[n2][3] -= r12[0] * fvec[1]; // xy
        g_virial[n2][4] -= r12[0] * fvec[2]; // xz
        g_virial[n2][5] -= r12[1] * fvec[2]; // yz
        g_virial[n2][6] -= r12[1] * fvec[0]; // yx
        g_virial[n2][7] -= r12[2] * fvec[0]; // zx
        g_virial[n2][8] -= r12[2] * fvec[1]; // zy
      }
    }
  }
}

void find_mforce_radial_spin_spherical_fused_for_lammps(
  NEP::ParaMB& paramb,
  NEP::ANN& annmb,
  int nlocal,
  int N,
  int* g_ilist,
  int* g_NN,
  int** g_NL,
  int* g_type,
  int* type_map,
  double** g_pos,
  double** g_spin,
  double* g_Fp,
  double** g_mforce)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  if (spin_blocks.pair_blocks == 0) {
    return;
  }
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const SpinCMode spin_c_mode = nep_spin_get_c_mode(
    static_cast<int>(paramb.num_c_spin),
    static_cast<int>(paramb.c_spin_block_stride));
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  const double msign = paramb.mforce_sign;

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    int t1 = type_map[g_type[n1]]; // from LAMMPS to NEP convention
    double si_mag = g_spin[n1][3];
    double si[3] = {g_spin[n1][0] * si_mag, g_spin[n1][1] * si_mag, g_spin[n1][2] * si_mag};
    double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) {
      continue;
    }
    const double si_norm = sqrt(si2);
    const double inv_si = 1.0 / (si_norm + 1.0e-12);

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = lammps_unpack_neigh_index(g_NL[n1][i1]);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};
      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      double rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5;
      if (d12sq >= rc * rc) {
        continue;
      }
      double d12 = sqrt(d12sq);
      if (d12 <= 0.0) {
        continue;
      }
      double rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

      double rcinv = 1.0 / rc;
      double fc12;
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(bs, rcinv, d12, fc12, fn12);

      double sj_mag = g_spin[n2][3];
      double sj[3] = {g_spin[n2][0] * sj_mag, g_spin[n2][1] * sj_mag, g_spin[n2][2] * sj_mag};
      double sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
      const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
      const double sj_norm = neighbor_has_spin ? sqrt(sj2) : 0.0;
      const double inv_sj = neighbor_has_spin ? (1.0 / (sj_norm + 1.0e-12)) : 0.0;
      const double sdot = neighbor_has_spin ? nep_spin_dot3(si, sj) : 0.0;
      const double denom = neighbor_has_spin ? (si_norm * sj_norm) : 0.0;
      const double c =
        neighbor_has_spin ? nep_spin_clamp_unit(sdot / (denom + 1.0e-12)) : 0.0;

      double mvec_i[3] = {0.0, 0.0, 0.0};
      double mvec_j[3] = {0.0, 0.0, 0.0};

      if (spin_blocks.kmax_ex >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_ex, Tk, Uk);
        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ex; ++k) {
            const double fp_val = g_Fp[(spin_offset + k * nspin + n) * nlocal + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            double term_i_si = 0.0;
            double term_i_sj = 0.0;
            double term_j_si = 0.0;
            double term_j_sj = 0.0;

            if (paramb.spin_ex_phi_mode == 0) {
              const double ratio = sj_norm * inv_si;
              term_i_si = (Tk[k] - c * Uk[k]) * ratio;
              term_i_sj = Uk[k];
              const double ratio_j = si_norm * inv_sj;
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
              const double inv_denom = 1.0 / (denom + 1.0e-12);
              const double ratio_i = sj_norm * inv_si;
              term_i_si = -c * Uk[k] * ratio_i * inv_denom;
              term_i_sj = Uk[k] * inv_denom;
              const double ratio_j = si_norm * inv_sj;
              term_j_sj = -c * Uk[k] * ratio_j * inv_denom;
              term_j_si = Uk[k] * inv_denom;
            }

            const double pre = fp_val * C_val;
            mvec_i[0] += pre * (term_i_si * si[0] + term_i_sj * sj[0]);
            mvec_i[1] += pre * (term_i_si * si[1] + term_i_sj * sj[1]);
            mvec_i[2] += pre * (term_i_si * si[2] + term_i_sj * sj[2]);
            mvec_j[0] += pre * (term_j_si * si[0] + term_j_sj * sj[0]);
            mvec_j[1] += pre * (term_j_si * si[1] + term_j_sj * sj[1]);
            mvec_j[2] += pre * (term_j_si * si[2] + term_j_sj * sj[2]);
          }
        }
      }

      if (spin_blocks.kmax_dmi >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_dmi, Tk, Uk);
        double dc_dsi[3] = {0.0, 0.0, 0.0};
        double dc_dsj[3] = {0.0, 0.0, 0.0};
        nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
        double sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        const double dmi_val = nep_spin_dot3(sixsj, rhat);
        double dDMI_dsi[3] = {
          sj[1] * rhat[2] - sj[2] * rhat[1],
          sj[2] * rhat[0] - sj[0] * rhat[2],
          sj[0] * rhat[1] - sj[1] * rhat[0]};
        double dDMI_dsj[3] = {
          rhat[1] * si[2] - rhat[2] * si[1],
          rhat[2] * si[0] - rhat[0] * si[2],
          rhat[0] * si[1] - rhat[1] * si[0]};

        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_dmi; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.dmi_block0 + k) * nspin + n) * nlocal + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.dmi_block0 + k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            const double term1 = C_val * Tk[k];
            const double term2 = C_val * dmi_val * Uk[k];
            for (int d = 0; d < 3; ++d) {
              mvec_i[d] += fp_val * (term1 * dDMI_dsi[d] + term2 * dc_dsi[d]);
              mvec_j[d] += fp_val * (term1 * dDMI_dsj[d] + term2 * dc_dsj[d]);
            }
          }
        }
      }

      if (spin_blocks.kmax_ani >= 0 && neighbor_has_spin) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_ani, Tk, Uk);
        double dc_dsi[3] = {0.0, 0.0, 0.0};
        double dc_dsj[3] = {0.0, 0.0, 0.0};
        nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
        const double si_r = nep_spin_dot3(si, rhat);
        const double sj_r = nep_spin_dot3(sj, rhat);
        const double ani_scalar = si_r * sj_r;

        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_ani; ++k) {
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.ani_block0 + k) * nspin + n) * nlocal + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.ani_block0 + k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            const double term1 = C_val * ani_scalar * Uk[k];
            const double term2 = C_val * Tk[k];
            for (int d = 0; d < 3; ++d) {
              mvec_i[d] += fp_val * (term1 * dc_dsi[d] + term2 * rhat[d] * sj_r);
              mvec_j[d] += fp_val * (term1 * dc_dsj[d] + term2 * rhat[d] * si_r);
            }
          }
        }
      }

      if (spin_blocks.kmax_sia >= 0) {
        double Tk[9] = {0.0};
        double Uk[9] = {0.0};
        double dc_dsi[3] = {0.0, 0.0, 0.0};
        double dc_dsj[3] = {0.0, 0.0, 0.0};
        Tk[0] = 1.0;
        if (neighbor_has_spin) {
          nep_spin_fill_Tk_and_dTk(c, spin_blocks.kmax_sia, Tk, Uk);
          nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
        }
        const double si_r = nep_spin_dot3(si, rhat);
        const double sia_scalar = si_r * si_r;

        for (int n = 0; n < nspin; ++n) {
          for (int k = 0; k <= spin_blocks.kmax_sia; ++k) {
            if (k > 0 && !neighbor_has_spin) {
              continue;
            }
            const double fp_val =
              g_Fp[(spin_offset + (spin_blocks.sia_block0 + k) * nspin + n) * nlocal + n1];
            double C_val = 0.0;
            for (int kb = 0; kb <= bs; ++kb) {
              int c_idx = nep_spin_get_c_index(
                spin_c_mode,
                static_cast<int>(paramb.c_spin_offset),
                static_cast<int>(paramb.c_spin_block_stride),
                static_cast<int>(paramb.num_types_sq),
                static_cast<int>(paramb.num_types),
                bs,
                spin_blocks.sia_block0 + k,
                n,
                kb,
                t1,
                t2);
              C_val += fn12[kb] * annmb.c[c_idx];
            }

            const double term1 = C_val * sia_scalar * Uk[k];
            const double term2 = C_val * Tk[k] * 2.0 * si_r;
            for (int d = 0; d < 3; ++d) {
              mvec_i[d] += fp_val * (term1 * dc_dsi[d] + term2 * rhat[d]);
              mvec_j[d] += fp_val * (term1 * dc_dsj[d]);
            }
          }
        }
      }

      g_mforce[n1][0] += msign * mvec_i[0];
      g_mforce[n1][1] += msign * mvec_i[1];
      g_mforce[n1][2] += msign * mvec_i[2];
      g_mforce[n2][0] += msign * mvec_j[0];
      g_mforce[n2][1] += msign * mvec_j[1];
      g_mforce[n2][2] += msign * mvec_j[2];
    }
  }
}

void find_mforce_radial_spin_spherical_onsite_for_lammps(
  NEP::ParaMB& paramb,
  int nlocal,
  int N,
  int* g_ilist,
  double** g_spin,
  double* g_Fp,
  double** g_mforce)
{
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  if (spin_pmax <= 0) return;
  const int nspin = paramb.spin_n_max + 1;
  const int spin_offset = (paramb.n_max_radial + 1) + paramb.dim_angular;
  const int onsite_offset = spin_offset + nspin * spin_blocks.pair_blocks;
  const double msign = paramb.mforce_sign;

  for (int ii = 0; ii < N; ++ii) {
    int n1 = g_ilist[ii];
    double si_mag = g_spin[n1][3];
    double si[3] = {g_spin[n1][0] * si_mag, g_spin[n1][1] * si_mag, g_spin[n1][2] * si_mag};
    const double si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
    if (si2 <= kSpinZeroEpsSph) {
      continue;
    }

    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;

    if (paramb.spin_onsite_basis_mode == 0) {
      const double m2 = si2;
      double m2pow = 1.0;
      for (int p = 1; p <= spin_pmax; ++p) {
        const double Fp_p = g_Fp[(onsite_offset + (p - 1)) * nlocal + n1];
        const double coeff = msign * Fp_p * (2.0 * p) * m2pow;
        mx += coeff * si[0];
        my += coeff * si[1];
        mz += coeff * si[2];
        m2pow *= m2;
      }
    } else {
      double y = si2;
      double yref = paramb.spin_mref * paramb.spin_mref;
      const double si_norm = sqrt(si2);
      const double inv_si_norm = 1.0 / (si_norm + 1.0e-12);
      if (paramb.spin_onsite_basis_mode == 2) {
        y = si_norm;
        yref = paramb.spin_mref;
      }
      if (yref <= 0.0) yref = 1.0;

      const double denom = y + yref;
      const double inv_denom = 1.0 / (denom + 1.0e-12);
      double x = (y - yref) * inv_denom;
      x = std::max(-1.0, std::min(1.0, x));
      const double dx_dy = (2.0 * yref) * inv_denom * inv_denom;

      double Tp[9] = {0.0};
      double dTp[9] = {0.0};
      Tp[0] = 1.0;
      dTp[0] = 0.0;
      if (spin_pmax >= 1) {
        Tp[1] = x;
        dTp[1] = 1.0;
      }
      for (int p = 2; p <= spin_pmax; ++p) {
        Tp[p] = 2.0 * x * Tp[p - 1] - Tp[p - 2];
        dTp[p] = 2.0 * Tp[p - 1] + 2.0 * x * dTp[p - 1] - dTp[p - 2];
      }

      double dy_dsi[3];
      if (paramb.spin_onsite_basis_mode == 2) {
        dy_dsi[0] = inv_si_norm * si[0];
        dy_dsi[1] = inv_si_norm * si[1];
        dy_dsi[2] = inv_si_norm * si[2];
      } else {
        dy_dsi[0] = 2.0 * si[0];
        dy_dsi[1] = 2.0 * si[1];
        dy_dsi[2] = 2.0 * si[2];
      }

      for (int p = 1; p <= spin_pmax; ++p) {
        const double Fp_p = g_Fp[(onsite_offset + (p - 1)) * nlocal + n1];
        const double coeff = msign * Fp_p * dTp[p] * dx_dy;
        mx += coeff * dy_dsi[0];
        my += coeff * dy_dsi[1];
        mz += coeff * dy_dsi[2];
      }
    }

    g_mforce[n1][0] += mx;
    g_mforce[n1][1] += my;
    g_mforce[n1][2] += mz;
  }
}


std::vector<std::string> get_tokens(std::ifstream& input)
{
  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

void print_tokens(const std::vector<std::string>& tokens)
{
  std::cout << "Line:";
  for (const auto& token : tokens) {
    std::cout << " " << token;
  }
  std::cout << std::endl;
}

int get_int_from_token(const std::string& token, const char* filename, const int line)
{
  int value = 0;
  try {
    value = std::stoi(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

double get_double_from_token(const std::string& token, const char* filename, const int line)
{
  double value = 0;
  try {
    value = std::stod(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

} // namespace

NEP::NEP() {}

NEP::NEP(const std::string& potential_filename) { init_from_file(potential_filename, true); }

void NEP::init_from_file(const std::string& potential_filename, const bool is_rank_0)
{
  std::ifstream input(potential_filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << potential_filename << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  bool is_spin_model = false;
  if (tokens[0] == "nep3") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep3_dipole") {
    paramb.model_type = 1;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_polarizability") {
    paramb.model_type = 2;
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_dipole") {
    paramb.model_type = 1;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_polarizability") {
    paramb.model_type = 2;
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5") {
    paramb.model_type = 0;
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.model_type = 0;
    paramb.version = 5;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_charge1") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_zbl_charge1") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_charge2") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_charge3") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    paramb.charge_mode = 3;
  } else if (tokens[0] == "nep4_zbl_charge3") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = true;
    paramb.charge_mode = 3;
  } else if (tokens[0] == "nep3_spin") {
    paramb.model_type = 0;
    paramb.version = 3;
    zbl.enabled = false;
    is_spin_model = true;
  } else if (tokens[0] == "nep4_spin") {
    paramb.model_type = 0;
    paramb.version = 4;
    zbl.enabled = false;
    is_spin_model = true;
  } else {
    std::cout << tokens[0] << " is an unsupported NEP model." << std::endl;
    exit(1);
  }

  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  element_list.resize(paramb.num_types);
  for (std::size_t n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    element_list[n] = tokens[2 + n];
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m;
        break;
      }
    }
    paramb.atomic_numbers[n] = atomic_number;
    dftd3.atomic_number[n] = atomic_number;
  }

  if (is_spin_model) {
    int spin_header_lines = 1;
    tokens = get_tokens(input);
    if ((tokens.size() != 2 && tokens.size() != 3) || tokens[0] != "spin_mode") {
      print_tokens(tokens);
      std::cout << "Second line of nep*_spin must be spin_mode <mode> [spin_header_lines]." << std::endl;
      exit(1);
    }
    paramb.spin_mode = get_int_from_token(tokens[1], __FILE__, __LINE__);
    if (tokens.size() == 3) {
      spin_header_lines = get_int_from_token(tokens[2], __FILE__, __LINE__);
    }
    if (paramb.spin_mode <= 0) {
      std::cout << "spin_mode should be positive for spin models." << std::endl;
      exit(1);
    }
    if (spin_header_lines != 1 && spin_header_lines != 2) {
      std::cout << "spin_header_lines must be 1 or 2 for nep*_spin." << std::endl;
      exit(1);
    }

    tokens = get_tokens(input);
    if (tokens.size() < 5 || tokens.size() > 9 || tokens[0] != "spin_feature") {
      print_tokens(tokens);
      std::cout << "Third line of nep*_spin must be spin_feature <kmax_ex> <kmax_dmi> <kmax_ani> <kmax_sia> "
                   "[pmax] [ex_phi_mode] [onsite_basis_mode] [mref]." << std::endl;
      exit(1);
    }

    paramb.spin_kmax_ex = get_int_from_token(tokens[1], __FILE__, __LINE__);
    paramb.spin_kmax_dmi = get_int_from_token(tokens[2], __FILE__, __LINE__);
    paramb.spin_kmax_ani = get_int_from_token(tokens[3], __FILE__, __LINE__);
    paramb.spin_kmax_sia = get_int_from_token(tokens[4], __FILE__, __LINE__);

    auto check_kmax = [&](int kmax, const char* name) {
      if (kmax < -1 || kmax > 8) {
        std::cout << "Invalid " << name << " in spin_feature (must be in [-1,8])." << std::endl;
        exit(1);
      }
    };
    check_kmax(paramb.spin_kmax_ex, "kmax_ex");
    check_kmax(paramb.spin_kmax_dmi, "kmax_dmi");
    check_kmax(paramb.spin_kmax_ani, "kmax_ani");
    check_kmax(paramb.spin_kmax_sia, "kmax_sia");

    if (tokens.size() >= 6) {
      paramb.spin_pmax = get_int_from_token(tokens[5], __FILE__, __LINE__);
    } else {
      paramb.spin_pmax = 0;
    }
    if (paramb.spin_pmax < 0 || paramb.spin_pmax > 8) {
      std::cout << "spin_feature pmax must be in [0,8]." << std::endl;
      exit(1);
    }

    if (tokens.size() >= 7) {
      paramb.spin_ex_phi_mode = get_int_from_token(tokens[6], __FILE__, __LINE__);
    } else {
      paramb.spin_ex_phi_mode = 0;
    }
    if (paramb.spin_ex_phi_mode < 0 || paramb.spin_ex_phi_mode > 3) {
      std::cout << "spin_feature ex_phi_mode must be in [0,3]." << std::endl;
      exit(1);
    }

    if (tokens.size() >= 8) {
      paramb.spin_onsite_basis_mode = get_int_from_token(tokens[7], __FILE__, __LINE__);
    } else {
      paramb.spin_onsite_basis_mode = 0;
    }
    if (paramb.spin_onsite_basis_mode < 0 || paramb.spin_onsite_basis_mode > 2) {
      std::cout << "spin_feature onsite_basis_mode must be in [0,2]." << std::endl;
      exit(1);
    }

    if (tokens.size() == 9) {
      paramb.spin_mref = get_double_from_token(tokens[8], __FILE__, __LINE__);
      if (!(paramb.spin_mref > 0.0)) {
        std::cout << "spin_feature mref must be > 0." << std::endl;
        exit(1);
      }
    } else {
      paramb.spin_mref = 1.0;
    }

    paramb.spin_blocks = nep_spin_blocks_from_kmax(paramb.spin_kmax_ex) +
                         nep_spin_blocks_from_kmax(paramb.spin_kmax_dmi) +
                         nep_spin_blocks_from_kmax(paramb.spin_kmax_ani) +
                         nep_spin_blocks_from_kmax(paramb.spin_kmax_sia);
    if (paramb.spin_blocks == 0 && paramb.spin_pmax == 0) {
      std::cout << "spin_mode>0 requires at least one spin block (kmax>=0) or spin_pmax>0."
                << std::endl;
      exit(1);
    }

    if (spin_header_lines == 2) {
      tokens = get_tokens(input);
      if (tokens.size() != 2 || tokens[0] != "spin_n_max") {
        print_tokens(tokens);
        std::cout << "Fourth line of nep*_spin must be spin_n_max <spin_n_max>." << std::endl;
        exit(1);
      }
      paramb.spin_n_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
    }
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      print_tokens(tokens);
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
      }
    }
  }

  // cutoff
  tokens = get_tokens(input);
  if (is_spin_model) {
    if (tokens.size() != 5 && tokens.size() != 8) {
      print_tokens(tokens);
      std::cout << "cutoff should have 4 or 7 parameters for spin models.\n";
      exit(1);
    }
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (std::size_t n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
    if (tokens.size() == 8) {
      paramb.typewise_cutoff_radial_factor = get_double_from_token(tokens[5], __FILE__, __LINE__);
      paramb.typewise_cutoff_angular_factor = get_double_from_token(tokens[6], __FILE__, __LINE__);
      paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[7], __FILE__, __LINE__);
      paramb.use_typewise_cutoff = (paramb.typewise_cutoff_radial_factor > 0.0);
      paramb.use_typewise_cutoff_zbl = (paramb.typewise_cutoff_zbl_factor > 0.0);
    }
  } else {
    if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
      print_tokens(tokens);
      std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
      exit(1);
    }
    if (tokens.size() == 5) {
      paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
      paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
      for (std::size_t n = 0; n < paramb.num_types; ++n) {
        paramb.rc_radial[n] = paramb.rc_radial[0];
        paramb.rc_angular[n] = paramb.rc_angular[0];
      }
    } else {
      for (std::size_t n = 0; n < paramb.num_types; ++n) {
        paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
        paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
      }
    }
  }
  for (std::size_t n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
    if (paramb.rc_angular[n] > paramb.rc_angular_max) {
      paramb.rc_angular_max = paramb.rc_angular[n];
    }
  }

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  if (is_spin_model && paramb.spin_n_max < 0) {
    // Default spin_n_max to the (non-spin) radial order unless explicitly provided in the header.
    paramb.spin_n_max = paramb.n_max_radial;
  }
  if (is_spin_model) {
    if (paramb.spin_n_max < 0) {
      std::cout << "spin_n_max should be >= 0." << std::endl;
      exit(1);
    }
    if (paramb.spin_n_max > 12) {
      std::cout << "spin_n_max should be <= 12." << std::endl;
      exit(1);
    }
    if (paramb.spin_n_max + 1 > MAX_NUM_N) {
      std::cout << "spin_n_max is too large for compiled MAX_NUM_N." << std::endl;
      exit(1);
    }
  }

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() != 4) {
    print_tokens(tokens);
    std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.num_L = paramb.L_max;

  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  if (is_spin_model) {
    const int nspin = paramb.spin_n_max + 1;
    const int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
    annmb.dim += nspin * paramb.spin_blocks + spin_pmax;
  }

  // calculated parameters:
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  if (paramb.version == 3) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else {
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }
  if (paramb.model_type == 2) {
    annmb.num_para_ann *= 2;
  }
  if (paramb.charge_mode > 0) {
    annmb.num_para_ann += annmb.num_neurons1 * paramb.num_types + 1;
  }
  int num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (is_spin_model) {
    const int nspin = paramb.spin_n_max + 1;
    paramb.c_spin_block_stride =
      paramb.num_types_sq * nspin * (paramb.basis_size_radial + 1);
    int num_c_angular =
      paramb.num_types_sq * (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
    if (paramb.spin_mode == 2) {
      if (paramb.spin_blocks > 0) {
        paramb.num_c_spin = paramb.c_spin_block_stride;
        paramb.c_spin_offset = num_c_radial + num_c_angular;
      } else {
        paramb.num_c_spin = 0;
        paramb.c_spin_offset = 0;
      }
    } else if (paramb.spin_mode == 3) {
      paramb.num_c_spin = paramb.c_spin_block_stride * paramb.spin_blocks;
      paramb.c_spin_offset = num_c_radial + num_c_angular;
    } else {
      paramb.num_c_spin = 0;
      paramb.c_spin_offset = 0;
    }
    num_para_descriptor += static_cast<int>(paramb.num_c_spin);
  }
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;

  paramb.num_c_radial = num_c_radial;

  // NN and descriptor parameters
  parameters.resize(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  update_potential(parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters if (zbl.flexibled)
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
  input.close();


  // charge related parameters and data
  if (paramb.charge_mode > 0) {
    charge_para.alpha = PI / paramb.rc_radial_max; // a good value
    ewald.initialize(charge_para.alpha);
    charge_para.two_alpha_over_sqrt_pi = 2.0 * charge_para.alpha / sqrt(PI);
    charge_para.A = erfc(PI) / (paramb.rc_radial_max * paramb.rc_radial_max);
    charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-PI * PI) / paramb.rc_radial_max;
    charge_para.B = - erfc(PI) / paramb.rc_radial_max - charge_para.A * paramb.rc_radial_max;
  }

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  if (paramb.use_typewise_cutoff) {
    std::cout << "Cannot use tabulated radial functions with typewise cutoff." << std::endl;
    exit(1);
  }
  construct_table(parameters.data());
#endif

  // only report for rank_0
  if (is_rank_0) {

    if (is_spin_model) {
      if (paramb.num_types == 1) {
        std::cout << "Use the NEP" << paramb.version << "-Spin potential with " << paramb.num_types
                  << " atom type.\n";
      } else {
        std::cout << "Use the NEP" << paramb.version << "-Spin potential with " << paramb.num_types
                  << " atom types.\n";
      }
    } else if (paramb.charge_mode > 0) {
      if (paramb.num_types == 1) {
        std::cout << "Use the NEP4-Charge" << paramb.charge_mode << " potential with " << paramb.num_types
                  << " atom type.\n";
      } else {
        std::cout << "Use the NEP4-Charge" << paramb.charge_mode << " potential with " << paramb.num_types
                  << " atom types.\n";
      }
    } else {
      if (paramb.num_types == 1) {
        std::cout << "Use the NEP" << paramb.version << " potential with " << paramb.num_types
                  << " atom type.\n";
      } else {
        std::cout << "Use the NEP" << paramb.version << " potential with " << paramb.num_types
                  << " atom types.\n";
      }
    }

    for (std::size_t n = 0; n < paramb.num_types; ++n) {
      std::cout << "    type " << n << " (" << element_list[n]
                << " with Z = " << paramb.atomic_numbers[n] + 1 << ")"
                << " has cutoffs " << "(" << paramb.rc_radial[n] << " A, "
                << paramb.rc_angular[n] << " A).\n";
    }

    if (zbl.enabled) {
      if (zbl.flexibled) {
        std::cout << "    has flexible ZBL.\n";
      } else {
        std::cout << "    has universal ZBL with inner cutoff " << zbl.rc_inner
                  << " A and outer cutoff " << zbl.rc_outer << " A.\n";
        if (paramb.use_typewise_cutoff_zbl) {
          std::cout << "    ZBL typewise cutoff is enabled with factor "
                    << paramb.typewise_cutoff_zbl_factor << ".\n";
        }
      }
    }

    std::cout << "    n_max_radial = " << paramb.n_max_radial << ".\n";
    std::cout << "    n_max_angular = " << paramb.n_max_angular << ".\n";
    std::cout << "    basis_size_radial = " << paramb.basis_size_radial << ".\n";
    std::cout << "    basis_size_angular = " << paramb.basis_size_angular << ".\n";
    std::cout << "    l_max_3body = " << paramb.L_max << ".\n";
    std::cout << "    l_max_4body = " << (paramb.num_L >= 5 ? 2 : 0) << ".\n";
    std::cout << "    l_max_5body = " << (paramb.num_L >= 6 ? 1 : 0) << ".\n";
    std::cout << "    ANN = " << annmb.dim << "-" << annmb.num_neurons1 << "-1.\n";
    std::cout << "    number of neural network parameters = " << annmb.num_para_ann << ".\n";
    std::cout << "    number of descriptor parameters = " << num_para_descriptor << ".\n";
    std::cout << "    total number of parameters = " << annmb.num_para << ".\n";
  }
}

void NEP::update_type_map(const int ntype, int* type_map, char** elements)
{
  std::size_t n = 0;
  for (int itype = 0; itype < ntype + 1; ++itype) {
    // check if set NULL in lammps input file
    if (type_map[itype] == -1) {
      continue;
    }

    // find the same element name in potential file
    std::string element_name = elements[type_map[itype]];
    for (n = 0; n < paramb.num_types; ++n) {
      if (element_name == element_list[n]) {
        type_map[itype] = n;
        break;
      }
    }

    // check if no corresponding element
    if (n == paramb.num_types) {
      std::cout << "There is no element " << element_name << " in the potential file." << std::endl;
      exit(1);
    }
  }
}

void NEP::update_potential(double* parameters, ANN& ann)
{
  double* pointer = parameters;
  for (std::size_t t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    if (paramb.charge_mode > 0) {
      pointer += ann.num_neurons1 * 2;
    } else {
      pointer += ann.num_neurons1;
    }
    
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }

  if (paramb.charge_mode > 0) {
    ann.sqrt_epsilon_inf = pointer;
    pointer += 1;
  }

  ann.b1 = pointer;
  pointer += 1;

  if (paramb.model_type == 2) {
    for (std::size_t t = 0; t < paramb.num_types; ++t) {
      if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
        pointer -= (ann.dim + 2) * ann.num_neurons1;
      }
      ann.w0_pol[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0_pol[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1_pol[t] = pointer;
      pointer += ann.num_neurons1;
    }
    ann.b1_pol = pointer;
    pointer += 1;
  }

  ann.c = pointer;
}

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
void NEP::construct_table(double* parameters)
{
  gn_radial.resize(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  gnp_radial.resize(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  gn_angular.resize(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  gnp_angular.resize(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  double* c_pointer = parameters + annmb.num_para_ann;
  construct_table_radial_or_angular(
    paramb.version, paramb.num_types, paramb.num_types_sq, paramb.n_max_radial,
    paramb.basis_size_radial, paramb.rc_radial, paramb.rcinv_radial, c_pointer, gn_radial.data(),
    gnp_radial.data());
  construct_table_radial_or_angular(
    paramb.version, paramb.num_types, paramb.num_types_sq, paramb.n_max_angular,
    paramb.basis_size_angular, paramb.rc_angular, paramb.rcinv_angular,
    c_pointer + paramb.num_c_radial, gn_angular.data(), gnp_angular.data());
}
#endif

void NEP::allocate_memory(const int N)
{
  if (num_atoms < N) {
    NN_radial.resize(N);
    NL_radial.resize(N * MN);
    NN_angular.resize(N);
    NL_angular.resize(N * MN);
    r12.resize(N * MN * 6);
    Fp.resize(N * annmb.dim);
    sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    if (paramb.charge_mode > 0) {
      D_real.resize(N);
      charge_derivative.resize(N * annmb.dim);
    }
    dftd3.cn.resize(N);
    dftd3.dc6_sum.resize(N);
    dftd3.dc8_sum.resize(N);
    num_atoms = N;
  }
}

void NEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }

  if (paramb.charge_mode != 0) {
    std::cout << "Cannot use this compute for a qNEP model.\n";
    exit(1);
  }
  if (paramb.spin_mode > 0) {
    std::cout << "NEP spin model requires spin input; use compute with spin overload.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const int size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr, nullptr, nullptr, false, nullptr);

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  find_force_angular_small_box(
    false, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), force.data() + N,
      force.data() + N * 2, virial.data(), potential.data());
  }
}

void NEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial,
  std::vector<double>& charge,
  std::vector<double>& bec)
{
  if (paramb.charge_mode == 0) {
    std::cout << "Can only use this compute for a qNEP model.\n";
    exit(1);
  }
  if (paramb.spin_mode > 0) {
    std::cout << "Cannot use qNEP compute for a spin model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }
  if (N != charge.size()) {
    std::cout << "Type and charge sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != bec.size()) {
    std::cout << "Type and BEC sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }
  for (std::size_t n = 0; n < charge.size(); ++n) {
    charge[n] = 0.0;
  }
  for (std::size_t n = 0; n < bec.size(); ++n) {
    bec[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(), charge.data(), charge_derivative.data(), potential.data(), nullptr);

  zero_total_charge(N, charge.data());

  find_bec_diagonal(N, charge.data(), bec.data());
  find_bec_radial_small_box(
    paramb,
    annmb,
    N,
    NN_radial.data(),
    NL_radial.data(),
    type.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    charge_derivative.data(),
    bec.data());
  find_bec_angular_small_box(
    paramb,
    annmb,
    N,
    NN_angular.data(),
    NL_angular.data(),
    type.data(),
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    charge_derivative.data(),
    sum_fxyz.data(),
    bec.data());
  scale_bec(N, annmb.sqrt_epsilon_inf, bec.data());

  if (paramb.charge_mode == 1 || paramb.charge_mode == 2) {
    ewald.find_force(
      N,
      box.data(),
      charge,
      position,
      D_real,
      force,
      virial,
      potential);
  }

  if (paramb.charge_mode == 1) {
    find_force_charge_real_space_small_box(
      N,
      charge_para,
      NN_radial.data(),
      NL_radial.data(),
      charge.data(),
      r12.data(),
      r12.data() + size_x12,
      r12.data() + size_x12 * 2,
      force.data(),
      force.data() + N,
      force.data() + N * 2,
      virial.data(),
      potential.data(),
      D_real.data());
  }

  if (paramb.charge_mode == 3) {
    find_force_charge_real_space_only_small_box(
      N,
      charge_para,
      NN_radial.data(),
      NL_radial.data(),
      charge.data(),
      r12.data(),
      r12.data() + size_x12,
      r12.data() + size_x12 * 2,
      force.data(),
      force.data() + N,
      force.data() + N * 2,
      virial.data(),
      potential.data(),
      D_real.data());
  }

  find_force_radial_small_box(
    paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
    charge_derivative.data(), D_real.data(),
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  find_force_angular_small_box(
    paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    charge_derivative.data(), D_real.data(), sum_fxyz.data(),
    force.data(), force.data() + N, force.data() + N * 2, virial.data());

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), force.data() + N,
      force.data() + N * 2, virial.data(), potential.data());
  }
}

void NEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  const std::vector<double>& spin,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial,
  std::vector<double>& mforce)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }
  if (paramb.spin_mode == 0) {
    std::cout << "Spin mode is disabled; use compute without spin input.\n";
    exit(1);
  }
  if (paramb.charge_mode != 0) {
    std::cout << "Spin and charge modes cannot be enabled together.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != spin.size()) {
    std::cout << "Type and spin sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != mforce.size()) {
    std::cout << "Type and mforce sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(static_cast<int>(N));

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }
  for (std::size_t n = 0; n < mforce.size(); ++n) {
    mforce[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, static_cast<int>(N), MN, box, position, num_cells, ebox,
    NN_radial, NL_radial, NN_angular, NL_angular, r12);

  find_descriptor_spin_small_box(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), NN_angular.data(),
    NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, spin.data(),
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr);

  find_force_radial_spinbase_small_box(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(), force.data(), force.data() + N,
    force.data() + N * 2, virial.data());

  find_force_angular_spinbase_small_box(
    paramb, annmb, static_cast<int>(N), NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    sum_fxyz.data(), force.data(), force.data() + N, force.data() + N * 2, virial.data());

  find_force_radial_spin_spherical_fused_small_box(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, spin.data(), Fp.data(), force.data(),
    force.data() + N, force.data() + N * 2, virial.data());

  find_mforce_radial_spin_spherical_fused_small_box(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, spin.data(), Fp.data(), mforce.data(),
    mforce.data() + N, mforce.data() + N * 2);
  find_mforce_radial_spin_spherical_onsite_small_box(
    paramb, static_cast<int>(N), spin.data(), Fp.data(), mforce.data(), mforce.data() + N,
    mforce.data() + N * 2);
}

void NEP::compute_with_dftd3(
  const std::string& xc,
  const double rc_potential,
  const double rc_coordination_number,
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial)
{
  compute(type, box, position, potential, force, virial);
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;
  set_dftd3_para_all(xc, rc_potential, rc_coordination_number);

  find_neighbor_list_small_box(
    dftd3.rc_radial, dftd3.rc_angular, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
  find_dftd3_coordination_number(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5);
  add_dftd3_force(
    dftd3, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data() + size_x12 * 0,
    r12.data() + size_x12 * 1, r12.data() + size_x12 * 2, potential.data(), force.data(),
    virial.data());
  add_dftd3_force_extra(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), virial.data());
}

void NEP::compute_dftd3(
  const std::string& xc,
  const double rc_potential,
  const double rc_coordination_number,
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N != potential.size()) {
    std::cout << "Type and potential sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != force.size()) {
    std::cout << "Type and force sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != virial.size()) {
    std::cout << "Type and virial sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  set_dftd3_para_all(xc, rc_potential, rc_coordination_number);

  find_neighbor_list_small_box(
    dftd3.rc_radial, dftd3.rc_angular, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);
  find_dftd3_coordination_number(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5);
  add_dftd3_force(
    dftd3, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data() + size_x12 * 0,
    r12.data() + size_x12 * 1, r12.data() + size_x12 * 2, potential.data(), force.data(),
    virial.data());
  add_dftd3_force_extra(
    dftd3, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force.data(), virial.data());
}

void NEP::find_descriptor(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& descriptor)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.dim != descriptor.size()) {
    std::cout << "Type and descriptor sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  if (paramb.charge_mode > 0) {
    find_descriptor_small_box(
      false, true, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
      NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
      r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
      r12.data() + size_x12 * 5,
      Fp.data(), sum_fxyz.data(), nullptr, nullptr, nullptr, descriptor.data());
  } else {
    find_descriptor_small_box(
      false, true, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
      NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
      r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
      r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      gn_radial.data(), gn_angular.data(),
#endif
      Fp.data(), sum_fxyz.data(), nullptr, descriptor.data(), nullptr, nullptr, false, nullptr);
  }
}

void NEP::find_descriptor(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  const std::vector<double>& spin,
  std::vector<double>& descriptor)
{
  if (paramb.spin_mode == 0) {
    std::cout << "Spin mode is disabled; use find_descriptor without spin input.\n";
    exit(1);
  }
  if (paramb.charge_mode != 0) {
    std::cout << "Spin and charge modes cannot be enabled together.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 3 != spin.size()) {
    std::cout << "Type and spin sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.dim != descriptor.size()) {
    std::cout << "Type and descriptor sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial,
    NL_radial, NN_angular, NL_angular, r12);

  std::vector<double> potential(N, 0.0);
  find_descriptor_spin_small_box(
    paramb, annmb, static_cast<int>(N), NN_radial.data(), NL_radial.data(), NN_angular.data(),
    NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5, spin.data(), Fp.data(), sum_fxyz.data(), potential.data(),
    descriptor.data());
}

void NEP::find_latent_space(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& latent_space)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.num_neurons1 != latent_space.size()) {
    std::cout << "Type and latent_space sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    false, false, true, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), nullptr, nullptr, latent_space.data(), nullptr, false, nullptr);
}

void NEP::find_B_projection(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& B_projection)
{
  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N * annmb.num_neurons1 * (annmb.dim + 2) != B_projection.size()) {
    std::cout << "Type and B_projection sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    false, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), nullptr, nullptr, nullptr, nullptr, true, B_projection.data());
}

void NEP::find_dipole(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& dipole)
{
  if (paramb.model_type != 1) {
    std::cout << "Cannot compute dipole using a non-dipole NEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  std::vector<double> potential(N);  // not used but needed for find_descriptor_small_box
  std::vector<double> virial(N * 3); // need the 3 diagonal components only

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr, nullptr, nullptr, false, nullptr);

  find_force_radial_small_box(
    true, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  find_force_angular_small_box(
    true, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  for (int d = 0; d < 3; ++d) {
    dipole[d] = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
      dipole[d] += virial[d * N + n];
    }
  }
}

void NEP::find_polarizability(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& polarizability)
{
  if (paramb.model_type != 2) {
    std::cout << "Cannot compute polarizability using a non-polarizability NEP model.\n";
    exit(1);
  }

  const std::size_t N = type.size();
  const std::size_t size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  std::vector<double> potential(N);  // not used but needed for find_descriptor_small_box
  std::vector<double> virial(N * 9); // per-atom polarizability

  for (std::size_t n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (std::size_t n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial_max, paramb.rc_angular_max, N, MN, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, false, true, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), potential.data(), nullptr, nullptr, virial.data(), false, nullptr);

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  find_force_angular_small_box(
    false, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    nullptr, nullptr, nullptr, virial.data());

  for (int d = 0; d < 6; ++d) {
    polarizability[d] = 0.0;
  }
  for (std::size_t n = 0; n < N; ++n) {
    polarizability[0] += virial[0 * N + n]; // xx
    polarizability[1] += virial[4 * N + n]; // yy
    polarizability[2] += virial[8 * N + n]; // zz
    polarizability[3] += virial[1 * N + n]; // xy
    polarizability[4] += virial[5 * N + n]; // yz
    polarizability[5] += virial[6 * N + n]; // zx
  }
}

void NEP::compute_for_lammps(
  int nlocal,
  int N,
  int* ilist,
  int* NN,
  int** NL,
  int* type,
  int* type_map,
  double** pos,
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** virial)
{
  if (paramb.spin_mode > 0) {
    std::cout << "Spin models require per-atom spin input; use compute_for_lammps with spin data.\n";
    exit(1);
  }
  if (num_atoms < nlocal) {
    Fp.resize(nlocal * annmb.dim);
    sum_fxyz.resize(nlocal * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    num_atoms = nlocal;
  }
  find_descriptor_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), total_potential, potential);
  find_force_radial_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    force, total_virial, virial);
  find_force_angular_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, Fp.data(), sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force, total_virial, virial);
  if (zbl.enabled) {
    find_force_ZBL_for_lammps(
      paramb, zbl, N, ilist, NN, NL, type, type_map, pos, force, total_virial, virial,
      total_potential, potential);
  }
}

void NEP::compute_for_lammps(
  int nlocal,
  int N,
  int* ilist,
  int* NN,
  int** NL,
  int* type,
  int* type_map,
  double** pos,
  double** sp,
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** fm,
  double** virial)
{
  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }
  if (paramb.spin_mode == 0) {
    std::cout << "Spin mode is disabled; use compute_for_lammps without spin input.\n";
    exit(1);
  }
  if (paramb.charge_mode != 0) {
    std::cout << "Spin and charge modes cannot be enabled together.\n";
    exit(1);
  }
  if (!sp) {
    std::cout << "Spin models require per-atom sp input.\n";
    exit(1);
  }
  if (num_atoms < nlocal) {
    Fp.resize(nlocal * annmb.dim);
    sum_fxyz.resize(nlocal * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    num_atoms = nlocal;
  }

  find_descriptor_spin_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, sp, Fp.data(),
    sum_fxyz.data(), total_potential, potential);

  find_force_radial_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    force, total_virial, virial);

  find_force_angular_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, Fp.data(), sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    force, total_virial, virial);

  find_force_radial_spin_spherical_fused_for_lammps(
    paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, sp, Fp.data(), force,
    total_virial, virial);

  if (fm) {
    find_mforce_radial_spin_spherical_fused_for_lammps(
      paramb, annmb, nlocal, N, ilist, NN, NL, type, type_map, pos, sp, Fp.data(), fm);
    find_mforce_radial_spin_spherical_onsite_for_lammps(
      paramb, nlocal, N, ilist, sp, Fp.data(), fm);
  }

  if (zbl.enabled) {
    find_force_ZBL_for_lammps(
      paramb, zbl, N, ilist, NN, NL, type, type_map, pos, force, total_virial, virial,
      total_potential, potential);
  }
}

bool NEP::set_dftd3_para_one(
  const std::string& functional_input,
  const std::string& functional_library,
  const double s6,
  const double a1,
  const double s8,
  const double a2)
{
  if (functional_input == functional_library) {
    dftd3.s6 = s6;
    dftd3.a1 = a1;
    dftd3.s8 = s8;
    dftd3.a2 = a2 * dftd3para::Bohr;
    return true;
  }
  return false;
}

void NEP::set_dftd3_para_all(
  const std::string& functional_input,
  const double rc_potential,
  const double rc_coordination_number)
{

  dftd3.rc_radial = rc_potential;
  dftd3.rc_angular = rc_coordination_number;

  std::string functional = functional_input;
  std::transform(functional.begin(), functional.end(), functional.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  bool valid = false;
  valid = valid || set_dftd3_para_one(functional, "b1b95", 1.000, 0.2092, 1.4507, 5.5545);
  valid = valid || set_dftd3_para_one(functional, "b2gpplyp", 0.560, 0.0000, 0.2597, 6.3332);
  valid = valid || set_dftd3_para_one(functional, "b2plyp", 0.640, 0.3065, 0.9147, 5.0570);
  valid = valid || set_dftd3_para_one(functional, "b3lyp", 1.000, 0.3981, 1.9889, 4.4211);
  valid = valid || set_dftd3_para_one(functional, "b3pw91", 1.000, 0.4312, 2.8524, 4.4693);
  valid = valid || set_dftd3_para_one(functional, "b97d", 1.000, 0.5545, 2.2609, 3.2297);
  valid = valid || set_dftd3_para_one(functional, "bhlyp", 1.000, 0.2793, 1.0354, 4.9615);
  valid = valid || set_dftd3_para_one(functional, "blyp", 1.000, 0.4298, 2.6996, 4.2359);
  valid = valid || set_dftd3_para_one(functional, "bmk", 1.000, 0.1940, 2.0860, 5.9197);
  valid = valid || set_dftd3_para_one(functional, "bop", 1.000, 0.4870, 3.295, 3.5043);
  valid = valid || set_dftd3_para_one(functional, "bp86", 1.000, 0.3946, 3.2822, 4.8516);
  valid = valid || set_dftd3_para_one(functional, "bpbe", 1.000, 0.4567, 4.0728, 4.3908);
  valid = valid || set_dftd3_para_one(functional, "camb3lyp", 1.000, 0.3708, 2.0674, 5.4743);
  valid = valid || set_dftd3_para_one(functional, "dsdblyp", 0.500, 0.0000, 0.2130, 6.0519);
  valid = valid || set_dftd3_para_one(functional, "hcth120", 1.000, 0.3563, 1.0821, 4.3359);
  valid = valid || set_dftd3_para_one(functional, "hf", 1.000, 0.3385, 0.9171, 2.883);
  valid = valid || set_dftd3_para_one(functional, "hse-hjs", 1.000, 0.3830, 2.3100, 5.685);
  valid = valid || set_dftd3_para_one(functional, "lc-wpbe08", 1.000, 0.3919, 1.8541, 5.0897);
  valid = valid || set_dftd3_para_one(functional, "lcwpbe", 1.000, 0.3919, 1.8541, 5.0897);
  valid = valid || set_dftd3_para_one(functional, "m11", 1.000, 0.0000, 2.8112, 10.1389);
  valid = valid || set_dftd3_para_one(functional, "mn12l", 1.000, 0.0000, 2.2674, 9.1494);
  valid = valid || set_dftd3_para_one(functional, "mn12sx", 1.000, 0.0983, 1.1674, 8.0259);
  valid = valid || set_dftd3_para_one(functional, "mpw1b95", 1.000, 0.1955, 1.0508, 6.4177);
  valid = valid || set_dftd3_para_one(functional, "mpwb1k", 1.000, 0.1474, 0.9499, 6.6223);
  valid = valid || set_dftd3_para_one(functional, "mpwlyp", 1.000, 0.4831, 2.0077, 4.5323);
  valid = valid || set_dftd3_para_one(functional, "n12sx", 1.000, 0.3283, 2.4900, 5.7898);
  valid = valid || set_dftd3_para_one(functional, "olyp", 1.000, 0.5299, 2.6205, 2.8065);
  valid = valid || set_dftd3_para_one(functional, "opbe", 1.000, 0.5512, 3.3816, 2.9444);
  valid = valid || set_dftd3_para_one(functional, "otpss", 1.000, 0.4634, 2.7495, 4.3153);
  valid = valid || set_dftd3_para_one(functional, "pbe", 1.000, 0.4289, 0.7875, 4.4407);
  valid = valid || set_dftd3_para_one(functional, "pbe0", 1.000, 0.4145, 1.2177, 4.8593);
  valid = valid || set_dftd3_para_one(functional, "pbe38", 1.000, 0.3995, 1.4623, 5.1405);
  valid = valid || set_dftd3_para_one(functional, "pbesol", 1.000, 0.4466, 2.9491, 6.1742);
  valid = valid || set_dftd3_para_one(functional, "ptpss", 0.750, 0.000, 0.2804, 6.5745);
  valid = valid || set_dftd3_para_one(functional, "pw6b95", 1.000, 0.2076, 0.7257, 6.375);
  valid = valid || set_dftd3_para_one(functional, "pwb6k", 1.000, 0.1805, 0.9383, 7.7627);
  valid = valid || set_dftd3_para_one(functional, "pwpb95", 0.820, 0.0000, 0.2904, 7.3141);
  valid = valid || set_dftd3_para_one(functional, "revpbe", 1.000, 0.5238, 2.3550, 3.5016);
  valid = valid || set_dftd3_para_one(functional, "revpbe0", 1.000, 0.4679, 1.7588, 3.7619);
  valid = valid || set_dftd3_para_one(functional, "revpbe38", 1.000, 0.4309, 1.4760, 3.9446);
  valid = valid || set_dftd3_para_one(functional, "revssb", 1.000, 0.4720, 0.4389, 4.0986);
  valid = valid || set_dftd3_para_one(functional, "rpbe", 1.000, 0.1820, 0.8318, 4.0094);
  valid = valid || set_dftd3_para_one(functional, "rpw86pbe", 1.000, 0.4613, 1.3845, 4.5062);
  valid = valid || set_dftd3_para_one(functional, "scan", 1.000, 0.5380, 0.0000, 5.42);
  valid = valid || set_dftd3_para_one(functional, "sogga11x", 1.000, 0.1330, 1.1426, 5.7381);
  valid = valid || set_dftd3_para_one(functional, "ssb", 1.000, -0.0952, -0.1744, 5.2170);
  valid = valid || set_dftd3_para_one(functional, "tpss", 1.000, 0.4535, 1.9435, 4.4752);
  valid = valid || set_dftd3_para_one(functional, "tpss0", 1.000, 0.3768, 1.2576, 4.5865);
  valid = valid || set_dftd3_para_one(functional, "tpssh", 1.000, 0.4529, 2.2382, 4.6550);
  valid = valid || set_dftd3_para_one(functional, "b2kplyp", 0.64, 0.0000, 0.1521, 7.1916);
  valid = valid || set_dftd3_para_one(functional, "dsd-pbep86", 0.418, 0.0000, 0.0000, 5.6500);
  valid = valid || set_dftd3_para_one(functional, "b97m", 1.0000, -0.0780, 0.1384, 5.5946);
  valid = valid || set_dftd3_para_one(functional, "wb97x", 1.0000, 0.0000, 0.2641, 5.4959);
  valid = valid || set_dftd3_para_one(functional, "wb97m", 1.0000, 0.5660, 0.3908, 3.1280);

  if (!valid) {
    std::cout << "The " << functional
              << " functional is not supported for DFT-D3 with BJ damping.\n"
              << std::endl;
    exit(1);
  }
};
