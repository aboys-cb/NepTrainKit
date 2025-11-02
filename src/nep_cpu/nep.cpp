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
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <tuple>
#ifndef NEIGHMASK
#define NEIGHMASK 0x3FFFFFFF
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{
const int MAX_NEURON = 120; // maximum number of neurons in the hidden layer
const int MN = 1000;        // maximum number of neighbors for one atom
const int NUM_OF_ABC = 80;  // 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 for L_max = 8
const int MAX_NUM_N = 17;   // basis_size_radial+1 = 16+1
const int MAX_DIM = 103;
const int MAX_DIM_ANGULAR = 90;
const double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435, 0.596831036594608,
  0.596831036594608, 0.149207759148652, 0.149207759148652, 0.139260575205408, 0.104445431404056,
  0.104445431404056, 1.044454314040563, 1.044454314040563, 0.174075719006761, 0.174075719006761,
  0.011190581936149, 0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606, 0.013677377921960,
  0.102580334414698, 0.102580334414698, 2.872249363611549, 2.872249363611549, 0.119677056817148,
  0.119677056817148, 2.154187022708661, 2.154187022708661, 0.215418702270866, 0.215418702270866,
  0.004041043476943, 0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
  0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694, 2.800443129521260,
  2.800443129521260, 0.233370260793438, 0.233370260793438, 0.004662742473395, 0.004079899664221,
  0.004079899664221, 0.024479397985326, 0.024479397985326, 0.012239698992663, 0.012239698992663,
  0.538546755677165, 0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
  3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966, 0.005944996653579,
  0.005944996653579, 0.104037441437634, 0.104037441437634, 0.762941237209318, 0.762941237209318,
  0.114441185581398, 0.114441185581398, 5.950941650232678, 5.950941650232678, 0.141689086910302,
  0.141689086910302, 4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816};
const double C4B[5] = {
  -0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723};
const double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};

const double Z_COEFFICIENT_1[2][2] = {{0.0, 1.0}, {1.0, 0.0}};

const double Z_COEFFICIENT_2[3][3] = {{-1.0, 0.0, 3.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};

const double Z_COEFFICIENT_3[4][4] = {
  {0.0, -3.0, 0.0, 5.0}, {-1.0, 0.0, 5.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_4[5][5] = {
  {3.0, 0.0, -30.0, 0.0, 35.0},
  {0.0, -3.0, 0.0, 7.0, 0.0},
  {-1.0, 0.0, 7.0, 0.0, 0.0},
  {0.0, 1.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_5[6][6] = {
  {0.0, 15.0, 0.0, -70.0, 0.0, 63.0}, {1.0, 0.0, -14.0, 0.0, 21.0, 0.0},
  {0.0, -1.0, 0.0, 3.0, 0.0, 0.0},    {-1.0, 0.0, 9.0, 0.0, 0.0, 0.0},
  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},     {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_6[7][7] = {
  {-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0}, {0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0},
  {1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0},      {0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0},
  {-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0},       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_7[8][8] = {{0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0},
                                      {-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0},
                                      {0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0},
                                      {3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0},
                                      {0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0},
                                      {-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double Z_COEFFICIENT_8[9][9] = {
  {35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0},
  {0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0},
  {-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0},
  {0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

const double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0)
const double PI = 3.141592653589793;
const double PI_HALF = 1.570796326794897;
const int NUM_ELEMENTS = 94;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};
double COVALENT_RADIUS[NUM_ELEMENTS] = {
  0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333, 0.893333,
  1.86667,  1.66667,  1.50667, 1.38667, 1.46667, 1.36,    1.32,     1.28,    2.34667,  2.05333,
  1.77333,  1.62667,  1.61333, 1.46667, 1.42667, 1.38667, 1.33333,  1.32,    1.34667,  1.45333,
  1.49333,  1.45333,  1.53333, 1.46667, 1.52,    1.56,    2.52,     2.22667, 1.96,     1.85333,
  1.76,     1.65333,  1.53333, 1.50667, 1.50667, 1.44,    1.53333,  1.64,    1.70667,  1.68,
  1.68,     1.64,     1.76,    1.74667, 2.78667, 2.34667, 2.16,     1.96,    2.10667,  2.09333,
  2.08,     2.06667,  2.01333, 2.02667, 2.01333, 2.0,     1.98667,  1.98667, 1.97333,  2.04,
  1.94667,  1.82667,  1.74667, 1.64,    1.57333, 1.54667, 1.48,     1.49333, 1.50667,  1.76,
  1.73333,  1.73333,  1.81333, 1.74667, 1.84,    1.89333, 2.68,     2.41333, 2.22667,  2.10667,
  2.02667,  2.04,     2.05333, 2.06667};

void complex_product(const double a, const double b, double& real_part, double& imag_part)
{
  const double real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
}

void apply_ann_one_layer(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double* latent_space,
  bool need_B_projection,
  double* B_projection)
{
  for (int n = 0; n < num_neurons1; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tan_der = 1.0 - x1 * x1;

    if (need_B_projection) {
      // calculate B_projection:
      // dE/dw0
      for (int d = 0; d < dim; ++d)
        B_projection[n * (dim + 2) + d] = tan_der * q[d] * w1[n];
      // dE/db0
      B_projection[n * (dim + 2) + dim] = -tan_der * w1[n];
      // dE/dw1
      B_projection[n * (dim + 2) + dim + 1] = x1;
    }

    latent_space[n] = w1[n] * x1; // also try x1
    energy += w1[n] * x1;
    for (int d = 0; d < dim; ++d) {
      double y1 = tan_der * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[0];
}

void apply_ann_one_layer_nep5(
  const int dim,
  const int num_neurons1,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double* latent_space)
{
  for (int n = 0; n < num_neurons1; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    latent_space[n] = w1[n] * x1; // also try x1
    energy += w1[n] * x1;
    for (int d = 0; d < dim; ++d) {
      double y1 = (1.0 - x1 * x1) * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= w1[num_neurons1] + b1[0]; // typewise bias + common bias
}

void find_fc(double rc, double rcinv, double d12, double& fc)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
  } else {
    fc = 0.0;
  }
}

void find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
    fcp = -PI_HALF * sin(PI * x);
    fcp *= rcinv;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

void find_fc_and_fcp_zbl(double r1, double r2, double d12, double& fc, double& fcp)
{
  if (d12 < r1) {
    fc = 1.0;
    fcp = 0.0;
  } else if (d12 < r2) {
    double pi_factor = PI / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

void find_phi_and_phip_zbl(double a, double b, double x, double& phi, double& phip)
{
  double tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

void find_f_and_fp_zbl(
  double zizj,
  double a_inv,
  double rc_inner,
  double rc_outer,
  double d12,
  double d12inv,
  double& f,
  double& fp)
{
  double x = d12 * a_inv;
  f = fp = 0.0;
  double Zbl_para[8] = {0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

void find_f_and_fp_zbl(
  double* zbl_para, double zizj, double a_inv, double d12, double d12inv, double& f, double& fp)
{
  double x = d12 * a_inv;
  f = fp = 0.0;
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

void find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5 * fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0) * 0.5 * fc12;
  }
}

void find_fn_and_fnp(
  const int n,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double& fn,
  double& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5;
    fnp = 2.0 * (d12 * rcinv - 1.0) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    double u0 = 1.0;
    double u1 = 2.0 * x;
    double u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0 * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0) * 0.5;
    fnp = n * u0 * 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

void find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5 * fc12;
  }
}

void find_fn_and_fnp(
  const int n_max,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double* fn,
  double* fnp)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fnp[0] = 0.0;
  fn[1] = x;
  fnp[1] = 1.0;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5;
    fnp[m] *= 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

void get_f12_4body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double y20 = (3.0 * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  double tmp0 = C4B[0] * 3.0 * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
                C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  double tmp1 = tmp0 * y20 * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * 2.0 - C4B[3] * s[3] * s[1] * 2.0 + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * 2.0 + C4B[3] * s[3] * s[2] * 2.0 + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * 2.0 + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * 2.0 + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
  f12[2] += tmp1 * r12[2];
}

void get_f12_5body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  double tmp0 = C5B[0] * 4.0 * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * 2.0 * s[0];
  double tmp1 = tmp0 * r12[2] * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s[1]
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[1] * 4.0;
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[2] * 4.0;
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

template <int L>
void calculate_s_one(
  const int n, const int n_max_angular_plus_1, const double* Fp, const double* sum_fxyz, double* s)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  double Fp_factor = 2.0 * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  s[0] = sum_fxyz[n * NUM_OF_ABC + L_square_minus_1] * C3B[L_square_minus_1] * Fp_factor;
  Fp_factor *= 2.0;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    s[k] = sum_fxyz[n * NUM_OF_ABC + L_square_minus_1 + k] * C3B[L_square_minus_1 + k] * Fp_factor;
  }
}

template <int L>
void accumulate_f12_one(
  const double d12inv,
  const double fn,
  const double fnp,
  const double* s,
  const double* r12,
  double* f12)
{
  const double dx[3] = {
    (1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {
    -r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {
    -r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};

  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      if (L == 1) {
        z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_1[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 2) {
        z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_2[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 3) {
        z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_3[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 4) {
        z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_4[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_5[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_6[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_7[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_8[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        double real_part_dx = dx[d];
        double imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const double xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    }
  }
}

void accumulate_f12(
  const int L_max,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double fn_original = fn;
  const double fnp_original = fnp;
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 2) {
    double s1[3] = {
      sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
    get_f12_5body(d12, d12inv, fn, fnp, Fp[(L_max + 1) * n_max_angular_plus_1 + n], s1, r12, f12);
  }

  if (L_max >= 1) {
    double s1[3];
    calculate_s_one<1>(n, n_max_angular_plus_1, Fp, sum_fxyz, s1);
    accumulate_f12_one<1>(d12inv, fn_original, fnp_original, s1, r12unit, f12);
  }

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 1) {
    double s2[5] = {
      sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
      sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};
    get_f12_4body(d12, d12inv, fn, fnp, Fp[L_max * n_max_angular_plus_1 + n], s2, r12, f12);
  }

  if (L_max >= 2) {
    double s2[5];
    calculate_s_one<2>(n, n_max_angular_plus_1, Fp, sum_fxyz, s2);
    accumulate_f12_one<2>(d12inv, fn_original, fnp_original, s2, r12unit, f12);
  }

  if (L_max >= 3) {
    double s3[7];
    calculate_s_one<3>(n, n_max_angular_plus_1, Fp, sum_fxyz, s3);
    accumulate_f12_one<3>(d12inv, fn_original, fnp_original, s3, r12unit, f12);
  }

  if (L_max >= 4) {
    double s4[9];
    calculate_s_one<4>(n, n_max_angular_plus_1, Fp, sum_fxyz, s4);
    accumulate_f12_one<4>(d12inv, fn_original, fnp_original, s4, r12unit, f12);
  }

  if (L_max >= 5) {
    double s5[11];
    calculate_s_one<5>(n, n_max_angular_plus_1, Fp, sum_fxyz, s5);
    accumulate_f12_one<5>(d12inv, fn_original, fnp_original, s5, r12unit, f12);
  }

  if (L_max >= 6) {
    double s6[13];
    calculate_s_one<6>(n, n_max_angular_plus_1, Fp, sum_fxyz, s6);
    accumulate_f12_one<6>(d12inv, fn_original, fnp_original, s6, r12unit, f12);
  }

  if (L_max >= 7) {
    double s7[15];
    calculate_s_one<7>(n, n_max_angular_plus_1, Fp, sum_fxyz, s7);
    accumulate_f12_one<7>(d12inv, fn_original, fnp_original, s7, r12unit, f12);
  }

  if (L_max >= 8) {
    double s8[17];
    calculate_s_one<8>(n, n_max_angular_plus_1, Fp, sum_fxyz, s8);
    accumulate_f12_one<8>(d12inv, fn_original, fnp_original, s8, r12unit, f12);
  }
}

template <int L>
void accumulate_s_one(
  const double x12, const double y12, const double z12, const double fn, double* s)
{
  int s_index = L * L - 1;
  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  double real_part = x12;
  double imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      if (L == 1) {
        z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
      }
      if (L == 2) {
        z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
      }
      if (L == 3) {
        z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
      }
      if (L == 4) {
        z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
      }
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
      }
    }
    z_factor *= fn;
    if (n1 == 0) {
      s[s_index++] += z_factor;
    } else {
      s[s_index++] += z_factor * real_part;
      s[s_index++] += z_factor * imag_part;
      complex_product(x12, y12, real_part, imag_part);
    }
  }
}

void accumulate_s(
  const int L_max, const double d12, double x12, double y12, double z12, const double fn, double* s)
{
  double d12inv = 1.0 / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  if (L_max >= 1) {
    accumulate_s_one<1>(x12, y12, z12, fn, s);
  }
  if (L_max >= 2) {
    accumulate_s_one<2>(x12, y12, z12, fn, s);
  }
  if (L_max >= 3) {
    accumulate_s_one<3>(x12, y12, z12, fn, s);
  }
  if (L_max >= 4) {
    accumulate_s_one<4>(x12, y12, z12, fn, s);
  }
  if (L_max >= 5) {
    accumulate_s_one<5>(x12, y12, z12, fn, s);
  }
  if (L_max >= 6) {
    accumulate_s_one<6>(x12, y12, z12, fn, s);
  }
  if (L_max >= 7) {
    accumulate_s_one<7>(x12, y12, z12, fn, s);
  }
  if (L_max >= 8) {
    accumulate_s_one<8>(x12, y12, z12, fn, s);
  }
}

template <int L>
double find_q_one(const double* s)
{
  const int start_index = L * L - 1;
  const int num_terms = 2 * L + 1;
  double q = 0.0;
  for (int k = 1; k < num_terms; ++k) {
    q += C3B[start_index + k] * s[start_index + k] * s[start_index + k];
  }
  q *= 2.0;
  q += C3B[start_index] * s[start_index] * s[start_index];
  return q;
}

void find_q(
  const int L_max,
  const int num_L,
  const int n_max_angular_plus_1,
  const int n,
  const double* s,
  double* q)
{
  if (L_max >= 1) {
    q[0 * n_max_angular_plus_1 + n] = find_q_one<1>(s);
  }
  if (L_max >= 2) {
    q[1 * n_max_angular_plus_1 + n] = find_q_one<2>(s);
  }
  if (L_max >= 3) {
    q[2 * n_max_angular_plus_1 + n] = find_q_one<3>(s);
  }
  if (L_max >= 4) {
    q[3 * n_max_angular_plus_1 + n] = find_q_one<4>(s);
  }
  if (L_max >= 5) {
    q[4 * n_max_angular_plus_1 + n] = find_q_one<5>(s);
  }
  if (L_max >= 6) {
    q[5 * n_max_angular_plus_1 + n] = find_q_one<6>(s);
  }
  if (L_max >= 7) {
    q[6 * n_max_angular_plus_1 + n] = find_q_one<7>(s);
  }
  if (L_max >= 8) {
    q[7 * n_max_angular_plus_1 + n] = find_q_one<8>(s);
  }
  if (num_L >= L_max + 1) {
    q[L_max * n_max_angular_plus_1 + n] =
      C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      C4B[4] * s[4] * s[5] * s[7];
  }
  if (num_L >= L_max + 2) {
    double s0_sq = s[0] * s[0];
    double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[(L_max + 1) * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq +
                                                C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                                C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
  }
}

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
const int table_length = 2001;
const int table_segments = table_length - 1;
const double table_resolution = 0.0005;

void find_index_and_weight(
  const double d12_reduced,
  int& index_left,
  int& index_right,
  double& weight_left,
  double& weight_right)
{
  double d12_index = d12_reduced * table_segments;
  index_left = int(d12_index);
  if (index_left == table_segments) {
    --index_left;
  }
  index_right = index_left + 1;
  weight_right = d12_index - index_left;
  weight_left = 1.0 - weight_right;
}

void construct_table_radial_or_angular(
  const int version,
  const int num_types,
  const int num_types_sq,
  const int n_max,
  const int basis_size,
  const double rc,
  const double rcinv,
  const double* c,
  double* gn,
  double* gnp)
{
  for (int table_index = 0; table_index < table_length; ++table_index) {
    double d12 = table_index * table_resolution * rc;
    double fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int t12 = t1 * num_types + t2;
        double fn12[MAX_NUM_N];
        double fnp12[MAX_NUM_N];
        find_fn_and_fnp(basis_size, rcinv, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= n_max; ++n) {
          double gn12 = 0.0;
          double gnp12 = 0.0;
          for (int k = 0; k <= basis_size; ++k) {
            gn12 += fn12[k] * c[(n * (basis_size + 1) + k) * num_types_sq + t12];
            gnp12 += fnp12[k] * c[(n * (basis_size + 1) + k) * num_types_sq + t12];
          }
          int index_all = (table_index * num_types_sq + t12) * (n_max + 1) + n;
          gn[index_all] = gn12;
          gnp[index_all] = gnp12;
        }
      }
    }
  }
}
#endif

void find_descriptor_small_box(
  const bool calculating_potential,
  const bool calculating_descriptor,
  const bool calculating_latent_space,
  const bool calculating_polarizability,
  NEP3::ParaMB& paramb,
  NEP3::ANN& annmb,
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
  const int NT  = (paramb.spin_mode == 1) ? paramb.num_types_total : paramb.num_types;
  const int NT2 = (paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    // ANN weights are defined only for real types. In spin mode, map
    // any virtual (pseudo) type back to its corresponding real type
    // when indexing ANN parameter arrays.
    int t1_ann = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
    double q[MAX_DIM] = {0.0};

    const double tiny = 1e-12;
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      double r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (d12 <= tiny) continue;

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      if (paramb.spin_mode == 1) {
        std::cout << "Tabulated radial/angular functions are not supported in spin mode." << std::endl;
        exit(1);
      }
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * NT + g_type[n2];
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * NT2 + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * NT2 + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      double fc12;
      int t2 = g_type[n2];
      double rc = paramb.rc_radial;
      double rcinv = paramb.rcinv_radial;
      if (paramb.use_typewise_cutoff) {
        int tr1 = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
        int tr2 = (paramb.spin_mode == 1) ? (t2 % paramb.num_types_real) : t2;
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[tr1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[tr2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
        rcinv = 1.0 / rc;
      }
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * NT2;
          c_index += t1 * NT + t2;
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
        if (d12 <= tiny) continue;
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
        if (paramb.spin_mode == 1) {
          std::cout << "Tabulated radial/angular functions are not supported in spin mode." << std::endl;
          exit(1);
        }
        int index_left, index_right;
        double weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * NT + g_type[n2];
        double gn12 =
          g_gn_angular[(index_left * NT2 + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * NT2 + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
#else
        int t2 = g_type[n2];
        double fc12;
        double rc = paramb.rc_angular;
        double rcinv = paramb.rcinv_angular;
        if (paramb.use_typewise_cutoff) {
          int tr1 = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
          int tr2 = (paramb.spin_mode == 1) ? (t2 % paramb.num_types_real) : t2;
          rc = std::min(
            (COVALENT_RADIUS[paramb.atomic_numbers[tr1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[tr2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
          rcinv = 1.0 / rc;
        }
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * NT2;
          c_index += t1 * NT + t2 + paramb.num_c_radial;
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
          annmb.dim, annmb.num_neurons1, annmb.w0_pol[t1_ann], annmb.b0_pol[t1_ann],
          annmb.w1_pol[t1_ann], annmb.b1_pol, q, F, Fp, latent_space, false, nullptr);
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
          annmb.dim, annmb.num_neurons1, annmb.w0[t1_ann], annmb.b0[t1_ann], annmb.w1[t1_ann],
          annmb.b1, q, F, Fp, latent_space);
      } else {
        apply_ann_one_layer(
          annmb.dim, annmb.num_neurons1, annmb.w0[t1_ann], annmb.b0[t1_ann], annmb.w1[t1_ann],
          annmb.b1, q, F, Fp, latent_space, calculating_B_projection,
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
  NEP3::ParaMB& paramb,
  NEP3::ANN& annmb,
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
  double* g_virial,
  bool accumulate_on_center = false,
  // spin-optimized extras (optional): if provided, treat n2>=pseudo_start as pseudo
  int pseudo_start = -1,
  const int* pseudo_to_host = nullptr,
  double* g_pseudo_fx = nullptr,
  double* g_pseudo_fy = nullptr,
  double* g_pseudo_fz = nullptr,
  bool enable_parallel_center_only = false)
{
  const bool pseudo_mode = (pseudo_to_host && pseudo_start >= 0 && g_pseudo_fx && g_pseudo_fy && g_pseudo_fz);
  // Fast path: in spin mode, if requested, parallelize across centers and avoid neighbor writes
#if defined(_OPENMP)
  if (enable_parallel_center_only && pseudo_mode && !is_dipole) {
    int nthreads = omp_get_max_threads();
    std::vector<double> tls_pfx(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_pfy(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_pfz(static_cast<size_t>(nthreads) * N, 0.0);
    // TLS for deterministic accumulation of atom forces (center + neighbors)
    std::vector<double> tls_fx(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_fy(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_fz(static_cast<size_t>(nthreads) * N, 0.0);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      double* pfx = tls_pfx.data() + static_cast<size_t>(tid) * N;
      double* pfy = tls_pfy.data() + static_cast<size_t>(tid) * N;
      double* pfz = tls_pfz.data() + static_cast<size_t>(tid) * N;
      double* lfx = tls_fx.data() + static_cast<size_t>(tid) * N;
      double* lfy = tls_fy.data() + static_cast<size_t>(tid) * N;
      double* lfz = tls_fz.data() + static_cast<size_t>(tid) * N;

      const double tiny = 1e-12;
      #pragma omp for schedule(static)
      for (int n1 = 0; n1 < N; ++n1) {
        int t1 = g_type[n1];
        for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
          int index = i1 * N + n1;
          int n2 = g_NL[index];
          int t2 = g_type[n2];
          double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
          double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
          if (d12 <= tiny) continue; // avoid singularities
          double d12inv = 1.0 / d12;
          double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
          int t12 = t1 * ((paramb.spin_mode == 1) ? paramb.num_types_total : paramb.num_types) + t2;
          int index_left, index_right; double weight_left, weight_right;
          find_index_and_weight(d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
          for (int n = 0; n <= paramb.n_max_radial; ++n) {
            double gnp12 =
              g_gnp_radial[(index_left * ((paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq) + t12) * (paramb.n_max_radial + 1) + n] * weight_left +
              g_gnp_radial[(index_right * ((paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq) + t12) * (paramb.n_max_radial + 1) + n] * weight_right;
            double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
            f12[0] += tmp12 * r12[0];
            f12[1] += tmp12 * r12[1];
            f12[2] += tmp12 * r12[2];
          }
#else
          double fc12, fcp12;
          double rc = paramb.rc_radial; double rcinv = paramb.rcinv_radial;
          if (paramb.use_typewise_cutoff) {
            int tr1 = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
            int tr2 = (paramb.spin_mode == 1) ? (t2 % paramb.num_types_real) : t2;
            rc = std::min((COVALENT_RADIUS[paramb.atomic_numbers[tr1]] + COVALENT_RADIUS[paramb.atomic_numbers[tr2]]) * paramb.typewise_cutoff_radial_factor, rc);
            rcinv = 1.0 / rc;
          }
          find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
          double fn12[MAX_NUM_N]; double fnp12[MAX_NUM_N];
          find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
          const int NT2 = (paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq;
          const int NT  = (paramb.spin_mode == 1) ? paramb.num_types_total    : paramb.num_types;
          for (int n = 0; n <= paramb.n_max_radial; ++n) {
            double gnp12 = 0.0;
            for (int k = 0; k <= paramb.basis_size_radial; ++k) {
              int c_index = (n * (paramb.basis_size_radial + 1) + k) * NT2;
              c_index += t1 * NT + t2;
              gnp12 += fnp12[k] * annmb.c[c_index];
            }
            double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
            f12[0] += tmp12 * r12[0];
            f12[1] += tmp12 * r12[1];
            f12[2] += tmp12 * r12[2];
          }
#endif
          // Deterministic TLS accumulation: center and neighbor contributions into TLS
          lfx[n1] += f12[0];
          lfy[n1] += f12[1];
          lfz[n1] += f12[2];

          const bool is_pseudo = (n2 >= pseudo_start);
          if (is_pseudo) {
            int host = pseudo_to_host[n2];
            if (host >= 0) {
              pfx[host] -= f12[0];
              pfy[host] -= f12[1];
              pfz[host] -= f12[2];
            }
          } else {
            lfx[n2] -= f12[0];
            lfy[n2] -= f12[1];
            lfz[n2] -= f12[2];
          }

          // virial: accumulate on center only (spin path provides accumulate_on_center=true)
          int idx = n1;
          g_virial[idx + 0 * N] -= r12[0] * f12[0];
          g_virial[idx + 1 * N] -= r12[0] * f12[1];
          g_virial[idx + 2 * N] -= r12[0] * f12[2];
          g_virial[idx + 3 * N] -= r12[1] * f12[0];
          g_virial[idx + 4 * N] -= r12[1] * f12[1];
          g_virial[idx + 5 * N] -= r12[1] * f12[2];
          g_virial[idx + 6 * N] -= r12[2] * f12[0];
          g_virial[idx + 7 * N] -= r12[2] * f12[1];
          g_virial[idx + 8 * N] -= r12[2] * f12[2];
        }
      }
    }

    // reduction of TLS atom forces (deterministic)
    for (int i = 0; i < N; ++i) {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (int t = 0; t < nthreads; ++t) {
        sx += tls_fx[static_cast<size_t>(t) * N + i];
        sy += tls_fy[static_cast<size_t>(t) * N + i];
        sz += tls_fz[static_cast<size_t>(t) * N + i];
      }
      if (g_fx) g_fx[i] += sx;
      if (g_fy) g_fy[i] += sy;
      if (g_fz) g_fz[i] += sz;
    }

    // reduction of TLS pseudo reactions
    for (int i = 0; i < N; ++i) {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (int t = 0; t < nthreads; ++t) {
        sx += tls_pfx[static_cast<size_t>(t) * N + i];
        sy += tls_pfy[static_cast<size_t>(t) * N + i];
        sz += tls_pfz[static_cast<size_t>(t) * N + i];
      }
      g_pseudo_fx[i] += sx;
      g_pseudo_fy[i] += sy;
      g_pseudo_fz[i] += sz;
    }
    return;
  }
#endif
  const int NT  = (paramb.spin_mode == 1) ? paramb.num_types_total : paramb.num_types;
  const int NT2 = (paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq;
  const double tiny = 1e-12;
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (d12 <= tiny) continue; // avoid singularities
      double d12inv = 1.0 / d12;
      double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      if (paramb.spin_mode == 1) {
        std::cout << "Tabulated radial/angular functions are not supported in spin mode." << std::endl;
        exit(1);
      }
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * NT + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 =
          g_gnp_radial[(index_left * NT2 + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * NT2 + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#else
      double fc12, fcp12;
      double rc = paramb.rc_radial;
      double rcinv = paramb.rcinv_radial;
      if (paramb.use_typewise_cutoff) {
        int tr1 = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
        int tr2 = (paramb.spin_mode == 1) ? (t2 % paramb.num_types_real) : t2;
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[tr1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[tr2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
        rcinv = 1.0 / rc;
      }
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * NT2;
          c_index += t1 * NT + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        double tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#endif

      if (!pseudo_mode) {
        if (g_fx) { g_fx[n1] += f12[0]; g_fx[n2] -= f12[0]; }
        if (g_fy) { g_fy[n1] += f12[1]; g_fy[n2] -= f12[1]; }
        if (g_fz) { g_fz[n1] += f12[2]; g_fz[n2] -= f12[2]; }
      } else {
        const bool is_pseudo = (n2 >= pseudo_start);
        if (g_fx) {
          g_fx[n1] += f12[0];
          if (is_pseudo) { int host = pseudo_to_host[n2]; if (host >= 0) g_pseudo_fx[host] -= f12[0]; }
          else { g_fx[n2] -= f12[0]; }
        }
        if (g_fy) {
          g_fy[n1] += f12[1];
          if (is_pseudo) { int host = pseudo_to_host[n2]; if (host >= 0) g_pseudo_fy[host] -= f12[1]; }
          else { g_fy[n2] -= f12[1]; }
        }
        if (g_fz) {
          g_fz[n1] += f12[2];
          if (is_pseudo) { int host = pseudo_to_host[n2]; if (host >= 0) g_pseudo_fz[host] -= f12[2]; }
          else { g_fz[n2] -= f12[2]; }
        }
      }

      if (!is_dipole) {
        int idx = accumulate_on_center ? n1 : n2;
        g_virial[idx + 0 * N] -= r12[0] * f12[0];
        g_virial[idx + 1 * N] -= r12[0] * f12[1];
        g_virial[idx + 2 * N] -= r12[0] * f12[2];
        g_virial[idx + 3 * N] -= r12[1] * f12[0];
        g_virial[idx + 4 * N] -= r12[1] * f12[1];
        g_virial[idx + 5 * N] -= r12[1] * f12[2];
        g_virial[idx + 6 * N] -= r12[2] * f12[0];
        g_virial[idx + 7 * N] -= r12[2] * f12[1];
        g_virial[idx + 8 * N] -= r12[2] * f12[2];
      } else {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        int idx = accumulate_on_center ? n1 : n2;
        g_virial[idx + 0 * N] -= r12_square * f12[0];
        g_virial[idx + 1 * N] -= r12_square * f12[1];
        g_virial[idx + 2 * N] -= r12_square * f12[2];
      }
    }
  }
}

void find_force_angular_small_box(
  const bool is_dipole,
  NEP3::ParaMB& paramb,
  NEP3::ANN& annmb,
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
  double* g_virial,
  bool accumulate_on_center = false,
  // spin-optimized extras (optional): if provided, treat n2>=pseudo_start as pseudo
  int pseudo_start = -1,
  const int* pseudo_to_host = nullptr,
  double* g_pseudo_fx = nullptr,
  double* g_pseudo_fy = nullptr,
  double* g_pseudo_fz = nullptr,
  bool enable_parallel_center_only = false)
{
  const bool pseudo_mode = (pseudo_to_host && pseudo_start >= 0 && g_pseudo_fx && g_pseudo_fy && g_pseudo_fz);
  // Fast path: in spin mode, if requested, parallelize across centers and avoid neighbor writes
#if defined(_OPENMP)
  if (enable_parallel_center_only && pseudo_mode && !is_dipole) {
    int nthreads = omp_get_max_threads();
    std::vector<double> tls_pfx(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_pfy(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_pfz(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_fx(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_fy(static_cast<size_t>(nthreads) * N, 0.0);
    std::vector<double> tls_fz(static_cast<size_t>(nthreads) * N, 0.0);

    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      double* pfx = tls_pfx.data() + static_cast<size_t>(tid) * N;
      double* pfy = tls_pfy.data() + static_cast<size_t>(tid) * N;
      double* pfz = tls_pfz.data() + static_cast<size_t>(tid) * N;
      double* lfx = tls_fx.data() + static_cast<size_t>(tid) * N;
      double* lfy = tls_fy.data() + static_cast<size_t>(tid) * N;
      double* lfz = tls_fz.data() + static_cast<size_t>(tid) * N;

      const double tiny = 1e-12;
      #pragma omp for schedule(static)
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
          int n2 = g_NL_angular[index];
          double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
          double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
          if (d12 <= tiny) continue; // avoid singularities
          double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
          int t12 = t1 * ((paramb.spin_mode == 1) ? paramb.num_types_total : paramb.num_types) + g_type[n2];
          int index_left, index_right; double weight_left, weight_right;
          find_index_and_weight(d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
          for (int n = 0; n <= paramb.n_max_angular; ++n) {
            int index_left_all  = (index_left  * ((paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq) + t12) * (paramb.n_max_angular + 1) + n;
            int index_right_all = (index_right * ((paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq) + t12) * (paramb.n_max_angular + 1) + n;
            double gn12  = g_gn_angular[index_left_all]  * weight_left + g_gn_angular[index_right_all]  * weight_right;
            double gnp12 = g_gnp_angular[index_left_all] * weight_left + g_gnp_angular[index_right_all] * weight_right;
            accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          }
#else
          int t2 = g_type[n2];
          double fc12, fcp12;
          double rc = paramb.rc_angular; double rcinv = paramb.rcinv_angular;
          if (paramb.use_typewise_cutoff) {
            int tr1 = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
            int tr2 = (paramb.spin_mode == 1) ? (t2 % paramb.num_types_real) : t2;
            rc = std::min((COVALENT_RADIUS[paramb.atomic_numbers[tr1]] + COVALENT_RADIUS[paramb.atomic_numbers[tr2]]) * paramb.typewise_cutoff_angular_factor, rc);
            rcinv = 1.0 / rc;
          }
          find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
          double fn12[MAX_NUM_N]; double fnp12[MAX_NUM_N];
          find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
          const int NT2 = (paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq;
          const int NT  = (paramb.spin_mode == 1) ? paramb.num_types_total    : paramb.num_types;
          for (int n = 0; n <= paramb.n_max_angular; ++n) {
            double gn12 = 0.0, gnp12 = 0.0;
            for (int k = 0; k <= paramb.basis_size_angular; ++k) {
              int c_index = (n * (paramb.basis_size_angular + 1) + k) * NT2;
              c_index += t1 * NT + t2 + paramb.num_c_radial;
              gn12  += fn12[k]  * annmb.c[c_index];
              gnp12 += fnp12[k] * annmb.c[c_index];
            }
            accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          }
#endif
          // Deterministic TLS accumulation
          lfx[n1] += f12[0];
          lfy[n1] += f12[1];
          lfz[n1] += f12[2];

          const bool is_pseudo = (n2 >= pseudo_start);
          if (is_pseudo) {
            int host = pseudo_to_host[n2];
            if (host >= 0) {
              pfx[host] -= f12[0];
              pfy[host] -= f12[1];
              pfz[host] -= f12[2];
            }
          } else {
            lfx[n2] -= f12[0];
            lfy[n2] -= f12[1];
            lfz[n2] -= f12[2];
          }

          int idx = n1; // accumulate_on_center expected true
          g_virial[idx + 0 * N] -= r12[0] * f12[0];
          g_virial[idx + 1 * N] -= r12[0] * f12[1];
          g_virial[idx + 2 * N] -= r12[0] * f12[2];
          g_virial[idx + 3 * N] -= r12[1] * f12[0];
          g_virial[idx + 4 * N] -= r12[1] * f12[1];
          g_virial[idx + 5 * N] -= r12[1] * f12[2];
          g_virial[idx + 6 * N] -= r12[2] * f12[0];
          g_virial[idx + 7 * N] -= r12[2] * f12[1];
          g_virial[idx + 8 * N] -= r12[2] * f12[2];
        }
      }
    }

    // reduction of TLS atom forces (deterministic)
    for (int i = 0; i < N; ++i) {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (int t = 0; t < nthreads; ++t) {
        sx += tls_fx[static_cast<size_t>(t) * N + i];
        sy += tls_fy[static_cast<size_t>(t) * N + i];
        sz += tls_fz[static_cast<size_t>(t) * N + i];
      }
      if (g_fx) g_fx[i] += sx;
      if (g_fy) g_fy[i] += sy;
      if (g_fz) g_fz[i] += sz;
    }

    // reduction of TLS pseudo reactions
    for (int i = 0; i < N; ++i) {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (int t = 0; t < nthreads; ++t) {
        sx += tls_pfx[static_cast<size_t>(t) * N + i];
        sy += tls_pfy[static_cast<size_t>(t) * N + i];
        sz += tls_pfz[static_cast<size_t>(t) * N + i];
      }
      g_pseudo_fx[i] += sx;
      g_pseudo_fy[i] += sy;
      g_pseudo_fz[i] += sz;
    }
    return;
  }
#endif
  const int NT  = (paramb.spin_mode == 1) ? paramb.num_types_total : paramb.num_types;
  const int NT2 = (paramb.spin_mode == 1) ? paramb.num_types_sq_total : paramb.num_types_sq;
  const double tiny = 1e-12;
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
      int n2 = g_NL_angular[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (d12 <= tiny) continue; // avoid singularities
      double f12[3] = {0.0};
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
      if (paramb.spin_mode == 1) {
        std::cout << "Tabulated radial/angular functions are not supported in spin mode." << std::endl;
        exit(1);
      }
      int index_left, index_right;
      double weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * NT + g_type[n2];
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        int index_left_all =
          (index_left * NT2 + t12) * (paramb.n_max_angular + 1) + n;
        int index_right_all =
          (index_right * NT2 + t12) * (paramb.n_max_angular + 1) + n;
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
      double rc = paramb.rc_angular;
      double rcinv = paramb.rcinv_angular;
      if (paramb.use_typewise_cutoff) {
        int tr1 = (paramb.spin_mode == 1) ? (t1 % paramb.num_types_real) : t1;
        int tr2 = (paramb.spin_mode == 1) ? (t2 % paramb.num_types_real) : t2;
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[tr1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[tr2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
        rcinv = 1.0 / rc;
      }
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * NT2;
          c_index += t1 * NT + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp,
          sum_fxyz, f12);
      }
#endif

      if (!pseudo_mode) {
        if (g_fx) { g_fx[n1] += f12[0]; g_fx[n2] -= f12[0]; }
        if (g_fy) { g_fy[n1] += f12[1]; g_fy[n2] -= f12[1]; }
        if (g_fz) { g_fz[n1] += f12[2]; g_fz[n2] -= f12[2]; }
      } else {
        const bool is_pseudo = (n2 >= pseudo_start);
        if (g_fx) {
          g_fx[n1] += f12[0];
          if (is_pseudo) { int host = pseudo_to_host[n2]; if (host >= 0) g_pseudo_fx[host] -= f12[0]; }
          else { g_fx[n2] -= f12[0]; }
        }
        if (g_fy) {
          g_fy[n1] += f12[1];
          if (is_pseudo) { int host = pseudo_to_host[n2]; if (host >= 0) g_pseudo_fy[host] -= f12[1]; }
          else { g_fy[n2] -= f12[1]; }
        }
        if (g_fz) {
          g_fz[n1] += f12[2];
          if (is_pseudo) { int host = pseudo_to_host[n2]; if (host >= 0) g_pseudo_fz[host] -= f12[2]; }
          else { g_fz[n2] -= f12[2]; }
        }
      }

      if (!is_dipole) {
        int idx = accumulate_on_center ? n1 : n2;
        g_virial[idx + 0 * N] -= r12[0] * f12[0];
        g_virial[idx + 1 * N] -= r12[0] * f12[1];
        g_virial[idx + 2 * N] -= r12[0] * f12[2];
        g_virial[idx + 3 * N] -= r12[1] * f12[0];
        g_virial[idx + 4 * N] -= r12[1] * f12[1];
        g_virial[idx + 5 * N] -= r12[1] * f12[2];
        g_virial[idx + 6 * N] -= r12[2] * f12[0];
        g_virial[idx + 7 * N] -= r12[2] * f12[1];
        g_virial[idx + 8 * N] -= r12[2] * f12[2];
      } else {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        int idx = accumulate_on_center ? n1 : n2;
        g_virial[idx + 0 * N] -= r12_square * f12[0];
        g_virial[idx + 1 * N] -= r12_square * f12[1];
        g_virial[idx + 2 * N] -= r12_square * f12[2];
      }
    }
  }
}

void find_force_ZBL_small_box(
  const int N,
  NEP3::ParaMB& paramb,
  const NEP3::ZBL& zbl,
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
  double* g_pe,
  bool skip_pseudo_pairs = false,
  bool accumulate_on_center = false,
  // spin-optimized extras
  int pseudo_start = -1,
  const int* pseudo_to_host = nullptr,
  double* g_pseudo_fx = nullptr,
  double* g_pseudo_fy = nullptr,
  double* g_pseudo_fz = nullptr)
{
  const bool pseudo_mode = (pseudo_to_host && pseudo_start >= 0 && g_pseudo_fx && g_pseudo_fy && g_pseudo_fz);
  for (int n1 = 0; n1 < N; ++n1) {
    int type1_all = g_type[n1];
    int type1 = (paramb.spin_mode == 1) ? (type1_all % paramb.num_types_real) : type1_all;
    int zi = paramb.atomic_numbers[type1] + 1;
    double pow_zi = pow(double(zi), 0.23);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      if (paramb.spin_mode == 1 && skip_pseudo_pairs) {
        // Skip any interaction involving a pseudo atom (type >= num_types_real)
        if (g_type[n2] >= paramb.num_types_real) continue;
      }
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f, fp;
      int type2_all = g_type[n2];
      int type2 = (paramb.spin_mode == 1) ? (type2_all % paramb.num_types_real) : type2_all;
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
          rc_inner = rc_outer * 0.5f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      if (!pseudo_mode) {
        g_fx[n1] += f12[0];
        g_fy[n1] += f12[1];
        g_fz[n1] += f12[2];
        g_fx[n2] -= f12[0];
        g_fy[n2] -= f12[1];
        g_fz[n2] -= f12[2];
      } else {
        const bool is_pseudo = (n2 >= pseudo_start);
        g_fx[n1] += f12[0];
        g_fy[n1] += f12[1];
        g_fz[n1] += f12[2];
        if (is_pseudo) {
          int host = pseudo_to_host[n2];
          if (host >= 0) {
            g_pseudo_fx[host] -= f12[0];
            g_pseudo_fy[host] -= f12[1];
            g_pseudo_fz[host] -= f12[2];
          }
        } else {
          g_fx[n2] -= f12[0];
          g_fy[n2] -= f12[1];
          g_fz[n2] -= f12[2];
        }
      }
      {
        int idx = accumulate_on_center ? n1 : n2;
        g_virial[idx + 0 * N] -= r12[0] * f12[0];
        g_virial[idx + 1 * N] -= r12[0] * f12[1];
        g_virial[idx + 2 * N] -= r12[0] * f12[2];
        g_virial[idx + 3 * N] -= r12[1] * f12[0];
        g_virial[idx + 4 * N] -= r12[1] * f12[1];
        g_virial[idx + 5 * N] -= r12[1] * f12[2];
        g_virial[idx + 6 * N] -= r12[2] * f12[0];
        g_virial[idx + 7 * N] -= r12[2] * f12[1];
        g_virial[idx + 8 * N] -= r12[2] * f12[2];
      }
      g_pe[n1] += f * 0.5;
    }
  }
}

void find_dftd3_coordination_number(
  NEP3::DFTD3& dftd3,
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
  NEP3::DFTD3& dftd3,
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
  const NEP3::DFTD3& dftd3,
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
  NEP3::ParaMB& paramb,
  NEP3::ANN& annmb,
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
      int n2 = g_NL[n1][i1];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= paramb.rc_radial * paramb.rc_radial) {
        continue;
      }
      double d12 = sqrt(d12sq);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention

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
      double rc = paramb.rc_radial;
      double rcinv = paramb.rcinv_radial;
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
        rcinv = 1.0 / rc;
      }
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
        int n2 = g_NL[n1][i1];
        double r12[3] = {
          g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

        double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        if (d12sq >= paramb.rc_angular * paramb.rc_angular) {
          continue;
        }
        double d12 = sqrt(d12sq);
        int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention

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
        double rc = paramb.rc_angular;
        double rcinv = paramb.rcinv_angular;
        if (paramb.use_typewise_cutoff) {
          rc = std::min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
          rcinv = 1.0 / rc;
        }
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
  NEP3::ParaMB& paramb,
  NEP3::ANN& annmb,
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
      int n2 = g_NL[n1][i1];
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= paramb.rc_radial * paramb.rc_radial) {
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
      double rc = paramb.rc_radial;
      double rcinv = paramb.rcinv_radial;
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
        rcinv = 1.0 / rc;
      }
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
  NEP3::ParaMB& paramb,
  NEP3::ANN& annmb,
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
      int n2 = g_NL[n1][i1];
      double r12[3] = {
        g_pos[n2][0] - g_pos[n1][0], g_pos[n2][1] - g_pos[n1][1], g_pos[n2][2] - g_pos[n1][2]};

      double d12sq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12sq >= paramb.rc_angular * paramb.rc_angular) {
        continue;
      }
      double d12 = sqrt(d12sq);
      int t2 = type_map[g_type[n2]]; // from LAMMPS to NEP convention
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
      double rc = paramb.rc_angular;
      double rcinv = paramb.rcinv_angular;
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
        rcinv = 1.0 / rc;
      }
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
  NEP3::ParaMB& paramb,
  const NEP3::ZBL& zbl,
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
      int n2 = g_NL[n1][i1];
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
          rc_inner = rc_outer * 0.5f;
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

double get_area_one_direction(const double* a, const double* b)
{
  double s1 = a[1] * b[2] - a[2] * b[1];
  double s2 = a[2] * b[0] - a[0] * b[2];
  double s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

double get_area(const int d, const double* cpu_h)
{
  double area;
  double a[3] = {cpu_h[0], cpu_h[3], cpu_h[6]};
  double b[3] = {cpu_h[1], cpu_h[4], cpu_h[7]};
  double c[3] = {cpu_h[2], cpu_h[5], cpu_h[8]};
  if (d == 0) {
    area = get_area_one_direction(b, c);
  } else if (d == 1) {
    area = get_area_one_direction(c, a);
  } else {
    area = get_area_one_direction(a, b);
  }
  return area;
}

double get_det(const double* cpu_h)
{
  return cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
         cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
         cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
}

double get_volume(const double* cpu_h) { return abs(get_det(cpu_h)); }

void get_inverse(double* cpu_h)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  double det = get_det(cpu_h);
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= det;
  }
}

bool get_expanded_box(const double rc, const double* box, int* num_cells, double* ebox)
{
  double volume = get_volume(box);
  double thickness_x = volume / get_area(0, box);
  double thickness_y = volume / get_area(1, box);
  double thickness_z = volume / get_area(2, box);
  num_cells[0] = int(ceil(2.0 * rc / thickness_x));
  num_cells[1] = int(ceil(2.0 * rc / thickness_y));
  num_cells[2] = int(ceil(2.0 * rc / thickness_z));

  bool is_small_box = false;
  if (thickness_x <= 2.5 * rc) {
    is_small_box = true;
  }
  if (thickness_y <= 2.5 * rc) {
    is_small_box = true;
  }
  if (thickness_z <= 2.5 * rc) {
    is_small_box = true;
  }

  ebox[0] = box[0] * num_cells[0];
  ebox[3] = box[3] * num_cells[0];
  ebox[6] = box[6] * num_cells[0];
  ebox[1] = box[1] * num_cells[1];
  ebox[4] = box[4] * num_cells[1];
  ebox[7] = box[7] * num_cells[1];
  ebox[2] = box[2] * num_cells[2];
  ebox[5] = box[5] * num_cells[2];
  ebox[8] = box[8] * num_cells[2];

  get_inverse(ebox);

  return is_small_box;
}

void applyMicOne(double& x12)
{
  while (x12 < -0.5)
    x12 += 1.0;
  while (x12 > +0.5)
    x12 -= 1.0;
}

void apply_mic_small_box(const double* ebox, double& x12, double& y12, double& z12)
{
  double sx12 = ebox[9] * x12 + ebox[10] * y12 + ebox[11] * z12;
  double sy12 = ebox[12] * x12 + ebox[13] * y12 + ebox[14] * z12;
  double sz12 = ebox[15] * x12 + ebox[16] * y12 + ebox[17] * z12;
  applyMicOne(sx12);
  applyMicOne(sy12);
  applyMicOne(sz12);
  x12 = ebox[0] * sx12 + ebox[1] * sy12 + ebox[2] * sz12;
  y12 = ebox[3] * sx12 + ebox[4] * sy12 + ebox[5] * sz12;
  z12 = ebox[6] * sx12 + ebox[7] * sy12 + ebox[8] * sz12;
}

void findCell(
  const double* box,
  const double* thickness,
  const double* r,
  double cutoffInverse,
  const int* numCells,
  int* cell)
{
  double s[3];
  s[0] = box[9] * r[0] + box[10] * r[1] + box[11] * r[2];
  s[1] = box[12] * r[0] + box[13] * r[1] + box[14] * r[2];
  s[2] = box[15] * r[0] + box[16] * r[1] + box[17] * r[2];
  for (int d = 0; d < 3; ++d) {
    cell[d] = floor(s[d] * thickness[d] * cutoffInverse);
    if (cell[d] < 0)
      cell[d] += numCells[d];
    if (cell[d] >= numCells[d])
      cell[d] -= numCells[d];
  }
  cell[3] = cell[0] + numCells[0] * (cell[1] + numCells[1] * cell[2]);
}

void applyPbcOne(double& sx)
{
  while (sx < 0.0) {
    sx += 1.0;
  }
  while (sx > 1.0) {
    sx -= 1.0;
  }
}

void applyPbc(const int N, const double* box, double* x, double* y, double* z)
{
  for (int n = 0; n < N; ++n) {
    double sx = box[9] * x[n] + box[10] * y[n] + box[11] * z[n];
    double sy = box[12] * x[n] + box[13] * y[n] + box[14] * z[n];
    double sz = box[15] * x[n] + box[16] * y[n] + box[17] * z[n];
    applyPbcOne(sx);
    applyPbcOne(sy);
    applyPbcOne(sz);
    x[n] = box[0] * sx + box[1] * sy + box[2] * sz;
    y[n] = box[3] * sx + box[4] * sy + box[5] * sz;
    z[n] = box[6] * sx + box[7] * sy + box[8] * sz;
  }
}

void find_neighbor_list_large_box(
  const double rc_radial,
  const double rc_angular,
  const int N,
  const std::vector<double>& box,
  const std::vector<double>& position,
  int* num_cells,
  double* ebox,
  std::vector<int>& g_NN_radial,
  std::vector<int>& g_NL_radial,
  std::vector<int>& g_NN_angular,
  std::vector<int>& g_NL_angular,
  std::vector<double>& r12)
{
  const int size_x12 = N * MN;
  std::vector<double> position_copy(position);
  double* g_x = position_copy.data();
  double* g_y = position_copy.data() + N;
  double* g_z = position_copy.data() + N * 2;
  double* g_x12_radial = r12.data();
  double* g_y12_radial = r12.data() + size_x12;
  double* g_z12_radial = r12.data() + size_x12 * 2;
  double* g_x12_angular = r12.data() + size_x12 * 3;
  double* g_y12_angular = r12.data() + size_x12 * 4;
  double* g_z12_angular = r12.data() + size_x12 * 5;

  applyPbc(N, ebox, g_x, g_y, g_z);

  const double cutoffInverse = 2.0 / rc_radial;
  double thickness[3];
  double volume = get_volume(box.data());
  thickness[0] = volume / get_area(0, box.data());
  thickness[1] = volume / get_area(1, box.data());
  thickness[2] = volume / get_area(2, box.data());

  int numCells[4];

  for (int d = 0; d < 3; ++d) {
    numCells[d] = floor(thickness[d] * cutoffInverse);
  }

  numCells[3] = numCells[0] * numCells[1] * numCells[2];
  int cell[4];

  std::vector<int> cellCount(numCells[3], 0);
  std::vector<int> cellCountSum(numCells[3], 0);

  for (int n = 0; n < N; ++n) {
    const double r[3] = {g_x[n], g_y[n], g_z[n]};
    findCell(ebox, thickness, r, cutoffInverse, numCells, cell);
    ++cellCount[cell[3]];
  }

  for (int i = 1; i < numCells[3]; ++i) {
    cellCountSum[i] = cellCountSum[i - 1] + cellCount[i - 1];
  }

  std::fill(cellCount.begin(), cellCount.end(), 0);

  std::vector<int> cellContents(N, 0);

  for (int n = 0; n < N; ++n) {
    const double r[3] = {g_x[n], g_y[n], g_z[n]};
    findCell(ebox, thickness, r, cutoffInverse, numCells, cell);
    cellContents[cellCountSum[cell[3]] + cellCount[cell[3]]] = n;
    ++cellCount[cell[3]];
  }

  for (int n1 = 0; n1 < N; ++n1) {
    int count_radial = 0;
    int count_angular = 0;
    const double r1[3] = {g_x[n1], g_y[n1], g_z[n1]};
    findCell(ebox, thickness, r1, cutoffInverse, numCells, cell);
    for (int k = -2; k <= 2; ++k) {
      for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
          int neighborCell = cell[3] + (k * numCells[1] + j) * numCells[0] + i;
          if (cell[0] + i < 0)
            neighborCell += numCells[0];
          if (cell[0] + i >= numCells[0])
            neighborCell -= numCells[0];
          if (cell[1] + j < 0)
            neighborCell += numCells[1] * numCells[0];
          if (cell[1] + j >= numCells[1])
            neighborCell -= numCells[1] * numCells[0];
          if (cell[2] + k < 0)
            neighborCell += numCells[3];
          if (cell[2] + k >= numCells[2])
            neighborCell -= numCells[3];

          for (int m = 0; m < cellCount[neighborCell]; ++m) {
            const int n2 = cellContents[cellCountSum[neighborCell] + m];
            if (n1 != n2) {
              double x12 = g_x[n2] - r1[0];
              double y12 = g_y[n2] - r1[1];
              double z12 = g_z[n2] - r1[2];
              apply_mic_small_box(ebox, x12, y12, z12);
              const double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
              if (distance_square < rc_radial * rc_radial) {
                g_NL_radial[count_radial * N + n1] = n2;
                g_x12_radial[count_radial * N + n1] = x12;
                g_y12_radial[count_radial * N + n1] = y12;
                g_z12_radial[count_radial * N + n1] = z12;
                count_radial++;
              }
              if (distance_square < rc_angular * rc_angular) {
                g_NL_angular[count_angular * N + n1] = n2;
                g_x12_angular[count_angular * N + n1] = x12;
                g_y12_angular[count_angular * N + n1] = y12;
                g_z12_angular[count_angular * N + n1] = z12;
                count_angular++;
              }
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

void find_neighbor_list_small_box(
  const double rc_radial,
  const double rc_angular,
  const int N,
  const std::vector<double>& box,
  const std::vector<double>& position,
  int* num_cells,
  double* ebox,
  std::vector<int>& g_NN_radial,
  std::vector<int>& g_NL_radial,
  std::vector<int>& g_NN_angular,
  std::vector<int>& g_NL_angular,
  std::vector<double>& r12)
{
  bool is_small_box = get_expanded_box(rc_radial, box.data(), num_cells, ebox);

  if (!is_small_box) {
    find_neighbor_list_large_box(
      rc_radial, rc_angular, N, box, position, num_cells, ebox, g_NN_radial, g_NL_radial,
      g_NN_angular, g_NL_angular, r12);
    return;
  }

  const int size_x12 = N * MN;
  const double* g_x = position.data();
  const double* g_y = position.data() + N;
  const double* g_z = position.data() + N * 2;
  double* g_x12_radial = r12.data();
  double* g_y12_radial = r12.data() + size_x12;
  double* g_z12_radial = r12.data() + size_x12 * 2;
  double* g_x12_angular = r12.data() + size_x12 * 3;
  double* g_y12_angular = r12.data() + size_x12 * 4;
  double* g_z12_angular = r12.data() + size_x12 * 5;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = 0; n2 < N; ++n2) {
      for (int ia = 0; ia < num_cells[0]; ++ia) {
        for (int ib = 0; ib < num_cells[1]; ++ib) {
          for (int ic = 0; ic < num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }

            double delta[3];
            delta[0] = box[0] * ia + box[1] * ib + box[2] * ic;
            delta[1] = box[3] * ia + box[4] * ib + box[5] * ic;
            delta[2] = box[6] * ia + box[7] * ib + box[8] * ic;

            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(ebox, x12, y12, z12);

            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < rc_radial * rc_radial) {
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = x12;
              g_y12_radial[count_radial * N + n1] = y12;
              g_z12_radial[count_radial * N + n1] = z12;
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = x12;
              g_y12_angular[count_angular * N + n1] = y12;
              g_z12_angular[count_angular * N + n1] = z12;
              count_angular++;
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
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

 

NEP3::NEP3() {}

NEP3::NEP3(const std::string& potential_filename) { init_from_file(potential_filename, true); }

void NEP3::init_from_file(const std::string& potential_filename, const bool is_rank_0)
{
  std::ifstream input(potential_filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << potential_filename << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
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
  } else if (tokens[0] == "nep4_spin") {
    paramb.model_type = 0;
    paramb.version = 4;
    paramb.spin_mode = 1;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl_spin") {
    paramb.model_type = 0;
    paramb.version = 4;
    paramb.spin_mode = 1;
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
  }

  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    print_tokens(tokens);
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  element_list.resize(paramb.num_types);
  for (int n = 0; n < paramb.num_types; ++n) {
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

  // zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      print_tokens(tokens);
      std::cout << "This line should be zbl rc_inner rc_outer." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
    }
  }

  // Optional: virtual_scale (spin model only)
  if (paramb.spin_mode == 1) {
    paramb.has_spin = 1;
    // Peek next line; it can be either 'virtual_scale ...' or 'cutoff ...'
    std::streampos pos = input.tellg();
    std::vector<std::string> peek = get_tokens(input);
    if (!peek.empty() && peek[0] == "virtual_scale") {
      if ((int)peek.size() != 1 + paramb.num_types) {
        print_tokens(peek);
        std::cout << "virtual_scale expects " << paramb.num_types << " values." << std::endl;
        exit(1);
      }
      paramb.virtual_scale_by_type.resize(paramb.num_types);
      for (int i = 0; i < paramb.num_types; ++i) {
        paramb.virtual_scale_by_type[i] = get_double_from_token(peek[1 + i], __FILE__, __LINE__);
      }
      // compatibility alias
      paramb.virtual_scale = paramb.virtual_scale_by_type;
      paramb.is_virtual_scale_set = true;
    } else {
      // rewind if not virtual_scale
      input.clear();
      input.seekg(pos);
      paramb.is_virtual_scale_set = false;
    }
  } else {
    paramb.has_spin = 0;
  }

  // cutoff 4.2 3.7 80 47
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != 8) {
    print_tokens(tokens);
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                 "[radial_factor] [angular_factor] [zbl_factor].\n";
    exit(1);
  }
  paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);  // not used
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__); // not used
  if (tokens.size() == 8) {
    paramb.typewise_cutoff_radial_factor = get_double_from_token(tokens[5], __FILE__, __LINE__);
    paramb.typewise_cutoff_angular_factor = get_double_from_token(tokens[6], __FILE__, __LINE__);
    paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[7], __FILE__, __LINE__);
    if (paramb.typewise_cutoff_radial_factor > 0.0) {
      paramb.use_typewise_cutoff = true;
    }
    if (paramb.typewise_cutoff_zbl_factor > 0.0) {
      paramb.use_typewise_cutoff_zbl = true;
    }
  }

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    print_tokens(tokens);
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

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

  // calculated parameters:
  paramb.rcinv_radial = 1.0 / paramb.rc_radial;
  paramb.rcinv_angular = 1.0 / paramb.rc_angular;
  paramb.num_types_real = paramb.num_types;
  paramb.num_types_total = (paramb.spin_mode == 1) ? (paramb.num_types_real * 2) : paramb.num_types_real;
  paramb.num_types_sq = paramb.num_types_real * paramb.num_types_real;
  paramb.num_types_sq_total = paramb.num_types_total * paramb.num_types_total;
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
  int num_para_descriptor =
    paramb.num_types_sq_total * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                                 (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;

  paramb.num_c_radial =
    paramb.num_types_sq_total * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

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

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  if (paramb.use_typewise_cutoff) {
    std::cout << "Cannot use tabulated radial functions with typewise cutoff." << std::endl;
    exit(1);
  }
  construct_table(parameters.data());
#endif

  // only report for rank_0
  if (is_rank_0) {

    if (paramb.num_types == 1) {
      std::cout << "Use the NEP" << paramb.version << " potential with " << paramb.num_types
                << " atom type.\n";
    } else {
      std::cout << "Use the NEP" << paramb.version << " potential with " << paramb.num_types
                << " atom types.\n";
    }

    for (int n = 0; n < paramb.num_types; ++n) {
      std::cout << "    type " << n << "( " << element_list[n]
                << " with Z = " << paramb.atomic_numbers[n] + 1 << ").\n";
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
    std::cout << "    radial cutoff = " << paramb.rc_radial << " A.\n";
    std::cout << "    angular cutoff = " << paramb.rc_angular << " A.\n";
    if (paramb.use_typewise_cutoff) {
      std::cout << "    typewise cutoff is enabled with radial factor "
                << paramb.typewise_cutoff_radial_factor << " and angular factor "
                << paramb.typewise_cutoff_angular_factor << ".\n";
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

void NEP3::update_type_map(const int ntype, int* type_map, char** elements)
{
  int n = 0;
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

void NEP3::update_potential(double* parameters, ANN& ann)
{
  double* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }

  ann.b1 = pointer;
  pointer += 1;

  if (paramb.model_type == 2) {
    for (int t = 0; t < paramb.num_types; ++t) {
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
void NEP3::construct_table(double* parameters)
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

void NEP3::allocate_memory(const int N)
{
  if (num_atoms < N) {
    NN_radial.resize(N);
    NL_radial.resize(N * MN);
    NN_angular.resize(N);
    NL_angular.resize(N * MN);
    r12.resize(N * MN * 6);
    Fp.resize(N * annmb.dim);
    sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    dftd3.cn.resize(N);
    dftd3.dc6_sum.resize(N);
    dftd3.dc8_sum.resize(N);
    num_atoms = N;
  }
}

void NEP3::compute(
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

  const int N = type.size();
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

  ComputeWorkspace& W = compute_workspace_;
  W.potential.resize(N);
  W.fx.resize(N);
  W.fy.resize(N);
  W.fz.resize(N);
  W.virial.resize(static_cast<size_t>(N) * 9);

  std::fill(W.potential.begin(), W.potential.end(), 0.0);
  std::fill(W.fx.begin(), W.fx.end(), 0.0);
  std::fill(W.fy.begin(), W.fy.end(), 0.0);
  std::fill(W.fz.begin(), W.fz.end(), 0.0);
  std::fill(W.virial.begin(), W.virial.end(), 0.0);

  find_neighbor_list_small_box(
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

  find_descriptor_small_box(
    true, false, false, false, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    Fp.data(), sum_fxyz.data(), W.potential.data(), nullptr, nullptr, nullptr, false, nullptr);

  find_force_radial_small_box(
    false, paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    W.fx.data(), W.fy.data(), W.fz.data(), W.virial.data());

  find_force_angular_small_box(
    false, paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(),
    sum_fxyz.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    W.fx.data(), W.fy.data(), W.fz.data(), W.virial.data());

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, paramb, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, W.fx.data(), W.fy.data(),
      W.fz.data(), W.virial.data(), W.potential.data());
  }

  std::copy_n(W.potential.data(), N, potential.data());
  std::copy_n(W.fx.data(), N, force.data());
  std::copy_n(W.fy.data(), N, force.data() + N);
  std::copy_n(W.fz.data(), N, force.data() + 2 * N);
  for (int c = 0; c < 9; ++c) {
    std::copy_n(W.virial.data() + static_cast<size_t>(c) * N, N, virial.data() + static_cast<size_t>(c) * N);
  }
}

void NEP3::compute_with_dftd3(
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
  const int N = type.size();
  const int size_x12 = N * MN;
  set_dftd3_para_all(xc, rc_potential, rc_coordination_number);

  find_neighbor_list_small_box(
    dftd3.rc_radial, dftd3.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
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

void NEP3::compute_dftd3(
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

  const int N = type.size();
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

  for (int n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (int n = 0; n < force.size(); ++n) {
    force[n] = 0.0;
  }
  for (int n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  set_dftd3_para_all(xc, rc_potential, rc_coordination_number);

  find_neighbor_list_small_box(
    dftd3.rc_radial, dftd3.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
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

void NEP3::find_descriptor(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& descriptor)
{
  const int N = type.size();
  const int size_x12 = N * MN;

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
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
    NN_angular, NL_angular, r12);

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

void NEP3::build_spin_augmented_small_box(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  const std::vector<double>& spin,
  SpinWorkspace& S)
{
  const int N_real = static_cast<int>(type.size());
  S.N_real = N_real;

  if (N_real == 0) {
    S.N_total = 0;
    S.host2pseudo.clear();
    S.pseudo2host.clear();
    S.alpha.clear();
    S.type_total.clear();
    S.rx.clear();
    S.ry.clear();
    S.rz.clear();
    S.NN_radial.clear();
    S.NN_angular.clear();
    S.NL_radial.clear();
    S.NL_angular.clear();
    S.x12_radial.clear();
    S.y12_radial.clear();
    S.z12_radial.clear();
    S.x12_angular.clear();
    S.y12_angular.clear();
    S.z12_angular.clear();
    S.NN_radial_base.clear();
    S.NL_radial_base.clear();
    S.NN_angular_base.clear();
    S.NL_angular_base.clear();
    S.r12_base.clear();
    S.pos_real.clear();
    return;
  }

  S.alpha.resize(N_real);
  std::fill(S.alpha.begin(), S.alpha.end(), 0.0);
  const bool has_alpha_table =
    static_cast<int>(paramb.virtual_scale_by_type.size()) == paramb.num_types_real;
  if (has_alpha_table) {
    for (int i = 0; i < N_real; ++i) {
      int t = type[i];
      if (t < 0 || t >= paramb.num_types_real) {
        std::cout << "Invalid type index in build_spin_augmented_small_box." << std::endl;
        exit(1);
      }
      S.alpha[i] = paramb.virtual_scale_by_type[t];
    }
  }

  S.host2pseudo.resize(N_real);
  std::fill(S.host2pseudo.begin(), S.host2pseudo.end(), -1);

  int pseudo_count = 0;
  if (paramb.spin_mode == 1) {
    for (int i = 0; i < N_real; ++i) {
      double a = S.alpha[i];
      if (a == 0.0) continue;
      double sx = spin[i];
      double sy = spin[i + N_real];
      double sz = spin[i + 2 * N_real];
      double mag2 = sx * sx + sy * sy + sz * sz;
      if (mag2 > 0.0) {
        S.host2pseudo[i] = N_real + pseudo_count;
        ++pseudo_count;
      }
    }
  }

  S.N_total = N_real + pseudo_count;
  S.pseudo2host.resize(S.N_total);
  std::fill(S.pseudo2host.begin(), S.pseudo2host.end(), -1);
  for (int i = 0; i < N_real; ++i) {
    int p = S.host2pseudo[i];
    if (p >= 0) {
      S.pseudo2host[p] = i;
    }
  }

  S.type_total.resize(S.N_total);
  S.rx.resize(S.N_total);
  S.ry.resize(S.N_total);
  S.rz.resize(S.N_total);
  for (int i = 0; i < N_real; ++i) {
    S.type_total[i] = type[i];
    S.rx[i] = position[i];
    S.ry[i] = position[i + N_real];
    S.rz[i] = position[i + 2 * N_real];
  }
  for (int i = 0; i < N_real; ++i) {
    int p = S.host2pseudo[i];
    if (p >= 0) {
      S.type_total[p] = type[i] + paramb.num_types_real;
      double a = S.alpha[i];
      double sx = spin[i];
      double sy = spin[i + N_real];
      double sz = spin[i + 2 * N_real];
      S.rx[p] = S.rx[i] + a * sx;
      S.ry[p] = S.ry[i] + a * sy;
      S.rz[p] = S.rz[i] + a * sz;
    }
  }

  S.pos_real.resize(static_cast<size_t>(3) * N_real);
  for (int i = 0; i < N_real; ++i) {
    S.pos_real[i] = S.rx[i];
    S.pos_real[i + N_real] = S.ry[i];
    S.pos_real[i + 2 * N_real] = S.rz[i];
  }

  const size_t neigh_stride = static_cast<size_t>(N_real) * MN;
  S.NN_radial_base.resize(N_real);
  S.NL_radial_base.resize(neigh_stride);
  S.NN_angular_base.resize(N_real);
  S.NL_angular_base.resize(neigh_stride);
  S.r12_base.resize(neigh_stride * 6);

  find_neighbor_list_small_box(
    paramb.rc_radial,
    paramb.rc_angular,
    N_real,
    box,
    S.pos_real,
    S.num_cells_base,
    S.ebox_base,
    S.NN_radial_base,
    S.NL_radial_base,
    S.NN_angular_base,
    S.NL_angular_base,
    S.r12_base);

  const int size_x12_center = S.N_real * MN;
  S.NN_radial.resize(S.N_real);
  S.NN_angular.resize(S.N_real);
  S.NL_radial.resize(size_x12_center);
  S.NL_angular.resize(size_x12_center);
  S.x12_radial.resize(size_x12_center);
  S.y12_radial.resize(size_x12_center);
  S.z12_radial.resize(size_x12_center);
  S.x12_angular.resize(size_x12_center);
  S.y12_angular.resize(size_x12_center);
  S.z12_angular.resize(size_x12_center);

  const int size_x12_base = N_real * MN;
  const double* x12r_base = S.r12_base.data() + 0 * size_x12_base;
  const double* y12r_base = S.r12_base.data() + 1 * size_x12_base;
  const double* z12r_base = S.r12_base.data() + 2 * size_x12_base;
  const double* x12a_base = S.r12_base.data() + 3 * size_x12_base;
  const double* y12a_base = S.r12_base.data() + 4 * size_x12_base;
  const double* z12a_base = S.r12_base.data() + 5 * size_x12_base;
  const double* ebox = S.ebox_base;

  auto append_neighbor = [&](int n1, int& write_count, int n2, double dx, double dy, double dz,
                             bool radial) {
    if (write_count >= MN) return;
    int idx = write_count * S.N_real + n1;
    if (radial) {
      S.NL_radial[idx] = n2;
      S.x12_radial[idx] = dx;
      S.y12_radial[idx] = dy;
      S.z12_radial[idx] = dz;
    } else {
      S.NL_angular[idx] = n2;
      S.x12_angular[idx] = dx;
      S.y12_angular[idx] = dy;
      S.z12_angular[idx] = dz;
    }
    ++write_count;
  };

  const double eps2 = 1e-24;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N_real; ++n1) {
    int p1 = S.host2pseudo[n1];
    int base_cr = S.NN_radial_base[n1];
    int wr = 0;
    // cutoff helpers: respect typewise cutoffs when enabled
    auto rc2_radial = [&](int t1_total, int t2_total) -> double {
      double rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        int z1 = paramb.atomic_numbers[(t1_total % paramb.num_types_real)];
        int z2 = paramb.atomic_numbers[(t2_total % paramb.num_types_real)];
        rc = std::min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor, rc);
      }
      return rc * rc;
    };
    auto rc2_angular = [&](int t1_total, int t2_total) -> double {
      double rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        int z1 = paramb.atomic_numbers[(t1_total % paramb.num_types_real)];
        int z2 = paramb.atomic_numbers[(t2_total % paramb.num_types_real)];
        rc = std::min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor, rc);
      }
      return rc * rc;
    };
    for (int i = 0; i < base_cr; ++i) {
      int idx_b = i * N_real + n1;
      int n2 = S.NL_radial_base[idx_b];
      double dx = x12r_base[idx_b];
      double dy = y12r_base[idx_b];
      double dz = z12r_base[idx_b];
      if (dx * dx + dy * dy + dz * dz > eps2) {
        append_neighbor(n1, wr, n2, dx, dy, dz, true);
      }
    }
    if (p1 >= 0) {
      double dx = S.rx[p1] - S.rx[n1];
      double dy = S.ry[p1] - S.ry[n1];
      double dz = S.rz[p1] - S.rz[n1];
      apply_mic_small_box(ebox, dx, dy, dz);
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > eps2) {
        int t1_total = S.type_total[n1];
        int t2_total = S.type_total[p1];
        if (d2 <= rc2_radial(t1_total, t2_total)) {
          append_neighbor(n1, wr, p1, dx, dy, dz, true);
        }
      }
    }
    for (int i = 0; i < base_cr; ++i) {
      int idx_b = i * N_real + n1;
      int n2 = S.NL_radial_base[idx_b];
      int p2 = S.host2pseudo[n2];
      if (p2 >= 0) {
        double dx = x12r_base[idx_b] + (S.rx[p2] - S.rx[n2]);
        double dy = y12r_base[idx_b] + (S.ry[p2] - S.ry[n2]);
        double dz = z12r_base[idx_b] + (S.rz[p2] - S.rz[n2]);
        apply_mic_small_box(ebox, dx, dy, dz);
        double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > eps2) {
          int t1_total = S.type_total[n1];
          int t2_total = S.type_total[p2];
          if (d2 <= rc2_radial(t1_total, t2_total)) {
            append_neighbor(n1, wr, p2, dx, dy, dz, true);
          }
        }
      }
    }
    S.NN_radial[n1] = wr;

    int base_ca = S.NN_angular_base[n1];
    int wa = 0;
    for (int i = 0; i < base_ca; ++i) {
      int idx_b = i * N_real + n1;
      int n2 = S.NL_angular_base[idx_b];
      double dx = x12a_base[idx_b];
      double dy = y12a_base[idx_b];
      double dz = z12a_base[idx_b];
      if (dx * dx + dy * dy + dz * dz > eps2) {
        append_neighbor(n1, wa, n2, dx, dy, dz, false);
      }
    }
    if (p1 >= 0) {
      double dx = S.rx[p1] - S.rx[n1];
      double dy = S.ry[p1] - S.ry[n1];
      double dz = S.rz[p1] - S.rz[n1];
      apply_mic_small_box(ebox, dx, dy, dz);
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > eps2) {
        int t1_total = S.type_total[n1];
        int t2_total = S.type_total[p1];
        if (d2 <= rc2_angular(t1_total, t2_total)) {
          append_neighbor(n1, wa, p1, dx, dy, dz, false);
        }
      }
    }
    for (int i = 0; i < base_ca; ++i) {
      int idx_b = i * N_real + n1;
      int n2 = S.NL_angular_base[idx_b];
      int p2 = S.host2pseudo[n2];
      if (p2 >= 0) {
        double dx = x12a_base[idx_b] + (S.rx[p2] - S.rx[n2]);
        double dy = y12a_base[idx_b] + (S.ry[p2] - S.ry[n2]);
        double dz = z12a_base[idx_b] + (S.rz[p2] - S.rz[n2]);
        apply_mic_small_box(ebox, dx, dy, dz);
        double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > eps2) {
          int t1_total = S.type_total[n1];
          int t2_total = S.type_total[p2];
          if (d2 <= rc2_angular(t1_total, t2_total)) {
            append_neighbor(n1, wa, p2, dx, dy, dz, false);
          }
        }
      }
    }
    S.NN_angular[n1] = wa;
  }
}

void NEP3::find_descriptor(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  const std::vector<double>& spin,
  std::vector<double>& descriptor)
{
  const int N_real = (int)type.size();
  if (N_real * 3 != (int)position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }
  if (N_real * 3 != (int)spin.size()) {
    std::cout << "Type and spin sizes are inconsistent.\n";
    exit(1);
  }
  if (N_real * annmb.dim != (int)descriptor.size()) {
    std::cout << "Type and descriptor sizes are inconsistent.\n";
    exit(1);
  }

  if (paramb.spin_mode != 1) {
    find_descriptor(type, box, position, descriptor);
    return;
  }

  SpinWorkspace& S = spin_workspace_;
  build_spin_augmented_small_box(type, box, position, spin, S);

  const size_t fp_size = static_cast<size_t>(S.N_real) * annmb.dim;
  const size_t sum_size =
    static_cast<size_t>(S.N_real) * (paramb.n_max_angular + 1) * NUM_OF_ABC;

  S.Fp_tot.resize(fp_size);
  std::fill(S.Fp_tot.begin(), S.Fp_tot.end(), 0.0);

  S.sum_fxyz_tot.resize(sum_size);
  std::fill(S.sum_fxyz_tot.begin(), S.sum_fxyz_tot.end(), 0.0);

  S.descriptor_buffer.resize(fp_size);
  std::fill(S.descriptor_buffer.begin(), S.descriptor_buffer.end(), 0.0);

  find_descriptor_small_box(
    false, true, false, false, paramb, annmb, S.N_real, S.NN_radial.data(), S.NL_radial.data(),
    S.NN_angular.data(), S.NL_angular.data(), S.type_total.data(), S.x12_radial.data(),
    S.y12_radial.data(), S.z12_radial.data(), S.x12_angular.data(), S.y12_angular.data(),
    S.z12_angular.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    S.Fp_tot.data(), S.sum_fxyz_tot.data(), nullptr, S.descriptor_buffer.data(), nullptr, nullptr,
    false, nullptr);

  for (int d = 0; d < annmb.dim; ++d) {
    for (int i = 0; i < N_real; ++i) {
      descriptor[d * N_real + i] = S.descriptor_buffer[d * S.N_real + i];
    }
  }
}

void NEP3::find_latent_space(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& latent_space)
{
  const int N = type.size();
  const int size_x12 = N * MN;

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
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
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

// --- Spin-enabled small-box compute implementation ---
void NEP3::compute(
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

  const int N_real = (int)type.size();
  if (N_real * 3 != (int)position.size()) {
    std::cout << "Type and position sizes are inconsistent in compute(spin).\n";
    exit(1);
  }
  if ((int)spin.size() != N_real * 3) {
    std::cout << "Spin size mismatch in compute(spin).\n";
    exit(1);
  }
  if ((int)potential.size() != N_real || (int)force.size() != N_real * 3 ||
      (int)virial.size() != N_real * 9 || (int)mforce.size() != N_real * 3) {
    std::cout << "Output buffers have wrong sizes in compute(spin).\n";
    exit(1);
  }

  if (paramb.spin_mode != 1) {
    compute(type, box, position, potential, force, virial);
    for (int i = 0; i < N_real; ++i) {
      mforce[i] = mforce[i + N_real] = mforce[i + 2 * N_real] = 0.0;
    }
    return;
  }

  SpinWorkspace& S = spin_workspace_;
  build_spin_augmented_small_box(type, box, position, spin, S);

  const size_t fp_size = static_cast<size_t>(S.N_real) * annmb.dim;
  const size_t sum_size =
    static_cast<size_t>(S.N_real) * (paramb.n_max_angular + 1) * NUM_OF_ABC;

  S.Fp_tot.resize(fp_size);

  S.sum_fxyz_tot.resize(sum_size);

  S.potential_tot.resize(S.N_real);
  std::fill(S.potential_tot.begin(), S.potential_tot.end(), 0.0);

  S.fx.resize(S.N_real);
  S.fy.resize(S.N_real);
  S.fz.resize(S.N_real);
  std::fill(S.fx.begin(), S.fx.end(), 0.0);
  std::fill(S.fy.begin(), S.fy.end(), 0.0);
  std::fill(S.fz.begin(), S.fz.end(), 0.0);

  S.virial_tot.resize(static_cast<size_t>(S.N_real) * 9);
  std::fill(S.virial_tot.begin(), S.virial_tot.end(), 0.0);

  S.pseudo_fx.resize(S.N_real);
  S.pseudo_fy.resize(S.N_real);
  S.pseudo_fz.resize(S.N_real);
  std::fill(S.pseudo_fx.begin(), S.pseudo_fx.end(), 0.0);
  std::fill(S.pseudo_fy.begin(), S.pseudo_fy.end(), 0.0);
  std::fill(S.pseudo_fz.begin(), S.pseudo_fz.end(), 0.0);

  find_descriptor_small_box(
    true, false, false, false, paramb, annmb, S.N_real, S.NN_radial.data(), S.NL_radial.data(),
    S.NN_angular.data(), S.NL_angular.data(), S.type_total.data(), S.x12_radial.data(),
    S.y12_radial.data(), S.z12_radial.data(), S.x12_angular.data(), S.y12_angular.data(),
    S.z12_angular.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_radial.data(), gn_angular.data(),
#endif
    S.Fp_tot.data(), S.sum_fxyz_tot.data(), S.potential_tot.data(), nullptr, nullptr, nullptr,
    false, nullptr);

  // Radial forces: accumulate on center (real) and record pseudo reactions
  find_force_radial_small_box(
    false, paramb, annmb, S.N_real, S.NN_radial.data(), S.NL_radial.data(), S.type_total.data(),
    S.x12_radial.data(), S.y12_radial.data(), S.z12_radial.data(), S.Fp_tot.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gnp_radial.data(),
#endif
    S.fx.data(), S.fy.data(), S.fz.data(), S.virial_tot.data(), true,
    /*pseudo_start*/ S.N_real, /*pseudo_to_host*/ S.pseudo2host.data(),
    S.pseudo_fx.data(), S.pseudo_fy.data(), S.pseudo_fz.data(),
    /*enable_parallel_center_only*/ true);

  // Angular forces: accumulate on center (real) and record pseudo reactions
  find_force_angular_small_box(
    false, paramb, annmb, S.N_real, S.NN_angular.data(), S.NL_angular.data(), S.type_total.data(),
    S.x12_angular.data(), S.y12_angular.data(), S.z12_angular.data(), S.Fp_tot.data(),
    S.sum_fxyz_tot.data(),
#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
    gn_angular.data(), gnp_angular.data(),
#endif
    S.fx.data(), S.fy.data(), S.fz.data(), S.virial_tot.data(), true,
    /*pseudo_start*/ S.N_real, /*pseudo_to_host*/ S.pseudo2host.data(),
    S.pseudo_fx.data(), S.pseudo_fy.data(), S.pseudo_fz.data(),
    /*enable_parallel_center_only*/ true);

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      S.N_real, paramb, zbl, S.NN_angular.data(), S.NL_angular.data(), S.type_total.data(),
      S.x12_angular.data(), S.y12_angular.data(), S.z12_angular.data(), S.fx.data(), S.fy.data(),
      S.fz.data(), S.virial_tot.data(), S.potential_tot.data(), /*skip_pseudo_pairs=*/true,
      /*accumulate_on_center=*/true,
      /*pseudo_start*/ S.N_real, /*pseudo_to_host*/ S.pseudo2host.data(),
      S.pseudo_fx.data(), S.pseudo_fy.data(), S.pseudo_fz.data());
  }

  for (int i = 0; i < N_real; ++i) {
    potential[i] = S.potential_tot[i];
    force[i] = S.fx[i] + S.pseudo_fx[i];
    force[i + N_real] = S.fy[i] + S.pseudo_fy[i];
    force[i + 2 * N_real] = S.fz[i] + S.pseudo_fz[i];
    mforce[i] = S.alpha[i] * S.pseudo_fx[i];
    mforce[i + N_real] = S.alpha[i] * S.pseudo_fy[i];
    mforce[i + 2 * N_real] = S.alpha[i] * S.pseudo_fz[i];
    // extra virial contribution from pseudo displacement
    if (S.host2pseudo[i] >= 0 && S.alpha[i] != 0.0) {
      double drx = -S.alpha[i] * spin[i];
      double dry = -S.alpha[i] * spin[i + N_real];
      double drz = -S.alpha[i] * spin[i + 2 * N_real];
      double Fpx = S.pseudo_fx[i];
      double Fpy = S.pseudo_fy[i];
      double Fpz = S.pseudo_fz[i];
      S.virial_tot[i + 0 * S.N_real] += drx * Fpx;
      S.virial_tot[i + 1 * S.N_real] += drx * Fpy;
      S.virial_tot[i + 2 * S.N_real] += drx * Fpz;
      S.virial_tot[i + 3 * S.N_real] += dry * Fpx;
      S.virial_tot[i + 4 * S.N_real] += dry * Fpy;
      S.virial_tot[i + 5 * S.N_real] += dry * Fpz;
      S.virial_tot[i + 6 * S.N_real] += drz * Fpx;
      S.virial_tot[i + 7 * S.N_real] += drz * Fpy;
      S.virial_tot[i + 8 * S.N_real] += drz * Fpz;
    }
    for (int c = 0; c < 9; ++c) {
      virial[i + c * N_real] = S.virial_tot[i + c * S.N_real];
    }
  }
}

namespace {
struct SpinNeighborEntry {
  int index;
  bool is_pseudo;
  double dx;
  double dy;
  double dz;
  double dist2;
  int type;       // NEP total type (real or virtual)
  bool in_radial;
  bool in_angular;
};

 

inline void apply_mic_vec(const double* ebox_local, double& dx, double& dy, double& dz)
{
  apply_mic_small_box(ebox_local, dx, dy, dz);
}

 

inline int real_type(const NEP3::ParaMB& paramb, int t_total)
{
  return t_total % paramb.num_types_real;
}

inline void typewise_rc(const NEP3::ParaMB& paramb,
                        int t1_total,
                        int t2_total,
                        bool for_radial,
                        double base_rc,
                        double base_rcinv,
                        double& rc,
                        double& rcinv)
{
  rc = base_rc;
  rcinv = base_rcinv;
  if (!paramb.use_typewise_cutoff) return;
  int z1 = paramb.atomic_numbers[real_type(paramb, t1_total)];
  int z2 = paramb.atomic_numbers[real_type(paramb, t2_total)];
  double factor = for_radial ? paramb.typewise_cutoff_radial_factor
                             : paramb.typewise_cutoff_angular_factor;
  rc = std::min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * factor, base_rc);
  rcinv = 1.0 / rc;
}

inline void accumulate_spin_descriptor_for_center(
  const NEP3::ParaMB& paramb,
  const NEP3::ANN& annmb,
  int type1,
  const std::vector<SpinNeighborEntry>& entries,
  double* q,
  double* sum_buffer)
{
  for (const auto& nb : entries) {
    if (!nb.in_radial || nb.dist2 == 0.0) continue;
    double d12 = sqrt(nb.dist2);
    double rc, rcinv, fc12;
    typewise_rc(paramb, type1, nb.type, true, paramb.rc_radial, paramb.rcinv_radial, rc, rcinv);
    find_fc(rc, rcinv, d12, fc12);
    double fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      double gn12 = 0.0;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq_total;
        c_index += type1 * paramb.num_types_total + nb.type;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      q[n] += gn12;
    }
  }

  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    double s[NUM_OF_ABC] = {0.0};
    for (const auto& nb : entries) {
      if (!nb.in_angular || nb.dist2 == 0.0) continue;
      double d12 = sqrt(nb.dist2);
      double rc, rcinv, fc12;
      typewise_rc(paramb, type1, nb.type, false, paramb.rc_angular, paramb.rcinv_angular, rc, rcinv);
      find_fc(rc, rcinv, d12, fc12);
      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
      double gn12 = 0.0;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq_total;
        c_index += type1 * paramb.num_types_total + nb.type + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      accumulate_s(paramb.L_max, d12, nb.dx, nb.dy, nb.dz, gn12, s);
    }
    find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
    for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
      sum_buffer[n * NUM_OF_ABC + abc] = s[abc];
    }
  }
}

inline void accumulate_spin_radial_forces_for_center(
  const NEP3::ParaMB& paramb,
  const NEP3::ANN& annmb,
  int n1,
  int type1,
  const std::vector<SpinNeighborEntry>& entries,
  const double* Fp_scaled,
  double** force,
  std::vector<double>& pseudo_fx,
  std::vector<double>& pseudo_fy,
  std::vector<double>& pseudo_fz,
  double* virial_local,
  int nlocal)
{
  for (const auto& nb : entries) {
    if (!nb.in_radial || nb.dist2 == 0.0) continue;
    double d12 = sqrt(nb.dist2);
    double d12inv = 1.0 / d12;
    double rc, rcinv, fc12, fcp12;
    typewise_rc(paramb, type1, nb.type, true, paramb.rc_radial, paramb.rcinv_radial, rc, rcinv);
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    double fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
    double f12[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      double gnp12 = 0.0;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq_total;
        c_index += type1 * paramb.num_types_total + nb.type;
        gnp12 += fnp12[k] * annmb.c[c_index];
      }
      double tmp = Fp_scaled[n] * gnp12 * d12inv;
      f12[0] += tmp * nb.dx;
      f12[1] += tmp * nb.dy;
      f12[2] += tmp * nb.dz;
    }
    force[n1][0] += f12[0];
    force[n1][1] += f12[1];
    force[n1][2] += f12[2];
    if (nb.is_pseudo) {
      int host = nb.index;
      pseudo_fx[host] -= f12[0];
      pseudo_fy[host] -= f12[1];
      pseudo_fz[host] -= f12[2];
    } else {
      int n2 = nb.index;
      force[n2][0] -= f12[0];
      force[n2][1] -= f12[1];
      force[n2][2] -= f12[2];
    }
    virial_local[0 * nlocal + n1] -= nb.dx * f12[0];
    virial_local[1 * nlocal + n1] -= nb.dx * f12[1];
    virial_local[2 * nlocal + n1] -= nb.dx * f12[2];
    virial_local[3 * nlocal + n1] -= nb.dy * f12[0];
    virial_local[4 * nlocal + n1] -= nb.dy * f12[1];
    virial_local[5 * nlocal + n1] -= nb.dy * f12[2];
    virial_local[6 * nlocal + n1] -= nb.dz * f12[0];
    virial_local[7 * nlocal + n1] -= nb.dz * f12[1];
    virial_local[8 * nlocal + n1] -= nb.dz * f12[2];
  }
}

inline void accumulate_spin_angular_forces_for_center(
  const NEP3::ParaMB& paramb,
  const NEP3::ANN& annmb,
  int n1,
  int type1,
  const std::vector<SpinNeighborEntry>& entries,
  const double* Fp_ang_scaled,
  const double* sum_buffer,
  double** force,
  std::vector<double>& pseudo_fx,
  std::vector<double>& pseudo_fy,
  std::vector<double>& pseudo_fz,
  double* virial_local,
  int nlocal)
{
  for (const auto& nb : entries) {
    if (!nb.in_angular || nb.dist2 == 0.0) continue;
    double d12 = sqrt(nb.dist2);
    double rc, rcinv, fc12, fcp12;
    typewise_rc(paramb, type1, nb.type, false, paramb.rc_angular, paramb.rcinv_angular, rc, rcinv);
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    double fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
    double f12[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq_total;
        c_index += type1 * paramb.num_types_total + nb.type + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
        gnp12 += fnp12[k] * annmb.c[c_index];
      }
      double r12vec[3] = {nb.dx, nb.dy, nb.dz};
      accumulate_f12(
        paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12vec, gn12, gnp12,
        Fp_ang_scaled, sum_buffer, f12);
    }

    force[n1][0] += f12[0];
    force[n1][1] += f12[1];
    force[n1][2] += f12[2];
    if (nb.is_pseudo) {
      int host = nb.index;
      pseudo_fx[host] -= f12[0];
      pseudo_fy[host] -= f12[1];
      pseudo_fz[host] -= f12[2];
    } else {
      int n2 = nb.index;
      force[n2][0] -= f12[0];
      force[n2][1] -= f12[1];
      force[n2][2] -= f12[2];
    }
    virial_local[0 * nlocal + n1] -= nb.dx * f12[0];
    virial_local[1 * nlocal + n1] -= nb.dx * f12[1];
    virial_local[2 * nlocal + n1] -= nb.dx * f12[2];
    virial_local[3 * nlocal + n1] -= nb.dy * f12[0];
    virial_local[4 * nlocal + n1] -= nb.dy * f12[1];
    virial_local[5 * nlocal + n1] -= nb.dy * f12[2];
    virial_local[6 * nlocal + n1] -= nb.dz * f12[0];
    virial_local[7 * nlocal + n1] -= nb.dz * f12[1];
    virial_local[8 * nlocal + n1] -= nb.dz * f12[2];
  }
}

inline void build_spin_neighbors_for_center(
  const NEP3::ParaMB& paramb,
  int n1,
  int ii,
  int nall,
  const int* type_nep,
  int* numneigh,
  int** firstneigh,
  double** x,
  const double* ebox_local,
  const std::vector<int>& has_pseudo,
  const std::vector<double>& pseudo_dx,
  const std::vector<double>& pseudo_dy,
  const std::vector<double>& pseudo_dz,
  std::vector<SpinNeighborEntry>& entries)
{
  entries.clear();

  const double rc_radial_sq = paramb.rc_radial * paramb.rc_radial;
  const double rc_angular_sq = paramb.rc_angular * paramb.rc_angular;
  // robust guard to avoid zero-distance pairs which cause 1/0 in force/descriptor
  const double eps2 = 1e-24;

  int jnum = numneigh[ii];
  int* neighs = firstneigh[ii];
  for (int jj = 0; jj < jnum; ++jj) {
    int n2 = neighs[jj] & NEIGHMASK;
    if (n2 < 0 || n2 >= nall) continue;

    double dx = x[n2][0] - x[n1][0];
    double dy = x[n2][1] - x[n1][1];
    double dz = x[n2][2] - x[n1][2];
    apply_mic_vec(ebox_local, dx, dy, dz);
    double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= eps2) {
      // skip near-zero separation to prevent singularities
      continue;
    }
    bool rin = (d2 < rc_radial_sq);
    bool ain = (d2 < rc_angular_sq);
    if (rin || ain) {
      entries.push_back({n2, false, dx, dy, dz, d2, type_nep[n2], rin, ain});
    }

    if (has_pseudo[n2] && (rin || ain)) {
      int t_pseudo = type_nep[n2] + paramb.num_types_real;
      double dxp = dx + pseudo_dx[n2];
      double dyp = dy + pseudo_dy[n2];
      double dzp = dz + pseudo_dz[n2];
      apply_mic_vec(ebox_local, dxp, dyp, dzp);
      double d2p = dxp * dxp + dyp * dyp + dzp * dzp;
      if (d2p <= eps2) {
        // skip pseudo pair if it collapses onto center
        continue;
      }
      bool rinp = (d2p < rc_radial_sq);
      bool ainp = (d2p < rc_angular_sq);
      // only add pseudo neighbor when the corresponding real-real pair
      // is already within cutoff, i.e., neighbors are created by real atoms
      bool in_r_final = rin && rinp;
      bool in_a_final = ain && ainp;
      if (in_r_final || in_a_final) {
        entries.push_back({n2, true, dxp, dyp, dzp, d2p, t_pseudo, in_r_final, in_a_final});
      }
    }
  }

  if (has_pseudo[n1]) {
    int t1 = type_nep[n1];
    int t_pseudo = t1 + paramb.num_types_real;
    double dxp = pseudo_dx[n1];
    double dyp = pseudo_dy[n1];
    double dzp = pseudo_dz[n1];
    apply_mic_vec(ebox_local, dxp, dyp, dzp);
    double d2p = dxp * dxp + dyp * dyp + dzp * dzp;
    if (d2p > eps2) {
    bool rinp = (d2p < rc_radial_sq);
    bool ainp = (d2p < rc_angular_sq);
    if (rinp || ainp) {
      entries.push_back({n1, true, dxp, dyp, dzp, d2p, t_pseudo, rinp, ainp});
    }
    }
  }
}

 
} // namespace

void NEP3::compute_spin_for_lammps(
  int nlocal,
  int nall,
  int inum,
  int* ilist,
  int* numneigh,
  int** firstneigh,
  int* type,
  int* type_map,
  double** x,
  double** sp,
  const double box[9],
  double& total_potential,
  double total_virial[6],
  double* potential,
  double** force,
  double** virial,
  double** mforce)
{
 
  // debug comparison paths removed

  if (paramb.model_type != 0) {
    std::cout << "Cannot compute potential using a non-potential NEP model.\n";
    exit(1);
  }

  if (sp == nullptr) {
    std::cout << "Spin data is required for compute_spin_for_lammps.\n";
    exit(1);
  }

  if (paramb.spin_mode != 1) {
    compute_for_lammps(
      nlocal, inum, ilist, numneigh, firstneigh, type, type_map, x, total_potential, total_virial,
      potential, force, virial);
    if (mforce) {
      for (int i = 0; i < nlocal; ++i) {
        mforce[i][0] = 0.0;
        mforce[i][1] = 0.0;
        mforce[i][2] = 0.0;
      }
    }
    return;
  }

  if (nlocal <= 0 || inum <= 0 || nall <= 0) {
    if (mforce) {
      for (int i = 0; i < nlocal; ++i) {
        mforce[i][0] = 0.0;
        mforce[i][1] = 0.0;
        mforce[i][2] = 0.0;
      }
    }
    return;
  }

  if (num_atoms < nlocal) {
    Fp.resize(nlocal * annmb.dim);
    sum_fxyz.resize(nlocal * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    num_atoms = nlocal;
  }

  double ebox_local[18];
  int num_cells_local[3] = {1, 1, 1};
  // Use the larger cutoff to build the MIC box to be safe for both
  // radial and angular neighbor vectors.
  const double rc_max = std::max(paramb.rc_radial, paramb.rc_angular);
  get_expanded_box(rc_max, box, num_cells_local, ebox_local);

  const int num_types_real = paramb.num_types_real;
  const bool has_alpha_table =
    (int)paramb.virtual_scale_by_type.size() == num_types_real;

  std::vector<int> type_nep(nall, 0);
  std::vector<double> alpha_atom(nall, 0.0);
  std::vector<double> spin_x(nall, 0.0), spin_y(nall, 0.0), spin_z(nall, 0.0);
  std::vector<double> pseudo_dx(nall, 0.0), pseudo_dy(nall, 0.0), pseudo_dz(nall, 0.0);
  std::vector<int> has_pseudo(nall, 0);

  for (int idx = 0; idx < nall; ++idx) {
    int t_lammps = type[idx];
    int t_nep = type_map[t_lammps];
    type_nep[idx] = t_nep;

    double sx_raw = sp[idx][0];
    double sy_raw = sp[idx][1];
    double sz_raw = sp[idx][2];
    double mag = sp[idx][3];
    // default spin scaling: normalize raw vector to magnitude
    double sx, sy, sz;
    double raw_norm_sq = sx_raw * sx_raw + sy_raw * sy_raw + sz_raw * sz_raw;
    double scale = 1.0;
    if (mag > 0.0 && raw_norm_sq > 0.0) {
      double raw_norm = sqrt(raw_norm_sq);
      if (raw_norm > 0.0) {
        scale = mag / raw_norm;
      }
    }
    sx = sx_raw * scale;
    sy = sy_raw * scale;
    sz = sz_raw * scale;
    spin_x[idx] = sx;
    spin_y[idx] = sy;
    spin_z[idx] = sz;

    double alpha_value = 0.0;
    if (has_alpha_table) {
      if (t_nep < 0 || t_nep >= num_types_real) {
        std::cout << "Invalid type index in compute_spin_for_lammps." << std::endl;
        exit(1);
      }
      alpha_value = paramb.virtual_scale_by_type[t_nep];
    }
    alpha_atom[idx] = alpha_value;

    double mag2 = sx * sx + sy * sy + sz * sz;
    if (alpha_value != 0.0 && mag2 > 0.0) {
      has_pseudo[idx] = 1;
      pseudo_dx[idx] = alpha_value * sx;
      pseudo_dy[idx] = alpha_value * sy;
      pseudo_dz[idx] = alpha_value * sz;
    }
  }

  std::vector<double> pseudo_fx(nall, 0.0);
  std::vector<double> pseudo_fy(nall, 0.0);
  std::vector<double> pseudo_fz(nall, 0.0);



  // Fast per-center path (avoid large small-box allocations)
  std::vector<double> virial_local(9 * nlocal, 0.0);
  double virial_corr_ghost[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (mforce) {
    for (int i = 0; i < nall; ++i) {
      mforce[i][0] = 0.0;
      mforce[i][1] = 0.0;
      mforce[i][2] = 0.0;
    }
  }

  std::vector<SpinNeighborEntry> entries;
  entries.reserve(96);

  for (int ii = 0; ii < inum; ++ii) {
    int n1 = ilist[ii];
    if (n1 < 0 || n1 >= nlocal) continue;

    int type1 = type_nep[n1];
    build_spin_neighbors_for_center(
      paramb, n1, ii, nall, type_nep.data(), numneigh, firstneigh, x, ebox_local, has_pseudo,
      pseudo_dx, pseudo_dy, pseudo_dz, entries);

    double q[MAX_DIM] = {0.0};
    double sum_buffer[NUM_OF_ABC * MAX_NUM_N];
    std::fill(sum_buffer, sum_buffer + NUM_OF_ABC * MAX_NUM_N, 0.0);

    accumulate_spin_descriptor_for_center(paramb, annmb, type1, entries, q, sum_buffer);

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    double F = 0.0;
    double Fp_local[MAX_DIM] = {0.0};
    double latent_space[MAX_NEURON] = {0.0};

    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[type1], annmb.b0[type1], annmb.w1[type1], annmb.b1,
        q, F, Fp_local, latent_space);
    } else {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[type1], annmb.b0[type1], annmb.w1[type1], annmb.b1,
        q, F, Fp_local, latent_space, false, nullptr);
    }

    total_potential += F;
    if (potential) {
      potential[n1] += F;
    }

    double Fp_scaled[MAX_DIM] = {0.0};
    for (int d = 0; d < annmb.dim; ++d) {
      Fp_scaled[d] = Fp_local[d] * paramb.q_scaler[d];
    }

    // radial forces
    accumulate_spin_radial_forces_for_center(
      paramb, annmb, n1, type1, entries, Fp_scaled, force, pseudo_fx, pseudo_fy, pseudo_fz,
      virial_local.data(), nlocal);

    double Fp_ang_local[MAX_DIM_ANGULAR] = {0.0};
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp_ang_local[d] = Fp_scaled[paramb.n_max_radial + 1 + d];
    }

    // angular forces
    accumulate_spin_angular_forces_for_center(
      paramb, annmb, n1, type1, entries, Fp_ang_local, sum_buffer, force, pseudo_fx, pseudo_fy,
      pseudo_fz, virial_local.data(), nlocal);

    if (zbl.enabled) {
      int type1_total = type1;
      int zi = paramb.atomic_numbers[type1_total] + 1;
      double pow_zi = pow(double(zi), 0.23);
      for (const auto& nb : entries) {
        if (nb.is_pseudo) continue;
        if (!nb.in_angular) continue;

        int n2 = nb.index;
        int type2 = type_nep[n2];
        int zj = paramb.atomic_numbers[type2] + 1;

        double dx = nb.dx;
        double dy = nb.dy;
        double dz = nb.dz;
        double d12sq = dx * dx + dy * dy + dz * dz;
        if (d12sq == 0.0) continue;

        double d12 = sqrt(d12sq);
        double d12inv = 1.0 / d12;

        double f, fp;
        double a_inv = (pow_zi + pow(double(zj), 0.23)) * 2.134563;
        double zizj = K_C_SP * zi * zj;
        if (zbl.flexibled) {
          int t1 = std::min(type1_total, type2);
          int t2 = std::max(type1_total, type2);
          int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
          double ZBL_para[10];
          for (int i10 = 0; i10 < 10; ++i10) {
            ZBL_para[i10] = zbl.para[10 * zbl_index + i10];
          }
          find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
        } else {
          double rc_in = zbl.rc_inner;
          double rc_out = zbl.rc_outer;
          if (paramb.use_typewise_cutoff_zbl) {
            rc_out = std::min(
              (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor, rc_out);
            rc_in = rc_out * 0.5;
          }
          find_f_and_fp_zbl(zizj, a_inv, rc_in, rc_out, d12, d12inv, f, fp);
        }

        double f2 = fp * d12inv * 0.5;
        double f12[3] = {dx * f2, dy * f2, dz * f2};

        force[n1][0] += f12[0];
        force[n1][1] += f12[1];
        force[n1][2] += f12[2];
        force[n2][0] -= f12[0];
        force[n2][1] -= f12[1];
        force[n2][2] -= f12[2];

        // accumulate ZBL energy consistently with non-spin path
        total_potential += f * 0.5;
        if (potential) {
          potential[n1] += f * 0.5;
        }

        virial_local[0 * nlocal + n1] -= dx * f12[0];
        virial_local[1 * nlocal + n1] -= dx * f12[1];
        virial_local[2 * nlocal + n1] -= dx * f12[2];
        virial_local[3 * nlocal + n1] -= dy * f12[0];
        virial_local[4 * nlocal + n1] -= dy * f12[1];
        virial_local[5 * nlocal + n1] -= dy * f12[2];
        virial_local[6 * nlocal + n1] -= dz * f12[0];
        virial_local[7 * nlocal + n1] -= dz * f12[1];
        virial_local[8 * nlocal + n1] -= dz * f12[2];
      }
    }
  }

  for (int idx = 0; idx < nall; ++idx) {
    if (!has_pseudo[idx]) continue;

    double fxp = pseudo_fx[idx];
    double fyp = pseudo_fy[idx];
    double fzp = pseudo_fz[idx];
    force[idx][0] += fxp;
    force[idx][1] += fyp;
    force[idx][2] += fzp;

    double alpha_value = alpha_atom[idx];
    double drx = -alpha_value * spin_x[idx];
    double dry = -alpha_value * spin_y[idx];
    double drz = -alpha_value * spin_z[idx];
    double vxx = drx * fxp;
    double vxy = drx * fyp;
    double vxz = drx * fzp;
    double vyx = dry * fxp;
    double vyy = dry * fyp;
    double vyz = dry * fzp;
    double vzx = drz * fxp;
    double vzy = drz * fyp;
    double vzz = drz * fzp;

    if (idx < nlocal) {
      virial_local[0 * nlocal + idx] += vxx;
      virial_local[1 * nlocal + idx] += vxy;
      virial_local[2 * nlocal + idx] += vxz;
      virial_local[3 * nlocal + idx] += vyx;
      virial_local[4 * nlocal + idx] += vyy;
      virial_local[5 * nlocal + idx] += vyz;
      virial_local[6 * nlocal + idx] += vzx;
      virial_local[7 * nlocal + idx] += vzy;
      virial_local[8 * nlocal + idx] += vzz;
    } else {
      virial_corr_ghost[0] += vxx;
      virial_corr_ghost[1] += vyy;
      virial_corr_ghost[2] += vzz;
      virial_corr_ghost[3] += vxy;
      virial_corr_ghost[4] += vxz;
      virial_corr_ghost[5] += vyz;
    }

    double fm_x = alpha_value * fxp;
    double fm_y = alpha_value * fyp;
    double fm_z = alpha_value * fzp;
    if (mforce) {
      mforce[idx][0] += fm_x;
      mforce[idx][1] += fm_y;
      mforce[idx][2] += fm_z;
    }
  }

  if (virial) {
    for (int i = 0; i < nlocal; ++i) {
      virial[i][0] += virial_local[0 * nlocal + i];
      virial[i][1] += virial_local[4 * nlocal + i];
      virial[i][2] += virial_local[8 * nlocal + i];
      virial[i][3] += virial_local[1 * nlocal + i];
      virial[i][4] += virial_local[2 * nlocal + i];
      virial[i][5] += virial_local[5 * nlocal + i];
    }
  }

 
  double tv[6] = {0.0};
  for (int i = 0; i < nlocal; ++i) {
    tv[0] += virial_local[0 * nlocal + i];
    tv[1] += virial_local[4 * nlocal + i];
    tv[2] += virial_local[8 * nlocal + i];
    tv[3] += virial_local[1 * nlocal + i];
    tv[4] += virial_local[2 * nlocal + i];
    tv[5] += virial_local[5 * nlocal + i];
  }
  for (int k = 0; k < 6; ++k) {
    total_virial[k] += tv[k];
  }
  total_virial[0] += virial_corr_ghost[0];
  total_virial[1] += virial_corr_ghost[1];
  total_virial[2] += virial_corr_ghost[2];
  total_virial[3] += virial_corr_ghost[3];
  total_virial[4] += virial_corr_ghost[4];
  total_virial[5] += virial_corr_ghost[5];
  return;

}

void NEP3::find_dipole(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& dipole)
{
  if (paramb.model_type != 1) {
    std::cout << "Cannot compute dipole using a non-dipole NEP model.\n";
    exit(1);
  }

  const int N = type.size();
  const int size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  std::vector<double> potential(N);  // not used but needed for find_descriptor_small_box
  std::vector<double> virial(N * 3); // need the 3 diagonal components only

  for (int n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (int n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
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
    for (int n = 0; n < N; ++n) {
      dipole[d] += virial[d * N + n];
    }
  }
}


void NEP3::find_polarizability(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& polarizability)
{
  if (paramb.model_type != 2) {
    std::cout << "Cannot compute polarizability using a non-polarizability NEP model.\n";
    exit(1);
  }

  const int N = type.size();
  const int size_x12 = N * MN;

  if (N * 3 != position.size()) {
    std::cout << "Type and position sizes are inconsistent.\n";
    exit(1);
  }

  allocate_memory(N);
  std::vector<double> potential(N);  // not used but needed for find_descriptor_small_box
  std::vector<double> virial(N * 9); // per-atom polarizability

  for (int n = 0; n < potential.size(); ++n) {
    potential[n] = 0.0;
  }
  for (int n = 0; n < virial.size(); ++n) {
    virial[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
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
  for (int n = 0; n < N; ++n) {
    polarizability[0] += virial[0 * N + n]; // xx
    polarizability[1] += virial[4 * N + n]; // yy
    polarizability[2] += virial[8 * N + n]; // zz
    polarizability[3] += virial[1 * N + n]; // xy
    polarizability[4] += virial[5 * N + n]; // yz
    polarizability[5] += virial[6 * N + n]; // zx
  }
}

void NEP3::compute_for_lammps(
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

bool NEP3::set_dftd3_para_one(
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

void NEP3::set_dftd3_para_all(
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
