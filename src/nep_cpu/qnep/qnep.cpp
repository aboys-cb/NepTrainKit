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

#include "qnep.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

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

void apply_ann_one_layer_charge(
  const int N_des,
  const int N_neu,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative,
  double& charge,
  double* charge_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tanh_der = 1.0 - x1 * x1;
    energy += w1[n] * x1;
    charge += w1[n + N_neu] * x1;
    for (int d = 0; d < N_des; ++d) {
      double y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
      charge_derivative[d] += w1[n + N_neu] * y1;
    }
  }
  energy -= b1[0];
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

void find_descriptor_small_box(
  const bool calculating_potential,
  const bool calculating_descriptor,
  QNEP::ParaMB& paramb,
  QNEP::ANN& annmb,
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
  QNEP::ParaMB& paramb,
  QNEP::ANN& annmb,
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
  QNEP::ParaMB& paramb,
  QNEP::ANN& annmb,
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

void find_force_ZBL_small_box(
  const int N,
  QNEP::ParaMB& paramb,
  const QNEP::ZBL& zbl,
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
          rc_inner = rc_outer * 0.5;
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
  const QNEP::ParaMB paramb,
  const QNEP::ANN annmb,
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
      double rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) * paramb.typewise_cutoff_radial_factor,
          rc);
      }
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
  QNEP::ParaMB paramb,
  QNEP::ANN annmb,
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
      double rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        rc = std::min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
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
  const QNEP::Charge_Para charge_para,
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
  const QNEP::Charge_Para charge_para,
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

QNEP::QNEP() {}

QNEP::QNEP(const std::string& potential_filename) { init_from_file(potential_filename, true); }

void QNEP::init_from_file(const std::string& potential_filename, const bool is_rank_0)
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
  if (tokens[0] == "nep4_charge1") {
    zbl.enabled = false;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_zbl_charge1") {
    zbl.enabled = true;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_charge2") {
    zbl.enabled = false;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    zbl.enabled = true;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_charge3") {
    zbl.enabled = false;
    paramb.charge_mode = 3;
  } else if (tokens[0] == "nep4_zbl_charge3") {
    zbl.enabled = true;
    paramb.charge_mode = 3;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 charge models now."
              << std::endl;
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
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  annmb.num_para_ann = (annmb.dim + 3) * annmb.num_neurons1 * paramb.num_types + 2;

  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

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
  charge_para.alpha = PI / paramb.rc_radial; // a good value
  ewald.initialize(charge_para.alpha);
  charge_para.two_alpha_over_sqrt_pi = 2.0 * charge_para.alpha / sqrt(PI);
  charge_para.A = erfc(PI) / (paramb.rc_radial * paramb.rc_radial);
  charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-PI * PI) / paramb.rc_radial;
  charge_para.B = - erfc(PI) / paramb.rc_radial - charge_para.A * paramb.rc_radial;

  // only report for rank_0
  if (is_rank_0) {

    if (paramb.num_types == 1) {
      std::cout << "Use the NEP4-Charge" << paramb.charge_mode << " potential with " << paramb.num_types
                << " atom type.\n";
    } else {
      std::cout << "Use the NEP4-Charge" << paramb.charge_mode << " potential with " << paramb.num_types
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

void QNEP::update_potential(double* parameters, ANN& ann)
{
  double* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * 2;
  }
  ann.sqrt_epsilon_inf = pointer;
  pointer += 1;
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
}

void QNEP::allocate_memory(const int N)
{
  if (num_atoms < N) {
    NN_radial.resize(N);
    NL_radial.resize(N * MN);
    NN_angular.resize(N);
    NL_angular.resize(N * MN);
    r12.resize(N * MN * 6);
    Fp.resize(N * annmb.dim);
    sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    D_real.resize(N);
    charge_derivative.resize(N * annmb.dim);

    num_atoms = N;
  }
}

void QNEP::compute(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& potential,
  std::vector<double>& force,
  std::vector<double>& virial,
  std::vector<double>& charge,
  std::vector<double>& bec)
{
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
  if (N != charge.size()) {
    std::cout << "Type and charge sizes are inconsistent.\n";
    exit(1);
  }
  if (N * 9 != bec.size()) {
    std::cout << "Type and BEC sizes are inconsistent.\n";
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
  for (int n = 0; n < charge.size(); ++n) {
    charge[n] = 0.0;
  }
  for (int n = 0; n < bec.size(); ++n) {
    bec[n] = 0.0;
  }

  find_neighbor_list_small_box(
    paramb.rc_radial, paramb.rc_angular, N, box, position, num_cells, ebox, NN_radial, NL_radial,
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

void QNEP::find_descriptor(
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
    false, true, paramb, annmb, N, NN_radial.data(), NL_radial.data(),
    NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
    r12.data() + size_x12 * 2, r12.data() + size_x12 * 3, r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    Fp.data(), sum_fxyz.data(), nullptr, nullptr, nullptr, descriptor.data());
}
