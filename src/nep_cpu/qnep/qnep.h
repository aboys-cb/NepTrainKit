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

#pragma once
#include <string>
#include <vector>
#include "ewald.h"

class QNEP
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    double typewise_cutoff_radial_factor = 2.5;
    double typewise_cutoff_angular_factor = 2.0;
    double typewise_cutoff_zbl_factor = 0.65;

    int charge_mode = 0;
    double rc_radial = 0.0;
    double rc_angular = 0.0;
    double rcinv_radial = 0.0;
    double rcinv_angular = 0.0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    double q_scaler[140];
    int atomic_numbers[94];
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    int num_para_ann = 0;
    const double* w0[94];
    const double* b0[94];
    const double* w1[94];
    const double* b1;
    const double* c;
    const double* sqrt_epsilon_inf;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    int num_types;
    double rc_inner = 1.0;
    double rc_outer = 2.0;
    double para[550];
  };

  struct Charge_Para {
    int num_kpoints_max = 1;
    double alpha = 0.5; // 1 / (2 Angstrom)
    double two_alpha_over_sqrt_pi = 0.564189583547756;
    double A;
    double B;
  };

  QNEP();
  QNEP(const std::string& potential_filename);

  void init_from_file(const std::string& potential_filename, const bool is_rank_0);

  // type[num_atoms] should be integers 0, 1, ..., mapping to the atom types in nep.txt in order
  // box[9] is ordered as ax, bx, cx, ay, by, cy, az, bz, cz
  // position[num_atoms * 3] is ordered as x[num_atoms], y[num_atoms], z[num_atoms]
  // potential[num_atoms]
  // force[num_atoms * 3] is ordered as fx[num_atoms], fy[num_atoms], fz[num_atoms]
  // virial[num_atoms * 9] is ordered as v_xx[num_atoms], v_xy[num_atoms], v_xz[num_atoms],
  // v_yx[num_atoms], v_yy[num_atoms], v_yz[num_atoms], v_zx[num_atoms], v_zy[num_atoms],
  // v_zz[num_atoms]
  // descriptor[num_atoms * dim] is ordered as d0[num_atoms], d1[num_atoms], ...

  void compute(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& potential,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& charge,
    std::vector<double>& bec);

  void find_descriptor(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& descriptor);

  int num_atoms = 0;
  int num_cells[3];
  double ebox[18];
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  Charge_Para charge_para;
  Ewald ewald;
  std::vector<int> NN_radial, NL_radial, NN_angular, NL_angular;
  std::vector<double> r12;
  std::vector<double> Fp;
  std::vector<double> sum_fxyz;
  std::vector<double> D_real;
  std::vector<double> charge_derivative;
  std::vector<double> parameters;
  std::vector<std::string> element_list;
  void update_potential(double* parameters, ANN& ann);
  void allocate_memory(const int N);
};
