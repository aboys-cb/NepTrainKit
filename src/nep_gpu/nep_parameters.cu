#include "nep_parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
// SPDX-License-Identifier: GPL-3.0-or-later
/*
    NepTrainKit NEP parameter utilities (loader/extractor)
    Copyright (C) 2025 NepTrainKit contributors

    This file adapts logic from GPUMD
    (https://github.com/brucefan1983/GPUMD) by Zheyong Fan and the
    GPUMD development team, licensed under the GNU General Public License
    version 3 (or later).

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

#include <iostream>
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};


void NepParameters::load_from_nep_txt(const std::string& filename, std::vector<float>& elite)
{
  set_default_parameters();
  prediction = 1;

  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  std::string head = tokens[0];

  version = (head.find("nep3") != std::string::npos) ? 3 : 4;
  enable_zbl = (head.find("zbl") != std::string::npos);

  if (head.find("dipole") != std::string::npos) {
    train_mode = 1;
  } else if (head.find("polarizability") != std::string::npos) {
    train_mode = 2;
  } else if (head.find("temperature") != std::string::npos) {
    train_mode = 3;
  } else {
    train_mode = 0;
  }

  size_t pos = head.find("charge");
  if (pos != std::string::npos) {
    charge_mode = std::stoi(head.substr(pos + 6, 1));
  }

  num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  elements.resize(num_types);
  atomic_numbers.resize(num_types);
  for (int n = 0; n < num_types; ++n) {
    elements[n] = tokens[2 + n];
    bool found = false;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (elements[n] == ELEMENTS[m]) {
        atomic_numbers[n] = m + 1;
        found = true;
        break;
      }
    }
    if (!found) {
      PRINT_INPUT_ERROR("Element not recognized in nep.txt.");
    }
  }

  tokens = get_tokens(input);
  if (tokens[0] == "zbl") {
    zbl_rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl_rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    flexible_zbl = (zbl_rc_inner == 0.0f && zbl_rc_outer == 0.0f);
    tokens = get_tokens(input);
  }

  if (tokens[0] == "cutoff") {
    rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
    rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
  }

  tokens = get_tokens(input); // n_max
  n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  tokens = get_tokens(input); // basis_size
  basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  tokens = get_tokens(input); // l_max
  L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);

  tokens = get_tokens(input); // ANN
  num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);

  calculate_parameters();

  elite.resize(number_of_variables);
  for (int n = 0; n < number_of_variables; ++n) {
    tokens = get_tokens(input);
    elite[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  for (int d = 0; d < dim; ++d) {
    tokens = get_tokens(input);
    q_scaler_cpu[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  q_scaler_gpu[0].resize(dim);
  q_scaler_gpu[0].copy_from_host(q_scaler_cpu.data());

  if (flexible_zbl) {
    for (int d = 0; d < 10 * (num_types * (num_types + 1) / 2); ++d) {
      tokens = get_tokens(input);
      zbl_para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
  }

  input.close();
}
