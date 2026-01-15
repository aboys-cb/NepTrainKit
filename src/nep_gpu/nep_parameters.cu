#include "nep_parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cctype>
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
  if (tokens.empty()) {
    PRINT_INPUT_ERROR("nep.txt header line is empty.");
  }
  std::string head = tokens[0];
  const bool header_is_spin = (head.find("_spin") != std::string::npos);

  if (head.find("nep3") != std::string::npos) {
    version = 3;
  } else if (head.find("nep4") != std::string::npos) {
    version = 4;
  } else if (head.find("nep5") != std::string::npos) {
    version = 5;
  } else {
    PRINT_INPUT_ERROR("nep.txt header does not specify a supported nep version.");
  }
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
    size_t start = pos + 6;
    size_t end = start;
    while (end < head.size() && std::isdigit(static_cast<unsigned char>(head[end]))) {
      ++end;
    }
    if (end == start) {
      PRINT_INPUT_ERROR("Invalid charge mode in nep.txt header.");
    }
    charge_mode = std::stoi(head.substr(start, end - start));
  }

  if (tokens.size() < 2) {
    PRINT_INPUT_ERROR("nep.txt header line is incomplete.");
  }
  num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  elements.resize(num_types);
  atomic_numbers.resize(num_types);
  if (tokens.size() < static_cast<size_t>(2 + num_types)) {
    PRINT_INPUT_ERROR("nep.txt header line does not include all element symbols.");
  }
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

  // Spin models are written as:
  //   nep*_spin <num_types> <elements...>
  //   spin_mode <0..3>
  //   spin_feature <kmax_ex> <kmax_dmi> <kmax_ani> <kmax_sia> [pmax] [ex_phi_mode] [onsite_basis_mode] [mref]
  // then the usual nep.txt fields ("cutoff", "n_max", ...).
  //
  // Default to spin_mode=1 if the header advertises *_spin but the explicit
  // line is missing (for forward/backward compatibility).
  if (header_is_spin && spin_mode == 0) {
    spin_mode = 1;
  }

  tokens = get_tokens(input);
  if (tokens.empty()) {
    PRINT_INPUT_ERROR("Unexpected EOF after nep.txt header.");
  }

  // Optional spin lines (appear before cutoff, and before any zbl line).
  while (!tokens.empty() && (tokens[0] == "spin_mode" || tokens[0] == "spin_feature")) {
    if (tokens[0] == "spin_mode") {
      if (tokens.size() != 2) {
        PRINT_INPUT_ERROR("Invalid spin_mode line in nep.txt.");
      }
      spin_mode = get_int_from_token(tokens[1], __FILE__, __LINE__);
      if (spin_mode < 0 || spin_mode > 3) {
        PRINT_INPUT_ERROR("spin_mode should be 0, 1, 2 or 3.");
      }
    } else if (tokens[0] == "spin_feature") {
      if (tokens.size() < 5 || tokens.size() > 9) {
        PRINT_INPUT_ERROR(
          "Invalid spin_feature line in nep.txt (expected: spin_feature <kmax_ex> <kmax_dmi> <kmax_ani> <kmax_sia> [onsite_pmax] [ex_phi_mode] [onsite_basis_mode] [mref]).");
      }
      spin_kmax_ex = get_int_from_token(tokens[1], __FILE__, __LINE__);
      spin_kmax_dmi = get_int_from_token(tokens[2], __FILE__, __LINE__);
      spin_kmax_ani = get_int_from_token(tokens[3], __FILE__, __LINE__);
      spin_kmax_sia = get_int_from_token(tokens[4], __FILE__, __LINE__);
      if (tokens.size() >= 6) {
        spin_pmax = get_int_from_token(tokens[5], __FILE__, __LINE__);
      }
      if (tokens.size() >= 7) {
        spin_ex_phi_mode = get_int_from_token(tokens[6], __FILE__, __LINE__);
      }
      if (tokens.size() >= 8) {
        spin_onsite_basis_mode = get_int_from_token(tokens[7], __FILE__, __LINE__);
      }
      if (tokens.size() == 9) {
        spin_mref = static_cast<float>(get_double_from_token(tokens[8], __FILE__, __LINE__));
      }
    }

    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF after nep.txt spin lines.");
    }
  }

  if (tokens[0] == "zbl") {
    if (header_is_spin) {
      PRINT_INPUT_ERROR("ZBL is not supported for *_spin models.");
    }
    enable_zbl = true;
    use_typewise_cutoff_zbl = false;
    if (tokens.size() < 3) {
      PRINT_INPUT_ERROR("Invalid zbl line in nep.txt.");
    }
    zbl_rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl_rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    flexible_zbl = (zbl_rc_inner == 0.0f && zbl_rc_outer == 0.0f);
    if (tokens.size() == 4) {
      use_typewise_cutoff_zbl = true;
      typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
    } else if (tokens.size() != 3) {
      PRINT_INPUT_ERROR("Invalid zbl line in nep.txt.");
    }
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF after nep.txt zbl line.");
    }
  }

  if (tokens[0] != "cutoff") {
    PRINT_INPUT_ERROR("Missing cutoff line in nep.txt.");
  }
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("Invalid cutoff line in nep.txt.");
  }
  {
    int max_nn_radial = -1;
    int max_nn_angular = -1;
    bool has_max_nn = false;
    if (tokens.size() >= 5) {
      int tmp_r = -1;
      int tmp_a = -1;
      if (is_valid_int(tokens[tokens.size() - 2].c_str(), &tmp_r) &&
          is_valid_int(tokens[tokens.size() - 1].c_str(), &tmp_a)) {
        has_max_nn = true;
        max_nn_radial = tmp_r;
        max_nn_angular = tmp_a;
      }
    }

    size_t cutoff_tokens_end = has_max_nn ? (tokens.size() - 2) : tokens.size();
    const size_t num_cutoff_values = cutoff_tokens_end - 1; // exclude "cutoff"

    // Keep the same vector sizing as Parameters::set_default_parameters()
    // (NUM_ELEMENTS), since downstream code assumes rc_* can be indexed by
    // element/type slot without bounds checks.
    if (static_cast<int>(rc_radial.size()) != NUM_ELEMENTS) rc_radial.resize(NUM_ELEMENTS);
    if (static_cast<int>(rc_angular.size()) != NUM_ELEMENTS) rc_angular.resize(NUM_ELEMENTS);
    rc_radial_max = 0.0f;
    rc_angular_max = 0.0f;

    if (num_types > 1 && num_cutoff_values == static_cast<size_t>(2 * num_types)) {
      has_multiple_cutoffs = true;
      for (int n = 0; n < num_types; ++n) {
        const float rc_r = static_cast<float>(
          get_double_from_token(tokens[1 + 2 * n], __FILE__, __LINE__));
        const float rc_a = static_cast<float>(
          get_double_from_token(tokens[1 + 2 * n + 1], __FILE__, __LINE__));
        rc_radial[n] = rc_r;
        rc_angular[n] = rc_a;
        if (rc_r > rc_radial_max) rc_radial_max = rc_r;
        if (rc_a > rc_angular_max) rc_angular_max = rc_a;
      }
    } else if (num_cutoff_values == 2) {
      has_multiple_cutoffs = false;
      const float rc_r = static_cast<float>(get_double_from_token(tokens[1], __FILE__, __LINE__));
      const float rc_a = static_cast<float>(get_double_from_token(tokens[2], __FILE__, __LINE__));
      for (int n = 0; n < num_types; ++n) {
        rc_radial[n] = rc_r;
        rc_angular[n] = rc_a;
      }
      rc_radial_max = rc_r;
      rc_angular_max = rc_a;
    } else {
      PRINT_INPUT_ERROR("Invalid cutoff line in nep.txt.");
    }

    (void)max_nn_radial;
    (void)max_nn_angular;
  }

  tokens = get_tokens(input); // n_max
  if (tokens.empty() || tokens[0] != "n_max" || tokens.size() < 3) {
    PRINT_INPUT_ERROR("Invalid n_max line in nep.txt.");
  }
  n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  tokens = get_tokens(input); // basis_size
  if (tokens.empty() || tokens[0] != "basis_size" || tokens.size() < 3) {
    PRINT_INPUT_ERROR("Invalid basis_size line in nep.txt.");
  }
  basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  tokens = get_tokens(input); // l_max
  if (tokens.empty() || tokens[0] != "l_max" || tokens.size() < 4) {
    PRINT_INPUT_ERROR("Invalid l_max line in nep.txt.");
  }
  L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);

  tokens = get_tokens(input); // ANN
  if (tokens.empty() || tokens[0] != "ANN" || tokens.size() < 2) {
    PRINT_INPUT_ERROR("Invalid ANN line in nep.txt.");
  }
  num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);

  calculate_parameters();

  elite.resize(number_of_variables);
  for (int n = 0; n < number_of_variables; ++n) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF while reading nep.txt parameters.");
    }
    elite[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  for (int d = 0; d < dim; ++d) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF while reading nep.txt q_scaler.");
    }
    q_scaler_cpu[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  q_scaler_gpu[0].resize(dim);
  q_scaler_gpu[0].copy_from_host(q_scaler_cpu.data());

  if (flexible_zbl) {
    for (int d = 0; d < 10 * (num_types * (num_types + 1) / 2); ++d) {
      tokens = get_tokens(input);
      if (tokens.empty()) {
        PRINT_INPUT_ERROR("Unexpected EOF while reading nep.txt zbl parameters.");
      }
      zbl_para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
  }

  input.close();
}
