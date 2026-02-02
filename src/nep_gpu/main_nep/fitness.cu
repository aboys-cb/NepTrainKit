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
Get the fitness
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "nep_spin.cuh"
#include "tnep.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <cstring>

Fitness::Fitness(Parameters& para)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));

  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  printf("Number of devices = %d\n", deviceCount);
  printf("Number of batches = %d\n", num_batches);
  int batch_size_old = para.batch_size;
  para.batch_size = (structures_train.size() - 1) / num_batches + 1;
  if (batch_size_old != para.batch_size) {
    printf("Hello, I changed the batch_size from %d to %d.\n", batch_size_old, para.batch_size);
  }

  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    train_set[batch_id].resize(deviceCount);
  }
  int count = 0;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    const int batch_size_minimal = structures_train.size() / num_batches;
    const bool is_larger_batch =
      batch_id + batch_size_minimal * num_batches < structures_train.size();
    const int batch_size = is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
    count += batch_size;
    printf("\nBatch %d:\n", batch_id);
    printf("Number of configurations = %d.\n", batch_size);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing train_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      train_set[batch_id][device_id].construct(
        para, structures_train, count - batch_size, count, device_id);
      print_line_2();
    }
  }

  std::vector<Structure> structures_test;
  has_test_set = read_structures(false, para, structures_test);
  if (has_test_set) {
    test_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing test_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id);
      print_line_2();
    }
  }

  int N = -1;
  int Nc = -1;
  int N_times_max_NN_radial = -1;
  int N_times_max_NN_angular = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  if (has_test_set) {
    N = test_set[0].N;
    Nc = test_set[0].Nc;
    N_times_max_NN_radial = test_set[0].N * test_set[0].max_NN_radial;
    N_times_max_NN_angular = test_set[0].N * test_set[0].max_NN_angular;
    max_NN_radial = test_set[0].max_NN_radial;
    max_NN_angular = test_set[0].max_NN_angular;
  }
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n][0].N > N) {
      N = train_set[n][0].N;
    };
    if (train_set[n][0].Nc > Nc) {
      Nc = train_set[n][0].Nc;
    };
    if (train_set[n][0].N * train_set[n][0].max_NN_radial > N_times_max_NN_radial) {
      N_times_max_NN_radial = train_set[n][0].N * train_set[n][0].max_NN_radial;
    };
    if (train_set[n][0].N * train_set[n][0].max_NN_angular > N_times_max_NN_angular) {
      N_times_max_NN_angular = train_set[n][0].N * train_set[n][0].max_NN_angular;
    };

    if (train_set[n][0].max_NN_radial > max_NN_radial) {
      max_NN_radial = train_set[n][0].max_NN_radial;
    }
    if (train_set[n][0].max_NN_angular > max_NN_angular) {
      max_NN_angular = train_set[n][0].max_NN_angular;
    }
  }

  if (para.train_mode == 1 || para.train_mode == 2) {
    potential.reset(
      new TNEP(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
  } else if (para.spin_mode > 0) {
    potential.reset(
      new NEP_Spin(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
  } else {
    if (para.charge_mode) {
      potential.reset(
        new NEP_Charge(para, N, Nc, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
    } else {
      potential.reset(
        new NEP(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
    }
  }

  if (para.prediction == 0) {
    fid_loss_out = my_fopen("loss.out", "a");
  }
}

Fitness::~Fitness()
{
  if (fid_loss_out != NULL) {
    fclose(fid_loss_out);
  }
}

void Fitness::compute(
  const int generation,
  Parameters& para,
  const float* population,
  float* fitness_energy,
  float* fitness_force,
  float* fitness_virial,
  float* fitness_charge,
  float* fitness_bec,
  float* fitness_mforce,
  float* fitness_torque
  )
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  int population_iter = (para.population_size - 1) / deviceCount + 1;

  if (generation == 0) {
    std::vector<float> dummy_solution(para.number_of_variables * deviceCount, para.initial_para);
    for (int n = 0; n < num_batches; ++n) {
      potential->find_force(
        para,
        dummy_solution.data(),
        train_set[n],
        (para.fine_tune ? false : true),
        true,
        deviceCount);
    }
  } else {
    int batch_id = generation % num_batches;
    bool calculate_neighbor = (num_batches > 1) || (generation % 100 == 0);

    // If q_scaler contains saturated entries (e.g., still at the initial 1e10),
    // some descriptor components can blow up, saturating the NN so that Fp->0 and thus force->0.
    // Detect this cheaply (dim is small) and update q_scaler at most once per generation by
    // re-running find_max_min using the first evaluated individual of this generation.
    const bool force_update_q_scaler = []() {
      const char* s = std::getenv("NEP_UPDATE_Q_SCALER");
      return s && s[0] != '\0' && s[0] != '0';
    }();
    bool want_update_q_scaler = force_update_q_scaler;
    if (!want_update_q_scaler) {
      CHECK(gpuSetDevice(0));
      std::vector<float> q_cpu(para.q_scaler_cpu.size());
      para.q_scaler_gpu[0].copy_to_host(q_cpu.data());
      for (float v : q_cpu) {
        if (v >= 1.0e8f) { // heuristic threshold; 1e10 indicates "never updated"
          want_update_q_scaler = true;
          break;
        }
      }
    }
    bool updated_q_scaler_this_generation = false;
    for (int n = 0; n < population_iter; ++n) {
      const float* individual = population + deviceCount * n * para.number_of_variables;
      const bool calculate_q_scaler = want_update_q_scaler && (!updated_q_scaler_this_generation);
      potential->find_force(
        para, individual, train_set[batch_id], calculate_q_scaler, calculate_neighbor, deviceCount);
      if (calculate_q_scaler) {
        updated_q_scaler_this_generation = true;
      }
      for (int m = 0; m < deviceCount; ++m) {
        float energy_shift_per_structure_not_used;
        auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
          para, energy_shift_per_structure_not_used, true, true, m);
        auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
        auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
        auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);
        auto rmse_bec_array = train_set[batch_id][m].get_rmse_bec(para, m);
        std::vector<float> rmse_mforce_array(para.num_types + 1, 0.0f);
        std::vector<float> rmse_torque_array(para.num_types + 1, 0.0f);
        if (para.spin_mode > 0 &&
            train_set[batch_id][m].mforce_ref_gpu.size() == (size_t)train_set[batch_id][m].N * 3) {
          rmse_mforce_array = train_set[batch_id][m].get_rmse_mforce(para, true, m);
          if (para.lambda_tau > 0.0f) {
            rmse_torque_array = train_set[batch_id][m].get_rmse_torque(para, true, m);
          }
        }
        for (int t = 0; t <= para.num_types; ++t) {
          fitness_energy[deviceCount * n + m + t * para.population_size] =
            para.lambda_e * rmse_energy_array[t];
          fitness_force[deviceCount * n + m + t * para.population_size] =
            para.lambda_f * rmse_force_array[t];
          fitness_virial[deviceCount * n + m + t * para.population_size] =
            para.lambda_v * rmse_virial_array[t];
          fitness_charge[deviceCount * n + m + t * para.population_size] =
            para.lambda_q * rmse_charge_array[t];
          fitness_bec[deviceCount * n + m + t * para.population_size] =
            para.lambda_z * rmse_bec_array[t];
          fitness_mforce[deviceCount * n + m + t * para.population_size] =
            para.lambda_m * rmse_mforce_array[t];
          fitness_torque[deviceCount * n + m + t * para.population_size] =
            para.lambda_tau * rmse_torque_array[t];

      }
      }
    }

    if (para.use_full_batch) {
      int count_batch = 0;
      for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
        if (batch_id == generation % num_batches) {
          continue; // skip the batch that has already been calculated
        }
        ++count_batch;
        for (int n = 0; n < population_iter; ++n) {
          const float* individual = population + deviceCount * n * para.number_of_variables;
          const bool calculate_q_scaler = want_update_q_scaler && (!updated_q_scaler_this_generation);
          potential->find_force(
            para, individual, train_set[batch_id], calculate_q_scaler, calculate_neighbor, deviceCount);
          if (calculate_q_scaler) {
            updated_q_scaler_this_generation = true;
          }
          for (int m = 0; m < deviceCount; ++m) {
            float energy_shift_per_structure_not_used;
            auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
              para, energy_shift_per_structure_not_used, true, true, m);
            auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
            auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
            auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);
            auto rmse_bec_array = train_set[batch_id][m].get_rmse_bec(para, m);
            std::vector<float> rmse_mforce_array(para.num_types + 1, 0.0f);
            std::vector<float> rmse_torque_array(para.num_types + 1, 0.0f);
            if (para.spin_mode > 0 &&
                train_set[batch_id][m].mforce_ref_gpu.size() == (size_t)train_set[batch_id][m].N * 3) {
              rmse_mforce_array = train_set[batch_id][m].get_rmse_mforce(para, true, m);
              if (para.lambda_tau > 0.0f) {
                rmse_torque_array = train_set[batch_id][m].get_rmse_torque(para, true, m);
              }
            }
            for (int t = 0; t <= para.num_types; ++t) {
              // energy
              float old_value = fitness_energy[deviceCount * n + m + t * para.population_size];
              float new_value = para.lambda_e * rmse_energy_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_energy[deviceCount * n + m + t * para.population_size] = new_value;
              // force
              old_value = fitness_force[deviceCount * n + m + t * para.population_size];
              new_value = para.lambda_f * rmse_force_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_force[deviceCount * n + m + t * para.population_size] = new_value;
              // virial
              old_value = fitness_virial[deviceCount * n + m + t * para.population_size];
              new_value = para.lambda_v * rmse_virial_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_virial[deviceCount * n + m + t * para.population_size] = new_value;
              // charge
              old_value = fitness_charge[deviceCount * n + m + t * para.population_size];
              new_value = para.lambda_q * rmse_charge_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_charge[deviceCount * n + m + t * para.population_size] = new_value;
              // BEC
              old_value = fitness_bec[deviceCount * n + m + t * para.population_size];
              new_value = para.lambda_z * rmse_bec_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_bec[deviceCount * n + m + t * para.population_size] = new_value;
              // mforce
              old_value = fitness_mforce[deviceCount * n + m + t * para.population_size];
              new_value = para.lambda_m * rmse_mforce_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_mforce[deviceCount * n + m + t * para.population_size] = new_value;
              // torque
              old_value = fitness_torque[deviceCount * n + m + t * para.population_size];
              new_value = para.lambda_tau * rmse_torque_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness_torque[deviceCount * n + m + t * para.population_size] = new_value;
           }
          }
        }
      }
    }
  }
}

void Fitness::output(
  bool is_stress,
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int n = 0; n < num_components; ++n) {
      int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
      // Use double for the per-configuration reduction to reduce round-off when
      // energy differences are tiny (e.g., SOC-scale).
      double data_nc = 0.0;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        data_nc += (double)prediction[offset + m];
      }
      if (!is_stress) {
        fprintf(fid, "%.10g ", data_nc / (double)dataset.Na_cpu[nc]);
      } else {
        fprintf(
          fid,
          "%.10g ",
          data_nc / (double)dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
      }
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.Nc + nc];
      if (is_stress) {
        ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
      if (n == num_components - 1) {
        fprintf(fid, "%.10g\n", ref_value);
      } else {
        fprintf(fid, "%.10g ", ref_value);
      }
    }
  }
}

void Fitness::output_atomic(
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
for (int nc = 0; nc < dataset.Nc; ++nc) {
  int offset = dataset.Na_sum_cpu[nc];
  for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
    for (int n = 0; n < num_components; ++n) {
      int index = n * dataset.N + offset + m;
      fprintf(fid, "%g ", prediction[index]);
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.N + offset + m];
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}
}

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, float* elite)
{
  const bool is_spin_model = (para.spin_mode > 0);

  // Model header on first line
  if (is_spin_model) {
    // Spin model: use nep*_spin tag and do not advertise ZBL here.
    if (para.version == 3) {
      fprintf(fid_nep, "nep3_spin %d ", para.num_types);
    } else if (para.version == 4) {
      fprintf(fid_nep, "nep4_spin %d ", para.num_types);
    } else {
      // Fallback for future versions: still mark as *_spin but keep version number.
      fprintf(fid_nep, "nep%d_spin %d ", para.version, para.num_types);
    }
  } else if (para.train_mode == 0) { // potential model
    if (!para.charge_mode) {
      if (para.version == 3) {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep3_zbl %d ", para.num_types);
        } else {
          fprintf(fid_nep, "nep3 %d ", para.num_types);
        }
      } else if (para.version == 4) {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl %d ", para.num_types);
        } else {
          fprintf(fid_nep, "nep4 %d ", para.num_types);
        }
      } 
    } else {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep4_zbl_charge%d %d ", para.charge_mode, para.num_types);
      } else {
        fprintf(fid_nep, "nep4_charge%d %d ", para.charge_mode, para.num_types);
      }
    }
  } else if (para.train_mode == 1) { // dipole model
    if (para.version == 3) {
      fprintf(fid_nep, "nep3_dipole %d ", para.num_types);
    } else if (para.version == 4) {
      fprintf(fid_nep, "nep4_dipole %d ", para.num_types);
    }
  } else if (para.train_mode == 2) { // polarizability model
    if (para.version == 3) {
      fprintf(fid_nep, "nep3_polarizability %d ", para.num_types);
    } else if (para.version == 4) {
      fprintf(fid_nep, "nep4_polarizability %d ", para.num_types);
    }
  } else if (para.train_mode == 3) { // temperature model
    if (para.version == 3) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep3_zbl_temperature %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep3_temperature %d ", para.num_types);
      }
    } else if (para.version == 4) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep4_zbl_temperature %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep4_temperature %d ", para.num_types);
      }
    }
  }

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_nep, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_nep, "\n");

  // Spin model header (NEP spin potentials insert extra header lines)
  if (is_spin_model) {
    // Always write spin_n_max (and advertise 2 header lines) to make the output
    // robust against future default-parameter changes.
    const int spin_header_lines = 2;
    fprintf(
      fid_nep,
      "spin_mode %d %d\n",
      para.spin_mode,
      spin_header_lines);
    fprintf(
      fid_nep,
      "spin_feature %d %d %d %d %d %d %d %g\n",
      para.spin_kmax_ex,
      para.spin_kmax_dmi,
      para.spin_kmax_ani,
      para.spin_kmax_sia,
      para.spin_pmax,
      para.spin_ex_phi_mode,
      para.spin_onsite_basis_mode,
      para.spin_mref);

    fprintf(fid_nep, "spin_n_max %d\n", para.spin_n_max);

  }

  // For now, do not export ZBL lines for spin models (NEP_Spin does not include ZBL).
  if (para.enable_zbl && !is_spin_model) {
    if (para.flexible_zbl) {
      fprintf(fid_nep, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(fid_nep, "zbl %g %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(fid_nep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }

  fprintf(fid_nep, "cutoff %g %g ", para.rc_radial[0], para.rc_angular[0]);
  if (para.has_multiple_cutoffs) {
    for (int n = 1; n < para.num_types; ++n) {
      fprintf(fid_nep, "%g %g ", para.rc_radial[n], para.rc_angular[n]);
    }
  }
  fprintf(fid_nep, "%d %d\n", max_NN_radial, max_NN_angular);

  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_nep, "l_max %d %d %d\n", para.L_max, para.L_max_4body, para.L_max_5body);

  fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", elite[m]);
  }
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_nep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl && !is_spin_model) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_nep, "%15.7e\n", para.zbl_para[d]);
    }
  }
}

void Fitness::get_save_potential_label(Parameters& para, const int generation, std::string& label) {
    if (para.save_potential_format == 1) {
      time_t rawtime;
      time(&rawtime);
      struct tm* timeinfo = localtime(&rawtime);
      char buffer[200];
      strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
      label = std::string(buffer) + std::to_string(generation + 1);
    } else {
      label = "nep_gen" + std::to_string(generation + 1);
    }
}

void Fitness::report_error(
  Parameters& para,
  const int generation,
  const float loss_total,
  const float loss_L1,
  const float loss_L2,
  float* elite)
{
  if (0 == (generation + 1) % 100) {
    int batch_id = generation % num_batches;
    potential->find_force(para, elite, train_set[batch_id], false, true, 1);
    float energy_shift_per_structure;
    auto rmse_energy_train_array =
      train_set[batch_id][0].get_rmse_energy(para, energy_shift_per_structure, false, true, 0);
    auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(para, false, 0);
    auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(para, false, 0);
    auto rmse_charge_train_array = train_set[batch_id][0].get_rmse_charge(para, 0);
    auto rmse_bec_train_array = train_set[batch_id][0].get_rmse_bec(para, 0);
    std::vector<float> rmse_mforce_train_array;
    std::vector<float> rmse_torque_train_array;
    const bool has_mforce_train_data =
      (para.spin_mode > 0) &&
      (train_set[batch_id][0].mforce_ref_gpu.size() == (size_t)train_set[batch_id][0].N * 3);
    if (has_mforce_train_data) {
      rmse_mforce_train_array = train_set[batch_id][0].get_rmse_mforce(para, false, 0);
      if (para.lambda_tau > 0.0f) {
        rmse_torque_train_array = train_set[batch_id][0].get_rmse_torque(para, false, 0);
      }
    }

    float rmse_energy_train = rmse_energy_train_array.back();
    float rmse_force_train = rmse_force_train_array.back();
    float rmse_virial_train = rmse_virial_train_array.back();
    float rmse_charge_train = rmse_charge_train_array.back();
    float rmse_bec_train = rmse_bec_train_array.back();
    float rmse_mforce_train = rmse_mforce_train_array.empty() ? 0.0f : rmse_mforce_train_array.back();
    float rmse_torque_train = rmse_torque_train_array.empty() ? 0.0f : rmse_torque_train_array.back();

    // correct the last bias parameter in the NN
    // Apply for potential(0), temperature(3), and spin so saved model energies are offset-free.
    if (para.train_mode == 0 || para.train_mode == 3 || (para.spin_mode > 0)) {
      elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
    }

    float rmse_energy_test = 0.0f;
    float rmse_force_test = 0.0f;
    float rmse_virial_test = 0.0f;
    float rmse_charge_test = 0.0f;
    float rmse_bec_test = 0.0f;
    float rmse_mforce_test = 0.0f;
    float rmse_torque_test = 0.0f;
    bool has_mforce_test_data = false;
    if (has_test_set) {
      potential->find_force(para, elite, test_set, false, true, 1);
      float energy_shift_per_structure_not_used;
      auto rmse_energy_test_array =
        test_set[0].get_rmse_energy(para, energy_shift_per_structure_not_used, false, false, 0);
      auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
      auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
      auto rmse_charge_test_array = test_set[0].get_rmse_charge(para, 0);
      auto rmse_bec_test_array = test_set[0].get_rmse_bec(para, 0);
      std::vector<float> rmse_mforce_test_array;
      std::vector<float> rmse_torque_test_array;
      has_mforce_test_data =
        (para.spin_mode > 0) && (test_set[0].mforce_ref_gpu.size() == (size_t)test_set[0].N * 3);
      if (has_mforce_test_data) {
        rmse_mforce_test_array = test_set[0].get_rmse_mforce(para, false, 0);
        if (para.lambda_tau > 0.0f) {
          rmse_torque_test_array = test_set[0].get_rmse_torque(para, false, 0);
        }
      }
      rmse_energy_test = rmse_energy_test_array.back();
      rmse_force_test = rmse_force_test_array.back();
      rmse_virial_test = rmse_virial_test_array.back();
      rmse_charge_test = rmse_charge_test_array.back();
      rmse_bec_test = rmse_bec_test_array.back();
      rmse_mforce_test = rmse_mforce_test_array.empty() ? 0.0f : rmse_mforce_test_array.back();
      rmse_torque_test = rmse_torque_test_array.empty() ? 0.0f : rmse_torque_test_array.back();
    }

    FILE* fid_nep = my_fopen("nep.txt", "w");
    write_nep_txt(fid_nep, para, elite);
    fclose(fid_nep);

    if (0 == (generation + 1) % para.save_potential) {
      std::string filename;
      get_save_potential_label(para, generation, filename);
      filename += ".txt";

      FILE* fid_nep = my_fopen(filename.c_str(), "w");
      write_nep_txt(fid_nep, para, elite);
      fclose(fid_nep);
    }

    if (para.train_mode == 0 || para.train_mode == 3) {
      if ((para.spin_mode > 0) && (has_mforce_train_data || has_mforce_test_data)) {
      // Spin mode with mforce labels: report E/F/V and (mforce, torque) for train and test.
      if (para.lambda_tau > 0.0f) {
        printf(
          "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_mforce_train,
          rmse_torque_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_mforce_test,
          rmse_torque_test);
        fprintf(
          fid_loss_out,
          "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_mforce_train,
          rmse_torque_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_mforce_test,
          rmse_torque_test);
      } else {
        printf(
          "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_mforce_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_mforce_test);
        fprintf(
          fid_loss_out,
          "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_mforce_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_mforce_test);
      }

      }else if (!para.charge_mode) {
        // NEP models
        printf(
          "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test);
        fprintf(
          fid_loss_out,
          "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test);
      } else {
        // qNEP models:
        printf(
          "%-8d%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_charge_train,
          rmse_bec_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_charge_test,
          rmse_bec_test);
        fprintf(
          fid_loss_out,
          "%-8d%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f%-9.5f\n",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_charge_train,
          rmse_bec_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_charge_test,
          rmse_bec_test);
      }
    } else {
      // TNEP models:
      printf(
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
      fprintf(
        fid_loss_out,
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
    }
    fflush(stdout);
    fflush(fid_loss_out);

    if (has_test_set) {
      if (para.train_mode == 0 || para.train_mode == 3) {
        FILE* fid_force = my_fopen("force_test.out", "w");
        FILE* fid_energy = my_fopen("energy_test.out", "w");
        FILE* fid_virial = my_fopen("virial_test.out", "w");
        FILE* fid_stress = my_fopen("stress_test.out", "w");
        update_energy_force_virial(fid_energy, fid_force, fid_virial, fid_stress, test_set[0]);
        fclose(fid_energy);
        fclose(fid_force);
        fclose(fid_virial);
        fclose(fid_stress);
        if (para.charge_mode) {
          FILE* fid_charge = my_fopen("charge_test.out", "w");
          update_charge(fid_charge, test_set[0]);
          fclose(fid_charge);
          if (para.has_bec) {
            FILE* fid_bec = my_fopen("bec_test.out", "w");
            update_bec(fid_bec, test_set[0]);
            fclose(fid_bec);
          }
        }
        // Only output mforce if reference data exists to avoid out-of-bounds
        if (para.spin_mode) {
          FILE* fid_mforce = my_fopen("mforce_test.out", "w");
          update_mforce(fid_mforce, test_set[0]);
          fclose(fid_mforce);
        }

      } else if (para.train_mode == 1) {
        FILE* fid_dipole = my_fopen("dipole_test.out", "w");
        update_dipole(fid_dipole, test_set[0], para.atomic_v);
        fclose(fid_dipole);
      } else if (para.train_mode == 2) {
        FILE* fid_polarizability = my_fopen("polarizability_test.out", "w");
        update_polarizability(fid_polarizability, test_set[0], para.atomic_v);
        fclose(fid_polarizability);
      }
    }

    if (0 == (generation + 1) % 1000) {
      predict(para, elite);
    }
  }
}

void Fitness::update_energy_force_virial(
  FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());

  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
      int n = offset + m;
      fprintf(
        fid_force,
        "%g %g %g %g %g %g\n",
        dataset.force_cpu[n],
        dataset.force_cpu[n + dataset.N],
        dataset.force_cpu[n + dataset.N * 2],
        dataset.force_ref_cpu[n],
        dataset.force_ref_cpu[n + dataset.N],
        dataset.force_ref_cpu[n + dataset.N * 2]);
    }
  }

  output(false, 1, fid_energy, dataset.energy_cpu.data(), dataset.energy_ref_cpu.data(), dataset);

  output(false, 6, fid_virial, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  output(true, 6, fid_stress, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
}

void Fitness::update_charge(FILE* fid_charge, Dataset& dataset)
{
  dataset.charge.copy_to_host(dataset.charge_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      fprintf(fid_charge, "%g\n", dataset.charge_cpu[dataset.Na_sum_cpu[nc] + m]);
    }
  }
}

void Fitness::update_bec(FILE* fid_bec, Dataset& dataset)
{
  dataset.bec.copy_to_host(dataset.bec_cpu.data());
  output_atomic(9, fid_bec, dataset.bec_cpu.data(), dataset.bec_ref_cpu.data(), dataset);
}

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(3, fid_dipole, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::update_polarizability(FILE* fid_polarizability, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 6, fid_polarizability, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(6, fid_polarizability, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::update_mforce(FILE* fid_mforce, Dataset& dataset)
{
  // Only output when reference magnetic force exists to avoid ambiguity
  if (dataset.mforce_ref_cpu.size() != (size_t)dataset.N * 3) {
    if (std::getenv("NEP_SPIN_LOG")) {
      printf("[spin][write] skip mforce output: no reference present (size=%zu expected=%zu)\n",
             dataset.mforce_ref_cpu.size(), (size_t)dataset.N * 3);
    }
    return;
  }
  dataset.mforce_cpu.resize(dataset.N * 3);
  dataset.mforce.copy_to_host(dataset.mforce_cpu.data());
  output_atomic(3, fid_mforce, dataset.mforce_cpu.data(), dataset.mforce_ref_cpu.data(), dataset);
}

 void Fitness::predict(Parameters& para, float* elite)
{
  if (para.train_mode == 0 || para.train_mode == 3) {
    FILE* fid_force = my_fopen("force_train.out", "w");
    FILE* fid_energy = my_fopen("energy_train.out", "w");
    FILE* fid_virial = my_fopen("virial_train.out", "w");
    FILE* fid_stress = my_fopen("stress_train.out", "w");
    FILE* fid_charge;
    FILE* fid_bec;
    FILE* fid_mforce;
    if (para.charge_mode) {
      fid_charge = my_fopen("charge_train.out", "w");
      if (para.has_bec) {
        fid_bec = my_fopen("bec_train.out", "w");
      }
    }
    if ((para.spin_mode > 0)) {
      // Create mforce output file; writing is gated per-batch by reference presence
      fid_mforce = my_fopen("mforce_train.out", "w");
    }
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, true, 1);
      update_energy_force_virial(
        fid_energy, fid_force, fid_virial, fid_stress, train_set[batch_id][0]);
      if (para.charge_mode) {
        update_charge(fid_charge, train_set[batch_id][0]);
        if (para.has_bec) {
          update_bec(fid_bec, train_set[batch_id][0]);
        }
      }
      if ((para.spin_mode > 0) &&
          (train_set[batch_id][0].mforce_ref_cpu.size() == (size_t)train_set[batch_id][0].N * 3)) {
        update_mforce(fid_mforce, train_set[batch_id][0]);
      }
    }
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
    if (para.charge_mode) {
      fclose(fid_charge);
      if (para.has_bec) {
        fclose(fid_bec);
      }
    }
    if ((para.spin_mode > 0)) {
      fclose(fid_mforce);
    }
  } else if (para.train_mode == 1) {
    FILE* fid_dipole = my_fopen("dipole_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, true, 1);
      update_dipole(fid_dipole, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_dipole);
  } else if (para.train_mode == 2) {
    FILE* fid_polarizability = my_fopen("polarizability_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, true, 1);
      update_polarizability(fid_polarizability, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_polarizability);
  }
}
