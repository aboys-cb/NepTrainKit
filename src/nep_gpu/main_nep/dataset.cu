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

#include "dataset.cuh"
#include "mic.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

void Dataset::copy_structures(std::vector<Structure>& structures_input, int n1, int n2)
{
  Nc = n2 - n1;
  structures.resize(Nc);

  for (int n = 0; n < Nc; ++n) {
    int n_input = n + n1;
    structures[n].num_atom = structures_input[n_input].num_atom;
    structures[n].weight = structures_input[n_input].weight;
    structures[n].has_virial = structures_input[n_input].has_virial;
    structures[n].has_spin = structures_input[n_input].has_spin;
    structures[n].has_bec = structures_input[n_input].has_bec;
    structures[n].has_atomic_virial = structures_input[n_input].has_atomic_virial;
    structures[n].atomic_virial_diag_only = structures_input[n_input].atomic_virial_diag_only;
    structures[n].charge = structures_input[n_input].charge;
    structures[n].energy = structures_input[n_input].energy;
    structures[n].energy_weight = structures_input[n_input].energy_weight;
    structures[n].has_temperature = structures_input[n_input].has_temperature;
    structures[n].temperature = structures_input[n_input].temperature;
    structures[n].volume = structures_input[n_input].volume;
    for (int k = 0; k < 6; ++k) {
      structures[n].virial[k] = structures_input[n_input].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures[n].box[k] = structures_input[n_input].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      structures[n].box_original[k] = structures_input[n_input].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      structures[n].num_cell[k] = structures_input[n_input].num_cell[k];
    }

    structures[n].type.resize(structures[n].num_atom);
    structures[n].x.resize(structures[n].num_atom);
    structures[n].y.resize(structures[n].num_atom);
    structures[n].z.resize(structures[n].num_atom);
    structures[n].fx.resize(structures[n].num_atom);
    structures[n].fy.resize(structures[n].num_atom);
    structures[n].fz.resize(structures[n].num_atom);
    structures[n].bec.resize(structures[n].num_atom * 9);
    // spin/mforce optional buffers
    if (!structures_input[n_input].spinx.empty()) {
      structures[n].spinx.resize(structures[n].num_atom);
      structures[n].spiny.resize(structures[n].num_atom);
      structures[n].spinz.resize(structures[n].num_atom);
    }
    if (!structures_input[n_input].mforce_mx.empty()) {
      structures[n].mforce_mx.resize(structures[n].num_atom);
      structures[n].mforce_my.resize(structures[n].num_atom);
      structures[n].mforce_mz.resize(structures[n].num_atom);
    }

    for (int na = 0; na < structures[n].num_atom; ++na) {
      structures[n].type[na] = structures_input[n_input].type[na];
      structures[n].x[na] = structures_input[n_input].x[na];
      structures[n].y[na] = structures_input[n_input].y[na];
      structures[n].z[na] = structures_input[n_input].z[na];
      structures[n].fx[na] = structures_input[n_input].fx[na];
      structures[n].fy[na] = structures_input[n_input].fy[na];
      structures[n].fz[na] = structures_input[n_input].fz[na];
      for (int d = 0; d < 9; ++d) {
        structures[n].bec[na * 9 + d] = structures_input[n_input].bec[na * 9 + d];
      }
      if (!structures[n].spinx.empty()) {
        structures[n].spinx[na] = structures_input[n_input].spinx[na];
        structures[n].spiny[na] = structures_input[n_input].spiny[na];
        structures[n].spinz[na] = structures_input[n_input].spinz[na];
      }
      if (!structures[n].mforce_mx.empty()) {
        structures[n].mforce_mx[na] = structures_input[n_input].mforce_mx[na];
        structures[n].mforce_my[na] = structures_input[n_input].mforce_my[na];
        structures[n].mforce_mz[na] = structures_input[n_input].mforce_mz[na];
      }
    }

    if (structures[n].has_atomic_virial != structures[0].has_atomic_virial) {
      throw std::runtime_error("All structures must have the same has_atomic_virial flag.");
    }
    if (structures[n].atomic_virial_diag_only != structures[0].atomic_virial_diag_only) {
      throw std::runtime_error("All structures must have the same atomic_virial_diag_only flag.");
    }
    if (structures[n].has_atomic_virial) {
      structures[n].avirialxx.resize(structures[n].num_atom);
      structures[n].avirialyy.resize(structures[n].num_atom);
      structures[n].avirialzz.resize(structures[n].num_atom);
      for (int na = 0; na < structures[n].num_atom; ++na) {
        structures[n].avirialxx[na] = structures_input[n_input].avirialxx[na];
        structures[n].avirialyy[na] = structures_input[n_input].avirialyy[na];
        structures[n].avirialzz[na] = structures_input[n_input].avirialzz[na];
      }
      if (!structures[n].atomic_virial_diag_only) {
        structures[n].avirialxy.resize(structures[n].num_atom);
        structures[n].avirialyz.resize(structures[n].num_atom);
        structures[n].avirialzx.resize(structures[n].num_atom);
        for (int na = 0; na < structures[n].num_atom; ++na) {
          structures[n].avirialxy[na] = structures_input[n_input].avirialxy[na];
          structures[n].avirialyz[na] = structures_input[n_input].avirialyz[na];
          structures[n].avirialzx[na] = structures_input[n_input].avirialzx[na];
        }
      }
    }
  }
}

void Dataset::find_has_type(Parameters& para)
{
  has_type.resize((para.num_types + 1) * Nc, false);
  for (int n = 0; n < Nc; ++n) {
    has_type[para.num_types * Nc + n] = true;
    for (int na = 0; na < structures[n].num_atom; ++na) {
      int t = structures[n].type[na];
      if (para.spin_mode == 1 && t >= para.num_types) {
        t -= para.num_types; // fold virtual to its real type bucket
      }
      has_type[t * Nc + n] = true;
    }
  }
}

void Dataset::find_Na(Parameters& para)
{
  Na_cpu.resize(Nc);
  Na_sum_cpu.resize(Nc);
  Na_real_cpu.resize(Nc);
  Na_real_sum_cpu.resize(Nc);

  N = 0;
  N_real = 0;
  max_Na = 0;
  int num_virial_configurations = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    int real_na = structures[nc].num_atom;
    int total_na = real_na;
    if (para.spin_mode == 1) {
      int pseudo_count = 0;
      for (int na = 0; na < real_na; ++na) {
        float a = 0.0f;
        if ((int)para.virtual_scale_by_type.size() == para.num_types) {
          a = para.virtual_scale_by_type[structures[nc].type[na]];
        }
        // alpha = virtual_scale_by_type
        if (a != 0.0f) {
          float sx = (na < (int)structures[nc].spinx.size()) ? structures[nc].spinx[na] : 0.0f;
          float sy = (na < (int)structures[nc].spiny.size()) ? structures[nc].spiny[na] : 0.0f;
          float sz = (na < (int)structures[nc].spinz.size()) ? structures[nc].spinz[na] : 0.0f;
          float s2 = sx * sx + sy * sy + sz * sz;
          if (s2 > 0.0f) {
            ++pseudo_count;
          }
        }
      }
      total_na = real_na + pseudo_count;
    }
    Na_cpu[nc] = total_na;
    Na_real_cpu[nc] = real_na;
    Na_sum_cpu[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    int real_na = structures[nc].num_atom;
    int total_na = real_na;
    if (para.spin_mode == 1) {
      int pseudo_count = 0;
      for (int na = 0; na < real_na; ++na) {
        float a = 0.0f;
        if ((int)para.virtual_scale_by_type.size() == para.num_types) {
          a = para.virtual_scale_by_type[structures[nc].type[na]];
        }
        if (a != 0.0f) {
          float sx = (na < (int)structures[nc].spinx.size()) ? structures[nc].spinx[na] : 0.0f;
          float sy = (na < (int)structures[nc].spiny.size()) ? structures[nc].spiny[na] : 0.0f;
          float sz = (na < (int)structures[nc].spinz.size()) ? structures[nc].spinz[na] : 0.0f;
          float s2 = sx * sx + sy * sy + sz * sz;
          if (s2 > 0.0f) {
            ++pseudo_count;
          }
        }
      }
      total_na = real_na + pseudo_count;
    }
    N += total_na;
    N_real += real_na;
    if (total_na > max_Na) {
      max_Na = total_na;
    }
    num_virial_configurations += structures[nc].has_virial;
  }

  // initialize real prefix sum base
  if (Nc > 0) {
    Na_real_sum_cpu[0] = 0;
  }
  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum_cpu[nc] = Na_sum_cpu[nc - 1] + Na_cpu[nc - 1];
    Na_real_sum_cpu[nc] = Na_real_sum_cpu[nc - 1] + Na_real_cpu[nc - 1];
  }

  printf("Total number of atoms = %d.\n", N);
  printf("Number of atoms in the largest configuration = %d.\n", max_Na);
  if (para.train_mode == 0 || para.train_mode == 3) {
    printf("Number of configurations having virial = %d.\n", num_virial_configurations);
  }

  Na.resize(Nc);
  Na_sum.resize(Nc);
  Na.copy_from_host(Na_cpu.data());
  Na_sum.copy_from_host(Na_sum_cpu.data());
  // spin: copy real-atom counters/prefix if available
  if (para.spin_mode == 1) {
    Na_real.resize(Nc);
    Na_real_sum.resize(Nc);
    Na_real.copy_from_host(Na_real_cpu.data());
    Na_real_sum.copy_from_host(Na_real_sum_cpu.data());
  }
}

void Dataset::initialize_gpu_data(Parameters& para)
{
  std::vector<float> box_cpu(Nc * 18);
  std::vector<float> box_original_cpu(Nc * 9);
  std::vector<int> num_cell_cpu(Nc * 3);
  std::vector<float> r_cpu(N * 3);
  std::vector<int> type_cpu(N);

  if (para.charge_mode) {
    charge.resize(N);
    charge_shifted.resize(N);
    charge_cpu.resize(N);
    charge_ref_cpu.resize(Nc);
    charge_ref_gpu.resize(Nc);
    bec.resize(N * 9);
    bec_cpu.resize(N * 9);
    bec_ref_cpu.resize(N * 9);
    bec_ref_gpu.resize(N * 9);
  }

  energy.resize(N);
  virial.resize(N * 6);
  force.resize(N * 3);
  energy_cpu.resize(N);
  virial_cpu.resize(N * 6);
  force_cpu.resize(N * 3);
  weight_cpu.resize(Nc);
  energy_ref_cpu.resize(Nc);
  energy_weight_cpu.resize(Nc);
  virial_ref_cpu.resize(Nc * 6);
  force_ref_cpu.resize(N * 3);
  if (structures[0].has_atomic_virial) {
    avirial_ref_cpu.resize(N * (structures[0].atomic_virial_diag_only ? 3 : 6));
  }
  temperature_ref_cpu.resize(N);

  // Prepare spin-mode buffers on host
  std::vector<int> host2pseudo_cpu;
  std::vector<int> host2real_cpu;
  std::vector<int> is_pseudo_cpu(N, 0);
  std::vector<float> spin_x_cpu, spin_y_cpu, spin_z_cpu;
  std::vector<float> mforce_x_cpu, mforce_y_cpu, mforce_z_cpu;
  std::vector<float> alpha_cpu;
  if (para.spin_mode == 1) {
    host2pseudo_cpu.resize(N_real);
    host2real_cpu.resize(N_real);
    spin_x_cpu.resize(N_real);
    spin_y_cpu.resize(N_real);
    spin_z_cpu.resize(N_real);
    mforce_x_cpu.resize(N_real);
    mforce_y_cpu.resize(N_real);
    mforce_z_cpu.resize(N_real);
    alpha_cpu.resize(N_real);
  }

  int real_running = 0;
  for (int n = 0; n < Nc; ++n) {
    weight_cpu[n] = structures[n].weight;
    if (para.charge_mode) {
      charge_ref_cpu[n] = structures[n].charge;
    }
    energy_ref_cpu[n] = structures[n].energy;
    energy_weight_cpu[n] = structures[n].energy_weight;
    for (int k = 0; k < 6; ++k) {
      virial_ref_cpu[k * Nc + n] = structures[n].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      box_cpu[k + n * 18] = structures[n].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      box_original_cpu[k + n * 9] = structures[n].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      num_cell_cpu[k + n * 3] = structures[n].num_cell[k];
    }
    int real_na = structures[n].num_atom;
    int offset_total = Na_sum_cpu[n];
    int pseudo_created_in_config = 0;
    // fill real atoms
    for (int na = 0; na < real_na; ++na) {
      int gt_real = offset_total + na;
      type_cpu[gt_real] = structures[n].type[na];
      r_cpu[gt_real] = structures[n].x[na];
      r_cpu[gt_real + N] = structures[n].y[na];
      r_cpu[gt_real + N * 2] = structures[n].z[na];
      force_ref_cpu[gt_real] = structures[n].fx[na];
      force_ref_cpu[gt_real + N] = structures[n].fy[na];
      force_ref_cpu[gt_real + N * 2] = structures[n].fz[na];
      temperature_ref_cpu[gt_real] = structures[n].temperature;
      if (structures[n].has_atomic_virial) {
        avirial_ref_cpu[gt_real] = structures[n].avirialxx[na];
        avirial_ref_cpu[gt_real + N] = structures[n].avirialyy[na];
        avirial_ref_cpu[gt_real + N * 2] = structures[n].avirialzz[na];
        if (!structures[n].atomic_virial_diag_only) {
          avirial_ref_cpu[gt_real + N * 3] = structures[n].avirialxy[na];
          avirial_ref_cpu[gt_real + N * 4] = structures[n].avirialyz[na];
          avirial_ref_cpu[gt_real + N * 5] = structures[n].avirialzx[na];
        }
      }
      if (para.charge_mode) {
        for (int d = 0; d < 9; ++d) {
          bec_ref_cpu[gt_real + N * d] = structures[n].bec[na * 9 + d];
        }
      }
      if (para.spin_mode == 1) {
        // compute alpha for this real atom
        float a = 0.0f;
        if ((int)para.virtual_scale_by_type.size() == para.num_types) {
          a = para.virtual_scale_by_type[structures[n].type[na]];
        }
        alpha_cpu[real_running] = a;
        host2real_cpu[real_running] = gt_real;
        // spin/mforce
        spin_x_cpu[real_running] = (na < structures[n].spinx.size()) ? structures[n].spinx[na] : 0.0f;
        spin_y_cpu[real_running] = (na < structures[n].spiny.size()) ? structures[n].spiny[na] : 0.0f;
        spin_z_cpu[real_running] = (na < structures[n].spinz.size()) ? structures[n].spinz[na] : 0.0f;
        mforce_x_cpu[real_running] = (na < structures[n].mforce_mx.size()) ? structures[n].mforce_mx[na] : 0.0f;
        mforce_y_cpu[real_running] = (na < structures[n].mforce_my.size()) ? structures[n].mforce_my[na] : 0.0f;
        mforce_z_cpu[real_running] = (na < structures[n].mforce_mz.size()) ? structures[n].mforce_mz[na] : 0.0f;
        // create pseudo atom only when alpha != 0 and spin magnitude > 0
        if (a != 0.0f) {
          float sx = (na < (int)structures[n].spinx.size()) ? structures[n].spinx[na] : 0.0f;
          float sy = (na < (int)structures[n].spiny.size()) ? structures[n].spiny[na] : 0.0f;
          float sz = (na < (int)structures[n].spinz.size()) ? structures[n].spinz[na] : 0.0f;
          float s2 = sx * sx + sy * sy + sz * sz;
          if (s2 > 0.0f) {
          int gt_pseudo = offset_total + real_na + pseudo_created_in_config;
          host2pseudo_cpu[real_running] = gt_pseudo;
          is_pseudo_cpu[gt_pseudo] = 1;
          // initialize pseudo entry with shifted position: r_pseudo = r_real + alpha * spin
          // assign virtual type index = real type + num_types (when spin_mode==1)
          type_cpu[gt_pseudo] = structures[n].type[na] + (para.spin_mode == 1 ? para.num_types : 0);
          r_cpu[gt_pseudo] = structures[n].x[na] + a * sx;
          r_cpu[gt_pseudo + N] = structures[n].y[na] + a * sy;
          r_cpu[gt_pseudo + N * 2] = structures[n].z[na] + a * sz;
          force_ref_cpu[gt_pseudo] = 0.0f;
          force_ref_cpu[gt_pseudo + N] = 0.0f;
          force_ref_cpu[gt_pseudo + N * 2] = 0.0f;
          temperature_ref_cpu[gt_pseudo] = structures[n].temperature;
          ++pseudo_created_in_config;
          } else {
            host2pseudo_cpu[real_running] = -1;
          }
        } else {
          host2pseudo_cpu[real_running] = -1;
        }
      }
      real_running++;
    }
  }

  type_weight_gpu.resize(NUM_ELEMENTS);
  
  energy_ref_gpu.resize(Nc);
  energy_weight_gpu.resize(Nc);
  virial_ref_gpu.resize(Nc * 6);
  force_ref_gpu.resize(N * 3);
  if (structures[0].has_atomic_virial) {
    avirial_ref_gpu.resize(N * (structures[0].atomic_virial_diag_only ? 3 : 6));
  }
  temperature_ref_gpu.resize(N);
  type_weight_gpu.copy_from_host(para.type_weight_cpu.data());
  if (para.charge_mode) {
    charge_ref_gpu.copy_from_host(charge_ref_cpu.data());
    bec_ref_gpu.copy_from_host(bec_ref_cpu.data());
  }
  energy_ref_gpu.copy_from_host(energy_ref_cpu.data());
  energy_weight_gpu.copy_from_host(energy_weight_cpu.data());
  virial_ref_gpu.copy_from_host(virial_ref_cpu.data());
  force_ref_gpu.copy_from_host(force_ref_cpu.data());
  if (structures[0].has_atomic_virial) {
    avirial_ref_gpu.copy_from_host(avirial_ref_cpu.data());
  }
  temperature_ref_gpu.copy_from_host(temperature_ref_cpu.data());

  if (para.spin_mode == 1) {
    // allocate and copy spin-mode buffers
    spin.resize(N_real * 3);
    mforce_ref.resize(N_real * 3);
    fm_pred.resize(N_real * 3);
    host2pseudo.resize(N_real);
    host2real.resize(N_real);
    is_pseudo.resize(N);
    alpha.resize(N_real);

    host2pseudo.copy_from_host(host2pseudo_cpu.data());
    host2real.copy_from_host(host2real_cpu.data());
    is_pseudo.copy_from_host(is_pseudo_cpu.data());
    alpha.copy_from_host(alpha_cpu.data());

    // assemble component-major buffers for spin and mforce_ref
    std::vector<float> tmp(N_real * 3);
    for (int i = 0; i < N_real; ++i) {
      tmp[i] = spin_x_cpu[i];
      tmp[i + N_real] = spin_y_cpu[i];
      tmp[i + 2 * N_real] = spin_z_cpu[i];
    }
    spin.copy_from_host(tmp.data());
    for (int i = 0; i < N_real; ++i) {
      tmp[i] = mforce_x_cpu[i];
      tmp[i + N_real] = mforce_y_cpu[i];
      tmp[i + 2 * N_real] = mforce_z_cpu[i];
    }
    mforce_ref.copy_from_host(tmp.data());
  }

  box.resize(Nc * 18);
  box_original.resize(Nc * 9);
  num_cell.resize(Nc * 3);
  r.resize(N * 3);
  type.resize(N);
  box.copy_from_host(box_cpu.data());
  box_original.copy_from_host(box_original_cpu.data());
  num_cell.copy_from_host(num_cell_cpu.data());
  r.copy_from_host(r_cpu.data());
  type.copy_from_host(type_cpu.data());
}

static __global__ void gpu_find_neighbor_number(
  const int N,
  const int* Na,
  const int* Na_sum,
  const bool use_typewise_cutoff,
  const float typewise_cutoff_radial_factor,
  const float typewise_cutoff_angular_factor,
  const int* g_type,
  const int* g_atomic_numbers,
  const float g_rc_radial,
  const float g_rc_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NN_angular,
  const int* __restrict__ g_is_pseudo)
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
      // Skip pseudo-pseudo pairs entirely to save work; they do not contribute
      // to forces/energy in NEP (pseudo centers have zero gradients).
      if (g_is_pseudo && g_is_pseudo[n1] && g_is_pseudo[n2]) {
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
            if (distance_square < 1.0e-12f) {
              // Skip degenerate pairs; overlap detection removed.
              continue;
            }
            int t2 = g_type[n2];
            float rc_radial = g_rc_radial;
            float rc_angular = g_rc_angular;
            if (use_typewise_cutoff) {
              int z1 = g_atomic_numbers[t1];
              int z2 = g_atomic_numbers[t2];
              rc_radial = min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * typewise_cutoff_radial_factor, rc_radial);
              rc_angular = min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * typewise_cutoff_angular_factor, rc_angular);
            }
            if (distance_square < rc_radial * rc_radial) {
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              count_angular++;
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

void Dataset::find_neighbor(Parameters& para)
{
  // Overlap checking removed; neighbor counts proceed without overlap aborts.

  GPU_Vector<int> NN_radial_gpu(N);
  GPU_Vector<int> NN_angular_gpu(N);
  std::vector<int> NN_radial_cpu(N);
  std::vector<int> NN_angular_cpu(N);

  // Build atomic number table for all types used in neighbor kernel
  int num_types_real = para.num_types;
  int num_types_total = (para.spin_mode == 1 ? para.num_types * 2 : para.num_types);
  std::vector<int> atomic_numbers_from_zero(num_types_total);
  for (int n = 0; n < num_types_real; ++n) {
    atomic_numbers_from_zero[n] = para.atomic_numbers[n] - 1;
  }
  for (int n = num_types_real; n < num_types_total; ++n) {
    atomic_numbers_from_zero[n] = para.atomic_numbers[n - num_types_real] - 1;
  }
  GPU_Vector<int> atomic_numbers(num_types_total);
  atomic_numbers.copy_from_host(atomic_numbers_from_zero.data());

  gpu_find_neighbor_number<<<Nc, 256>>>(
    N,
    Na.data(),
    Na_sum.data(),
    para.use_typewise_cutoff,
    para.typewise_cutoff_radial_factor,
    para.typewise_cutoff_angular_factor,
    type.data(),
    atomic_numbers.data(),
    para.rc_radial,
    para.rc_angular,
    box.data(),
    box_original.data(),
    num_cell.data(),
    r.data(),
    r.data() + N,
    r.data() + N * 2,
    NN_radial_gpu.data(),
    NN_angular_gpu.data(),
    (para.spin_mode == 1) ? is_pseudo.data() : (int*)nullptr);
  GPU_CHECK_KERNEL


  NN_radial_gpu.copy_to_host(NN_radial_cpu.data());
  NN_angular_gpu.copy_to_host(NN_angular_cpu.data());

  int min_NN_radial = 10000;
  max_NN_radial = -1;
  for (int n = 0; n < N; ++n) {
    if (NN_radial_cpu[n] < min_NN_radial) {
      min_NN_radial = NN_radial_cpu[n];
    }
    if (NN_radial_cpu[n] > max_NN_radial) {
      max_NN_radial = NN_radial_cpu[n];
    }
  }
  int min_NN_angular = 10000;
  max_NN_angular = -1;
  for (int n = 0; n < N; ++n) {
    if (NN_angular_cpu[n] < min_NN_angular) {
      min_NN_angular = NN_angular_cpu[n];
    }
    if (NN_angular_cpu[n] > max_NN_angular) {
      max_NN_angular = NN_angular_cpu[n];
    }
  }

  // In spin mode, NEP will inject additional neighbors after the raw cutoff-based
  // list is built: (1) host real <-> its own pseudo, and (2) for each neighbor b
  // of a, also add b's mirror partner. To avoid any truncation of these injected
  // entries due to capacity limits, reserve enough headroom by expanding the
  // per-atom maximum neighbor capacities here. A conservative and simple choice
  // is to double the maxima computed from the raw counts.
  if (para.spin_mode == 1) {
    max_NN_radial = max_NN_radial * 2;
    max_NN_angular = max_NN_angular * 2;
  }

  printf("Radial descriptor with a cutoff of %g A:\n", para.rc_radial);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_radial);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_radial);
  printf("Angular descriptor with a cutoff of %g A:\n", para.rc_angular);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_angular);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_angular);
}

void Dataset::construct(
  Parameters& para, std::vector<Structure>& structures_input, int n1, int n2, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  copy_structures(structures_input, n1, n2);
  find_has_type(para);
  error_cpu.resize(Nc);
  error_gpu.resize(Nc);

  find_Na(para);
  initialize_gpu_data(para);
  find_neighbor(para);
}

static __global__ void gpu_sum_force_error(
  bool use_weight,
  float force_delta,
  int* g_Na,
  int* g_Na_sum,
  const int* __restrict__ g_is_pseudo,
  int* g_type,
  float* g_type_weight,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_fx_ref,
  float* g_fy_ref,
  float* g_fz_ref,
  float* error_gpu,
  int num_types_real)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    if (g_is_pseudo && g_is_pseudo[n]) {
      continue; // exclude pseudo atoms from mechanical force RMSE
    }
    float fx_ref = g_fx_ref[n];
    float fy_ref = g_fy_ref[n];
    float fz_ref = g_fz_ref[n];
    float dx = g_fx[n] - fx_ref;
    float dy = g_fy[n] - fy_ref;
    float dz = g_fz[n] - fz_ref;
    float diff_square = dx * dx + dy * dy + dz * dz;
    if (use_weight) {
      int t = g_type[n];
      if (t >= num_types_real) t -= num_types_real;
      float type_weight = g_type_weight[t];
      diff_square *= type_weight * type_weight;
    }
    if (use_weight && force_delta > 0.0f) {
      float force_magnitude = sqrt(fx_ref * fx_ref + fy_ref * fy_ref + fz_ref * fz_ref);
      diff_square *= force_delta / (force_delta + force_magnitude);
    }
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> Dataset::get_rmse_force(Parameters& para, const bool use_weight, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;
  gpu_sum_force_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    use_weight,
    para.force_delta,
    Na.data(),
    Na_sum.data(),
    (para.spin_mode == 1) ? is_pseudo.data() : (int*)nullptr,
    type.data(),
    type_weight_gpu.data(),
    force.data(),
    force.data() + N,
    force.data() + N * 2,
    force_ref_gpu.data(),
    force_ref_gpu.data() + N,
    force_ref_gpu.data() + N * 2,
    error_gpu.data(),
    para.num_types);
  int mem = sizeof(float) * Nc;
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += (para.spin_mode == 1 ? Na_real_cpu[n] : Na_cpu[n]);
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 3));
    }
  }
  return rmse_array;
}

// Sum per-configuration magnetic-force squared error over real atoms only
static __global__ void gpu_sum_mforce_error(
  bool use_weight,
  int* g_Na,
  int* g_Na_sum,
  int* g_Na_real,
  int* g_Na_real_sum,
  int* g_type,
  float* g_type_weight,
  const float* __restrict__ g_fm_x,
  const float* __restrict__ g_fm_y,
  const float* __restrict__ g_fm_z,
  const float* __restrict__ g_mref_x,
  const float* __restrict__ g_mref_y,
  const float* __restrict__ g_mref_z,
  float* error_gpu,
  int* active_count_gpu,
  int num_types_real)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int real_na = g_Na_real[bid];
  int r_start = g_Na_real_sum[bid];
  extern __shared__ float s_buf[];
  float* s_error = s_buf;
  float* s_count = s_buf + blockDim.x;
  s_error[tid] = 0.0f;
  s_count[tid] = 0.0f;

  for (int i = tid; i < real_na; i += blockDim.x) {
    int n = N1 + i;              // global index among all atoms
    int r = r_start + i;         // index among real atoms
    float mrx = g_mref_x[r];
    float mry = g_mref_y[r];
    float mrz = g_mref_z[r];
    float mr2 = mrx * mrx + mry * mry + mrz * mrz;
    if (mr2 < 1.0e-24f) {
      continue; // mask non-magnetic atoms (zero reference torque)
    }
    float dx = g_fm_x[r] - mrx;
    float dy = g_fm_y[r] - mry;
    float dz = g_fm_z[r] - mrz;
    float diff_square = dx * dx + dy * dy + dz * dz;
    if (use_weight) {
      int t = g_type[n];
      if (t >= num_types_real) t -= num_types_real;
      float type_weight = g_type_weight[t];
      diff_square *= type_weight * type_weight;
    }
    s_error[tid] += diff_square;
    s_count[tid] += 1.0f;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
      s_count[tid] += s_count[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
    if (active_count_gpu) active_count_gpu[bid] = (int)(s_count[0] + 0.5f);
  }
}

std::vector<float> Dataset::get_rmse_magnetic_force(Parameters& para, const bool use_weight, int device_id)
{
  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  if (para.spin_mode != 1 || para.lambda_mf <= 0.0f) {
    return rmse_array;
  }

  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;
  // prepare active-count buffers
  mforce_active_count_gpu.resize(Nc);
  mforce_active_count_cpu.resize(Nc);
  gpu_sum_mforce_error<<<Nc, block_size, sizeof(float) * block_size * 2>>>(
    use_weight,
    Na.data(),
    Na_sum.data(),
    Na_real.data(),
    Na_real_sum.data(),
    type.data(),
    type_weight_gpu.data(),
    fm_pred.data(),
    fm_pred.data() + N_real,
    fm_pred.data() + 2 * N_real,
    mforce_ref.data(),
    mforce_ref.data() + N_real,
    mforce_ref.data() + 2 * N_real,
    error_gpu.data(),
    mforce_active_count_gpu.data(),
    para.num_types);

  int mem = sizeof(float) * Nc;
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(mforce_active_count_cpu.data(), mforce_active_count_gpu.data(), sizeof(int) * Nc, gpuMemcpyDeviceToHost));

  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += mforce_active_count_cpu[n];
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 3));
    }
  }
  return rmse_array;
}

static __global__ void gpu_sum_avirial_diag_only_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_virial,
  float* g_avxx_ref,
  float* g_avyy_ref,
  float* g_avzz_ref,
  float* error_gpu,
  const int* __restrict__ g_is_pseudo)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    if (g_is_pseudo && g_is_pseudo[n]) {
      continue; // exclude pseudo atoms
    }
    float avxx_ref = g_avxx_ref[n];
    float avyy_ref = g_avyy_ref[n];
    float avzz_ref = g_avzz_ref[n];
    float dxx = g_virial[n] - avxx_ref;
    float dyy = g_virial[1 * N + n] - avyy_ref;
    float dzz = g_virial[2 * N + n] - avzz_ref;
    float diff_square = dxx * dxx + dyy * dyy + dzz * dzz;
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

static __global__ void gpu_sum_avirial_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_virial,
  float* g_avxx_ref,
  float* g_avyy_ref,
  float* g_avzz_ref,
  float* g_avxy_ref,
  float* g_avyz_ref,
  float* g_avzx_ref,
  float* error_gpu,
  const int* __restrict__ g_is_pseudo)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    if (g_is_pseudo && g_is_pseudo[n]) {
      continue; // exclude pseudo atoms
    }
    float avxx_ref = g_avxx_ref[n];
    float avyy_ref = g_avyy_ref[n];
    float avzz_ref = g_avzz_ref[n];
    float avxy_ref = g_avxy_ref[n];
    float avyz_ref = g_avyz_ref[n];
    float avzx_ref = g_avzx_ref[n];
    float dxx = g_virial[n] - avxx_ref;
    float dyy = g_virial[1 * N + n] - avyy_ref;
    float dzz = g_virial[2 * N + n] - avzz_ref;
    float dxy = g_virial[3 * N + n] - avxy_ref;
    float dyz = g_virial[4 * N + n] - avyz_ref;
    float dzx = g_virial[5 * N + n] - avzx_ref;
    float diff_square = dxx * dxx + dyy * dyy + dzz * dzz + dxy * dxy + dyz * dyz + dzx * dzx;
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> Dataset::get_rmse_avirial(Parameters& para, const bool use_weight, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;

  if (structures[0].atomic_virial_diag_only) {
    gpu_sum_avirial_diag_only_error<<<Nc, block_size, sizeof(float) * block_size>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      type_weight_gpu.data(),
      virial.data(),
      avirial_ref_gpu.data(),
      avirial_ref_gpu.data() + N,
      avirial_ref_gpu.data() + N * 2,
      error_gpu.data(),
      (para.spin_mode == 1) ? is_pseudo.data() : (int*)nullptr);
  } else {
    gpu_sum_avirial_error<<<Nc, block_size, sizeof(float) * block_size>>>(
      N,
      Na.data(),
      Na_sum.data(),
      type.data(),
      type_weight_gpu.data(),
      virial.data(),
      avirial_ref_gpu.data(),
      avirial_ref_gpu.data() + N,
      avirial_ref_gpu.data() + N * 2,
      avirial_ref_gpu.data() + N * 3,
      avirial_ref_gpu.data() + N * 4,
      avirial_ref_gpu.data() + N * 5,
      error_gpu.data(),
      (para.spin_mode == 1) ? is_pseudo.data() : (int*)nullptr);
  }
  int mem = sizeof(float) * Nc;
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += (para.spin_mode == 1 ? Na_real_cpu[n] : Na_cpu[n]);
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 6));
    }
  }
  return rmse_array;
}

static __global__ void
gpu_get_energy_shift(
  int* g_Na, 
  int* g_Na_sum, 
  float* g_pe, 
  float* g_pe_ref, 
  float* g_pe_weight, 
  float* g_energy_shift,
  const int* __restrict__ g_Na_den)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int Na_den = g_Na_den ? g_Na_den[bid] : Na;
    float diff = s_pe[0] / Na_den - g_pe_ref[bid];
    g_energy_shift[bid] = diff * g_pe_weight[bid];
  }
}

static __global__ void gpu_sum_pe_error(
  float energy_shift, 
  int* g_Na, 
  int* g_Na_sum, 
  float* g_pe, 
  float* g_pe_ref, 
  float* g_pe_weight, 
  float* error_gpu,
  const int* __restrict__ g_Na_den)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int Na_den = g_Na_den ? g_Na_den[bid] : Na;
    float diff = s_pe[0] / Na_den - g_pe_ref[bid] - energy_shift;
    error_gpu[bid] = diff * diff * g_pe_weight[bid];
  }
}

std::vector<float> Dataset::get_rmse_energy(
  Parameters& para,
  float& energy_shift_per_structure,
  const bool use_weight,
  const bool do_shift,
  int device_id)
{
  CHECK(gpuSetDevice(device_id));
  energy_shift_per_structure = 0.0f;

  const int block_size = 256;
  int mem = sizeof(float) * Nc;

  if (do_shift) {
    gpu_get_energy_shift<<<Nc, block_size, sizeof(float) * block_size>>>(
      Na.data(), 
      Na_sum.data(), 
      energy.data(), 
      energy_ref_gpu.data(), 
      energy_weight_gpu.data(), 
      error_gpu.data(),
      Na_real.data());
    CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
    float Nc_with_weight = 0.0f;
    for (int n = 0; n < Nc; ++n) {
      Nc_with_weight += energy_weight_cpu[n];
      energy_shift_per_structure += error_cpu[n];
    }
    if (Nc_with_weight > 0.0f) {
      energy_shift_per_structure /= Nc_with_weight;
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    energy_shift_per_structure,
    Na.data(),
    Na_sum.data(),
    energy.data(),
    energy_ref_gpu.data(),
    energy_weight_gpu.data(), 
    error_gpu.data(),
    Na_real.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        ++count_array[t];
      }
    }
  }
  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

static __global__ void gpu_sum_virial_error(
  const int N,
  const float shear_weight,
  int* g_Na,
  int* g_Na_sum,
  float* g_virial,
  float* g_virial_ref,
  float* error_gpu,
  const int* __restrict__ g_is_pseudo,
  const int* __restrict__ g_Na_den)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_virial[];
  for (int d = 0; d < 6; ++d) {
    s_virial[d * blockDim.x + tid] = 0.0f;
  }

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    if (g_is_pseudo && g_is_pseudo[n]) {
      continue; // exclude pseudo atoms
    }
    for (int d = 0; d < 6; ++d) {
      s_virial[d * blockDim.x + tid] += g_virial[d * N + n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int Na_den = g_Na_den ? g_Na_den[bid] : Na;
    float error_sum = 0.0f;
    for (int d = 0; d < 6; ++d) {
      float diff = s_virial[d * blockDim.x + 0] / Na_den - g_virial_ref[d * gridDim.x + bid];
      error_sum += (d >= 3) ? (shear_weight * diff * diff) : (diff * diff);
    }
    error_gpu[bid] = error_sum;
  }
}

std::vector<float> Dataset::get_rmse_virial(Parameters& para, const bool use_weight, int device_id)
{
  if (para.atomic_v) {
    return get_rmse_avirial(para, use_weight, device_id);
  }
  CHECK(gpuSetDevice(device_id));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * Nc;
  const int block_size = 256;

  float shear_weight =
    (para.train_mode != 1) ? (use_weight ? para.lambda_shear * para.lambda_shear : 1.0f) : 0.0f;
  gpu_sum_virial_error<<<Nc, block_size, sizeof(float) * block_size * 6>>>(
    N,
    shear_weight,
    Na.data(),
    Na_sum.data(),
    virial.data(),
    virial_ref_gpu.data(),
    error_gpu.data(),
    (para.spin_mode == 1) ? is_pseudo.data() : (int*)nullptr,
    (para.spin_mode == 1) ? Na_real.data() : (int*)nullptr);
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (has_type[t * Nc + n]) {
          rmse_array[t] += rmse_temp;
          count_array[t] += (para.train_mode != 1) ? 6 : 3;
        }
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

static __global__ void gpu_sum_charge_error(
  int* g_Na, 
  int* g_Na_sum, 
  float* g_charge, 
  float* g_charge_ref,  
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_charge[];
  s_charge[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_charge[tid] += g_charge[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float diff = (s_charge[0] - g_charge_ref[bid]) / Na;
    error_gpu[bid] = diff * diff;
  }
}

static __global__ void gpu_sum_bec_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  float* g_bec,
  float* g_bec_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float diff_square = 0.0f;
    for (int d = 0; d < 9; ++d) {
      const float diff = g_bec[n + N * d] - g_bec_ref[n + N * d];
      diff_square += diff * diff;
    }
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> Dataset::get_rmse_charge(Parameters& para, int device_id)
{
  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  if (!para.charge_mode) {
    return rmse_array;
  }

  CHECK(gpuSetDevice(device_id));

  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * Nc;
  const int block_size = 256;

  gpu_sum_charge_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(),
    Na_sum.data(),
    charge.data(),
    charge_ref_gpu.data(),
    error_gpu.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
      float rmse_temp = error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (has_type[t * Nc + n]) {
          rmse_array[t] += rmse_temp;
          count_array[t] += 1;
        }
      }
  }

  if (para.has_bec) {
    gpu_sum_bec_error<<<Nc, block_size, sizeof(float) * block_size>>>(
      N,
      Na.data(),
      Na_sum.data(),
      bec.data(),
      bec_ref_gpu.data(),
      error_gpu.data());
    CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
    for (int n = 0; n < Nc; ++n) {
      if (structures[n].has_bec) {
        float rmse_temp = error_cpu[n];
        for (int t = 0; t < para.num_types + 1; ++t) {
          if (has_type[t * Nc + n]) {
            rmse_array[t] += rmse_temp / (Na_cpu[n]);
            count_array[t] += 9;
          }
        }
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}
