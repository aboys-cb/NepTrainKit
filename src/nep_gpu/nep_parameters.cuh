// SPDX-License-Identifier: GPL-3.0-or-later
/*
    NepTrainKit NEP parameter interfaces
    Copyright (C) 2025 NepTrainKit contributors

    This file interfaces with and extends types from GPUMD
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

#pragma once

#include "parameters.cuh"

// A derived Parameters class that provides the bool-ctor
// previously declared in main_nep::Parameters.
class NepParameters : public Parameters {
public:
//   explicit NepParameters(bool skip_nep_in);

  void load_from_nep_txt(const std::string& filename, std::vector<float>& elite);

};

