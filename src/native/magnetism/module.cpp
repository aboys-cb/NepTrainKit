#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "neptrainkit/native/periodic_neighbors.hpp"

namespace py = pybind11;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kMomentEps = 1.0e-7;

std::array<double, 3> symmetric_eigenvalues(std::array<double, 9> matrix) {
    for (int sweep = 0; sweep < 12; ++sweep) {
        int p = 0;
        int q = 1;
        double maximum = std::abs(matrix[1]);
        for (const auto pair : {std::array<int, 2>{{0, 2}}, std::array<int, 2>{{1, 2}}}) {
            const double value = std::abs(matrix[pair[0] * 3 + pair[1]]);
            if (value > maximum) {
                maximum = value;
                p = pair[0];
                q = pair[1];
            }
        }
        if (maximum < 1.0e-12) break;
        const double app = matrix[p * 3 + p];
        const double aqq = matrix[q * 3 + q];
        const double apq = matrix[p * 3 + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int row = 0; row < 3; ++row) {
            if (row == p || row == q) continue;
            const double arp = matrix[row * 3 + p];
            const double arq = matrix[row * 3 + q];
            matrix[row * 3 + p] = matrix[p * 3 + row] = cosine * arp - sine * arq;
            matrix[row * 3 + q] = matrix[q * 3 + row] = sine * arp + cosine * arq;
        }
        matrix[p * 3 + p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * 3 + q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * 3 + q] = matrix[q * 3 + p] = 0.0;
    }
    std::array<double, 3> values{{matrix[0], matrix[4], matrix[8]}};
    std::sort(values.begin(), values.end());
    return values;
}

std::array<double, 9> inverse_cell(const float* cell) {
    const double determinant =
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
        cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
        cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
    if (std::abs(determinant) <= 1.0e-12) {
        throw py::value_error("periodic cell must be invertible");
    }
    const double scale = 1.0 / determinant;
    return {{
        (cell[4] * cell[8] - cell[5] * cell[7]) * scale,
        (cell[2] * cell[7] - cell[1] * cell[8]) * scale,
        (cell[1] * cell[5] - cell[2] * cell[4]) * scale,
        (cell[5] * cell[6] - cell[3] * cell[8]) * scale,
        (cell[0] * cell[8] - cell[2] * cell[6]) * scale,
        (cell[2] * cell[3] - cell[0] * cell[5]) * scale,
        (cell[3] * cell[7] - cell[4] * cell[6]) * scale,
        (cell[1] * cell[6] - cell[0] * cell[7]) * scale,
        (cell[0] * cell[4] - cell[1] * cell[3]) * scale,
    }};
}

std::pair<std::string, std::string> classify_order(
    const std::int64_t active_count,
    const double net_ratio,
    const double collinearity,
    const double coplanarity,
    const double neighbor_correlation,
    const double antiparallel_fraction,
    const double q_peak
) {
    if (active_count < 2) return {"low_moment", "strong"};
    if (collinearity >= 0.90) {
        if (net_ratio >= 0.82 && neighbor_correlation >= 0.20) {
            return {"fm", net_ratio >= 0.92 ? "strong" : "mixed"};
        }
        if (q_peak >= 0.45 && antiparallel_fraction >= 0.20) {
            if (net_ratio <= 0.20) return {"afm", "strong"};
            if (net_ratio <= 0.95) return {"ferrimagnetic", "strong"};
        }
        if (net_ratio <= 0.20 && neighbor_correlation <= -0.25) {
            return {"afm", "mixed"};
        }
        return {"collinear_mixed", "mixed"};
    }
    if (q_peak >= 0.52 && coplanarity >= 0.72) {
        return {"spin_spiral", q_peak >= 0.70 ? "strong" : "mixed"};
    }
    if (q_peak >= 0.32 || std::abs(neighbor_correlation) >= 0.30) {
        return {"noncollinear", "mixed"};
    }
    return {"spin_disordered", q_peak <= 0.22 ? "strong" : "mixed"};
}

py::tuple magnetic_order_evidence(
    py::array_t<float, py::array::c_style | py::array::forcecast> positions_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> cell_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> pbc_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> spins_array,
    const int requested_neighbors,
    const int q_max
) {
    if (positions_array.ndim() != 2 || positions_array.shape(1) != 3 ||
        spins_array.ndim() != 2 || spins_array.shape(1) != 3 ||
        positions_array.shape(0) != spins_array.shape(0) ||
        cell_array.ndim() != 2 || cell_array.shape(0) != 3 || cell_array.shape(1) != 3 ||
        pbc_array.ndim() != 1 || pbc_array.shape(0) != 3) {
        throw py::value_error("positions, spins, cell, and pbc must have shapes (N,3), (N,3), (3,3), and (3,)");
    }
    if (requested_neighbors <= 0 || q_max <= 0) {
        throw py::value_error("neighbors and q_max must be positive");
    }
    const std::int64_t atom_count = static_cast<std::int64_t>(positions_array.shape(0));
    const float* spins = spins_array.data();
    std::vector<double> magnitudes(static_cast<std::size_t>(atom_count), 0.0);
    std::vector<std::array<double, 3>> unit_spins(static_cast<std::size_t>(atom_count));
    std::int64_t active_count = 0;
    double total_magnitude = 0.0;
    double sum_squared_magnitude = 0.0;
    std::array<double, 3> net{{0.0, 0.0, 0.0}};
    std::array<double, 9> orientation{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
        const double x = spins[atom * 3];
        const double y = spins[atom * 3 + 1];
        const double z = spins[atom * 3 + 2];
        const double magnitude = std::sqrt(x * x + y * y + z * z);
        magnitudes[static_cast<std::size_t>(atom)] = magnitude;
        total_magnitude += magnitude;
        sum_squared_magnitude += magnitude * magnitude;
        net[0] += x;
        net[1] += y;
        net[2] += z;
        if (magnitude <= kMomentEps) continue;
        ++active_count;
        auto& unit = unit_spins[static_cast<std::size_t>(atom)];
        unit = {{x / magnitude, y / magnitude, z / magnitude}};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                orientation[row * 3 + column] += unit[row] * unit[column];
            }
        }
    }
    const double mean_magnitude = atom_count > 0 ? total_magnitude / atom_count : 0.0;
    const double variance = atom_count > 0
        ? std::max(0.0, sum_squared_magnitude / atom_count - mean_magnitude * mean_magnitude)
        : 0.0;
    const double net_ratio = total_magnitude > kMomentEps
        ? std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]) / total_magnitude
        : 0.0;
    double collinearity = 0.0;
    double coplanarity = 0.0;
    if (active_count > 0) {
        for (double& value : orientation) value /= active_count;
        const auto eigenvalues = symmetric_eigenvalues(orientation);
        collinearity = std::max(0.0, std::min(1.0, eigenvalues[2]));
        coplanarity = std::max(0.0, std::min(1.0, 1.0 - 3.0 * eigenvalues[0]));
    }

    double neighbor_correlation = 0.0;
    double neighbor_abs_correlation = 0.0;
    std::int64_t parallel_count = 0;
    std::int64_t antiparallel_count = 0;
    std::int64_t pair_count = 0;
    if (atom_count > 1 && active_count > 1) {
        const neptrainkit::native::PeriodicNeighborSearch<float> search(
            positions_array.data(), atom_count, cell_array.data(), pbc_array.data());
        const auto neighbors = search.query_knn(std::min<int>(requested_neighbors, atom_count - 1));
        for (std::int64_t atom = 0; atom < atom_count; ++atom) {
            if (magnitudes[static_cast<std::size_t>(atom)] <= kMomentEps) continue;
            const auto& left = unit_spins[static_cast<std::size_t>(atom)];
            for (const auto& neighbor : neighbors[static_cast<std::size_t>(atom)]) {
                const std::int64_t source = neighbor.source;
                if (source < 0 || magnitudes[static_cast<std::size_t>(source)] <= kMomentEps) continue;
                const auto& right = unit_spins[static_cast<std::size_t>(source)];
                const double dot = std::max(-1.0, std::min(1.0,
                    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]));
                neighbor_correlation += dot;
                neighbor_abs_correlation += std::abs(dot);
                parallel_count += dot >= 0.80;
                antiparallel_count += dot <= -0.80;
                ++pair_count;
            }
        }
    }
    if (pair_count > 0) {
        neighbor_correlation /= pair_count;
        neighbor_abs_correlation /= pair_count;
    }
    const double parallel_fraction = pair_count > 0 ? static_cast<double>(parallel_count) / pair_count : 0.0;
    const double antiparallel_fraction = pair_count > 0 ? static_cast<double>(antiparallel_count) / pair_count : 0.0;

    double q_peak = 0.0;
    std::array<int, 3> q_vector{{0, 0, 0}};
    if (atom_count > 1 && total_magnitude > kMomentEps) {
        const auto inverse = inverse_cell(cell_array.data());
        std::vector<std::array<double, 3>> fractional(static_cast<std::size_t>(atom_count));
        const float* positions = positions_array.data();
        for (std::int64_t atom = 0; atom < atom_count; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                double value = positions[atom * 3] * inverse[axis]
                    + positions[atom * 3 + 1] * inverse[3 + axis]
                    + positions[atom * 3 + 2] * inverse[6 + axis];
                fractional[static_cast<std::size_t>(atom)][axis] = value - std::floor(value);
            }
        }
        const bool* pbc = pbc_array.data();
        for (int h = pbc[0] ? -q_max : 0; h <= (pbc[0] ? q_max : 0); ++h) {
            for (int k = pbc[1] ? -q_max : 0; k <= (pbc[1] ? q_max : 0); ++k) {
                for (int l = pbc[2] ? -q_max : 0; l <= (pbc[2] ? q_max : 0); ++l) {
                    if (h == 0 && k == 0 && l == 0) continue;
                    if (h < 0 || (h == 0 && k < 0) || (h == 0 && k == 0 && l < 0)) continue;
                    std::array<double, 3> real{{0.0, 0.0, 0.0}};
                    std::array<double, 3> imaginary{{0.0, 0.0, 0.0}};
                    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
                        const auto& value = fractional[static_cast<std::size_t>(atom)];
                        const double phase = kTwoPi * (h * value[0] + k * value[1] + l * value[2]);
                        const double cosine = std::cos(phase);
                        const double sine = std::sin(phase);
                        for (int axis = 0; axis < 3; ++axis) {
                            real[axis] += spins[atom * 3 + axis] * cosine;
                            imaginary[axis] += spins[atom * 3 + axis] * sine;
                        }
                    }
                    double intensity = 0.0;
                    for (int axis = 0; axis < 3; ++axis) {
                        intensity += real[axis] * real[axis] + imaginary[axis] * imaginary[axis];
                    }
                    intensity = std::min(1.0, 2.0 * intensity / (total_magnitude * total_magnitude));
                    if (intensity > q_peak) {
                        q_peak = intensity;
                        q_vector = {{h, k, l}};
                    }
                }
            }
        }
    }
    const auto classification = classify_order(
        active_count, net_ratio, collinearity, coplanarity,
        neighbor_correlation, antiparallel_fraction, q_peak);
    return py::make_tuple(
        active_count, mean_magnitude, std::sqrt(variance), net_ratio,
        collinearity, coplanarity, neighbor_correlation, neighbor_abs_correlation,
        parallel_fraction, antiparallel_fraction, q_peak,
        q_vector[0], q_vector[1], q_vector[2], classification.first, classification.second);
}

std::string classify_element_order(
    const std::int64_t active_count,
    const double net_ratio,
    const double collinearity,
    const double coplanarity,
    const double q_peak,
    const double intra_correlation,
    const std::int64_t intra_pair_count
) {
    if (active_count == 0) return "low_moment";
    if (active_count < 2) return "insufficient";
    if (collinearity >= 0.90) {
        if (net_ratio >= 0.80) return "aligned";
        if (net_ratio <= 0.20 && (
                q_peak >= 0.45 ||
                (intra_pair_count > 0 && intra_correlation <= -0.25))) {
            return "compensated";
        }
        return "collinear_mixed";
    }
    if (q_peak >= 0.52 && coplanarity >= 0.72) return "modulated";
    if (q_peak >= 0.32) return "noncollinear";
    return "disordered";
}

py::tuple element_magnetic_evidence(
    py::array_t<float, py::array::c_style | py::array::forcecast> positions_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> cell_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> pbc_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> spins_array,
    py::array_t<std::int16_t, py::array::c_style | py::array::forcecast> atomic_numbers_array,
    const int requested_neighbors,
    const int q_max
) {
    if (positions_array.ndim() != 2 || positions_array.shape(1) != 3 ||
        spins_array.ndim() != 2 || spins_array.shape(1) != 3 ||
        positions_array.shape(0) != spins_array.shape(0) ||
        atomic_numbers_array.ndim() != 1 ||
        atomic_numbers_array.shape(0) != positions_array.shape(0) ||
        cell_array.ndim() != 2 || cell_array.shape(0) != 3 || cell_array.shape(1) != 3 ||
        pbc_array.ndim() != 1 || pbc_array.shape(0) != 3) {
        throw py::value_error(
            "positions, spins, atomic_numbers, cell, and pbc must have shapes "
            "(N,3), (N,3), (N,), (3,3), and (3,)"
        );
    }
    if (requested_neighbors <= 0 || q_max <= 0) {
        throw py::value_error("neighbors and q_max must be positive");
    }
    const std::int64_t atom_count = static_cast<std::int64_t>(positions_array.shape(0));
    const float* spins = spins_array.data();
    const std::int16_t* atomic_numbers = atomic_numbers_array.data();
    std::map<std::int16_t, std::vector<std::int64_t>> atoms_by_element;
    std::vector<double> magnitudes(static_cast<std::size_t>(atom_count), 0.0);
    std::vector<std::array<double, 3>> unit_spins(static_cast<std::size_t>(atom_count));
    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
        atoms_by_element[atomic_numbers[atom]].push_back(atom);
        const double x = spins[atom * 3];
        const double y = spins[atom * 3 + 1];
        const double z = spins[atom * 3 + 2];
        const double magnitude = std::sqrt(x * x + y * y + z * z);
        magnitudes[static_cast<std::size_t>(atom)] = magnitude;
        if (magnitude > kMomentEps) {
            unit_spins[static_cast<std::size_t>(atom)] = {{
                x / magnitude, y / magnitude, z / magnitude
            }};
        }
    }

    std::vector<std::array<double, 3>> fractional(static_cast<std::size_t>(atom_count));
    if (atom_count > 0) {
        const auto inverse = inverse_cell(cell_array.data());
        const float* positions = positions_array.data();
        for (std::int64_t atom = 0; atom < atom_count; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                double value = positions[atom * 3] * inverse[axis]
                    + positions[atom * 3 + 1] * inverse[3 + axis]
                    + positions[atom * 3 + 2] * inverse[6 + axis];
                fractional[static_cast<std::size_t>(atom)][axis] = value - std::floor(value);
            }
        }
    }

    neptrainkit::native::PeriodicNeighborSearch<float>::NeighborRows neighbors;
    if (atom_count > 1) {
        const neptrainkit::native::PeriodicNeighborSearch<float> search(
            positions_array.data(), atom_count, cell_array.data(), pbc_array.data());
        neighbors = search.query_knn(std::min<int>(requested_neighbors, atom_count - 1));
    } else {
        neighbors.resize(static_cast<std::size_t>(atom_count));
    }

    py::list element_rows;
    const bool* pbc = pbc_array.data();
    for (const auto& entry : atoms_by_element) {
        const std::int16_t atomic_number = entry.first;
        const auto& atoms = entry.second;
        std::int64_t active_count = 0;
        double total_magnitude = 0.0;
        std::array<double, 3> net{{0.0, 0.0, 0.0}};
        std::array<double, 9> orientation{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        for (const std::int64_t atom : atoms) {
            const double magnitude = magnitudes[static_cast<std::size_t>(atom)];
            total_magnitude += magnitude;
            net[0] += spins[atom * 3];
            net[1] += spins[atom * 3 + 1];
            net[2] += spins[atom * 3 + 2];
            if (magnitude <= kMomentEps) continue;
            ++active_count;
            const auto& unit = unit_spins[static_cast<std::size_t>(atom)];
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    orientation[row * 3 + column] += unit[row] * unit[column];
                }
            }
        }
        const double mean_magnitude = atoms.empty() ? 0.0 : total_magnitude / atoms.size();
        const double net_ratio = total_magnitude > kMomentEps
            ? std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]) / total_magnitude
            : 0.0;
        double collinearity = 0.0;
        double coplanarity = 0.0;
        if (active_count > 0) {
            for (double& value : orientation) value /= active_count;
            const auto eigenvalues = symmetric_eigenvalues(orientation);
            collinearity = std::max(0.0, std::min(1.0, eigenvalues[2]));
            coplanarity = std::max(0.0, std::min(1.0, 1.0 - 3.0 * eigenvalues[0]));
        }

        double intra_correlation = 0.0;
        std::int64_t intra_pair_count = 0;
        for (const std::int64_t atom : atoms) {
            if (magnitudes[static_cast<std::size_t>(atom)] <= kMomentEps) continue;
            const auto& left = unit_spins[static_cast<std::size_t>(atom)];
            for (const auto& neighbor : neighbors[static_cast<std::size_t>(atom)]) {
                const std::int64_t source = neighbor.source;
                if (source < 0 || source == atom || atomic_numbers[source] != atomic_number ||
                    magnitudes[static_cast<std::size_t>(source)] <= kMomentEps) continue;
                const auto& right = unit_spins[static_cast<std::size_t>(source)];
                intra_correlation += std::max(-1.0, std::min(1.0,
                    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]));
                ++intra_pair_count;
            }
        }
        if (intra_pair_count > 0) intra_correlation /= intra_pair_count;

        double q_peak = 0.0;
        std::array<int, 3> q_vector{{0, 0, 0}};
        if (active_count > 1 && total_magnitude > kMomentEps) {
            for (int h = pbc[0] ? -q_max : 0; h <= (pbc[0] ? q_max : 0); ++h) {
                for (int k = pbc[1] ? -q_max : 0; k <= (pbc[1] ? q_max : 0); ++k) {
                    for (int l = pbc[2] ? -q_max : 0; l <= (pbc[2] ? q_max : 0); ++l) {
                        if (h == 0 && k == 0 && l == 0) continue;
                        if (h < 0 || (h == 0 && k < 0) || (h == 0 && k == 0 && l < 0)) continue;
                        std::array<double, 3> real{{0.0, 0.0, 0.0}};
                        std::array<double, 3> imaginary{{0.0, 0.0, 0.0}};
                        for (const std::int64_t atom : atoms) {
                            const auto& value = fractional[static_cast<std::size_t>(atom)];
                            const double phase = kTwoPi * (h * value[0] + k * value[1] + l * value[2]);
                            const double cosine = std::cos(phase);
                            const double sine = std::sin(phase);
                            for (int axis = 0; axis < 3; ++axis) {
                                real[axis] += spins[atom * 3 + axis] * cosine;
                                imaginary[axis] += spins[atom * 3 + axis] * sine;
                            }
                        }
                        double intensity = 0.0;
                        for (int axis = 0; axis < 3; ++axis) {
                            intensity += real[axis] * real[axis] + imaginary[axis] * imaginary[axis];
                        }
                        intensity = std::min(1.0, 2.0 * intensity / (total_magnitude * total_magnitude));
                        if (intensity > q_peak) {
                            q_peak = intensity;
                            q_vector = {{h, k, l}};
                        }
                    }
                }
            }
        }
        const std::string label = classify_element_order(
            active_count, net_ratio, collinearity, coplanarity, q_peak,
            intra_correlation, intra_pair_count);
        element_rows.append(py::make_tuple(
            atomic_number, static_cast<std::int64_t>(atoms.size()), active_count,
            mean_magnitude, net_ratio, collinearity, intra_correlation,
            intra_pair_count, q_peak, q_vector[0], q_vector[1], q_vector[2], label));
    }

    struct PairAccumulator {
        std::int64_t count = 0;
        double correlation = 0.0;
    };
    std::map<std::pair<std::int16_t, std::int16_t>, PairAccumulator> pair_accumulators;
    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
        if (magnitudes[static_cast<std::size_t>(atom)] <= kMomentEps) continue;
        const auto& left = unit_spins[static_cast<std::size_t>(atom)];
        for (const auto& neighbor : neighbors[static_cast<std::size_t>(atom)]) {
            const std::int64_t source = neighbor.source;
            if (source < 0 || source == atom || atomic_numbers[source] == atomic_numbers[atom] ||
                magnitudes[static_cast<std::size_t>(source)] <= kMomentEps) continue;
            const auto key = std::minmax(atomic_numbers[atom], atomic_numbers[source]);
            auto& accumulator = pair_accumulators[{key.first, key.second}];
            const auto& right = unit_spins[static_cast<std::size_t>(source)];
            accumulator.correlation += std::max(-1.0, std::min(1.0,
                left[0] * right[0] + left[1] * right[1] + left[2] * right[2]));
            ++accumulator.count;
        }
    }
    py::list pair_rows;
    for (const auto& entry : pair_accumulators) {
        const double correlation = entry.second.count > 0
            ? entry.second.correlation / entry.second.count : 0.0;
        const std::string label = correlation >= 0.50
            ? "parallel" : correlation <= -0.50 ? "antiparallel" : "mixed";
        pair_rows.append(py::make_tuple(
            entry.first.first, entry.first.second, entry.second.count,
            correlation, label));
    }
    return py::make_tuple(element_rows, pair_rows);
}

}  // namespace

PYBIND11_MODULE(_magnetism, module) {
    module.doc() = "Native magnetic-order evidence for Training Set Audit";
    module.def(
        "magnetic_order_evidence", &magnetic_order_evidence,
        py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::arg("spins"),
        py::arg("neighbors") = 12, py::arg("q_max") = 3);
    module.def(
        "element_magnetic_evidence", &element_magnetic_evidence,
        py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::arg("spins"),
        py::arg("atomic_numbers"), py::arg("neighbors") = 12, py::arg("q_max") = 3);
}
