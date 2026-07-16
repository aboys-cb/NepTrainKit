// Batched geometry primitives for Training Set Audit.
// The Python layer owns audit rules and findings; this module only answers
// whether each structure contains a distinct atom pair within a cutoff.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "neptrainkit/native/periodic_neighbors.hpp"

namespace py = pybind11;

namespace {

struct NeighborFrame {
    std::vector<std::int32_t> centers;
    std::vector<std::int32_t> neighbors;
    std::vector<double> distances;
};

struct NormalizedContactFrame {
    std::vector<std::int32_t> codes;
    std::vector<double> values;
};

double determinant(const double* a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
         - a[1] * (a[3] * a[8] - a[5] * a[6])
         + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

int periodic_independence(const double* cell, const std::uint8_t* pbc) {
    double basis[9] = {0.0};
    int basis_count = 0;
    bool ambiguous = false;
    for (int row = 0; row < 3; ++row) {
        if (!pbc[row]) {
            continue;
        }
        double vector[3] = {cell[3 * row], cell[3 * row + 1], cell[3 * row + 2]};
        for (int pass = 0; pass < 2; ++pass) {
            for (int previous = 0; previous < basis_count; ++previous) {
                const double* direction = basis + 3 * previous;
                const double projection =
                    vector[0] * direction[0] + vector[1] * direction[1] + vector[2] * direction[2];
                vector[0] -= projection * direction[0];
                vector[1] -= projection * direction[1];
                vector[2] -= projection * direction[2];
            }
        }
        const double residual = std::sqrt(
            vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
        if (residual <= 1.0e-14) {
            return 0;
        }
        if (residual <= 1.0e-10) {
            ambiguous = true;
        }
        double* direction = basis + 3 * basis_count;
        direction[0] = vector[0] / residual;
        direction[1] = vector[1] / residual;
        direction[2] = vector[2] / residual;
        ++basis_count;
    }
    return ambiguous ? 2 : 1;
}

std::uint8_t cell_status(const double* cell, const std::uint8_t* pbc) {
    for (int value = 0; value < 9; ++value) {
        if (!std::isfinite(cell[value])) {
            return 0;
        }
    }
    const bool periodic = pbc[0] || pbc[1] || pbc[2];
    if (!periodic) {
        return 3;
    }
    for (int row = 0; row < 3; ++row) {
        if (!pbc[row]) {
            continue;
        }
        const double squared_norm =
            cell[3 * row] * cell[3 * row]
            + cell[3 * row + 1] * cell[3 * row + 1]
            + cell[3 * row + 2] * cell[3 * row + 2];
        if (squared_norm <= 1.0e-24) {
            return 0;
        }
    }
    const int independence = periodic_independence(cell, pbc);
    if (independence == 0) {
        return 0;
    }
    if (independence == 2) {
        return 4;
    }
    const bool invertible = std::abs(determinant(cell)) > 1.0e-12;
    return static_cast<std::uint8_t>(1 | (invertible ? 2 : 0));
}


void collect_cutoff_neighbors_frame(
    const double* positions,
    std::int64_t begin,
    std::int64_t end,
    const double* cell,
    const std::uint8_t* pbc,
    double cutoff,
    NeighborFrame& output
) {
    const std::int64_t atom_count = end - begin;
    if (atom_count <= 0) return;
    const bool flags[3] = {pbc[0] != 0, pbc[1] != 0, pbc[2] != 0};
    const neptrainkit::native::PeriodicNeighborSearch<double> search(
        positions + 3 * begin, atom_count, cell, flags
    );
    auto neighbors = search.query_radius(cutoff);
    output.centers = std::move(neighbors.centers);
    output.neighbors = std::move(neighbors.sources);
    output.distances = std::move(neighbors.distances);
}

}  // namespace

py::array_t<std::uint8_t> short_distance_mask(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> cells,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> pbc,
    double cutoff
) {
    const auto position_info = positions.request();
    const auto offset_info = offsets.request();
    const auto cell_info = cells.request();
    const auto pbc_info = pbc.request();

    if (cutoff < 0.0 || !std::isfinite(cutoff)) {
        throw std::invalid_argument("cutoff must be finite and non-negative");
    }
    if (position_info.ndim != 2 || position_info.shape[1] != 3) {
        throw std::invalid_argument("positions must have shape (N, 3)");
    }
    if (offset_info.ndim != 1 || offset_info.shape[0] < 1) {
        throw std::invalid_argument("offsets must have shape (M + 1,)");
    }
    const py::ssize_t frame_count = offset_info.shape[0] - 1;
    if (cell_info.ndim != 3 || cell_info.shape[0] != frame_count ||
        cell_info.shape[1] != 3 || cell_info.shape[2] != 3) {
        throw std::invalid_argument("cells must have shape (M, 3, 3)");
    }
    if (pbc_info.ndim != 2 || pbc_info.shape[0] != frame_count || pbc_info.shape[1] != 3) {
        throw std::invalid_argument("pbc must have shape (M, 3)");
    }

    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto atom_count = static_cast<std::int64_t>(position_info.shape[0]);
    if (offset_data[0] != 0 || offset_data[frame_count] != atom_count) {
        throw std::invalid_argument("offsets must start at zero and end at the number of positions");
    }
    for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
        if (offset_data[frame] > offset_data[frame + 1]) {
            throw std::invalid_argument("offsets must be monotonic");
        }
    }

    const auto* cell_data = static_cast<const double*>(cell_info.ptr);
    const auto* pbc_data = static_cast<const std::uint8_t*>(pbc_info.ptr);
    for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
        const std::uint8_t* flags = pbc_data + 3 * frame;
        if ((flags[0] || flags[1] || flags[2]) &&
            std::abs(determinant(cell_data + 9 * frame)) <= 1.0e-12) {
            throw std::invalid_argument("periodic native scans require a nonsingular cell");
        }
    }

    py::array_t<std::uint8_t> result(frame_count);
    auto* result_data = result.mutable_data();
    const auto* position_data = static_cast<const double*>(position_info.ptr);

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
            const std::int64_t begin = offset_data[frame];
            const std::int64_t end = offset_data[frame + 1];
            if (end - begin < 2) {
                result_data[frame] = 0;
                continue;
            }
            const std::uint8_t* raw_flags = pbc_data + 3 * frame;
            const bool flags[3] = {
                raw_flags[0] != 0, raw_flags[1] != 0, raw_flags[2] != 0
            };
            const neptrainkit::native::PeriodicNeighborSearch<double> search(
                position_data + 3 * begin,
                end - begin,
                cell_data + 9 * frame,
                flags
            );
            result_data[frame] = search.any_distinct_pair_within(cutoff) ? 1 : 0;
        }
    }
    return result;
}

py::array_t<std::uint8_t> cell_status_mask(
    py::array_t<double, py::array::c_style | py::array::forcecast> cells,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> pbc
) {
    const auto cell_info = cells.request();
    const auto pbc_info = pbc.request();
    if (cell_info.ndim != 3 || cell_info.shape[1] != 3 || cell_info.shape[2] != 3) {
        throw std::invalid_argument("cells must have shape (M, 3, 3)");
    }
    const py::ssize_t frame_count = cell_info.shape[0];
    if (pbc_info.ndim != 2 || pbc_info.shape[0] != frame_count || pbc_info.shape[1] != 3) {
        throw std::invalid_argument("pbc must have shape (M, 3)");
    }
    const auto* cell_data = static_cast<const double*>(cell_info.ptr);
    const auto* pbc_data = static_cast<const std::uint8_t*>(pbc_info.ptr);
    py::array_t<std::uint8_t> result(frame_count);
    auto* result_data = result.mutable_data();
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
            result_data[frame] = cell_status(cell_data + 9 * frame, pbc_data + 3 * frame);
        }
    }
    return result;
}

py::tuple cutoff_neighbor_pairs(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> cells,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> pbc,
    double cutoff
) {
    const auto position_info = positions.request();
    const auto offset_info = offsets.request();
    const auto cell_info = cells.request();
    const auto pbc_info = pbc.request();
    if (cutoff <= 0.0 || !std::isfinite(cutoff)) {
        throw std::invalid_argument("cutoff must be finite and positive");
    }
    if (position_info.ndim != 2 || position_info.shape[1] != 3) {
        throw std::invalid_argument("positions must have shape (N, 3)");
    }
    if (offset_info.ndim != 1 || offset_info.shape[0] < 1) {
        throw std::invalid_argument("offsets must have shape (M + 1,)");
    }
    const py::ssize_t frame_count = offset_info.shape[0] - 1;
    if (cell_info.ndim != 3 || cell_info.shape[0] != frame_count ||
        cell_info.shape[1] != 3 || cell_info.shape[2] != 3) {
        throw std::invalid_argument("cells must have shape (M, 3, 3)");
    }
    if (pbc_info.ndim != 2 || pbc_info.shape[0] != frame_count || pbc_info.shape[1] != 3) {
        throw std::invalid_argument("pbc must have shape (M, 3)");
    }
    const auto* position_data = static_cast<const double*>(position_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* cell_data = static_cast<const double*>(cell_info.ptr);
    const auto* pbc_data = static_cast<const std::uint8_t*>(pbc_info.ptr);
    const std::int64_t atom_count = static_cast<std::int64_t>(position_info.shape[0]);
    if (offset_data[0] != 0 || offset_data[frame_count] != atom_count) {
        throw std::invalid_argument("offsets must start at zero and end at the number of positions");
    }
    for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
        if (offset_data[frame] > offset_data[frame + 1]) {
            throw std::invalid_argument("offsets must be monotonic");
        }
        const std::uint8_t* flags = pbc_data + 3 * frame;
        if ((flags[0] || flags[1] || flags[2]) &&
            std::abs(determinant(cell_data + 9 * frame)) <= 1.0e-12) {
            throw std::invalid_argument("periodic native scans require a nonsingular cell");
        }
    }

    std::vector<NeighborFrame> frames(static_cast<std::size_t>(frame_count));
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
            collect_cutoff_neighbors_frame(
                position_data,
                offset_data[frame],
                offset_data[frame + 1],
                cell_data + 9 * frame,
                pbc_data + 3 * frame,
                cutoff,
                frames[static_cast<std::size_t>(frame)]);
        }
    }

    py::array_t<std::int64_t> pair_offsets(frame_count + 1);
    auto* pair_offset_data = pair_offsets.mutable_data();
    pair_offset_data[0] = 0;
    for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
        pair_offset_data[frame + 1] = pair_offset_data[frame]
            + static_cast<std::int64_t>(frames[static_cast<std::size_t>(frame)].centers.size());
    }
    const py::ssize_t pair_count = static_cast<py::ssize_t>(pair_offset_data[frame_count]);
    py::array_t<std::int32_t> centers(pair_count);
    py::array_t<std::int32_t> neighbors(pair_count);
    py::array_t<double> distances(pair_count);
    auto* center_data = centers.mutable_data();
    auto* neighbor_data = neighbors.mutable_data();
    auto* distance_data = distances.mutable_data();
    for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
        const auto& source = frames[static_cast<std::size_t>(frame)];
        const std::int64_t target_offset = pair_offset_data[frame];
        for (std::size_t pair = 0; pair < source.centers.size(); ++pair) {
            const std::int64_t target = target_offset + static_cast<std::int64_t>(pair);
            center_data[target] = source.centers[pair];
            neighbor_data[target] = source.neighbors[pair];
            distance_data[target] = source.distances[pair];
        }
    }
    return py::make_tuple(pair_offsets, centers, neighbors, distances);
}

py::tuple local_chemistry_summary(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> atom_offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> cells,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> pbc,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_types,
    py::array_t<double, py::array::c_style | py::array::forcecast> cutoff_matrices,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> detail_mask
) {
    const auto position_info = positions.request();
    const auto offset_info = atom_offsets.request();
    const auto cell_info = cells.request();
    const auto pbc_info = pbc.request();
    const auto type_info = atom_types.request();
    const auto cutoff_info = cutoff_matrices.request();
    const auto detail_info = detail_mask.request();
    if (position_info.ndim != 2 || position_info.shape[1] != 3 ||
        offset_info.ndim != 1 || offset_info.shape[0] < 1 ||
        type_info.ndim != 1 || type_info.shape[0] != position_info.shape[0]) {
        throw std::invalid_argument("local chemistry positions, offsets, and atom types are incompatible");
    }
    const py::ssize_t frame_count = offset_info.shape[0] - 1;
    if (cell_info.ndim != 3 || cell_info.shape[0] != frame_count ||
        cell_info.shape[1] != 3 || cell_info.shape[2] != 3 ||
        pbc_info.ndim != 2 || pbc_info.shape[0] != frame_count || pbc_info.shape[1] != 3) {
        throw std::invalid_argument("local chemistry cells and pbc have incompatible shapes");
    }
    if (cutoff_info.ndim != 3 || cutoff_info.shape[1] != cutoff_info.shape[2]) {
        throw std::invalid_argument("cutoff_matrices must have shape (S, T, T)");
    }
    const py::ssize_t scope_count = cutoff_info.shape[0];
    const py::ssize_t type_count = cutoff_info.shape[1];
    const py::ssize_t type_pair_count = type_count * (type_count + 1) / 2;
    if (detail_info.ndim != 2 || detail_info.shape[0] != scope_count ||
        detail_info.shape[1] != type_pair_count) {
        throw std::invalid_argument("detail_mask must have shape (S, T * (T + 1) / 2)");
    }

    const auto* position_data = static_cast<const double*>(position_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* cell_data = static_cast<const double*>(cell_info.ptr);
    const auto* pbc_data = static_cast<const std::uint8_t*>(pbc_info.ptr);
    const auto* type_data = static_cast<const std::int32_t*>(type_info.ptr);
    const auto* cutoff_data = static_cast<const double*>(cutoff_info.ptr);
    const auto* detail_data = static_cast<const std::uint8_t*>(detail_info.ptr);
    const py::ssize_t total_atoms = position_info.shape[0];
    if (offset_data[0] != 0 || offset_data[frame_count] != total_atoms) {
        throw std::invalid_argument("atom_offsets must span every input atom");
    }
    double maximum_cutoff = 0.0;
    for (py::ssize_t value = 0; value < cutoff_info.size; ++value) {
        if (!(cutoff_data[value] > 0.0) || !std::isfinite(cutoff_data[value])) {
            throw std::invalid_argument("cutoff_matrices must contain finite positive values");
        }
        maximum_cutoff = std::max(maximum_cutoff, cutoff_data[value]);
    }
    for (py::ssize_t atom = 0; atom < total_atoms; ++atom) {
        if (type_data[atom] < 0 || type_data[atom] >= type_count) {
            throw std::invalid_argument("atom type is outside cutoff_matrices");
        }
    }
    for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
        if (offset_data[frame] > offset_data[frame + 1]) {
            throw std::invalid_argument("atom_offsets must be monotonic");
        }
        const std::uint8_t* flags = pbc_data + 3 * frame;
        if ((flags[0] || flags[1] || flags[2]) &&
            std::abs(determinant(cell_data + 9 * frame)) <= 1.0e-12) {
            throw std::invalid_argument("periodic local chemistry requires a nonsingular cell");
        }
    }

    py::array_t<std::int32_t> counts({scope_count, total_atoms});
    py::array_t<std::int32_t> type_counts({scope_count, total_atoms, type_count});
    py::array_t<double> metrics({frame_count, scope_count, type_pair_count, py::ssize_t(8)});
    auto* count_data = counts.mutable_data();
    auto* type_count_data = type_counts.mutable_data();
    auto* metric_data = metrics.mutable_data();
    std::fill(count_data, count_data + counts.size(), std::int32_t(0));
    std::fill(type_count_data, type_count_data + type_counts.size(), std::int32_t(0));
    std::fill(metric_data, metric_data + metrics.size(), 0.0);
    std::vector<NormalizedContactFrame> normalized(static_cast<std::size_t>(frame_count));

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
            const std::int64_t atom_begin = offset_data[frame];
            const std::int64_t atom_end = offset_data[frame + 1];
            const std::int32_t frame_atom_count = static_cast<std::int32_t>(atom_end - atom_begin);
            NeighborFrame edges;
            collect_cutoff_neighbors_frame(
                position_data,
                atom_begin,
                atom_end,
                cell_data + 9 * frame,
                pbc_data + 3 * frame,
                maximum_cutoff,
                edges
            );

            std::vector<std::int32_t> populations(static_cast<std::size_t>(type_count), 0);
            for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
                ++populations[static_cast<std::size_t>(type_data[atom])];
            }
            const std::size_t scope_pair_count = static_cast<std::size_t>(scope_count * type_pair_count);
            std::vector<std::int32_t> nonself_opportunities(scope_pair_count, 0);
            std::vector<std::int32_t> self_opportunities(scope_pair_count, 0);
            std::vector<std::int32_t> contacts(scope_pair_count, 0);
            std::vector<std::uint8_t> first_exposed(
                scope_pair_count * static_cast<std::size_t>(frame_atom_count), 0
            );
            std::vector<std::uint8_t> second_exposed(
                scope_pair_count * static_cast<std::size_t>(frame_atom_count), 0
            );
            std::vector<std::vector<double>> normalized_by_code(scope_pair_count);
            std::vector<double> code_cutoffs(scope_pair_count, 0.0);
            std::vector<std::uint8_t> active_codes(scope_pair_count, 0);
            std::vector<std::size_t> nonself_codes;
            std::vector<std::size_t> self_codes;
            nonself_codes.reserve(scope_pair_count);
            self_codes.reserve(static_cast<std::size_t>(scope_count * type_count));
            std::int32_t active_type_pair = 0;
            for (std::int32_t first = 0; first < type_count; ++first) {
                for (
                    std::int32_t second = first;
                    second < type_count;
                    ++second, ++active_type_pair
                ) {
                    const bool same_type = first == second;
                    const bool co_sampled =
                        populations[static_cast<std::size_t>(first)] >= (same_type ? 2 : 1) &&
                        populations[static_cast<std::size_t>(second)] >= 1;
                    if (!co_sampled) continue;
                    for (py::ssize_t scope = 0; scope < scope_count; ++scope) {
                        const std::size_t code = static_cast<std::size_t>(
                            scope * type_pair_count + active_type_pair
                        );
                        code_cutoffs[code] = cutoff_data[
                            (scope * type_count + first) * type_count + second
                        ];
                        active_codes[code] = 1;
                        nonself_codes.push_back(code);
                        if (same_type) self_codes.push_back(code);
                    }
                }
            }

            for (std::size_t edge = 0; edge < edges.centers.size(); ++edge) {
                const std::int32_t center = edges.centers[edge];
                const std::int32_t neighbor = edges.neighbors[edge];
                const double distance = edges.distances[edge];
                const std::int32_t center_type = type_data[atom_begin + center];
                const std::int32_t neighbor_type = type_data[atom_begin + neighbor];
                const bool same_parent = center == neighbor;
                for (py::ssize_t scope = 0; scope < scope_count; ++scope) {
                    const double cutoff = cutoff_data[
                        (scope * type_count + center_type) * type_count + neighbor_type
                    ];
                    if (distance < cutoff) {
                        const std::int64_t global_center = atom_begin + center;
                        ++count_data[scope * total_atoms + global_center];
                        ++type_count_data[
                            (scope * total_atoms + global_center) * type_count + neighbor_type
                        ];
                    }
                }

                const auto& opportunity_codes = same_parent ? self_codes : nonself_codes;
                for (const std::size_t code : opportunity_codes) {
                    if (distance >= code_cutoffs[code]) continue;
                    if (same_parent) {
                        ++self_opportunities[code];
                    } else {
                        ++nonself_opportunities[code];
                    }
                }

                const std::int32_t first_type = std::min(center_type, neighbor_type);
                const std::int32_t second_type = std::max(center_type, neighbor_type);
                const std::int32_t actual_type_pair =
                    first_type * type_count - first_type * (first_type - 1) / 2
                    + second_type - first_type;
                for (py::ssize_t scope = 0; scope < scope_count; ++scope) {
                    const std::size_t code = static_cast<std::size_t>(
                        scope * type_pair_count + actual_type_pair
                    );
                    const double cutoff = code_cutoffs[code];
                    if (!active_codes[code] || distance >= cutoff) continue;
                    ++contacts[code];
                    const std::size_t exposure_base =
                        code * static_cast<std::size_t>(frame_atom_count);
                    if (center_type == first_type) {
                        first_exposed[exposure_base + static_cast<std::size_t>(center)] = 1;
                    } else {
                        second_exposed[exposure_base + static_cast<std::size_t>(center)] = 1;
                    }
                    if (detail_data[code]) {
                        normalized_by_code[code].push_back(distance / cutoff);
                    }
                }
            }

            std::int32_t type_pair = 0;
            for (std::int32_t first = 0; first < type_count; ++first) {
                for (std::int32_t second = first; second < type_count; ++second, ++type_pair) {
                    const bool same_type = first == second;
                    const std::int32_t first_count = populations[static_cast<std::size_t>(first)];
                    const std::int32_t second_count = populations[static_cast<std::size_t>(second)];
                    const bool co_sampled = first_count >= (same_type ? 2 : 1) && second_count >= 1;
                    if (!co_sampled || frame_atom_count == 0) continue;
                    for (py::ssize_t scope = 0; scope < scope_count; ++scope) {
                        const std::size_t code = static_cast<std::size_t>(
                            scope * type_pair_count + type_pair
                        );
                        const std::int64_t metric_base =
                            (((frame * scope_count + scope) * type_pair_count + type_pair) * 8);
                        metric_data[metric_base] = 1.0;
                        metric_data[metric_base + 1] =
                            nonself_opportunities[code] + self_opportunities[code];
                        const double denominator = frame_atom_count < 2
                            ? 1.0
                            : static_cast<double>(frame_atom_count) * (frame_atom_count - 1);
                        const double probability = frame_atom_count < 2
                            ? 0.0
                            : same_type
                                ? static_cast<double>(first_count) * (first_count - 1) / denominator
                                : 2.0 * first_count * second_count / denominator;
                        metric_data[metric_base + 2] = nonself_opportunities[code] * probability
                            + (same_type
                                ? self_opportunities[code] * first_count /
                                    static_cast<double>(frame_atom_count)
                                : 0.0);
                        metric_data[metric_base + 3] = contacts[code];
                        metric_data[metric_base + 4] = first_count;
                        metric_data[metric_base + 5] = same_type ? 0.0 : second_count;
                        const std::size_t exposure_base =
                            code * static_cast<std::size_t>(frame_atom_count);
                        metric_data[metric_base + 6] = std::count(
                            first_exposed.begin() + exposure_base,
                            first_exposed.begin() + exposure_base + frame_atom_count,
                            std::uint8_t(1)
                        );
                        metric_data[metric_base + 7] = std::count(
                            second_exposed.begin() + exposure_base,
                            second_exposed.begin() + exposure_base + frame_atom_count,
                            std::uint8_t(1)
                        );
                        for (const double value : normalized_by_code[code]) {
                            normalized[static_cast<std::size_t>(frame)].codes.push_back(
                                static_cast<std::int32_t>(code)
                            );
                            normalized[static_cast<std::size_t>(frame)].values.push_back(value);
                        }
                    }
                }
            }
        }
    }

    std::size_t normalized_count = 0;
    for (const auto& frame : normalized) normalized_count += frame.codes.size();
    py::array_t<std::int32_t> normalized_codes(static_cast<py::ssize_t>(normalized_count));
    py::array_t<double> normalized_values(static_cast<py::ssize_t>(normalized_count));
    auto* code_data = normalized_codes.mutable_data();
    auto* value_data = normalized_values.mutable_data();
    std::size_t target = 0;
    for (const auto& frame : normalized) {
        for (std::size_t index = 0; index < frame.codes.size(); ++index, ++target) {
            code_data[target] = frame.codes[index];
            value_data[target] = frame.values[index];
        }
    }
    return py::make_tuple(counts, type_counts, metrics, normalized_codes, normalized_values);
}


PYBIND11_MODULE(_audit, module) {
    module.doc() = "Batched native geometry primitives for Training Set Audit";
    module.def(
        "short_distance_mask",
        &short_distance_mask,
        py::arg("positions"),
        py::arg("offsets"),
        py::arg("cells"),
        py::arg("pbc"),
        py::arg("cutoff"),
        "Return one uint8 collision flag per structure."
    );
    module.def(
        "cell_status_mask",
        &cell_status_mask,
        py::arg("cells"),
        py::arg("pbc"),
        "Return bit 0 for valid cells and bit 1 for native-neighbor support."
    );
    module.def(
        "cutoff_neighbor_pairs",
        &cutoff_neighbor_pairs,
        py::arg("positions"),
        py::arg("offsets"),
        py::arg("cells"),
        py::arg("pbc"),
        py::arg("cutoff"),
        "Return pair offsets, local centers, local neighbors, and distances for a structure batch."
    );
    module.def(
        "local_chemistry_summary",
        &local_chemistry_summary,
        py::arg("positions"),
        py::arg("atom_offsets"),
        py::arg("cells"),
        py::arg("pbc"),
        py::arg("atom_types"),
        py::arg("cutoff_matrices"),
        py::arg("detail_mask"),
        "Fuse periodic neighbors, typed counts, and contact summaries for one structure batch."
    );
}
