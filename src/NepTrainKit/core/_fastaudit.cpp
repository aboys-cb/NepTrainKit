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

namespace py = pybind11;

namespace {

struct Vec3 {
    double x;
    double y;
    double z;
};

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

void inverse3x3(const double* a, double* out, double det) {
    const double inv_det = 1.0 / det;
    out[0] =  (a[4] * a[8] - a[5] * a[7]) * inv_det;
    out[1] =  (a[2] * a[7] - a[1] * a[8]) * inv_det;
    out[2] =  (a[1] * a[5] - a[2] * a[4]) * inv_det;
    out[3] =  (a[5] * a[6] - a[3] * a[8]) * inv_det;
    out[4] =  (a[0] * a[8] - a[2] * a[6]) * inv_det;
    out[5] =  (a[2] * a[3] - a[0] * a[5]) * inv_det;
    out[6] =  (a[3] * a[7] - a[4] * a[6]) * inv_det;
    out[7] =  (a[1] * a[6] - a[0] * a[7]) * inv_det;
    out[8] =  (a[0] * a[4] - a[1] * a[3]) * inv_det;
}

bool within_cutoff(const Vec3& delta, double cutoff_squared) {
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z <= cutoff_squared;
}

bool scan_nonperiodic(
    const double* positions,
    std::int64_t begin,
    std::int64_t end,
    double cutoff_squared
) {
    for (std::int64_t i = begin; i < end; ++i) {
        const double* first = positions + 3 * i;
        for (std::int64_t j = i + 1; j < end; ++j) {
            const double* second = positions + 3 * j;
            if (within_cutoff(
                    {first[0] - second[0], first[1] - second[1], first[2] - second[2]},
                    cutoff_squared)) {
                return true;
            }
        }
    }
    return false;
}

bool scan_periodic(
    const double* positions,
    std::int64_t begin,
    std::int64_t end,
    const double* cell,
    const std::uint8_t* pbc,
    double cutoff_squared
) {
    double inverse[9];
    inverse3x3(cell, inverse, determinant(cell));

    std::vector<Vec3> wrapped(static_cast<std::size_t>(end - begin));
    for (std::int64_t atom = begin; atom < end; ++atom) {
        const double* position = positions + 3 * atom;
        double fractional[3] = {
            position[0] * inverse[0] + position[1] * inverse[3] + position[2] * inverse[6],
            position[0] * inverse[1] + position[1] * inverse[4] + position[2] * inverse[7],
            position[0] * inverse[2] + position[1] * inverse[5] + position[2] * inverse[8],
        };
        for (int axis = 0; axis < 3; ++axis) {
            if (pbc[axis]) {
                fractional[axis] -= std::floor(fractional[axis]);
            }
        }
        wrapped[static_cast<std::size_t>(atom - begin)] = {
            fractional[0] * cell[0] + fractional[1] * cell[3] + fractional[2] * cell[6],
            fractional[0] * cell[1] + fractional[1] * cell[4] + fractional[2] * cell[7],
            fractional[0] * cell[2] + fractional[1] * cell[5] + fractional[2] * cell[8],
        };
    }

    std::vector<Vec3> translations;
    translations.reserve(27);
    for (int sx = pbc[0] ? -1 : 0; sx <= (pbc[0] ? 1 : 0); ++sx) {
        for (int sy = pbc[1] ? -1 : 0; sy <= (pbc[1] ? 1 : 0); ++sy) {
            for (int sz = pbc[2] ? -1 : 0; sz <= (pbc[2] ? 1 : 0); ++sz) {
                translations.push_back({
                    sx * cell[0] + sy * cell[3] + sz * cell[6],
                    sx * cell[1] + sy * cell[4] + sz * cell[7],
                    sx * cell[2] + sy * cell[5] + sz * cell[8],
                });
            }
        }
    }

    for (std::size_t i = 0; i < wrapped.size(); ++i) {
        for (std::size_t j = i + 1; j < wrapped.size(); ++j) {
            const Vec3 base = {
                wrapped[i].x - wrapped[j].x,
                wrapped[i].y - wrapped[j].y,
                wrapped[i].z - wrapped[j].z,
            };
            for (const Vec3& shift : translations) {
                if (within_cutoff(
                        {base.x + shift.x, base.y + shift.y, base.z + shift.z},
                        cutoff_squared)) {
                    return true;
                }
            }
        }
    }
    return false;
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
    if (atom_count <= 0) {
        return;
    }
    const bool periodic = pbc[0] || pbc[1] || pbc[2];
    double inverse[9] = {0.0};
    std::vector<Vec3> wrapped(static_cast<std::size_t>(atom_count));
    if (periodic) {
        inverse3x3(cell, inverse, determinant(cell));
    }
    for (std::int64_t local_atom = 0; local_atom < atom_count; ++local_atom) {
        const double* position = positions + 3 * (begin + local_atom);
        if (!periodic) {
            wrapped[static_cast<std::size_t>(local_atom)] = {position[0], position[1], position[2]};
            continue;
        }
        double fractional[3] = {
            position[0] * inverse[0] + position[1] * inverse[3] + position[2] * inverse[6],
            position[0] * inverse[1] + position[1] * inverse[4] + position[2] * inverse[7],
            position[0] * inverse[2] + position[1] * inverse[5] + position[2] * inverse[8],
        };
        for (int axis = 0; axis < 3; ++axis) {
            if (pbc[axis]) {
                fractional[axis] -= std::floor(fractional[axis]);
            }
        }
        wrapped[static_cast<std::size_t>(local_atom)] = {
            fractional[0] * cell[0] + fractional[1] * cell[3] + fractional[2] * cell[6],
            fractional[0] * cell[1] + fractional[1] * cell[4] + fractional[2] * cell[7],
            fractional[0] * cell[2] + fractional[1] * cell[5] + fractional[2] * cell[8],
        };
    }

    int image_counts[3] = {0, 0, 0};
    if (periodic) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!pbc[axis]) {
                continue;
            }
            const double reciprocal_norm = std::sqrt(
                inverse[axis] * inverse[axis]
                + inverse[3 + axis] * inverse[3 + axis]
                + inverse[6 + axis] * inverse[6 + axis]);
            image_counts[axis] = static_cast<int>(std::ceil(cutoff * reciprocal_norm));
        }
    }

    std::vector<Vec3> translations;
    const std::size_t translation_count = static_cast<std::size_t>(
        (2 * image_counts[0] + 1) * (2 * image_counts[1] + 1) * (2 * image_counts[2] + 1));
    translations.reserve(translation_count);
    for (int sx = -image_counts[0]; sx <= image_counts[0]; ++sx) {
        for (int sy = -image_counts[1]; sy <= image_counts[1]; ++sy) {
            for (int sz = -image_counts[2]; sz <= image_counts[2]; ++sz) {
                translations.push_back({
                    sx * cell[0] + sy * cell[3] + sz * cell[6],
                    sx * cell[1] + sy * cell[4] + sz * cell[7],
                    sx * cell[2] + sy * cell[5] + sz * cell[8],
                });
            }
        }
    }

    const double cutoff_squared = cutoff * cutoff;
    output.centers.reserve(static_cast<std::size_t>(atom_count * 64));
    output.neighbors.reserve(static_cast<std::size_t>(atom_count * 64));
    output.distances.reserve(static_cast<std::size_t>(atom_count * 64));
    for (std::int32_t center = 0; center < atom_count; ++center) {
        const Vec3 first = wrapped[static_cast<std::size_t>(center)];
        for (std::int32_t neighbor = 0; neighbor < atom_count; ++neighbor) {
            const Vec3 second = wrapped[static_cast<std::size_t>(neighbor)];
            for (std::size_t shift_index = 0; shift_index < translations.size(); ++shift_index) {
                if (center == neighbor && shift_index == translations.size() / 2) {
                    continue;
                }
                const Vec3 shift = translations[shift_index];
                const double dx = second.x + shift.x - first.x;
                const double dy = second.y + shift.y - first.y;
                const double dz = second.z + shift.z - first.z;
                const double squared = dx * dx + dy * dy + dz * dz;
                if (squared < cutoff_squared) {
                    output.centers.push_back(center);
                    output.neighbors.push_back(neighbor);
                    output.distances.push_back(std::sqrt(squared));
                }
            }
        }
    }
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
    const double cutoff_squared = cutoff * cutoff;

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
            const std::uint8_t* flags = pbc_data + 3 * frame;
            const bool periodic = flags[0] || flags[1] || flags[2];
            const bool found = periodic
                ? scan_periodic(
                      position_data,
                      begin,
                      end,
                      cell_data + 9 * frame,
                      flags,
                      cutoff_squared)
                : scan_nonperiodic(position_data, begin, end, cutoff_squared);
            result_data[frame] = found ? 1 : 0;
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

py::tuple typed_contact_summary(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> atom_offsets,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_types,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pair_offsets,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> centers,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> neighbors,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<double, py::array::c_style | py::array::forcecast> cutoff_matrices,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> detail_mask
) {
    const auto atom_offset_info = atom_offsets.request();
    const auto atom_type_info = atom_types.request();
    const auto pair_offset_info = pair_offsets.request();
    const auto center_info = centers.request();
    const auto neighbor_info = neighbors.request();
    const auto distance_info = distances.request();
    const auto cutoff_info = cutoff_matrices.request();
    const auto detail_info = detail_mask.request();
    if (atom_offset_info.ndim != 1 || atom_offset_info.shape[0] < 1 ||
        pair_offset_info.ndim != 1 || pair_offset_info.shape[0] != atom_offset_info.shape[0]) {
        throw std::invalid_argument("atom_offsets and pair_offsets must have matching shape (M + 1,)");
    }
    if (atom_type_info.ndim != 1 || center_info.ndim != 1 || neighbor_info.ndim != 1 ||
        distance_info.ndim != 1 || center_info.shape[0] != neighbor_info.shape[0] ||
        center_info.shape[0] != distance_info.shape[0]) {
        throw std::invalid_argument("atom types and pair arrays must be one dimensional");
    }
    if (cutoff_info.ndim != 3 || cutoff_info.shape[1] != cutoff_info.shape[2]) {
        throw std::invalid_argument("cutoff_matrices must have shape (S, T, T)");
    }
    const py::ssize_t frame_count = atom_offset_info.shape[0] - 1;
    const py::ssize_t scope_count = cutoff_info.shape[0];
    const py::ssize_t type_count = cutoff_info.shape[1];
    const py::ssize_t type_pair_count = type_count * (type_count + 1) / 2;
    if (detail_info.ndim != 2 || detail_info.shape[0] != scope_count ||
        detail_info.shape[1] != type_pair_count) {
        throw std::invalid_argument("detail_mask must have shape (S, T * (T + 1) / 2)");
    }
    const auto* atom_offset_data = static_cast<const std::int64_t*>(atom_offset_info.ptr);
    const auto* atom_type_data = static_cast<const std::int32_t*>(atom_type_info.ptr);
    const auto* pair_offset_data = static_cast<const std::int64_t*>(pair_offset_info.ptr);
    const auto* center_data = static_cast<const std::int32_t*>(center_info.ptr);
    const auto* neighbor_data = static_cast<const std::int32_t*>(neighbor_info.ptr);
    const auto* distance_data = static_cast<const double*>(distance_info.ptr);
    const auto* cutoff_data = static_cast<const double*>(cutoff_info.ptr);
    const auto* detail_data = static_cast<const std::uint8_t*>(detail_info.ptr);
    if (atom_offset_data[0] != 0 || atom_offset_data[frame_count] != atom_type_info.shape[0] ||
        pair_offset_data[0] != 0 || pair_offset_data[frame_count] != center_info.shape[0]) {
        throw std::invalid_argument("offset arrays do not match their flattened values");
    }

    py::array_t<double> metrics({frame_count, scope_count, type_pair_count, py::ssize_t(8)});
    auto* metric_data = metrics.mutable_data();
    std::fill(metric_data, metric_data + metrics.size(), 0.0);
    std::vector<NormalizedContactFrame> normalized(static_cast<std::size_t>(frame_count));
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
            const std::int64_t atom_begin = atom_offset_data[frame];
            const std::int64_t atom_end = atom_offset_data[frame + 1];
            const std::int32_t frame_atom_count = static_cast<std::int32_t>(atom_end - atom_begin);
            std::vector<std::int32_t> counts(static_cast<std::size_t>(type_count), 0);
            for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
                const std::int32_t type = atom_type_data[atom];
                if (type < 0 || type >= type_count) {
                    continue;
                }
                ++counts[static_cast<std::size_t>(type)];
            }
            std::int32_t type_pair = 0;
            for (std::int32_t first = 0; first < type_count; ++first) {
                for (std::int32_t second = first; second < type_count; ++second, ++type_pair) {
                    const bool same_type = first == second;
                    const std::int32_t first_count = counts[static_cast<std::size_t>(first)];
                    const std::int32_t second_count = counts[static_cast<std::size_t>(second)];
                    const bool co_sampled = first_count >= (same_type ? 2 : 1) && second_count >= 1;
                    if (!co_sampled || frame_atom_count == 0) {
                        continue;
                    }
                    for (py::ssize_t scope = 0; scope < scope_count; ++scope) {
                        const std::int64_t metric_base =
                            (((frame * scope_count + scope) * type_pair_count + type_pair) * 8);
                        metric_data[metric_base] = 1.0;
                        metric_data[metric_base + 4] = first_count;
                        metric_data[metric_base + 5] = same_type ? 0.0 : second_count;
                        const double cutoff = cutoff_data[(scope * type_count + first) * type_count + second];
                        std::int32_t nonself_opportunities = 0;
                        std::int32_t self_opportunities = 0;
                        std::int32_t contacts = 0;
                        std::vector<std::uint8_t> first_exposed(static_cast<std::size_t>(frame_atom_count), 0);
                        std::vector<std::uint8_t> second_exposed(static_cast<std::size_t>(frame_atom_count), 0);
                        for (std::int64_t edge = pair_offset_data[frame]; edge < pair_offset_data[frame + 1]; ++edge) {
                            const std::int32_t center = center_data[edge];
                            const std::int32_t neighbor = neighbor_data[edge];
                            const bool same_parent = center == neighbor;
                            if (distance_data[edge] >= cutoff || (!same_type && same_parent)) {
                                continue;
                            }
                            if (same_parent) {
                                ++self_opportunities;
                            } else {
                                ++nonself_opportunities;
                            }
                            const std::int32_t center_type = atom_type_data[atom_begin + center];
                            const std::int32_t neighbor_type = atom_type_data[atom_begin + neighbor];
                            const bool actual = same_type
                                ? center_type == first && neighbor_type == first
                                : ((center_type == first && neighbor_type == second)
                                   || (center_type == second && neighbor_type == first));
                            if (!actual) {
                                continue;
                            }
                            ++contacts;
                            if (center_type == first) {
                                first_exposed[static_cast<std::size_t>(center)] = 1;
                            }
                            if (!same_type && center_type == second) {
                                second_exposed[static_cast<std::size_t>(center)] = 1;
                            }
                            if (detail_data[scope * type_pair_count + type_pair]) {
                                normalized[static_cast<std::size_t>(frame)].codes.push_back(
                                    static_cast<std::int32_t>(scope * type_pair_count + type_pair));
                                normalized[static_cast<std::size_t>(frame)].values.push_back(
                                    distance_data[edge] / cutoff);
                            }
                        }
                        metric_data[metric_base + 1] = nonself_opportunities + self_opportunities;
                        const double denominator = frame_atom_count < 2
                            ? 1.0
                            : static_cast<double>(frame_atom_count) * (frame_atom_count - 1);
                        const double probability = frame_atom_count < 2
                            ? 0.0
                            : same_type
                                ? static_cast<double>(first_count) * (first_count - 1) / denominator
                                : 2.0 * first_count * second_count / denominator;
                        metric_data[metric_base + 2] = nonself_opportunities * probability
                            + (same_type ? self_opportunities * first_count / static_cast<double>(frame_atom_count) : 0.0);
                        metric_data[metric_base + 3] = contacts;
                        metric_data[metric_base + 6] = std::count(
                            first_exposed.begin(), first_exposed.end(), std::uint8_t(1));
                        metric_data[metric_base + 7] = std::count(
                            second_exposed.begin(), second_exposed.end(), std::uint8_t(1));
                    }
                }
            }
        }
    }

    std::size_t normalized_count = 0;
    for (const auto& frame : normalized) {
        normalized_count += frame.codes.size();
    }
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
    return py::make_tuple(metrics, normalized_codes, normalized_values);
}

py::tuple typed_neighbor_counts(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> atom_offsets,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_types,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pair_offsets,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> centers,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> neighbors,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<double, py::array::c_style | py::array::forcecast> cutoff_matrices
) {
    const auto atom_offset_info = atom_offsets.request();
    const auto atom_type_info = atom_types.request();
    const auto pair_offset_info = pair_offsets.request();
    const auto center_info = centers.request();
    const auto neighbor_info = neighbors.request();
    const auto distance_info = distances.request();
    const auto cutoff_info = cutoff_matrices.request();
    if (atom_offset_info.ndim != 1 || atom_offset_info.shape[0] < 1 ||
        pair_offset_info.ndim != 1 || pair_offset_info.shape[0] != atom_offset_info.shape[0] ||
        atom_type_info.ndim != 1 || center_info.ndim != 1 || neighbor_info.ndim != 1 ||
        distance_info.ndim != 1 || center_info.shape[0] != neighbor_info.shape[0] ||
        center_info.shape[0] != distance_info.shape[0] || cutoff_info.ndim != 3 ||
        cutoff_info.shape[1] != cutoff_info.shape[2]) {
        throw std::invalid_argument("typed neighbor count inputs have incompatible shapes");
    }
    const py::ssize_t frame_count = atom_offset_info.shape[0] - 1;
    const py::ssize_t total_atoms = atom_type_info.shape[0];
    const py::ssize_t scope_count = cutoff_info.shape[0];
    const py::ssize_t type_count = cutoff_info.shape[1];
    const auto* atom_offset_data = static_cast<const std::int64_t*>(atom_offset_info.ptr);
    const auto* atom_type_data = static_cast<const std::int32_t*>(atom_type_info.ptr);
    const auto* pair_offset_data = static_cast<const std::int64_t*>(pair_offset_info.ptr);
    const auto* center_data = static_cast<const std::int32_t*>(center_info.ptr);
    const auto* neighbor_data = static_cast<const std::int32_t*>(neighbor_info.ptr);
    const auto* distance_data = static_cast<const double*>(distance_info.ptr);
    const auto* cutoff_data = static_cast<const double*>(cutoff_info.ptr);
    if (atom_offset_data[0] != 0 || atom_offset_data[frame_count] != total_atoms ||
        pair_offset_data[0] != 0 || pair_offset_data[frame_count] != center_info.shape[0]) {
        throw std::invalid_argument("offset arrays do not match typed neighbor values");
    }

    py::array_t<std::int32_t> counts({scope_count, total_atoms});
    py::array_t<std::int32_t> type_counts({scope_count, total_atoms, type_count});
    auto* count_data = counts.mutable_data();
    auto* type_count_data = type_counts.mutable_data();
    std::fill(count_data, count_data + counts.size(), std::int32_t(0));
    std::fill(type_count_data, type_count_data + type_counts.size(), std::int32_t(0));
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
            const std::int64_t atom_begin = atom_offset_data[frame];
            for (std::int64_t edge = pair_offset_data[frame]; edge < pair_offset_data[frame + 1]; ++edge) {
                const std::int64_t center = atom_begin + center_data[edge];
                const std::int64_t neighbor = atom_begin + neighbor_data[edge];
                const std::int32_t center_type = atom_type_data[center];
                const std::int32_t neighbor_type = atom_type_data[neighbor];
                for (py::ssize_t scope = 0; scope < scope_count; ++scope) {
                    const double cutoff = cutoff_data[
                        (scope * type_count + center_type) * type_count + neighbor_type];
                    if (distance_data[edge] >= cutoff) {
                        continue;
                    }
                    ++count_data[scope * total_atoms + center];
                    ++type_count_data[(scope * total_atoms + center) * type_count + neighbor_type];
                }
            }
        }
    }
    return py::make_tuple(counts, type_counts);
}

PYBIND11_MODULE(_fastaudit, module) {
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
        "typed_contact_summary",
        &typed_contact_summary,
        py::arg("atom_offsets"),
        py::arg("atom_types"),
        py::arg("pair_offsets"),
        py::arg("centers"),
        py::arg("neighbors"),
        py::arg("distances"),
        py::arg("cutoff_matrices"),
        py::arg("detail_mask"),
        "Aggregate typed contact metrics for a neighbor batch without applying audit policy."
    );
    module.def(
        "typed_neighbor_counts",
        &typed_neighbor_counts,
        py::arg("atom_offsets"),
        py::arg("atom_types"),
        py::arg("pair_offsets"),
        py::arg("centers"),
        py::arg("neighbors"),
        py::arg("distances"),
        py::arg("cutoff_matrices"),
        "Count typed neighbors per atom and scope without applying histogram policy."
    );
}
