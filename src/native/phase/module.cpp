#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "neptrainkit/native/periodic_neighbors.hpp"

namespace py = pybind11;

namespace {

constexpr float kEps = 1.0e-12f;

using Neighbor = neptrainkit::native::Neighbor<float>;

inline bool neighbor_less(const Neighbor& left, const Neighbor& right) {
    return neptrainkit::native::neighbor_less(left, right);
}

py::tuple periodic_knn_vectors(
    py::array_t<float, py::array::c_style | py::array::forcecast> positions_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> cell_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> pbc_array,
    const int requested_neighbors
) {
    if (positions_array.ndim() != 2 || positions_array.shape(1) != 3 ||
        cell_array.ndim() != 2 || cell_array.shape(0) != 3 || cell_array.shape(1) != 3 ||
        pbc_array.ndim() != 1 || pbc_array.shape(0) != 3) {
        throw py::value_error("positions, cell, and pbc must have shapes (N,3), (3,3), and (3,)");
    }
    if (requested_neighbors <= 0) {
        throw py::value_error("neighbors must be positive");
    }

    const py::ssize_t atom_count = positions_array.shape(0);
    const py::ssize_t neighbor_count = requested_neighbors;
    py::array_t<float> vectors({atom_count, neighbor_count, py::ssize_t{3}});
    py::array_t<std::int32_t> indices({atom_count, neighbor_count});
    py::array_t<bool> valid({atom_count, neighbor_count});
    std::fill(
        vectors.mutable_data(),
        vectors.mutable_data() + atom_count * requested_neighbors * 3,
        0.0f
    );
    std::fill(
        indices.mutable_data(),
        indices.mutable_data() + atom_count * requested_neighbors,
        -1
    );
    std::fill(
        valid.mutable_data(),
        valid.mutable_data() + atom_count * requested_neighbors,
        false
    );
    if (atom_count == 0) return py::make_tuple(vectors, indices, valid);

    neptrainkit::native::PeriodicNeighborSearch<float>::NeighborRows selected;
    {
        py::gil_scoped_release release;
        const neptrainkit::native::PeriodicNeighborSearch<float> search(
            positions_array.data(),
            static_cast<std::int64_t>(atom_count),
            cell_array.data(),
            pbc_array.data()
        );
        selected = search.query_knn(requested_neighbors);
    }

    float* vector_output = vectors.mutable_data();
    std::int32_t* index_output = indices.mutable_data();
    bool* valid_output = valid.mutable_data();
    for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
        const auto& neighbors = selected[static_cast<std::size_t>(atom)];
        for (py::ssize_t slot = 0; slot < static_cast<py::ssize_t>(neighbors.size()); ++slot) {
            const py::ssize_t offset = atom * requested_neighbors + slot;
            vector_output[offset * 3] = neighbors[static_cast<std::size_t>(slot)].x;
            vector_output[offset * 3 + 1] = neighbors[static_cast<std::size_t>(slot)].y;
            vector_output[offset * 3 + 2] = neighbors[static_cast<std::size_t>(slot)].z;
            index_output[offset] = neighbors[static_cast<std::size_t>(slot)].source;
            valid_output[offset] = true;
        }
    }
    return py::make_tuple(std::move(vectors), std::move(indices), std::move(valid));
}

py::tuple translational_order_evidence(
    py::array_t<float, py::array::c_style | py::array::forcecast> positions_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> cell_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> pbc_array
) {
    if (positions_array.ndim() != 2 || positions_array.shape(1) != 3 ||
        cell_array.ndim() != 2 || cell_array.shape(0) != 3 || cell_array.shape(1) != 3 ||
        pbc_array.ndim() != 1 || pbc_array.shape(0) != 3) {
        throw py::value_error("positions, cell, and pbc must have shapes (N,3), (3,3), and (3,)");
    }
    const py::ssize_t atom_count = positions_array.shape(0);
    if (atom_count == 0) {
        throw py::value_error("translational order requires at least one atom");
    }
    if (!pbc_array.data()[0] || !pbc_array.data()[1] || !pbc_array.data()[2]) {
        const double unavailable = std::numeric_limits<double>::quiet_NaN();
        return py::make_tuple(unavailable, unavailable);
    }

    const float* cell = cell_array.data();
    const float determinant =
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
        cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
        cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
    if (std::abs(determinant) <= kEps) {
        throw py::value_error("periodic cell must be invertible");
    }
    const float inverse_determinant = 1.0f / determinant;
    const std::array<float, 9> inverse{{
        (cell[4] * cell[8] - cell[5] * cell[7]) * inverse_determinant,
        (cell[2] * cell[7] - cell[1] * cell[8]) * inverse_determinant,
        (cell[1] * cell[5] - cell[2] * cell[4]) * inverse_determinant,
        (cell[5] * cell[6] - cell[3] * cell[8]) * inverse_determinant,
        (cell[0] * cell[8] - cell[2] * cell[6]) * inverse_determinant,
        (cell[2] * cell[3] - cell[0] * cell[5]) * inverse_determinant,
        (cell[3] * cell[7] - cell[4] * cell[6]) * inverse_determinant,
        (cell[1] * cell[6] - cell[0] * cell[7]) * inverse_determinant,
        (cell[0] * cell[4] - cell[1] * cell[3]) * inverse_determinant,
    }};
    std::vector<std::array<float, 3>> fractional(static_cast<std::size_t>(atom_count));
    const float* positions = positions_array.data();
    for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            float value =
                positions[atom * 3] * inverse[axis] +
                positions[atom * 3 + 1] * inverse[3 + axis] +
                positions[atom * 3 + 2] * inverse[6 + axis];
            value -= std::floor(value);
            fractional[static_cast<std::size_t>(atom)][axis] = value;
        }
    }

    static constexpr std::array<std::array<int, 3>, 13> directions{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}},
        {{1, 1, 0}}, {{1, -1, 0}}, {{1, 0, 1}}, {{1, 0, -1}},
        {{0, 1, 1}}, {{0, 1, -1}},
        {{1, 1, 1}}, {{1, 1, -1}}, {{1, -1, 1}}, {{-1, 1, 1}},
    }};
    const int maximum_harmonic = std::min(
        64,
        static_cast<int>(std::ceil(4.0 * std::cbrt(static_cast<double>(atom_count))))
    );
    const int wave_count = static_cast<int>(directions.size()) * maximum_harmonic;
    std::vector<double> intensities(static_cast<std::size_t>(wave_count), 0.0);
    constexpr double two_pi = 6.283185307179586476925286766559;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int wave = 0; wave < wave_count; ++wave) {
            const auto& direction = directions[static_cast<std::size_t>(wave / maximum_harmonic)];
            const int harmonic = wave % maximum_harmonic + 1;
            double real = 0.0;
            double imaginary = 0.0;
            for (const auto& value : fractional) {
                const double phase = two_pi * harmonic * (
                    direction[0] * value[0] +
                    direction[1] * value[1] +
                    direction[2] * value[2]
                );
                real += std::cos(phase);
                imaginary += std::sin(phase);
            }
            const double denominator = static_cast<double>(atom_count) * atom_count;
            intensities[static_cast<std::size_t>(wave)] =
                (real * real + imaginary * imaginary) / denominator;
        }
    }
    const double score = *std::max_element(intensities.begin(), intensities.end());
    const double random_limit = std::min(
        1.0,
        std::log(static_cast<double>(wave_count) / 0.01) / atom_count
    );
    return py::make_tuple(score, random_limit);
}

inline float legendre(const int order, const float value) {
    if (order == 0) return 1.0f;
    if (order == 1) return value;
    float previous = 1.0f;
    float current = value;
    for (int degree = 2; degree <= order; ++degree) {
        const float next = ((2.0f * degree - 1.0f) * value * current -
                            (degree - 1.0f) * previous) /
                           degree;
        previous = current;
        current = next;
    }
    return current;
}

inline float dot_unit(const Neighbor& left, const Neighbor& right) {
    const float denominator = std::max(left.distance * right.distance, kEps);
    const float value = (left.x * right.x + left.y * right.y + left.z * right.z) /
                        denominator;
    return std::max(-1.0f, std::min(1.0f, value));
}

void append_bond_order(
    const std::vector<Neighbor>& neighbors,
    std::vector<float>& output
) {
    static constexpr std::array<int, 7> counts{{4, 6, 8, 12, 14, 18, 24}};
    static constexpr std::array<int, 6> orders{{2, 4, 6, 8, 10, 12}};
    for (const int requested : counts) {
        const int count = std::min(requested, static_cast<int>(neighbors.size()));
        for (const int order : orders) {
            if (count == 0) {
                output.push_back(0.0f);
                continue;
            }
            float sum = 0.0f;
            for (int left = 0; left < count; ++left) {
                for (int right = 0; right < count; ++right) {
                    sum += legendre(order, dot_unit(neighbors[left], neighbors[right]));
                }
            }
            output.push_back(std::sqrt(std::max(sum / (count * count), 0.0f)));
        }
    }
}

void append_topology(
    const std::vector<Neighbor>& neighbors,
    const float scale,
    std::vector<float>& output
) {
    static constexpr std::array<float, 5> thresholds{{1.08f, 1.22f, 1.42f, 1.75f, 2.10f}};
    constexpr float edge_threshold = 1.24f;
    for (const float threshold : thresholds) {
        int count = 0;
        while (count < static_cast<int>(neighbors.size()) &&
               neighbors[count].distance / scale <= threshold) {
            ++count;
        }
        if (count < 2) {
            output.insert(output.end(), {count / 24.0f, 0.0f, 0.0f, 0.0f, 0.0f});
            continue;
        }
        std::vector<unsigned char> adjacency(static_cast<std::size_t>(count * count), 0);
        std::vector<int> degrees(static_cast<std::size_t>(count), 0);
        int edges = 0;
        for (int left = 0; left < count; ++left) {
            for (int right = left + 1; right < count; ++right) {
                const float dx = (neighbors[left].x - neighbors[right].x) / scale;
                const float dy = (neighbors[left].y - neighbors[right].y) / scale;
                const float dz = (neighbors[left].z - neighbors[right].z) / scale;
                if (std::sqrt(dx * dx + dy * dy + dz * dz) <= edge_threshold) {
                    adjacency[left * count + right] = 1;
                    adjacency[right * count + left] = 1;
                    ++degrees[left];
                    ++degrees[right];
                    ++edges;
                }
            }
        }
        int triangles = 0;
        for (int first = 0; first < count; ++first) {
            for (int second = first + 1; second < count; ++second) {
                if (!adjacency[first * count + second]) continue;
                for (int third = second + 1; third < count; ++third) {
                    triangles += adjacency[first * count + third] &&
                                 adjacency[second * count + third];
                }
            }
        }
        float degree_sum = 0.0f;
        for (const int degree : degrees) degree_sum += degree;
        const float degree_mean = degree_sum / count;
        float degree_variance = 0.0f;
        for (const int degree : degrees) {
            const float delta = degree - degree_mean;
            degree_variance += delta * delta;
        }
        degree_variance /= count;
        const float possible_edges = std::max(count * (count - 1) / 2.0f, 1.0f);
        const float possible_triangles =
            std::max(count * (count - 1) * (count - 2) / 6.0f, 1.0f);
        output.insert(
            output.end(),
            {
                count / 24.0f,
                edges / possible_edges,
                triangles / possible_triangles,
                degree_mean / std::max(count - 1, 1),
                std::sqrt(degree_variance) / std::max(count - 1, 1),
            }
        );
    }
}

void append_cosine_histogram(
    const std::vector<Neighbor>& neighbors,
    std::vector<float>& output
) {
    std::array<float, 8> histogram{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    const int count = std::min(12, static_cast<int>(neighbors.size()));
    int samples = 0;
    for (int left = 0; left < count; ++left) {
        for (int right = left + 1; right < count; ++right) {
            const float cosine = dot_unit(neighbors[left], neighbors[right]);
            const int bin = std::max(0, std::min(7, static_cast<int>((cosine + 1.0f) * 4.0f)));
            histogram[bin] += 1.0f;
            ++samples;
        }
    }
    for (const float value : histogram) {
        output.push_back(samples ? value / samples : 0.0f);
    }
}

void append_chemistry(
    const std::vector<Neighbor>& neighbors,
    const std::int32_t center_type,
    const std::int32_t* atom_types,
    const float scale,
    std::vector<float>& output
) {
    static constexpr std::array<float, 3> centers{{0.95f, 1.18f, 1.42f}};
    static constexpr std::array<float, 2> widths{{0.08f, 0.16f}};
    for (const float center : centers) {
        for (const float width : widths) {
            std::unordered_map<std::int32_t, float> weights_by_species;
            float total = 0.0f;
            float same = 0.0f;
            for (const Neighbor& neighbor : neighbors) {
                const float distance = neighbor.distance / scale;
                const float cutoff = 1.0f / (1.0f + std::exp((distance - 1.55f) / 0.04f));
                const float delta = (distance - center) / width;
                const float weight = std::exp(-0.5f * delta * delta) * cutoff;
                total += weight;
                weights_by_species[atom_types[neighbor.source]] += weight;
                if (atom_types[neighbor.source] == center_type) same += weight;
            }
            if (total <= kEps) {
                output.insert(output.end(), 7, 0.0f);
                continue;
            }
            std::vector<float> fractions;
            fractions.reserve(weights_by_species.size());
            float pair_equal = 0.0f;
            float entropy = 0.0f;
            for (const auto& item : weights_by_species) {
                const float fraction = item.second / total;
                fractions.push_back(fraction);
                pair_equal += fraction * fraction;
                entropy -= fraction * std::log(fraction + kEps) / std::log(4.0f);
            }
            std::sort(fractions.begin(), fractions.end(), std::greater<float>());
            while (fractions.size() < 3) fractions.push_back(0.0f);
            const float effective_species =
                std::min(1.0f / std::max(pair_equal, kEps), 4.0f) / 4.0f;
            output.insert(
                output.end(),
                {
                    same / total,
                    effective_species,
                    fractions[0],
                    fractions[1],
                    fractions[2],
                    pair_equal,
                    entropy,
                }
            );
        }
    }
}

py::tuple phase_features(
    py::array_t<float, py::array::c_style | py::array::forcecast> vectors_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> indices_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> valid_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types_array
) {
    if (vectors_array.ndim() != 3 || vectors_array.shape(2) != 3 ||
        indices_array.ndim() != 2 || valid_array.ndim() != 2 || types_array.ndim() != 1 ||
        vectors_array.shape(0) != indices_array.shape(0) ||
        vectors_array.shape(0) != valid_array.shape(0) ||
        vectors_array.shape(0) != types_array.shape(0) ||
        vectors_array.shape(1) != indices_array.shape(1) ||
        vectors_array.shape(1) != valid_array.shape(1)) {
        throw py::value_error("invalid phase feature input shapes");
    }
    const py::ssize_t atom_count = vectors_array.shape(0);
    const py::ssize_t neighbor_count = vectors_array.shape(1);
    const py::ssize_t geometry_width = 2 * neighbor_count - 1 + 42 + 25 + 8;
    constexpr py::ssize_t chemistry_width = 42;
    py::array_t<float> geometry({atom_count, geometry_width});
    py::array_t<float> chemistry({atom_count, chemistry_width});

    const float* vectors = vectors_array.data();
    const std::int32_t* indices = indices_array.data();
    const bool* valid = valid_array.data();
    const std::int32_t* types = types_array.data();
    float* geometry_output = geometry.mutable_data();
    float* chemistry_output = chemistry.mutable_data();

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
            std::vector<Neighbor> neighbors;
            neighbors.reserve(static_cast<std::size_t>(neighbor_count));
            for (py::ssize_t slot = 0; slot < neighbor_count; ++slot) {
                const py::ssize_t offset = atom * neighbor_count + slot;
                if (!valid[offset]) continue;
                const float x = vectors[offset * 3];
                const float y = vectors[offset * 3 + 1];
                const float z = vectors[offset * 3 + 2];
                neighbors.push_back({x, y, z, std::sqrt(x * x + y * y + z * z), indices[offset]});
            }
            std::stable_sort(
                neighbors.begin(), neighbors.end(),
                neighbor_less
            );
            if (neighbors.empty()) continue;
            const int scale_count = std::min(6, static_cast<int>(neighbors.size()));
            float scale = 0.0f;
            for (int index = 0; index < scale_count; ++index) scale += neighbors[index].distance;
            scale /= scale_count;

            std::vector<float> geometry_row;
            geometry_row.reserve(static_cast<std::size_t>(geometry_width));
            for (py::ssize_t slot = 0; slot < neighbor_count; ++slot) {
                geometry_row.push_back(
                    slot < static_cast<py::ssize_t>(neighbors.size())
                        ? std::min(neighbors[slot].distance / scale, 3.0f)
                        : 3.0f
                );
            }
            for (py::ssize_t slot = 0; slot + 1 < neighbor_count; ++slot) {
                if (slot + 1 < static_cast<py::ssize_t>(neighbors.size())) {
                    geometry_row.push_back(std::log(
                        std::max(neighbors[slot + 1].distance / scale, kEps) /
                        std::max(neighbors[slot].distance / scale, kEps)
                    ));
                } else {
                    geometry_row.push_back(0.0f);
                }
            }
            append_bond_order(neighbors, geometry_row);
            append_topology(neighbors, scale, geometry_row);
            append_cosine_histogram(neighbors, geometry_row);
            std::copy(
                geometry_row.begin(), geometry_row.end(),
                geometry_output + atom * geometry_width
            );

            std::vector<float> chemistry_row;
            chemistry_row.reserve(chemistry_width);
            append_chemistry(neighbors, types[atom], types, scale, chemistry_row);
            std::copy(
                chemistry_row.begin(), chemistry_row.end(),
                chemistry_output + atom * chemistry_width
            );
        }
    }
    return py::make_tuple(std::move(geometry), std::move(chemistry));
}

inline bool type_in_group(
    const std::int32_t value,
    const std::int32_t* group,
    const py::ssize_t group_size
) {
    for (py::ssize_t index = 0; index < group_size; ++index) {
        if (group[index] == value) return true;
    }
    return false;
}

void validate_refinement_inputs(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& vectors,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& indices,
    const py::array_t<bool, py::array::c_style | py::array::forcecast>& valid,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& types,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& a_types
) {
    if (vectors.ndim() != 3 || vectors.shape(2) != 3 ||
        indices.ndim() != 2 || valid.ndim() != 2 || types.ndim() != 1 ||
        a_types.ndim() != 1 ||
        vectors.shape(0) != indices.shape(0) ||
        vectors.shape(0) != valid.shape(0) ||
        vectors.shape(0) != types.shape(0) ||
        vectors.shape(1) != indices.shape(1) ||
        vectors.shape(1) != valid.shape(1)) {
        throw py::value_error("invalid phase refinement input shapes");
    }
    if (vectors.shape(0) == 0 || a_types.shape(0) == 0) {
        throw py::value_error("phase refinement requires atoms and a non-empty A group");
    }
    const std::int32_t* sources = indices.data();
    const bool* mask = valid.data();
    const py::ssize_t value_count = indices.shape(0) * indices.shape(1);
    for (py::ssize_t offset = 0; offset < value_count; ++offset) {
        if (mask[offset] && (sources[offset] < 0 || sources[offset] >= vectors.shape(0))) {
            throw py::value_error("valid neighbor index is outside atom_types");
        }
    }
}

std::vector<Neighbor> sorted_neighbors(
    const float* vectors,
    const std::int32_t* indices,
    const bool* valid,
    const py::ssize_t atom,
    const py::ssize_t neighbor_count
) {
    std::vector<Neighbor> neighbors;
    neighbors.reserve(static_cast<std::size_t>(neighbor_count));
    for (py::ssize_t slot = 0; slot < neighbor_count; ++slot) {
        const py::ssize_t offset = atom * neighbor_count + slot;
        if (!valid[offset]) continue;
        const float x = vectors[offset * 3];
        const float y = vectors[offset * 3 + 1];
        const float z = vectors[offset * 3 + 2];
        neighbors.push_back({
            x, y, z, std::sqrt(x * x + y * y + z * z), indices[offset]
        });
    }
    std::stable_sort(
        neighbors.begin(), neighbors.end(),
        neighbor_less
    );
    return neighbors;
}

int bit_count(unsigned int value) {
    int count = 0;
    while (value) {
        value &= value - 1;
        ++count;
    }
    return count;
}

std::int8_t classify_close_packed_cna(const std::vector<Neighbor>& neighbors) {
    if (neighbors.size() < 13 ||
        neighbors[11].distance <= kEps ||
        neighbors[12].distance / neighbors[11].distance < 1.08f) {
        return 0;
    }
    const float cutoff = 0.5f * (
        neighbors[11].distance + neighbors[12].distance
    );
    const float cutoff_squared = cutoff * cutoff;
    std::array<unsigned int, 12> adjacency{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    for (int left = 0; left < 12; ++left) {
        for (int right = left + 1; right < 12; ++right) {
            const float dx = neighbors[left].x - neighbors[right].x;
            const float dy = neighbors[left].y - neighbors[right].y;
            const float dz = neighbors[left].z - neighbors[right].z;
            if (dx * dx + dy * dy + dz * dz < cutoff_squared) {
                adjacency[left] |= 1u << right;
                adjacency[right] |= 1u << left;
            }
        }
    }
    int signature_421 = 0;
    int signature_422 = 0;
    for (int bonded = 0; bonded < 12; ++bonded) {
        const unsigned int common = adjacency[bonded];
        if (bit_count(common) != 4) return 0;
        int degree_sum = 0;
        int maximum_degree = 0;
        for (int vertex = 0; vertex < 12; ++vertex) {
            if (!(common & (1u << vertex))) continue;
            const int degree = bit_count(adjacency[vertex] & common);
            degree_sum += degree;
            maximum_degree = std::max(maximum_degree, degree);
        }
        const int bond_count = degree_sum / 2;
        if (bond_count != 2) return 0;
        if (maximum_degree == 1) {
            ++signature_421;
        } else if (maximum_degree == 2) {
            ++signature_422;
        } else {
            return 0;
        }
    }
    if (signature_421 == 12) return 1;
    if (signature_421 == 6 && signature_422 == 6) return 2;
    return 0;
}

std::int8_t classify_adaptive_cna(const std::vector<Neighbor>& neighbors) {
    const std::int8_t close_packed = classify_close_packed_cna(neighbors);
    if (close_packed) return close_packed;
    if (neighbors.size() < 15 ||
        neighbors[13].distance <= kEps ||
        neighbors[14].distance / neighbors[13].distance < 1.08f) {
        return 0;
    }
    const float cutoff = 0.5f * (
        neighbors[13].distance + neighbors[14].distance
    );
    const float cutoff_squared = cutoff * cutoff;
    std::array<unsigned int, 14> adjacency{};
    for (int left = 0; left < 14; ++left) {
        for (int right = left + 1; right < 14; ++right) {
            const float dx = neighbors[left].x - neighbors[right].x;
            const float dy = neighbors[left].y - neighbors[right].y;
            const float dz = neighbors[left].z - neighbors[right].z;
            if (dx * dx + dy * dy + dz * dz < cutoff_squared) {
                adjacency[left] |= 1u << right;
                adjacency[right] |= 1u << left;
            }
        }
    }
    int signature_444 = 0;
    int signature_666 = 0;
    for (int bonded = 0; bonded < 14; ++bonded) {
        const unsigned int common = adjacency[bonded];
        const int common_count = bit_count(common);
        if (common_count != 4 && common_count != 6) return 0;
        int degree_sum = 0;
        for (int vertex = 0; vertex < 14; ++vertex) {
            if (!(common & (1u << vertex))) continue;
            const int degree = bit_count(adjacency[vertex] & common);
            if (degree != 2) return 0;
            degree_sum += degree;
        }
        if (degree_sum / 2 != common_count) return 0;
        if (common_count == 4) {
            ++signature_444;
        } else {
            ++signature_666;
        }
    }
    return signature_444 == 6 && signature_666 == 8 ? 3 : 0;
}

py::array_t<std::int8_t> adaptive_cna_labels(
    py::array_t<float, py::array::c_style | py::array::forcecast> vectors_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> indices_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> valid_array
) {
    if (vectors_array.ndim() != 3 || vectors_array.shape(2) != 3 ||
        indices_array.ndim() != 2 || valid_array.ndim() != 2 ||
        vectors_array.shape(0) != indices_array.shape(0) ||
        vectors_array.shape(0) != valid_array.shape(0) ||
        vectors_array.shape(1) != indices_array.shape(1) ||
        vectors_array.shape(1) != valid_array.shape(1)) {
        throw py::value_error("invalid adaptive CNA input shapes");
    }
    const py::ssize_t atom_count = vectors_array.shape(0);
    const py::ssize_t neighbor_count = vectors_array.shape(1);
    py::array_t<std::int8_t> labels(atom_count);
    std::fill(labels.mutable_data(), labels.mutable_data() + atom_count, 0);
    if (neighbor_count < 13) return labels;

    const float* vectors = vectors_array.data();
    const std::int32_t* indices = indices_array.data();
    const bool* valid = valid_array.data();
    std::int8_t* output = labels.mutable_data();
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(atom_count >= 256)
#endif
        for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
            const std::vector<Neighbor> neighbors = sorted_neighbors(
                vectors, indices, valid, atom, neighbor_count
            );
            output[atom] = classify_adaptive_cna(neighbors);
        }
    }
    return labels;
}

py::tuple phase_partition_primitives(
    py::array_t<float, py::array::c_style | py::array::forcecast> positions_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> cell_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> pbc_array,
    const int requested_neighbors
) {
    if (requested_neighbors < 24) {
        throw py::value_error(
            "phase partition primitives require at least 24 neighbors"
        );
    }
    py::tuple neighbors = periodic_knn_vectors(
        positions_array,
        cell_array,
        pbc_array,
        requested_neighbors
    );
    auto vectors = neighbors[0].cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>
    >();
    auto indices = neighbors[1].cast<
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
    >();
    auto valid = neighbors[2].cast<
        py::array_t<bool, py::array::c_style | py::array::forcecast>
    >();
    py::array_t<std::int8_t> labels = adaptive_cna_labels(
        vectors,
        indices,
        valid
    );
    return py::make_tuple(
        std::move(vectors),
        std::move(indices),
        std::move(valid),
        std::move(labels)
    );
}

bool matches_shape_template(
    const std::vector<Neighbor>& neighbors,
    const int coordination,
    const float* templates,
    const py::ssize_t template_count,
    const py::ssize_t template_width,
    const float threshold
) {
    const py::ssize_t expected_width = coordination + coordination * (coordination - 1) / 2;
    if (neighbors.size() < static_cast<std::size_t>(coordination) ||
        template_count == 0 || template_width != expected_width ||
        neighbors.front().distance <= kEps) {
        return false;
    }
    float scale = 0.0f;
    for (int index = 0; index < coordination; ++index) {
        scale += neighbors[index].distance;
    }
    scale /= coordination;
    if (scale <= kEps) return false;

    std::vector<float> descriptor;
    descriptor.reserve(static_cast<std::size_t>(expected_width));
    for (int index = 0; index < coordination; ++index) {
        descriptor.push_back(neighbors[index].distance / scale);
    }
    std::sort(descriptor.begin(), descriptor.end());
    std::vector<float> pairwise;
    pairwise.reserve(static_cast<std::size_t>(coordination * (coordination - 1) / 2));
    for (int left = 0; left < coordination; ++left) {
        for (int right = left + 1; right < coordination; ++right) {
            const float dx = neighbors[left].x - neighbors[right].x;
            const float dy = neighbors[left].y - neighbors[right].y;
            const float dz = neighbors[left].z - neighbors[right].z;
            pairwise.push_back(std::sqrt(dx * dx + dy * dy + dz * dz) / scale);
        }
    }
    std::sort(pairwise.begin(), pairwise.end());
    descriptor.insert(descriptor.end(), pairwise.begin(), pairwise.end());

    const float squared_limit = threshold * threshold * expected_width;
    for (py::ssize_t row = 0; row < template_count; ++row) {
        const float* current = templates + row * template_width;
        float squared_error = 0.0f;
        for (py::ssize_t column = 0; column < template_width; ++column) {
            const float delta = descriptor[column] - current[column];
            squared_error += delta * delta;
            if (squared_error > squared_limit) break;
        }
        if (squared_error <= squared_limit) return true;
    }
    return false;
}

double common_prototype_shape_rms(
    const float* vectors,
    const py::ssize_t neighbor_capacity,
    const py::ssize_t atom,
    const int coordination,
    const double* reference
) {
    std::vector<float> radial(static_cast<std::size_t>(coordination));
    std::vector<std::array<float, 3>> normalized(
        static_cast<std::size_t>(coordination)
    );
    float scale = 0.0f;
    for (int slot = 0; slot < coordination; ++slot) {
        const py::ssize_t offset = (atom * neighbor_capacity + slot) * 3;
        const float x = vectors[offset];
        const float y = vectors[offset + 1];
        const float z = vectors[offset + 2];
        const float distance = std::sqrt(x * x + y * y + z * z);
        radial[static_cast<std::size_t>(slot)] = distance;
        scale += distance;
    }
    scale /= coordination;
    if (scale <= kEps) return std::numeric_limits<double>::infinity();

    for (int slot = 0; slot < coordination; ++slot) {
        const py::ssize_t offset = (atom * neighbor_capacity + slot) * 3;
        normalized[static_cast<std::size_t>(slot)] = {{
            vectors[offset] / scale,
            vectors[offset + 1] / scale,
            vectors[offset + 2] / scale,
        }};
        radial[static_cast<std::size_t>(slot)] /= scale;
    }
    std::sort(radial.begin(), radial.end());
    std::vector<float> pairwise;
    pairwise.reserve(
        static_cast<std::size_t>(coordination * (coordination - 1) / 2)
    );
    for (int left = 0; left < coordination; ++left) {
        const auto& first = normalized[static_cast<std::size_t>(left)];
        for (int right = left + 1; right < coordination; ++right) {
            const auto& second = normalized[static_cast<std::size_t>(right)];
            const float dx = first[0] - second[0];
            const float dy = first[1] - second[1];
            const float dz = first[2] - second[2];
            pairwise.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    std::sort(pairwise.begin(), pairwise.end());

    double squared_error = 0.0;
    for (int slot = 0; slot < coordination; ++slot) {
        const double delta = radial[static_cast<std::size_t>(slot)] - reference[slot];
        squared_error += delta * delta;
    }
    for (py::ssize_t slot = 0; slot < static_cast<py::ssize_t>(pairwise.size()); ++slot) {
        const double delta = pairwise[static_cast<std::size_t>(slot)]
            - reference[coordination + slot];
        squared_error += delta * delta;
    }
    const py::ssize_t width = coordination
        + coordination * (coordination - 1) / 2;
    return std::sqrt(squared_error / static_cast<double>(width));
}

py::tuple common_prototype_mapping_metrics(
    py::array_t<float, py::array::c_style | py::array::forcecast> vectors_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> indices_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> valid_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> mapped_roles_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> template_roles_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> neighbor_counts_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> shell_sizes_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> shell_role_counts_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> descriptors_array,
    const double shape_threshold,
    const double maximum_shell_error_fraction
) {
    if (vectors_array.ndim() != 3 || vectors_array.shape(2) != 3 ||
        indices_array.ndim() != 2 || valid_array.ndim() != 2 ||
        mapped_roles_array.ndim() != 1 || template_roles_array.ndim() != 1 ||
        neighbor_counts_array.ndim() != 1 || shell_sizes_array.ndim() != 2 ||
        shell_role_counts_array.ndim() != 3 || descriptors_array.ndim() != 2 ||
        vectors_array.shape(0) != indices_array.shape(0) ||
        vectors_array.shape(0) != valid_array.shape(0) ||
        vectors_array.shape(0) != mapped_roles_array.shape(0) ||
        vectors_array.shape(1) != indices_array.shape(1) ||
        vectors_array.shape(1) != valid_array.shape(1)) {
        throw py::value_error("invalid common-prototype mapping input shapes");
    }
    const py::ssize_t atom_count = vectors_array.shape(0);
    const py::ssize_t neighbor_capacity = vectors_array.shape(1);
    const py::ssize_t template_count = template_roles_array.shape(0);
    const py::ssize_t shell_capacity = shell_sizes_array.shape(1);
    const py::ssize_t role_count = shell_role_counts_array.shape(2);
    if (atom_count <= 0 || template_count <= 0 || role_count <= 0 ||
        neighbor_counts_array.shape(0) != template_count ||
        shell_sizes_array.shape(0) != template_count ||
        shell_role_counts_array.shape(0) != template_count ||
        shell_role_counts_array.shape(1) != shell_capacity ||
        descriptors_array.shape(0) != template_count ||
        shape_threshold < 0.0 || maximum_shell_error_fraction < 0.0) {
        throw py::value_error("invalid common-prototype template metadata");
    }

    const std::int32_t* mapped_roles = mapped_roles_array.data();
    const std::int32_t* template_roles = template_roles_array.data();
    const std::int32_t* neighbor_counts = neighbor_counts_array.data();
    const std::int32_t* shell_sizes = shell_sizes_array.data();
    const std::int32_t* shell_role_counts = shell_role_counts_array.data();
    const std::int32_t* indices = indices_array.data();
    const bool* valid = valid_array.data();
    for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
        if (mapped_roles[atom] < 0 || mapped_roles[atom] >= role_count) {
            throw py::value_error("mapped atom roles are outside the template role range");
        }
        for (py::ssize_t slot = 0; slot < neighbor_capacity; ++slot) {
            const py::ssize_t offset = atom * neighbor_capacity + slot;
            if (valid[offset] && (indices[offset] < 0 || indices[offset] >= atom_count)) {
                throw py::value_error("common-prototype neighbor index is out of range");
            }
        }
    }
    for (py::ssize_t row = 0; row < template_count; ++row) {
        const int coordination = neighbor_counts[row];
        const py::ssize_t descriptor_width = coordination
            + coordination * (coordination - 1) / 2;
        if (template_roles[row] < 0 || template_roles[row] >= role_count ||
            coordination <= 0 || coordination > neighbor_capacity ||
            descriptor_width > descriptors_array.shape(1)) {
            throw py::value_error("common-prototype template dimensions are inconsistent");
        }
        int shell_total = 0;
        for (py::ssize_t shell = 0; shell < shell_capacity; ++shell) {
            const int size = shell_sizes[row * shell_capacity + shell];
            if (size < 0) {
                throw py::value_error("common-prototype shell size must be non-negative");
            }
            shell_total += size;
        }
        if (shell_total != coordination) {
            throw py::value_error("common-prototype shells do not cover the template");
        }
    }

    const float* vectors = vectors_array.data();
    const double* descriptors = descriptors_array.data();
    const py::ssize_t descriptor_capacity = descriptors_array.shape(1);
    std::int64_t geometry_count = 0;
    std::int64_t chemistry_count = 0;
    std::int64_t joint_count = 0;
    std::int64_t finite_rms_count = 0;
    double finite_rms_sum = 0.0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
            int best_score = -1;
            bool best_geometry = false;
            bool best_chemistry = false;
            double best_rms = std::numeric_limits<double>::infinity();
            for (py::ssize_t row = 0; row < template_count; ++row) {
                if (template_roles[row] != mapped_roles[atom]) continue;
                const int coordination = neighbor_counts[row];
                bool eligible = true;
                for (int slot = 0; slot < coordination; ++slot) {
                    if (!valid[atom * neighbor_capacity + slot]) {
                        eligible = false;
                        break;
                    }
                }
                bool chemistry = false;
                bool geometry = false;
                double rms = std::numeric_limits<double>::infinity();
                if (eligible) {
                    int shell_errors = 0;
                    int start = 0;
                    for (py::ssize_t shell = 0; shell < shell_capacity; ++shell) {
                        const int size = shell_sizes[row * shell_capacity + shell];
                        if (size == 0) continue;
                        std::vector<int> observed(static_cast<std::size_t>(role_count), 0);
                        for (int slot = start; slot < start + size; ++slot) {
                            const std::int32_t source = indices[
                                atom * neighbor_capacity + slot
                            ];
                            ++observed[static_cast<std::size_t>(mapped_roles[source])];
                        }
                        int current_error = 0;
                        for (py::ssize_t role = 0; role < role_count; ++role) {
                            const py::ssize_t expected_offset =
                                (row * shell_capacity + shell) * role_count + role;
                            current_error += std::abs(
                                observed[static_cast<std::size_t>(role)]
                                - shell_role_counts[expected_offset]
                            );
                        }
                        shell_errors += current_error / 2;
                        start += size;
                    }
                    const int allowed_errors = std::max(
                        1,
                        static_cast<int>(std::floor(
                            coordination * maximum_shell_error_fraction
                        ))
                    );
                    chemistry = shell_errors <= allowed_errors;
                    if (chemistry) {
                        rms = common_prototype_shape_rms(
                            vectors,
                            neighbor_capacity,
                            atom,
                            coordination,
                            descriptors + row * descriptor_capacity
                        );
                        geometry = rms <= shape_threshold;
                    }
                }
                const bool joint = geometry && chemistry;
                const int score = 4 * static_cast<int>(joint)
                    + 2 * static_cast<int>(chemistry)
                    + static_cast<int>(geometry);
                if (score > best_score || (score == best_score && rms < best_rms)) {
                    best_score = score;
                    best_geometry = geometry;
                    best_chemistry = chemistry;
                    best_rms = rms;
                }
            }
            geometry_count += best_geometry;
            chemistry_count += best_chemistry;
            joint_count += best_geometry && best_chemistry;
            if (std::isfinite(best_rms)) {
                finite_rms_sum += best_rms;
                ++finite_rms_count;
            }
        }
    }
    const double denominator = static_cast<double>(atom_count);
    const py::object mean_rms = finite_rms_count
        ? py::cast(finite_rms_sum / static_cast<double>(finite_rms_count))
        : py::none();
    return py::make_tuple(
        geometry_count / denominator,
        chemistry_count / denominator,
        joint_count / denominator,
        mean_rms
    );
}

float minimum_pairing_cost(
    const std::array<Neighbor, 6>& vectors,
    const std::array<int, 6>& remaining,
    const int count
) {
    if (count == 0) return 0.0f;
    const int first = remaining[0];
    float best = std::numeric_limits<float>::max();
    for (int partner_slot = 1; partner_slot < count; ++partner_slot) {
        const int partner = remaining[partner_slot];
        std::array<int, 6> next{{0, 0, 0, 0, 0, 0}};
        int next_count = 0;
        for (int slot = 1; slot < count; ++slot) {
            if (slot != partner_slot) next[next_count++] = remaining[slot];
        }
        const float x = vectors[first].x + vectors[partner].x;
        const float y = vectors[first].y + vectors[partner].y;
        const float z = vectors[first].z + vectors[partner].z;
        best = std::min(
            best,
            x * x + y * y + z * z + minimum_pairing_cost(vectors, next, next_count)
        );
    }
    return best;
}

float normalized_csp(const std::array<Neighbor, 6>& vectors) {
    float mean_squared_bond = 0.0f;
    for (const Neighbor& vector : vectors) {
        mean_squared_bond += vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
    }
    mean_squared_bond /= vectors.size();
    const std::array<int, 6> all{{0, 1, 2, 3, 4, 5}};
    return minimum_pairing_cost(vectors, all, 6) / std::max(mean_squared_bond, kEps);
}

py::tuple l12_refinement_metrics(
    py::array_t<float, py::array::c_style | py::array::forcecast> vectors_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> indices_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> valid_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> a_types_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> templates_array,
    const float shape_threshold
) {
    validate_refinement_inputs(vectors_array, indices_array, valid_array, types_array, a_types_array);
    constexpr int coordination = 12;
    constexpr py::ssize_t template_width = coordination + coordination * (coordination - 1) / 2;
    if (templates_array.ndim() != 2 || templates_array.shape(1) != template_width) {
        throw py::value_error("L1_2 templates must have shape (M, 78)");
    }

    const py::ssize_t atom_count = vectors_array.shape(0);
    const py::ssize_t neighbor_count = vectors_array.shape(1);
    const float* vectors = vectors_array.data();
    const std::int32_t* indices = indices_array.data();
    const bool* valid = valid_array.data();
    const std::int32_t* types = types_array.data();
    const std::int32_t* a_types = a_types_array.data();
    const float* templates = templates_array.data();
    const py::ssize_t a_type_count = a_types_array.shape(0);
    const py::ssize_t template_count = templates_array.shape(0);
    std::int64_t geometry_count = 0;
    std::int64_t chemistry_count = 0;
    std::int64_t joint_count = 0;

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:geometry_count,chemistry_count,joint_count)
#endif
        for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
            const std::vector<Neighbor> neighbors = sorted_neighbors(
                vectors, indices, valid, atom, neighbor_count
            );
            if (neighbors.size() < 18) continue;
            const bool geometry = matches_shape_template(
                neighbors, coordination, templates, template_count, template_width, shape_threshold
            );
            int first_shell_a = 0;
            int second_shell_a = 0;
            for (int slot = 0; slot < 12; ++slot) {
                first_shell_a += type_in_group(
                    types[neighbors[slot].source], a_types, a_type_count
                );
            }
            for (int slot = 12; slot < 18; ++slot) {
                second_shell_a += type_in_group(
                    types[neighbors[slot].source], a_types, a_type_count
                );
            }
            const bool center_is_a = type_in_group(types[atom], a_types, a_type_count);
            const bool chemistry = center_is_a
                ? first_shell_a == 0 && second_shell_a == 6
                : first_shell_a == 4 && second_shell_a == 0;
            geometry_count += geometry;
            chemistry_count += chemistry;
            joint_count += geometry && chemistry;
        }
    }
    const double denominator = static_cast<double>(atom_count);
    return py::make_tuple(
        geometry_count / denominator,
        chemistry_count / denominator,
        joint_count / denominator
    );
}

py::tuple laves_refinement_metrics(
    py::array_t<float, py::array::c_style | py::array::forcecast> vectors_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> indices_array,
    py::array_t<bool, py::array::c_style | py::array::forcecast> valid_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> types_array,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> a_types_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> z12_templates_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> z16_templates_array,
    const float shape_threshold,
    const float csp_threshold
) {
    validate_refinement_inputs(vectors_array, indices_array, valid_array, types_array, a_types_array);
    constexpr py::ssize_t z12_width = 12 + 12 * 11 / 2;
    constexpr py::ssize_t z16_width = 16 + 16 * 15 / 2;
    if (z12_templates_array.ndim() != 2 || z12_templates_array.shape(1) != z12_width ||
        z16_templates_array.ndim() != 2 || z16_templates_array.shape(1) != z16_width) {
        throw py::value_error("Laves templates must have shapes (M, 78) and (M, 136)");
    }

    const py::ssize_t atom_count = vectors_array.shape(0);
    const py::ssize_t neighbor_count = vectors_array.shape(1);
    const float* vectors = vectors_array.data();
    const std::int32_t* indices = indices_array.data();
    const bool* valid = valid_array.data();
    const std::int32_t* types = types_array.data();
    const std::int32_t* a_types = a_types_array.data();
    const float* z12_templates = z12_templates_array.data();
    const float* z16_templates = z16_templates_array.data();
    const py::ssize_t a_type_count = a_types_array.shape(0);
    std::int64_t geometry_count = 0;
    std::int64_t chemistry_count = 0;
    std::int64_t joint_count = 0;
    std::int64_t csp_count = 0;
    std::int64_t b2_count = 0;

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:geometry_count,chemistry_count,joint_count,csp_count,b2_count)
#endif
        for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
            const std::vector<Neighbor> neighbors = sorted_neighbors(
                vectors, indices, valid, atom, neighbor_count
            );
            const bool center_is_a = type_in_group(types[atom], a_types, a_type_count);
            const int coordination = center_is_a ? 16 : 12;
            if (neighbors.size() < static_cast<std::size_t>(coordination)) continue;
            const bool geometry = center_is_a
                ? matches_shape_template(
                    neighbors, 16, z16_templates, z16_templates_array.shape(0),
                    z16_width, shape_threshold
                )
                : matches_shape_template(
                    neighbors, 12, z12_templates, z12_templates_array.shape(0),
                    z12_width, shape_threshold
                );
            int a_neighbors = 0;
            for (int slot = 0; slot < coordination; ++slot) {
                a_neighbors += type_in_group(
                    types[neighbors[slot].source], a_types, a_type_count
                );
            }
            const bool chemistry = a_neighbors == (center_is_a ? 4 : 6);
            geometry_count += geometry;
            chemistry_count += chemistry;
            joint_count += geometry && chemistry;
            if (center_is_a || !geometry || !chemistry) continue;

            std::array<Neighbor, 6> b_vectors;
            int b_count = 0;
            for (const Neighbor& neighbor : neighbors) {
                if (!type_in_group(types[neighbor.source], a_types, a_type_count)) {
                    b_vectors[b_count++] = neighbor;
                    if (b_count == 6) break;
                }
            }
            if (b_count == 6) {
                ++csp_count;
                b2_count += normalized_csp(b_vectors) > csp_threshold;
            }
        }
    }
    const double denominator = static_cast<double>(atom_count);
    const py::object b2_fraction = csp_count
        ? py::cast(static_cast<double>(b2_count) / static_cast<double>(csp_count))
        : py::none();
    return py::make_tuple(
        geometry_count / denominator,
        chemistry_count / denominator,
        joint_count / denominator,
        b2_fraction
    );
}

}  // namespace

PYBIND11_MODULE(_phase, module) {
    module.doc() = "Single-precision OpenMP kernels for experimental phase fingerprints";
    module.def(
        "periodic_knn_vectors",
        &periodic_knn_vectors,
        py::arg("positions"),
        py::arg("cell"),
        py::arg("pbc"),
        py::arg("neighbors") = 24
    );
    module.def(
        "phase_features",
        &phase_features,
        py::arg("vectors"),
        py::arg("indices"),
        py::arg("valid"),
        py::arg("atom_types")
    );
    module.def(
        "translational_order_evidence",
        &translational_order_evidence,
        py::arg("positions"),
        py::arg("cell"),
        py::arg("pbc")
    );
    module.def(
        "adaptive_cna_labels",
        &adaptive_cna_labels,
        py::arg("vectors"),
        py::arg("indices"),
        py::arg("valid")
    );
    module.def(
        "phase_partition_primitives",
        &phase_partition_primitives,
        py::arg("positions"),
        py::arg("cell"),
        py::arg("pbc"),
        py::arg("neighbors") = 32,
        "Build one reusable neighbor field plus adaptive-CNA labels."
    );
    module.def(
        "common_prototype_mapping_metrics",
        &common_prototype_mapping_metrics,
        py::arg("vectors"),
        py::arg("indices"),
        py::arg("valid"),
        py::arg("mapped_roles"),
        py::arg("template_roles"),
        py::arg("neighbor_counts"),
        py::arg("shell_sizes"),
        py::arg("shell_role_counts"),
        py::arg("descriptors"),
        py::arg("shape_threshold"),
        py::arg("maximum_shell_error_fraction"),
        "Evaluate one common-prototype role mapping."
    );
    module.def(
        "l12_refinement_metrics",
        &l12_refinement_metrics,
        py::arg("vectors"),
        py::arg("indices"),
        py::arg("valid"),
        py::arg("atom_types"),
        py::arg("a_types"),
        py::arg("templates"),
        py::arg("shape_threshold")
    );
    module.def(
        "laves_refinement_metrics",
        &laves_refinement_metrics,
        py::arg("vectors"),
        py::arg("indices"),
        py::arg("valid"),
        py::arg("atom_types"),
        py::arg("a_types"),
        py::arg("z12_templates"),
        py::arg("z16_templates"),
        py::arg("shape_threshold"),
        py::arg("csp_threshold")
    );
}
