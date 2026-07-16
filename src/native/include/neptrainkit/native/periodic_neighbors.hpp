#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace neptrainkit {
namespace native {

template <typename Scalar>
struct Neighbor {
    Scalar x;
    Scalar y;
    Scalar z;
    Scalar distance;
    std::int32_t source;
};

template <typename Scalar>
struct RadiusNeighbors {
    std::vector<std::int32_t> centers;
    std::vector<std::int32_t> sources;
    std::vector<Scalar> distances;
};

template <typename Scalar>
inline bool neighbor_less(const Neighbor<Scalar>& left, const Neighbor<Scalar>& right) {
    if (left.distance != right.distance) return left.distance < right.distance;
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    if (left.z != right.z) return left.z < right.z;
    return left.source < right.source;
}

template <typename Scalar>
class PeriodicNeighborSearch {
public:
    using NeighborType = Neighbor<Scalar>;
    using NeighborRows = std::vector<std::vector<NeighborType>>;
    using RadiusResult = RadiusNeighbors<Scalar>;

    PeriodicNeighborSearch(
        const Scalar* positions,
        std::int64_t atom_count,
        const Scalar* cell,
        const bool* pbc
    )
        : atom_count_(atom_count), periodic_(pbc[0] || pbc[1] || pbc[2]) {
        if (atom_count < 0) throw std::invalid_argument("atom_count must be non-negative");
        for (int value = 0; value < 9; ++value) cell_[value] = cell[value];
        for (int axis = 0; axis < 3; ++axis) pbc_[axis] = pbc[axis];

        const Scalar det = determinant(cell_.data());
        if (periodic_) {
            const Scalar tolerance = std::is_same<Scalar, float>::value
                ? static_cast<Scalar>(1.0e-8)
                : static_cast<Scalar>(1.0e-12);
            if (std::abs(det) <= tolerance) {
                throw std::invalid_argument("periodic cell must be safely invertible");
            }
            inverse3x3(cell_.data(), inverse_.data(), det);
        } else {
            inverse_.fill(static_cast<Scalar>(0));
        }

        wrapped_.resize(static_cast<std::size_t>(atom_count_));
        for (std::int64_t atom = 0; atom < atom_count_; ++atom) {
            const Scalar* position = positions + 3 * atom;
            if (!periodic_) {
                wrapped_[static_cast<std::size_t>(atom)] = {
                    position[0], position[1], position[2]
                };
                continue;
            }
            Scalar fractional[3] = {
                position[0] * inverse_[0] + position[1] * inverse_[3] + position[2] * inverse_[6],
                position[0] * inverse_[1] + position[1] * inverse_[4] + position[2] * inverse_[7],
                position[0] * inverse_[2] + position[1] * inverse_[5] + position[2] * inverse_[8],
            };
            for (int axis = 0; axis < 3; ++axis) {
                if (pbc_[axis]) fractional[axis] -= std::floor(fractional[axis]);
            }
            wrapped_[static_cast<std::size_t>(atom)] = {
                fractional[0] * cell_[0] + fractional[1] * cell_[3] + fractional[2] * cell_[6],
                fractional[0] * cell_[1] + fractional[1] * cell_[4] + fractional[2] * cell_[7],
                fractional[0] * cell_[2] + fractional[1] * cell_[5] + fractional[2] * cell_[8],
            };
        }
    }

    NeighborRows query_knn(int requested_neighbors) const {
        if (requested_neighbors <= 0) {
            throw std::invalid_argument("requested_neighbors must be positive");
        }
        NeighborRows selected(static_cast<std::size_t>(atom_count_));
        if (atom_count_ == 0) return selected;

        Scalar cutoff = initial_knn_cutoff(requested_neighbors);
        for (int attempt = 0; attempt < 8; ++attempt) {
            const SpatialIndex index(this, cutoff);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t center = 0; center < atom_count_; ++center) {
                std::vector<NeighborType> candidates;
                candidates.reserve(static_cast<std::size_t>(requested_neighbors) * 3);
                index.visit(center, [&](
                    const ImagePoint& point,
                    Scalar dx,
                    Scalar dy,
                    Scalar dz,
                    Scalar squared
                ) {
                    candidates.push_back({
                        dx, dy, dz, std::sqrt(squared), point.source
                    });
                    return true;
                });
                const std::size_t keep = std::min<std::size_t>(
                    static_cast<std::size_t>(requested_neighbors), candidates.size()
                );
                std::partial_sort(
                    candidates.begin(), candidates.begin() + keep, candidates.end(),
                    neighbor_less<Scalar>
                );
                candidates.resize(keep);
                selected[static_cast<std::size_t>(center)] = std::move(candidates);
            }

            bool complete = !periodic_;
            if (periodic_) {
                complete = true;
                for (const auto& neighbors : selected) {
                    if (neighbors.size() < static_cast<std::size_t>(requested_neighbors) ||
                        neighbors.back().distance >= cutoff * static_cast<Scalar>(0.9)) {
                        complete = false;
                        break;
                    }
                }
            }
            if (complete) return selected;
            cutoff *= static_cast<Scalar>(1.7);
        }
        throw std::runtime_error("failed to recover the requested periodic neighbors");
    }

    RadiusResult query_radius(Scalar cutoff) const {
        if (!(cutoff > static_cast<Scalar>(0)) || !std::isfinite(cutoff)) {
            throw std::invalid_argument("cutoff must be finite and positive");
        }
        RadiusResult selected;
        if (atom_count_ == 0) return selected;
        if (atom_count_ <= kDirectScanAtomLimit) return query_radius_direct(cutoff);
        const SpatialIndex index(this, cutoff);
        const Scalar cutoff_squared = cutoff * cutoff;
        NeighborRows rows(static_cast<std::size_t>(atom_count_));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t center = 0; center < atom_count_; ++center) {
            std::vector<NeighborType> neighbors;
            index.visit(center, [&](
                const ImagePoint& point,
                Scalar dx,
                Scalar dy,
                Scalar dz,
                Scalar squared
            ) {
                if (squared < cutoff_squared) {
                    neighbors.push_back({
                        dx, dy, dz, std::sqrt(squared), point.source
                    });
                }
                return true;
            });
            rows[static_cast<std::size_t>(center)] = std::move(neighbors);
        }
        std::size_t pair_count = 0;
        for (const auto& row : rows) pair_count += row.size();
        selected.centers.reserve(pair_count);
        selected.sources.reserve(pair_count);
        selected.distances.reserve(pair_count);
        for (std::int32_t center = 0; center < atom_count_; ++center) {
            for (const auto& neighbor : rows[static_cast<std::size_t>(center)]) {
                selected.centers.push_back(center);
                selected.sources.push_back(neighbor.source);
                selected.distances.push_back(neighbor.distance);
            }
        }
        return selected;
    }

    bool any_distinct_pair_within(Scalar cutoff) const {
        if (atom_count_ < 2) return false;
        if (!(cutoff >= static_cast<Scalar>(0)) || !std::isfinite(cutoff)) {
            throw std::invalid_argument("cutoff must be finite and non-negative");
        }
        if (cutoff == static_cast<Scalar>(0)) return false;
        if (atom_count_ <= kDirectScanAtomLimit) return any_distinct_pair_direct(cutoff);
        const SpatialIndex index(this, cutoff);
        const Scalar cutoff_squared = cutoff * cutoff;
        for (std::int64_t center = 0; center < atom_count_; ++center) {
            bool found = false;
            index.visit(center, [&](
                const ImagePoint& point,
                Scalar,
                Scalar,
                Scalar,
                Scalar squared
            ) {
                if (point.source != center && squared <= cutoff_squared) {
                    found = true;
                    return false;
                }
                return true;
            });
            if (found) return true;
        }
        return false;
    }

private:
    // Measured on Audit's 32-atom frame workload: below this point the setup
    // cost of a cell-list is larger than the pair scan it replaces.
    static constexpr std::int64_t kDirectScanAtomLimit = 96;

    struct Translation {
        Scalar x;
        Scalar y;
        Scalar z;
        bool central;
    };

    struct ImagePoint {
        Scalar x;
        Scalar y;
        Scalar z;
        std::int32_t source;
        bool central_image;
    };

    class SpatialIndex {
    public:
        SpatialIndex(const PeriodicNeighborSearch* owner, Scalar cutoff)
            : owner_(owner), cutoff_(cutoff), bin_width_(std::max(
                  cutoff, std::numeric_limits<Scalar>::epsilon()
              )) {
            int image_counts[3] = {0, 0, 0};
            if (owner_->periodic_) {
                for (int axis = 0; axis < 3; ++axis) {
                    if (!owner_->pbc_[axis]) continue;
                    const Scalar reciprocal_norm = std::sqrt(
                        owner_->inverse_[axis] * owner_->inverse_[axis] +
                        owner_->inverse_[3 + axis] * owner_->inverse_[3 + axis] +
                        owner_->inverse_[6 + axis] * owner_->inverse_[6 + axis]
                    );
                    // Wrapped fractional differences are strictly below one.
                    image_counts[axis] = static_cast<int>(
                        std::ceil(cutoff_ * reciprocal_norm)
                    );
                }
            }

            const std::size_t image_total = static_cast<std::size_t>(
                2 * image_counts[0] + 1
            ) * static_cast<std::size_t>(
                2 * image_counts[1] + 1
            ) * static_cast<std::size_t>(
                2 * image_counts[2] + 1
            );
            points_.reserve(static_cast<std::size_t>(owner_->atom_count_) * image_total);
            lower_.fill(std::numeric_limits<Scalar>::max());
            upper_.fill(std::numeric_limits<Scalar>::lowest());
            for (int sx = -image_counts[0]; sx <= image_counts[0]; ++sx) {
                for (int sy = -image_counts[1]; sy <= image_counts[1]; ++sy) {
                    for (int sz = -image_counts[2]; sz <= image_counts[2]; ++sz) {
                        const Scalar shift_x = sx * owner_->cell_[0] +
                                               sy * owner_->cell_[3] +
                                               sz * owner_->cell_[6];
                        const Scalar shift_y = sx * owner_->cell_[1] +
                                               sy * owner_->cell_[4] +
                                               sz * owner_->cell_[7];
                        const Scalar shift_z = sx * owner_->cell_[2] +
                                               sy * owner_->cell_[5] +
                                               sz * owner_->cell_[8];
                        const bool central_image = sx == 0 && sy == 0 && sz == 0;
                        for (std::int64_t source = 0; source < owner_->atom_count_; ++source) {
                            const auto& wrapped = owner_->wrapped_[static_cast<std::size_t>(source)];
                            const ImagePoint point{
                                wrapped[0] + shift_x,
                                wrapped[1] + shift_y,
                                wrapped[2] + shift_z,
                                static_cast<std::int32_t>(source),
                                central_image,
                            };
                            points_.push_back(point);
                            lower_[0] = std::min(lower_[0], point.x);
                            lower_[1] = std::min(lower_[1], point.y);
                            lower_[2] = std::min(lower_[2], point.z);
                            upper_[0] = std::max(upper_[0], point.x);
                            upper_[1] = std::max(upper_[1], point.y);
                            upper_[2] = std::max(upper_[2], point.z);
                        }
                    }
                }
            }

            bins_[0] = bin_count(0);
            bins_[1] = bin_count(1);
            bins_[2] = bin_count(2);
            const std::size_t total_bins = static_cast<std::size_t>(bins_[0]) *
                                           bins_[1] * bins_[2];
            heads_.assign(total_bins, -1);
            next_.assign(points_.size(), -1);
            for (std::size_t point_index = 0; point_index < points_.size(); ++point_index) {
                const ImagePoint& point = points_[point_index];
                const int x = coordinate(point.x, 0);
                const int y = coordinate(point.y, 1);
                const int z = coordinate(point.z, 2);
                const std::size_t bin = flat_bin(x, y, z);
                next_[point_index] = heads_[bin];
                heads_[bin] = static_cast<std::int32_t>(point_index);
            }
        }

        template <typename Visitor>
        bool visit(std::int64_t center, Visitor visitor) const {
            const auto& origin = owner_->wrapped_[static_cast<std::size_t>(center)];
            const int center_x = coordinate(origin[0], 0);
            const int center_y = coordinate(origin[1], 1);
            const int center_z = coordinate(origin[2], 2);
            const Scalar squared_limit = cutoff_ * cutoff_ *
                (static_cast<Scalar>(1) + static_cast<Scalar>(1.0e-6));
            for (int x = std::max(0, center_x - 1);
                 x <= std::min(bins_[0] - 1, center_x + 1); ++x) {
                for (int y = std::max(0, center_y - 1);
                     y <= std::min(bins_[1] - 1, center_y + 1); ++y) {
                    for (int z = std::max(0, center_z - 1);
                         z <= std::min(bins_[2] - 1, center_z + 1); ++z) {
                        std::int32_t point_index = heads_[flat_bin(x, y, z)];
                        while (point_index >= 0) {
                            const ImagePoint& point = points_[static_cast<std::size_t>(point_index)];
                            if (!(point.source == center && point.central_image)) {
                                const Scalar dx = point.x - origin[0];
                                const Scalar dy = point.y - origin[1];
                                const Scalar dz = point.z - origin[2];
                                const Scalar squared = dx * dx + dy * dy + dz * dz;
                                if (squared <= squared_limit &&
                                    !visitor(point, dx, dy, dz, squared)) {
                                    return false;
                                }
                            }
                            point_index = next_[static_cast<std::size_t>(point_index)];
                        }
                    }
                }
            }
            return true;
        }

    private:
        int bin_count(int axis) const {
            return std::max(
                1,
                static_cast<int>(std::floor(
                    (upper_[axis] - lower_[axis]) / bin_width_
                )) + 1
            );
        }

        int coordinate(Scalar value, int axis) const {
            return std::max(
                0,
                std::min(
                    bins_[axis] - 1,
                    static_cast<int>(std::floor((value - lower_[axis]) / bin_width_))
                )
            );
        }

        std::size_t flat_bin(int x, int y, int z) const {
            return (static_cast<std::size_t>(x) * bins_[1] + y) * bins_[2] + z;
        }

        const PeriodicNeighborSearch* owner_;
        Scalar cutoff_;
        Scalar bin_width_;
        std::vector<ImagePoint> points_;
        std::array<Scalar, 3> lower_;
        std::array<Scalar, 3> upper_;
        int bins_[3] = {1, 1, 1};
        std::vector<std::int32_t> heads_;
        std::vector<std::int32_t> next_;
    };

    static Scalar determinant(const Scalar* cell) {
        return cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
               cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
               cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
    }

    static void inverse3x3(const Scalar* cell, Scalar* inverse, Scalar det) {
        const Scalar factor = static_cast<Scalar>(1) / det;
        inverse[0] = (cell[4] * cell[8] - cell[5] * cell[7]) * factor;
        inverse[1] = (cell[2] * cell[7] - cell[1] * cell[8]) * factor;
        inverse[2] = (cell[1] * cell[5] - cell[2] * cell[4]) * factor;
        inverse[3] = (cell[5] * cell[6] - cell[3] * cell[8]) * factor;
        inverse[4] = (cell[0] * cell[8] - cell[2] * cell[6]) * factor;
        inverse[5] = (cell[2] * cell[3] - cell[0] * cell[5]) * factor;
        inverse[6] = (cell[3] * cell[7] - cell[4] * cell[6]) * factor;
        inverse[7] = (cell[1] * cell[6] - cell[0] * cell[7]) * factor;
        inverse[8] = (cell[0] * cell[4] - cell[1] * cell[3]) * factor;
    }

    std::vector<Translation> image_translations(Scalar cutoff) const {
        int image_counts[3] = {0, 0, 0};
        if (periodic_) {
            for (int axis = 0; axis < 3; ++axis) {
                if (!pbc_[axis]) continue;
                const Scalar reciprocal_norm = std::sqrt(
                    inverse_[axis] * inverse_[axis] +
                    inverse_[3 + axis] * inverse_[3 + axis] +
                    inverse_[6 + axis] * inverse_[6 + axis]
                );
                image_counts[axis] = static_cast<int>(
                    std::ceil(cutoff * reciprocal_norm)
                );
            }
        }
        std::vector<Translation> translations;
        const std::size_t count = static_cast<std::size_t>(2 * image_counts[0] + 1) *
                                  static_cast<std::size_t>(2 * image_counts[1] + 1) *
                                  static_cast<std::size_t>(2 * image_counts[2] + 1);
        translations.reserve(count);
        for (int sx = -image_counts[0]; sx <= image_counts[0]; ++sx) {
            for (int sy = -image_counts[1]; sy <= image_counts[1]; ++sy) {
                for (int sz = -image_counts[2]; sz <= image_counts[2]; ++sz) {
                    translations.push_back({
                        sx * cell_[0] + sy * cell_[3] + sz * cell_[6],
                        sx * cell_[1] + sy * cell_[4] + sz * cell_[7],
                        sx * cell_[2] + sy * cell_[5] + sz * cell_[8],
                        sx == 0 && sy == 0 && sz == 0,
                    });
                }
            }
        }
        return translations;
    }

    RadiusResult query_radius_direct(Scalar cutoff) const {
        RadiusResult selected;
        const auto translations = image_translations(cutoff);
        const Scalar cutoff_squared = cutoff * cutoff;
        selected.centers.reserve(static_cast<std::size_t>(atom_count_) * 32);
        selected.sources.reserve(static_cast<std::size_t>(atom_count_) * 32);
        selected.distances.reserve(static_cast<std::size_t>(atom_count_) * 32);
        for (std::int64_t center = 0; center < atom_count_; ++center) {
            const auto& first = wrapped_[static_cast<std::size_t>(center)];
            for (std::int64_t source = 0; source < atom_count_; ++source) {
                const auto& second = wrapped_[static_cast<std::size_t>(source)];
                for (const auto& shift : translations) {
                    if (source == center && shift.central) continue;
                    const Scalar dx = second[0] + shift.x - first[0];
                    const Scalar dy = second[1] + shift.y - first[1];
                    const Scalar dz = second[2] + shift.z - first[2];
                    const Scalar squared = dx * dx + dy * dy + dz * dz;
                    if (squared < cutoff_squared) {
                        selected.centers.push_back(static_cast<std::int32_t>(center));
                        selected.sources.push_back(static_cast<std::int32_t>(source));
                        selected.distances.push_back(std::sqrt(squared));
                    }
                }
            }
        }
        return selected;
    }

    bool any_distinct_pair_direct(Scalar cutoff) const {
        const auto translations = image_translations(cutoff);
        const Scalar cutoff_squared = cutoff * cutoff;
        for (std::int64_t first_index = 0; first_index < atom_count_; ++first_index) {
            const auto& first = wrapped_[static_cast<std::size_t>(first_index)];
            for (std::int64_t second_index = first_index + 1;
                 second_index < atom_count_; ++second_index) {
                const auto& second = wrapped_[static_cast<std::size_t>(second_index)];
                const Scalar base_x = second[0] - first[0];
                const Scalar base_y = second[1] - first[1];
                const Scalar base_z = second[2] - first[2];
                for (const auto& shift : translations) {
                    const Scalar dx = base_x + shift.x;
                    const Scalar dy = base_y + shift.y;
                    const Scalar dz = base_z + shift.z;
                    if (dx * dx + dy * dy + dz * dz <= cutoff_squared) return true;
                }
            }
        }
        return false;
    }

    Scalar initial_knn_cutoff(int requested_neighbors) const {
        if (periodic_) {
            const Scalar volume = std::abs(determinant(cell_.data()));
            const Scalar density = static_cast<Scalar>(atom_count_) / volume;
            const Scalar pi = static_cast<Scalar>(3.14159265358979323846);
            const Scalar radius = std::cbrt(
                static_cast<Scalar>(3) * static_cast<Scalar>(requested_neighbors + 2) /
                (static_cast<Scalar>(4) * pi * density)
            );
            Scalar minimum_cell_length = std::numeric_limits<Scalar>::max();
            for (int row = 0; row < 3; ++row) {
                const Scalar length = std::sqrt(
                    cell_[3 * row] * cell_[3 * row] +
                    cell_[3 * row + 1] * cell_[3 * row + 1] +
                    cell_[3 * row + 2] * cell_[3 * row + 2]
                );
                minimum_cell_length = std::min(minimum_cell_length, length);
            }
            return std::max(
                radius * static_cast<Scalar>(1.8),
                minimum_cell_length * static_cast<Scalar>(0.55)
            );
        }

        std::array<Scalar, 3> lower{{
            std::numeric_limits<Scalar>::max(),
            std::numeric_limits<Scalar>::max(),
            std::numeric_limits<Scalar>::max(),
        }};
        std::array<Scalar, 3> upper{{
            std::numeric_limits<Scalar>::lowest(),
            std::numeric_limits<Scalar>::lowest(),
            std::numeric_limits<Scalar>::lowest(),
        }};
        for (const auto& position : wrapped_) {
            for (int axis = 0; axis < 3; ++axis) {
                lower[axis] = std::min(lower[axis], position[axis]);
                upper[axis] = std::max(upper[axis], position[axis]);
            }
        }
        const Scalar x = upper[0] - lower[0];
        const Scalar y = upper[1] - lower[1];
        const Scalar z = upper[2] - lower[2];
        return std::max(
            std::sqrt(x * x + y * y + z * z), static_cast<Scalar>(1)
        );
    }

    std::int64_t atom_count_;
    bool periodic_;
    bool pbc_[3] = {false, false, false};
    std::array<Scalar, 9> cell_;
    std::array<Scalar, 9> inverse_;
    std::vector<std::array<Scalar, 3>> wrapped_;
};

}  // namespace native
}  // namespace neptrainkit
