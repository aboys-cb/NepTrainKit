#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

template <typename Scalar>
using ContiguousArray = py::array_t<
    Scalar,
    py::array::c_style | py::array::forcecast>;

template <typename Scalar>
std::vector<py::ssize_t> farthest_point_sampling_impl(
    const ContiguousArray<Scalar>& points,
    py::ssize_t requested_samples,
    double min_distance,
    const py::object& selected_data_object) {
    const auto point_view = points.template unchecked<2>();
    const py::ssize_t point_count = point_view.shape(0);
    const py::ssize_t dimensions = point_view.shape(1);
    if (point_count == 0 || requested_samples <= 0) {
        return {};
    }
    requested_samples = std::min(requested_samples, point_count);

    const bool has_warm_start = !selected_data_object.is_none();
    ContiguousArray<Scalar> selected_data;
    if (has_warm_start) {
        selected_data = ContiguousArray<Scalar>::ensure(selected_data_object);
        if (!selected_data) {
            throw py::type_error("selected_data must be a numeric array");
        }
        if (selected_data.ndim() != 2) {
            throw py::value_error("selected_data must be two dimensional");
        }
        if (selected_data.shape(1) != dimensions) {
            throw py::value_error(
                "points and selected_data must have the same feature dimension");
        }
    }

    const Scalar threshold = static_cast<Scalar>(
        min_distance < 0.0 ? 0.0 : min_distance);
    const Scalar threshold_squared = threshold * threshold;
    std::vector<Scalar> nearest_squared(
        static_cast<std::size_t>(point_count),
        std::numeric_limits<Scalar>::infinity());
    std::vector<py::ssize_t> sampled_indices;
    sampled_indices.reserve(static_cast<std::size_t>(requested_samples));

    if (has_warm_start) {
        const auto selected_view = selected_data.template unchecked<2>();
        const py::ssize_t selected_count = selected_view.shape(0);
        if (selected_count == 0) {
            throw py::value_error("selected_data must contain at least one row");
        }
        py::gil_scoped_release release;
        for (py::ssize_t point = 0; point < point_count; ++point) {
            Scalar best = std::numeric_limits<Scalar>::infinity();
            for (py::ssize_t center = 0; center < selected_count; ++center) {
                Scalar distance_squared = Scalar{0};
                for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
                    const Scalar difference =
                        point_view(point, feature) - selected_view(center, feature);
                    distance_squared += difference * difference;
                }
                if (distance_squared < best) {
                    best = distance_squared;
                }
            }
            nearest_squared[static_cast<std::size_t>(point)] = best;
        }
    } else {
        sampled_indices.push_back(0);
        py::gil_scoped_release release;
        for (py::ssize_t point = 0; point < point_count; ++point) {
            Scalar distance_squared = Scalar{0};
            for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
                const Scalar difference =
                    point_view(point, feature) - point_view(0, feature);
                distance_squared += difference * difference;
            }
            nearest_squared[static_cast<std::size_t>(point)] = distance_squared;
        }
        nearest_squared[0] = -std::numeric_limits<Scalar>::infinity();
    }

    {
        py::gil_scoped_release release;
        while (static_cast<py::ssize_t>(sampled_indices.size()) < requested_samples) {
            py::ssize_t farthest = 0;
            Scalar farthest_squared = nearest_squared[0];
            for (py::ssize_t point = 1; point < point_count; ++point) {
                const Scalar value = nearest_squared[static_cast<std::size_t>(point)];
                if (value > farthest_squared) {
                    farthest_squared = value;
                    farthest = point;
                }
            }
            if (!std::isfinite(farthest_squared)
                || farthest_squared < threshold_squared) {
                break;
            }

            sampled_indices.push_back(farthest);
            nearest_squared[static_cast<std::size_t>(farthest)] =
                -std::numeric_limits<Scalar>::infinity();
            for (py::ssize_t point = 0; point < point_count; ++point) {
                Scalar distance_squared = Scalar{0};
                for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
                    const Scalar difference =
                        point_view(point, feature) - point_view(farthest, feature);
                    distance_squared += difference * difference;
                }
                Scalar& nearest = nearest_squared[static_cast<std::size_t>(point)];
                if (distance_squared < nearest) {
                    nearest = distance_squared;
                }
            }
            nearest_squared[static_cast<std::size_t>(farthest)] =
                -std::numeric_limits<Scalar>::infinity();
        }
    }
    return sampled_indices;
}

struct CoverageTraceResult {
    std::vector<py::ssize_t> sampled_indices;
    std::vector<double> coverage;
    std::vector<double> radii;
    double initial_coverage = 0.0;
};

template <typename Scalar>
CoverageTraceResult farthest_point_sampling_trace_impl(
    const ContiguousArray<Scalar>& points,
    py::ssize_t requested_samples,
    double target_coverage,
    const py::object& selected_data_object) {
    const auto point_view = points.template unchecked<2>();
    const py::ssize_t point_count = point_view.shape(0);
    const py::ssize_t dimensions = point_view.shape(1);
    CoverageTraceResult result;
    if (point_count == 0 || requested_samples <= 0) {
        return result;
    }
    requested_samples = std::min(requested_samples, point_count);
    if (!(target_coverage > 0.0 && target_coverage <= 1.0)) {
        throw py::value_error("target_coverage must be in (0, 1]");
    }

    const bool has_warm_start = !selected_data_object.is_none();
    ContiguousArray<Scalar> selected_data;
    if (has_warm_start) {
        selected_data = ContiguousArray<Scalar>::ensure(selected_data_object);
        if (!selected_data) {
            throw py::type_error("selected_data must be a numeric array");
        }
        if (selected_data.ndim() != 2) {
            throw py::value_error("selected_data must be two dimensional");
        }
        if (selected_data.shape(1) != dimensions) {
            throw py::value_error(
                "points and selected_data must have the same feature dimension");
        }
        if (selected_data.shape(0) == 0) {
            throw py::value_error("selected_data must contain at least one row");
        }
    }

    std::vector<Scalar> mean(static_cast<std::size_t>(dimensions), Scalar{0});
    for (py::ssize_t point = 0; point < point_count; ++point) {
        for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
            mean[static_cast<std::size_t>(feature)] += point_view(point, feature);
        }
    }
    for (Scalar& value : mean) {
        value /= static_cast<Scalar>(point_count);
    }
    double total_variance = 0.0;
    for (py::ssize_t point = 0; point < point_count; ++point) {
        for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
            const double difference = static_cast<double>(point_view(point, feature))
                - static_cast<double>(mean[static_cast<std::size_t>(feature)]);
            total_variance += difference * difference;
        }
    }

    std::vector<Scalar> nearest_squared(
        static_cast<std::size_t>(point_count),
        std::numeric_limits<Scalar>::infinity());
    std::vector<unsigned char> selected_mask(
        static_cast<std::size_t>(point_count), 0);
    result.sampled_indices.reserve(static_cast<std::size_t>(requested_samples));
    result.coverage.reserve(static_cast<std::size_t>(requested_samples));
    result.radii.reserve(static_cast<std::size_t>(requested_samples));

    auto summarize = [&]() -> std::pair<double, double> {
        double residual = 0.0;
        Scalar farthest_squared = Scalar{0};
        for (py::ssize_t point = 0; point < point_count; ++point) {
            if (selected_mask[static_cast<std::size_t>(point)]) continue;
            const Scalar value = std::max(
                nearest_squared[static_cast<std::size_t>(point)], Scalar{0});
            residual += static_cast<double>(value);
            farthest_squared = std::max(farthest_squared, value);
        }
        const double coverage = total_variance <= std::numeric_limits<double>::epsilon()
            ? 1.0
            : std::max(0.0, std::min(1.0, 1.0 - residual / total_variance));
        return {coverage, std::sqrt(static_cast<double>(farthest_squared))};
    };

    {
        py::gil_scoped_release release;
        if (has_warm_start) {
            const auto selected_view = selected_data.template unchecked<2>();
            const py::ssize_t selected_count = selected_view.shape(0);
            for (py::ssize_t point = 0; point < point_count; ++point) {
                Scalar best = std::numeric_limits<Scalar>::infinity();
                for (py::ssize_t center = 0; center < selected_count; ++center) {
                    Scalar distance_squared = Scalar{0};
                    for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
                        const Scalar difference =
                            point_view(point, feature) - selected_view(center, feature);
                        distance_squared += difference * difference;
                    }
                    best = std::min(best, distance_squared);
                }
                nearest_squared[static_cast<std::size_t>(point)] = best;
            }
            result.initial_coverage = summarize().first;
            if (result.initial_coverage >= target_coverage) {
                return result;
            }
        } else {
            result.sampled_indices.push_back(0);
            selected_mask[0] = 1;
            for (py::ssize_t point = 0; point < point_count; ++point) {
                Scalar distance_squared = Scalar{0};
                for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
                    const Scalar difference =
                        point_view(point, feature) - point_view(0, feature);
                    distance_squared += difference * difference;
                }
                nearest_squared[static_cast<std::size_t>(point)] = distance_squared;
            }
            nearest_squared[0] = Scalar{0};
            const auto summary = summarize();
            result.coverage.push_back(summary.first);
            result.radii.push_back(summary.second);
            if (summary.first >= target_coverage) {
                return result;
            }
        }

        while (static_cast<py::ssize_t>(result.sampled_indices.size())
               < requested_samples) {
            py::ssize_t farthest = -1;
            Scalar farthest_squared = Scalar{-1};
            for (py::ssize_t point = 0; point < point_count; ++point) {
                if (selected_mask[static_cast<std::size_t>(point)]) continue;
                const Scalar value = nearest_squared[static_cast<std::size_t>(point)];
                if (value > farthest_squared) {
                    farthest_squared = value;
                    farthest = point;
                }
            }
            if (farthest < 0 || !std::isfinite(farthest_squared)
                || farthest_squared <= Scalar{0}) {
                break;
            }

            result.sampled_indices.push_back(farthest);
            selected_mask[static_cast<std::size_t>(farthest)] = 1;
            nearest_squared[static_cast<std::size_t>(farthest)] = Scalar{0};
            for (py::ssize_t point = 0; point < point_count; ++point) {
                if (selected_mask[static_cast<std::size_t>(point)]) continue;
                Scalar distance_squared = Scalar{0};
                for (py::ssize_t feature = 0; feature < dimensions; ++feature) {
                    const Scalar difference =
                        point_view(point, feature) - point_view(farthest, feature);
                    distance_squared += difference * difference;
                }
                Scalar& nearest = nearest_squared[static_cast<std::size_t>(point)];
                nearest = std::min(nearest, distance_squared);
            }
            const auto summary = summarize();
            result.coverage.push_back(summary.first);
            result.radii.push_back(summary.second);
            if (summary.first >= target_coverage) {
                break;
            }
        }
    }
    return result;
}

std::vector<py::ssize_t> farthest_point_sampling(
    const py::array& points,
    py::ssize_t requested_samples,
    double min_distance,
    const py::object& selected_data) {
    if (points.ndim() != 2) {
        throw py::value_error("points must be two dimensional");
    }
    const auto selected_array = selected_data.is_none()
        ? py::array()
        : py::array::ensure(selected_data);
    const bool use_float32 = points.dtype().is(py::dtype::of<float>())
        && (selected_data.is_none()
            || (selected_array
                && selected_array.dtype().is(py::dtype::of<float>())));
    if (use_float32) {
        auto typed_points = ContiguousArray<float>::ensure(points);
        if (!typed_points) {
            throw py::type_error("points must be a numeric array");
        }
        return farthest_point_sampling_impl<float>(
            typed_points,
            requested_samples,
            min_distance,
            selected_data);
    }
    auto typed_points = ContiguousArray<double>::ensure(points);
    if (!typed_points) {
        throw py::type_error("points must be a numeric array");
    }
    return farthest_point_sampling_impl<double>(
        typed_points,
        requested_samples,
        min_distance,
        selected_data);
}

py::tuple farthest_point_sampling_trace(
    const py::array& points,
    py::ssize_t requested_samples,
    double target_coverage,
    const py::object& selected_data) {
    if (points.ndim() != 2) {
        throw py::value_error("points must be two dimensional");
    }
    const auto selected_array = selected_data.is_none()
        ? py::array()
        : py::array::ensure(selected_data);
    const bool use_float32 = points.dtype().is(py::dtype::of<float>())
        && (selected_data.is_none()
            || (selected_array
                && selected_array.dtype().is(py::dtype::of<float>())));
    CoverageTraceResult result;
    if (use_float32) {
        auto typed_points = ContiguousArray<float>::ensure(points);
        if (!typed_points) {
            throw py::type_error("points must be a numeric array");
        }
        result = farthest_point_sampling_trace_impl<float>(
            typed_points,
            requested_samples,
            target_coverage,
            selected_data);
    } else {
        auto typed_points = ContiguousArray<double>::ensure(points);
        if (!typed_points) {
            throw py::type_error("points must be a numeric array");
        }
        result = farthest_point_sampling_trace_impl<double>(
            typed_points,
            requested_samples,
            target_coverage,
            selected_data);
    }
    return py::make_tuple(
        result.sampled_indices,
        result.coverage,
        result.radii,
        result.initial_coverage);
}

// Exact structure-view distances and conservative local-environment witnesses.
// Retiring an atom within the finest supported resolution preserves an upper
// bound on its distance. No pairwise distance matrix or group-local metric.
class CoverageState {
    ContiguousArray<double> points_, atoms_;
    ContiguousArray<long long> ends_, elements_, offsets_, species_;
    py::ssize_t count_, dimensions_, blocks_, atom_dimensions_, atom_count_;
    std::vector<double> nearest_, atomic_, local_scores_;
    std::vector<py::ssize_t> atom_order_, owners_, dirty_rows_;
    std::vector<unsigned char> chosen_;
    std::vector<unsigned char> dirty_;
    double atom_resolution_squared = 0.18 * 0.18;
    // A static index over candidate atoms with dynamic covering radii. Bounds
    // use leading orthogonal coordinates; leaf distances still use every axis.
    static constexpr int bound_dimensions = 8;
    struct AtomNode {
        std::array<double, bound_dimensions> low, high;
        py::ssize_t begin, end;
        int left = -1, right = -1;
        double radius = std::numeric_limits<double>::infinity();
    };
    std::vector<AtomNode> atom_nodes_;
    std::vector<std::pair<long long, int>> atom_roots_;

    int build_atom_node(py::ssize_t begin, py::ssize_t end) {
        AtomNode node;
        node.begin=begin; node.end=end;
        node.low.fill(std::numeric_limits<double>::infinity());
        node.high.fill(-std::numeric_limits<double>::infinity());
        const int dims=std::min<py::ssize_t>(bound_dimensions, atom_dimensions_);
        for (auto at=begin; at<end; ++at) {
            const double* point=atoms_.data()+atom_order_[at]*atom_dimensions_;
            for (int col=0; col<dims; ++col) {
                node.low[col]=std::min(node.low[col],point[col]);
                node.high[col]=std::max(node.high[col],point[col]);
            }
        }
        int index=static_cast<int>(atom_nodes_.size());
        atom_nodes_.push_back(node);
        if (end-begin>32 && dims) {
            int axis=0;
            for (int col=1; col<dims; ++col)
                if (node.high[col]-node.low[col]>node.high[axis]-node.low[axis]) axis=col;
            auto middle=begin+(end-begin)/2;
            std::nth_element(atom_order_.begin()+begin,atom_order_.begin()+middle,atom_order_.begin()+end,
                [&](py::ssize_t a,py::ssize_t b) {
                    return atoms_.data()[a*atom_dimensions_+axis]<atoms_.data()[b*atom_dimensions_+axis];
                });
            int left=build_atom_node(begin,middle), right=build_atom_node(middle,end);
            atom_nodes_[index].left=left; atom_nodes_[index].right=right;
        }
        return index;
    }

    void mark_dirty(py::ssize_t row) {
        if (!dirty_[row]) { dirty_[row]=1; dirty_rows_.push_back(row); }
    }

    void update_atom_node(int index, const std::vector<py::ssize_t>& centers) {
        AtomNode& node=atom_nodes_[index];
        if (node.radius<=atom_resolution_squared) return;
        const double* atoms=atoms_.data();
        const int dims=std::min<py::ssize_t>(bound_dimensions,atom_dimensions_);
        bool intersects=false;
        for (auto center:centers) {
            double lower=0;
            for (int col=0; col<dims; ++col) {
                double value=atoms[center*atom_dimensions_+col];
                double delta=std::max(node.low[col]-value,std::max(value-node.high[col],0.0));
                lower+=delta*delta;
            }
            // Roundoff margin keeps this an exact pruning bound.
            if (lower<=node.radius*(1.0+1e-14)) { intersects=true; break; }
        }
        if (!intersects) return;
        if (node.left>=0) {
            update_atom_node(node.left,centers); update_atom_node(node.right,centers);
            node.radius=std::max(atom_nodes_[node.left].radius,atom_nodes_[node.right].radius);
            return;
        }
        const double retirement=atom_resolution_squared;
        double radius=0;
        for (auto at=node.begin; at<node.end; ++at) {
            auto row=atom_order_[at];
            if (chosen_[owners_[row]] || atomic_[row]<=retirement) continue;
            double best=atomic_[row];
            for (auto center:centers) {
                double squared=0;
                for (py::ssize_t col=0; col<atom_dimensions_ && squared<best; ++col) {
                    double delta=atoms[row*atom_dimensions_+col]-atoms[center*atom_dimensions_+col];
                    squared+=delta*delta;
                }
                best=std::min(best,squared);
                if (best<=retirement) break;
            }
            if (best<atomic_[row]) { atomic_[row]=best; mark_dirty(owners_[row]); }
            if (best>retirement) radius=std::max(radius,best);
        }
        node.radius=radius;
    }
public:
    CoverageState(ContiguousArray<double> points, ContiguousArray<long long> ends,
                  ContiguousArray<long long> elements, ContiguousArray<double> atoms,
                  ContiguousArray<long long> offsets, ContiguousArray<long long> species,
                  py::ssize_t count)
        : points_(points), atoms_(atoms), ends_(ends), elements_(elements),
          offsets_(offsets), species_(species), count_(count) {
        if (points.ndim() != 2 || atoms.ndim() != 2 || ends.ndim() != 1 ||
            elements.ndim() != 1 || offsets.ndim() != 1 || species.ndim() != 1 ||
            count < 0 || count > points.shape(0) || elements.size() != points.shape(0) ||
            offsets.size() != points.shape(0) + 1 || species.size() != atoms.shape(0))
            throw py::value_error("coverage arrays are not aligned");
        dimensions_ = points.shape(1); blocks_ = ends.size(); atom_dimensions_ = atoms.shape(1);
        long long previous = 0;
        for (auto end : ends) {
            auto value = py::cast<long long>(end);
            if (value <= previous || value > dimensions_) throw py::value_error("invalid block offsets");
            previous = value;
        }
        if (!blocks_ || previous != dimensions_) throw py::value_error("invalid block dimensions");
        auto off = offsets.unchecked<1>();
        if (off(0) != 0 || off(off.shape(0)-1) != atoms.shape(0))
            throw py::value_error("invalid atom offsets");
        for (py::ssize_t i=1; i<off.shape(0); ++i)
            if (off(i) < off(i-1)) throw py::value_error("invalid atom offsets");
        for (py::ssize_t i=0; i<points.size(); ++i)
            if (!std::isfinite(points.data()[i])) throw py::value_error("features must be finite");
        for (py::ssize_t i=0; i<atoms.size(); ++i)
            if (!std::isfinite(atoms.data()[i])) throw py::value_error("features must be finite");
        atom_count_ = off(count);
        py::gil_scoped_release release;
        nearest_.assign(count*(blocks_+1), std::numeric_limits<double>::infinity());
        atomic_.assign(atom_count_, std::numeric_limits<double>::infinity());
        chosen_.assign(count, 0);
        dirty_.assign(count, 0);
        local_scores_.assign(count, atom_count_ ? std::numeric_limits<double>::infinity() : 0.0);
        owners_.resize(atom_count_);
        for (py::ssize_t row=0; row<count; ++row)
            for (py::ssize_t atom=off(row); atom<off(row+1); ++atom) {
                atom_order_.push_back(atom); owners_[atom]=row;
            }
        const auto* sp=species_.data();
        std::sort(atom_order_.begin(),atom_order_.end(),[&](py::ssize_t a,py::ssize_t b) {
            return sp[a]<sp[b];
        });
        for (py::ssize_t begin=0; begin<atom_count_;) {
            auto end=begin+1;
            while (end<atom_count_ && sp[atom_order_[end]]==sp[atom_order_[begin]]) ++end;
            auto species_id=sp[atom_order_[begin]];
            atom_roots_.emplace_back(species_id,build_atom_node(begin,end));
            begin=end;
        }
    }
    void refine_atoms(double resolution=0.0) {
        if (!std::isfinite(resolution) || resolution<0.0 || resolution*resolution>atom_resolution_squared)
            throw py::value_error("atomic resolution must be finite, nonnegative and no coarser");
        py::gil_scoped_release release;
        atom_resolution_squared = resolution*resolution;
        for (auto& node:atom_nodes_) node.radius=std::numeric_limits<double>::infinity();
        std::fill(local_scores_.begin(), local_scores_.end(), 0.0);
        for (py::ssize_t atom=0; atom<atom_count_; ++atom) {
            if (!chosen_[owners_[atom]] && atomic_[atom] > atom_resolution_squared) {
                local_scores_[owners_[atom]] = std::max(local_scores_[owners_[atom]], atomic_[atom]);
            }
        }
    }
    void add(py::ssize_t center) {
        if (center < 0 || center >= points_.shape(0)) throw py::value_error("invalid center");
        const double* points = points_.data();
        const auto* ends = ends_.data(); const auto* elements = elements_.data();
        const auto* offsets = offsets_.data(); const auto* species = species_.data();
        py::gil_scoped_release release;
        for (py::ssize_t row=0; row<count_; ++row) {
            if (chosen_[row] || elements[row] != elements[center]) continue;
            double joint = 0.0;
            py::ssize_t start = 0;
            for (py::ssize_t block=0; block<blocks_; ++block) {
                double squared = 0.0;
                const double bound = std::max(nearest_[row*(blocks_+1)+block],
                                              nearest_[row*(blocks_+1)+blocks_]);
                for (py::ssize_t col=start; col<ends[block] && squared < bound; ++col) {
                    double delta = points[row*dimensions_+col] - points[center*dimensions_+col];
                    squared += delta*delta;
                }
                double& best = nearest_[row*(blocks_+1)+block];
                best = std::min(best, squared);
                joint = std::max(joint, squared);
                start = ends[block];
            }
            double& best = nearest_[row*(blocks_+1)+blocks_];
            best = std::min(best, joint);
        }
        if (center < count_) {
            chosen_[center] = 1;
            std::fill(nearest_.begin()+center*(blocks_+1), nearest_.begin()+(center+1)*(blocks_+1), 0.0);
            std::fill(atomic_.begin()+offsets[center], atomic_.begin()+offsets[center+1], 0.0);
            mark_dirty(center);
        }
        for (const auto& root:atom_roots_) {
            std::vector<py::ssize_t> centers;
            for (auto atom=offsets[center]; atom<offsets[center+1]; ++atom)
                if (species[atom]==root.first) centers.push_back(atom);
            if (!centers.empty()) update_atom_node(root.second,centers);
        }
        for (auto row:dirty_rows_) {
            double best=0;
            for (auto atom=offsets[row]; atom<offsets[row+1]; ++atom)
                if (atomic_[atom]>atom_resolution_squared) best=std::max(best,atomic_[atom]);
            local_scores_[row]=best; dirty_[row]=0;
        }
        dirty_rows_.clear();
    }
    py::array_t<double> scores() const {
        py::array_t<double> result(count_);
        auto output = result.mutable_unchecked<1>();
        py::gil_scoped_release release;
        for (py::ssize_t row=0; row<count_; ++row) {
            if (chosen_[row]) { output(row) = -1.0; continue; }
            double best = std::max(nearest_[row*(blocks_+1)+blocks_], local_scores_[row]);
            output(row) = std::sqrt(best);
        }
        return result;
    }
    py::array_t<double> distances() const {
        py::array_t<double> result({count_, blocks_+2});
        auto output = result.mutable_unchecked<2>();
        const auto* offsets = offsets_.data();
        py::gil_scoped_release release;
        for (py::ssize_t row=0; row<count_; ++row) {
            for (py::ssize_t col=0; col<=blocks_; ++col)
                output(row,col) = std::sqrt(nearest_[row*(blocks_+1)+col]);
            double best = 0.0;
            for (py::ssize_t atom=offsets[row]; atom<offsets[row+1]; ++atom) best=std::max(best,atomic_[atom]);
            output(row,blocks_+1)=std::sqrt(best);
        }
        return result;
    }
};

}  // namespace

PYBIND11_MODULE(_sampling, module) {
    module.attr("coverage_api_version") = 2;
    py::class_<CoverageState>(module, "CoverageState")
        .def(py::init<ContiguousArray<double>, ContiguousArray<long long>,
                      ContiguousArray<long long>, ContiguousArray<double>,
                      ContiguousArray<long long>, ContiguousArray<long long>, py::ssize_t>())
        .def("add", &CoverageState::add)
        .def("refine_atoms", &CoverageState::refine_atoms, py::arg("resolution")=0.0)
        .def("scores", &CoverageState::scores)
        .def("distances", &CoverageState::distances);
    module.doc() = "Native numerical primitives for sampling workflows.";
    module.def(
        "farthest_point_sampling",
        &farthest_point_sampling,
        py::arg("points"),
        py::arg("n_samples"),
        py::arg("min_distance") = 0.1,
        py::arg("selected_data") = py::none(),
        "Run deterministic greedy FPS with a streaming distance field.");
    module.def(
        "farthest_point_sampling_trace",
        &farthest_point_sampling_trace,
        py::arg("points"),
        py::arg("n_samples"),
        py::arg("target_coverage") = 0.995,
        py::arg("selected_data") = py::none(),
        "Run FPS and return nested coverage and radius traces.");
}
