#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
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

}  // namespace

PYBIND11_MODULE(_sampling, module) {
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
