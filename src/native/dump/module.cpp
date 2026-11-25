// Native LAMMPS dump atom-block parser.
//
// The Python layer owns frame headers, column semantics, error policy and
// batching.  This module only turns whitespace-separated tables into typed
// arrays, which keeps it independent of LAMMPS column names and styles.
// ``parse_blocks`` parses several independent frames in parallel.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../io/fast_float.h"

namespace py = pybind11;

namespace {

constexpr int kKindFloat = 0;
constexpr int kKindInt = 1;
constexpr int kKindString = 2;
constexpr int kMaxTokens = 128;
// Below this many atoms a batch is cheaper to parse sequentially: thread setup
// and wake-up would dominate a handful of short rows.
constexpr int64_t kParallelAtomThreshold = 25000;

inline bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

struct Row {
    const char *ptr[kMaxTokens];
    int len[kMaxTokens];
    int count{0};
    bool overflow{false};
};

// Advance to the next non-empty line and split it into tokens.
inline bool next_row(const char *&cursor, const char *end, Row &row) {
    while (cursor < end) {
        const char c = *cursor;
        if (c == '\n' || is_space(c)) {
            ++cursor;
            continue;
        }
        break;
    }
    if (cursor >= end) return false;
    row.count = 0;
    row.overflow = false;
    while (cursor < end && *cursor != '\n') {
        while (cursor < end && is_space(*cursor)) ++cursor;
        if (cursor >= end || *cursor == '\n') break;
        const char *start = cursor;
        while (cursor < end && *cursor != '\n' && !is_space(*cursor)) ++cursor;
        if (row.count < kMaxTokens) {
            row.ptr[row.count] = start;
            row.len[row.count] = static_cast<int>(cursor - start);
            ++row.count;
        } else {
            row.overflow = true;
        }
    }
    if (cursor < end && *cursor == '\n') ++cursor;
    return true;
}

inline bool parse_float_token(const char *p, int n, double &out) {
    const auto result = fast_float::from_chars(p, p + n, out, fast_float::chars_format::general);
    return result.ec == std::errc() && result.ptr == p + n;
}

inline bool parse_int_token(const char *p, int n, int64_t &out) {
    int i = 0;
    bool negative = false;
    if (i < n && (p[i] == '-' || p[i] == '+')) {
        negative = p[i] == '-';
        ++i;
    }
    if (i >= n) return false;
    int64_t value = 0;
    for (; i < n; ++i) {
        const char c = p[i];
        if (c < '0' || c > '9') return false;
        value = value * 10 + (c - '0');
    }
    out = negative ? -value : value;
    return true;
}

// Interns the few unique labels (element symbols, type labels) without
// allocating a std::string per row.
class InternTable {
public:
    int intern(const char *p, int n) {
        const uint64_t hash = fnv1a(p, n);
        std::vector<int> &bucket = buckets_[hash];
        for (const int id : bucket) {
            const std::string &candidate = values_[static_cast<size_t>(id)];
            if (static_cast<int>(candidate.size()) == n &&
                std::memcmp(candidate.data(), p, static_cast<size_t>(n)) == 0) {
                return id;
            }
        }
        const int id = static_cast<int>(values_.size());
        values_.emplace_back(p, static_cast<size_t>(n));
        bucket.push_back(id);
        return id;
    }

    const std::vector<std::string> &values() const { return values_; }

private:
    static uint64_t fnv1a(const char *p, int n) {
        uint64_t hash = 1469598103934665603ULL;
        for (int i = 0; i < n; ++i) {
            hash ^= static_cast<unsigned char>(p[i]);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    std::vector<std::string> values_;
    std::unordered_map<uint64_t, std::vector<int>> buckets_;
};

struct FrameOutput {
    std::vector<py::object> outputs;
    std::vector<double *> float_columns;
    std::vector<int32_t *> int_columns;
    std::vector<InternTable> intern_tables;
};

struct FrameStatus {
    int64_t rows{0};
    int64_t problem_row{-1};
    int64_t problem_width{0};
    int64_t bad_row{-1};
    bool problem_at_eof{false};
    bool failed{false};
};

void allocate_output(FrameOutput &output, py::ssize_t n_atoms, const std::vector<int> &kinds) {
    const size_t n_columns = kinds.size();
    output.outputs.resize(n_columns);
    output.float_columns.assign(n_columns, nullptr);
    output.int_columns.assign(n_columns, nullptr);
    output.intern_tables.resize(n_columns);
    for (size_t index = 0; index < n_columns; ++index) {
        if (kinds[index] == kKindFloat) {
            py::array_t<double> values(n_atoms);
            output.float_columns[index] = static_cast<double *>(values.mutable_data());
            output.outputs[index] = std::move(values);
        } else if (kinds[index] == kKindInt || kinds[index] == kKindString) {
            py::array_t<int32_t> values(n_atoms);
            output.int_columns[index] = static_cast<int32_t *>(values.mutable_data());
            output.outputs[index] = std::move(values);
        } else {
            throw std::runtime_error("unknown dump column kind");
        }
    }
}

// Pure C++: no Python API calls, safe to run on a worker thread.
void parse_frame(const char *base, const char *block_end, int64_t n_atoms, int n_tokens,
                 const std::vector<int> &tokens, const std::vector<int> &kinds,
                 FrameOutput &output, FrameStatus &status) {
    const char *cursor = base;
    Row row;
    while (status.rows < n_atoms && next_row(cursor, block_end, row)) {
        if (row.overflow || row.count != n_tokens) {
            status.problem_row = status.rows;
            status.problem_width = row.overflow ? n_tokens + 1 : row.count;
            const char *rest = cursor;
            while (rest < block_end && (is_space(*rest) || *rest == '\n')) ++rest;
            status.problem_at_eof = rest >= block_end;
            return;
        }
        for (size_t index = 0; index < tokens.size(); ++index) {
            const int token = tokens[index];
            const char *text = row.ptr[token];
            const int length = row.len[token];
            if (kinds[index] == kKindFloat) {
                double value = 0.0;
                if (parse_float_token(text, length, value)) {
                    output.float_columns[index][status.rows] = value;
                } else {
                    output.float_columns[index][status.rows] = std::nan("");
                    if (status.bad_row < 0) status.bad_row = status.rows;
                }
            } else if (kinds[index] == kKindInt) {
                int64_t value = 0;
                if (parse_int_token(text, length, value)) {
                    output.int_columns[index][status.rows] = static_cast<int32_t>(value);
                } else {
                    output.int_columns[index][status.rows] = 0;
                    if (status.bad_row < 0) status.bad_row = status.rows;
                }
            } else {
                output.int_columns[index][status.rows] = output.intern_tables[index].intern(text, length);
            }
        }
        ++status.rows;
        if (status.bad_row >= 0) return;
    }
    if (status.problem_row < 0 && status.bad_row < 0) {
        Row extra;
        const char *rest = cursor;
        if (next_row(rest, block_end, extra)) {
            status.problem_row = status.rows;
            status.problem_width = -1;
        }
    }
}

py::tuple make_result(FrameOutput &output, const FrameStatus &status, const std::vector<int> &kinds) {
    py::list columns;
    for (size_t index = 0; index < kinds.size(); ++index) {
        if (kinds[index] == kKindString) {
            py::list labels;
            for (const auto &value : output.intern_tables[index].values()) {
                labels.append(py::str(value));
            }
            columns.append(py::make_tuple(output.outputs[index], labels));
        } else {
            columns.append(output.outputs[index]);
        }
    }
    return py::make_tuple(columns, status.rows, status.problem_row, status.problem_width,
                          status.bad_row, status.problem_at_eof);
}

int resolve_workers(int max_workers, size_t frames, int64_t total_atoms) {
    int workers = max_workers > 0 ? max_workers : static_cast<int>(std::thread::hardware_concurrency());
    if (workers < 1) workers = 1;
    workers = std::min(workers, static_cast<int>(frames));
    if (total_atoms < kParallelAtomThreshold) workers = 1;
    return workers;
}

const char *checked_pointer(const char *base, py::ssize_t size, py::ssize_t offset) {
    if (offset < 0 || offset > size) {
        throw std::runtime_error("dump frame offset is outside the block buffer");
    }
    return base + offset;
}

py::tuple parse_block(py::buffer buffer, py::ssize_t n_atoms, py::ssize_t n_tokens,
                      std::vector<int> tokens, std::vector<int> kinds) {
    const py::buffer_info info = buffer.request();
    if (info.ndim != 1 || info.itemsize != 1) {
        throw std::runtime_error("dump block buffer must be one-dimensional bytes");
    }
    if (tokens.size() != kinds.size()) {
        throw std::runtime_error("dump block column specification mismatch");
    }
    for (const int token : tokens) {
        if (token < 0 || token >= n_tokens) {
            throw std::runtime_error("dump token index is outside the frame width");
        }
    }
    const char *const base = static_cast<const char *>(info.ptr);
    FrameOutput output;
    FrameStatus status;
    allocate_output(output, n_atoms, kinds);
    {
        py::gil_scoped_release release;
        parse_frame(base, base + info.size, n_atoms, static_cast<int>(n_tokens), tokens, kinds, output,
                    status);
    }
    return make_result(output, status, kinds);
}

py::list parse_blocks(py::buffer buffer, std::vector<py::ssize_t> starts,
                      std::vector<py::ssize_t> ends, std::vector<py::ssize_t> n_atoms,
                      py::ssize_t n_tokens, std::vector<int> tokens, std::vector<int> kinds,
                      int max_workers) {
    const py::buffer_info info = buffer.request();
    if (info.ndim != 1 || info.itemsize != 1) {
        throw std::runtime_error("dump block buffer must be one-dimensional bytes");
    }
    const size_t frames = starts.size();
    if (ends.size() != frames || n_atoms.size() != frames) {
        throw std::runtime_error("dump frame list length mismatch");
    }
    if (tokens.size() != kinds.size()) {
        throw std::runtime_error("dump block column specification mismatch");
    }
    for (const int token : tokens) {
        if (token < 0 || token >= n_tokens) {
            throw std::runtime_error("dump token index is outside the frame width");
        }
    }
    const char *const base = static_cast<const char *>(info.ptr);
    int64_t total_atoms = 0;
    std::vector<const char *> frame_starts(frames);
    std::vector<const char *> frame_ends(frames);
    for (size_t index = 0; index < frames; ++index) {
        frame_starts[index] = checked_pointer(base, info.size, starts[index]);
        frame_ends[index] = checked_pointer(base, info.size, ends[index]);
        if (frame_ends[index] < frame_starts[index]) {
            throw std::runtime_error("dump frame range is reversed");
        }
        total_atoms += n_atoms[index];
    }

    std::vector<FrameOutput> outputs(frames);
    std::vector<FrameStatus> statuses(frames);
    for (size_t index = 0; index < frames; ++index) {
        allocate_output(outputs[index], n_atoms[index], kinds);
    }

    const int workers = resolve_workers(max_workers, frames, total_atoms);
    {
        py::gil_scoped_release release;
        if (workers <= 1) {
            for (size_t index = 0; index < frames; ++index) {
                try {
                    parse_frame(frame_starts[index], frame_ends[index], n_atoms[index],
                                static_cast<int>(n_tokens), tokens, kinds, outputs[index],
                                statuses[index]);
                } catch (...) {
                    statuses[index].failed = true;
                }
            }
        } else {
            std::atomic<size_t> next{0};
            std::vector<std::thread> pool;
            pool.reserve(static_cast<size_t>(workers));
            try {
                for (int worker = 0; worker < workers; ++worker) {
                    pool.emplace_back([&]() {
                        for (;;) {
                            const size_t index = next.fetch_add(1);
                            if (index >= frames) break;
                            try {
                                parse_frame(frame_starts[index], frame_ends[index], n_atoms[index],
                                            static_cast<int>(n_tokens), tokens, kinds,
                                            outputs[index], statuses[index]);
                            } catch (...) {
                                statuses[index].failed = true;
                            }
                        }
                    });
                }
            } catch (...) {
                for (auto &thread : pool) {
                    if (thread.joinable()) thread.join();
                }
                throw;
            }
            for (auto &thread : pool) thread.join();
        }
    }

    py::list results;
    for (size_t index = 0; index < frames; ++index) {
        if (statuses[index].failed) {
            throw std::runtime_error("dump frame parsing failed");
        }
        results.append(make_result(outputs[index], statuses[index], kinds));
    }
    return results;
}

}  // namespace

PYBIND11_MODULE(_dump, module) {
    module.doc() =
        "Native LAMMPS dump atom-block parser (token layout supplied by Python).";
    module.def("parse_block", &parse_block, py::arg("buffer"), py::arg("n_atoms"),
               py::arg("n_tokens"), py::arg("tokens"), py::arg("kinds"),
               "Parse one atom table into typed arrays.");
    module.def("parse_blocks", &parse_blocks, py::arg("buffer"), py::arg("starts"),
               py::arg("ends"), py::arg("n_atoms"), py::arg("n_tokens"), py::arg("tokens"),
               py::arg("kinds"), py::arg("max_workers") = -1,
               "Parse several atom tables in parallel; one result per frame.");
}
