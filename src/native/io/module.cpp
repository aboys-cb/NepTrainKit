// Fast EXTXYZ parsing with mmap buffer input
// Build with pybind11; exposed as NepTrainKit._native._io.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <Python.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <system_error>
#include <thread>

// Use single-header fast_float colocated in this directory
#include "fast_float.h"
#define NEPKIT_HAVE_FAST_FLOAT 1

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Release the GIL only if currently held by this thread.
struct ScopedReleaseIfHeld {
    PyThreadState* state{nullptr};
    ScopedReleaseIfHeld() {
        if (PyGILState_Check()) {
            state = PyEval_SaveThread();
        }
    }
    ~ScopedReleaseIfHeld() {
        if (state) {
            PyEval_RestoreThread(state);
        }
    }
    ScopedReleaseIfHeld(const ScopedReleaseIfHeld&) = delete;
    ScopedReleaseIfHeld& operator=(const ScopedReleaseIfHeld&) = delete;
};

constexpr size_t kPythonYieldInterval = 256;

inline void cooperative_python_yield(const size_t completed) {
    if (completed == 0 || completed % kPythonYieldInterval != 0) return;
    ScopedReleaseIfHeld release;
    std::this_thread::yield();
}

struct PropDesc {
    std::string name;
    char dtype{'S'}; // 'S','R','I','L'
    int count{1};
};

struct FrameIndex {
    size_t off_num{0};
    size_t off_header{0};
    size_t off_data{0};
    size_t end{0};
    int num_atoms{0};
};

inline const char* skip_ws(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
    return p;
}

inline const char* find_eol(const char* p, const char* end) {
    // Use memchr to speed up newline search on large buffers
    const void* res = std::memchr(p, '\n', static_cast<size_t>(end - p));
    return res ? static_cast<const char*>(res) : end;
}

// parse integer at start of line
inline bool parse_int(const char* p, const char* end, int& out, const char** next) {
    const char* s = skip_ws(p, end);
    bool neg = false;
    if (s < end && (*s == '+' || *s == '-')) { neg = (*s == '-'); ++s; }
    long long val = 0;
    const char* start = s;
    while (s < end && std::isdigit(static_cast<unsigned char>(*s))) { val = val * 10 + (*s - '0'); ++s; }
    if (s == start) return false;
    out = static_cast<int>(neg ? -val : val);
    if (next) *next = s;
    return true;
}

inline double parse_double(const char*& p, const char* end) {
    const char* q = p;
    while (q < end && !std::isspace(static_cast<unsigned char>(*q))) ++q;
#if defined(NEPKIT_HAVE_FAST_FLOAT)
    double v_ff = 0.0;
    auto ffres = fast_float::from_chars(p, q, v_ff, fast_float::chars_format::general);
    if (ffres.ec == std::errc() && ffres.ptr == q && std::isfinite(v_ff)) {
        p = q;
        return v_ff;
    }
#endif
    std::string token(p, q);
    char* e = nullptr;
    double v = std::strtod(token.c_str(), &e);
    p = q;
    if (e != token.c_str() + token.size() || !std::isfinite(v)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return v;
}

inline bool parse_double_token(const char*& p, const char* end, double& out) {
    const char* q = p;
    while (q < end && !std::isspace(static_cast<unsigned char>(*q))) ++q;
    const char* next = p;
    out = parse_double(next, q);
    p = q;
    return std::isfinite(out);
}

inline bool parse_int_token(const char*& p, const char* end, int& out) {
    const char* s = p;
    const char* q = p;
    while (q < end && !std::isspace(static_cast<unsigned char>(*q))) ++q;
    bool neg = false;
    if (s < q && (*s == '+' || *s == '-')) { neg = (*s == '-'); ++s; }
    long long val = 0;
    const char* start = s;
    while (s < q && std::isdigit(static_cast<unsigned char>(*s))) { val = val * 10 + (*s - '0'); ++s; }
    if (s == start || s != q) {
        p = q;
        return false;
    }
    out = static_cast<int>(neg ? -val : val);
    p = q;
    return true;
}

inline bool parse_bool_token(const char*& p, const char* end, uint8_t& out) {
    const char* s = p;
    const char* q = s;
    while (q < end && !std::isspace(static_cast<unsigned char>(*q))) ++q;
    size_t len = static_cast<size_t>(q - s);
    bool valid = false;
    bool v = false;
    if (len == 1) {
        if (*s == '1' || *s == 'T' || *s == 't') {
            valid = true;
            v = true;
        } else if (*s == '0' || *s == 'F' || *s == 'f') {
            valid = true;
        }
    } else {
        bool is_true = (len == 4 && (s[0]=='t'||s[0]=='T') && (s[1]=='r'||s[1]=='R') && (s[2]=='u'||s[2]=='U') && (s[3]=='e'||s[3]=='E'));
        bool is_false = (len == 5 && (s[0]=='f'||s[0]=='F') && (s[1]=='a'||s[1]=='A') && (s[2]=='l'||s[2]=='L') && (s[3]=='s'||s[3]=='S') && (s[4]=='e'||s[4]=='E'));
        valid = is_true || is_false;
        v = is_true;
    }
    out = v ? 1 : 0;
    p = q;
    return valid;
}

inline std::string parse_token(const char*& p, const char* end) {
    p = skip_ws(p, end);
    const char* q = p;
    while (q < end && !std::isspace(static_cast<unsigned char>(*q))) ++q;
    std::string out(p, q);
    p = q;
    return out;
}

static std::vector<PropDesc> parse_properties_desc(const std::string& s) {
    std::vector<PropDesc> props;
    size_t i = 0, n = s.size();
    auto next_token = [&](std::string& tok)->bool{
        if (i >= n) return false;
        size_t j = i;
        while (j < n && s[j] != ':') ++j;
        tok.assign(s.data() + i, j - i);
        i = (j < n ? j + 1 : j);
        return true;
    };
    while (i < n) {
        std::string name, dtype, count;
        if (!next_token(name)) break;
        if (!next_token(dtype)) break;
        // count may be missing -> default 1
        size_t save = i;
        if (!next_token(count)) { count = "1"; i = save; }
        PropDesc d;
        d.name = name;
        d.dtype = dtype.empty() ? 'S' : static_cast<char>(dtype[0]);
        try { d.count = std::stoi(count); } catch (...) { d.count = 0; }
        props.push_back(std::move(d));
    }
    return props;
}

enum class AddType { STRING, DOUBLE, FLOATS };
struct AddValue {
    AddType type{AddType::STRING};
    std::string s;
    double d{0.0};
    std::vector<float> vf;
};

static bool normalise_pbc_and_validate_cell(
    const std::vector<float>& lattice,
    std::unordered_map<std::string, AddValue>& additional,
    std::string& error
) {
    auto pbc_it = additional.find("pbc");
    std::string raw = pbc_it == additional.end() ? "T T T" : pbc_it->second.s;
    std::replace(raw.begin(), raw.end(), ',', ' ');
    std::vector<uint8_t> flags;
    const char* cursor = raw.data();
    const char* end = raw.data() + raw.size();
    while (cursor < end) {
        cursor = skip_ws(cursor, end);
        if (cursor >= end) break;
        uint8_t value = 0;
        if (!parse_bool_token(cursor, end, value)) {
            error = "pbc contains an invalid logical value";
            return false;
        }
        flags.push_back(value);
    }
    if (flags.size() == 1) flags.resize(3, flags.front());
    if (flags.size() != 3) {
        error = "pbc must contain exactly three logical values";
        return false;
    }

    AddValue canonical;
    canonical.type = AddType::STRING;
    canonical.s =
        std::string(flags[0] ? "T" : "F") + " "
        + (flags[1] ? "T" : "F") + " "
        + (flags[2] ? "T" : "F");
    additional["pbc"] = std::move(canonical);

    std::vector<int> periodic;
    for (int axis = 0; axis < 3; ++axis) {
        if (flags[static_cast<size_t>(axis)] != 0) periodic.push_back(axis);
    }
    if (periodic.empty()) return true;
    if (lattice.size() != 9) {
        error = "periodic cell is missing";
        return false;
    }

    constexpr double tolerance = 1.0e-12;
    auto component = [&](int row, int column) {
        return static_cast<double>(lattice[static_cast<size_t>(row * 3 + column)]);
    };
    for (int row : periodic) {
        const double norm2 =
            component(row, 0) * component(row, 0)
            + component(row, 1) * component(row, 1)
            + component(row, 2) * component(row, 2);
        if (norm2 <= tolerance * tolerance) {
            error = "periodic cell contains a zero-length periodic vector";
            return false;
        }
    }
    if (periodic.size() == 2) {
        const int a = periodic[0];
        const int b = periodic[1];
        const double cx = component(a, 1) * component(b, 2) - component(a, 2) * component(b, 1);
        const double cy = component(a, 2) * component(b, 0) - component(a, 0) * component(b, 2);
        const double cz = component(a, 0) * component(b, 1) - component(a, 1) * component(b, 0);
        if (cx * cx + cy * cy + cz * cz <= tolerance * tolerance) {
            error = "periodic cell vectors are linearly dependent";
            return false;
        }
    } else if (periodic.size() == 3) {
        const double determinant =
            component(0, 0) * (component(1, 1) * component(2, 2) - component(1, 2) * component(2, 1))
            - component(0, 1) * (component(1, 0) * component(2, 2) - component(1, 2) * component(2, 0))
            + component(0, 2) * (component(1, 0) * component(2, 1) - component(1, 1) * component(2, 0));
        if (std::abs(determinant) <= tolerance) {
            error = "periodic cell vectors are linearly dependent";
            return false;
        }
    }
    return true;
}

static void parse_header_line(const char* b, const char* e,
                              std::vector<float>& lattice_out,
                              std::vector<PropDesc>& props_out,
                              std::unordered_map<std::string, AddValue>& add_out) {
    // header format: key=value tokens separated by spaces
    const char* p = b;
    while (p < e) {
        p = skip_ws(p, e);
        if (p >= e) break;
        const char* k0 = p;
        while (p < e && *p != '=' && !std::isspace(static_cast<unsigned char>(*p))) ++p;
        std::string key(k0, p);
        p = skip_ws(p, e);
        if (p < e && *p == '=') ++p; else { // malformed, skip to next space
            p = find_eol(p, e);
            break;
        }
        p = skip_ws(p, e);
        std::string value;
        if (p < e && *p == '"') {
            ++p;
            while (p < e) {
                const char c = *p++;
                if (c == '\\' && p < e) {
                    value.push_back(*p++);
                } else if (c == '"') {
                    break;
                } else {
                    value.push_back(c);
                }
            }
        } else {
            const char* v0 = p;
            while (p < e && !std::isspace(static_cast<unsigned char>(*p))) ++p;
            value.assign(v0, p);
        }
        if (!key.empty()) {
            if (key == "Lattice" || key == "lattice" || key == "LATTICE") {
                // value: 9 floats separated by spaces
                lattice_out.clear();
                lattice_out.reserve(9);
                const char* vp = value.data();
                const char* ve = value.data() + value.size();
                while (vp < ve) {
                    vp = skip_ws(vp, ve);
                    if (vp >= ve) break;
                    lattice_out.push_back(static_cast<float>(parse_double(vp, ve)));
                }
                // ensure 9
                if (lattice_out.size() != 9) {
                    // leave as-is; higher-level code can reshape or ignore
                }
            } else if (key == "Properties" || key == "properties" || key == "PROPERTIES") {
                props_out = parse_properties_desc(value);
            } else {
                // additional fields: collect as pure C++ types (no Python in threads)
                if (key == "energy" || key == "Energy") {
                    AddValue v; v.type = AddType::DOUBLE;
                    try { v.d = std::stod(value); } catch (...) { v.type = AddType::STRING; v.s = value; }
                    add_out["energy"] = std::move(v);
                } else if (key == "weight" || key == "Weight" || key == "WEIGHT") {
                    AddValue v; v.type = AddType::DOUBLE;
                    try { v.d = std::stod(value); } catch (...) { v.type = AddType::STRING; v.s = value; }
                    add_out["weight"] = std::move(v);

                } else if (key == "pbc" || key == "PBC") {
                    AddValue v; v.type = AddType::STRING; v.s = value;
                    add_out["pbc"] = std::move(v);
                } else if (key == "virial" || key == "stress" || key == "VIRIAL" || key == "STRESS" || key == "Virial" || key == "Stress") {
                    AddValue v; v.type = AddType::FLOATS;
                    const char* vp = value.data();
                    const char* ve = value.data() + value.size();
                    while (vp < ve) {
                        vp = skip_ws(vp, ve);
                        if (vp >= ve) break;
                        v.vf.push_back(static_cast<float>(parse_double(vp, ve)));
                    }
                    const std::string norm = (key[0]=='s'||key[0]=='S')?"stress":"virial";
                    add_out[norm] = std::move(v);
                } else if (key == "config_type" || key == "Config_type" ) {
                    AddValue v; v.type = AddType::STRING; v.s = value;
                    add_out["Config_type"] = std::move(v);
                } else {
                    AddValue v; v.type = AddType::STRING; v.s = value;
                    add_out[key] = std::move(v);
                }
            }
        }
    }
}

static std::vector<FrameIndex> index_frames(const char* buf, size_t nbytes) {
    std::vector<FrameIndex> out;
    const char* p = buf;
    const char* end = buf + nbytes;
    while (p < end) {
        // line 1: num atoms
        const char* l1 = p;
        const char* e1 = find_eol(l1, end);
        if (l1 == e1) { if (e1 < end) { p = e1 + 1; continue; } else break; }
        int num = 0; const char* after = nullptr;
        if (!parse_int(l1, e1, num, &after) || skip_ws(after, e1) != e1 || num < 0) {
            throw std::runtime_error(
                "Malformed EXTXYZ frame " + std::to_string(out.size() + 1)
                + ": invalid atom count"
            );
        }
        // line 2: header
        const char* l2 = (e1 < end) ? e1 + 1 : end;
        if (l2 >= end) {
            throw std::runtime_error(
                "Malformed EXTXYZ frame " + std::to_string(out.size() + 1)
                + ": missing header"
            );
        }
        const char* e2 = find_eol(l2, end);
        // data lines
        const char* d0 = (e2 < end) ? e2 + 1 : end;
        const char* d = d0;
        for (int i = 0; i < num && d < end; ++i) {
            d = find_eol(d, end);
            if (d < end) ++d;
        }
        FrameIndex fi;
        fi.off_num = static_cast<size_t>(l1 - buf);
        fi.off_header = static_cast<size_t>(l2 - buf);
        fi.off_data = static_cast<size_t>(d0 - buf);
        fi.end = static_cast<size_t>(d - buf);
        fi.num_atoms = num;
        out.push_back(fi);
        p = d; // next frame
    }
    return out;
}

// Thread count for parallel per-frame parsing.
static int compute_threads(int max_workers) {
    int nthreads = 1;
#ifdef _OPENMP
    // base on available hardware threads
    nthreads = omp_get_max_threads();
    if (max_workers > 0) nthreads = std::min(nthreads, max_workers);
    // optional env override
    if (const char* env = std::getenv("NEPKIT_FASTXYZ_THREADS")) {
        int v = std::atoi(env);
        if (v > 0) nthreads = std::min(nthreads, v);
    }
#else
    (void)max_workers;
#endif
    if (nthreads < 1) nthreads = 1;
    return nthreads;
}

// Parse frames into Python-friendly dicts
static bool _env_debug_on() {
    const char* e = std::getenv("NEPKIT_FASTXYZ_DEBUG");
    if (!e) e = std::getenv("FASTXYZ_DEBUG");
    if (!e) return false;
    return (e[0]=='1' || e[0]=='t' || e[0]=='T' || e[0]=='y' || e[0]=='Y');
}

// Species mode selection (string vs id). Define before use in parse_all_impl.
enum class SpeciesMode { STR, ID, Z_UNSUPPORTED };
static SpeciesMode _species_mode() {
    const char* s = std::getenv("NEPKIT_FASTXYZ_SPECIES_MODE");
    if (!s) return SpeciesMode::STR;
    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
    if (v == "str" || v.empty()) return SpeciesMode::STR;
    if (v == "id") return SpeciesMode::ID;
    if (v == "z") return SpeciesMode::Z_UNSUPPORTED; // fallback warning later
    return SpeciesMode::STR;
}

static py::list parse_all_impl(py::buffer bbuf, int max_workers) {
    py::buffer_info info = bbuf.request();
    if (info.ndim != 1 || info.itemsize != 1) {
        throw std::runtime_error("buffer must be 1D bytes");
    }
    const char* base = static_cast<const char*>(info.ptr);
    size_t nbytes = static_cast<size_t>(info.size);

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    bool dbg = _env_debug_on();
    if (dbg) {
        std::fprintf(stderr, "[fastxyz] parse_all begin: bytes=%zu max_workers=%d OMP=%s\n",
                     nbytes, max_workers,
#ifdef _OPENMP
                     "on"
#else
                     "off"
#endif
        );
        std::fflush(stderr);
    }

    auto t_idx0 = clock::now();
    std::vector<FrameIndex> frames;
    {
        // Indexing is pure C++; release the GIL here as well.
        ScopedReleaseIfHeld _nogil_idx;
        // Frame boundaries are stateful: the atom count determines exactly how
        // many lines must be skipped before the next frame can begin.  Splitting
        // the raw buffer at arbitrary line boundaries can start inside an atom
        // block, where a numeric first property (for example pos:R:3) may be
        // mistaken for the next atom count.  Keep indexing serial and parallelise
        // the substantially heavier per-frame parsing below.
        frames = index_frames(base, nbytes);
    }
    auto t_idx1 = clock::now();
    if (dbg) {
        std::fprintf(stderr, "[fastxyz] index done: frames=%zu (%.2f ms)\n",
                     frames.size(), std::chrono::duration<double, std::milli>(t_idx1 - t_idx0).count());
        std::fflush(stderr);
    }

    struct Parsed {
        std::vector<float> lattice; // 9
        std::vector<PropDesc> props;
        std::unordered_map<std::string, AddValue> add;
        // numeric buffers (zero-copy into NumPy via capsule)
        std::unordered_map<std::string, std::unique_ptr<float[]>> rbuf;
        std::unordered_map<std::string, std::unique_ptr<int32_t[]>> ibuf;
        std::unordered_map<std::string, std::unique_ptr<uint8_t[]>> lbuf;
        std::unordered_map<std::string, size_t> totals;
        // string properties stored as tokens, converted later
        std::unordered_map<std::string, std::vector<std::string>> sprops;
        int num_atoms{0};
        std::string error;
    };

    std::vector<Parsed> parsed(frames.size());

    // Determine threads
    int nthreads = compute_threads(max_workers);
    if (dbg) { nthreads = 1; }

    // First, parse all frames in parallel into plain C++ storage
    auto t_par0 = clock::now();
    {
    ScopedReleaseIfHeld _nogil;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
#endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(frames.size()); ++i) {
        const FrameIndex& fi = frames[static_cast<size_t>(i)];
        Parsed p;
        // Validate offsets to avoid invalid memory access on malformed files
        if (!(fi.off_header <= fi.off_data && fi.off_data <= fi.end)) {
            if (dbg) {
                std::fprintf(stderr, "[fastxyz] frame %lld invalid offsets: header=%zu data=%zu end=%zu\n",
                             (long long)i, (size_t)fi.off_header, (size_t)fi.off_data, (size_t)fi.end);
                std::fflush(stderr);
            }
            p.error = "invalid frame offsets";
            parsed[static_cast<size_t>(i)] = std::move(p);
            continue;
        }
        p.num_atoms = std::max(0, fi.num_atoms);
        if (fi.num_atoms < 0) {
            p.error = "atom count must be non-negative";
        }

        const char* h0 = base + fi.off_header;
        const char* h1 = (fi.off_data > fi.off_header) ? base + fi.off_data - 1 : base + fi.off_header;
        if (h1 > h0 && *(h1-1) == '\r') --h1; // trim CR
        if (dbg) {
            std::fprintf(stderr, "[fastxyz] frame %lld header parse: num_atoms=%d header_len=%lld\n",
                         (long long)i, p.num_atoms, (long long)(h1 - h0));
            std::fflush(stderr);
        }
        parse_header_line(h0, h1, p.lattice, p.props, p.add);

        if (p.lattice.size() != 9) {
            p.error = "Lattice must contain exactly nine finite values";
        } else {
            for (float value : p.lattice) {
                if (!std::isfinite(value)) {
                    p.error = "Lattice must contain exactly nine finite values";
                    break;
                }
            }
        }
        if (p.props.empty()) {
            p.error = "Properties is required";
        }
        bool has_species = false;
        bool has_pos = false;
        std::unordered_map<std::string, bool> property_names;
        for (const auto& desc : p.props) {
            if (desc.name.empty() || property_names.count(desc.name) != 0) {
                p.error = "property names must be non-empty and unique";
                break;
            }
            property_names[desc.name] = true;
            if (
                desc.count <= 0
                || (desc.dtype != 'S' && desc.dtype != 'R' && desc.dtype != 'I' && desc.dtype != 'L')
            ) {
                p.error = "Properties contains an invalid type or column count";
                break;
            }
            if (desc.name == "species") {
                has_species = desc.dtype == 'S' && desc.count == 1;
            } else if (desc.name == "pos") {
                has_pos = desc.dtype == 'R' && desc.count == 3;
            }
        }
        if (!has_species || !has_pos) {
            p.error = "Properties must include species:S:1 and pos:R:3";
        }
        if (p.error.empty()) {
            normalise_pbc_and_validate_cell(p.lattice, p.add, p.error);
        }
        if (dbg) {
            std::fprintf(stderr, "[fastxyz] frame %lld props=%zu\n", (long long)i, p.props.size());
            std::fflush(stderr);
        }

        // allocate arrays
        for (const auto& d : p.props) {
            size_t total = (p.num_atoms > 0 && d.count > 0)
                           ? (static_cast<size_t>(p.num_atoms) * static_cast<size_t>(d.count))
                           : 0u;
            p.totals[d.name] = total;
            switch (d.dtype) {
                case 'R': p.rbuf[d.name] = std::unique_ptr<float[]>(new float[total]()); break;
                case 'I': p.ibuf[d.name] = std::unique_ptr<int32_t[]>(new int32_t[total]()); break;
                case 'L': p.lbuf[d.name] = std::unique_ptr<uint8_t[]>(new uint8_t[total]()); break;
                case 'S': default: p.sprops[d.name].assign(total, std::string()); break;
            }
            if (dbg) {
                std::fprintf(stderr, "[fastxyz] frame %lld alloc prop name=%s type=%c count=%d total=%zu\n",
                             (long long)i, d.name.c_str(), d.dtype, d.count, total);
                std::fflush(stderr);
            }
        }

        // parse atom lines
        const char* d = base + fi.off_data;
        const char* dend = base + fi.end;
        int actual_rows = 0;
        for (; actual_rows < p.num_atoms && d < dend && p.error.empty(); ++actual_rows) {
            const char* l0 = d;
            const char* le = find_eol(l0, dend);
            const char* q = l0;
            bool row_ok = true;
            // For speed, scan tokens sequentially across properties
            for (const auto& desc : p.props) {
                for (int k = 0; k < desc.count; ++k) {
                    q = skip_ws(q, le);
                    if (q >= le) {
                        row_ok = false;
                        p.error = "atom row has fewer columns than declared by Properties";
                        break;
                    }
                    if (desc.dtype == 'R') {
                        double v = 0.0;
                        if (!parse_double_token(q, le, v)) {
                            row_ok = false;
                            p.error = "atom row contains an invalid floating-point value";
                            break;
                        }
                        size_t idx = static_cast<size_t>(actual_rows) * static_cast<size_t>(desc.count) + static_cast<size_t>(k);
                        p.rbuf[desc.name][idx] = static_cast<float>(v);
                    } else if (desc.dtype == 'I') {
                        int v = 0;
                        if (!parse_int_token(q, le, v)) {
                            row_ok = false;
                            p.error = "atom row contains an invalid integer value";
                            break;
                        }
                        size_t idx = static_cast<size_t>(actual_rows) * static_cast<size_t>(desc.count) + static_cast<size_t>(k);
                        p.ibuf[desc.name][idx] = v;
                    } else if (desc.dtype == 'L') {
                        uint8_t v = 0;
                        if (!parse_bool_token(q, le, v)) {
                            row_ok = false;
                            p.error = "atom row contains an invalid logical value";
                            break;
                        }
                        size_t idx = static_cast<size_t>(actual_rows) * static_cast<size_t>(desc.count) + static_cast<size_t>(k);
                        p.lbuf[desc.name][idx] = v;
                    } else { // 'S'
                        // string token
                        const char* t0 = q;
                        while (q < le && !std::isspace(static_cast<unsigned char>(*q))) ++q;
                        std::string tok(t0, q);
                        size_t idx = static_cast<size_t>(actual_rows) * static_cast<size_t>(desc.count) + static_cast<size_t>(k);
                        p.sprops[desc.name][idx] = std::move(tok);
                    }
                }
                if (!row_ok) break;
            }
            if (!row_ok) break;
            q = skip_ws(q, le);
            if (q != le) {
                p.error = "atom row has more columns than declared by Properties";
                break;
            }
            d = (le < dend ? le + 1 : le);
        }
        if (p.error.empty() && actual_rows < p.num_atoms) {
            p.error = "frame ended before all declared atom rows were read";
        }

            parsed[static_cast<size_t>(i)] = std::move(p);
        }
    }

    auto t_par1 = clock::now();
    for (size_t index = 0; index < parsed.size(); ++index) {
        if (!parsed[index].error.empty()) {
            throw std::runtime_error(
                "Malformed EXTXYZ frame " + std::to_string(index + 1)
                + ": " + parsed[index].error
            );
        }
    }
    size_t total_atoms = 0; for (const auto& p : parsed) total_atoms += static_cast<size_t>(p.num_atoms);

    // Convert to Python list of dicts
    auto t_cvt0 = clock::now();
    py::list out;
    if (dbg) {
        std::fprintf(stderr, "[fastxyz] convert begin: frames=%zu\n", parsed.size());
        std::fflush(stderr);
    }
    // Intern cache local to this call to avoid lifetime/threading issues
    std::unordered_map<std::string, py::object> s_intern_cache;
    s_intern_cache.reserve(256);
    SpeciesMode s_mode = _species_mode();
    static bool warned_z = false;
    if (s_mode == SpeciesMode::Z_UNSUPPORTED && !warned_z) {
        std::fprintf(stderr, "[fastxyz] species mode 'z' not implemented; falling back to 'id'\n");
        warned_z = true;
        s_mode = SpeciesMode::ID;
    }

    for (size_t fi = 0; fi < parsed.size(); ++fi) {
        auto& p = parsed[fi];
        if (dbg) {
            std::fprintf(stderr, "[fastxyz] convert frame %zu: num_atoms=%d props=%zu\n", fi, p.num_atoms, p.props.size());
            std::fflush(stderr);
        }
        py::dict frame;
        // lattice (list of 9 floats or empty)
        if (!p.lattice.empty()) {
            py::array_t<float> arr(p.lattice.size());
            std::memcpy(arr.mutable_data(), p.lattice.data(), p.lattice.size()*sizeof(float));
            frame["lattice"] = std::move(arr);
        } else {
            frame["lattice"] = py::array_t<float>(0);
        }
        // properties (list of dicts)
        py::list props;
        bool has_species_prop = false;
        for (auto& d : p.props) {
            py::dict pd;
            pd["name"] = py::str(d.name);
            pd["type"] = py::str(std::string(1, d.dtype));
            pd["count"] = d.count;
            props.append(pd);
            if (d.name == "species" && d.dtype == 'S' && d.count == 1) has_species_prop = true;
        }
        if (s_mode != SpeciesMode::STR && has_species_prop) {
            // expose species_id property when generated
            py::dict pd;
            pd["name"] = py::str("species_id");
            pd["type"] = py::str("I");
            pd["count"] = 1;
            props.append(pd);
        }
        frame["properties"] = props;

        // atomic_properties
        py::dict ap;
    for (auto& kv : p.rbuf) {
            const std::string& name = kv.first;
            float* raw = kv.second.release();
            size_t total = p.totals[name];
            py::capsule c(raw, [](void* p){ delete[] static_cast<float*>(p); });
            py::array_t<float> arr({ (py::ssize_t) total }, raw, c);
            // reshape if count > 1
            int count = 1;
            for (auto& d : p.props) if (d.name == name) { count = d.count; break; }
            if (count > 1) {
                if (dbg && (size_t)p.num_atoms * (size_t)count != total) {
                    std::fprintf(stderr, "[fastxyz] reshape mismatch R: total=%zu != %zu*%d\n", total, (size_t)p.num_atoms, count);
                    std::fflush(stderr);
                }
                if (total == (size_t)p.num_atoms * (size_t)count) {
                    arr = arr.reshape({ p.num_atoms, count });
                }
            }
            ap[name.c_str()] = arr;
        }
    for (auto& kv : p.ibuf) {
            const std::string& name = kv.first;
            int32_t* raw = kv.second.release();
            size_t total = p.totals[name];
            py::capsule c(raw, [](void* p){ delete[] static_cast<int32_t*>(p); });
            py::array_t<int32_t> arr({ (py::ssize_t) total }, raw, c);
            int count = 1;
            for (auto& d : p.props) if (d.name == name) { count = d.count; break; }
            if (count > 1) {
                if (dbg && (size_t)p.num_atoms * (size_t)count != total) {
                    std::fprintf(stderr, "[fastxyz] reshape mismatch I: total=%zu != %zu*%d\n", total, (size_t)p.num_atoms, count);
                    std::fflush(stderr);
                }
                if (total == (size_t)p.num_atoms * (size_t)count) {
                    arr = arr.reshape({p.num_atoms, count});
                }
            }
            ap[name.c_str()] = arr;
        }
    for (auto& kv : p.lbuf) {
            const std::string& name = kv.first;
            uint8_t* raw = kv.second.release();
            size_t total = p.totals[name];
            py::capsule c(raw, [](void* p){ delete[] static_cast<uint8_t*>(p); });
            py::array_t<uint8_t> arr({ (py::ssize_t) total }, raw, c);
            int count = 1;
            for (auto& d : p.props) if (d.name == name) { count = d.count; break; }
            if (count > 1) {
                if (dbg && (size_t)p.num_atoms * (size_t)count != total) {
                    std::fprintf(stderr, "[fastxyz] reshape mismatch L: total=%zu != %zu*%d\n", total, (size_t)p.num_atoms, count);
                    std::fflush(stderr);
                }
                if (total == (size_t)p.num_atoms * (size_t)count) {
                    arr = arr.reshape({p.num_atoms, count});
                }
            }
            ap[name.c_str()] = arr;
        }
        for (auto& kv : p.sprops) {
            const std::string& name = kv.first;
            const auto& vec = kv.second;
            // count can be >1; pack into list of lists or list
            int count = 1;
            for (auto& d : p.props) if (d.name == name) { count = d.count; break; }
            if (count == 1) {
                py::list lst;
                // If this is species and ID mode, build species_id instead of strings
                if (name == "species" && s_mode != SpeciesMode::STR) {
                    // global type_map buildup for this frame
                    std::unordered_map<std::string,int> local;
                    std::vector<std::string> type_map;
                    type_map.reserve(8);
                    py::array_t<int32_t> ids((py::ssize_t)vec.size());
                    auto m = ids.mutable_unchecked<1>();
                    for (py::ssize_t i = 0; i < m.shape(0); ++i) {
                        const std::string& sym = vec[static_cast<size_t>(i)];
                        auto it = local.find(sym);
                        int idx;
                        if (it == local.end()) { idx = (int)type_map.size(); local.emplace(sym, idx); type_map.push_back(sym); }
                        else { idx = it->second; }
                        m(i) = idx;
                    }
                    ap["species_id"] = ids;
                    // attach type_map into additional_fields
                    py::list tm;
                    for (auto& s : type_map) tm.append(py::str(s));
                    // merge into frame additional_fields after ap is set
                    // We'll set into frame-level additional_fields dict below
                    // Temporarily stash into ap under a reserved key
                    ap["__type_map__"] = tm;
                } else {
                    // Use intern cache to avoid constructing duplicate Python str objects
                    for (const auto& s : vec) {
                        auto it = s_intern_cache.find(s);
                        if (it == s_intern_cache.end()) {
                            py::str ps(s);
                            it = s_intern_cache.emplace(s, ps).first;
                        }
                        lst.append(it->second);
                    }
                    ap[name.c_str()] = lst;
                }
            } else {
                // shape (num_atoms, count)
                py::list outer;
                for (int i = 0; i < p.num_atoms; ++i) {
                    py::list inner;
                    for (int k = 0; k < count; ++k) {
                        const auto& s = vec[static_cast<size_t>(i) * count + k];
                        auto it = s_intern_cache.find(s);
                        if (it == s_intern_cache.end()) {
                            py::str ps(s);
                            it = s_intern_cache.emplace(s, ps).first;
                        }
                        inner.append(it->second);
                    }
                    outer.append(inner);
                }
                ap[name.c_str()] = outer;
            }
        }
        frame["atomic_properties"] = ap;

        // additional_fields (convert now under GIL)
        py::dict add;
        for (auto& kv : p.add) {
            const auto& key = kv.first;
            const auto& val = kv.second;
            switch (val.type) {
                case AddType::DOUBLE: add[key.c_str()] = py::float_(val.d); break;
                case AddType::FLOATS: {
                    py::array_t<float> arr(val.vf.size());
                    std::memcpy(arr.mutable_data(), val.vf.data(), val.vf.size()*sizeof(float));
                    add[key.c_str()] = std::move(arr);
                    break;
                }
                case AddType::STRING: default:
                    add[key.c_str()] = py::str(val.s);
                    break;
            }
        }
        // propagate type_map if species_id was created
        if (ap.contains("__type_map__")) {
            add["type_map"] = ap["__type_map__"];
            ap.attr("pop")("__type_map__");
        }
        frame["additional_fields"] = add;

        out.append(frame);
        cooperative_python_yield(fi + 1);
        if (dbg) {
            std::fprintf(stderr, "[fastxyz] convert frame %zu done\n", fi);
            std::fflush(stderr);
        }
    }

    auto t_cvt1 = clock::now();

    if (dbg) {
        double t_index = std::chrono::duration<double, std::milli>(t_idx1 - t_idx0).count();
        double t_parse = std::chrono::duration<double, std::milli>(t_par1 - t_par0).count();
        double t_convert = std::chrono::duration<double, std::milli>(t_cvt1 - t_cvt0).count();
        double t_total = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        std::fprintf(stderr,
            "[fastxyz] bytes=%zu frames=%zu atoms=%zu threads=%d | index=%.2f ms parse=%.2f ms convert=%.2f ms total=%.2f ms\n",
            nbytes, frames.size(), total_atoms, nthreads,
            t_index, t_parse, t_convert, t_total);
        std::fflush(stderr);
    }

    // The parsed C++ graph can contain millions of temporary strings.  Free it
    // without monopolising the interpreter after the Python result is ready.
    {
        ScopedReleaseIfHeld release;
        std::vector<Parsed>().swap(parsed);
        std::vector<FrameIndex>().swap(frames);
    }
    return out;
}

static py::list index_only_impl(py::buffer bbuf) {
    py::buffer_info info = bbuf.request();
    if (info.ndim != 1 || info.itemsize != 1) {
        throw std::runtime_error("buffer must be 1D bytes");
    }
    const char* base = static_cast<const char*>(info.ptr);
    size_t nbytes = static_cast<size_t>(info.size);
    std::vector<FrameIndex> frames;
    {
        ScopedReleaseIfHeld release;
        frames = index_frames(base, nbytes);
    }
    py::list out;
    for (size_t index = 0; index < frames.size(); ++index) {
        const auto& fi = frames[index];
        py::dict d;
        d["offset_num"] = static_cast<py::ssize_t>(fi.off_num);
        d["offset_header"] = static_cast<py::ssize_t>(fi.off_header);
        d["offset_data"] = static_cast<py::ssize_t>(fi.off_data);
        d["end"] = static_cast<py::ssize_t>(fi.end);
        d["num_atoms"] = fi.num_atoms;
        out.append(d);
        cooperative_python_yield(index + 1);
    }
    return out;
}

PYBIND11_MODULE(_io, m) {
    m.doc() = "Fast EXTXYZ parser for NepTrainKit (mmap buffer input)";
    m.def("index_frames", &index_only_impl, "Return frame offsets from a memory buffer");
    m.def("parse_all", &parse_all_impl, py::arg("buffer"), py::arg("max_workers") = -1,
          "Parse all frames from an mmap-backed bytes-like object. Set NEPKIT_FASTXYZ_DEBUG=1 to print timings.");
}
