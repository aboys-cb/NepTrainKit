// Fast EXTXYZ parsing with mmap buffer input
// Build with pybind11; exposed as NepTrainKit.core._fastxyz

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <system_error>

// Use single-header fast_float colocated in this directory
#include "fast_float.h"
#define NEPKIT_HAVE_FAST_FLOAT 1

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

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
    if (ffres.ec == std::errc()) {
        p = q;
        return v_ff;
    }
#endif
    std::string token(p, q);
    char* e = nullptr;
    double v = std::strtod(token.c_str(), &e);
    p = q;
    return v;
}

inline bool parse_int_token(const char*& p, const char* end, int& out) {
    const char* s = p;
    bool neg = false;
    if (s < end && (*s == '+' || *s == '-')) { neg = (*s == '-'); ++s; }
    long long val = 0;
    const char* start = s;
    while (s < end && std::isdigit(static_cast<unsigned char>(*s))) { val = val * 10 + (*s - '0'); ++s; }
    if (s == start) return false;
    out = static_cast<int>(neg ? -val : val);
    p = s;
    return true;
}

inline bool parse_bool_token(const char*& p, const char* end, uint8_t& out) {
    const char* s = p;
    const char* q = s;
    while (q < end && !std::isspace(static_cast<unsigned char>(*q))) ++q;
    size_t len = static_cast<size_t>(q - s);
    bool v = false;
    if (len == 1) {
        v = (*s == '1' || *s == 'T' || *s == 't');
    } else {
        // Compare lowercase
        bool is_true = (len == 4 && (s[0]=='t'||s[0]=='T') && (s[1]=='r'||s[1]=='R') && (s[2]=='u'||s[2]=='U') && (s[3]=='e'||s[3]=='E'));
        v = is_true;
    }
    out = v ? 1 : 0;
    p = q;
    return true;
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
        try { d.count = std::stoi(count); } catch (...) { d.count = 1; }
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
        bool quoted = false;
        if (p < e && *p == '"') {
            quoted = true;
            ++p;
            const char* v0 = p;
            while (p < e && *p != '"') ++p;
            value.assign(v0, p);
            if (p < e && *p == '"') ++p;
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
                } else if (key == "pbc" || key == "PBC") {
                    AddValue v; v.type = AddType::STRING; v.s = value;
                    add_out["pbc"] = std::move(v);
                } else if (key == "virial" || key == "stress" || key == "Virial" || key == "Stress"  || key == "VIRIAL" || key == "STRESS") {
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
                } else if (key == "Weight" || key == "weight") {
                   AddValue v; v.type = AddType::DOUBLE;
                    try { v.d = std::stod(value); } catch (...) { v.type = AddType::STRING; v.s = value; }
                    add_out["weight"] = std::move(v);
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
        if (!parse_int(l1, e1, num, &after)) { // skip invalid line
            if (e1 < end) { p = e1 + 1; continue; } else break;
        }
        // line 2: header
        const char* l2 = (e1 < end) ? e1 + 1 : end;
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

// Parallel frame indexing: split buffer into coarse chunks, align to line boundaries,
// and scan frame starts within each chunk. Header and data line scans may cross chunk
// boundaries; a frame is only emitted by the chunk that contains its first line.
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

static std::vector<FrameIndex> index_frames_parallel(const char* buf, size_t nbytes, int max_workers) {
    const char* gend = buf + nbytes;
    size_t chunk_bytes = 32ull * 1024ull * 1024ull; // 32 MiB default
    if (const char* env = std::getenv("NEPKIT_FASTXYZ_CHUNK_MB")) {
        long mb = std::strtol(env, nullptr, 10);
        if (mb > 0) chunk_bytes = static_cast<size_t>(mb) * 1024ull * 1024ull;
    }
    if (chunk_bytes == 0) chunk_bytes = nbytes;
    size_t nchunks = (nbytes + chunk_bytes - 1) / chunk_bytes;

    int nthreads = compute_threads(max_workers);
    // Ensure at least as many chunks as threads to keep workers busy
    if (nchunks < static_cast<size_t>(nthreads)) {
        nchunks = static_cast<size_t>(nthreads);
        chunk_bytes = (nbytes + nchunks - 1) / nchunks;
    }

    std::vector<std::vector<FrameIndex>> parts(nchunks);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (ptrdiff_t ci = 0; ci < static_cast<ptrdiff_t>(nchunks); ++ci) {
        const size_t start = static_cast<size_t>(ci) * chunk_bytes;
        const size_t stop = std::min(nbytes, (static_cast<size_t>(ci) + 1) * chunk_bytes);
        const char* cs = buf + start;
        const char* ce = buf + stop;
        if (cs >= ce) {
            continue;
        }
        // Align chunk start to next line start (skip a partial line at the beginning)
        if (cs != buf) {
            const void* nl = std::memchr(cs, '\n', static_cast<size_t>(ce - cs));
            if (!nl) {
                continue; // no newline in this window; nothing to do
            }
            cs = static_cast<const char*>(nl) + 1;
        }
        std::vector<FrameIndex> local;
        const char* p = cs;
        while (p < ce) {
            const char* l1 = p;
            const char* e1 = find_eol(l1, gend);
            if (l1 == e1) { // blank line
                p = (e1 < gend ? e1 + 1 : e1);
                continue;
            }
            int num = 0; const char* after = nullptr;
            if (!parse_int(l1, e1, num, &after)) {
                // not a valid frame start; skip line
                p = (e1 < gend ? e1 + 1 : e1);
                continue;
            }
            // second line (header)
            const char* l2 = (e1 < gend) ? e1 + 1 : gend;
            const char* e2 = find_eol(l2, gend);
            const char* d0 = (e2 < gend) ? e2 + 1 : gend;
            const char* d = d0;
            for (int i = 0; i < num && d < gend; ++i) {
                d = find_eol(d, gend);
                if (d < gend) ++d;
            }
            FrameIndex fi;
            fi.off_num = static_cast<size_t>(l1 - buf);
            fi.off_header = static_cast<size_t>(l2 - buf);
            fi.off_data = static_cast<size_t>(d0 - buf);
            fi.end = static_cast<size_t>(d - buf);
            fi.num_atoms = num;
            local.push_back(fi);

            // Advance p to the frame end (may move beyond this chunk window)
            p = d;
            if (p < gend && p <= ce) {
                // no-op
            }
        }
        parts[static_cast<size_t>(ci)] = std::move(local);
    }

    // Merge and sort by start offset to ensure global ordering
    std::vector<FrameIndex> out;
    size_t total = 0; for (auto& v : parts) total += v.size();
    out.reserve(total);
    for (auto& v : parts) {
        out.insert(out.end(), v.begin(), v.end());
    }
    std::sort(out.begin(), out.end(), [](const FrameIndex& a, const FrameIndex& b){ return a.off_num < b.off_num; });
    return out;
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
        py::gil_scoped_release _nogil_idx;
        frames = index_frames_parallel(base, nbytes, max_workers);
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
    };

    std::vector<Parsed> parsed(frames.size());

    // Determine threads
    int nthreads = compute_threads(max_workers);
    if (dbg) { nthreads = 1; }

    // First, parse all frames in parallel into plain C++ storage
    auto t_par0 = clock::now();
    {
    py::gil_scoped_release _nogil;

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
            // skip malformed frame
            parsed[static_cast<size_t>(i)] = std::move(p);
            continue;
        }
        p.num_atoms = std::max(0, fi.num_atoms);

        const char* h0 = base + fi.off_header;
        const char* h1 = (fi.off_data > fi.off_header) ? base + fi.off_data - 1 : base + fi.off_header;
        if (h1 > h0 && *(h1-1) == '\r') --h1; // trim CR
        if (dbg) {
            std::fprintf(stderr, "[fastxyz] frame %lld header parse: num_atoms=%d header_len=%lld\n",
                         (long long)i, p.num_atoms, (long long)(h1 - h0));
            std::fflush(stderr);
        }
        parse_header_line(h0, h1, p.lattice, p.props, p.add);

        // ensure essential props exist; if none, synthesize species:S:1 pos:R:3
        if (p.props.empty()) {
            PropDesc d1; d1.name = "species"; d1.dtype = 'S'; d1.count = 1; p.props.push_back(d1);
            PropDesc d2; d2.name = "pos";     d2.dtype = 'R'; d2.count = 3; p.props.push_back(d2);
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
        for (; actual_rows < p.num_atoms && d < dend; ++actual_rows) {
            const char* l0 = d;
            const char* le = find_eol(l0, dend);
            const char* q = l0;
            // For speed, scan tokens sequentially across properties
            for (const auto& desc : p.props) {
                for (int k = 0; k < desc.count; ++k) {
                    q = skip_ws(q, le);
                    if (q >= le) break;
                    if (desc.dtype == 'R') {
                        double v = parse_double(q, le);
                        size_t idx = static_cast<size_t>(actual_rows) * static_cast<size_t>(desc.count) + static_cast<size_t>(k);
                        p.rbuf[desc.name][idx] = static_cast<float>(v);
                    } else if (desc.dtype == 'I') {
                        int v = 0; (void)parse_int_token(q, le, v);
                        size_t idx = static_cast<size_t>(actual_rows) * static_cast<size_t>(desc.count) + static_cast<size_t>(k);
                        p.ibuf[desc.name][idx] = v;
                    } else if (desc.dtype == 'L') {
                        uint8_t v = 0; (void)parse_bool_token(q, le, v);
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
            }
            d = (le < dend ? le + 1 : le);
        }
        // If file ended early, shrink num_atoms to the number of parsed rows
        if (actual_rows < p.num_atoms) {
            p.num_atoms = actual_rows;
            // also shrink totals for each property to reflect actual rows
            for (const auto& desc : p.props) {
                p.totals[desc.name] = static_cast<size_t>(p.num_atoms) * static_cast<size_t>(desc.count);
            }
        }

            parsed[static_cast<size_t>(i)] = std::move(p);
        }
    }

    auto t_par1 = clock::now();
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

    return out;
}

static py::list index_only_impl(py::buffer bbuf) {
    py::buffer_info info = bbuf.request();
    if (info.ndim != 1 || info.itemsize != 1) {
        throw std::runtime_error("buffer must be 1D bytes");
    }
    const char* base = static_cast<const char*>(info.ptr);
    size_t nbytes = static_cast<size_t>(info.size);
    auto frames = index_frames(base, nbytes);
    py::list out;
    for (auto& fi : frames) {
        py::dict d;
        d["offset_num"] = static_cast<py::ssize_t>(fi.off_num);
        d["offset_header"] = static_cast<py::ssize_t>(fi.off_header);
        d["offset_data"] = static_cast<py::ssize_t>(fi.off_data);
        d["end"] = static_cast<py::ssize_t>(fi.end);
        d["num_atoms"] = fi.num_atoms;
        out.append(d);
    }
    return out;
}

PYBIND11_MODULE(_fastxyz, m) {
    m.doc() = "Fast EXTXYZ parser for NepTrainKit (mmap buffer input)";
    m.def("index_frames", &index_only_impl, "Return frame offsets from a memory buffer");
    m.def("parse_all", &parse_all_impl, py::arg("buffer"), py::arg("max_workers") = -1,
          "Parse all frames from an mmap-backed bytes-like object. Set NEPKIT_FASTXYZ_DEBUG=1 to print timings.");
}
