/*
  Lightweight GPU kernel timing (CUDA/HIP) using events.
  Intended for debugging/performance profiling; avoid enabling in production runs.
*/

#pragma once

#include "error.cuh"
#include "gpu_macro.cuh"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef USE_HIP
using gpuEvent_t = hipEvent_t;
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#else
using gpuEvent_t = cudaEvent_t;
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#endif

class KernelTiming
{
public:
  struct Stat {
    double total_ms = 0.0;
    double max_ms = 0.0;
    long long count = 0;
  };

  KernelTiming() = default;
  KernelTiming(const KernelTiming&) = delete;
  KernelTiming& operator=(const KernelTiming&) = delete;

  ~KernelTiming()
  {
    for (auto& ev : event_pool_) {
      if (ev.created) {
        gpuEventDestroy(ev.ev);
        ev.created = false;
      }
    }
  }

  int begin(const char* name, gpuStream_t stream = 0)
  {
    Entry entry;
    entry.name = name ? name : "(null)";
    entry.start = acquire_event_();
    entry.stop = acquire_event_();
    CHECK(gpuEventRecord(entry.start, stream));
    entries_.push_back(entry);
    return static_cast<int>(entries_.size() - 1);
  }

  void end(const int token, gpuStream_t stream = 0)
  {
    if (token < 0 || token >= static_cast<int>(entries_.size())) {
      return;
    }
    CHECK(gpuEventRecord(entries_[token].stop, stream));
  }

  void flush()
  {
    if (entries_.empty()) {
      return;
    }
    // Make sure everything in the stream up to the last recorded stop event is complete.
    CHECK(gpuEventSynchronize(entries_.back().stop));
    for (auto& e : entries_) {
      float ms = 0.0f;
      CHECK(gpuEventElapsedTime(&ms, e.start, e.stop));
      auto& st = stats_[e.name];
      st.total_ms += static_cast<double>(ms);
      st.max_ms = std::max(st.max_ms, static_cast<double>(ms));
      st.count += 1;
    }
    // Return events to pool for reuse.
    for (auto& e : entries_) {
      release_event_(e.start);
      release_event_(e.stop);
    }
    entries_.clear();
  }

  void reset() { stats_.clear(); }

  bool empty() const { return stats_.empty(); }

  void merge_from(const KernelTiming& other)
  {
    for (const auto& kv : other.stats_) {
      auto& st = stats_[kv.first];
      st.total_ms += kv.second.total_ms;
      st.max_ms = std::max(st.max_ms, kv.second.max_ms);
      st.count += kv.second.count;
    }
  }

  double total_ms() const
  {
    double sum = 0.0;
    for (const auto& kv : stats_) {
      sum += kv.second.total_ms;
    }
    return sum;
  }

  void print_top(const char* title, int topk) const
  {
    if (stats_.empty()) {
      printf("[kernel_timing] %s: (no data)\n", title ? title : "(null)");
      return;
    }
    if (topk <= 0) topk = 20;
    std::vector<std::pair<std::string, Stat>> items;
    items.reserve(stats_.size());
    for (const auto& kv : stats_) {
      items.push_back(kv);
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
      return a.second.total_ms > b.second.total_ms;
    });

    const double total = total_ms();
    printf("[kernel_timing] %s total_gpu_ms=%.3f kernels=%zu\n", title ? title : "(null)", total, items.size());
    const int n = std::min<int>(topk, static_cast<int>(items.size()));
    for (int i = 0; i < n; ++i) {
      const auto& name = items[i].first;
      const auto& st = items[i].second;
      const double pct = (total > 0.0) ? (100.0 * st.total_ms / total) : 0.0;
      const double avg = (st.count > 0) ? (st.total_ms / static_cast<double>(st.count)) : 0.0;
      printf(
        "  %6.2f%%  total=%8.3f ms  avg=%7.3f ms  max=%7.3f ms  count=%lld  %s\n",
        pct,
        st.total_ms,
        avg,
        st.max_ms,
        st.count,
        name.c_str());
    }
  }

private:
  struct PoolEvent {
    gpuEvent_t ev{};
    bool created = false;
    bool in_use = false;
  };

  struct Entry {
    std::string name;
    gpuEvent_t start{};
    gpuEvent_t stop{};
  };

  gpuEvent_t acquire_event_()
  {
    for (auto& pe : event_pool_) {
      if (!pe.in_use) {
        if (!pe.created) {
          CHECK(gpuEventCreate(&pe.ev));
          pe.created = true;
        }
        pe.in_use = true;
        return pe.ev;
      }
    }
    PoolEvent pe;
    CHECK(gpuEventCreate(&pe.ev));
    pe.created = true;
    pe.in_use = true;
    event_pool_.push_back(pe);
    return pe.ev;
  }

  void release_event_(gpuEvent_t ev)
  {
    for (auto& pe : event_pool_) {
      if (pe.created && pe.ev == ev) {
        pe.in_use = false;
        return;
      }
    }
  }

  std::vector<Entry> entries_;
  std::unordered_map<std::string, Stat> stats_;
  std::vector<PoolEvent> event_pool_;
};

