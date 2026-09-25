#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>

#include "../common/metrics_writer.h"

namespace graphsignal {

// Bounded, allocation-free-on-the-hot-path aggregator for the NInfer NVTX
// domain. CUPTI supplies a domain handle rather than the domain name on range
// callbacks, so the adapter records domain creation/destruction here.
//
// Scoped push/pop ranges record the outermost active range of each category on
// their thread. Asynchronous ranges are independent lifecycles and may overlap
// across requests, so every completed async range is recorded. Cross-category
// nesting remains visible (for example runtime -> prefill -> decode) while
// same-category nesting is suppressed for scoped ranges.
class NvtxRangeAggregator {
 public:
  static constexpr size_t kMaxDepth = 64;
  static constexpr size_t kMaxNameBytes = 96;
  static constexpr size_t kNumCategories = 9;

  void configure(MetricsWriter* writer, bool adopt_existing_domain = false);
  void on_domain_create(uintptr_t domain, const char* name);
  void on_domain_destroy(uintptr_t domain);
  void on_register(uintptr_t domain, uintptr_t handle, const char* name);
  void on_start(uintptr_t domain, const char* name, uintptr_t id, bool has_id = true);
  void on_start_category(uintptr_t domain, uint32_t category, const char* name,
                         uintptr_t id, bool has_id);
  void on_start_registered(uintptr_t domain, uintptr_t handle, uintptr_t id, bool has_id);
  void on_end(uintptr_t domain, uintptr_t id, bool has_id);
  void reset();

  const char* resolve_registered(uintptr_t handle) const;

  // Exposed for deterministic parser tests.
  static const char* category(const char* name);

 private:
  struct Entry {
    uintptr_t domain; uintptr_t id; uint64_t start_ns; bool has_id;
    uint8_t category; bool record; char name[kMaxNameBytes];
  };
  void record(const Entry& entry, uint64_t duration_ns) noexcept;
  void start_entry(uintptr_t domain, const char* name, uintptr_t id, bool has_id, size_t index);
  void drop() noexcept;
  void copy_name(char* dst, const char* src) noexcept;

  MetricsWriter* writer_ = nullptr;
  Instrument* histograms_[kNumCategories]{};
  Instrument* counters_[kNumCategories]{};
  Instrument* dropped_counter_ = nullptr;
  struct Registered { uintptr_t handle; char name[kMaxNameBytes]; };
  Registered registered_[256]{};
  mutable std::mutex registered_mu_;
  Entry async_[256]{};
  std::mutex async_mu_;
  std::atomic<uintptr_t> ninfer_domain_{0};
  bool adopt_existing_domain_ = false;
  // NVTX callbacks may arrive on many host threads. Keep the small stack and
  // category occupancy thread-local so recording does not serialize workers.
  inline static thread_local Entry stack_[kMaxDepth]{};
  inline static thread_local size_t depth_ = 0;
  inline static thread_local uint8_t active_[kNumCategories]{};
  std::atomic<uint64_t> dropped_{0};
};

}  // namespace graphsignal
