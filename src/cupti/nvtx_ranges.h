#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>

#include "../common/metrics_writer.h"

namespace graphsignal {

// Bounded, allocation-free-on-the-hot-path aggregator for NVTX domains. CUPTI
// supplies a domain handle rather than the domain name on range callbacks, so
// the adapter records domain creation/destruction here.
//
// Domains are opt-in. The NInfer domain is adopted by default (its launcher also
// opts the process in, because NInfer creates the domain before this library
// loads); GRAPHSIGNAL_NVTX_DOMAINS adopts others by name, or every domain with
// "all". The NInfer domain keeps the `ninfer_*` metric names; any other adopted
// domain reports the same category structure under `nvtx_*`, so a phase breakdown
// of vLLM/SGLang/TensorRT-LLM lands in the familiar shape without being
// mislabelled as NInfer's.
//
// Scoped push/pop ranges record the outermost active range of each category on
// their thread. Asynchronous ranges are independent lifecycles and may overlap
// across requests, so every completed async range is recorded. Cross-category
// nesting remains visible (for example runtime -> prefill -> decode) while
// same-category nesting is suppressed for scoped ranges. Marks are point events
// with no duration, so they are counted rather than timed.
class NvtxRangeAggregator {
 public:
  static constexpr size_t kMaxDepth = 64;
  static constexpr size_t kMaxNameBytes = 96;
  static constexpr size_t kNumCategories = 9;
  static constexpr size_t kMaxAdoptedDomains = 16;
  static constexpr size_t kMaxMarkNames = 64;
  // 0 = the NInfer domain (ninfer_* names), 1 = any other adopted domain
  // (the generic nvtx_* family, which carries domain and category as labels).
  static constexpr size_t kNumPrefixes = 2;
  // Generic (non-NInfer) domains report each range under its own name, because
  // NInfer's nine-word vocabulary is its own and does not describe anyone
  // else's phases. Bounded, with overflow folded into "other".
  static constexpr size_t kMaxGenericSeries = 48;
  static constexpr size_t kMaxCategoryBytes = 48;
  // Slot space: the NInfer categories first, then the generic series.
  static constexpr size_t kNumSlots = kNumCategories + kMaxGenericSeries;

  // `domains` is the parsed GRAPHSIGNAL_NVTX_DOMAINS value: "all", a
  // comma-separated name list, or empty for the NInfer-only default.
  void configure(MetricsWriter* writer, bool adopt_existing_domain = false,
                 const char* domains = nullptr);
  void on_domain_create(uintptr_t domain, const char* name);
  void on_domain_destroy(uintptr_t domain);
  void on_register(uintptr_t domain, uintptr_t handle, const char* name);
  void on_start(uintptr_t domain, const char* name, uintptr_t id, bool has_id = true);
  void on_start_category(uintptr_t domain, uint32_t category, const char* name,
                         uintptr_t id, bool has_id);
  void on_start_registered(uintptr_t domain, uintptr_t handle, uintptr_t id, bool has_id);
  void on_end(uintptr_t domain, uintptr_t id, bool has_id);
  void on_mark(uintptr_t domain, const char* name) noexcept;
  void reset();

  const char* resolve_registered(uintptr_t handle) const;

  // Exposed for deterministic parser tests.
  static const char* category(const char* name);

 private:
  struct Entry {
    uintptr_t domain; uintptr_t id; uint64_t start_ns; bool has_id;
    uint16_t slot; uint8_t prefix; bool record;
    char name[kMaxNameBytes];
  };
  struct Domain {
    uintptr_t handle;
    uint8_t prefix;
    char name[kMaxCategoryBytes];
  };
  struct GenericSlot {
    uintptr_t domain;
    char category[kMaxCategoryBytes];
    Instrument* histogram;
    Instrument* counter;
  };
  struct MarkSlot {
    char name[kMaxNameBytes];
    Instrument* counter;
  };
  // 0 when the domain is not adopted, else 1 + prefix index.
  uint8_t prefix_for_domain(uintptr_t domain) const noexcept;
  // Label value for a domain: its name, or "default"/"adopted".
  std::string domain_name(uintptr_t domain) const;
  // Slot for a generic (non-NInfer) range, allocated on first sight of the
  // (domain, category) pair. kNumSlots means "no room", which records under
  // "other" instead of growing without bound.
  size_t generic_slot(uintptr_t domain, const char* category) noexcept;
  void record(const Entry& entry, uint64_t duration_ns) noexcept;
  void start_entry(uintptr_t domain, const char* name, uintptr_t id, bool has_id,
                   size_t slot, uint8_t prefix);
  void drop() noexcept;
  void copy_name(char* dst, const char* src) noexcept;

  MetricsWriter* writer_ = nullptr;
  Instrument* histograms_[kNumCategories]{};
  Instrument* counters_[kNumCategories]{};
  GenericSlot generic_[kMaxGenericSeries]{};
  std::mutex generic_mu_;
  Instrument* dropped_counter_ = nullptr;
  struct Registered { uintptr_t handle; char name[kMaxNameBytes]; };
  Registered registered_[256]{};
  mutable std::mutex registered_mu_;
  Entry async_[256]{};
  std::mutex async_mu_;
  Domain domains_[kMaxAdoptedDomains]{};
  std::atomic<size_t> num_domains_{0};
  bool adopt_all_domains_ = false;
  bool adopt_default_domain_ = false;
  char domain_filter_[256]{};  // comma-separated names, empty = ninfer only
  MarkSlot marks_[kMaxMarkNames]{};
  mutable std::mutex marks_mu_;
  bool adopt_existing_domain_ = false;
  // NVTX callbacks may arrive on many host threads. Keep the small stack and
  // category occupancy thread-local so recording does not serialize workers.
  inline static thread_local Entry stack_[kMaxDepth]{};
  inline static thread_local size_t depth_ = 0;
  inline static thread_local uint8_t active_[kNumSlots]{};
  std::atomic<uint64_t> dropped_{0};
};

}  // namespace graphsignal
