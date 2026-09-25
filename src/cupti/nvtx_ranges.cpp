#include "nvtx_ranges.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace graphsignal {
namespace {
constexpr const char* kNames[NvtxRangeAggregator::kNumCategories] = {
    "runtime", "engine", "prefill", "decode", "mtp", "dflash", "cuda_graph", "vision", "moe"};
// Metric-name prefix per prefix index. NInfer keeps its own namespace; other
// adopted domains share the generic one.
constexpr const char* kPrefixes[NvtxRangeAggregator::kNumPrefixes] = {"ninfer", "nvtx"};

// True when `name` is listed in the comma-separated `list`.
bool name_listed(const char* list, const char* name) {
  if (!list || !*list || !name) return false;
  const size_t len = std::strlen(name);
  const char* p = list;
  while (*p) {
    const char* comma = std::strchr(p, ',');
    const size_t seg = comma ? static_cast<size_t>(comma - p) : std::strlen(p);
    if (seg == len && std::strncmp(p, name, len) == 0) return true;
    if (!comma) break;
    p = comma + 1;
  }
  return false;
}
}  // namespace

void NvtxRangeAggregator::configure(MetricsWriter* writer, bool adopt_existing_domain,
                                    const char* domains) {
  writer_ = writer;
  adopt_existing_domain_ = adopt_existing_domain;
  if (domains && *domains) {
    if (std::strcmp(domains, "all") == 0) {
      adopt_all_domains_ = true;
      adopt_default_domain_ = true;
    } else {
      std::strncpy(domain_filter_, domains, sizeof(domain_filter_) - 1);
      domain_filter_[sizeof(domain_filter_) - 1] = '\0';
      // "default" names the implicit domain that is never created via the API.
      adopt_default_domain_ = name_listed(domain_filter_, "default");
    }
  }
  if (!writer_) return;
  dropped_counter_ = writer_->register_instrument(
      InstrumentType::Counter, "ninfer_ranges_dropped_total", {});
  // NInfer's categories are its own closed vocabulary and keep the published
  // ninfer_* names.
  for (size_t i = 0; i < kNumCategories; ++i) {
    histograms_[i] = writer_->register_instrument(
        InstrumentType::Histogram, std::string("ninfer_") + kNames[i] + "_nanoseconds", {});
    counters_[i] = writer_->register_instrument(
        InstrumentType::Counter, std::string("ninfer_") + kNames[i] + "_total", {});
  }
  // Every other adopted domain lands in one family, with the domain and the
  // category as labels: the domain is a namespace chosen by the workload
  // (vllm, sglang, torch, ...), so it belongs in a label rather than in the
  // metric name, and one family means a third engine costs a label value
  // instead of nine more metric names.
  for (auto& slot : generic_) { slot = GenericSlot{}; }
}

// 0 = not adopted, else 1 + prefix index.
uint8_t NvtxRangeAggregator::prefix_for_domain(uintptr_t domain) const noexcept {
  // The default domain is never "created" through the API, so it cannot be
  // observed; it is adopted by configuration instead. Most engines that emit
  // NVTX without creating a domain of their own use it.
  if (domain == 0) return adopt_default_domain_ ? 2 : 0;
  const size_t n = num_domains_.load(std::memory_order_acquire);
  for (size_t i = 0; i < n && i < kMaxAdoptedDomains; ++i) {
    if (domains_[i].handle == domain) return static_cast<uint8_t>(1 + domains_[i].prefix);
  }
  return 0;
}

void NvtxRangeAggregator::on_domain_create(uintptr_t domain, const char* name) {
  if (!domain || !name) return;
  // NInfer is adopted by default. Any other domain needs to be named, unless
  // the process opted into every domain.
  const bool is_ninfer = std::strcmp(name, "ninfer") == 0;
  if (!is_ninfer && !adopt_all_domains_ && !name_listed(domain_filter_, name)) return;

  size_t n = num_domains_.load(std::memory_order_acquire);
  for (size_t i = 0; i < n && i < kMaxAdoptedDomains; ++i) {
    if (domains_[i].handle == domain) return;  // already adopted
  }
  if (n >= kMaxAdoptedDomains) { drop(); return; }
  domains_[n].handle = domain;
  domains_[n].prefix = is_ninfer ? 0 : 1;
  // Keep the name for the metric label: the handle is not stable across runs.
  copy_name(domains_[n].name, name);
  num_domains_.store(n + 1, std::memory_order_release);
}

void NvtxRangeAggregator::on_register(uintptr_t domain, uintptr_t handle, const char* name) {
  if (!writer_ || !prefix_for_domain(domain) || !handle || !name) return;
  try {
    std::lock_guard<std::mutex> lock(registered_mu_);
    for (auto& item : registered_) {
      if (item.handle == 0) {
        item.handle = handle;
        std::strncpy(item.name, name, kMaxNameBytes - 1);
        return;
      }
    }
    drop();
  } catch (...) { drop(); }
}

const char* NvtxRangeAggregator::resolve_registered(uintptr_t handle) const {
  std::lock_guard<std::mutex> lock(registered_mu_);
  for (const auto& item : registered_) if (item.handle == handle) return item.name;
  return nullptr;
}

void NvtxRangeAggregator::on_start_registered(uintptr_t domain, uintptr_t handle,
                                               uintptr_t id, bool has_id) {
  const char* name = resolve_registered(handle);
  if (name) on_start(domain, name, id, has_id);
}

void NvtxRangeAggregator::on_domain_destroy(uintptr_t domain) {
  // Compact the adopted-domain list rather than clearing it: other domains in
  // the same process keep reporting.
  const size_t n = num_domains_.load(std::memory_order_acquire);
  size_t keep = 0;
  for (size_t i = 0; i < n && i < kMaxAdoptedDomains; ++i) {
    if (domains_[i].handle == domain) continue;
    if (keep != i) domains_[keep] = domains_[i];
    ++keep;
  }
  for (size_t i = keep; i < n && i < kMaxAdoptedDomains; ++i) domains_[i] = Domain{};
  num_domains_.store(keep, std::memory_order_release);
  for (size_t i = 0; i < depth_; ++i) {
    if (stack_[i].domain == domain) {
      if (active_[stack_[i].slot] > 0) --active_[stack_[i].slot];
      stack_[i] = Entry{};
    }
  }
}

const char* NvtxRangeAggregator::category(const char* name) {
  if (!name) return "other";
  std::string lower;
  try { lower.reserve(std::strlen(name)); } catch (...) { return "other"; }
  for (const char* p = name; *p && p - name < static_cast<ptrdiff_t>(kMaxNameBytes); ++p) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
  }
  for (size_t i = 0; i < kNumCategories; ++i) {
    const char* needle = kNames[i];
    if (i == 6) needle = "cuda graph";
    if (lower.find(needle) != std::string::npos) return kNames[i];
  }
  return "other";
}

void NvtxRangeAggregator::copy_name(char* dst, const char* src) noexcept {
  std::memset(dst, 0, kMaxNameBytes);
  if (!src) return;
  size_t i = 0;
  for (; src[i] && i + 1 < kMaxNameBytes; ++i) {
    unsigned char c = static_cast<unsigned char>(src[i]);
    dst[i] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '_';
  }
}

// The label value for a domain: its name when we observed it being created,
// "default" for the implicit one, and "adopted" for a domain the launcher
// opted in without CUPTI ever reporting it. The handle is never the value —
// handles are not stable across runs.
std::string NvtxRangeAggregator::domain_name(uintptr_t domain) const {
  if (domain == 0) return std::string("default");
  const size_t n = num_domains_.load(std::memory_order_acquire);
  for (size_t i = 0; i < n && i < kMaxAdoptedDomains; ++i) {
    if (domains_[i].handle == domain && domains_[i].name[0] != '\0') {
      return std::string(domains_[i].name);
    }
  }
  return std::string("adopted");
}

// A range name is a category for anyone but NInfer. Reduce it to a bounded,
// tag-safe label value: printable ASCII, capped, and the whole name is not
// allowed to become a cardinality explosion.
static std::string sanitize_category(const char* name) {
  std::string out;
  if (!name) return std::string("other");
  for (const char* p = name; *p && out.size() + 1 < NvtxRangeAggregator::kMaxCategoryBytes; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    out.push_back((c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '_');
  }
  if (out.empty()) return std::string("other");
  return out;
}

size_t NvtxRangeAggregator::generic_slot(uintptr_t domain, const char* category) noexcept {
  const std::string cat = sanitize_category(category);
  try {
    std::lock_guard<std::mutex> lock(generic_mu_);
    size_t free_index = kMaxGenericSeries;
    for (size_t i = 0; i < kMaxGenericSeries; ++i) {
      if (generic_[i].histogram != nullptr) {
        if (generic_[i].domain == domain && std::strcmp(generic_[i].category, cat.c_str()) == 0) {
          return kNumCategories + i;
        }
      } else if (free_index == kMaxGenericSeries) {
        free_index = i;
      }
    }
    if (free_index == kMaxGenericSeries) {
      // Out of room: fold into a per-domain "other" so the data still lands
      // somewhere attributable instead of being dropped entirely.
      for (size_t i = 0; i < kMaxGenericSeries; ++i) {
        if (generic_[i].domain == domain && std::strcmp(generic_[i].category, "other") == 0) {
          return kNumCategories + i;
        }
      }
      drop();
      return kNumSlots;
    }
    GenericSlot& slot = generic_[free_index];
    slot.domain = domain;
    std::strncpy(slot.category, cat.c_str(), kMaxCategoryBytes - 1);
    slot.category[kMaxCategoryBytes - 1] = '\0';
    // Labels: the domain (namespace chosen by the workload) and the range
    // name that acts as the category within it. The domain's *name* is the
    // label value, not its runtime handle: handles are not stable across runs.
    std::vector<std::pair<std::string, std::string>> tags;
    tags.emplace_back("domain", domain_name(domain));
    tags.emplace_back("category", cat);
    slot.histogram = writer_->register_instrument(InstrumentType::Histogram,
                                                  "nvtx_range_nanoseconds", tags);
    slot.counter = writer_->register_instrument(InstrumentType::Counter,
                                                "nvtx_ranges_total", tags);
    if (slot.histogram == nullptr || slot.counter == nullptr) {
      slot = GenericSlot{};
      drop();
      return kNumSlots;
    }
    return kNumCategories + free_index;
  } catch (...) {
    drop();
    return kNumSlots;
  }
}

void NvtxRangeAggregator::on_start(uintptr_t domain, const char* name, uintptr_t id, bool has_id) {
  // NInfer publishes a category enum, so its ranges map onto a fixed
  // vocabulary. Anyone else describes their own phases, so the range's own name
  // is the category — routing another engine's names through NInfer's nine
  // words would drop nearly all of them.
  const uint8_t prefix = prefix_for_domain(domain);
  if (prefix == 2) {  // a named, adopted, non-NInfer domain
    start_entry(domain, name, id, has_id, generic_slot(domain, name), 1);
    return;
  }
  const char* cat = category(name);
  size_t index = kNumCategories;
  for (size_t i = 0; i < kNumCategories; ++i) if (std::strcmp(cat, kNames[i]) == 0) { index = i; break; }
  start_entry(domain, name, id, has_id, index, 0);
}

void NvtxRangeAggregator::on_start_category(uintptr_t domain, uint32_t ninfer_category,
                                             const char* name, uintptr_t id, bool has_id) {
  // These values are the stable Category enum in ninfer::nvtx, not request data.
  size_t index = kNumCategories;
  switch (ninfer_category) {
    case 1: index = 0; break;  // Runtime
    case 2: index = 2; break;  // Prefill
    case 3: index = 3; break;  // Decode
    case 4: index = 4; break;  // Mtp
    case 5: index = 5; break;  // DFlash
    case 9: index = 8; break;  // Moe
    case 11: index = 6; break; // Graph
    case 12: index = 7; break; // Vision
    default: break;
  }
  if (index == kNumCategories) on_start(domain, name, id, has_id);
  else start_entry(domain, name, id, has_id, index, /*prefix=*/0);
}

void NvtxRangeAggregator::start_entry(uintptr_t domain, const char* name, uintptr_t id,
                                      bool has_id, size_t slot, uint8_t prefix) {
  if (!writer_ || !domain || slot >= kNumSlots) return;
  // NInfer creates and registers its domain before the injected library loads,
  // so CUPTI cannot report either event. The dedicated NInfer launcher makes
  // this process opt in to NInfer category IDs without a pre-observed domain.
  // Every other domain must have been observed being created, so a workload
  // cannot inject ranges into an adopted name by accident.
  if (prefix == 0) {
    uint8_t known = prefix_for_domain(domain);
    if (known == 0) {
      if (!adopt_existing_domain_ || domain == 0) return;
      // Adopted-but-unnamed (the NInfer launcher's pre-observed domain): it
      // still speaks NInfer's category vocabulary.
      prefix = 0;
    } else if (known == 2) {
      return;  // a generic domain that reached here with an NInfer slot
    }
  }
  Entry e{};
  e.domain = domain; e.id = id; e.has_id = has_id; e.start_ns = now_ns();
  e.slot = static_cast<uint16_t>(slot);
  e.prefix = prefix;
  if (has_id) {
    std::lock_guard<std::mutex> lock(async_mu_);
    Entry* slot = nullptr;
    for (auto& candidate : async_) if (candidate.domain == 0) { slot = &candidate; break; }
    if (!slot) { drop(); return; }
    // Async ranges may legitimately overlap across requests/threads. They are
    // independent lifetimes rather than nested scopes, so record each one.
    e.record = true;
    copy_name(e.name, name);
    *slot = e;
    return;
  }
  if (depth_ >= kMaxDepth) { drop(); return; }
  e.record = active_[slot] == 0;
  ++active_[slot];
  copy_name(e.name, name);
  stack_[depth_++] = e;
}

void NvtxRangeAggregator::on_end(uintptr_t domain, uintptr_t id, bool has_id) {
  if (!writer_) return;
  if (has_id) {
    std::lock_guard<std::mutex> lock(async_mu_);
    for (auto& e : async_) {
      if (e.domain == domain && e.id == id) {
        const uint64_t end = now_ns();
        if (e.record && end >= e.start_ns) record(e, end - e.start_ns);
        e = Entry{};
        return;
      }
    }
    return;
  }
  if (depth_ == 0) return;
  size_t index = depth_;
  while (index > 0) {
    --index;
    const Entry& e = stack_[index];
    if (e.domain == domain && (!has_id || e.id == id)) {
      const uint64_t end = now_ns();
      if (e.record && end >= e.start_ns) record(e, end - e.start_ns);
      if (active_[e.slot] > 0) --active_[e.slot];
      // Remove only the matched entry; malformed/out-of-order end events must
      // not discard unrelated active ranges.
      for (size_t move = index; move + 1 < depth_; ++move) stack_[move] = stack_[move + 1];
      --depth_;
      return;
    }
  }
}

void NvtxRangeAggregator::drop() noexcept {
  ++dropped_;
  MetricsWriter::add(dropped_counter_, 1);
}

void NvtxRangeAggregator::record(const Entry& entry, uint64_t duration_ns) noexcept {
  const size_t slot = entry.slot;
  if (slot >= kNumSlots) { drop(); return; }
  try {
    if (entry.prefix == 0) {
      // NInfer: the published ninfer_<category>_* names.
      MetricsWriter::record(histograms_[slot], duration_ns);
      MetricsWriter::add(counters_[slot], 1);
      return;
    }
    // Any other domain: one family, labelled with the domain and the range
    // name. The slot was allocated when the range started.
    const size_t index = slot - kNumCategories;
    if (index >= kMaxGenericSeries) { drop(); return; }
    Instrument* histogram = nullptr;
    Instrument* counter = nullptr;
    {
      std::lock_guard<std::mutex> lock(generic_mu_);
      if (generic_[index].domain == entry.domain) {
        histogram = generic_[index].histogram;
        counter = generic_[index].counter;
      }
    }
    if (histogram == nullptr || counter == nullptr) { drop(); return; }
    MetricsWriter::record(histogram, duration_ns);
    MetricsWriter::add(counter, 1);
  } catch (...) {}
}

// A mark is a point event with no duration, so it is counted, not timed. One
// counter per distinct mark name, bounded: a workload that invents unbounded
// mark names (a request id in the name, say) hits the cap and is reported
// through the dropped counter rather than growing the metric set without limit.
void NvtxRangeAggregator::on_mark(uintptr_t domain, const char* name) noexcept {
  if (!writer_ || !prefix_for_domain(domain) || !name || !*name) return;
  Instrument* counter = nullptr;
  try {
    std::lock_guard<std::mutex> lock(marks_mu_);
    size_t free_slot = kMaxMarkNames;
    for (size_t i = 0; i < kMaxMarkNames; ++i) {
      if (std::strcmp(marks_[i].name, name) == 0) { counter = marks_[i].counter; break; }
      if (free_slot == kMaxMarkNames && marks_[i].counter == nullptr) free_slot = i;
    }
    if (counter == nullptr) {
      if (free_slot == kMaxMarkNames) { drop(); return; }
      // Register lazily: registration is the slow path, and a mark name seen
      // once should still be counted.
      const std::string metric = "nvtx_marks_total";
      std::vector<std::pair<std::string, std::string>> tags;
      tags.emplace_back("name", std::string(name));
      counter = writer_->register_instrument(InstrumentType::Counter, metric, tags);
      if (counter == nullptr) { drop(); return; }
      copy_name(marks_[free_slot].name, name);
      marks_[free_slot].counter = counter;
    }
  } catch (...) { drop(); return; }
  MetricsWriter::add(counter, 1);
}

void NvtxRangeAggregator::reset() {
  writer_ = nullptr; dropped_counter_ = nullptr; adopt_existing_domain_ = false;
  adopt_all_domains_ = false;
  adopt_default_domain_ = false;
  domain_filter_[0] = '\0';
  num_domains_.store(0, std::memory_order_release);
  for (auto& d : domains_) d = Domain{};
  { std::lock_guard<std::mutex> lock(registered_mu_); for (auto& item : registered_) item = Registered{}; }
  { std::lock_guard<std::mutex> lock(async_mu_); for (auto& item : async_) item = Entry{}; }
  { std::lock_guard<std::mutex> lock(marks_mu_); for (auto& m : marks_) { m.name[0] = '\0'; m.counter = nullptr; } }
  { std::lock_guard<std::mutex> lock(generic_mu_); for (auto& g : generic_) g = GenericSlot{}; }
  depth_ = 0; dropped_ = 0;
  for (auto& count : active_) count = 0;
  for (auto& p : histograms_) p = nullptr;
  for (auto& p : counters_) p = nullptr;
}

}  // namespace graphsignal
