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
  for (size_t p = 0; p < kNumPrefixes; ++p) {
    for (size_t i = 0; i < kNumCategories; ++i) {
      histograms_[p][i] = writer_->register_instrument(
          InstrumentType::Histogram,
          std::string(kPrefixes[p]) + "_" + kNames[i] + "_nanoseconds", {});
      counters_[p][i] = writer_->register_instrument(
          InstrumentType::Counter,
          std::string(kPrefixes[p]) + "_" + kNames[i] + "_total", {});
    }
  }
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
      if (active_[stack_[i].category] > 0) --active_[stack_[i].category];
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

void NvtxRangeAggregator::on_start(uintptr_t domain, const char* name, uintptr_t id, bool has_id) {
  const char* cat = category(name);
  size_t index = kNumCategories;
  for (size_t i = 0; i < kNumCategories; ++i) if (std::strcmp(cat, kNames[i]) == 0) { index = i; break; }
  start_entry(domain, name, id, has_id, index);
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
  else start_entry(domain, name, id, has_id, index);
}

void NvtxRangeAggregator::start_entry(uintptr_t domain, const char* name, uintptr_t id,
                                      bool has_id, size_t index) {
  if (!writer_ || !domain || index >= kNumCategories) return;
  // NInfer creates and registers its domain before the injected library loads,
  // so CUPTI cannot report either event. The dedicated NInfer launcher makes
  // this process opt in to NInfer category IDs without a pre-observed domain.
  // Every other domain must have been observed being created, so a workload
  // cannot inject ranges into an adopted name by accident.
  uint8_t prefix = prefix_for_domain(domain);
  if (prefix == 0) {
    if (!adopt_existing_domain_ || domain == 0) return;
    prefix = 1;  // generic prefix: the domain was never named
  }
  Entry e{};
  e.domain = domain; e.id = id; e.has_id = has_id; e.start_ns = now_ns();
  e.category = static_cast<uint8_t>(index);
  e.prefix = static_cast<uint8_t>(prefix - 1);
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
  e.record = active_[index] == 0;
  ++active_[index];
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
      if (active_[e.category] > 0) --active_[e.category];
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
  const size_t index = entry.category;
  const size_t prefix = entry.prefix;
  if (index >= kNumCategories || prefix >= kNumPrefixes) { drop(); return; }
  try {
    MetricsWriter::record(histograms_[prefix][index], duration_ns);
    MetricsWriter::add(counters_[prefix][index], 1);
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
  depth_ = 0; dropped_ = 0;
  for (auto& count : active_) count = 0;
  for (auto& p : histograms_) for (auto& q : p) q = nullptr;
  for (auto& p : counters_) for (auto& q : p) q = nullptr;
}

}  // namespace graphsignal
