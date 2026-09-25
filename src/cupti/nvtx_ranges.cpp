#include "nvtx_ranges.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace graphsignal {
namespace {
constexpr const char* kNames[NvtxRangeAggregator::kNumCategories] = {
    "runtime", "engine", "prefill", "decode", "mtp", "dflash", "cuda_graph", "vision", "moe"};
}

void NvtxRangeAggregator::configure(MetricsWriter* writer, bool adopt_existing_domain) {
  writer_ = writer;
  adopt_existing_domain_ = adopt_existing_domain;
  if (!writer_) return;
  dropped_counter_ = writer_->register_instrument(
      InstrumentType::Counter, "ninfer_ranges_dropped_total", {});
  for (size_t i = 0; i < kNumCategories; ++i) {
    histograms_[i] = writer_->register_instrument(InstrumentType::Histogram,
                                                    std::string("ninfer_") + kNames[i] + "_nanoseconds", {});
    counters_[i] = writer_->register_instrument(InstrumentType::Counter,
                                                 std::string("ninfer_") + kNames[i] + "_total", {});
  }
}

void NvtxRangeAggregator::on_domain_create(uintptr_t domain, const char* name) {
  if (!name || std::strcmp(name, "ninfer") != 0) return;
  uintptr_t expected = 0;
  if (domain) ninfer_domain_.compare_exchange_strong(
      expected, domain, std::memory_order_release, std::memory_order_relaxed);
}

void NvtxRangeAggregator::on_register(uintptr_t domain, uintptr_t handle, const char* name) {
  if (!writer_ || domain != ninfer_domain_.load(std::memory_order_acquire) || !handle || !name) return;
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
  uintptr_t expected = domain;
  ninfer_domain_.compare_exchange_strong(expected, 0, std::memory_order_release,
                                        std::memory_order_relaxed);
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
  const uintptr_t known_domain = ninfer_domain_.load(std::memory_order_acquire);
  if (!writer_ || !domain || index >= kNumCategories) return;
  // NInfer creates and registers its domain before the injected library loads,
  // so CUPTI cannot report either event. The dedicated NInfer launcher makes
  // this process opt in to NInfer category IDs without a pre-observed domain.
  // Generic workloads keep strict handle matching.
  if (!adopt_existing_domain_ && (known_domain == 0 || known_domain != domain)) return;
  Entry e{};
  e.domain = domain; e.id = id; e.has_id = has_id; e.start_ns = now_ns();
  e.category = static_cast<uint8_t>(index);
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
  if (index >= kNumCategories) { drop(); return; }
  try {
    MetricsWriter::record(histograms_[index], duration_ns);
    MetricsWriter::add(counters_[index], 1);
  } catch (...) {}
}

void NvtxRangeAggregator::reset() {
  writer_ = nullptr; dropped_counter_ = nullptr; adopt_existing_domain_ = false;
  ninfer_domain_.store(0, std::memory_order_release);
  { std::lock_guard<std::mutex> lock(registered_mu_); for (auto& item : registered_) item = Registered{}; }
  { std::lock_guard<std::mutex> lock(async_mu_); for (auto& item : async_) item = Entry{}; }
  depth_ = 0; dropped_ = 0;
  for (auto& count : active_) count = 0;
  for (auto& p : histograms_) p = nullptr;
  for (auto& p : counters_) p = nullptr;
}

}  // namespace graphsignal
