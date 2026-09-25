// Internal metric + writer layer shared by the CUPTI and ROCm injection
// libraries. Not part of the public probe API (that's include/graphsignal/).
//
// A Writer owns the library's own instruments (kernel/graph/memcpy/memset/
// sync histograms and counters), a retained log ring, the process context
// (rank/SLURM env), and a background thread that serializes everything —
// including user probes discovered via the probe.h registry — to
// `<base>/graphsignal_<pid>/<libname>.json` every interval, atomically
// (write tmp + rename). All values are cumulative since start_ts; the file is
// a full-state overwrite, so every read is an immutable snapshot.
//
// Runs inside the inference engine's process: it must never crash, hang, or
// slow the workload. No exceptions escape, memory is bounded, and the final
// shutdown write never calls the flush hook (no CUDA/ROCm on teardown).

#pragma once

#include <graphsignal/probe.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace graphsignal {

constexpr uint32_t kMaxInstruments = 4096;
constexpr uint32_t kMaxProfileFrames = 250;
constexpr uint32_t kMaxLogEntries = 256;
constexpr size_t kMaxLogLineBytes = 2048;
constexpr uint64_t kDefaultIntervalNs = 1000000000ull;

enum class InstrumentType { Gauge, Counter, Histogram, Profile };

struct Instrument {
  InstrumentType type;
  std::string name;
  std::vector<std::pair<std::string, std::string>> tags;

  // Guards all mutable fields below. These are not user-facing hot paths
  // (recorded from the activity-processing thread), so a mutex is fine and
  // gives the serializer torn-free snapshots.
  std::mutex mu;
  double gauge_value = 0.0;
  uint64_t counter_total = 0;
  uint64_t bins[GRAPHSIGNAL_PROBE_HIST_BINS] = {0};
  // Exact histogram aggregates next to the 25 %-wide bins: a reader that
  // needs totals (time per iteration = sum / iterations) or the true mean
  // cannot get them from bins; the probe.h device block already keeps them.
  uint64_t hist_count = 0;
  uint64_t hist_sum = 0;
  uint64_t hist_min = UINT64_MAX;
  uint64_t hist_max = 0;
  // Profile instruments: frame name -> cumulative value and how many
  // recordings contributed to it, capped at kMaxProfileFrames (new frames
  // past the cap are dropped).
  struct FrameAgg {
    uint64_t value = 0;
    uint64_t samples = 0;
  };
  std::unordered_map<std::string, FrameAgg> frames;
  uint64_t dropped_frames = 0;
};

inline uint64_t now_ns() {
  return graphsignal_now_ns();
}

inline void json_escape_append(std::string& out, const char* s, size_t max_len) {
  for (size_t i = 0; s[i] && i < max_len; i++) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
}

inline void json_escape_append(std::string& out, const std::string& s) {
  json_escape_append(out, s.c_str(), s.size());
}

inline void append_double(std::string& out, double v) {
  char buf[64];
  if (v == static_cast<double>(static_cast<int64_t>(v)) &&
      v >= -9.007199254740992e15 && v <= 9.007199254740992e15) {
    std::snprintf(buf, sizeof(buf), "%" PRId64, static_cast<int64_t>(v));
  } else {
    std::snprintf(buf, sizeof(buf), "%.17g", v);
  }
  out += buf;
}

inline void append_u64(std::string& out, uint64_t v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%" PRIu64, v);
  out += buf;
}

class MetricsWriter {
 public:
  using FlushHook = void (*)(void*);
  // Copies a DEVICE-storage probe's data block to host memory. Returns false
  // to skip the probe (e.g. device not initialized yet, or during teardown).
  using DeviceProbeReader = bool (*)(const graphsignal_probe_entry*,
                                     graphsignal_instrument_data*, void*);
  // Copies `n` consecutive device blocks starting at `first` (the data pointer
  // of one entry; the blocks of the run are contiguous in device memory) into
  // out[0..n). Returns false to skip the whole run. When set, the serializer
  // groups the registry's device probes into contiguous runs and issues one
  // copy per run instead of one per probe — a process with a thousand device
  // probes (one per SM per phase) then costs a few copies per write, not a
  // thousand driver round trips.
  using DeviceProbeBatchReader = bool (*)(const graphsignal_instrument_data* first,
                                          size_t n, graphsignal_instrument_data* out,
                                          void*);

  // Never throws; returns nullptr on failure (callers degrade to no-op).
  static MetricsWriter* init(const char* libname, const char* base_dir,
                      uint64_t interval_ns, bool debug_enabled) {
    try {
      MetricsWriter* w = new MetricsWriter();
      if (!w->setup(libname, base_dir, interval_ns, debug_enabled)) {
        delete w;
        return nullptr;
      }
      return w;
    } catch (...) {
      return nullptr;
    }
  }

  void set_flush_hook(FlushHook fn, void* arg) {
    std::lock_guard<std::mutex> g(hook_mu_);
    flush_hook_ = fn;
    flush_arg_ = arg;
  }

  void set_device_probe_reader(DeviceProbeReader fn, void* arg) {
    std::lock_guard<std::mutex> g(hook_mu_);
    device_reader_ = fn;
    device_reader_arg_ = arg;
  }

  void set_device_probe_batch_reader(DeviceProbeBatchReader fn, void* arg) {
    std::lock_guard<std::mutex> g(hook_mu_);
    device_batch_reader_ = fn;
    device_batch_reader_arg_ = arg;
  }

  // Statistics of the last serialize (debug log + tests): device probes seen,
  // contiguous runs copied, time spent reading device probes.
  uint64_t last_device_probes() const { return last_device_probes_.load(std::memory_order_relaxed); }
  uint64_t last_device_runs() const { return last_device_runs_.load(std::memory_order_relaxed); }
  uint64_t last_device_read_ns() const { return last_device_read_ns_.load(std::memory_order_relaxed); }

  // --- instruments (registration is idempotent per name+tags) ---

  Instrument* register_instrument(
      InstrumentType type, const std::string& name,
      std::vector<std::pair<std::string, std::string>> tags) {
    try {
      std::string key = name;
      key += '\x1f';
      for (const auto& tag : tags) {
        key += tag.first;
        key += '\x1e';
        key += tag.second;
        key += '\x1f';
      }

      std::lock_guard<std::mutex> g(reg_mu_);
      auto it = by_key_.find(key);
      if (it != by_key_.end()) return it->second;
      if (ordered_.size() >= kMaxInstruments) {
        dropped_instruments_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }
      Instrument* inst = new Instrument();
      inst->type = type;
      inst->name = name;
      inst->tags = std::move(tags);
      by_key_.emplace(std::move(key), inst);
      ordered_.push_back(inst);
      return inst;
    } catch (...) {
      return nullptr;
    }
  }

  // Histogram: observe one value.
  static void record(Instrument* inst, uint64_t value) {
    if (!inst) return;
    std::lock_guard<std::mutex> g(inst->mu);
    inst->bins[graphsignal_probe_bin_index(value)]++;
    inst->hist_count++;
    inst->hist_sum += value;
    if (value < inst->hist_min) inst->hist_min = value;
    if (value > inst->hist_max) inst->hist_max = value;
  }

  // Counter: add to the cumulative total.
  static void add(Instrument* inst, uint64_t delta) {
    if (!inst) return;
    std::lock_guard<std::mutex> g(inst->mu);
    inst->counter_total += delta;
  }

  // Gauge: set the current value.
  static void set(Instrument* inst, double value) {
    if (!inst) return;
    std::lock_guard<std::mutex> g(inst->mu);
    inst->gauge_value = value;
  }

  // Profile: add to a frame's cumulative value and count the recording.
  static void profile_add(Instrument* inst, const char* frame_name,
                          uint64_t delta) {
    if (!inst || !frame_name || !frame_name[0]) return;
    try {
      std::lock_guard<std::mutex> g(inst->mu);
      auto it = inst->frames.find(frame_name);
      if (it != inst->frames.end()) {
        it->second.value += delta;
        it->second.samples++;
        return;
      }
      if (inst->frames.size() >= kMaxProfileFrames) {
        inst->dropped_frames++;
        return;
      }
      inst->frames.emplace(frame_name, Instrument::FrameAgg{delta, 1});
    } catch (...) {
    }
  }

  // --- logging (retained ring, serialized on every write) ---

  void set_debug(bool enabled) { debug_enabled_.store(enabled, std::memory_order_relaxed); }
  bool debug_enabled() const { return debug_enabled_.load(std::memory_order_relaxed); }

  void debug(const char* fmt, ...) {
    if (!debug_enabled()) return;
    va_list args;
    va_start(args, fmt);
    log_v("debug", fmt, args);
    va_end(args);
  }

  void error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v("error", fmt, args);
    va_end(args);
  }

  // --- lifecycle ---

  // Serializes and writes the file immediately (used by the writer thread and
  // tests). Never throws.
  void write_now(bool call_hook) {
    try {
      if (call_hook) {
        FlushHook hook = nullptr;
        void* arg = nullptr;
        {
          std::lock_guard<std::mutex> g(hook_mu_);
          hook = flush_hook_;
          arg = flush_arg_;
        }
        if (hook) hook(arg);
      }
      std::string json = serialize();
      write_file(json);
    } catch (...) {
    }
  }

  // Final write (without the flush hook) + writer thread join. The Writer
  // object is deliberately leaked by callers: no destructor work on teardown.
  void shutdown() {
    bool expected = false;
    if (!stopped_.compare_exchange_strong(expected, true)) return;
    {
      std::lock_guard<std::mutex> g(cv_mu_);
      cv_.notify_all();
    }
    if (writer_thread_.joinable()) {
      try {
        writer_thread_.join();
      } catch (...) {
      }
    }
  }

  const std::string& file_path() const { return file_path_; }
  const std::string& dir_path() const { return dir_path_; }
  uint64_t start_ts() const { return start_ts_; }
  uint64_t dropped_instruments() const {
    return dropped_instruments_.load(std::memory_order_relaxed);
  }

 private:
  MetricsWriter() = default;

  bool setup(const char* libname, const char* base_dir, uint64_t interval_ns,
             bool debug_enabled) {
    if (!libname || !libname[0]) return false;
    const char* base = (base_dir && base_dir[0]) ? base_dir : "/dev/shm";

    start_ts_ = now_ns();
    interval_ns_ = interval_ns ? interval_ns : kDefaultIntervalNs;
    debug_enabled_.store(debug_enabled, std::memory_order_relaxed);
    pid_ = static_cast<uint64_t>(getpid());

    dir_path_ = std::string(base) + "/graphsignal_" + std::to_string(pid_);
    if (mkdir(dir_path_.c_str(), 0700) != 0 && errno != EEXIST) return false;
    file_path_ = dir_path_ + "/" + libname + ".json";
    tmp_path_ = file_path_ + ".tmp";

    capture_context();

    try {
      writer_thread_ = std::thread([this]() { writer_loop(); });
    } catch (...) {
      return false;
    }
    return true;
  }

  void writer_loop() {
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(cv_mu_);
        cv_.wait_for(lk, std::chrono::nanoseconds(interval_ns_),
                     [this]() { return stopped_.load(); });
      }
      bool stopping = stopped_.load();
      // The flush hook (e.g. cuptiActivityFlushAll) is never called on the
      // final write: no CUDA/ROCm calls on the teardown path.
      write_now(/*call_hook=*/!stopping);
      if (stopping) return;
    }
  }

  void log_v(const char* level, const char* fmt, va_list args) {
    char buf[kMaxLogLineBytes];
    va_list args_copy;
    va_copy(args_copy, args);
    std::vsnprintf(buf, sizeof(buf), fmt, args_copy);
    va_end(args_copy);

    std::fprintf(stderr, "graphsignal: %s\n", buf);

    try {
      std::lock_guard<std::mutex> g(log_mu_);
      LogEntry entry;
      entry.ts = now_ns();
      entry.level = level;
      entry.msg = buf;
      if (log_ring_.size() >= kMaxLogEntries) {
        log_ring_[log_head_ % kMaxLogEntries] = std::move(entry);
        log_head_++;
      } else {
        log_ring_.push_back(std::move(entry));
      }
    } catch (...) {
    }
  }

  void capture_context() {
    struct Field {
      const char* key;
      std::vector<const char*> vars;
    };
    const Field fields[] = {
        {"rank", {"RANK", "NCCL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"}},
        {"local_rank", {"LOCAL_RANK", "NCCL_LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"}},
        {"world_size", {"WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"}},
        {"local_world_size", {"LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE"}},
        {"master_addr", {"MASTER_ADDR"}},
        {"master_port", {"MASTER_PORT"}},
        {"slurm_job_id", {"SLURM_JOB_ID", "SLURM_JOBID"}},
        {"slurm_step_id", {"SLURM_STEP_ID", "SLURM_STEPID"}},
        {"slurm_node_id", {"SLURM_NODEID"}},
        {"slurm_node_count", {"SLURM_NNODES", "SLURM_JOB_NUM_NODES"}},
    };
    for (const Field& field : fields) {
      std::string value = read_first_env(field.vars);
      if (!value.empty()) context_.emplace_back(field.key, std::move(value));
    }
  }

  static std::string read_first_env(const std::vector<const char*>& names) {
    for (const char* name : names) {
      const char* v = std::getenv(name);
      if (v && *v) {
        // Sanitize: keep only printable ASCII, cap length, strip JSON-hostile chars.
        std::string out;
        out.reserve(64);
        for (size_t k = 0; v[k] && k < 256; ++k) {
          unsigned char c = static_cast<unsigned char>(v[k]);
          if (c >= 0x20 && c < 0x7f && c != '"' && c != '\\') out.push_back(static_cast<char>(c));
        }
        if (!out.empty()) return out;
      }
    }
    return {};
  }

  // --- serialization ---

  std::string serialize() {
    std::string j;
    j.reserve(16384);
    j += "{\"version\":1,\"pid\":";
    append_u64(j, pid_);
    j += ",\"start_ts\":";
    append_u64(j, start_ts_);
    j += ",\"write_ts\":";
    append_u64(j, now_ns());

    if (!context_.empty()) {
      j += ",\"context\":{";
      bool first = true;
      for (const auto& kv : context_) {
        if (!first) j += ',';
        first = false;
        j += '"';
        j += kv.first;
        j += "\":\"";
        j += kv.second;  // sanitized at capture
        j += '"';
      }
      j += '}';
    }

    j += ",\"metrics\":[";
    bool first_metric = true;
    serialize_instruments(j, first_metric);
    serialize_probes(j, first_metric);
    j += ']';

    serialize_log(j);

    j += '}';
    return j;
  }

  void serialize_instruments(std::string& j, bool& first_metric) {
    std::vector<Instrument*> instruments;
    {
      std::lock_guard<std::mutex> g(reg_mu_);
      instruments = ordered_;
    }
    for (Instrument* inst : instruments) {
      std::lock_guard<std::mutex> g(inst->mu);
      if (!first_metric) j += ',';
      first_metric = false;
      serialize_metric_head(j, inst->name.c_str(), type_str(inst->type));
      serialize_tags(j, inst->tags);
      switch (inst->type) {
        case InstrumentType::Gauge:
          j += ",\"value\":";
          append_double(j, inst->gauge_value);
          break;
        case InstrumentType::Counter:
          j += ",\"value\":";
          append_u64(j, inst->counter_total);
          break;
        case InstrumentType::Histogram:
          serialize_bins(j, inst->bins);
          serialize_hist_aggregates(j, inst->hist_count, inst->hist_sum, inst->hist_min, inst->hist_max);
          break;
        case InstrumentType::Profile:
          serialize_frames(j, inst->frames);
          break;
      }
      j += '}';
    }

    uint64_t writer_dropped = dropped_instruments();
    if (writer_dropped > 0) {
      if (!first_metric) j += ',';
      first_metric = false;
      serialize_metric_head(j, "graphsignal_writer_dropped_instruments", "counter");
      j += ",\"tags\":{},\"value\":";
      append_u64(j, writer_dropped);
      j += '}';
    }
  }

  // Device probes are read before serialization: sorted by device address,
  // grouped into runs of adjacent blocks, one batch copy per run (or one
  // single-probe copy each when only the per-probe reader is set). A probe
  // whose run failed to copy is skipped for this write, as before.
  struct DeviceRead {
    uint64_t index;    // registry index
    const graphsignal_instrument_data* dev;
    size_t slot;       // position in `host` after the copy
    bool ok;
  };

  void read_device_probes(const graphsignal_probe_registry* reg, uint64_t count,
                          std::vector<DeviceRead>& reads,
                          std::vector<graphsignal_instrument_data>& host) {
    DeviceProbeReader device_reader = nullptr;
    void* device_reader_arg = nullptr;
    DeviceProbeBatchReader batch_reader = nullptr;
    void* batch_reader_arg = nullptr;
    {
      std::lock_guard<std::mutex> g(hook_mu_);
      device_reader = device_reader_;
      device_reader_arg = device_reader_arg_;
      batch_reader = device_batch_reader_;
      batch_reader_arg = device_batch_reader_arg_;
    }
    reads.clear();
    for (uint64_t i = 0; i < count; i++) {
      const graphsignal_probe_entry* e = &reg->entries[i];
      if (e->type == GRAPHSIGNAL_PROFILE || e->storage != GRAPHSIGNAL_STORAGE_DEVICE || !e->data) continue;
      reads.push_back(DeviceRead{i, e->data, 0, false});
    }
    last_device_probes_.store(reads.size(), std::memory_order_relaxed);
    if (reads.empty() || (!device_reader && !batch_reader)) {
      last_device_runs_.store(0, std::memory_order_relaxed);
      last_device_read_ns_.store(0, std::memory_order_relaxed);
      return;
    }
    const uint64_t t0 = now_ns();
    host.resize(reads.size());
    uint64_t runs = 0;
    if (batch_reader) {
      std::vector<size_t> order(reads.size());
      for (size_t k = 0; k < order.size(); k++) order[k] = k;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return reads[a].dev < reads[b].dev; });
      size_t slot = 0;
      for (size_t k = 0; k < order.size();) {
        size_t m = k + 1;
        while (m < order.size() && reads[order[m]].dev == reads[order[m - 1]].dev + 1) m++;
        const size_t n = m - k;
        const bool ok = batch_reader(reads[order[k]].dev, n, host.data() + slot, batch_reader_arg);
        for (size_t q = k; q < m; q++) {
          reads[order[q]].slot = slot + (q - k);
          reads[order[q]].ok = ok;
        }
        slot += n;
        runs++;
        k = m;
      }
    } else {
      for (size_t k = 0; k < reads.size(); k++) {
        reads[k].slot = k;
        reads[k].ok = device_reader(&reg->entries[reads[k].index], &host[k], device_reader_arg);
        runs++;
      }
    }
    last_device_runs_.store(runs, std::memory_order_relaxed);
    last_device_read_ns_.store(now_ns() - t0, std::memory_order_relaxed);
    if (debug_enabled()) {
      debug("device probes: %llu read in %llu contiguous run(s), %llu us",
            static_cast<unsigned long long>(reads.size()), static_cast<unsigned long long>(runs),
            static_cast<unsigned long long>(last_device_read_ns_.load(std::memory_order_relaxed) / 1000ull));
    }
  }

  void serialize_probes(std::string& j, bool& first_metric) {
    graphsignal_probe_registry* reg = graphsignal_probe_reader_attach();
    if (!reg) return;

    uint64_t count = graphsignal_probe_reader_count(reg);
    std::vector<DeviceRead> reads;
    std::vector<graphsignal_instrument_data> host;
    read_device_probes(reg, count, reads, host);
    size_t next_read = 0;

    graphsignal_instrument_data snap;
    for (uint64_t i = 0; i < count; i++) {
      const graphsignal_probe_entry* e = &reg->entries[i];
      if (e->type == GRAPHSIGNAL_PROFILE) {
        serialize_probe_profile(j, first_metric, e);
        continue;
      }
      if (e->storage == GRAPHSIGNAL_STORAGE_HOST) {
        if (!graphsignal_probe_snapshot(e, &snap)) continue;
      } else {
        while (next_read < reads.size() && reads[next_read].index < i) next_read++;
        if (next_read >= reads.size() || reads[next_read].index != i || !reads[next_read].ok) continue;
        snap = host[reads[next_read].slot];
      }

      if (!first_metric) j += ',';
      first_metric = false;
      serialize_metric_head(
          j, e->name,
          e->type == GRAPHSIGNAL_GAUGE ? "gauge"
          : e->type == GRAPHSIGNAL_COUNTER ? "counter" : "histogram");
      serialize_probe_tags(j, e);

      switch (e->type) {
        case GRAPHSIGNAL_GAUGE:
          j += ",\"value\":";
          append_double(j, graphsignal_probe_gauge_value(&snap));
          break;
        case GRAPHSIGNAL_COUNTER:
          j += ",\"value\":";
          append_u64(j, snap.value_bits);
          break;
        default:
          serialize_bins(j, snap.bins);
          serialize_hist_aggregates(j, snap.count, snap.sum, snap.min, snap.max);
      }
      j += '}';
    }

    uint64_t probe_dropped = __atomic_load_n(&reg->dropped, __ATOMIC_RELAXED);
    if (probe_dropped > 0) {
      if (!first_metric) j += ',';
      first_metric = false;
      serialize_metric_head(j, "graphsignal_probe_dropped_instruments", "counter");
      j += ",\"tags\":{},\"value\":";
      append_u64(j, probe_dropped);
      j += '}';
    }
  }

  static void serialize_metric_head(std::string& j, const char* name,
                                    const char* type) {
    j += "{\"name\":\"";
    json_escape_append(j, name, GRAPHSIGNAL_PROBE_MAX_NAME);
    j += "\",\"type\":\"";
    j += type;
    j += '"';
  }

  static void serialize_tags(
      std::string& j, const std::vector<std::pair<std::string, std::string>>& tags) {
    j += ",\"tags\":{";
    bool first = true;
    for (const auto& tag : tags) {
      if (!first) j += ',';
      first = false;
      j += '"';
      json_escape_append(j, tag.first);
      j += "\":\"";
      json_escape_append(j, tag.second);
      j += '"';
    }
    j += '}';
  }

  static void serialize_probe_tags(std::string& j,
                                   const graphsignal_probe_entry* e) {
    j += ",\"tags\":{";
    for (uint32_t t = 0; t < e->ntags; t++) {
      if (t) j += ',';
      j += '"';
      json_escape_append(j, e->tag_keys[t], GRAPHSIGNAL_PROBE_MAX_TAG_KEY);
      j += "\":\"";
      json_escape_append(j, e->tag_values[t], GRAPHSIGNAL_PROBE_MAX_TAG_VALUE);
      j += '"';
    }
    j += '}';
  }

  // A frame serializes as "name":[value,samples].
  static void serialize_frames(
      std::string& j,
      const std::unordered_map<std::string, Instrument::FrameAgg>& frames) {
    j += ",\"frames\":{";
    bool first = true;
    for (const auto& frame : frames) {
      if (!first) j += ',';
      first = false;
      j += '"';
      json_escape_append(j, frame.first);
      j += "\":[";
      append_u64(j, frame.second.value);
      j += ',';
      append_u64(j, frame.second.samples);
      j += ']';
    }
    j += '}';
  }

  void serialize_probe_profile(std::string& j, bool& first_metric,
                               const graphsignal_probe_entry* e) {
    graphsignal_profile_data* p = graphsignal_probe_profile_data(
        const_cast<graphsignal_probe_entry*>(e));
    if (!p) return;

    if (!first_metric) j += ',';
    first_metric = false;
    serialize_metric_head(j, e->name, "profile");
    serialize_probe_tags(j, e);
    j += ",\"frames\":{";
    uint64_t frame_count = __atomic_load_n(&p->frame_count, __ATOMIC_ACQUIRE);
    for (uint64_t f = 0; f < frame_count; f++) {
      if (f) j += ',';
      j += '"';
      json_escape_append(j, p->frames[f].name, GRAPHSIGNAL_PROBE_MAX_FRAME_NAME);
      j += "\":[";
      append_u64(j, __atomic_load_n(&p->frames[f].value, __ATOMIC_RELAXED));
      j += ',';
      append_u64(j, __atomic_load_n(&p->frames[f].samples, __ATOMIC_RELAXED));
      j += ']';
    }
    j += "}}";
  }

  // Exact aggregates of a histogram ("count","sum","min","max"). Bins and
  // aggregates are independent halves of the same metric: either alone is a
  // whole histogram to the importer, and an instrument that has recorded
  // nothing emits neither, so it is skipped rather than stored as an empty
  // series.
  static void serialize_hist_aggregates(std::string& j, uint64_t count, uint64_t sum,
                                        uint64_t min, uint64_t max) {
    if (count == 0) return;
    j += ",\"count\":";
    append_u64(j, count);
    j += ",\"sum\":";
    append_u64(j, sum);
    j += ",\"min\":";
    append_u64(j, min);
    j += ",\"max\":";
    append_u64(j, max);
  }

  // Omitted entirely when every bin is empty — see serialize_hist_aggregates.
  static void serialize_bins(std::string& j, const uint64_t* bins) {
    std::string bins_json;
    std::string counts_json;
    for (uint32_t b = 0; b < GRAPHSIGNAL_PROBE_HIST_BINS; b++) {
      if (bins[b] == 0) continue;
      if (!bins_json.empty()) {
        bins_json += ',';
        counts_json += ',';
      }
      append_u64(bins_json, graphsignal_probe_bin_lower(b));
      append_u64(counts_json, bins[b]);
    }
    if (bins_json.empty()) return;
    j += ",\"bins\":[";
    j += bins_json;
    j += "],\"counts\":[";
    j += counts_json;
    j += ']';
  }

  void serialize_log(std::string& j) {
    std::lock_guard<std::mutex> g(log_mu_);
    if (log_ring_.empty()) return;
    j += ",\"log\":[";
    size_t n = log_ring_.size();
    // Oldest first: entries wrapped at log_head_ when the ring is full.
    size_t start = (n >= kMaxLogEntries) ? (log_head_ % kMaxLogEntries) : 0;
    for (size_t i = 0; i < n; i++) {
      const LogEntry& entry = log_ring_[(start + i) % n];
      if (i) j += ',';
      j += "{\"ts\":";
      append_u64(j, entry.ts);
      j += ",\"level\":\"";
      json_escape_append(j, entry.level);
      j += "\",\"msg\":\"";
      json_escape_append(j, entry.msg);
      j += "\"}";
    }
    j += ']';
  }

  void write_file(const std::string& data) {
    int fd = open(tmp_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return;
    const char* p = data.c_str();
    size_t rem = data.size();
    while (rem > 0) {
      ssize_t rc = write(fd, p, rem);
      if (rc < 0) {
        if (errno == EINTR) continue;
        close(fd);
        unlink(tmp_path_.c_str());
        return;
      }
      p += rc;
      rem -= static_cast<size_t>(rc);
    }
    close(fd);
    if (rename(tmp_path_.c_str(), file_path_.c_str()) != 0) unlink(tmp_path_.c_str());
  }

  static const char* type_str(InstrumentType type) {
    switch (type) {
      case InstrumentType::Gauge: return "gauge";
      case InstrumentType::Counter: return "counter";
      case InstrumentType::Histogram: return "histogram";
      default: return "profile";
    }
  }

  struct LogEntry {
    uint64_t ts = 0;
    std::string level;
    std::string msg;
  };

  uint64_t pid_ = 0;
  uint64_t start_ts_ = 0;
  uint64_t interval_ns_ = kDefaultIntervalNs;
  std::string dir_path_;
  std::string file_path_;
  std::string tmp_path_;
  std::vector<std::pair<std::string, std::string>> context_;

  std::mutex reg_mu_;
  std::unordered_map<std::string, Instrument*> by_key_;
  std::vector<Instrument*> ordered_;
  std::atomic<uint64_t> dropped_instruments_{0};

  std::mutex hook_mu_;
  FlushHook flush_hook_ = nullptr;
  void* flush_arg_ = nullptr;
  DeviceProbeReader device_reader_ = nullptr;
  void* device_reader_arg_ = nullptr;
  DeviceProbeBatchReader device_batch_reader_ = nullptr;
  void* device_batch_reader_arg_ = nullptr;
  std::atomic<uint64_t> last_device_probes_{0};
  std::atomic<uint64_t> last_device_runs_{0};
  std::atomic<uint64_t> last_device_read_ns_{0};

  std::mutex log_mu_;
  std::vector<LogEntry> log_ring_;
  size_t log_head_ = 0;
  std::atomic<bool> debug_enabled_{false};

  std::mutex cv_mu_;
  std::condition_variable cv_;
  std::atomic<bool> stopped_{false};
  std::thread writer_thread_;
};

}  // namespace graphsignal
