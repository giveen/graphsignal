#include "cupti_activity.h"

#include <cupti.h>
#include <cuda.h>

extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize(void* pfnGetExportTable);
#include <generated_nvtx_meta.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

#include "metrics_writer.h"
#include "nvtx_ranges.h"

#ifndef CUPTI_CALL
#define CUPTI_CALL(call)                                                        \
  do {                                                                          \
    CUptiResult _status = (call);                                               \
    if (_status != CUPTI_SUCCESS && g_writer) {                                 \
      const char *errstr = nullptr;                                             \
      cuptiGetResultString(_status, &errstr);                                   \
      g_writer->error("CUPTI error %d (%s) at %s:%d",                           \
              _status, errstr ? errstr : "unknown", __FILE__, __LINE__);        \
    }                                                                           \
  } while (0)
#endif

namespace {

// Keep buffers reasonably sized so CUPTI delivers records frequently.
constexpr size_t kBufferSize = 128 * 1024;
constexpr size_t kBufferAlign = 8;

std::atomic<bool> g_running{false};

// Owns the cumulative instruments, the retained log ring, and the background
// thread that serializes everything to /dev/shm/graphsignal_<pid>/cupti.json.
// Deliberately leaked at stop (no destructor work on teardown).
static graphsignal::MetricsWriter* g_writer = nullptr;

// ---------------------------------------------------------------------------
// Instrument lookups. Instruments are registered lazily on the first record of
// a given kernel/graph/kind and cached here so the per-record path is a map or
// array lookup. All registration runs under g_instruments_mu.
// ---------------------------------------------------------------------------

static std::mutex g_instruments_mu;

// GPU activity aggregates into profile instruments: one metric per activity
// family, frames keyed by kernel symbol / graph signature / kind. Frame
// cardinality is bounded by the writer (kMaxProfileFrames).
static graphsignal::Instrument* g_kernels_profile = nullptr;
static graphsignal::Instrument* g_graphs_profile = nullptr;
static graphsignal::Instrument* g_memcpy_profile = nullptr;
static graphsignal::Instrument* g_memset_profile = nullptr;
static graphsignal::Instrument* g_sync_profile = nullptr;

// Reports the CUDA graph tracing granularity in effect (0 = graph, 1 = node).
// Without it an empty cuda_graphs_nanoseconds is ambiguous: node mode, or a
// workload that launches no graphs at all.
static graphsignal::Instrument* g_graph_trace_mode_gauge = nullptr;
// Cumulative CUPTI records the driver could not hand us (buffer overflow under
// load). Published as a counter so a reader can tell "this engine was idle" from
// "we lost data and the timings below are incomplete" — a silent gap that used
// to be visible only in the debug log.
static graphsignal::Instrument* g_dropped_records_counter = nullptr;

// Graph tracing granularity for this process, set once by
// cupti_activity_start. Read on the record path only to decide nothing — the
// mode is expressed by which activity kinds are enabled — so a plain value is
// enough.
static uint32_t g_graph_trace_mode = GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;

// An unrecognized GRAPHSIGNAL_CUDA_GRAPH_TRACE value, kept until the writer's
// log ring exists (the variable is parsed in InitializeInjection, before the
// writer is created). Fixed size: the native lib keeps all memory bounded.
static char g_graph_trace_env_invalid[64] = {0};

// graphId -> frame name (16-hex signature hash, or graph_<id> fallback).
static std::unordered_map<uint32_t, std::string> g_graph_frames;

// Per-kind byte counters, indexed by the raw CUPTI enum value; values past
// the known range share slot 0, whose tag string the *KindToStr default
// covers. `registered` distinguishes "not yet seen" from "registration
// failed" so a failed registration isn't retried per record.
struct KindInstrument {
  graphsignal::Instrument* inst = nullptr;
  bool registered = false;
};
constexpr size_t kNumMemcpyKinds = 12;
constexpr size_t kNumMemsetKinds = 8;
static KindInstrument g_memcpy_bytes[kNumMemcpyKinds];
static KindInstrument g_memset_bytes[kNumMemsetKinds];

// CUPTI resource-callback subscriber (CUPTI_CB_DOMAIN_RESOURCE). Used to observe
// CUDA graph instantiation so we can compute a stable structural signature for
// each executable graph once, off the launch hot path.
static CUpti_SubscriberHandle g_subscriber{};
static graphsignal::NvtxRangeAggregator g_nvtx_ranges;

// graphId -> structural signature ("kernel[name=<symbol>,calls=n];" for kernel
// nodes, "<node_type>[calls=n];" otherwise; sorted, joined). A GRAPH_TRACE
// activity record carries only a graphId; this map lets the trace handler tag
// the instrument by structure (which aggregates across processes/hosts) instead
// of the process-local graphId. The signature
// is registered under BOTH the CUgraph id and the CUgraphExec id, since the id
// carried by GRAPH_TRACE differs across CUDA versions. Bounded with
// clear-on-overflow so a graph-churning workload can't grow it without bound.
constexpr size_t kMaxTrackedGraphs = 8192;
static std::unordered_map<uint32_t, std::string> g_graph_signatures;
static std::mutex g_graph_sig_mu;

// Bounded free-list of CUPTI activity buffers. 128 KiB sits at glibc's mmap
// threshold, so allocating each buffer fresh on the callback path means an
// mmap/munmap syscall pair (with page-zeroing) or arena-lock contention shared
// with the workload's own malloc; the free-list keeps a few buffers resident.
constexpr size_t kMaxPooledBuffers = 16;
static std::mutex g_buffer_pool_mu;
static std::vector<uint8_t*> g_buffer_pool;

static uint8_t* buffer_pool_acquire() {
  {
    std::lock_guard<std::mutex> g(g_buffer_pool_mu);
    if (!g_buffer_pool.empty()) {
      uint8_t* p = g_buffer_pool.back();
      g_buffer_pool.pop_back();
      return p;
    }
  }
  void* p = nullptr;
  if (posix_memalign(&p, kBufferAlign, kBufferSize) != 0 || !p) return nullptr;
  return static_cast<uint8_t*>(p);
}

static void buffer_pool_release(uint8_t* buf) {
  if (!buf) return;
  {
    std::lock_guard<std::mutex> g(g_buffer_pool_mu);
    if (g_buffer_pool.size() < kMaxPooledBuffers) {
      g_buffer_pool.push_back(buf);
      return;
    }
  }
  std::free(buf);
}

static void buffer_pool_drain() {
  std::lock_guard<std::mutex> g(g_buffer_pool_mu);
  for (uint8_t* p : g_buffer_pool) std::free(p);
  g_buffer_pool.clear();
}

// ---------------------------------------------------------------------------

static const char* syncTypeToStr(CUpti_ActivitySynchronizationType t) {
  switch (t) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE: return "context";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:  return "stream";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:   return "event";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:   return "stream_wait_event";
    default: return "unknown";
  }
}

static const char* memsetKindToStr(CUpti_ActivityMemoryKind k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:         return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:         return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:       return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:        return "managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:          return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:  return "device_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC: return "managed_static";
    default:                                        return "unknown";
  }
}

static const char* memcpyKindToStr(CUpti_ActivityMemcpyKind k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "host_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "device_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "device_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "host_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "peer_to_peer";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "host_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "array_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "array_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "array_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "device_to_array";
    default: return "other";
  }
}

// Short, stable token per CUDA graph node type — keeps the signature compact
// and human-skimmable.
static const char* graphNodeTypeToStr(CUgraphNodeType t) {
  switch (t) {
    case CU_GRAPH_NODE_TYPE_KERNEL:           return "kernel";
    case CU_GRAPH_NODE_TYPE_MEMCPY:           return "memcpy";
    case CU_GRAPH_NODE_TYPE_MEMSET:           return "memset";
    case CU_GRAPH_NODE_TYPE_HOST:             return "host";
    case CU_GRAPH_NODE_TYPE_GRAPH:            return "child";
    case CU_GRAPH_NODE_TYPE_EMPTY:            return "empty";
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT:       return "wait";
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:     return "event";
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: return "semsignal";
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT:   return "semwait";
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC:        return "memalloc";
    case CU_GRAPH_NODE_TYPE_MEM_FREE:         return "memfree";
    default:                                  return "other";
  }
}

// Build a structural signature for a CUDA graph: the multiset of node tuples,
// sorted and joined. Kernel nodes are keyed by their compiled symbol name
// ("kernel[name=<symbol>,calls=n];"); every other node type is keyed by type
// only ("memset[calls=n];"). The kernel symbol encodes the specialization
// (template params/tile sizes/dtype) but NOT the launch grid, so graphs that
// differ only by batch/token size (decode buckets) share one signature. The
// signature is order-independent (sorted) so it is stable across runs/processes
// for a structurally identical graph. Best-effort: returns "" on any failure.
static std::string compute_graph_signature(CUgraph graph) {
  if (!graph) return std::string();

  size_t num_nodes = 0;
  if (cuGraphGetNodes(graph, nullptr, &num_nodes) != CUDA_SUCCESS) return std::string();
  if (num_nodes == 0) return std::string();

  // Bound the work: a few thousand nodes is far more than any plausible decode
  // graph; cap to keep the one-time enumeration cheap and the string bounded.
  constexpr size_t kMaxNodes = 8192;
  if (num_nodes > kMaxNodes) num_nodes = kMaxNodes;

  std::vector<CUgraphNode> nodes(num_nodes);
  size_t got = num_nodes;
  if (cuGraphGetNodes(graph, nodes.data(), &got) != CUDA_SUCCESS) return std::string();
  if (got < num_nodes) num_nodes = got;

  // tuple key (everything up to the count) -> count. The "calls=" separator is
  // baked into the key so the emit loop can append the count uniformly.
  std::map<std::string, uint64_t> tuples;
  for (size_t i = 0; i < num_nodes; ++i) {
    CUgraphNodeType ntype;
    if (cuGraphNodeGetType(nodes[i], &ntype) != CUDA_SUCCESS) continue;

    std::string key = graphNodeTypeToStr(ntype);
    if (ntype == CU_GRAPH_NODE_TYPE_KERNEL) {
      // Identify the kernel by its compiled symbol name. cuFuncGetName returns
      // the mangled name (alphanumeric/underscore — no signature delimiters),
      // which is shape-independent: decode graphs that differ only by grid map
      // to the same name. Available since CUDA 12.3; the GRAPH_TRACE path that
      // calls this requires CUDA 12.4, so the symbol is always present at
      // runtime here (guarded for older build toolkits only).
      const char* kname = nullptr;
      CUDA_KERNEL_NODE_PARAMS kp;
      std::memset(&kp, 0, sizeof(kp));
      if (cuGraphKernelNodeGetParams(nodes[i], &kp) == CUDA_SUCCESS && kp.func) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12030
        if (cuFuncGetName(&kname, kp.func) != CUDA_SUCCESS) kname = nullptr;
#endif
      }
      key += "[name=";
      key += (kname ? kname : "");
      key += ",calls=";
    } else {
      key += "[calls=";
    }
    tuples[key] += 1;
  }

  if (tuples.empty()) return std::string();

  std::string sig;
  for (const auto& kv : tuples) {  // std::map iterates in sorted key order
    sig += kv.first;
    sig += std::to_string(kv.second);
    sig += "];";
  }
  return sig;
}

// Register a graph's signature under a graphId (CUgraph id and/or CUgraphExec
// id). Bounded with clear-on-overflow.
static void register_graph_signature(uint32_t graph_id, const std::string& sig) {
  if (sig.empty()) return;
  std::lock_guard<std::mutex> lock(g_graph_sig_mu);
  if (g_graph_signatures.size() >= kMaxTrackedGraphs &&
      g_graph_signatures.find(graph_id) == g_graph_signatures.end()) {
    g_graph_signatures.clear();
  }
  g_graph_signatures[graph_id] = sig;
}

// Returns the signature for a graphId, or "" if none is known yet.
static std::string lookup_graph_signature(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(g_graph_sig_mu);
  auto it = g_graph_signatures.find(graph_id);
  return it == g_graph_signatures.end() ? std::string() : it->second;
}

// 16-hex lowercase FNV-1a64 of a string — the compact, cross-process-stable
// tag value for a graph's structural signature.
static void fnv1a64_hex(const std::string& s, char out[17]) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  std::snprintf(out, 17, "%016llx", static_cast<unsigned long long>(h));
}

// Reduce one kernel symbol to a readable stem. Itanium-mangled names are
// demangled (best effort), then everything from the parameter list onward is
// dropped, and template arguments are dropped too: the structural hash already
// distinguishes two instantiations that share a stem, so the stem only has to
// answer "which kernel is this".
static std::string graph_kernel_stem(const std::string& symbol) {
  if (symbol.empty()) return std::string("kernel");
  std::string name = symbol;
#if defined(__GNUC__) || defined(__clang__)
  int status = 0;
  char* demangled = abi::__cxa_demangle(symbol.c_str(), nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    name = demangled;
  }
  std::free(demangled);
#endif
  const size_t paren = name.find('(');
  if (paren != std::string::npos) name.erase(paren);
  const size_t tmpl = name.find('<');
  if (tmpl != std::string::npos) name.erase(tmpl);
  // Trailing "::" left behind by a template-only qualified name.
  while (name.size() > 2 && name.compare(name.size() - 2, 2, "::") == 0) {
    name.erase(name.size() - 2);
  }
  if (name.empty()) name = "kernel";
  return name;
}

// Turn a structural signature into a short human label: the most-used node
// kinds with their multiplicities, e.g. "flash_attn+gemm*2+memcpy". This is
// what a reader actually wants from a CUDA-graph frame; the hash alongside it
// keeps distinct graphs distinct even when their labels collide.
static std::string graph_label_from_signature(const std::string& sig) {
  if (sig.empty()) return std::string();

  struct Token {
    std::string text;
    uint64_t count;
  };
  std::vector<Token> tokens;
  size_t pos = 0;
  while (pos < sig.size()) {
    const size_t end = sig.find(';', pos);
    const std::string seg =
        sig.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    pos = (end == std::string::npos) ? sig.size() : end + 1;
    if (seg.empty()) continue;

    const size_t lb = seg.find('[');
    if (lb == std::string::npos) continue;
    std::string head = seg.substr(0, lb);
    const size_t calls = seg.rfind(",calls=");
    uint64_t n = 1;
    if (calls != std::string::npos) {
      n = std::strtoull(seg.c_str() + calls + 7, nullptr, 10);
      if (n == 0) n = 1;
    }
    if (head == "kernel") {
      const std::string marker = "name=";
      const size_t nm = seg.find(marker);
      if (nm != std::string::npos) {
        const size_t start = nm + marker.size();
        const size_t stop = (calls == std::string::npos) ? seg.size() : calls;
        if (stop > start) head = graph_kernel_stem(seg.substr(start, stop - start));
      }
    }
    tokens.push_back(Token{head, n});
  }
  if (tokens.empty()) return std::string();

  // Most-used first; ties broken by name so the label is deterministic.
  std::sort(tokens.begin(), tokens.end(), [](const Token& a, const Token& b) {
    if (a.count != b.count) return a.count > b.count;
    return a.text < b.text;
  });

  constexpr size_t kMaxTokens = 3;
  std::string label;
  uint64_t shown = 0;
  for (size_t i = 0; i < tokens.size() && i < kMaxTokens; ++i) {
    if (i > 0) label += "+";
    label += tokens[i].text;
    if (tokens[i].count > 1) label += "*" + std::to_string(tokens[i].count);
    shown += tokens[i].count;
  }
  if (tokens.size() > kMaxTokens) {
    // Keep the tail bounded: say how much was left out rather than listing it.
    label += "+" + std::to_string(tokens.size() - kMaxTokens) + "more";
  }
  if (label.size() > 64) label.resize(64);
  return label;
}

// ---------------------------------------------------------------------------
// Lazy instrument registration
// ---------------------------------------------------------------------------

static graphsignal::Instrument* kind_instrument(KindInstrument& slot, graphsignal::InstrumentType type,
                                       const char* name,
                                       const char* tag_key, const char* tag_value) {
  std::lock_guard<std::mutex> g(g_instruments_mu);
  if (!slot.registered) {
    slot.registered = true;
    if (g_writer) {
      slot.inst = g_writer->register_instrument(type, name,
                                                {{tag_key, tag_value}});
    }
  }
  return slot.inst;
}

// Frame name for a graph: the structural signature (computed once at
// instantiation via the resource callback), hashed to a compact token so
// structurally identical graphs aggregate across processes/hosts. Falls back
// to the process-local graphId when no signature is known (e.g. the resource
// callback didn't fire, or graphId semantics didn't match).
static const std::string& graph_frame_name(uint32_t graph_id) {
  {
    std::lock_guard<std::mutex> g(g_instruments_mu);
    auto it = g_graph_frames.find(graph_id);
    if (it != g_graph_frames.end()) return it->second;
  }

  // Computed outside g_instruments_mu — lookup_graph_signature takes its own
  // lock.
  const std::string sig = lookup_graph_signature(graph_id);
  std::string frame;
  if (!sig.empty()) {
    char hex[17];
    fnv1a64_hex(sig, hex);
    // Readable label plus the hash: the label answers "what is in this graph",
    // the hash keeps two structurally different graphs from sharing a frame
    // when their labels happen to agree. The hash alone used to be the whole
    // frame name, which made the profile unreadable.
    const std::string label = graph_label_from_signature(sig);
    if (label.empty()) {
      frame = hex;
    } else {
      char short_hex[9];
      std::memcpy(short_hex, hex, 8);
      short_hex[8] = '\0';
      frame = label + " [" + short_hex + "]";
    }
  } else {
    frame = "graph_" + std::to_string(graph_id);
  }

  std::lock_guard<std::mutex> g(g_instruments_mu);
  auto it = g_graph_frames.find(graph_id);
  if (it != g_graph_frames.end()) return it->second;
  if (g_graph_frames.size() >= kMaxTrackedGraphs) g_graph_frames.clear();
  if (g_writer) {
    g_writer->debug("graph %s signature: %s", frame.c_str(), sig.c_str());
  }
  return g_graph_frames.emplace(graph_id, std::move(frame)).first->second;
}

// ---------------------------------------------------------------------------
// Writer hooks
// ---------------------------------------------------------------------------

// Called by the writer thread before each periodic serialize so buffered CUPTI
// activity reaches the instruments first. The Writer never calls it on the
// final shutdown write, and the g_running guard keeps it from touching CUPTI
// while stop is tearing profiling down.
static void writer_flush_hook(void*) {
  if (!g_running.load(std::memory_order_relaxed)) return;
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

// Copies DEVICE-storage probe blocks (probe.h DEVICE storage) to host memory
// via the driver API so user CUDA probes can be serialized. One call copies a
// run of `n` contiguous blocks (the writer coalesces adjacent registrations,
// e.g. a probe_cuda.h pool, into runs — one copy per run instead of one per
// probe). The copy runs on a private NON-BLOCKING stream of the device's
// primary context: a synchronous cuMemcpyDtoH on the legacy default stream
// would wait for, and be waited on by, the workload's own blocking-stream
// kernels — a device-wide serialization point once per probe per write that
// the workload would pay for being profiled. Every driver call is checked; any
// failure (including a not-yet-initialized driver) skips the run for this
// write. Never touches CUDA during/after teardown (g_running).
// The primary context is retained ONCE (the retain/release pair per probe was
// the dominant cost: ~0.25 ms per probe per write) and deliberately never
// released — it lives until process exit anyway, and teardown makes no CUDA
// calls. The stream is created once in that context.
static std::mutex g_probe_stream_mu;
static CUstream g_probe_stream = nullptr;
static CUcontext g_probe_ctx = nullptr;

static bool device_probe_batch_reader(const graphsignal_instrument_data* first,
                                      size_t n, graphsignal_instrument_data* out, void*) {
  if (!g_running.load(std::memory_order_relaxed)) return false;
  try {
    if (!first || !out || n == 0) return false;
    std::lock_guard<std::mutex> g(g_probe_stream_mu);
    if (!g_probe_ctx) {
      CUdevice dev = 0;
      if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;
      CUcontext ctx = nullptr;
      if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return false;
      g_probe_ctx = ctx;
    }
    if (cuCtxPushCurrent(g_probe_ctx) != CUDA_SUCCESS) return false;
    CUresult rc = CUDA_SUCCESS;
    if (!g_probe_stream) {
      rc = cuStreamCreate(&g_probe_stream, CU_STREAM_NON_BLOCKING);
      if (rc != CUDA_SUCCESS) g_probe_stream = nullptr;
    }
    if (rc == CUDA_SUCCESS) {
      rc = cuMemcpyDtoHAsync(out, (CUdeviceptr)(uintptr_t)first,
                             n * sizeof(graphsignal_instrument_data), g_probe_stream);
      if (rc == CUDA_SUCCESS) rc = cuStreamSynchronize(g_probe_stream);
    }
    cuCtxPopCurrent(nullptr);
    return rc == CUDA_SUCCESS;
  } catch (...) {
    return false;
  }
}

// Single-probe form kept for the per-probe reader hook (tests, fallback).
static bool device_probe_reader(const graphsignal_probe_entry* e,
                                graphsignal_instrument_data* out, void*) {
  if (!e || !e->data) return false;
  return device_probe_batch_reader(e->data, 1, out, nullptr);
}

// ---------------------------------------------------------------------------
// CUPTI callbacks
// ---------------------------------------------------------------------------

// Opt-in CUDA runtime API tracing. Off by default: the profiler's contract is
// that it costs nothing unless asked. When GRAPHSIGNAL_CUDA_API_TRACE selects
// it, only the named APIs are traced — CUPTI can enable a small set of runtime
// APIs individually, without turning on CUPTI_ACTIVITY_KIND_RUNTIME wholesale,
// so this stays a fraction of what a full API trace costs.
//
// The cbid -> name table is built at init by asking CUPTI what each callback id
// is, rather than by naming the CUPTI_RUNTIME_TRACE_CBID_* enum values: those
// carry per-API version suffixes that differ between CUDA releases
// (cudaGraphInstantiate exists as both _v10000 and _v12000), so compile-time
// references do not survive a toolkit upgrade.
static constexpr size_t kMaxTracedApis = 48;
static constexpr size_t kMaxApiNameBytes = 48;
struct TracedApi {
  uint32_t cbid;
  char name[kMaxApiNameBytes];
};
static TracedApi g_traced_apis[kMaxTracedApis];
static size_t g_num_traced_apis = 0;
static graphsignal::Instrument* g_api_profile = nullptr;

// The APIs worth tracing by default: allocator churn and graph lifecycle, the
// two costs that show up as unexplained host-side time in an inference engine
// and are invisible to kernel/memcpy/sync timing.
static const char* const kDefaultTracedApis[] = {
    "cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync",
    "cudaHostAlloc", "cudaHostRegister",
    "cudaGraphInstantiate", "cudaGraphInstantiateWithFlags", "cudaGraphLaunch",
    "cudaGraphExecUpdate",
};
static constexpr size_t kNumDefaultTracedApis =
    sizeof(kDefaultTracedApis) / sizeof(kDefaultTracedApis[0]);

// CUPTI reports runtime API names with the ABI version they were introduced in
// ("cudaMalloc_v3020"), and one API can appear more than once when it was
// revised ("cudaGraphInstantiate_v10000" and "_v12000"). Strip the suffix so
// names match the wanted set and read cleanly in the payload; both revisions
// then share one frame, which is what a reader wants.
static std::string strip_api_version(const char* name) {
  std::string s = name ? name : "";
  const size_t us = s.rfind("_v");
  if (us == std::string::npos || us + 2 >= s.size()) return s;
  for (size_t i = us + 2; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return s;  // not a version suffix
  }
  return s.substr(0, us);
}

enum class ApiTraceMode {
  kOff,
  kDefaultSet,   // the allocation + graph APIs above
  kAll,          // every runtime API: enables the kind, so this is the heavy one
  kList,         // an explicit comma-separated list
};

static bool env_truthy(const char* v) {
  if (!v || !*v) return false;
  return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

static ApiTraceMode api_trace_mode(const char* selection) {
  if (!env_truthy(selection)) return ApiTraceMode::kOff;
  if (!selection || std::strcmp(selection, "all") == 0) return ApiTraceMode::kAll;
  if (selection[0] == '1' || selection[0] == 't' || selection[0] == 'T' ||
      selection[0] == 'y' || selection[0] == 'Y' || selection[0] == 'o' ||
      selection[0] == 'O') {
    return ApiTraceMode::kDefaultSet;
  }
  return ApiTraceMode::kList;
}

static bool api_name_wanted(const char* name, ApiTraceMode mode, const char* list) {
  if (!name) return false;
  if (mode == ApiTraceMode::kList) return std::strstr(list, name) != nullptr;
  for (size_t i = 0; i < kNumDefaultTracedApis; ++i) {
    if (std::strcmp(name, kDefaultTracedApis[i]) == 0) return true;
  }
  return false;
}

// Resolve and enable the requested runtime APIs. Returns how many were enabled.
static size_t enable_traced_runtime_apis(ApiTraceMode mode, const char* list) {
  g_num_traced_apis = 0;
  if (mode == ApiTraceMode::kAll) {
    // Every runtime API. This is the expensive mode: the kind switch traces
    // every call, which is what nsys does. Off unless explicitly asked for.
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    return static_cast<size_t>(-1);  // "all": no per-API table
  }
  // Runtime cbids are a contiguous generated enum starting at 1. Names are
  // only available for the DRIVER and RUNTIME domains, and asking past the end
  // is an error, so walk until a run of misses rather than a hard-coded bound.
  int consecutive_misses = 0;
  for (uint32_t cbid = 1; cbid < 8192 && consecutive_misses < 16; ++cbid) {
    const char* name = nullptr;
    if (cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, cbid, &name) != CUPTI_SUCCESS ||
        name == nullptr) {
      ++consecutive_misses;
      continue;
    }
    consecutive_misses = 0;
    const std::string api = strip_api_version(name);
    if (api.empty() || !api_name_wanted(api.c_str(), mode, list)) continue;
    if (cuptiActivityEnableRuntimeApi(static_cast<CUpti_CallbackId>(cbid), 1) != CUPTI_SUCCESS) {
      continue;
    }
    if (g_num_traced_apis < kMaxTracedApis) {
      TracedApi& slot = g_traced_apis[g_num_traced_apis];
      slot.cbid = cbid;
      std::strncpy(slot.name, api.c_str(), kMaxApiNameBytes - 1);
      slot.name[kMaxApiNameBytes - 1] = '\0';
      ++g_num_traced_apis;
    }
  }
  return g_num_traced_apis;
}

static const char* traced_api_name(uint32_t cbid) {
  for (size_t i = 0; i < g_num_traced_apis; ++i) {
    if (g_traced_apis[i].cbid == cbid) return g_traced_apis[i].name;
  }
  return nullptr;
}

// CUPTI resource callback. Observes CUDA graph instantiation so we can compute
// each graph's structural signature once, off the launch hot path, and key it
// by the ids that GRAPH_TRACE may use.
static const char* nvtx_message(const nvtxEventAttributes_t* attrs) {
  if (!attrs) return nullptr;
  if (attrs->messageType == NVTX_MESSAGE_TYPE_ASCII) return attrs->message.ascii;
  // Registered and wide messages require a domain-owned string table. Do not
  // retain or guess those strings: NInfer's domain ranges use ASCII, and a
  // missing label is preferable to unbounded callback allocation.
  return nullptr;
}

static void CUPTIAPI graph_resource_callback(void* /*userdata*/, CUpti_CallbackDomain domain,
                                             CUpti_CallbackId cbid, const void* cbdata) {
  if (!g_running.load(std::memory_order_relaxed)) return;

  try {
    if (domain == CUPTI_CB_DOMAIN_NVTX) {
      const auto* nd = reinterpret_cast<const CUpti_NvtxData*>(cbdata);
      if (!nd) return;
      if (cbid == CUPTI_CBID_NVTX_nvtxDomainCreateA) {
        const auto* p = reinterpret_cast<const nvtxDomainCreateA_params*>(nd->functionParams);
        if (p) g_nvtx_ranges.on_domain_create(
            reinterpret_cast<uintptr_t>(nd->functionReturnValue), p->name);
      } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainRegisterStringA) {
        const auto* p = reinterpret_cast<const nvtxDomainRegisterStringA_params*>(nd->functionParams);
        if (p) g_nvtx_ranges.on_register(reinterpret_cast<uintptr_t>(p->domain),
                                         reinterpret_cast<uintptr_t>(nd->functionReturnValue),
                                         p->string);
      } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainDestroy) {
        const auto* p = reinterpret_cast<const nvtxDomainDestroy_params*>(nd->functionParams);
        if (p) g_nvtx_ranges.on_domain_destroy(reinterpret_cast<uintptr_t>(p->domain));
      } else if (cbid == CUPTI_CBID_NVTX_nvtxMarkEx) {
        // A mark is a point event with no duration, so it is counted rather
        // than timed. Marks were previously ignored entirely.
        const auto* p = reinterpret_cast<const nvtxMarkEx_params*>(nd->functionParams);
        const auto* a = p ? p->eventAttrib : nullptr;
        if (a) {
          const char* name = (a->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
                                 ? g_nvtx_ranges.resolve_registered(
                                       reinterpret_cast<uintptr_t>(a->message.registered))
                                 : nvtx_message(a);
          g_nvtx_ranges.on_mark(0, name);
        }
      } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainMarkEx) {
        const auto* p = reinterpret_cast<const nvtxDomainMarkEx_params*>(nd->functionParams);
        const auto* a = p ? p->core.eventAttrib : nullptr;
        if (a) {
          const char* name = (a->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
                                 ? g_nvtx_ranges.resolve_registered(
                                       reinterpret_cast<uintptr_t>(a->message.registered))
                                 : nvtx_message(a);
          g_nvtx_ranges.on_mark(reinterpret_cast<uintptr_t>(p->domain), name);
        }
      } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangeStartEx ||
                 cbid == CUPTI_CBID_NVTX_nvtxDomainRangePushEx) {
        if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangeStartEx) {
          const auto* p = reinterpret_cast<const nvtxDomainRangeStartEx_params*>(nd->functionParams);
          const auto* a = p ? p->core.eventAttrib : nullptr;
          if (a && a->messageType == NVTX_MESSAGE_TYPE_REGISTERED) {
            const char* name = g_nvtx_ranges.resolve_registered(
                reinterpret_cast<uintptr_t>(a->message.registered));
            g_nvtx_ranges.on_start_category(
                reinterpret_cast<uintptr_t>(p->domain), a->category, name,
                reinterpret_cast<uintptr_t>(nd->functionReturnValue), true);
          } else {
            g_nvtx_ranges.on_start_category(reinterpret_cast<uintptr_t>(p->domain), a->category,
                                             nvtx_message(a),
                                             reinterpret_cast<uintptr_t>(nd->functionReturnValue), true);
          }
        } else {
          const auto* p = reinterpret_cast<const nvtxDomainRangePushEx_params*>(nd->functionParams);
          const auto* a = p ? p->core.eventAttrib : nullptr;
          if (a && a->messageType == NVTX_MESSAGE_TYPE_REGISTERED) {
            const char* name = g_nvtx_ranges.resolve_registered(
                reinterpret_cast<uintptr_t>(a->message.registered));
            g_nvtx_ranges.on_start_category(reinterpret_cast<uintptr_t>(p->domain), a->category,
                                             name, 0, false);
          } else {
            g_nvtx_ranges.on_start_category(reinterpret_cast<uintptr_t>(p->domain), a->category,
                                           nvtx_message(a), 0, false);
          }
        }
      } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangeEnd ||
                 cbid == CUPTI_CBID_NVTX_nvtxDomainRangePop) {
        if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangeEnd) {
          const auto* p = reinterpret_cast<const nvtxDomainRangeEnd_params*>(nd->functionParams);
          if (p) g_nvtx_ranges.on_end(reinterpret_cast<uintptr_t>(p->domain), p->core.id, true);
        } else {
          const auto* p = reinterpret_cast<const nvtxDomainRangePop_params*>(nd->functionParams);
          if (p) g_nvtx_ranges.on_end(reinterpret_cast<uintptr_t>(p->domain), 0, false);
        }
      } else if (cbid == CUPTI_CBID_NVTX_nvtxRangeStartEx ||
                 cbid == CUPTI_CBID_NVTX_nvtxRangePushEx) {
        // Default-domain ranges: an engine that annotates without creating a
        // domain of its own lands here. Domain 0 is adopted only when asked
        // for, and the aggregator ignores everything else.
        if (cbid == CUPTI_CBID_NVTX_nvtxRangeStartEx) {
          const auto* p = reinterpret_cast<const nvtxRangeStartEx_params*>(nd->functionParams);
          const auto* a = p ? p->eventAttrib : nullptr;
          if (a) {
            const char* name = (a->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
                                   ? g_nvtx_ranges.resolve_registered(
                                         reinterpret_cast<uintptr_t>(a->message.registered))
                                   : nvtx_message(a);
            g_nvtx_ranges.on_start_category(0, a->category, name,
                                            reinterpret_cast<uintptr_t>(nd->functionReturnValue),
                                            true);
          }
        } else {
          const auto* p = reinterpret_cast<const nvtxRangePushEx_params*>(nd->functionParams);
          const auto* a = p ? p->eventAttrib : nullptr;
          if (a) {
            const char* name = (a->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
                                   ? g_nvtx_ranges.resolve_registered(
                                         reinterpret_cast<uintptr_t>(a->message.registered))
                                   : nvtx_message(a);
            g_nvtx_ranges.on_start_category(0, a->category, name, 0, false);
          }
        }
      } else if (cbid == CUPTI_CBID_NVTX_nvtxRangeEnd ||
                 cbid == CUPTI_CBID_NVTX_nvtxRangePop) {
        if (cbid == CUPTI_CBID_NVTX_nvtxRangeEnd) {
          const auto* p = reinterpret_cast<const nvtxRangeEnd_params*>(nd->functionParams);
          if (p) g_nvtx_ranges.on_end(0, p->id, true);
        } else {
          const auto* p = reinterpret_cast<const nvtxRangePop_params*>(nd->functionParams);
          if (p) g_nvtx_ranges.on_end(0, 0, false);
        }
      }
      return;
    }
    if (domain != CUPTI_CB_DOMAIN_RESOURCE) return;

    const auto* rd = reinterpret_cast<const CUpti_ResourceData*>(cbdata);
    if (!rd || !rd->resourceDescriptor) return;

    switch (cbid) {
      case CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED: {
        const auto* gd = reinterpret_cast<const CUpti_GraphData*>(rd->resourceDescriptor);
        const std::string sig = compute_graph_signature(gd->graph);
        if (sig.empty()) break;
        uint32_t id = 0;
        if (gd->graph && cuptiGetGraphId(gd->graph, &id) == CUPTI_SUCCESS) register_graph_signature(id, sig);
        if (gd->graphExec && cuptiGetGraphExecId(gd->graphExec, &id) == CUPTI_SUCCESS) register_graph_signature(id, sig);
        break;
      }
      default: break;
    }
  } catch (...) {
  }
}

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *buffer = nullptr;
  *size = 0;
  *maxNumRecords = 0;

  // After stop / during teardown, hand CUPTI nothing.
  if (!g_running.load(std::memory_order_relaxed)) return;

  try {
    if (g_writer) g_writer->debug("cupti bufferRequested: size=%zu", kBufferSize);

    uint8_t* p = buffer_pool_acquire();
    if (!p) {
      if (g_writer) g_writer->error("cupti bufferRequested: alloc FAILED");
      return;
    }

    *buffer = p;
    *size = kBufferSize;
  } catch (...) {
    // Never let an exception cross the CUPTI C callback boundary.
    *buffer = nullptr;
    *size = 0;
  }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
  // Late callback after stop / during teardown: globals (the instrument maps)
  // may be mid-destruction. Release the buffer and bail without touching any
  // shared state. g_running is a static atomic, safe to read post-destruction.
  if (!g_running.load(std::memory_order_relaxed)) {
    std::free(buffer);
    return;
  }

  try {
  if (validSize > 0) {
    CUpti_Activity* record = nullptr;
    CUptiResult status;
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS && record) {
        switch (record->kind) {
          case CUPTI_ACTIVITY_KIND_KERNEL:
          case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel4*>(record);
            if (k->end <= k->start) break;
            graphsignal::MetricsWriter::profile_add(
                g_kernels_profile, k->name ? k->name : "", k->end - k->start);
            break;
          }
          case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: {
            // With GRAPH_TRACE enabled, kernels replayed via a CUDA graph are
            // not reported individually; the whole graph launch arrives as one
            // record. Use the base CUpti_ActivityGraphTrace struct — its
            // streamId/graphId/start/end prefix is layout-stable across the
            // GraphTrace/GraphTrace2 record revisions, same pattern as the
            // kernel/sync casts.
            const auto* g = reinterpret_cast<const CUpti_ActivityGraphTrace*>(record);
            if (g->start == 0 || g->end == 0 || g->end <= g->start) break;
            graphsignal::MetricsWriter::profile_add(
                g_graphs_profile, graph_frame_name(g->graphId).c_str(),
                g->end - g->start);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY: {
            const auto* m = reinterpret_cast<const CUpti_ActivityMemcpy*>(record);
            if (m->end <= m->start) break;
            const size_t kind = m->copyKind < kNumMemcpyKinds ? m->copyKind : 0;
            const char* kind_str =
                memcpyKindToStr(static_cast<CUpti_ActivityMemcpyKind>(m->copyKind));
            graphsignal::MetricsWriter::profile_add(
                g_memcpy_profile, kind_str, m->end - m->start);
            graphsignal::MetricsWriter::add(
                kind_instrument(g_memcpy_bytes[kind], graphsignal::InstrumentType::Counter,
                                "cuda_memcpy_bytes", "kind", kind_str),
                m->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMSET: {
            const auto* m = reinterpret_cast<const CUpti_ActivityMemset*>(record);
            if (m->end <= m->start) break;
            const size_t kind = m->memoryKind < kNumMemsetKinds ? m->memoryKind : 0;
            const char* kind_str =
                memsetKindToStr(static_cast<CUpti_ActivityMemoryKind>(m->memoryKind));
            graphsignal::MetricsWriter::profile_add(
                g_memset_profile, kind_str, m->end - m->start);
            graphsignal::MetricsWriter::add(
                kind_instrument(g_memset_bytes[kind], graphsignal::InstrumentType::Counter,
                                "cuda_memset_bytes", "kind", kind_str),
                m->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            // Always use the base struct: it's CUPTI's layout-compatible prefix
            // (carries type/streamId/cudaEventId/start/end) and is defined across
            // CUDA 12 and all 13.x. CUpti_ActivitySynchronization2 is NOT present
            // in every CUDA 13.x cupti.h, so a CUDA_MAJOR-based cast both breaks
            // the build on those toolchains and, when cross-built against a
            // version that has it, produces an ABI mismatch with the runtime.
            const auto* s = reinterpret_cast<const CUpti_ActivitySynchronization*>(record);
            if (s->start == 0 && s->end == 0) break;  // timestamps unavailable
            if (s->end <= s->start) break;
            graphsignal::MetricsWriter::profile_add(
                g_sync_profile, syncTypeToStr(s->type), s->end - s->start);
            break;
          }
          case CUPTI_ACTIVITY_KIND_RUNTIME: {
            // Only reached for the APIs opted into by
            // GRAPHSIGNAL_CUDA_API_TRACE; anything else is filtered by CUPTI
            // and never arrives here.
            const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(record);
            if (a->start == 0 && a->end == 0) break;
            if (a->end <= a->start) break;
            const char* name = traced_api_name(a->cbid);
            if (name == nullptr) break;
            graphsignal::MetricsWriter::profile_add(
                g_api_profile, name, a->end - a->start);
            break;
          }
          default:
            break;
        }
      } else if (status != CUPTI_SUCCESS && status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
        const char* errstr = nullptr;
        cuptiGetResultString(status, &errstr);
        if (g_writer) {
          g_writer->error("cuptiActivityGetNextRecord error: %s",
                          errstr ? errstr : "unknown");
        }
      }
    } while (status == CUPTI_SUCCESS);
  }

  // Count any dropped records since the previous call. Use the
  // callback-provided (context, streamId) to be compatible across CUDA 12/13.
  // These are records the driver never produced, so every timing this run
  // reports is missing them: publish the total rather than leaving the loss
  // visible only in the debug log.
  size_t dropped = 0;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));

  if (g_writer) {
    if (dropped > 0) {
      graphsignal::MetricsWriter::add(g_dropped_records_counter, dropped);
    }
    g_writer->debug("cupti bufferCompleted: size=%zu validSize=%zu dropped=%zu",
                    size, validSize, dropped);
  }
  } catch (...) {
    // Never let an exception cross the CUPTI C callback boundary.
  }

  buffer_pool_release(buffer);
}

} // namespace

int cupti_activity_start(uint64_t write_interval_ns, uint32_t debug_mode,
                         uint32_t graph_trace_mode) {
  bool expected = false;
  if (!g_running.compare_exchange_strong(expected, true)) return 1;

  // Anything unexpected degrades to the default granularity rather than
  // leaving collection in an undefined state.
  g_graph_trace_mode = (graph_trace_mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE)
                           ? GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE
                           : GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;

  // The Writer owns the instruments, the log ring, the process context, and
  // the serialize thread. init never throws; nullptr means profiling stays
  // off — the workload runs unaffected.
  graphsignal::MetricsWriter* writer =
      graphsignal::MetricsWriter::init("cupti", nullptr, write_interval_ns, debug_mode != 0);
  if (!writer) {
    std::fprintf(stderr,
                 "graphsignal: cupti writer init failed; profiling disabled\n");
    g_running.store(false, std::memory_order_relaxed);
    return 0;
  }
  g_writer = writer;

  // Fresh run — instrument pointers belong to the current Writer only.
  {
    std::lock_guard<std::mutex> g(g_instruments_mu);
    g_graph_frames.clear();
    for (auto& s : g_memcpy_bytes) s = KindInstrument{};
    for (auto& s : g_memset_bytes) s = KindInstrument{};
    g_kernels_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_kernels_nanoseconds", {});
    g_graphs_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_graphs_nanoseconds", {});
    g_memcpy_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_memcpy_nanoseconds", {});
    g_memset_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_memset_nanoseconds", {});
    g_sync_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_sync_nanoseconds", {});
    g_graph_trace_mode_gauge = writer->register_instrument(
        graphsignal::InstrumentType::Gauge, "cuda_graph_trace_mode", {});
    g_dropped_records_counter = writer->register_instrument(
        graphsignal::InstrumentType::Counter, "cuda_dropped_records_total", {});
    g_api_profile = writer->register_instrument(
        graphsignal::InstrumentType::Profile, "cuda_api_nanoseconds", {});
  }
  graphsignal::MetricsWriter::set(g_graph_trace_mode_gauge,
                                  static_cast<double>(g_graph_trace_mode));
  {
    std::lock_guard<std::mutex> g(g_graph_sig_mu);
    g_graph_signatures.clear();
  }

  g_writer->set_flush_hook(writer_flush_hook, nullptr);
  g_writer->set_device_probe_reader(device_probe_reader, nullptr);
  g_writer->set_device_probe_batch_reader(device_probe_batch_reader, nullptr);
  g_nvtx_ranges.reset();
  const char* adopt_ninfer_domain = std::getenv("GRAPHSIGNAL_NINFER_NVTX");
  // Which NVTX domains to adopt beyond NInfer's: "all", or a comma-separated
  // list of domain names ("default" names the implicit default domain).
  // Unset keeps the previous behaviour — NInfer only.
  const char* nvtx_domains = std::getenv("GRAPHSIGNAL_NVTX_DOMAINS");
  g_nvtx_ranges.configure(
      writer, adopt_ninfer_domain && (adopt_ninfer_domain[0] == '1' ||
                                     adopt_ninfer_domain[0] == 't' ||
                                     adopt_ninfer_domain[0] == 'T'),
      nvtx_domains);

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));

  // Opt-in CUDA runtime API timing. Unset by default so the profiler costs
  // nothing here; the default selection traces only the allocation and graph
  // APIs rather than all of them, which is what keeps this affordable.
  {
    const char* api_trace = std::getenv("GRAPHSIGNAL_CUDA_API_TRACE");
    const ApiTraceMode mode = api_trace_mode(api_trace);
    if (mode != ApiTraceMode::kOff) {
      const size_t enabled = enable_traced_runtime_apis(mode, api_trace);
      if (g_writer) {
        if (mode == ApiTraceMode::kAll) {
          g_writer->debug("cupti: tracing ALL CUDA runtime APIs (GRAPHSIGNAL_CUDA_API_TRACE=all)");
        } else {
          g_writer->debug("cupti: tracing %zu CUDA runtime API(s) (selection: %s)",
                          enabled, api_trace ? api_trace : "1");
        }
      }
    }
  }

  // Trace CUDA graphs at GRAPH granularity rather than per node. When
  // GRAPH_TRACE is enabled CUPTI stops instrumenting the individual nodes of a
  // launched graph (the graph is reported as a single CUPTI_ACTIVITY_KIND_-
  // GRAPH_TRACE record instead). Lower overhead — same approach Nsight Systems
  // uses with --cuda-graph-trace=graph. Available since CUDA 12.4; if an older
  // CUPTI rejects it, CUPTI_CALL just logs and graph launches go unrecorded.
  //
  // In NODE mode this enable is deliberately skipped: with GRAPH_TRACE off,
  // CUPTI keeps instrumenting the graph's nodes and their kernels arrive
  // through the CONCURRENT_KERNEL kind enabled above, landing in
  // cuda_kernels_nanoseconds by symbol like eager launches. Nothing else
  // changes, so an older CUPTI that rejects a kind cannot break the rest of
  // collection either way.
  if (g_graph_trace_mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE));
  }

  // Observe CUDA graph instantiation so we can compute a stable structural
  // signature per executable graph (used to tag GRAPH_TRACE instruments across
  // processes/hosts). Resource callbacks are passive metadata notifications —
  // they don't instrument the launch path, so they don't reintroduce the
  // per-node graph instrumentation that GRAPH_TRACE avoids.
  if (cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)graph_resource_callback,
                     nullptr) == CUPTI_SUCCESS) {
    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_NVTX));
  }

  std::atexit([]() { cupti_activity_stop(); });

  if (g_graph_trace_env_invalid[0]) {
    g_writer->debug(
        "GRAPHSIGNAL_CUDA_GRAPH_TRACE=%s is not graph|node; using graph",
        g_graph_trace_env_invalid);
  }
  g_writer->debug(
      "cupti_activity_start: write_interval_ns=%llu debug_mode=%u "
      "graph_trace_mode=%s",
      static_cast<unsigned long long>(write_interval_ns), debug_mode,
      g_graph_trace_mode == GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE ? "node" : "graph");
  return 1;
}

void cupti_activity_stop(void) {
  bool expected = true;
  if (!g_running.compare_exchange_strong(expected, false)) return;

  // Stop receiving resource callbacks before tearing CUPTI down. cuptiFinalize()
  // below also unregisters everything, but unsubscribing first is the explicit
  // counterpart to cuptiSubscribe in start.
  if (g_subscriber) {
    cuptiUnsubscribe(g_subscriber);
    g_subscriber = nullptr;
  }

  // Detach CUPTI before our memory can go away. The CUDA_INJECTION64_PATH lib
  // can be unloaded while the process keeps running CUDA; if CUPTI still holds
  // our bufferRequested/bufferCompleted pointers, its next callback jumps into
  // the unmapped library -> SIGSEGV (observed in libcupti during cuLaunchKernel
  // on a forward-compat driver). cuptiFinalize() unregisters all callbacks and
  // disables activities. g_running is already false, so any flush callback it
  // triggers bails immediately (the buffer is freed, no shared state touched).
  cuptiFinalize();

  // Final serialize + writer-thread join. The final write skips the flush hook
  // and the device probe reader refuses because g_running is false — no CUDA
  // on the teardown path. The Writer object is deliberately leaked: no
  // destructor work on teardown.
  if (g_writer) {
    g_writer->debug("cupti_activity_stop");
    g_writer->shutdown();
  }
  g_nvtx_ranges.reset();

  // Safe now: cuptiFinalize() has unregistered the callbacks, so nothing can
  // call buffer_pool_acquire/release afterward.
  buffer_pool_drain();
}

void cupti_activity_set_debug_mode(uint32_t enabled) {
  if (g_writer) g_writer->set_debug(enabled != 0);
}

uint32_t cupti_activity_get_debug_mode(void) {
  return (g_writer && g_writer->debug_enabled()) ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// CUDA injection entry point
// Called by the CUDA driver when CUDA_INJECTION64_PATH points to this library.
// ---------------------------------------------------------------------------

static std::atomic<bool> g_injection_active{false};

static uint32_t read_env_bool(const char* name) {
  const char* s = std::getenv(name);
  if (!s || !*s) return 0;
  if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0) return 1;
  return 0;
}

// Returns the variable's value, or `def` when unset or empty. The pointer is
// getenv's — read it, never keep it.
static const char* read_env_string(const char* name, const char* def) {
  const char* s = std::getenv(name);
  if (!s || !*s) return def;
  return s;
}

extern "C" {

int InitializeInjectionNvtx(void* pfnGetExportTable) {
  return cuptiNvtxInitialize(pfnGetExportTable) == CUPTI_SUCCESS ? 1 : 0;
}

uint32_t cupti_activity_graph_trace_mode_from_env(void) {
  g_graph_trace_env_invalid[0] = '\0';
  const char* s = read_env_string("GRAPHSIGNAL_CUDA_GRAPH_TRACE", nullptr);
  if (!s) return GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;
  if (std::strcmp(s, "node") == 0) return GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE;
  if (std::strcmp(s, "graph") == 0) return GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;
  // Anything else: fall back to the default and keep the offending value for
  // the debug note cupti_activity_start emits once the writer exists.
  std::snprintf(g_graph_trace_env_invalid, sizeof(g_graph_trace_env_invalid),
                "%s", s);
  return GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH;
}

CUptiResult InitializeInjection() {
  bool expected = false;
  if (!g_injection_active.compare_exchange_strong(expected, true,
                                                   std::memory_order_acq_rel))
    return CUPTI_SUCCESS;

  const uint32_t debug_mode = read_env_bool("GRAPHSIGNAL_DEBUG");
  const uint32_t graph_trace_mode = cupti_activity_graph_trace_mode_from_env();

  // Never let an exception (e.g. std::thread creation, bad_alloc) escape into
  // the CUDA driver's init path — that would std::terminate the workload.
  try {
    cupti_activity_start(/*write_interval_ns=*/0, debug_mode, graph_trace_mode);
  } catch (...) {
    // Best-effort: profiling stays off; the workload runs unaffected.
  }

  return CUPTI_SUCCESS;
}

} // extern "C"
