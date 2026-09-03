// End-to-end test for the ROCm profiler (libgsrocmprof.so).
//
// rocprofiler-sdk owns the tool lifecycle: it invokes the library's exported
// rocprofiler_configure (tool_init/tool_fini) when the HIP/HSA runtime
// initializes. In production RocmProfiler.setup_env_vars registers the tool by
// pointing ROCP_TOOL_LIBRARIES at libgsrocmprof.so.
//
// This test therefore reproduces production exactly: it is a pure-HIP workload
// binary (NO reference to any profiler symbol) that is launched with
// ROCP_TOOL_LIBRARIES=<libgsrocmprof.so> (see the `test-rocm-activity` target
// in Makefile.rocm). tool_init fires on the first HIP call; the profiler's
// graphsignal::MetricsWriter serializes cumulative instruments to
// /dev/shm/graphsignal_<pid>/rocm.json every ~1s (full-state overwrite, so
// every read is an immutable snapshot).
//
// The test drives known workloads (memcpy with exact byte totals, two distinct
// kernels launched a fixed number of times, the traced HIP sync ops), then
// polls rocm.json and validates the instruments:
//   - rocm_kernels_nanoseconds      profile, frames = kernel symbols -> cumulative ns
//   - rocm_memcpy_nanoseconds       profile, frames = host_to_device|device_to_host|...
//     + rocm_memcpy_bytes counter, tags {"kind": <kind>}
//   - rocm_sync_nanoseconds         profile, frames = hip op names
//                       (e.g. hipStreamSynchronize) -> cumulative ns
//
// It also registers HOST-storage probes (a histogram and a profile) via
// <graphsignal/probe.h> and asserts the Writer discovers and serializes them.
// probe.h is a standalone header (no profiler symbol), so the binary stays
// pure HIP; the profiler finds the registry via dlsym(RTLD_DEFAULT), which
// requires this executable's dynamic symbols to be exported (-rdynamic in
// TEST_ROCM_ACT_FLAGS).

#include <hip/hip_runtime.h>

#include <graphsignal/probe.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// Simple test framework. g_checks counts every assertion that ran so the output
// makes it obvious the test body actually executed.
static int g_checks = 0;

#define ASSERT(cond, msg) \
  do { \
    ++g_checks; \
    if (!(cond)) { \
      std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      std::abort(); \
    } \
  } while (0)

#define HIP_OK(call, msg) ASSERT((call) == hipSuccess, msg)

#define CASE(name) std::printf("  [case] %s\n", name)

// ---------------------------------------------------------------------------
// rocm.json helpers
// ---------------------------------------------------------------------------

static std::string read_shm_json() {
  char path[256];
  std::snprintf(path, sizeof(path), "/dev/shm/graphsignal_%d/rocm.json",
                static_cast<int>(getpid()));
  FILE* f = std::fopen(path, "r");
  if (!f) return {};
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string content;
  if (sz > 0) {
    content.resize(static_cast<size_t>(sz));
    size_t nread = std::fread(&content[0], 1, static_cast<size_t>(sz), f);
    content.resize(nread);
  }
  std::fclose(f);
  return content;
}

// Splits the metrics array into per-metric text blocks. Every metric object
// starts with {"name":, so splitting on that marker yields one block per
// metric. The trailing "log" array is cut off first so log text can't leak
// into the last metric's block.
static std::vector<std::string> metric_blocks(const std::string& full_json) {
  std::string json = full_json;
  const size_t log_pos = json.rfind(",\"log\":");
  if (log_pos != std::string::npos) json.resize(log_pos);

  std::vector<std::string> out;
  const std::string marker = "{\"name\":";
  size_t pos = json.find(marker);
  while (pos != std::string::npos) {
    size_t next = json.find(marker, pos + marker.size());
    out.push_back(json.substr(pos, (next == std::string::npos ? json.size() : next) - pos));
    pos = next;
  }
  return out;
}

// Extracts an unsigned integer field ("key":123) from a metric block; returns
// UINT64_MAX when the key is absent.
static uint64_t block_u64(const std::string& block, const char* key) {
  std::string k = std::string("\"") + key + "\":";
  size_t p = block.find(k);
  if (p == std::string::npos) return UINT64_MAX;
  return std::strtoull(block.c_str() + p + k.size(), nullptr, 10);
}

// Finds the first metric block with the given name whose text contains
// `tag_sub` (e.g. a tag value). Returns an empty string when not found.
static std::string find_metric(const std::vector<std::string>& blocks,
                               const char* name, const char* tag_sub) {
  std::string name_field = std::string("\"name\":\"") + name + "\"";
  for (const std::string& b : blocks) {
    if (b.find(name_field) != std::string::npos &&
        b.find(tag_sub) != std::string::npos) {
      return b;
    }
  }
  return {};
}

// Extracts the "frames":{...} object of a profile metric block; "" if absent.
static std::string frames_object(const std::string& block) {
  const std::string m = "\"frames\":{";
  size_t p = block.find(m);
  if (p == std::string::npos) return {};
  size_t start = p + m.size() - 1;  // at '{'
  int depth = 0;
  bool in_str = false;
  for (size_t i = start; i < block.size(); ++i) {
    char c = block[i];
    if (in_str) {
      if (c == '\\') { i++; continue; }
      if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      in_str = true;
    } else if (c == '{') {
      depth++;
    } else if (c == '}') {
      depth--;
      if (depth == 0) return block.substr(start, i - start + 1);
    }
  }
  return {};
}

struct Frame {
  std::string name;
  uint64_t value;
  uint64_t samples;
};

// Parses every "name":[value,samples] entry of a profile block's frames
// object, unescaping \" and \\ in frame names.
static std::vector<Frame> parse_frames(const std::string& block) {
  std::vector<Frame> out;
  const std::string fo = frames_object(block);
  size_t i = 1;  // skip '{'
  while (i < fo.size()) {
    if (fo[i] != '"') { ++i; continue; }
    ++i;
    std::string name;
    while (i < fo.size() && fo[i] != '"') {
      if (fo[i] == '\\' && i + 1 < fo.size()) {
        name += fo[i + 1];
        i += 2;
        continue;
      }
      name += fo[i];
      ++i;
    }
    ++i;  // closing quote
    if (i + 1 < fo.size() && fo[i] == ':' && fo[i + 1] == '[') {
      i += 2;
      char* after = nullptr;
      uint64_t value = std::strtoull(fo.c_str() + i, &after, 10);
      uint64_t samples = 0;
      if (after && *after == ',') {
        samples = std::strtoull(after + 1, nullptr, 10);
      }
      out.push_back({std::move(name), value, samples});
    }
  }
  return out;
}

// Value of the frame with the exact name; 0 when absent.
static uint64_t frame_value(const std::string& block, const std::string& name) {
  for (const Frame& f : parse_frames(block)) {
    if (f.name == name) return f.value;
  }
  return 0;
}

// Value of the first frame whose name contains `sub`; 0 when absent. Kernel
// frames carry the compiled symbol name, so lookups match by substring.
static uint64_t frame_value_containing(const std::string& block,
                                       const std::string& sub) {
  for (const Frame& f : parse_frames(block)) {
    if (f.name.find(sub) != std::string::npos) return f.value;
  }
  return 0;
}

// Sums the sparse "counts":[...] array of a histogram block.
static uint64_t counts_total(const std::string& block) {
  const std::string m = "\"counts\":[";
  size_t p = block.find(m);
  if (p == std::string::npos) return 0;
  p += m.size();
  uint64_t total = 0;
  while (p < block.size() && block[p] != ']') {
    total += std::strtoull(block.c_str() + p, nullptr, 10);
    while (p < block.size() && block[p] != ',' && block[p] != ']') ++p;
    if (p < block.size() && block[p] == ',') ++p;
  }
  return total;
}

// Histograms serialize ONLY the sparse bins/counts arrays (no aggregate
// fields); a well-formed one observed at least one value.
static void assert_histogram_sane(const std::string& block, const char* what) {
  ASSERT(!block.empty(), what);
  ASSERT(block.find("\"type\":\"histogram\"") != std::string::npos, what);
  ASSERT(block.find("\"bins\":[") != std::string::npos,
         "histogram must serialize bins");
  ASSERT(counts_total(block) > 0, "histogram counts must sum to > 0");
  ASSERT(block.find("\"count\":") == std::string::npos,
         "histogram must not serialize count");
  ASSERT(block.find("\"sum\":") == std::string::npos,
         "histogram must not serialize sum");
  ASSERT(block.find("\"min\":") == std::string::npos,
         "histogram must not serialize min");
  ASSERT(block.find("\"max\":") == std::string::npos,
         "histogram must not serialize max");
}

// ---------------------------------------------------------------------------
// HIP kernels — distinct symbols so each gets its own rocm_kernels_nanoseconds frame
// ---------------------------------------------------------------------------

__global__ void gs_test_kernel_a(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * 2.0f + 1.0f;
  }
}

__global__ void gs_test_kernel_b(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * 0.5f - 1.0f;
  }
}

int main() {
  std::printf("Running rocm_activity tests...\n");

  int device_count = 0;
  hipError_t st = hipGetDeviceCount(&device_count);
  if (st != hipSuccess || device_count <= 0) {
    // No GPU (or no ROCm runtime): the shm-validation portion can't run. This
    // is expected on CPU-only CI; the writer/probe logic is covered by the
    // pure-C++ common_test / probe_test.
    std::printf("HIP not available (hipGetDeviceCount=%s, devices=%d), "
                "skipping GPU workload test\n",
                hipGetErrorString(st), device_count);
    std::printf("All rocm_activity tests passed! (%d assertions)\n", g_checks);
    return 0;
  }

  std::printf("HIP available (%d device(s)), testing with actual kernels/memcpy...\n",
              device_count);
  HIP_OK(hipSetDevice(0), "hipSetDevice failed");

  // NOTE: there is NO profiler start call here. The rocprofiler-sdk tool
  // (libgsrocmprof.so) must be injected via ROCP_TOOL_LIBRARIES; tool_init runs
  // on the first HIP call above. If the tool was not injected, rocm.json never
  // appears and the assertions below fail loudly (as they should).

  // ------------------------------------------------------------- host probes
  // User probes with HOST storage: the profiler's Writer discovers the
  // registry and serializes them alongside the rocm.* instruments.
  CASE("host probe registration");
  const char* probe_tag_keys[] = {"stage"};
  const char* probe_tag_vals[] = {"decode"};
  graphsignal_probe_entry* probe = graphsignal_probe_register(
      "gstest_batch_duration", GRAPHSIGNAL_HISTOGRAM,
      probe_tag_keys, probe_tag_vals, 1);
  ASSERT(probe != nullptr, "host probe registration failed");
  graphsignal_record(probe, 100);
  graphsignal_record(probe, 200);
  graphsignal_record(probe, 300);

  // A PROFILE probe: frames accumulate exact values.
  graphsignal_probe_entry* profile_probe = graphsignal_probe_register(
      "gstest_host_profile", GRAPHSIGNAL_PROFILE, NULL, NULL, 0);
  ASSERT(profile_probe != nullptr, "host profile probe registration failed");
  graphsignal_profile_add_by_name(profile_probe, "stage.prefill", 111);
  graphsignal_profile_add_by_name(profile_probe, "stage.decode", 222);
  graphsignal_profile_add_by_name(profile_probe, "stage.decode", 333);

  // ------------------------------------------------------------------ memcpy
  // Pinned host memory so every copy is a real DMA transfer reported by the
  // MEMORY_COPY tracing service (pageable copies may be staged differently).
  CASE("memcpy H2D/D2H with known byte totals");
  const int n = 1024;
  const size_t copy_bytes = n * sizeof(float);
  const int kH2DCopies = 3;
  const int kD2HCopies = 2;
  float* h_data = nullptr;
  float* d_data = nullptr;
  HIP_OK(hipHostMalloc(reinterpret_cast<void**>(&h_data), copy_bytes),
         "hipHostMalloc failed");
  HIP_OK(hipMalloc(&d_data, copy_bytes), "hipMalloc failed");
  for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i);

  for (int i = 0; i < kH2DCopies; ++i) {
    HIP_OK(hipMemcpy(d_data, h_data, copy_bytes, hipMemcpyHostToDevice),
           "hipMemcpy H2D failed");
  }
  for (int i = 0; i < kD2HCopies; ++i) {
    HIP_OK(hipMemcpy(h_data, d_data, copy_bytes, hipMemcpyDeviceToHost),
           "hipMemcpy D2H failed");
  }
  HIP_OK(hipDeviceSynchronize(), "sync after memcpy failed");

  // ----------------------------------------------------------------- kernels
  // Two distinct kernel symbols, each launched a fixed number of times.
  CASE("kernel dispatches (2 distinct symbols x N)");
  const int kKernelLaunches = 7;
  const int threads_per_block = 256;
  const int blocks = (n + threads_per_block - 1) / threads_per_block;
  for (int i = 0; i < kKernelLaunches; ++i) {
    gs_test_kernel_a<<<blocks, threads_per_block>>>(d_data, n);
    HIP_OK(hipGetLastError(), "gs_test_kernel_a launch failed");
    gs_test_kernel_b<<<blocks, threads_per_block>>>(d_data, n);
    HIP_OK(hipGetLastError(), "gs_test_kernel_b launch failed");
  }
  HIP_OK(hipDeviceSynchronize(), "sync after kernels failed");

  // -------------------------------------------------------------- HIP syncs
  // Exercise the traced sync ops: hipDeviceSynchronize (above),
  // hipStreamSynchronize, hipEventSynchronize.
  CASE("HIP sync ops");
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;
  HIP_OK(hipStreamCreate(&stream), "hipStreamCreate failed");
  HIP_OK(hipEventCreate(&event), "hipEventCreate failed");
  gs_test_kernel_a<<<blocks, threads_per_block, 0, stream>>>(d_data, n);
  HIP_OK(hipGetLastError(), "stream kernel launch failed");
  HIP_OK(hipStreamSynchronize(stream), "hipStreamSynchronize failed");
  gs_test_kernel_b<<<blocks, threads_per_block, 0, stream>>>(d_data, n);
  HIP_OK(hipGetLastError(), "stream kernel launch failed");
  HIP_OK(hipEventRecord(event, stream), "hipEventRecord failed");
  HIP_OK(hipEventSynchronize(event), "hipEventSynchronize failed");
  HIP_OK(hipEventDestroy(event), "hipEventDestroy failed");
  HIP_OK(hipStreamDestroy(stream), "hipStreamDestroy failed");

  HIP_OK(hipFree(d_data), "hipFree failed");
  HIP_OK(hipHostFree(h_data), "hipHostFree failed");

  // ----------------------------------------------------- collect + validate
  // The Writer flushes the rocprofiler buffer and overwrites rocm.json every
  // ~1s, so poll until a snapshot contains everything the workload produced
  // (worst case: records buffered just after a write need one more interval).
  CASE("collect + validate rocm.json");
  std::string json;
  const auto poll_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (std::chrono::steady_clock::now() < poll_deadline) {
    json = read_shm_json();
    if (json.find("gs_test_kernel_b") != std::string::npos &&
        json.find("hipEventSynchronize") != std::string::npos &&
        json.find("gstest_batch_duration") != std::string::npos &&
        json.find("gstest_host_profile") != std::string::npos) {
      // Candidate snapshot; verify both kernels' cumulative time has landed.
      std::vector<std::string> blocks_now = metric_blocks(json);
      std::string kernels_now =
          find_metric(blocks_now, "rocm_kernels_nanoseconds", "gs_test_kernel_a");
      if (!kernels_now.empty() &&
          frame_value_containing(kernels_now, "gs_test_kernel_a") > 0 &&
          frame_value_containing(kernels_now, "gs_test_kernel_b") > 0) {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::printf("Read %zu bytes of rocm.json\n", json.size());
  if (!json.empty()) {
    std::printf("rocm.json (first 512 bytes):\n%.512s\n", json.c_str());
  }

  ASSERT(!json.empty(),
         "expected rocm.json — is libgsrocmprof.so injected via "
         "ROCP_TOOL_LIBRARIES? (see Makefile.rocm test-rocm-activity)");
  ASSERT(json.find("\"version\":1") != std::string::npos,
         "rocm.json must carry version 1");
  ASSERT(json.find("\"log\":") != std::string::npos,
         "rocm.json should carry a log (GRAPHSIGNAL_DEBUG=1 logs tool_init)");

  std::vector<std::string> blocks_v = metric_blocks(json);

  // Kernels: one profile instrument; frames keyed by kernel symbol, values =
  // cumulative duration ns.
  const std::string kernels =
      find_metric(blocks_v, "rocm_kernels_nanoseconds", "\"frames\":{");
  ASSERT(!kernels.empty(), "rocm_kernels_nanoseconds profile missing");
  ASSERT(kernels.find("\"type\":\"profile\"") != std::string::npos,
         "rocm_kernels_nanoseconds must serialize as a profile");
  ASSERT(frame_value_containing(kernels, "gs_test_kernel_a") > 0,
         "rocm_kernels_nanoseconds frame for gs_test_kernel_a missing");
  ASSERT(frame_value_containing(kernels, "gs_test_kernel_b") > 0,
         "rocm_kernels_nanoseconds frame for gs_test_kernel_b missing");

  // Memcpy: per-kind duration frames + exact cumulative byte totals.
  const std::string memcpy_prof =
      find_metric(blocks_v, "rocm_memcpy_nanoseconds", "\"frames\":{");
  ASSERT(!memcpy_prof.empty(), "rocm_memcpy_nanoseconds profile missing");
  ASSERT(memcpy_prof.find("\"type\":\"profile\"") != std::string::npos,
         "rocm_memcpy_nanoseconds must serialize as a profile");
  ASSERT(frame_value(memcpy_prof, "host_to_device") > 0,
         "rocm_memcpy_nanoseconds frame for host_to_device missing");
  ASSERT(frame_value(memcpy_prof, "device_to_host") > 0,
         "rocm_memcpy_nanoseconds frame for device_to_host missing");

  const std::string h2d_bytes = find_metric(
      blocks_v, "rocm_memcpy_bytes", "\"kind\":\"host_to_device\"");
  const std::string d2h_bytes = find_metric(
      blocks_v, "rocm_memcpy_bytes", "\"kind\":\"device_to_host\"");
  ASSERT(!h2d_bytes.empty(), "rocm_memcpy_bytes host_to_device missing");
  ASSERT(!d2h_bytes.empty(), "rocm_memcpy_bytes device_to_host missing");
  ASSERT(block_u64(h2d_bytes, "value") == kH2DCopies * copy_bytes,
         "host_to_device byte total must match the copies issued");
  ASSERT(block_u64(d2h_bytes, "value") == kD2HCopies * copy_bytes,
         "device_to_host byte total must match the copies issued");

  // Sync: one profile; frames keyed by the HIP op name, one per op exercised.
  const std::string sync_prof =
      find_metric(blocks_v, "rocm_sync_nanoseconds", "\"frames\":{");
  ASSERT(!sync_prof.empty(), "rocm_sync_nanoseconds profile missing");
  ASSERT(sync_prof.find("\"type\":\"profile\"") != std::string::npos,
         "rocm_sync_nanoseconds must serialize as a profile");
  ASSERT(frame_value(sync_prof, "hipDeviceSynchronize") > 0,
         "rocm_sync_nanoseconds frame for hipDeviceSynchronize missing");
  ASSERT(frame_value(sync_prof, "hipStreamSynchronize") > 0,
         "rocm_sync_nanoseconds frame for hipStreamSynchronize missing");
  ASSERT(frame_value(sync_prof, "hipEventSynchronize") > 0,
         "rocm_sync_nanoseconds frame for hipEventSynchronize missing");

  // Host histogram probe: discovered from the probe.h registry, serialized
  // with its tags and ONLY bins/counts. 100/200/300 land in the log-linear
  // bins with lower bounds 96/192/256, one observation each.
  const std::string probe_block = find_metric(
      blocks_v, "gstest_batch_duration", "\"stage\":\"decode\"");
  assert_histogram_sane(probe_block, "host probe gstest_batch_duration missing");
  ASSERT(probe_block.find("\"bins\":[96,192,256],\"counts\":[1,1,1]") !=
             std::string::npos,
         "host probe bins/counts must match the recorded values exactly");

  // Host profile probe: frames with exact cumulative values.
  const std::string profile_block =
      find_metric(blocks_v, "gstest_host_profile", "\"frames\":{");
  ASSERT(!profile_block.empty(), "host profile probe gstest_host_profile missing");
  ASSERT(profile_block.find("\"type\":\"profile\"") != std::string::npos,
         "host profile probe must serialize as a profile");
  ASSERT(frame_value(profile_block, "stage.prefill") == 111,
         "host profile probe frame stage.prefill must be 111");
  ASSERT(frame_value(profile_block, "stage.decode") == 555,
         "host profile probe frame stage.decode must accumulate 222+333");

  // write_ts must be monotone across two reads (the Writer overwrites the file
  // every interval).
  CASE("monotone write_ts");
  const uint64_t write_ts_1 = block_u64(json, "write_ts");
  ASSERT(write_ts_1 != UINT64_MAX && write_ts_1 > 0, "write_ts missing");
  uint64_t write_ts_2 = 0;
  const auto ts_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < ts_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    write_ts_2 = block_u64(read_shm_json(), "write_ts");
    if (write_ts_2 != UINT64_MAX && write_ts_2 > write_ts_1) break;
  }
  ASSERT(write_ts_2 != UINT64_MAX && write_ts_2 > write_ts_1,
         "write_ts must increase across writes");

  std::printf("All rocm_activity tests passed! (%d assertions)\n", g_checks);
  return 0;
}
