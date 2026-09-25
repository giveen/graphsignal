#include "../../cupti/nvtx_ranges.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define ASSERT(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s: %s\n", __FILE__, m); std::abort(); } } while (0)

// Reads a counter's current value out of the serialized payload. The writer is
// shared across the blocks below, so instruments (and their cumulative totals)
// outlive an aggregator reset; a delta is what proves a range was or was not
// attributed to a domain.
static long long counter_value(const std::string& json, const std::string& name) {
  const std::string needle = "\"name\":\"" + name + "\"";
  const size_t at = json.find(needle);
  if (at == std::string::npos) return -1;
  const size_t v = json.find("\"value\":", at);
  if (v == std::string::npos) return -1;
  return std::strtoll(json.c_str() + v + 8, nullptr, 10);
}

static std::string read_all(const std::string& path) {
  std::string s;
  FILE* f = std::fopen(path.c_str(), "r");
  char b[8192];
  size_t n;
  while ((n = std::fread(b, 1, sizeof(b), f))) s.append(b, n);
  std::fclose(f);
  return s;
}

int main() {
  graphsignal::MetricsWriter* w = graphsignal::MetricsWriter::init(
      "nvtx_test", "/tmp", 3600000000000ull, false);
  ASSERT(w != nullptr, "writer init");
  graphsignal::NvtxRangeAggregator a;
  a.configure(w);
  a.on_domain_create(7, "other");
  a.on_start(7, "runtime", 90);
  a.on_end(7, 90, true);
  a.on_domain_create(9, "ninfer");
  a.on_start(9, "runtime request=123", 1);
  a.on_start(9, "prefill request=123", 0, false);
  a.on_start(9, "prefill nested", 0, false);
  a.on_end(9, 0, false);
  a.on_end(9, 0, false);
  a.on_end(9, 1, true);
  a.on_start(9, "decode", 0, false);
  a.on_start(9, "decode nested", 0, false);
  a.on_end(9, 0, false);
  a.on_end(9, 0, false);
  a.on_start_category(9, 4, "decode.mtp_round", 7, true);
  a.on_start_category(9, 4, "decode.mtp_round", 8, true);
  a.on_end(9, 7, true);
  a.on_end(9, 8, true);
  a.on_start(9, "unknown range", 0, false);
  a.on_end(9, 0, false);
  w->write_now(false);
  const std::string json = [&] { std::string s; FILE* f = std::fopen(w->file_path().c_str(), "r"); char b[4096]; size_t n; while ((n = std::fread(b, 1, sizeof(b), f))) s.append(b, n); std::fclose(f); return s; }();
  ASSERT(json.find("ninfer_runtime_total") != std::string::npos, "runtime metric");
  ASSERT(json.find("\"name\":\"ninfer_prefill_total\",\"type\":\"counter\",\"tags\":{},\"value\":1") != std::string::npos, "cross-category nested range recorded");
  ASSERT(json.find("\"name\":\"ninfer_decode_total\",\"type\":\"counter\",\"tags\":{},\"value\":1") != std::string::npos, "same-category nesting suppressed");
  ASSERT(json.find("\"name\":\"ninfer_mtp_total\",\"type\":\"counter\",\"tags\":{},\"value\":2") != std::string::npos, "overlapping async ranges are independent");
  ASSERT(json.find("ninfer_unknown_total") == std::string::npos, "unknown range dropped");
  ASSERT(std::string(graphsignal::NvtxRangeAggregator::category("CUDA Graph replay")) == "cuda_graph", "graph class");
  a.reset();
  a.configure(w, true);
  a.on_start_category(11, 2, nullptr, 0, false);
  a.on_end(11, 0, false);
  w->write_now(false);
  const std::string adopted_json = [&] { std::string s; FILE* f = std::fopen(w->file_path().c_str(), "r"); char b[4096]; size_t n; while ((n = std::fread(b, 1, sizeof(b), f))) s.append(b, n); std::fclose(f); return s; }();
  ASSERT(adopted_json.find("\"name\":\"ninfer_prefill_total\",\"type\":\"counter\",\"tags\":{},\"value\":2") != std::string::npos, "launcher-adopted existing domain");

  // --- A second engine's domain, adopted by name, reports the same category
  // structure under the generic nvtx_* names rather than ninfer_*.
  a.reset();
  a.configure(w, false, "vllm");
  a.on_domain_create(21, "unlisted");          // not named -> ignored
  a.on_start(21, "prefill", 1, true);
  a.on_end(21, 1, true);
  a.on_domain_create(22, "vllm");              // named -> adopted
  a.on_start(22, "prefill step=3", 2, true);
  a.on_end(22, 2, true);
  a.on_start(22, "decode step=3", 3, true);
  a.on_end(22, 3, true);
  // Marks are point events: counted, not timed.
  a.on_mark(22, "scheduler_step");
  a.on_mark(22, "scheduler_step");
  a.on_mark(21, "ignored_domain_mark");
  w->write_now(false);
  const std::string generic_json = [&] { std::string s; FILE* f = std::fopen(w->file_path().c_str(), "r"); char b[8192]; size_t n; while ((n = std::fread(b, 1, sizeof(b), f))) s.append(b, n); std::fclose(f); return s; }();
  ASSERT(generic_json.find("\"name\":\"nvtx_prefill_total\",\"type\":\"counter\",\"tags\":{},\"value\":1") != std::string::npos, "adopted domain reports under nvtx_ prefix");
  ASSERT(generic_json.find("\"name\":\"nvtx_decode_total\",\"type\":\"counter\",\"tags\":{},\"value\":1") != std::string::npos, "adopted domain decode category");
  ASSERT(generic_json.find("nvtx_prefill_nanoseconds") != std::string::npos, "adopted domain records durations too");
  ASSERT(generic_json.find("\"name\":\"nvtx_marks_total\",\"type\":\"counter\",\"tags\":{\"name\":\"scheduler_step\"},\"value\":2") != std::string::npos, "marks are counted per name");
  ASSERT(generic_json.find("ignored_domain_mark") == std::string::npos, "unlisted domain is not adopted");
  // The unlisted domain's prefill must not have landed in either prefix.
  ASSERT(generic_json.find("\"name\":\"nvtx_prefill_total\",\"type\":\"counter\",\"tags\":{},\"value\":2") == std::string::npos, "unlisted domain contributes nothing");

  // --- "all" adopts every domain, including the implicit default one. A named
  // domain still has to be observed being created (the default domain never is).
  a.reset();
  a.configure(w, false, "all");
  a.on_domain_create(31, "vllm");
  a.on_start(31, "prefill", 4, true);
  a.on_end(31, 4, true);
  a.on_start(0, "prefill default-domain", 5, true);  // default domain
  a.on_end(0, 5, true);
  a.on_mark(0, "default_mark");
  w->write_now(false);
  const std::string all_json = [&] { std::string s; FILE* f = std::fopen(w->file_path().c_str(), "r"); char b[8192]; size_t n; while ((n = std::fread(b, 1, sizeof(b), f))) s.append(b, n); std::fclose(f); return s; }();
  ASSERT(all_json.find("\"name\":\"nvtx_prefill_total\",\"type\":\"counter\",\"tags\":{},\"value\":2") != std::string::npos, "all adopts named and default domains");
  ASSERT(all_json.find("\"name\":\"nvtx_marks_total\",\"type\":\"counter\",\"tags\":{\"name\":\"default_mark\"},\"value\":1") != std::string::npos, "default-domain mark counted");

  // --- Unset (the default) keeps NInfer-only behaviour: another engine's
  // domain is not adopted, so its ranges are not attributed anywhere.
  const long long nvtx_prefill_before = counter_value(read_all(w->file_path()), "nvtx_prefill_total");
  const long long ninfer_prefill_before = counter_value(read_all(w->file_path()), "ninfer_prefill_total");
  a.reset();
  a.configure(w);
  a.on_domain_create(41, "vllm");
  a.on_start(41, "prefill", 6, true);
  a.on_end(41, 6, true);
  a.on_domain_create(42, "ninfer");
  a.on_start(42, "prefill", 7, true);
  a.on_end(42, 7, true);
  w->write_now(false);
  const std::string ninfer_only_json = read_all(w->file_path());
  ASSERT(ninfer_prefill_before >= 0 && ninfer_prefill_before < 1000000, "ninfer prefill counter readable");
  ASSERT(counter_value(ninfer_only_json, "ninfer_prefill_total") == ninfer_prefill_before + 1,
         "ninfer domain still adopted by default");
  ASSERT(counter_value(ninfer_only_json, "nvtx_prefill_total") == nvtx_prefill_before,
         "other engines stay off unless asked for");

  a.reset();
  w->shutdown();
  return 0;
}
