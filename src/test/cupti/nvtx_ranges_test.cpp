#include "../../cupti/nvtx_ranges.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define ASSERT(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s: %s\n", __FILE__, m); std::abort(); } } while (0)

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
  a.reset();
  w->shutdown();
  return 0;
}
