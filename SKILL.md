---
name: graphsignal
description: >-
  Profile AI inference workloads (vLLM, SGLang, TensorRT-LLM, PyTorch, any GPU
  application) with the Graphsignal profiler and read the results from its
  local /signals JSON endpoint. Use when the user wants GPU profiling,
  kernel/CUDA-graph timing, engine metrics, or error monitoring for a workload,
  asks about `graphsignal-run`, or wants an AI agent to inspect a running
  workload's performance.
---

# Graphsignal Profiler

Graphsignal is a GPU profiler built to be operated by an AI agent: install it, launch the workload under it, benchmark it, read the signals, change flags or code, measure again.

It observes a workload from a **sidecar process**: `graphsignal-run` launches the workload unmodified — no code changes, no imports — injects the CUPTI/ROCm activity library into it, and starts a watcher process that collects everything and serves it as one JSON document at `http://127.0.0.1:18259/signals`. Collected out of the box: GPU kernel/graph/transfer/sync timing (CUDA and ROCm), NVML device telemetry, process and host metrics, the engine's own Prometheus metrics (vLLM, SGLang, TensorRT-LLM), and errors extracted from console output. No profiling data is uploaded anywhere unless `GRAPHSIGNAL_API_KEY` is set.

## Install

Install as a uv tool, isolated from the workload environment:

```bash
UV_TOOL_BIN_DIR=/usr/local/bin uv tool install 'graphsignal[cu12]'   # CUDA 12.x
# or
UV_TOOL_BIN_DIR=/usr/local/bin uv tool install 'graphsignal[cu13]'   # CUDA 13.x
```

`UV_TOOL_BIN_DIR=/usr/local/bin` puts `graphsignal-run` on `PATH` for every shell, including non-interactive scripts and containers. `pip install 'graphsignal[cu12]'` works too. The `cu12`/`cu13` extras are Linux-only, needed for GPU profiling; on AMD install without an extra (`uv tool install graphsignal`) — GPU profiling goes through AMD's `rocprofiler-sdk`, which ships with ROCm 7+.

Root is not required. Verify the install with `graphsignal-run --version`, and readiness after a launch with `curl -s http://127.0.0.1:18259/health` → `{"status": "ok"}`.

## Run

Wrap the workload's launch command:

```bash
graphsignal-run vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
graphsignal-run sglang serve --model-path <model> --port 8000
graphsignal-run trtllm-serve <model> --port 8000
graphsignal-run python my_app.py
```

Leading flags (before the command):

- `--metrics-port PORT` — where to scrape the engine's Prometheus metrics (default: derived from the engine's `--port`).
- `--listen-host HOST` — host to bind the `/signals` endpoint to (default `127.0.0.1`; set e.g. `0.0.0.0` to expose it for remote access; also settable via `GRAPHSIGNAL_LISTEN_HOST`).
- `--listen-port PORT` — port for the `/signals` endpoint (default `18259`; also settable via `GRAPHSIGNAL_LISTEN_PORT`).

Set `GRAPHSIGNAL_DEBUG=1` for verbose diagnostics.

## Typical workflow (reference)

One optimization iteration, for reference — not a prescription. Steps marked *optional* are skipped when they don't apply, and the order is yours to adapt:

1. Change code or config — whatever the previous iteration pointed at (flags, engine settings, a kernel).
2. Add probes (*optional* — when the built-ins can't see the suspect; see GPU probes below).
3. Compile (*optional* — only when code or probes changed).
4. Run via `graphsignal-run`.
5. Run a benchmark, or put representative load on the workload.
6. Get signals from `/signals`.
7. Identify bottlenecks (the read order below).
8. Next iteration — keep what helped, revert what didn't.

## Read signals

While the workload is running:

```bash
curl -s http://127.0.0.1:18259/signals
```

`GET /health` returns `{"status": "ok"}` when the watcher is up. **The endpoint lives only as long as the profiled workload** — read it while the process runs; after a crash, the last errors are in the workload's console output.

### Reaching it from another machine or container

The endpoint binds to `127.0.0.1`, so by default it is readable only from where the workload runs. Two ways across, both of which keep it on loopback at each end:

```bash
# Workload over ssh — forward the port, change nothing on the remote side
ssh -N -L 18259:127.0.0.1:18259 <host>

# Workload in Docker — bind past the container's loopback, publish to the host's
docker run -p 127.0.0.1:18259:18259 ... \
  graphsignal-run --listen-host 0.0.0.0 <workload command>
```

Then read `http://127.0.0.1:18259/signals` locally either way. On Linux, `docker run --network host` avoids needing both.

**`--listen-host 0.0.0.0` is the part to be careful with**: the endpoint is read-only but unauthenticated, so anything that can reach the address can read the profiling data. Inside a container it is the only way out, which is why the publish above pins it to the host's `127.0.0.1` rather than to every interface. Never pass `--listen-host 0.0.0.0` on a host that is directly on an untrusted network.

### Response shape

```json
{
  "profiler": {"version": "..."},
  "payload_ns": 1756150000000000000,
  "start_ns": 1756149000000000000,
  "context": {"instance.id": "a1b2c3", "host.name": "gpu-node-1"},
  "metrics": [
    {"name": "gpu_utilization_percent", "type": "gauge",
     "tags": {"device.uuid": "GPU-..."},
     "stats": {"value": 88.0},
     "updated_ns": 1756150000000000000},
    {"name": "cuda_kernels_nanoseconds", "type": "profile",
     "tags": {"process.pid": "123"},
     "stats": {"frames": [
       {"name": "<raw kernel symbol>", "value": 8100000000, "samples": 3200},
       {"name": "<another kernel symbol>", "value": 550000000, "samples": 410}]},
     "updated_ns": 1756150000000000000},
    {"name": "vllm:e2e_request_latency_seconds", "type": "histogram",
     "tags": {"process.pid": "123"},
     "stats": {"mean": 0.42, "p50": 0.3, "p95": 1.5},
     "updated_ns": 1756150000000000000},
    {"name": "cuda_memcpy_bytes", "type": "counter",
     "tags": {"kind": "host_to_device", "process.pid": "123"},
     "stats": {"total": 1048576}, "updated_ns": 1756150000000000000}
  ],
  "errors": [
    {"level": "error", "log_ns": 1756150000000000000, "message": "CUDA out of memory ...",
     "exception": "Traceback ...", "tags": {"process.pid": "123", "stream": "stderr"}}
  ],
  "resources": [
    {"kind": "host", "tags": {}, "attributes": {"platform.name": "Linux"},
     "first_seen_ts": 1756150000, "last_seen_ts": 1756150000},
    {"kind": "process", "tags": {"process.pid": "123", "process.root_pid": "123"},
     "attributes": {"process.command_line": "..."}, "first_seen_ts": 1756150000, "last_seen_ts": 1756150000},
    {"kind": "device", "tags": {"device.uuid": "GPU-..."},
     "attributes": {"device.name": "NVIDIA ...", "mem_total": "..."}, "first_seen_ts": 1756150000, "last_seen_ts": 1756150000}
  ]
}
```

### Semantics

- **`null` = not measured, `0` = measured zero.**
- **Units live in field names**: a `_ns` field is epoch nanoseconds, a `_ts` field is epoch seconds, and query parameters without a suffix are epoch seconds.
- `start_ns` is when this instance started — the base every cumulative stat accumulates from.
- **gauge** — the latest sampled value (`stats.value`).
- **counter** — cumulative total since `start_ns` (`stats.total`).
- **summary** — cumulative `count`, `sum`, and derived `avg`.
- **histogram** — a cumulative distribution: `mean`, `p50`, `p95` computed from its bins (quantiles are bin values; resolution ≤25%).
- **profile** — cumulative counters per named frame (e.g. time per kernel) with the number of samples that produced each, as `stats.frames` sorted by value descending.
- Metrics not updated for 10 minutes disappear from the report (dead workers age out).

### Key metric families

- `cuda_kernels_nanoseconds` / `rocm_kernels_nanoseconds` — profile of cumulative execution time per kernel (frames are raw mangled symbols).
- `cuda_graphs_nanoseconds` — profile of cumulative replay time per unique CUDA-graph structure.
- `cuda_memcpy_nanoseconds` / `cuda_memset_nanoseconds` / `cuda_sync_nanoseconds` — profiles of cumulative time per transfer kind / sync type; `cuda_memcpy_bytes{kind}` / `cuda_memset_bytes{kind}` counters carry the volumes.
- `gpu_*` — NVML telemetry per device (utilization, memory, power, clocks, throttling, NVLink/PCIe, `gpu_xid_critical_errors`).
- `process_*`, `host_*` — CPU/memory per process and host.
- Engine metrics scraped from Prometheus (vLLM `vllm:*`, SGLang `sglang:*`, TRT-LLM) appear under their original names — latency metrics as both a summary (count/sum) and a histogram (distribution) of the same name.
- User probe metrics (see GPU probes below) appear under their registered names.

### How to interpret (suggested order)

1. Check `errors` first — a crash or XID error explains more than any metric.
2. Check `gpu_utilization_percent` and `gpu_memory_*` per device — is the GPU busy, starved, or memory-bound?
3. Read the `cuda_kernels_nanoseconds` and `cuda_graphs_nanoseconds` profiles — which frames dominate cumulative time (they are sorted descending)?
4. Check the `cuda_sync_nanoseconds` profile and `cuda_memcpy_nanoseconds` profile/byte counters — heavy host synchronization or transfer volume signals CPU/IO bottlenecks.
5. Correlate with engine metrics (queue depth, running requests, token throughput) from the Prometheus import.

Each read is the latest snapshot; trends come from diffing cumulative counts/sums between reads. When and how often to read is yours to decide — per benchmark run, per iteration, or continuously.

## GPU probes (instrumenting code)

The built-in `cuda_kernels_nanoseconds` profile already breaks time down per kernel symbol, for free. Probes are for what CUPTI structurally cannot see: the stages *inside* one kernel — a megakernel's phases, a fused op's steps — and anything whose name is a concept in your code rather than a linker symbol (per layer, per op, per request stage, queue depths). Reach for them when a built-in profile names the hot kernel and the next question is which part of it.

The probe API is a single vendorable header (Apache-2.0, C++17, no dependencies); an AI agent can insert probes, rebuild, rerun under `graphsignal-run`, and read the results from `/signals` in the same loop:

```bash
curl -fsSL https://raw.githubusercontent.com/graphsignal/graphsignal/main/include/graphsignal/probe.h -o third_party/graphsignal/probe.h --create-dirs
```

**Build requirements.** The file you edit is usually one `nvcc` (or `hipcc`) compiles, since the probe goes inside the kernel. Compile that translation unit with `-std=c++17` and put the vendored directory on the include path — `nvcc -std=c++17 -Ithird_party ...`. The header `#error`s on anything older, so a target still defaulting to `gnu++14` fails immediately and reads like a broken header. For device probes also vendor `probe_cuda.h` (or `probe_rocm.h`) beside `probe.h`.

Four instrument types, registered once (idempotent per name+tags) and recorded lock-free; every record function is a safe no-op on NULL:

```cpp
#include <graphsignal/probe.h>

static graphsignal_probe_entry* g_batch_ns;   // GRAPHSIGNAL_HISTOGRAM
static graphsignal_probe_entry* g_ops;        // GRAPHSIGNAL_PROFILE

void init() {
    g_batch_ns = graphsignal_probe_register(
        "myengine_batch_duration_nanoseconds", GRAPHSIGNAL_HISTOGRAM, NULL, NULL, 0);
    g_ops = graphsignal_probe_register(
        "myengine_ops_nanoseconds", GRAPHSIGNAL_PROFILE, NULL, NULL, 0);
}

void run_op(const char* op_name) {
    uint64_t t0 = graphsignal_now_ns();
    // ... work ...
    graphsignal_record(g_batch_ns, graphsignal_now_ns() - t0);                     // histogram
    graphsignal_profile_add_by_name(g_ops, op_name, graphsignal_now_ns() - t0);   // profile frame
}
```

`GRAPHSIGNAL_GAUGE` (`graphsignal_set`) and `GRAPHSIGNAL_COUNTER` (`graphsignal_add`) work the same way. A profile maps up to 250 named frames to a cumulative value plus a sample count — pick it when the breakdown itself is the question (time per op, per layer, per anything nameable).

Inside CUDA kernels, register with `<graphsignal/probe_cuda.h>` and record from device code (HIP: `<graphsignal/probe_rocm.h>`). Device-storage instruments are **histograms only** — register on the host, pass `graphsignal_probe_device_data(...)` into the launch, and record from whichever lane you choose (one per block is the usual answer):

```cpp
static graphsignal_probe_entry* g_tile_ns;

void init() {
    g_tile_ns = graphsignal_probe_register_cuda(
        "mykernels_tile_duration_nanoseconds", NULL, NULL, 0);
}

__global__ void my_kernel(..., graphsignal_instrument_data* tile_ns) {
    unsigned long long t0 = graphsignal_gtimer();
    // ... tile work ...
    if (threadIdx.x == 0) {  // one recording lane per block
        GRAPHSIGNAL_RECORD(tile_ns, graphsignal_gtimer() - t0);
    }
}

// launch: my_kernel<<<...>>>(..., graphsignal_probe_device_data(g_tile_ns));
```

On CUDA `graphsignal_gtimer()` reads `%globaltimer` and is already nanoseconds. **On HIP it is `wall_clock64()` ticks, not nanoseconds** — either convert with `hipDeviceAttributeWallClockRate` (kHz) or record raw ticks and name the metric for what it holds, so a `_nanoseconds` suffix never lies about ROCm numbers.

Conventions and semantics:

- Values are unsigned 64-bit integers (nanoseconds, bytes, counts). Instruments are cumulative forever — never reset; readers snapshot and diff.
- Name metrics with underscores and a unit suffix, like the built-ins: `myengine_batch_duration_nanoseconds`, `myengine_bytes_in`.
- Give a rebuild two ticks before you read it. A newly registered probe reaches the shm file on the writer's next 1s tick and `/signals` on the recorder's next 1s tick — let the workload run ~3s before curling, or a probe that works looks like one that never registered.
- Registration can fail silently, and this is the one failure worth guarding. Past 4096 instruments per process (or on allocation failure) `graphsignal_probe_register` returns `NULL`, every record on it becomes a no-op, and nothing is logged — the only evidence is the `graphsignal_probe_dropped_instruments` counter in `/signals`. Register each instrument once at init; never register per call in a loop over shapes, layer names, or request ids.
- Probes are inert without a reader. Under `graphsignal-run`, values appear in `/signals` automatically alongside the built-in metrics — histograms as summary + histogram stats, profiles as frames with values and sample counts.
- Full instructions: https://graphsignal.com/docs/guides/gpu-probes/

## Production feedback loop (optional)

With `GRAPHSIGNAL_API_KEY` set, the profiler also uploads signals to Graphsignal so they can be retrieved later from the server via the Signals API. Every request takes the key in an `X-API-Key` header; the base URL is `https://api.graphsignal.com` unless overridden with `GRAPHSIGNAL_API_BASE`.

List instances that have reported (each profiler run is one instance, identified by its `instance.id` tag):

```bash
curl -s -H "X-API-Key: $GRAPHSIGNAL_API_KEY" \
  "https://api.graphsignal.com/api/v1/instances"
```

Returns `{"instances": [{"instance_id", "resource_id", "tags", "attributes", "first_seen_ts", "last_seen_ts"}]}` for the last 24 hours by default; pass `start`/`end` (epoch seconds) for another window.

Read an instance's counter snapshot — the same shape as the local `/signals` payload above (the top block is `platform` instead of `profiler`):

```bash
curl -s -H "X-API-Key: $GRAPHSIGNAL_API_KEY" \
  "https://api.graphsignal.com/api/v1/signals?instance_id=<id>&start=<epoch seconds>"
```

`instance_id` is required (take it from `/api/v1/instances` or from the local payload's `context`). `start` is required: counters, summaries, histograms and profiles accumulate from that instant to the snapshot instant, rather than from run start — production instances can run longer than the server retains data. The payload echoes it as `start_ns`, where the local endpoint reports the instance start. Pass `end=<epoch seconds>` for a snapshot as of that moment instead of now. Consecutive reads tile without double counting when each passes the previous response's `payload_ns / 1e9` as its `start`.


## Reporting issues

If you find a **real bug in the profiler** — not a misconfiguration and not a problem in the workload — write up a report and ask the user to send it. Do not send it yourself.

It is a profiler bug when the profiler itself misbehaves: it crashes or hangs, `/signals` returns malformed JSON or an obviously wrong value (a negative duration, a counter that decreases, a metric that vanishes while the workload is plainly still doing the work), a probe that registered never appears, `graphsignal-run` mangles the command line it was given, or the profiler changes the workload's behavior.

It is **not** a profiler bug when: the workload itself fails (read `errors` and the console output — a CUDA OOM or an engine argument error is the workload's), a metric is `null` because nothing measured it (no GPU, no Prometheus endpoint, no `cu12`/`cu13` extra installed), `/signals` is unreachable because the workload already exited or is bound to another port, or a probe returned `NULL` at registration (check `graphsignal_probe_dropped_instruments`). Rule out all of these first — say which ones you ruled out in the report.

Collect, and reproduce with `GRAPHSIGNAL_DEBUG=1` before writing it up:

```bash
graphsignal-run --version
python -c "import platform,sys; print(platform.platform(), sys.version)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   # or rocm-smi
GRAPHSIGNAL_DEBUG=1 graphsignal-run <the exact command> 2>&1 | tail -50
```

The report should state: the profiler version and platform above, the exact `graphsignal-run` command, what you expected and what happened instead, the smallest reproduction you found, the `GRAPHSIGNAL_DEBUG=1` output, and the relevant slice of the `/signals` payload. **Redact before showing it to the user** — a payload carries command lines, host names and tags, which may hold model paths or internal identifiers.

Then give the user the report and the two ways to send it:

- email `support@graphsignal.com`
- open an issue at https://github.com/graphsignal/graphsignal/issues

Sending it is the user's decision, so present the report and let them choose. If the bug blocks the work, say what you are doing instead — a workaround, or continuing without the affected metric.
