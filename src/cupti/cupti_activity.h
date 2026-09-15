#pragma once

#include <cstdint>

extern "C" {

// CUDA graph tracing granularity.
//   GRAPH (default): CUPTI_ACTIVITY_KIND_GRAPH_TRACE is enabled, so a graph
//     replay arrives as one record and the graph's kernels are NOT
//     instrumented individually -> cuda_graphs_nanoseconds, cheap.
//   NODE: GRAPH_TRACE is left disabled, so the graph's kernels arrive through
//     CONCURRENT_KERNEL like eager launches -> cuda_kernels_nanoseconds, at a
//     higher per-node CUPTI cost, with cuda_graphs_nanoseconds left empty.
// Selected by GRAPHSIGNAL_CUDA_GRAPH_TRACE=graph|node and reported through the
// cuda_graph_trace_mode gauge.
#define GRAPHSIGNAL_GRAPH_TRACE_MODE_GRAPH 0u
#define GRAPHSIGNAL_GRAPH_TRACE_MODE_NODE 1u

// Reads GRAPHSIGNAL_CUDA_GRAPH_TRACE and maps it to one of the modes above.
// Unset, empty or unrecognized values yield GRAPH; the note about an
// unrecognized value is emitted by cupti_activity_start, once the writer's log
// ring exists. Never throws and makes no CUDA calls.
uint32_t cupti_activity_graph_trace_mode_from_env(void);

// Starts CUPTI activity collection and the shm metric writer
// (/dev/shm/graphsignal_<pid>/cupti.json). write_interval_ns is the writer's
// serialize interval; 0 selects the 1s default. graph_trace_mode is one of
// GRAPHSIGNAL_GRAPH_TRACE_MODE_*; anything else is treated as GRAPH. Returns 1
// when profiling is running (including when it already was), 0 when the writer
// could not be initialized — profiling stays off and the process is
// unaffected.
int cupti_activity_start(uint64_t write_interval_ns, uint32_t debug_mode,
                         uint32_t graph_trace_mode);
void cupti_activity_stop(void);
void cupti_activity_set_debug_mode(uint32_t enabled);
uint32_t cupti_activity_get_debug_mode(void);

}  // extern "C"
