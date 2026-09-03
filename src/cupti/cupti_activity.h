#pragma once

#include <cstdint>

extern "C" {

// Starts CUPTI activity collection and the shm metric writer
// (/dev/shm/graphsignal_<pid>/cupti.json). write_interval_ns is the writer's
// serialize interval; 0 selects the 1s default. Returns 1 when profiling is
// running (including when it already was), 0 when the writer could not be
// initialized — profiling stays off and the process is unaffected.
int cupti_activity_start(uint64_t write_interval_ns, uint32_t debug_mode);
void cupti_activity_stop(void);
void cupti_activity_set_debug_mode(uint32_t enabled);
uint32_t cupti_activity_get_debug_mode(void);

}  // extern "C"
