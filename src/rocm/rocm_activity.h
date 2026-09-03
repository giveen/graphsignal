#pragma once

#include <cstdint>

// Lifecycle is driven by rocprofiler-sdk via the exported rocprofiler_configure
// entry point (tool_init / tool_fini), NOT by explicit start/stop calls the way
// the CUPTI injection path works. These helpers only toggle/read the debug flag
// and are kept for symmetry with the CUPTI backend.
void rocm_activity_set_debug_mode(uint32_t debug_mode);
uint32_t rocm_activity_get_debug_mode();
