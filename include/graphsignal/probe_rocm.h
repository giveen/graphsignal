/*
 * Graphsignal probes — ROCm/HIP device-side instruments.
 *
 * Copyright 2026 Graphsignal, Inc.
 * Licensed under the Apache License, Version 2.0. See probe.h for the
 * license terms and the overall probe model, and probe_cuda.h for the usage
 * pattern (this header is the HIP analog).
 *
 * Compile with hipcc (-std=c++17). Note: graphsignal_gtimer() returns
 * wall_clock64() ticks, not nanoseconds — convert with the device's
 * wall clock frequency (hipDeviceAttributeWallClockRate, kHz) or record raw
 * ticks consistently and interpret downstream.
 */

#ifndef GRAPHSIGNAL_PROBE_ROCM_H
#define GRAPHSIGNAL_PROBE_ROCM_H

#include <graphsignal/probe.h>

#include <hip/hip_runtime.h>

/* Registers a histogram instrument stored in device memory on the current
   HIP device. Idempotent per (name, tags). Returns NULL on cap/allocation
   failure; passing NULL downstream is safe. */
static inline graphsignal_probe_entry* graphsignal_probe_register_rocm(
        const char* name, const char* const* tag_keys,
        const char* const* tag_vals, size_t ntags) {
    graphsignal_instrument_data* dev_data = NULL;
    if (hipMalloc((void**)&dev_data, sizeof(graphsignal_instrument_data)) != hipSuccess) {
        return NULL;
    }
    if (hipMemset(dev_data, 0, sizeof(graphsignal_instrument_data)) != hipSuccess) {
        hipFree(dev_data);
        return NULL;
    }
    uint64_t min_init = UINT64_MAX;
    if (hipMemcpy(&dev_data->min, &min_init, sizeof(min_init),
                  hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(dev_data);
        return NULL;
    }

    int device_id = -1;
    hipGetDevice(&device_id);

    graphsignal_probe_entry* e = graphsignal_probe_register_storage(
        name, GRAPHSIGNAL_HISTOGRAM, tag_keys, tag_vals, ntags,
        dev_data, device_id);
    if (!e || e->data != dev_data) {
        /* Registration failed or the instrument already existed. */
        hipFree(dev_data);
    }
    return e;
}

/* Device pointer to pass into kernels. NULL-safe. */
static inline graphsignal_instrument_data* graphsignal_probe_device_data_rocm(
        const graphsignal_probe_entry* e) {
    if (!e || e->storage != GRAPHSIGNAL_STORAGE_DEVICE) return NULL;
    return e->data;
}

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)

/* Constant-rate device wall clock, in ticks (see header comment). */
__device__ __forceinline__ unsigned long long graphsignal_gtimer() {
    return (unsigned long long)wall_clock64();
}

/* Records one histogram observation from device code. NULL-safe. */
__device__ __forceinline__ void graphsignal_device_record(
        graphsignal_instrument_data* d, unsigned long long value) {
    if (d == NULL) return;
    atomicAdd((unsigned long long*)&d->count, 1ull);
    atomicAdd((unsigned long long*)&d->sum, value);
    unsigned int e = 63u - (unsigned int)__clzll((long long)(value | 1ull));
    unsigned int idx;
    if (value < 4ull) {
        idx = (unsigned int)value;
    } else {
        idx = 4u * e + (unsigned int)((value >> (e - 2u)) & 3ull) - 4u;
        if (idx >= GRAPHSIGNAL_PROBE_HIST_BINS) idx = GRAPHSIGNAL_PROBE_HIST_BINS - 1u;
    }
    atomicAdd((unsigned long long*)&d->bins[idx], 1ull);
    atomicMin((unsigned long long*)&d->min, value);
    atomicMax((unsigned long long*)&d->max, value);
}

#define GRAPHSIGNAL_RECORD(data, value) graphsignal_device_record((data), (value))

#endif /* __HIPCC__ */

#endif /* GRAPHSIGNAL_PROBE_ROCM_H */
