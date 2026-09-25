"""Tails the ninfer-serve request log and turns it into signals.

`ninfer-serve` writes a JSONL request log: an append-only artifact, one JSON
object per line, flushed as requests complete. This recorder tails that file
and folds it into the metric, log and resource stores — no scraping, no extra
endpoint, nothing written into the workload's process.

The file is found in one of two ways, in this order:

1. `GRAPHSIGNAL_NINFER_JSONL` — an explicit path (absolute, or relative to the
   watcher's cwd). The launcher sets this for every `ninfer-serve` run.
2. `--request-log-jsonl` in the workload's argv, in both the space
   (`--request-log-jsonl /path`) and equal (`--request-log-jsonl=/path`) forms.
   The last occurrence wins, as it does in argparse.

The recorder is a strict no-op unless the target command's basename is
`ninfer-serve`: it is a root-pid-only recorder, so the JSONL a child process
happens to write is never read.

Envelope (schema 24)
--------------------

Every line is `event_base()` from `src/serve/request_log.cpp`:

    {"artifact_type": "ninfer_serve_request_log",
     "schema_version": 24,
     "event": "throughput",
     "timestamp_unix_ms": 1712345678901,
     "server_instance_id": "serve-1234-1712345678901234",
     …event payload…}

`timestamp_unix_ms` is epoch milliseconds; it is scaled to nanoseconds for
`measurement_ts`, which is what the signal stores take. There is no sequence
number: consumption is tracked by byte offset and file identity alone, and a
change of `server_instance_id` is the artifact's own signal that a new server
lifetime began (the log is opened in append mode, so one file legitimately
carries several).

Lines whose `artifact_type` is anything else are not ours and are skipped. A
`schema_version` below 24 predates the field layout this recorder reads and is
skipped. A *newer* version is tolerated: the events and fields read here are
stable, and anything the recorder does not know is ignored, so a future ninfer
that adds events or fields is read as far as it stays a superset of 24.

Events
------

* `server_start` — `server`, `artifact`, `engine`, `sampling_defaults`,
  `memory`, `environment`, `argv`. Configuration and startup accounting only.
* `request_start` — `request`, `preparation_seconds` (prompt preparation,
  media decode and media-cache accounting).
* `request_done` — `request`, `result`, `timings_seconds`, `engine_timing`,
  `speculative`, `materialization`. The request's measurements.
* `request_error` — `request`, `error.message`. Free text only.
* `request_rejected` — `phase`, `request`, `error` with the API error's
  `status`/`type`/`code`/`param`/`message`.
* `throughput` — `interval_seconds` and per-interval deltas: `tokens`,
  `throughput_tokens_per_second`, `scheduler`, `decode_batch`, `host_work`,
  `context_cache`.

Tailing
-------

The file is append-only, but not append-only forever: the server may truncate
it on restart, and something else may replace it. Each tick the recorder checks
the size and the inode; a file that shrank, or whose inode changed, is a new
file and is read from byte 0. `server_instance_id` is the second line of
defence — a change means a new server lifetime, which resets the recorder's
cumulative state (counters, histograms, published tags) so a restart is never
added on top of the run it replaced.

Only complete lines are consumed: the byte offset always stops at the last
newline, so a half-written record is retained and picked up whole on a later
tick, once the server has finished writing it.

Cardinality
-----------

Every tag this recorder emits comes from a closed set: a fixed list of
`server_start` values, the engine's own enumerations (`finish_reason`,
`prefix_reuse_path`, speculative `backend`, the context-cache selection
paths), and the request `protocol`. Where a value is not enumerated — an API
error `code`, a tool-call fallback reason — it is normalized and capped at
`MAX_TAG_VALUES` distinct values per metric, the overflow folding into
`other`. Nothing derived from an individual request — in particular no
`request_id` — is ever used as a tag; those go into log messages, which the
log store bounds by count, instead.
"""

import json
import logging
import math
import os
import re
import shlex
import time
from typing import Dict, List, Optional, Tuple

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.signals.logs import LogStore

logger = logging.getLogger('graphsignal')

# The command this recorder exists for, and the artifact it reads. Both must
# match src/serve/request_log.h.
COMMAND_BASENAME = 'ninfer-serve'
ARTIFACT_TYPE = 'ninfer_serve_request_log'
SCHEMA_VERSION = 24
JSONL_ENV_VAR = 'GRAPHSIGNAL_NINFER_JSONL'
REQUEST_LOG_FLAG = '--request-log-jsonl'

# Events read from the artifact. Anything else is ignored, which is what makes
# a newer schema version tolerable.
EVENT_SERVER_START = 'server_start'
EVENT_REQUEST_START = 'request_start'
EVENT_REQUEST_DONE = 'request_done'
EVENT_REQUEST_ERROR = 'request_error'
EVENT_REQUEST_REJECTED = 'request_rejected'
EVENT_THROUGHPUT = 'throughput'

# Bounded work per tick, so a burst (or a backlog after the watcher restarted)
# can never make a tick unbounded. Backlog is carried to the next tick through
# the offset rather than dropped.
MAX_READ_BYTES_PER_TICK = 4 * 1024 * 1024
MAX_LOG_ENTRIES_PER_TICK = 20

# A single record longer than this is not a record this recorder can use; the
# read stops there rather than growing without bound.
MAX_LINE_BYTES = 1024 * 1024

# Distinct tag values allowed per tagged metric. Values that are not part of
# the engine's own enumeration are normalized and capped here, which is what
# keeps a server that invents a new error code per request from growing the
# series count without bound.
MAX_TAG_VALUES = 32
OTHER_TAG_VALUE = 'other'

_NS_PER_MS = 1_000_000

# Anything at or below this is not worth a series: a zero duration is a phase
# the engine skipped, not a measurement of zero time.
_MIN_DURATION_SECONDS = 1e-9

# Latency buckets, in seconds. Two sets: request-level phases span hundreds of
# milliseconds to minutes, while queue wait and host/device exposure are
# sub-millisecond to seconds.
_REQUEST_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    120.0, 300.0, 600.0,
)
_ENGINE_LATENCY_BUCKETS = (
    0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)

# `host_work` groups whose seconds are recorded as cumulative work and as a
# per-second rate: the engine's own phase names, reused verbatim as tag values
# so a reader can line them up with the artifact.
HOST_WORK_ELAPSED = (
    'engine_boundary', 'program_submit', 'program_post', 'engine_commit_output',
    'engine_maintenance', 'total',
)
HOST_WORK_CLASSES = (
    'decode_host', 'decode_device_wait', 'prefill_host', 'prefill_device_wait',
    'control_host', 'control_device_wait',
)
HOST_WORK_DETAIL = ('admission_policy', 'context_progress', 'stats_publication')
# The scheduler's queue states, in the artifact's own names.
SCHEDULER_STATES = (
    'running', 'prefilling', 'decode_ready', 'waiting', 'materializing',
    'capture_pending', 'terminal_pending',
)
# Context-cache selection paths (`prefix_reuse_path_name`).
CACHE_SELECTION_PATHS = (
    'root', 'private_endpoint', 'private_turn_closure',
    'private_response_replay', 'private_long_anchor', 'shared_stable_prefix',
)
# Transfer directions, for state, main-KV and backend-KV movement.
TRANSFER_DIRECTIONS = ('d2h', 'h2d', 'd2d')

_CODE_SANITIZE_RE = re.compile(r'[^a-z0-9_.:-]+')
_EXECUTABLE_SUFFIXES = ('.exe', '.cmd', '.bat', '.com')


# ---------------------------------------------------------------------------
# Field maps
#
# Each entry is (metric name, dotted path into the record, tag key or None).
# The recorder walks these tables instead of hand-coding every field, so the
# mapping from artifact to signal is one readable list.
# ---------------------------------------------------------------------------

# `request_done.result` token accounting, all per-request quantities the
# recorder accumulates into cumulative totals.
REQUEST_DONE_COUNTERS: List[Tuple[str, str, Optional[str]]] = [
    ('ninfer_prompt_tokens_total', 'result.prompt_tokens', None),
    ('ninfer_completion_tokens_total', 'result.completion_tokens', None),
    ('ninfer_computed_prefill_tokens_total', 'result.computed_prefill_tokens', None),
    ('ninfer_prefix_cache_hit_tokens_total', 'result.prefix_cache_hit_tokens', None),
    ('ninfer_model_thinking_tokens_total', 'result.model_thinking_tokens', None),
    ('ninfer_thinking_control_tokens_total', 'result.thinking_control_tokens', None),
    ('ninfer_tool_calls_total', 'result.tool_call_count', None),
    ('ninfer_tool_call_parse_structured_total',
     'result.tool_call_parse.structured_call_count', None),
    ('ninfer_tool_call_parse_empty_arguments_omitted_total',
     'result.tool_call_parse.empty_arguments_omitted', None),
    ('ninfer_tool_call_parse_schema_mismatch_total',
     'result.tool_call_parse.schema_mismatch_arguments', None),
    ('ninfer_tool_call_parse_duplicate_parameters_repaired_total',
     'result.tool_call_parse.duplicate_parameters_repaired', None),
    # Enumerated by the engine; recorded as-is because the vocabulary is
    # closed (`finish_reason_name`, `protocol`).
    ('ninfer_requests_completed_by_finish_reason', 'result.finish_reason',
     'finish_reason'),
    ('ninfer_requests_completed_by_protocol', 'request.protocol', 'protocol'),
    ('ninfer_tool_call_parse_fallback_total',
     'result.tool_call_parse.fallback_reason', 'fallback_reason'),
    # Speculative decoding.
    ('ninfer_speculative_rounds_total', 'speculative.rounds', None),
    ('ninfer_speculative_drafted_tokens_total', 'speculative.drafted_tokens', None),
    ('ninfer_speculative_accepted_tokens_total', 'speculative.accepted_tokens', None),
    ('ninfer_speculative_fallback_steps_total', 'speculative.fallback_steps', None),
    # Engine scheduling units.
    ('ninfer_engine_prefill_units_total', 'engine_timing.units.prefill', None),
    ('ninfer_engine_control_units_total', 'engine_timing.units.control', None),
    # Materialization (prefix materialization planner), present only when the
    # feature runs.
    ('ninfer_materialization_search_renewals_total',
     'materialization.search_renewals', None),
    ('ninfer_materialization_incumbent_improvements_total',
     'materialization.incumbent_improvements', None),
    ('ninfer_materialization_search_work_total', 'materialization.search_work', None),
]

# Boolean diagnostics counted once each time they are reported true.
REQUEST_DONE_FLAGS: List[Tuple[str, str]] = [
    ('ninfer_requests_thinking_total', 'request.enable_thinking'),
    ('ninfer_requests_streamed_total', 'request.stream'),
    ('ninfer_thinking_control_applied_total', 'result.thinking_control_applied'),
    ('ninfer_tool_call_parse_marker_seen_total',
     'result.tool_call_parse.marker_seen'),
    ('ninfer_materialization_budget_exhausted_total',
     'materialization.budget_exhausted'),
    ('ninfer_materialization_maximal_fallback_total',
     'materialization.selected_maximal_fallback'),
]

REQUEST_DONE_GAUGES: List[Tuple[str, str, Optional[str]]] = [
    ('ninfer_speculative_accepted_per_position',
     'speculative.accepted_per_position', None),
    ('ninfer_materialization_planning_seconds',
     'materialization.planning_elapsed_ns', None),
    ('ninfer_materialization_search_seconds',
     'materialization.search_elapsed_ns', None),
    ('ninfer_materialization_selected_degradation_units',
     'materialization.selected_degradation_units', None),
    ('ninfer_materialization_targets_evaluated',
     'materialization.targets_evaluated', None),
]

# (metric name, path, tag key, bucket set key) — the two duration families.
REQUEST_DONE_HISTOGRAMS = [
    ('ninfer_request_duration_seconds', 'timings_seconds.total', 'request'),
    ('ninfer_request_prepare_seconds', 'timings_seconds.prepare', 'request'),
    ('ninfer_request_ttft_seconds', 'timings_seconds.ttft', 'request'),
    ('ninfer_request_vision_seconds', 'timings_seconds.vision', 'request'),
    ('ninfer_request_prefill_seconds', 'timings_seconds.prefill', 'request'),
    ('ninfer_request_decode_seconds', 'timings_seconds.decode', 'request'),
    ('ninfer_request_queue_wait_seconds', 'engine_timing.queue_wait_seconds',
     'engine'),
    ('ninfer_request_host_exposed_seconds',
     'engine_timing.host_exposed_seconds.total', 'engine'),
    ('ninfer_request_device_wait_seconds',
     'engine_timing.device_wait_exposed_seconds', 'engine'),
    ('ninfer_request_decode_host_seconds',
     'engine_timing.decode.host_exposed_seconds', 'engine'),
    ('ninfer_request_decode_device_wait_seconds',
     'engine_timing.decode.device_wait_exposed_seconds', 'engine'),
]

# `request_start.preparation_seconds`: prompt acquisition, tokenization, media
# preprocessing, and what the media cache did.
REQUEST_START_COUNTERS: List[Tuple[str, str, Optional[str]]] = [
    ('ninfer_requests_started_total', None, None),
    ('ninfer_media_items_total', 'preparation_seconds.media_items', None),
    ('ninfer_media_bytes_total', 'preparation_seconds.media_bytes', None),
    ('ninfer_raw_patches_total', 'preparation_seconds.raw_patches', None),
    ('ninfer_vision_tokens_total', 'preparation_seconds.vision_tokens', None),
    ('ninfer_patch_bytes_total', 'preparation_seconds.patch_bytes', None),
    ('ninfer_media_preprocess_built_patch_bytes_total',
     'preparation_seconds.built_patch_bytes', None),
    ('ninfer_media_preprocess_reused_patch_bytes_total',
     'preparation_seconds.reused_patch_bytes', None),
    ('ninfer_media_cache_hits_total', 'preparation_seconds.cache_hits', None),
    ('ninfer_media_cache_misses_total', 'preparation_seconds.cache_misses', None),
    ('ninfer_media_singleflight_waits_total',
     'preparation_seconds.singleflight_waits', None),
]

REQUEST_START_HISTOGRAMS = [
    ('ninfer_request_preparation_seconds', 'preparation_seconds.total', 'request'),
    ('ninfer_request_preparation_acquisition_seconds',
     'preparation_seconds.acquisition', 'request'),
    ('ninfer_request_preparation_tokenize_seconds',
     'preparation_seconds.tokenize', 'request'),
    ('ninfer_request_preparation_media_preprocess_seconds',
     'preparation_seconds.media_preprocess', 'request'),
]

# `throughput` carries per-interval deltas (the artifact already differences
# its runtime counters), so these are accumulated into cumulative totals.
# Token deltas are kept separate from the per-request token totals on
# purpose: the two describe the same work, and summing both into one series
# would double count it.
THROUGHPUT_COUNTERS: List[Tuple[str, str, Optional[str]]] = [
    ('ninfer_throughput_prefill_tokens_total', 'tokens.computed_prefill', None),
    ('ninfer_throughput_decode_tokens_total', 'tokens.committed_decode', None),
    ('ninfer_decode_rounds_total', 'decode_batch.rounds', None),
    ('ninfer_decode_row_rounds_total', 'decode_batch.row_rounds', None),
    ('ninfer_host_work_prefill_units_total', 'host_work.units.prefill', None),
    ('ninfer_host_work_control_units_total', 'host_work.units.control', None),
    ('ninfer_context_cache_captures_completed_total',
     'context_cache.captures.completed', None),
    ('ninfer_context_cache_captures_aborted_total',
     'context_cache.captures.aborted', None),
    ('ninfer_context_cache_captures_skipped_total',
     'context_cache.captures.skipped', None),
    ('ninfer_context_cache_salvaged_total', 'context_cache.salvage.published', None),
    ('ninfer_context_cache_reused_prompt_tokens_total',
     'context_cache.selections.reused_prompt_tokens', None),
    ('ninfer_context_cache_state_moves_total', 'context_cache.state_operations.moves',
     None),
    ('ninfer_context_cache_state_forks_total', 'context_cache.state_operations.forks',
     None),
    ('ninfer_context_cache_state_restores_total',
     'context_cache.state_operations.restores', None),
    ('ninfer_context_cache_spill_pages_total', 'context_cache.pressure.spill_pages',
     None),
    ('ninfer_context_cache_checkpoints_dropped_total',
     'context_cache.pressure.checkpoints_dropped', None),
    ('ninfer_context_cache_pressure_searches_total',
     'context_cache.pressure.searches', None),
    ('ninfer_context_cache_pressure_budget_exhaustions_total',
     'context_cache.pressure.search_budget_exhaustions', None),
    ('ninfer_context_cache_maximal_fallback_selections_total',
     'context_cache.pressure.maximal_fallback_selections', None),
    ('ninfer_context_cache_historical_fork_hits_total',
     'context_cache.pressure.historical_fork_hits', None),
    ('ninfer_context_cache_transfer_seconds_total',
     'context_cache.actual_transfer_seconds', None),
]

# Tagged throughput counters: (metric name, dotted path, tag key, tag value).
# The tag value is named explicitly rather than inferred from the path, because
# it is not always the last segment (`...transfers.h2d.count` is tagged
# `direction=h2d`). Every value is the artifact's own name.
THROUGHPUT_TAGGED_COUNTERS: List[Tuple[str, str, str, str]] = (
    [('ninfer_context_cache_selections_total',
      f'context_cache.selections.{path}', 'path', path)
     for path in CACHE_SELECTION_PATHS]
    + [('ninfer_context_cache_state_transfers_total',
        f'context_cache.state_transfers.{direction}.count', 'direction', direction)
       for direction in TRANSFER_DIRECTIONS]
    + [('ninfer_context_cache_state_transfer_bytes_total',
        f'context_cache.state_transfers.{direction}.bytes', 'direction', direction)
       for direction in TRANSFER_DIRECTIONS]
    + [('ninfer_context_cache_main_kv_pages_total',
        f'context_cache.main_kv_transfers.{direction}.pages', 'direction', direction)
       for direction in TRANSFER_DIRECTIONS]
    + [('ninfer_context_cache_main_kv_transfer_bytes_total',
        f'context_cache.main_kv_transfers.{direction}.bytes', 'direction', direction)
       for direction in TRANSFER_DIRECTIONS]
    + [('ninfer_context_cache_backend_kv_pages_total',
        f'context_cache.backend_kv_transfers.{direction}.pages', 'direction', direction)
       for direction in TRANSFER_DIRECTIONS]
    + [('ninfer_context_cache_owners_evicted_total',
        f'context_cache.pressure.{owner}_owners_evicted', 'owner', owner)
       for owner in ('private', 'shared')]
    + [('ninfer_context_cache_owners_degraded_total',
        f'context_cache.pressure.{owner}_owners_degraded', 'owner', owner)
       for owner in ('private', 'shared')]
    + [('ninfer_host_work_invocations_total',
        f'host_work.detail_invocations.{name}', 'detail', name)
       for name in HOST_WORK_DETAIL]
)

# Cumulative host work. One metric, tagged by the artifact's phase name, so the
# phases stay comparable with each other and with the artifact.
HOST_WORK_SECONDS_COUNTER: List[Tuple[str, str]] = (
    [(f'host_work.elapsed_seconds.{name}', name) for name in HOST_WORK_ELAPSED]
    + [('host_work.device_wait_seconds', 'device_wait')]
    + [(f'host_work.work_class_seconds.{name}', name) for name in HOST_WORK_CLASSES]
    + [(f'host_work.detail_subset_seconds.{name}', name) for name in HOST_WORK_DETAIL]
)

# Instantaneous state. Each throughput line overwrites the previous one, so a
# stale queue depth is never reported as current.
THROUGHPUT_GAUGES: List[Tuple[str, str, Optional[str]]] = (
    [(f'ninfer_requests_{state}', f'scheduler.{state}', None)
     for state in SCHEDULER_STATES]
    + [
        ('ninfer_prefill_tokens_per_second', 'throughput_tokens_per_second.prefill',
         None),
        ('ninfer_decode_tokens_per_second', 'throughput_tokens_per_second.decode',
         None),
        ('ninfer_decode_batch_average_size', 'decode_batch.average_size', None),
        ('ninfer_decode_host_microseconds_per_round',
         'host_work.decode_host_microseconds_per_round', None),
        ('ninfer_decode_host_microseconds_per_row_round',
         'host_work.decode_host_microseconds_per_row_round', None),
        ('ninfer_decode_device_wait_microseconds_per_round',
         'host_work.decode_device_wait_microseconds_per_round', None),
        ('ninfer_context_cache_frontier_tokens',
         'context_cache.last_selection.frontier_tokens', None),
    ]
    + [(f'ninfer_context_cache_occupancy_{field}',
        f'context_cache.occupancy.{field}', None)
       for field in ('device_state_slots', 'host_state_slots', 'device_main_kv_pages',
                     'device_main_kv_lease_pages', 'device_backend_kv_pages',
                     'device_backend_kv_lease_pages', 'host_kv_bytes',
                     'shared_active_references')]
)

# Tagged throughput gauges: (metric name, dotted path, tag key, tag value).
THROUGHPUT_TAGGED_GAUGES: List[Tuple[str, str, str, str]] = [
    ('ninfer_host_work_microseconds_per_invocation',
     f'host_work.detail_microseconds_per_invocation.{name}', 'detail', name)
    for name in HOST_WORK_DETAIL
]

# `server_start` values promoted to watch-wide tags: what is being served and
# how the engine is configured. One value each per server lifetime, all from
# fixed fields, so the tag count is bounded.
SERVER_START_TAGS = (
    ('model', ('server', 'public_model_id')),
    ('host', ('server', 'host')),
    ('port', ('server', 'port')),
    ('architecture', ('artifact', 'architecture')),
    ('kv_cache', ('engine', 'kv_cache')),
    ('speculative_backend', ('engine', 'speculative_backend')),
    ('ngram_residency', ('engine', 'ngram_residency')),
    ('kv_capacity_mode', ('engine', 'kv_capacity_mode')),
    ('gpu_name', ('environment', 'gpu_name')),
    # Two paths, joined into one value: compute capability as `12.1`.
    ('compute_capability', ('environment', 'compute_capability_major'),
     ('environment', 'compute_capability_minor')),
)

# `server_start` values recorded as resource attributes: the startup picture,
# which is large and run-scoped and has no business being a tag on every signal.
SERVER_START_ATTRIBUTES = (
    ('model', ('server', 'public_model_id')),
    ('host', ('server', 'host')),
    ('port', ('server', 'port')),
    ('artifact_path', ('artifact', 'path')),
    ('artifact_name', ('artifact', 'name')),
    ('artifact_size_bytes', ('artifact', 'size_bytes')),
    ('artifact_architecture', ('artifact', 'architecture')),
    ('artifact_formats', ('artifact', 'formats')),
    ('artifact_prefill_signature', ('artifact', 'prefill_signature')),
    ('artifact_bytes_read', ('artifact', 'bytes_read')),
    ('artifact_host_to_device_bytes', ('artifact', 'host_to_device_bytes')),
    ('device', ('engine', 'device')),
    ('max_context', ('engine', 'max_context')),
    ('max_concurrency', ('engine', 'max_concurrency')),
    ('max_pending_requests', ('engine', 'max_pending_requests')),
    ('pending_timeout_ms', ('engine', 'pending_timeout_ms')),
    ('prefill_chunk', ('engine', 'prefill_chunk')),
    ('kv_capacity', ('engine', 'kv_capacity')),
    ('kv_cache', ('engine', 'kv_cache')),
    ('cuda_graph', ('engine', 'cuda_graph')),
    ('vision', ('engine', 'vision')),
    ('prefix_reuse', ('engine', 'prefix_reuse')),
    ('speculative_backend', ('engine', 'speculative_backend')),
    ('speculative_draft_window', ('engine', 'speculative_draft_window')),
    ('proposal_head', ('engine', 'proposal_head')),
    ('context_cache_enabled', ('engine', 'context_cache', 'enabled')),
    ('context_cache_device_state_slots',
     ('engine', 'context_cache', 'device_state_slots')),
    ('context_cache_host_state_slots',
     ('engine', 'context_cache', 'host_state_slots')),
    ('context_cache_host_kv_capacity_bytes',
     ('engine', 'context_cache', 'host_kv_capacity_bytes')),
    ('weights_capacity_bytes', ('memory', 'weights', 'capacity_bytes')),
    ('weights_used_bytes', ('memory', 'weights', 'used_bytes')),
    ('weights_peak_used_bytes', ('memory', 'weights', 'peak_used_bytes')),
    ('sequence_capacity_bytes', ('memory', 'sequence', 'capacity_bytes')),
    ('sequence_used_bytes', ('memory', 'sequence', 'used_bytes')),
    ('workspace_capacity_bytes', ('memory', 'workspace', 'capacity_bytes')),
    ('workspace_used_bytes', ('memory', 'workspace', 'used_bytes')),
    ('kv_payload_bytes', ('memory', 'kv_payload_bytes')),
    ('kv_capacity_headroom_bytes', ('memory', 'kv_capacity_headroom_bytes')),
    ('planned_slack_bytes', ('memory', 'planned_slack_bytes')),
    ('available_after_startup_bytes', ('memory', 'available_after_startup_bytes')),
    ('host_kv_capacity_bytes', ('memory', 'host_kv_capacity_bytes')),
    ('host_state_capacity_slots', ('memory', 'host_state_capacity_slots')),
    ('gpu_name', ('environment', 'gpu_name')),
    ('gpu_uuid', ('environment', 'gpu_uuid')),
    ('total_device_memory_bytes', ('environment', 'total_device_memory_bytes')),
    ('cuda_compile_version', ('environment', 'cuda_compile_version')),
    ('cuda_runtime_version', ('environment', 'cuda_runtime_version')),
    ('cuda_driver_version', ('environment', 'cuda_driver_version')),
    ('load_seconds', ('artifact', 'load_seconds')),
)

# Startup memory as gauges: static for the server's lifetime, and the numbers a
# reader wants next to the throughput gauges.
SERVER_START_GAUGES: List[Tuple[str, Tuple[str, ...]]] = [
    ('ninfer_model_load_seconds', ('artifact', 'load_seconds')),
    ('ninfer_model_upload_seconds', ('artifact', 'upload_seconds')),
    ('ninfer_memory_weights_used_bytes', ('memory', 'weights', 'used_bytes')),
    ('ninfer_memory_weights_capacity_bytes', ('memory', 'weights', 'capacity_bytes')),
    ('ninfer_memory_sequence_used_bytes', ('memory', 'sequence', 'used_bytes')),
    ('ninfer_memory_sequence_capacity_bytes',
     ('memory', 'sequence', 'capacity_bytes')),
    ('ninfer_memory_workspace_used_bytes', ('memory', 'workspace', 'used_bytes')),
    ('ninfer_memory_workspace_capacity_bytes',
     ('memory', 'workspace', 'capacity_bytes')),
    ('ninfer_memory_kv_payload_bytes', ('memory', 'kv_payload_bytes')),
    ('ninfer_memory_kv_capacity_bytes', ('engine', 'kv_capacity')),
    ('ninfer_memory_kv_capacity_headroom_bytes',
     ('memory', 'kv_capacity_headroom_bytes')),
    ('ninfer_memory_host_kv_capacity_bytes', ('memory', 'host_kv_capacity_bytes')),
    ('ninfer_memory_available_after_startup_bytes',
     ('memory', 'available_after_startup_bytes')),
    ('ninfer_memory_host_state_capacity_slots',
     ('memory', 'host_state_capacity_slots')),
]


# ---------------------------------------------------------------------------
# Argument and value helpers
# ---------------------------------------------------------------------------

def command_basename(args) -> Optional[str]:
    """Basename of the launched command, without a platform executable
    suffix. Accepts argv lists and the space-joined command string supplied by
    the process monitor; None when there is no command."""
    if not args:
        return None
    if isinstance(args, str):
        try:
            argv = shlex.split(args)
        except ValueError:
            return None
        args = argv
    try:
        first = args[0]
    except (TypeError, IndexError, KeyError):
        return None
    if not isinstance(first, str) or not first:
        return None
    base = os.path.basename(first)
    lowered = base.lower()
    for suffix in _EXECUTABLE_SUFFIXES:
        if lowered.endswith(suffix):
            return base[:-len(suffix)]
    return base


def is_ninfer_serve(args) -> bool:
    """True iff the launched command is `ninfer-serve`. Compared on the exact
    basename: this recorder reads one specific engine's artifact and must not
    guess that a similarly named command writes the same schema."""
    return command_basename(args) == COMMAND_BASENAME


def parse_request_log_flag(args) -> Optional[str]:
    """The `--request-log-jsonl` path in argv, or None. Both the space and the
    equal form are accepted, and the last occurrence wins (as in argparse)."""
    if not args:
        return None
    if isinstance(args, str):
        try:
            args = shlex.split(args)
        except ValueError:
            return None
    path = None
    index = 0
    try:
        length = len(args)
    except TypeError:
        return None
    while index < length:
        arg = args[index]
        if not isinstance(arg, str):
            index += 1
            continue
        if arg == REQUEST_LOG_FLAG:
            # A trailing flag with no value is malformed; keep looking rather
            # than consuming the next flag as a path.
            if index + 1 < length:
                value = args[index + 1]
                if isinstance(value, str) and not value.startswith('-'):
                    path = value
                    index += 2
                    continue
            index += 1
            continue
        if arg.startswith(REQUEST_LOG_FLAG + '='):
            value = arg[len(REQUEST_LOG_FLAG) + 1:]
            if value:
                path = value
            index += 1
            continue
        index += 1
    return path


def resolve_jsonl_path(args, environ=None) -> Optional[str]:
    """The request-log path this run should tail: the env var wins, then the
    command line. Returns None when neither names a file."""
    environ = os.environ if environ is None else environ
    candidates = []
    env_value = environ.get(JSONL_ENV_VAR) if hasattr(environ, 'get') else None
    if env_value:
        candidates.append(env_value)
    flag_value = parse_request_log_flag(args)
    if flag_value:
        candidates.append(flag_value)
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        return os.path.abspath(os.path.expanduser(candidate))
    return None


def normalize_timestamp_ns(value, default: Optional[int] = None) -> Optional[int]:
    """`timestamp_unix_ms` in epoch nanoseconds, which is what the signal
    stores take as a measurement time. A value that is not a number falls back
    to `default`."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        milliseconds = value
    else:
        number = _as_number(value)
        if number is None:
            return default
        milliseconds = number
    if milliseconds <= 0:
        return default
    return int(milliseconds) * _NS_PER_MS


def _dig(record, path: Optional[str]):
    """Follow a dotted path through nested objects. Returns None when any
    step is missing or is not an object, so a field the artifact drops simply
    does not produce a metric rather than an error."""
    if path is None:
        return None
    current = record
    for key in path.split('.'):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _as_int(value) -> Optional[int]:
    number = _as_number(value)
    if number is None:
        return None
    return int(number)


def _as_flag(value) -> bool:
    return value is True


def _tags(tag_key: Optional[str], tag_value: Optional[str]) -> Optional[dict]:
    """The tag dict for one series, or None when the series is untagged."""
    if tag_key is None or tag_value is None:
        return None
    return {tag_key: tag_value}


def _text(value) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    text = str(value).strip()
    return text or None


def _attribute_text(value) -> Optional[str]:
    """A `server_start` value as a resource attribute. Unlike a tag value this
    keeps booleans (`cuda_graph`, `vision`, `prefix_reuse`), which the
    artifact writes as JSON booleans."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return _text(value)


def sanitize_code(value) -> Optional[str]:
    """Normalize a free-form tag value (an API error code, a fallback reason)
    into a short, tag-safe token. Returns None when the value carries nothing
    usable."""
    if value is None or isinstance(value, bool):
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip().lower()
    if not text:
        return None
    text = _CODE_SANITIZE_RE.sub('_', text).strip('_')[:40]
    return text or None


class _CodeSet:
    """A bounded set of tag values for one metric: the first `MAX_TAG_VALUES`
    distinct values are kept, everything after folds into `other`, so a value
    the engine does not enumerate cannot grow the series count without bound."""

    def __init__(self, limit: int = MAX_TAG_VALUES):
        self._limit = limit
        self._values = set()
        self._overflowed = False

    def normalize(self, value) -> str:
        code = sanitize_code(value) or OTHER_TAG_VALUE
        if code in self._values:
            return code
        if len(self._values) < self._limit:
            self._values.add(code)
            return code
        self._overflowed = True
        return OTHER_TAG_VALUE

    def reset(self):
        self._values.clear()
        self._overflowed = False


class _LatencyHistogram:
    """Cumulative duration distribution: the exact count/sum plus lifetime
    min/max, and per-bucket counts the recorder keeps so the emitted histogram
    is cumulative like every other histogram in the store."""

    def __init__(self, bounds):
        self._bounds = tuple(bounds)
        self._counts = [0] * len(self._bounds)
        self.count = 0
        self.sum = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        for index, bound in enumerate(self._bounds):
            if value <= bound:
                self._counts[index] += 1
                break
        # A value past the last bound lands in no bin: it still counts in the
        # exact total, and the widest edge is the one a reader can interpolate
        # from.

    def reset(self) -> None:
        self._counts = [0] * len(self._bounds)
        self.count = 0
        self.sum = 0.0
        self.min = None
        self.max = None

    def bins_and_counts(self):
        """Non-cumulative (bins, counts) with empty buckets dropped."""
        bins = []
        counts = []
        for bound, count in zip(self._bounds, self._counts):
            if count > 0:
                bins.append(bound)
                counts.append(count)
        return bins, counts


class NInferRecorder(BaseRecorder):
    """Reads the ninfer-serve request log and records what it says."""

    def __init__(self, root_pid=None, pid=None, args=None, jsonl_path=None,
                 environ=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._environ = os.environ if environ is None else environ
        self._jsonl_path = jsonl_path or resolve_jsonl_path(args, self._environ)
        self._disabled = True
        # Tail state. `_offset` is always the end of the last complete line
        # consumed, so a partial trailing line is retained, not lost.
        self._offset = 0
        self._inode: Optional[Tuple[int, int]] = None
        self._server_instance_id: Optional[str] = None
        # Cumulative state, reset with the server lifetime. Counters carry
        # totals, so the recorder keeps the running totals itself.
        self._counters: Dict[Tuple[str, Optional[str]], dict] = {}
        self._gauges: Dict[Tuple[str, Optional[str]], dict] = {}
        self._histograms: Dict[str, _LatencyHistogram] = {}
        self._histogram_ts: Dict[str, int] = {}
        self._code_sets: Dict[str, _CodeSet] = {}
        # Watch-wide tags this recorder published, so a new server lifetime can
        # retract the previous one's instead of leaving them lying around.
        self._published_tags: List[str] = []
        self._log_entries_this_tick = 0
        self._dropped_log_entries = 0
        self._missing_file_logged = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def jsonl_path(self) -> Optional[str]:
        return self._jsonl_path

    def setup(self):
        # One request log belongs to the server process. A child that happens to
        # be ninfer-serve-shaped, or that inherits the flag, is not the server.
        if self.pid is None or self.root_pid is None or self.pid != self.root_pid:
            logger.debug('NInferRecorder is root-pid only; skipping setup')
            return
        if not is_ninfer_serve(self.args):
            logger.debug('NInferRecorder: command is not %s; skipping setup',
                         COMMAND_BASENAME)
            return
        if not self._jsonl_path:
            logger.debug('NInferRecorder: no %s path; skipping setup',
                         JSONL_ENV_VAR)
            return
        self._disabled = False
        logger.debug('NInferRecorder tailing %s', self._jsonl_path)

    def on_tick(self):
        if self._disabled:
            return
        if not graphsignal.watcher.is_configured():
            return
        self._log_entries_this_tick = 0
        self._dropped_log_entries = 0
        try:
            self._drain()
        except OSError as exc:
            logger.debug('NInferRecorder: error reading %s: %s',
                         self._jsonl_path, exc)

    def finalize(self):
        if self._disabled:
            return
        # The server's last records are written as it exits; read them before
        # the watcher's final tick.
        try:
            self._drain()
        except Exception:
            logger.debug('NInferRecorder: error during finalize', exc_info=True)

    def shutdown(self):
        self._disabled = True

    # -- tailing -----------------------------------------------------------

    def _stat(self):
        try:
            stat = os.stat(self._jsonl_path)
        except FileNotFoundError:
            if not self._missing_file_logged:
                self._missing_file_logged = True
                logger.debug('NInferRecorder: %s does not exist yet',
                             self._jsonl_path)
            return None
        except OSError as exc:
            logger.debug('NInferRecorder: cannot stat %s: %s',
                         self._jsonl_path, exc)
            return None
        if not os.path.isfile(self._jsonl_path):
            return None
        self._missing_file_logged = False
        return stat

    def _drain(self):
        stat = self._stat()
        if stat is None:
            return

        identity = (stat.st_dev, stat.st_ino)
        replaced = self._inode is not None and identity != self._inode
        truncated = stat.st_size < self._offset
        if replaced or truncated:
            # A new file, or the same one cut short: read it from the start.
            # Whatever the previous lifetime counted belongs to a server that
            # is gone, so the cumulative state goes with it.
            logger.debug('NInferRecorder: %s was %s; restarting from offset 0',
                         self._jsonl_path,
                         'replaced' if replaced else 'truncated')
            self._reset_state()

        self._inode = identity
        if stat.st_size <= self._offset:
            return

        data, end = self._read_new_data()
        if end < 0:
            return
        self._offset += end + 1

        watcher = graphsignal.watcher.watcher()
        for raw in data[:end].split(b'\n'):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                logger.debug('NInferRecorder: skipping unparsable line')
                continue
            try:
                self._handle_record(watcher, record)
            except Exception:
                logger.debug('NInferRecorder: error handling record',
                             exc_info=True)

        if self._dropped_log_entries:
            logger.debug('NInferRecorder: rate cap dropped %s log entries this tick',
                         self._dropped_log_entries)

        self._flush(watcher)

    def _read_new_data(self):
        """Read from the offset, bounded per tick, and return the bytes plus
        the index of the last newline in them.

        Consuming complete lines only is what retains a half-written trailing
        record: the returned prefix stops at the last newline, and the offset
        moves exactly that far. A single read is not enough to guarantee a
        newline is in hand — a record longer than the per-tick budget would
        otherwise stall the tail forever — so the read is repeated until a
        newline appears, the file ends, or a single line exceeds
        `MAX_LINE_BYTES`, which is then dropped and the tail resynchronizes on
        the next newline.
        """
        data = b''
        with open(self._jsonl_path, 'rb') as f:
            f.seek(self._offset)
            while True:
                chunk = f.read(MAX_READ_BYTES_PER_TICK)
                if not chunk:
                    break
                data += chunk
                if b'\n' in data or len(data) >= MAX_LINE_BYTES:
                    break
            if len(data) >= MAX_LINE_BYTES and b'\n' not in data:
                # Skip the whole line: the offset moves past its newline, so
                # the tail does not stall on it and the next record is read
                # normally.
                logger.debug('NInferRecorder: dropping an oversized line in %s',
                             self._jsonl_path)
                self._offset = self._skip_to_next_line(f)
                return b'', -1
        end = data.rfind(b'\n')
        if end < 0:
            return b'', -1
        return data, end

    def _skip_to_next_line(self, f) -> int:
        """Offset just past the next newline at or after the current one."""
        position = self._offset + MAX_LINE_BYTES
        while True:
            f.seek(position)
            chunk = f.read(MAX_READ_BYTES_PER_TICK)
            if not chunk:
                return position
            index = chunk.find(b'\n')
            if index >= 0:
                return position + index + 1
            position += len(chunk)

    # -- record handling ---------------------------------------------------

    def _reset_state(self) -> None:
        self._offset = 0
        self._server_instance_id = None
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._histogram_ts.clear()
        for code_set in self._code_sets.values():
            code_set.reset()
        self._retract_published_tags()

    def _retract_published_tags(self) -> None:
        if not graphsignal.watcher.is_configured():
            self._published_tags = []
            return
        watcher = graphsignal.watcher.watcher()
        for key in self._published_tags:
            try:
                watcher.remove_tag(key)
            except Exception:
                logger.debug('NInferRecorder: could not retract tag %s', key)
        self._published_tags = []

    def _handle_record(self, watcher, record) -> None:
        if not isinstance(record, dict):
            return
        if record.get('artifact_type') != ARTIFACT_TYPE:
            return

        version = _as_int(_dig(record, 'schema_version'))
        if version is None:
            return
        if version < SCHEMA_VERSION:
            # Older layout: the fields this recorder reads are not there yet,
            # and reading them as if they were would invent numbers.
            logger.debug('NInferRecorder: skipping schema_version %s (< %s)',
                         version, SCHEMA_VERSION)
            return
        # A newer version is read, but only through the events and fields this
        # recorder knows; anything else falls through the dispatch below.

        event = record.get('event')
        if not isinstance(event, str) or not event:
            return

        instance_id = _text(record.get('server_instance_id'))
        if instance_id and instance_id != self._server_instance_id:
            # The artifact's own marker for a new server lifetime. The log is
            # opened in append mode, so this is expected mid-file, not an error.
            logger.debug('NInferRecorder: new server instance %s', instance_id)
            self._server_instance_id = instance_id
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._histogram_ts.clear()
            for code_set in self._code_sets.values():
                code_set.reset()
            self._retract_published_tags()

        ts_ns = normalize_timestamp_ns(record.get('timestamp_unix_ms'))
        if ts_ns is None:
            ts_ns = time.time_ns()

        if event == EVENT_SERVER_START:
            self._handle_server_start(watcher, record, ts_ns)
        elif event == EVENT_REQUEST_START:
            self._handle_request_start(watcher, record, ts_ns)
        elif event == EVENT_REQUEST_DONE:
            self._handle_request_done(watcher, record, ts_ns)
        elif event == EVENT_REQUEST_ERROR:
            self._handle_request_error(watcher, record, ts_ns)
        elif event == EVENT_REQUEST_REJECTED:
            self._handle_request_rejected(watcher, record, ts_ns)
        elif event == EVENT_THROUGHPUT:
            self._handle_throughput(watcher, record, ts_ns)
        # Unknown event names are ignored: a newer schema may add them, and
        # guessing at their meaning is worse than not recording them.

    # -- events ------------------------------------------------------------

    def _handle_server_start(self, watcher, record, ts_ns) -> None:
        self._publish_server_tags(watcher, record)

        attributes = {}
        for name, path in SERVER_START_ATTRIBUTES:
            value = _attribute_text(_dig(record, '.'.join(path)))
            if value is not None:
                attributes[name] = value
        if self._jsonl_path:
            attributes['request_log_path'] = self._jsonl_path
        if self._server_instance_id:
            attributes['server_instance_id'] = self._server_instance_id

        resource_tags = {'process.pid': str(self.pid)}
        if self.root_pid is not None:
            resource_tags['process.root_pid'] = str(self.root_pid)
        watcher.update_resource(
            'ninfer_server',
            tags=resource_tags,
            attributes=attributes,
            first_seen_ts=ts_ns,
            last_seen_ts=ts_ns)

        for name, path in SERVER_START_GAUGES:
            value = _as_number(_dig(record, '.'.join(path)))
            if value is not None:
                self._set_gauge(name, None, None, value, ts_ns)

        self._bump('ninfer_server_starts', None, None, 1, ts_ns)

    def _publish_server_tags(self, watcher, record) -> None:
        values = {}
        for name, *paths in SERVER_START_TAGS:
            if len(paths) == 1:
                value = _text(_dig(record, '.'.join(paths[0])))
            else:
                parts = [_as_number(_dig(record, '.'.join(path))) for path in paths]
                if any(part is None for part in parts):
                    value = None
                else:
                    value = '.'.join(
                        str(int(part)) if float(part).is_integer() else str(part)
                        for part in parts)
            if value:
                values[name] = value
        if not values:
            return
        # Retract first: a second server_start for a new instance must not
        # leave a stale field behind.
        self._retract_published_tags()
        published = []
        for name, value in values.items():
            key = f'ninfer.{name}'
            try:
                watcher.set_tag(key, value)
            except Exception:
                logger.debug('NInferRecorder: could not set tag %s', key)
                continue
            published.append(key)
        self._published_tags = published

    def _handle_request_start(self, watcher, record, ts_ns) -> None:
        # The first table row counts the record itself.
        self._bump_fields(record, REQUEST_START_COUNTERS, ts_ns)
        self._record_histograms(record, REQUEST_START_HISTOGRAMS, ts_ns)

    def _handle_request_done(self, watcher, record, ts_ns) -> None:
        self._bump('ninfer_requests_completed', None, None, 1, ts_ns)
        self._bump_fields(record, REQUEST_DONE_COUNTERS, ts_ns)
        for name, path in REQUEST_DONE_FLAGS:
            if _as_flag(_dig(record, path)):
                self._bump(name, None, None, 1, ts_ns)
        for name, path, tag_key in REQUEST_DONE_GAUGES:
            value = _as_number(_dig(record, path))
            if value is None:
                continue
            if path.endswith('_ns'):
                # Nanoseconds in the artifact, seconds in the signal store.
                value = value * 1e-9
            self._set_gauge(name, None, None, value, ts_ns)
        self._record_histograms(record, REQUEST_DONE_HISTOGRAMS, ts_ns)

    def _handle_request_error(self, watcher, record, ts_ns) -> None:
        # `request_error` carries a free-text message and nothing structured, so
        # the counter is untagged and the detail goes to the log.
        self._bump('ninfer_request_errors', None, None, 1, ts_ns)
        message = _text(_dig(record, 'error.message'))
        self._log(
            watcher,
            level='error',
            message=f'ninfer-serve request error: {message or "no detail"}',
            event=EVENT_REQUEST_ERROR,
            ts_ns=ts_ns)

    def _handle_request_rejected(self, watcher, record, ts_ns) -> None:
        code = self._code_for('ninfer_requests_rejected').normalize(
            _dig(record, 'error.code') or _dig(record, 'error.type'))
        self._bump('ninfer_requests_rejected', 'reason', code, 1, ts_ns)
        status = _text(_dig(record, 'error.status'))
        if status:
            status_code = self._code_for(
                'ninfer_requests_rejected_by_status').normalize(status)
            self._bump('ninfer_requests_rejected_by_status', 'status',
                       status_code, 1, ts_ns)
        message = _text(_dig(record, 'error.message'))
        phase = _text(record.get('phase')) or 'unknown'
        # A rejection is the server declining a request, not a fault in the
        # server: it is recorded as a warning so it does not read as a crash.
        self._log(
            watcher,
            level='warning',
            message=(f'ninfer-serve request rejected ({code}, phase {phase}): '
                     f'{message or "no detail"}'),
            event=EVENT_REQUEST_REJECTED,
            ts_ns=ts_ns)

    def _handle_throughput(self, watcher, record, ts_ns) -> None:
        interval = _as_number(_dig(record, 'interval_seconds'))

        self._bump_fields(record, THROUGHPUT_COUNTERS, ts_ns)
        self._bump_tagged_fields(record, THROUGHPUT_TAGGED_COUNTERS, ts_ns)
        self._bump_host_work_seconds(record, ts_ns)

        for name, path, _tag_key in THROUGHPUT_GAUGES:
            value = _as_number(_dig(record, path))
            if value is None:
                continue
            self._set_gauge(name, None, None, value, ts_ns)

        for name, path, tag_key, tag_value in THROUGHPUT_TAGGED_GAUGES:
            value = _as_number(_dig(record, path))
            if value is None:
                continue
            self._set_gauge(name, tag_key, self._code_for(name).normalize(tag_value),
                            value, ts_ns)

        # Host work arrives as seconds over the interval; a rate is only
        # meaningful with the interval, and without it the cumulative seconds
        # above stand on their own.
        if interval and interval > 0:
            for path, work in HOST_WORK_SECONDS_COUNTER:
                value = _as_number(_dig(record, path))
                if value is None:
                    continue
                self._set_gauge(
                    'ninfer_host_work_seconds_per_second', 'work',
                    self._code_for('ninfer_host_work_seconds_per_second')
                    .normalize(work),
                    value / interval, ts_ns)

    # -- state and emission ------------------------------------------------

    def _code_for(self, name: str) -> _CodeSet:
        code_set = self._code_sets.get(name)
        if code_set is None:
            code_set = _CodeSet()
            self._code_sets[name] = code_set
        return code_set

    def _tag_value(self, name: str, tag_key: str, value) -> Optional[str]:
        if tag_key is None or value is None:
            return None
        code = sanitize_code(value)
        if code is None:
            return None
        return self._code_for(name).normalize(code)

    def _bump_fields(self, record, table, ts_ns: int) -> None:
        for name, path, tag_key in table:
            if path is None:
                # A counted event rather than a field (one count per record).
                self._bump(name, None, None, 1, ts_ns)
                continue
            value = _dig(record, path)
            if value is None:
                continue
            tag_value = self._tag_value(name, tag_key, value)
            if tag_key is not None and isinstance(value, str):
                # A tag-only series (`finish_reason`, `protocol`, a fallback
                # reason): the value is the label, the count is the record.
                self._bump(name, tag_key, tag_value, 1, ts_ns)
                continue
            amount = _as_number(value)
            # Zero is not a measurement here and negative deltas do not occur;
            # either way there is nothing to add to a cumulative total.
            if not amount or amount < 0:
                continue
            self._bump(name, tag_key, tag_value, amount, ts_ns)

    def _bump_tagged_fields(self, record, table, ts_ns: int) -> None:
        for name, path, tag_key, tag_value in table:
            amount = _as_number(_dig(record, path))
            # Zero is not a measurement and deltas are not negative: either way
            # there is nothing to add to a cumulative total.
            if not amount or amount < 0:
                continue
            self._bump(name, tag_key, self._code_for(name).normalize(tag_value),
                       amount, ts_ns)

    def _bump_host_work_seconds(self, record, ts_ns: int) -> None:
        for path, work in HOST_WORK_SECONDS_COUNTER:
            amount = _as_number(_dig(record, path))
            if not amount or amount < 0:
                continue
            self._bump('ninfer_host_work_seconds_total', 'work',
                       self._code_for('ninfer_host_work_seconds_total')
                       .normalize(work),
                       amount, ts_ns)

    def _bump(self, name: str, tag_key: Optional[str], tag_value: Optional[str],
              amount: float, ts_ns: int) -> None:
        key = (name, tag_key, tag_value)
        entry = self._counters.get(key)
        if entry is None:
            entry = {'total': 0.0, 'ts': ts_ns}
            self._counters[key] = entry
        entry['total'] += amount
        entry['ts'] = ts_ns

    def _set_gauge(self, name: str, tag_key: Optional[str],
                   tag_value: Optional[str], value: float, ts_ns: int) -> None:
        key = (name, tag_key, tag_value)
        entry = self._gauges.get(key)
        if entry is None:
            entry = {'value': value, 'ts': ts_ns}
            self._gauges[key] = entry
        else:
            entry['value'] = value
            entry['ts'] = ts_ns

    def _record_histograms(self, record, table, ts_ns: int) -> None:
        for name, path, bucket_set in table:
            value = _as_number(_dig(record, path))
            if value is None or value < _MIN_DURATION_SECONDS:
                # A phase the engine skipped reports zero; that is "not
                # measured", not a request that took no time.
                continue
            bounds = (_ENGINE_LATENCY_BUCKETS if bucket_set == 'engine'
                      else _REQUEST_LATENCY_BUCKETS)
            histogram = self._histograms.get(name)
            if histogram is None:
                histogram = _LatencyHistogram(bounds)
                self._histograms[name] = histogram
            histogram.add(value)
            # The histogram's own measurement time: the event that last
            # extended it, which is not the newest event overall.
            self._histogram_ts[name] = ts_ns

    def _flush(self, watcher) -> None:
        for (name, tag_key, tag_value), entry in self._counters.items():
            try:
                watcher.set_counter(
                    name=name, total=entry['total'], measurement_ts=entry['ts'],
                    tags=_tags(tag_key, tag_value))
            except Exception:
                logger.debug('NInferRecorder: error setting counter %s', name,
                             exc_info=True)

        for (name, tag_key, tag_value), entry in self._gauges.items():
            try:
                watcher.set_gauge(
                    name=name, value=entry['value'], measurement_ts=entry['ts'],
                    tags=_tags(tag_key, tag_value))
            except Exception:
                logger.debug('NInferRecorder: error setting gauge %s', name,
                             exc_info=True)

        for name, histogram in self._histograms.items():
            if histogram.count <= 0:
                continue
            bins, counts = histogram.bins_and_counts()
            try:
                watcher.set_histogram(
                    name=name,
                    bins=bins or None,
                    counts=counts or None,
                    measurement_ts=self._histogram_ts.get(name),
                    count=histogram.count,
                    sum_val=histogram.sum,
                    min_val=histogram.min,
                    max_val=histogram.max)
            except Exception:
                logger.debug('NInferRecorder: error setting histogram %s', name,
                             exc_info=True)

    def _log(self, watcher, *, level, message, event, ts_ns) -> None:
        if self._log_entries_this_tick >= MAX_LOG_ENTRIES_PER_TICK:
            # The counters above already saw the event; only the log entry is
            # dropped, so a flood of rejections cannot flood the log ring.
            self._dropped_log_entries += 1
            return
        self._log_entries_this_tick += 1
        # No request id: the log store keeps 200 entries, but a consumer that
        # groups by tag would turn an id into unbounded cardinality.
        tags = {
            'process.pid': str(self.pid),
            'scope.name': 'workload',
            'event': event,
        }
        try:
            watcher.log_store().log_message(
                level=level,
                # Truncated rather than dropped: log_message rejects anything
                # over its own limit outright.
                message=str(message)[:LogStore.MESSAGE_SIZE_LIMIT],
                timestamp_ns=ts_ns,
                tags=tags)
        except Exception:
            logger.debug('NInferRecorder: error recording log entry',
                         exc_info=True)
