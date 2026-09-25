"""Turns an `ninfer_bench` report into signals.

`ninfer_bench` is the benchmark harness, not the server: it drives
`ninfer::Engine` directly and writes a single JSON report at the end of the run
(`-o json [--output-file PATH]`). It never writes the `ninfer-serve` request
log that `ninfer_recorder.py` tails, so under the profiler a benchmark produced
no `ninfer_*` metrics at all and its numbers could not be correlated with GPU
timing, memory, or the other instances from the same campaign.

This recorder imports that report instead. It reuses the serve recorder's
metric names — `ninfer_request_decode_seconds`, `ninfer_speculative_*`, the
`ninfer_memory_*` startup gauges — so a benchmark result and a served run are
read the same way, with the bench's own test label as a tag
(`{test: "pp512"}`) to keep the series apart. Values come from the report
verbatim; nothing is synthesized.

Timing: the report is written as the harness's last act before it exits, so the
import happens in `finalize()`, which the watcher runs after the target is gone
but before the collector and the /signals endpoint shut down. The numbers are
therefore in the stores, and uploaded on the final flush when an API key is
set, but a local `/signals` read only catches them in the brief window before
the endpoint closes — the same trade every end-of-run artifact has here.
"""

import json
import logging
import os
import shlex
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

COMMAND_BASENAME = 'ninfer-bench'
ARTIFACT_TYPE = 'ninfer_bench_report'
# Reported by the harness; bumped when its report shape changes.
SCHEMA_VERSION = 15

# The report's per-test seconds, and the serve-recorder metric each mirrors.
# Built from the measured per-repetition samples, not the report's own
# mean-and-stddev summary, so count/sum/min/max stay exact.
_SECONDS_METRICS: Tuple[Tuple[str, str], ...] = (
    # (metric name, per-rep timing key)
    ('ninfer_request_duration_seconds', 'total_seconds'),
    ('ninfer_request_prepare_seconds', 'prepare_seconds'),
    ('ninfer_request_prefill_seconds', 'prefill_seconds'),
    ('ninfer_request_decode_seconds', 'decode_seconds'),
    ('ninfer_request_vision_seconds', 'vision_seconds'),
)

# Startup gauges, mirroring ninfer_recorder's SERVER_START_GAUGES where the
# report carries the same field. (name, dotted path into the report)
_STARTUP_GAUGES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ('ninfer_model_load_seconds', ('load', 'load_seconds')),
    ('ninfer_model_upload_seconds', ('load', 'upload_seconds')),
    ('ninfer_memory_weights_used_bytes', ('memory', 'weights', 'used_bytes')),
    ('ninfer_memory_weights_capacity_bytes',
     ('memory', 'weights', 'capacity_bytes')),
    ('ninfer_memory_sequence_used_bytes', ('memory', 'sequence', 'used_bytes')),
    ('ninfer_memory_sequence_capacity_bytes',
     ('memory', 'sequence', 'capacity_bytes')),
    ('ninfer_memory_workspace_used_bytes', ('memory', 'workspace', 'used_bytes')),
    ('ninfer_memory_workspace_capacity_bytes',
     ('memory', 'workspace', 'capacity_bytes')),
    ('ninfer_memory_kv_payload_bytes', ('memory', 'kv_payload_bytes')),
    # The token-count capacity: same field the serve recorder reports.
    ('ninfer_engine_kv_capacity_tokens', ('memory', 'kv_capacity')),
    ('ninfer_model_max_context_tokens', ('memory', 'max_context')),
)

# Watch-wide tags describing the run, mirroring the serve recorder's
# SERVER_START_TAGS. All fixed report fields, so the count is bounded.
_RUN_TAGS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ('model', ('artifact', 'path')),
    ('architecture', ('load', 'architecture')),
    ('kv_cache', ('memory', 'kv_cache')),
    ('kv_capacity_mode', ('memory', 'kv_capacity_mode')),
    ('gpu_name', ('environment', 'gpu_name')),
    ('speculative_backend', ('config', 'speculative_backend')),
    ('draft_tokens', ('config', 'draft_tokens')),
    ('cuda_graph', ('config', 'use_cuda_graph')),
    ('repetitions', ('config', 'repetitions')),
    ('corpus_tokens', ('config', 'corpus_tokens')),
)

# Per-test speculative counters, named as the serve recorder names them.
_SPECULATIVE_COUNTERS: Tuple[Tuple[str, str], ...] = (
    ('ninfer_speculative_rounds_total', 'rounds'),
    ('ninfer_speculative_drafted_tokens_total', 'drafted_tokens'),
    ('ninfer_speculative_accepted_tokens_total', 'accepted_tokens'),
    ('ninfer_speculative_fallback_steps_total', 'fallback_steps'),
)


def _dig(payload: Any, path: Tuple[str, ...]) -> Any:
    """Walk a dotted path through nested dicts. None if any hop is missing,
    so a report that drops a section imports what it does have."""
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _finite_number(value: Any) -> Optional[float]:
    """None for null, bool, and non-finite values: the store takes numbers,
    and `null` must stay 'not measured' rather than become 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and value not in (float('inf'), float('-inf')) else None


def command_basename(args) -> Optional[str]:
    """Basename of the launched command without a platform executable suffix.
    Accepts an argv list or the space-joined command string the process
    monitor supplies."""
    if not args:
        return None
    if isinstance(args, str):
        try:
            args = shlex.split(args)
        except ValueError:
            return None
    try:
        return os.path.basename(args[0])
    except (IndexError, TypeError):
        return None


def is_ninfer_bench(args) -> bool:
    """True iff the launched command is the benchmark harness. The underscore
    spelling the build produces is accepted alongside the dashed one."""
    name = command_basename(args)
    if not name:
        return False
    stem = name[:-4] if name.endswith('.exe') else name
    return stem in (COMMAND_BASENAME, 'ninfer_bench')


def parse_report_path(args) -> Tuple[Optional[str], bool]:
    """Return the report path the harness will write and whether it was asked
    for JSON.

    Returns (path, is_json). `path` is None when the harness prints to stdout,
    which there is nothing to import afterwards; `is_json` is False when the
    user chose table or csv, which are not machine-readable.
    """
    if not args:
        return None, False
    if isinstance(args, str):
        try:
            args = shlex.split(args)
        except ValueError:
            return None, False
    args = list(args)
    path = None
    is_json = False
    for i, arg in enumerate(args):
        if arg in ('-o', '--output'):
            if i + 1 < len(args):
                is_json = args[i + 1].lower() == 'json'
        elif arg.startswith('--output='):
            is_json = arg.split('=', 1)[1].lower() == 'json'
        elif arg == '--output-file':
            if i + 1 < len(args):
                path = args[i + 1]
        elif arg.startswith('--output-file='):
            path = arg.split('=', 1)[1]
    return (path or None), is_json


def _histogram_kwargs(values: List[float]) -> Optional[Dict[str, Any]]:
    """count/sum/min/max for the store from measured per-rep values.

    No bins: these samples never passed through the writer's bin grid, so
    `p50`/`p95` in the payload read as null — a real "not measured" rather
    than a fabricated quantile. The exact four are measured and are what the
    report actually supports.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        'count': len(values),
        'sum_val': sum(values),
        'min_val': min(values),
        'max_val': max(values),
    }


class NinferBenchRecorder(BaseRecorder):
    """Imports one `ninfer_bench` report as `ninfer_*` metrics.

    Root-pid only, like the serve recorder: one report belongs to the harness
    process, and a child that merely looks like the harness is not it.
    """

    def __init__(self, root_pid=None, pid=None, args=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._disabled = True
        self._path = None
        self._imported = False
        self._missing_logged = False
        self._lock = threading.Lock()

    def setup(self):
        if self.pid is None or self.root_pid is None or self.pid != self.root_pid:
            logger.debug('NinferBenchRecorder is root-pid only; skipping setup')
            return
        if not is_ninfer_bench(self.args):
            logger.debug('NinferBenchRecorder: command is not %s; skipping setup',
                         COMMAND_BASENAME)
            return
        path, is_json = parse_report_path(self.args)
        if not path:
            logger.debug('NinferBenchRecorder: no --output-file to import; '
                         'the harness printed its report to stdout')
            return
        if not is_json:
            logger.debug('NinferBenchRecorder: %s is not a json report; '
                         'skipping setup', path)
            return
        self._path = path
        self._disabled = False
        logger.debug('NinferBenchRecorder importing %s', path)

    def on_tick(self):
        # The report appears only once, at the end of the run. Importing it
        # here when it happens to be readable gets the numbers into the stores
        # early; finalize() is the guarantee, not this path.
        self._import_if_readable()

    def finalize(self):
        # The harness writes its report as its last act before exiting, so by
        # the time the watcher finalizes us the file is complete. This is the
        # deterministic import.
        if self._disabled:
            return
        try:
            self._import_report()
        except Exception:
            logger.debug('NinferBenchRecorder: error during finalize',
                         exc_info=True)

    def shutdown(self):
        self._disabled = True

    # -- import -----------------------------------------------------------

    def _import_if_readable(self):
        if self._disabled or self._imported:
            return
        try:
            if not os.path.exists(self._path):
                if not self._missing_logged:
                    self._missing_logged = True
                    logger.debug('NinferBenchRecorder: %s does not exist yet',
                                 self._path)
                return
        except OSError:
            return
        self._import_report()

    def _import_report(self):
        """Read and record the report once. A partially written file fails to
        parse and is left for the next attempt; only a completed, well-formed
        report marks the import done."""
        with self._lock:
            if self._disabled or self._imported:
                return
            try:
                with open(self._path, 'r', encoding='utf-8') as handle:
                    report = json.load(handle)
            except FileNotFoundError:
                return
            except (OSError, ValueError) as exc:
                # Not written yet, or mid-write; finalize() gets the real one.
                logger.debug('NinferBenchRecorder: %s not readable yet: %s',
                             self._path, exc)
                return
            if not self._is_report(report):
                logger.warning('NinferBenchRecorder: %s is not a %s report; '
                               'skipping', self._path, ARTIFACT_TYPE)
                self._imported = True
                return
            self._imported = True
            self._record(report)

    def _is_report(self, report) -> bool:
        if not isinstance(report, dict):
            return False
        if report.get('artifact_type') != ARTIFACT_TYPE:
            return False
        version = report.get('schema_version')
        if not isinstance(version, int):
            return False
        if version > SCHEMA_VERSION:
            # Newer fields may be missing below; import what still lines up
            # rather than discarding a report we can partly read.
            logger.warning('NinferBenchRecorder: report schema %d is newer '
                           'than the %d this profiler knows; importing the '
                           'fields it recognizes', version, SCHEMA_VERSION)
        return True

    def _record(self, report):
        if not graphsignal.watcher.is_configured():
            return
        watcher = graphsignal.watcher.watcher()
        now_ns = time.time_ns()
        # Bench-wide context, so a bench and a served run read alike.
        for tag, path in _RUN_TAGS:
            value = _dig(report, path)
            if isinstance(value, bool):
                # A config flag like use_cuda_graph is exactly the kind of
                # thing worth tagging; spell it rather than drop it.
                watcher.set_tag(f'bench.{tag}', 'true' if value else 'false')
            elif isinstance(value, (str, int, float)):
                watcher.set_tag(f'bench.{tag}', str(value)[:200])

        for name, path in _STARTUP_GAUGES:
            value = _finite_number(_dig(report, path))
            if value is not None:
                watcher.set_gauge(name, value, now_ns)

        tests = report.get('tests')
        if not isinstance(tests, list):
            return
        for test in tests:
            if not isinstance(test, dict):
                continue
            self._record_test(watcher, test, now_ns)

    def _record_test(self, watcher, test, now_ns):
        label = test.get('label')
        if not isinstance(label, str) or not label:
            return
        tags = {'test': label}
        kind = test.get('kind')
        if isinstance(kind, str) and kind:
            tags['kind'] = kind

        reps = test.get('reps')
        reps = [r for r in reps if isinstance(r, dict)] if isinstance(reps, list) else []

        for name, timing_key in _SECONDS_METRICS:
            kwargs = _histogram_kwargs([
                _finite_number((r.get('timings') or {}).get(timing_key))
                for r in reps
            ])
            if kwargs is not None:
                watcher.set_histogram(
                    name, measurement_ts=now_ns, tags=dict(tags), **kwargs)

        # Workspace peaks are per test, not per repetition.
        for report_key, name in (
                ('workspace_peak_bytes', 'ninfer_memory_workspace_peak_bytes'),
                ('workspace_allocator_peak_bytes',
                 'ninfer_memory_workspace_allocator_peak_bytes')):
            value = _finite_number(test.get(report_key))
            if value is not None:
                watcher.set_gauge(name, value, tags=dict(tags),
                                  measurement_ts=now_ns)

        # Throughput: the benchmark's headline number. The harness already
        # computed these rates over exactly these repetitions, so take them
        # rather than re-deriving: a rate the report reports as null (a
        # prefill-only test has no decode rate) stays "not measured" instead of
        # becoming a fabricated 0.0. Token counters are recorded too, so the
        # read-order the serve recorder supports works here unchanged.
        self._record_throughput(watcher, tags, test, now_ns)
        self._record_speculative(watcher, tags, test, reps, now_ns)

    def _record_throughput(self, watcher, tags, test, now_ns):
        for report_key, fallback_key, gauge_name, counter_name, token_key in (
                ('prefill_tok_s_mean', None,
                 'ninfer_throughput_prefill_tokens_per_second',
                 'ninfer_throughput_prefill_tokens_total', 'n_prompt'),
                # Output-token rate is what a reader means by decode
                # throughput; the engine-token variant excludes rejected draft
                # tokens and is only a fallback.
                ('decode_output_tok_s_mean', 'decode_engine_tok_s_mean',
                 'ninfer_throughput_decode_tokens_per_second',
                 'ninfer_throughput_decode_tokens_total', 'n_gen'),
        ):
            rate = _finite_number(test.get(report_key))
            if rate is None and fallback_key is not None:
                rate = _finite_number(test.get(fallback_key))
            if rate is not None:
                watcher.set_gauge(gauge_name, rate, tags=dict(tags),
                                  measurement_ts=now_ns)
            tokens = _finite_number(test.get(token_key))
            if tokens is not None:
                watcher.set_counter(counter_name, int(tokens), tags=dict(tags),
                                    measurement_ts=now_ns)

    def _record_speculative(self, watcher, tags, test, reps, now_ns):
        summary = test.get('speculative')
        summary = summary if isinstance(summary, dict) else {}
        backend = summary.get('backend')
        spec_tags = dict(tags)
        if isinstance(backend, str) and backend:
            spec_tags['backend'] = backend

        # Totals over the measured repetitions, from the per-rep blocks.
        for name, key in _SPECULATIVE_COUNTERS:
            total = 0
            seen = False
            for rep in reps:
                block = rep.get('speculative')
                value = _finite_number(
                    block.get(key) if isinstance(block, dict) else None)
                if value is not None:
                    total += value
                    seen = True
            if seen:
                watcher.set_counter(name, int(total), tags=dict(spec_tags),
                                    measurement_ts=now_ns)

        # The harness already computed the rates over exactly these
        # repetitions; prefer them over re-deriving from the counters.
        for report_key, name in (
                ('acceptance_rate', 'ninfer_speculative_acceptance_rate'),
                ('acceptance_length', 'ninfer_speculative_acceptance_length')):
            value = _finite_number(summary.get(report_key))
            if value is not None:
                watcher.set_gauge(name, value, tags=dict(spec_tags),
                                  measurement_ts=now_ns)
