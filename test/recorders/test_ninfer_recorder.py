"""Fixture-driven tests for the ninfer-serve request log recorder.

The fixtures mirror the JSON that `src/serve/request_log.cpp` (schema 24)
actually writes: `event_base()` for the envelope, and the `server_start`,
`request_start`, `request_done`, `request_error`, `request_rejected` and
`throughput` payloads built by the `format_*_json` functions in that file.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import graphsignal.watcher
from graphsignal.recorders.ninfer_recorder import (
    ARTIFACT_TYPE, COMMAND_BASENAME, MAX_TAG_VALUES, SCHEMA_VERSION,
    NInferRecorder, command_basename, is_ninfer_serve, normalize_timestamp_ns,
    parse_request_log_flag, resolve_jsonl_path, sanitize_code)
from test.test_utils import configure_test_watcher, find_metric

# The artifact timestamps in epoch milliseconds; the stores take nanoseconds.
_TS_MS = 1_712_345_678_901
_TS_NS = _TS_MS * 1_000_000
_INSTANCE = 'serve-41234-1712345678901234'


def _event(event, ts_ms=_TS_MS, instance=_INSTANCE, **payload):
    """`event_base()` from request_log.cpp."""
    record = {
        'artifact_type': ARTIFACT_TYPE,
        'schema_version': SCHEMA_VERSION,
        'event': event,
        'timestamp_unix_ms': ts_ms,
        'server_instance_id': instance,
    }
    record.update(payload)
    return record


def _with(record, ts_ms=None, instance=None, **payload):
    """A copy of `record` with payload fields overridden, and the envelope
    (`timestamp_unix_ms`, `server_instance_id`) only when asked."""
    updated = dict(record)
    if ts_ms is not None:
        updated['timestamp_unix_ms'] = ts_ms
    if instance is not None:
        updated['server_instance_id'] = instance
    updated.update(payload)
    return updated


def _request(**overrides):
    """`request_json()` — the request half of request_start/request_done."""
    request = {
        'request_id': '1',
        'protocol': 'openai_chat_completions',
        'model': 'qwen3.6-27b',
        'stream': False,
        'message_count': 3,
        'media_item_count': 0,
        'requested_output_tokens': 128,
        'requested_output_tokens_source': 'client',
        'tool_count': 0,
        'tool_choice': 'auto',
        'has_tool_history': False,
        'enable_thinking': True,
        'thinking_budget': 1024,
        'requested_reasoning_effort': 'medium',
        'preserve_thinking': None,
        'preserve_thinking_semantic_change': False,
        'sampling': {'temperature': 0.7, 'top_p': 0.95, 'top_k': 40, 'min_p': 0.05,
                     'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'seed': 7},
    }
    request.update(overrides)
    return request


def _server_start(ts_ms=_TS_MS, instance=_INSTANCE, **overrides):
    """`format_server_start_json()` with the full v24 nesting."""
    record = _event('server_start', ts_ms=ts_ms, instance=instance, server={
        'host': '0.0.0.0', 'port': 8080, 'public_model_id': 'qwen3.6-27b',
        'api_key_configured': False, 'cors_enabled': False,
        'max_request_bytes': 33554432, 'media_cache_bytes': 1073741824,
        'media_live_bytes': 268435456, 'media_preprocess_threads': 8,
        'request_log_jsonl': '/tmp/ninfer/request_log.jsonl',
        'default_output_tokens': 512, 'default_thinking': True,
        'default_thinking_budget': 2048, 'default_preserve_thinking': None,
    }, artifact={
        'path': '/models/qwen3.6-27b.ninfer', 'size_bytes': 27487790694,
        'architecture': 'qwen3_moe', 'name': 'Qwen3.6-27B',
        'formats': ['bf16'], 'prefill_signature': 'abc123',
        'bytes_read': 27487790694, 'host_to_device_bytes': 27487790694,
        'peak_staging_bytes': 1073741824, 'device_object_count': 1013,
        'host_object_count': 0, 'load_seconds': 41.5, 'upload_seconds': 0.0,
    }, engine={
        'device': 0, 'max_context': 131072, 'kv_capacity_mode': 'auto',
        'kv_capacity': 4294967296, 'kv_capacity_page_groups': 2048,
        'kv_capacity_max_page_groups': 4096, 'expert_cache_slots': 0,
        'expert_cache_bytes': 0, 'ngram_residency': 'mtp3',
        'max_concurrency': 64, 'max_pending_requests': 256,
        'pending_timeout_ms': 60000, 'prefill_chunk': 8192,
        'log_stats_interval_ms': 1000, 'kv_cache': 'nvfp4', 'vision': True,
        'cuda_graph': True, 'prefix_reuse': True,
        'speculative_backend': 'mtp', 'speculative_draft_window': 3,
        'proposal_head': 'optimized',
        'context_cost': {'transfer_source': 'measured', 'prefill_source': 'measured',
                         'hardware_class': 'blackwell', 'prefill_signature': 'abc123',
                         'preset_path': ''},
        'context_cache': {'enabled': True, 'device_state_slots': 8,
                          'total_device_state_slots': 72, 'host_state_slots': 64,
                          'host_kv_capacity_bytes': 8589934592,
                          'max_private_continuations': 8,
                          'max_shared_prefixes': 16,
                          'max_long_anchors_per_continuation': 4},
    }, sampling_defaults={
        'thinking': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
                     'min_p': 0.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0},
        'non_thinking': {'temperature': 0.7, 'top_p': 0.95, 'top_k': 40,
                         'min_p': 0.05, 'presence_penalty': 0.0,
                         'frequency_penalty': 0.0},
        'server_overrides': {'temperature': None, 'top_p': None, 'top_k': None,
                             'min_p': None, 'presence_penalty': None,
                             'frequency_penalty': None, 'seed': None},
        'omitted_seed': 'random', 'greedy': False,
    }, memory={
        'weights': {'capacity_bytes': 27487790694, 'used_bytes': 27487790694,
                    'peak_used_bytes': 27487790694},
        'sequence': {'capacity_bytes': 1073741824, 'used_bytes': 268435456,
                     'peak_used_bytes': 536870912},
        'workspace': {'capacity_bytes': 2147483648, 'used_bytes': 1073741824,
                      'peak_used_bytes': 1610612736},
        'vision_workspace': {'aggregate_prompt_tokens': 4096, 'max_item_tokens': 1024,
                             'general_capacity_bytes': 536870912,
                             'encode_peak_bytes': 268435456,
                             'handoff_offset_bytes': 0, 'handoff_capacity_bytes': 0,
                             'handoff_active_bytes': 0, 'handoff_peak_bytes': 0},
        'minimum_runtime_reservation_bytes': 2147483648,
        'kv_capacity_increment_bytes': 1073741824,
        'runtime_reservation_bytes': 3221225472,
        'available_after_weights_bytes': 12884901888,
        'available_after_startup_bytes': 9663676416,
        'kv_capacity_headroom_bytes': 1073741824,
        'planned_slack_bytes': 2147483648,
        'cuda_graph_allowance_bytes': 1073741824,
        'kv_payload_bytes': 2147483648,
        'host_state_capacity_slots': 64, 'host_state_occupied_slots': 8,
        'host_state_image_bytes': 1073741824, 'host_kv_page_group_bytes': 4096,
        'host_cache_budget_bytes': 8589934592, 'host_kv_capacity_bytes': 8589934592,
        'host_kv_occupied_bytes': 1073741824,
    }, environment={
        'device': 0, 'gpu_name': 'NVIDIA GB10', 'gpu_uuid': 'GPU-9d2c1f4e-0000-0000-0000-000000000000',
        'total_device_memory_bytes': 137438953472,
        'compute_capability_major': 12, 'compute_capability_minor': 1,
        'cuda_compile_version': '13.0', 'cuda_runtime_version': '13.0',
        'cuda_driver_version': '580.65',
    }, argv=['ninfer-serve', '--model', '/models/qwen3.6-27b.ninfer',
             '--port', '8080', '--request-log-jsonl',
             '/tmp/ninfer/request_log.jsonl'])
    record.update(overrides)
    return record


def _request_start(ts_ms=_TS_MS, instance=_INSTANCE, **overrides):
    """`format_request_start_json()` — `preparation_json()`."""
    record = _event('request_start', ts_ms=ts_ms, instance=instance,
                    request=_request(), preparation_seconds={
        'total': 0.0125, 'acquisition': 0.002, 'media_preprocess': 0.0,
        'media_preprocess_work': 0.0, 'tokenize': 0.0031, 'media_items': 0,
        'media_bytes': 0, 'raw_patches': 0, 'vision_tokens': 0, 'patch_bytes': 0,
        'cache_hits': 0, 'cache_misses': 0, 'singleflight_waits': 0,
        'built_patch_bytes': 0, 'reused_patch_bytes': 0,
    })
    record.update(overrides)
    return record


def _request_start_with_media(ts_ms=_TS_MS, instance=_INSTANCE, **overrides):
    record = _request_start(ts_ms=ts_ms, instance=instance,
                            request=_request(media_item_count=2, message_count=5),
                            preparation_seconds={
                                'total': 0.412, 'acquisition': 0.05,
                                'media_preprocess': 0.31, 'media_preprocess_work': 0.28,
                                'tokenize': 0.052, 'media_items': 2,
                                'media_bytes': 4194304, 'raw_patches': 1536,
                                'vision_tokens': 12288, 'patch_bytes': 8388608,
                                'cache_hits': 1, 'cache_misses': 1,
                                'singleflight_waits': 0, 'built_patch_bytes': 4194304,
                                'reused_patch_bytes': 4194304,
                            })
    record.update(overrides)
    return record


def _request_done(ts_ms=_TS_MS, instance=_INSTANCE, **overrides):
    """`format_request_done_json()` — result / timings_seconds / engine_timing /
    speculative / materialization."""
    record = _event('request_done', ts_ms=ts_ms, instance=instance,
                    request=_request(), result={
        'finish_reason': 'stop_token', 'prompt_tokens': 1536,
        'completion_tokens': 128, 'computed_prefill_tokens': 1280,
        'prefix_cache_hit_tokens': 256, 'prefix_reuse_path': 'shared_stable_prefix',
        'thinking_budget': 1024, 'model_thinking_tokens': 64,
        'thinking_control_tokens': 8, 'thinking_control_applied': True,
        'tool_call_count': 0,
        'tool_call_parse': {'marker_seen': False, 'structured_call_count': 0,
                            'empty_arguments_omitted': 0,
                            'schema_mismatch_arguments': 0,
                            'duplicate_parameters_repaired': 0,
                            'fallback_reason': 'none'},
    }, timings_seconds={
        'prepare': 0.0125, 'ttft': 0.184, 'vision': 0.0, 'prefill': 0.121,
        'decode': 1.06, 'total': 1.256,
    }, engine_timing={
        'queue_wait_seconds': 0.0043,
        'host_exposed_seconds': {
            'engine_boundary': 0.0011, 'program_submit': 0.0022,
            'program_post': 0.0017, 'engine_commit_output': 0.0009,
            'engine_maintenance': 0.0006, 'total': 0.0065},
        'device_wait_exposed_seconds': 0.0412,
        'decode': {'host_exposed_seconds': 0.0044,
                   'device_wait_exposed_seconds': 0.0331, 'rounds': 128},
        'units': {'prefill': 2, 'control': 1},
    }, speculative={
        'backend': 'mtp', 'draft_window': 3, 'rounds': 128, 'drafted_tokens': 384,
        'accepted_tokens': 256, 'fallback_steps': 2,
        'accepted_per_position': 0.666, 'rounds_by_draft_length': [128, 118, 74],
    }, materialization={
        'predicted_now_ns': 1200000, 'predicted_future_loss_ns': 400000,
        'predicted_total_ns': 1600000, 'targets_evaluated': 4, 'projection_work': 96,
        'planning_elapsed_ns': 450000, 'search_elapsed_ns': 1800000,
        'stop_reason': 'budget', 'budget_exhausted': False,
        'selected_degradation_units': 0, 'selected_maximal_fallback': False,
        'initial_predicted_total_ns': 2400000, 'first_improvement_ns': 1500000,
        'incumbent_improvements': 2, 'search_work': 512, 'search_granted_ns': 2000000,
        'search_renewals': 1, 'search_discovery_used': True,
        'search_overshoot_ns': 20000, 'search_stop_phase': 'frontier',
        'search_boundary_limited': False,
    })
    record.update(overrides)
    return record


def _request_error(message='CUDA error: an illegal memory access was encountered',
                   ts_ms=_TS_MS, instance=_INSTANCE):
    return _event('request_error', ts_ms=ts_ms, instance=instance,
                  request=_request(), error={'message': message})


def _request_rejected(code='context_length_exceeded', status='400',
                      error_type='invalid_request_error',
                      message='maximum context length is 131072 tokens',
                      ts_ms=_TS_MS, instance=_INSTANCE):
    return _event('request_rejected', ts_ms=ts_ms, instance=instance, phase='prepare',
                  request=_request(enable_thinking=None, thinking_budget=None,
                                   requested_reasoning_effort=None),
                  error={'status': status, 'type': error_type, 'code': code,
                         'param': 'messages', 'message': message})


def _throughput(ts_ms=_TS_MS + 1000, instance=_INSTANCE, **overrides):
    """`format_throughput_json()` — per-interval deltas plus current state."""
    record = _event('throughput', ts_ms=ts_ms, instance=instance,
                    interval_seconds=1.0, tokens={
        'computed_prefill': 1536, 'committed_decode': 384,
    }, throughput_tokens_per_second={'prefill': 1536.0, 'decode': 384.0},
        scheduler={'running': 6, 'prefilling': 1, 'decode_ready': 5, 'waiting': 3,
                   'materializing': 0, 'capture_pending': 1,
                   'terminal_pending': 0},
        decode_batch={'rounds': 384, 'row_rounds': 2304, 'average_size': 6.0},
        host_work={
            'elapsed_seconds': {'engine_boundary': 0.0041, 'program_submit': 0.0082,
                                'program_post': 0.0063, 'engine_commit_output': 0.0027,
                                'engine_maintenance': 0.0011, 'total': 0.0224},
            'device_wait_seconds': 0.1536,
            'work_class_seconds': {'decode_host': 0.0071, 'decode_device_wait': 0.1121,
                                   'prefill_host': 0.0044, 'prefill_device_wait': 0.0225,
                                   'control_host': 0.0009, 'control_device_wait': 0.0031},
            'detail_subset_seconds': {'admission_policy': 0.0012,
                                      'context_progress': 0.0008,
                                      'stats_publication': 0.0004},
            'detail_invocations': {'admission_policy': 384, 'context_progress': 384,
                                   'stats_publication': 128},
            'units': {'prefill': 6, 'control': 128},
            'decode_host_microseconds_per_round': 18.5,
            'decode_host_microseconds_per_row_round': 3.08,
            'decode_device_wait_microseconds_per_round': 291.9,
            'detail_microseconds_per_invocation': {
                'admission_policy': 3.12, 'context_progress': 2.08,
                'stats_publication': 3.12},
        }, context_cache={
            'captures': {'completed': 2, 'aborted': 0, 'skipped': 1},
            'salvage': {'published': 1},
            'selections': {'root': 4, 'private_endpoint': 2, 'private_turn_closure': 1,
                           'private_response_replay': 0, 'private_long_anchor': 1,
                           'shared_stable_prefix': 11,
                           'reused_prompt_tokens': 1280},
            'last_selection': {'frontier_tokens': 4096},
            'state_operations': {'moves': 12, 'forks': 3, 'restores': 1},
            'state_transfers': {
                'd2h': {'count': 2, 'bytes': 4194304, 'seconds': 0.004},
                'h2d': {'count': 3, 'bytes': 8388608, 'seconds': 0.009},
                'd2d': {'count': 1, 'bytes': 1048576, 'seconds': 0.001}},
            'main_kv_transfers': {
                'd2h': {'pages': 8, 'bytes': 1048576, 'seconds': 0.002},
                'h2d': {'pages': 16, 'bytes': 2097152, 'seconds': 0.005},
                'd2d': {'pages': 4, 'bytes': 524288, 'seconds': 0.001}},
            'backend_kv_transfers': {
                'd2h': {'pages': 2, 'bytes': 262144, 'seconds': 0.0004},
                'h2d': {'pages': 4, 'bytes': 524288, 'seconds': 0.0008},
                'd2d': {'pages': 1, 'bytes': 131072, 'seconds': 0.0002}},
            'pressure': {'spill_pages': 6, 'partial_tail_cow_pages': 2,
                         'private_owners_degraded': 1, 'private_owners_evicted': 0,
                         'shared_owners_degraded': 2, 'shared_owners_evicted': 1,
                         'checkpoints_dropped': 0, 'searches': 3,
                         'search_budget_exhaustions': 0,
                         'maximal_fallback_selections': 0,
                         'historical_fork_hits': 5},
            'occupancy': {'device_state_slots': 4, 'host_state_slots': 8,
                          'device_main_kv_pages': 2048,
                          'device_main_kv_lease_pages': 2100,
                          'device_backend_kv_pages': 128,
                          'device_backend_kv_lease_pages': 130,
                          'host_kv_bytes': 1073741824,
                          'shared_active_references': 3},
            'actual_transfer_seconds': 0.014,
        })
    record.update(overrides)
    return record


class ArgParsingTest(unittest.TestCase):
    def test_command_basename(self):
        self.assertEqual(command_basename(['/usr/bin/ninfer-serve']), 'ninfer-serve')
        self.assertEqual(command_basename(['./venv/bin/ninfer-serve']), 'ninfer-serve')
        self.assertEqual(command_basename(['ninfer-serve.exe']), 'ninfer-serve')
        self.assertEqual(command_basename(['python', 'app.py']), 'python')
        self.assertIsNone(command_basename(None))
        self.assertIsNone(command_basename([]))
        self.assertIsNone(command_basename([None]))

    def test_is_ninfer_serve_is_exact(self):
        self.assertTrue(is_ninfer_serve('/usr/local/bin/ninfer-serve model.ninfer'))
        self.assertTrue(is_ninfer_serve(['/usr/local/bin/ninfer-serve']))
        self.assertTrue(is_ninfer_serve(['ninfer-serve', '--model', 'm']))
        # Not this recorder's engine: similar names must not be guessed at, and
        # the other ninfer entry points write no request log.
        self.assertFalse(is_ninfer_serve(['vllm', 'serve']))
        self.assertFalse(is_ninfer_serve(['python', '-m', 'ninfer.serve']))
        self.assertFalse(is_ninfer_serve(['ninfer-perplexity']))
        self.assertFalse(is_ninfer_serve(['ninfer-serve-wrapper']))
        self.assertFalse(is_ninfer_serve(None))

    def test_parse_request_log_flag_space_and_equal_forms(self):
        self.assertEqual(
            parse_request_log_flag('ninfer-serve --request-log-jsonl /tmp/a.jsonl'),
            '/tmp/a.jsonl')
        self.assertEqual(
            parse_request_log_flag(['ninfer-serve', '--request-log-jsonl', '/tmp/a.jsonl']),
            '/tmp/a.jsonl')
        self.assertEqual(
            parse_request_log_flag(['ninfer-serve', '--request-log-jsonl=/tmp/b.jsonl']),
            '/tmp/b.jsonl')

    def test_parse_request_log_flag_ignores_malformed_and_other_flags(self):
        self.assertIsNone(parse_request_log_flag(
            ['ninfer-serve', '--request-log-jsonl']))
        self.assertIsNone(parse_request_log_flag(
            ['ninfer-serve', '--request-log-jsonl', '--port', '8000']))
        self.assertIsNone(parse_request_log_flag(['ninfer-serve', '--port', '8000']))
        self.assertIsNone(parse_request_log_flag(None))
        self.assertIsNone(parse_request_log_flag(
            ['ninfer-serve', '--no-request-log-jsonl=/tmp/x.jsonl']))

    def test_parse_request_log_flag_last_occurrence_wins(self):
        self.assertEqual(parse_request_log_flag([
            'ninfer-serve', '--request-log-jsonl', '/tmp/first.jsonl',
            '--request-log-jsonl=/tmp/second.jsonl']), '/tmp/second.jsonl')

    def test_resolve_jsonl_path_env_wins_over_flag(self):
        path = resolve_jsonl_path(
            ['ninfer-serve', '--request-log-jsonl', '/tmp/from-flag.jsonl'],
            environ={'GRAPHSIGNAL_NINFER_JSONL': '/tmp/from-env.jsonl'})
        self.assertEqual(path, '/tmp/from-env.jsonl')

    def test_resolve_jsonl_path_falls_back_to_flag(self):
        self.assertEqual(
            resolve_jsonl_path(['ninfer-serve', '--request-log-jsonl', '/tmp/f.jsonl'],
                               environ={}),
            '/tmp/f.jsonl')

    def test_resolve_jsonl_path_none_when_unspecified(self):
        self.assertIsNone(resolve_jsonl_path(['ninfer-serve'], environ={}))
        self.assertIsNone(resolve_jsonl_path(
            ['ninfer-serve', '--request-log-jsonl='], environ={}))


class ValueNormalizationTest(unittest.TestCase):
    def test_normalize_timestamp_ns_scales_milliseconds(self):
        self.assertEqual(normalize_timestamp_ns(_TS_MS), _TS_NS)
        self.assertEqual(normalize_timestamp_ns(1), 1_000_000)

    def test_normalize_timestamp_ns_rejects_junk(self):
        self.assertIsNone(normalize_timestamp_ns(None))
        self.assertIsNone(normalize_timestamp_ns('later'))
        self.assertIsNone(normalize_timestamp_ns(0))
        self.assertIsNone(normalize_timestamp_ns(float('nan')))
        self.assertIsNone(normalize_timestamp_ns(True))
        self.assertEqual(normalize_timestamp_ns('later', default=42), 42)

    def test_sanitize_code(self):
        self.assertEqual(sanitize_code('Context Length Exceeded!'),
                         'context_length_exceeded')
        self.assertEqual(sanitize_code('HTTP-500'), 'http-500')
        self.assertEqual(sanitize_code('a' * 100), 'a' * 40)
        self.assertIsNone(sanitize_code('   '))
        self.assertIsNone(sanitize_code(None))
        self.assertIsNone(sanitize_code(True))
        self.assertEqual(sanitize_code(400), '400')


class NInferRecorderTestBase(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'request_log.jsonl')

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _recorder(self, args=None, environ=None, pid=100, root_pid=None,
                  jsonl_path=None):
        if root_pid is None:
            root_pid = pid
        recorder = NInferRecorder(
            root_pid=root_pid, pid=pid,
            args=args if args is not None else [f'/usr/bin/{COMMAND_BASENAME}'],
            jsonl_path=jsonl_path if jsonl_path is not None else self.path,
            environ=environ if environ is not None else {})
        recorder.setup()
        return recorder

    def _append(self, *records, trailing_newline=True):
        """Append records as JSONL. `trailing_newline=False` leaves the last
        record unterminated, as a half-written line looks on disk."""
        with open(self.path, 'a') as f:
            for index, record in enumerate(records):
                last = index == len(records) - 1
                f.write(json.dumps(record))
                if trailing_newline or not last:
                    f.write('\n')

    def _append_raw(self, text):
        with open(self.path, 'a') as f:
            f.write(text)

    def _exported(self):
        return self.watcher.metric_store().export()


class NInferRecorderActivationTest(NInferRecorderTestBase):
    def test_noop_unless_command_is_ninfer_serve(self):
        self.assertTrue(self._recorder(args=['vllm', 'serve'])._disabled)

    def test_noop_without_args(self):
        recorder = NInferRecorder(root_pid=100, pid=100, args=None,
                                  jsonl_path=self.path, environ={})
        recorder.setup()
        self.assertTrue(recorder._disabled)

    def test_noop_when_no_path_configured(self):
        self.assertTrue(self._recorder(jsonl_path='')._disabled)

    def test_root_only(self):
        # A child pid never tails a log, even with the right command and path.
        self.assertTrue(self._recorder(pid=101, root_pid=100)._disabled)
        self.assertFalse(self._recorder(pid=100, root_pid=100)._disabled)

    def test_setup_needs_both_pids(self):
        recorder = NInferRecorder(root_pid=None, pid=None, args=['ninfer-serve'],
                                  jsonl_path=self.path, environ={})
        recorder.setup()
        self.assertTrue(recorder._disabled)

    def test_path_from_env_and_flag_reaches_the_recorder(self):
        env_recorder = NInferRecorder(
            root_pid=100, pid=100, args=['/usr/bin/ninfer-serve'],
            environ={'GRAPHSIGNAL_NINFER_JSONL': self.path})
        self.assertEqual(env_recorder.jsonl_path, self.path)
        env_recorder.setup()
        self.assertFalse(env_recorder._disabled)

        flag_recorder = NInferRecorder(
            root_pid=100, pid=100,
            args=['/usr/bin/ninfer-serve', '--request-log-jsonl', self.path],
            environ={})
        self.assertEqual(flag_recorder.jsonl_path, self.path)
        flag_recorder.setup()
        self.assertFalse(flag_recorder._disabled)

    def test_missing_file_is_not_fatal(self):
        recorder = self._recorder(jsonl_path=os.path.join(self.tmpdir, 'nope.jsonl'))
        recorder.on_tick()
        self.assertEqual(self._exported(), [])

    def test_unconfigured_watcher_is_a_noop(self):
        graphsignal.watcher.shutdown()
        recorder = self._recorder()
        self._append(_server_start(), _request_done())
        recorder.on_tick()  # must not raise


class NInferRecorderTailTest(NInferRecorderTestBase):
    def test_only_new_bytes_are_read(self):
        recorder = self._recorder()
        self._append(_server_start(), _request_done())
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

        # A second tick over unchanged bytes re-reads nothing.
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

        self._append(_request_done())
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 2)

    def test_partial_line_is_retained_until_complete(self):
        recorder = self._recorder()
        self._append(_server_start(), _request_done(), trailing_newline=False)
        recorder.on_tick()
        # The unterminated record is held back, not lost and not half-parsed.
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_completed'))

        self._append_raw('\n')
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_split_across_many_ticks(self):
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()
        payload = json.dumps(_request_done())
        for index in range(0, len(payload), 7):
            self._append_raw(payload[index:index + 7])
            recorder.on_tick()
        self._append_raw('\n')
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_backlog_is_carried_across_ticks_not_dropped(self):
        # A bounded read per tick means a backlog spans ticks; the offset
        # carries it, so no record is lost or counted twice.
        recorder = self._recorder()
        self._append(_server_start())
        for _ in range(5):
            self._append(_request_done())
        with patch('graphsignal.recorders.ninfer_recorder.MAX_READ_BYTES_PER_TICK', 64):
            for _ in range(40):
                recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 5)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_request_duration_seconds',
                        metric_type='histogram').datapoint['count'], 5)

    def test_oversized_line_is_dropped_and_the_tail_resyncs(self):
        # A junk line past the per-line cap must not stall the tail, grow
        # without bound, or cost the records around it. The cap is well above a
        # real record (a full `server_start` is a few kB).
        recorder = self._recorder()
        with patch('graphsignal.recorders.ninfer_recorder.MAX_READ_BYTES_PER_TICK', 1024), \
                patch('graphsignal.recorders.ninfer_recorder.MAX_LINE_BYTES', 8192):
            self._append(_server_start())
            self._append_raw('x' * 40000 + '\n')
            self._append(_request_done())
            for _ in range(80):
                recorder.on_tick()
        exported = self._exported()
        # Both real records were read, and the junk line cost neither.
        self.assertEqual(
            find_metric(exported, 'ninfer_server_starts').datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_requests_completed').datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_prompt_tokens_total').datapoint['total'], 1536)

    def test_truncation_resets_cumulative_state(self):
        recorder = self._recorder()
        self._append(_server_start(), _request_done())
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

        # A restarted server cut the file back to the start.
        with open(self.path, 'w') as f:
            f.write(json.dumps(_server_start(instance='serve-999-1')) + '\n')
            f.write(json.dumps(_request_done(instance='serve-999-1')) + '\n')
        recorder.on_tick()
        # Counted once, for the new lifetime only.
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_replaced_file_restarts_from_zero(self):
        recorder = self._recorder()
        self._append(_server_start(), _request_done())
        recorder.on_tick()

        # A different file takes the path (rotation by rename).
        shutil.move(self.path, self.path + '.1')
        self._append(_server_start(instance='serve-999-1'),
                     _request_done(instance='serve-999-1'))
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_new_server_instance_id_resets_counters_and_tags(self):
        recorder = self._recorder()
        self._append(_server_start(), _request_done())
        recorder.on_tick()
        self.assertEqual(self.watcher.get_tag('ninfer.model'), 'qwen3.6-27b')

        # The log is append mode: a second block is a new server lifetime, and
        # its counters must not stack on the first one's.
        second = _server_start(instance='serve-999-1')
        second['server']['public_model_id'] = 'llama3-8b'
        second['engine']['kv_cache'] = 'bf16'
        self._append(second, _request_done(instance='serve-999-1'))
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)
        self.assertEqual(find_metric(self._exported(),
                                     'ninfer_server_starts').datapoint['total'], 1)
        self.assertEqual(self.watcher.get_tag('ninfer.model'), 'llama3-8b')
        self.assertEqual(self.watcher.get_tag('ninfer.kv_cache'), 'bf16')

    def test_server_instance_id_retracts_tags_it_no_longer_publishes(self):
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()
        self.assertEqual(self.watcher.get_tag('ninfer.gpu_name'), 'NVIDIA GB10')

        second = _server_start(instance='serve-999-1')
        second['environment']['gpu_name'] = ''
        self._append(second)
        recorder.on_tick()
        self.assertIsNone(self.watcher.get_tag('ninfer.gpu_name'))

    def test_malformed_and_blank_lines_are_skipped(self):
        recorder = self._recorder()
        self._append(_server_start())
        self._append_raw('not json\n\n{"truncated": \n')
        self._append(_request_done())
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_finalize_drains_the_last_records(self):
        recorder = self._recorder()
        self._append(_server_start())
        self._append(_request_done(), trailing_newline=False)
        recorder.finalize()
        # Still unterminated: finalize does not invent a record.
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_completed'))
        self._append_raw('\n')
        recorder.finalize()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_shutdown_disables_the_recorder(self):
        recorder = self._recorder()
        recorder.shutdown()
        self._append(_server_start(), _request_done())
        recorder.on_tick()
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_completed'))


class NInferRecorderSchemaTest(NInferRecorderTestBase):
    def test_foreign_artifact_type_is_ignored(self):
        recorder = self._recorder()
        foreign = _request_done()
        foreign['artifact_type'] = 'some_other_log'
        self._append(_server_start(), foreign)
        recorder.on_tick()
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_completed'))
        # The good line in the same file still landed.
        self.assertIsNotNone(find_metric(self._exported(), 'ninfer_server_starts'))

    def test_older_schema_version_is_skipped(self):
        recorder = self._recorder()
        old = _request_done()
        old['schema_version'] = SCHEMA_VERSION - 2  # 22, as the bench tool emits
        self._append(old)
        recorder.on_tick()
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_completed'))

    def test_missing_schema_version_is_skipped(self):
        recorder = self._recorder()
        record = _request_done()
        del record['schema_version']
        self._append(record)
        recorder.on_tick()
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_completed'))

    def test_future_schema_version_keeps_known_events(self):
        recorder = self._recorder()
        future = _request_done()
        future['schema_version'] = SCHEMA_VERSION + 3
        self._append(future)
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)

    def test_future_schema_version_ignores_unknown_events(self):
        recorder = self._recorder()
        unknown = _event('kv_cache_snapshot', kv_cache={'pages': 4096})
        self._append(_server_start(), unknown, _request_done())
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 1)
        self.assertIsNone(find_metric(self._exported(), 'kv_cache_snapshot'))

    def test_non_dict_lines_are_skipped(self):
        recorder = self._recorder()
        self._append_raw('[1, 2, 3]\n"a string"\nnull\n')
        recorder.on_tick()  # no exception

    def test_missing_timestamp_falls_back_to_now(self):
        recorder = self._recorder()
        record = _request_done()
        del record['timestamp_unix_ms']
        self._append(record)
        recorder.on_tick()
        metric = find_metric(self._exported(), 'ninfer_requests_completed')
        self.assertGreater(metric.datapoint['ts'], time.time_ns() - 60e9)

    def test_measurement_timestamp_is_the_event_time_in_ns(self):
        recorder = self._recorder()
        self._append(_server_start(ts_ms=1_712_345_678_901),
                     _request_done(ts_ms=1_712_345_679_555))
        recorder.on_tick()
        expected = 1_712_345_679_555 * 1_000_000
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['ts'], expected)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_request_duration_seconds',
                        metric_type='histogram').datapoint['ts'], expected)


class NInferRecorderServerStartTest(NInferRecorderTestBase):
    def _resource(self):
        return next(r for r in self.watcher.resource_store().export()
                    if r['kind'] == 'ninfer_server')

    def test_publishes_bounded_tags_from_the_v24_nesting(self):
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()

        self.assertEqual(self.watcher.get_tag('ninfer.model'), 'qwen3.6-27b')
        self.assertEqual(self.watcher.get_tag('ninfer.host'), '0.0.0.0')
        self.assertEqual(self.watcher.get_tag('ninfer.port'), '8080')
        self.assertEqual(self.watcher.get_tag('ninfer.architecture'), 'qwen3_moe')
        self.assertEqual(self.watcher.get_tag('ninfer.kv_cache'), 'nvfp4')
        self.assertEqual(self.watcher.get_tag('ninfer.speculative_backend'), 'mtp')
        self.assertEqual(self.watcher.get_tag('ninfer.ngram_residency'), 'mtp3')
        self.assertEqual(self.watcher.get_tag('ninfer.kv_capacity_mode'), 'auto')
        self.assertEqual(self.watcher.get_tag('ninfer.gpu_name'), 'NVIDIA GB10')
        self.assertEqual(self.watcher.get_tag('ninfer.compute_capability'), '12.1')

    def test_resource_carries_the_startup_picture(self):
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()

        resource = self._resource()
        self.assertEqual(resource['tags']['process.pid'], '100')
        self.assertEqual(resource['tags']['process.root_pid'], '100')
        attributes = resource['attributes']
        self.assertEqual(attributes['model'], 'qwen3.6-27b')
        self.assertEqual(attributes['artifact_path'],
                         '/models/qwen3.6-27b.ninfer')
        self.assertEqual(attributes['artifact_size_bytes'], '27487790694')
        self.assertEqual(attributes['max_context'], '131072')
        self.assertEqual(attributes['max_concurrency'], '64')
        self.assertEqual(attributes['cuda_graph'], 'true')
        self.assertEqual(attributes['context_cache_enabled'], 'true')
        self.assertEqual(attributes['weights_used_bytes'], '27487790694')
        self.assertEqual(attributes['gpu_uuid'],
                         'GPU-9d2c1f4e-0000-0000-0000-000000000000')
        self.assertEqual(attributes['load_seconds'], '41.5')
        self.assertEqual(attributes['server_instance_id'], _INSTANCE)
        self.assertEqual(resource['first_seen_ts'], _TS_NS)

    def test_argv_is_never_recorded(self):
        # argv can carry credentials, and the command line is already recorded
        # by ProcessRecorder.
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()
        serialized = str(self._resource()['attributes'])
        self.assertNotIn('argv', serialized)
        self.assertNotIn('request-log-jsonl', serialized)

    def test_startup_memory_gauges(self):
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()

        exported = self._exported()
        weights = find_metric(exported, 'ninfer_memory_weights_used_bytes',
                              metric_type='gauge')
        self.assertEqual(weights.datapoint['value'], 27487790694.0)
        self.assertEqual(weights.datapoint['ts'], _TS_NS)
        self.assertEqual(
            find_metric(exported, 'ninfer_memory_kv_capacity_bytes',
                        metric_type='gauge').datapoint['value'], 4294967296.0)
        self.assertEqual(
            find_metric(exported, 'ninfer_model_load_seconds',
                        metric_type='gauge').datapoint['value'], 41.5)
        self.assertEqual(
            find_metric(exported, 'ninfer_memory_host_state_capacity_slots',
                        metric_type='gauge').datapoint['value'], 64.0)

    def test_counts_server_starts(self):
        recorder = self._recorder()
        self._append(_server_start(), _server_start(ts_ms=_TS_MS + 10))
        recorder.on_tick()
        metric = find_metric(self._exported(), 'ninfer_server_starts')
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 2)
        self.assertEqual(metric.datapoint['ts'], (_TS_MS + 10) * 1_000_000)

    def test_startup_does_not_leak_per_request_or_instance_values(self):
        recorder = self._recorder()
        self._append(_server_start())
        recorder.on_tick()
        tags = {k: v for k, v in self.watcher.tags().items()
                if k.startswith('ninfer.')}
        self.assertEqual(sorted(tags), [
            'ninfer.architecture', 'ninfer.compute_capability', 'ninfer.gpu_name',
            'ninfer.host', 'ninfer.kv_cache', 'ninfer.kv_capacity_mode',
            'ninfer.model', 'ninfer.ngram_residency', 'ninfer.port',
            'ninfer.speculative_backend'])
        self.assertNotIn(_INSTANCE, str(self.watcher.tags()))


class NInferRecorderRequestStartTest(NInferRecorderTestBase):
    def test_preparation_histograms(self):
        recorder = self._recorder()
        self._append(_request_start())
        recorder.on_tick()

        exported = self._exported()
        total = find_metric(exported, 'ninfer_request_preparation_seconds',
                            metric_type='histogram')
        self.assertEqual(total.datapoint['count'], 1)
        self.assertAlmostEqual(total.datapoint['sum'], 0.0125, places=6)
        self.assertAlmostEqual(total.datapoint['min'], 0.0125, places=6)
        # 0.0125 lands in the 0.025s request bucket.
        self.assertEqual(total.datapoint['bins'], [0.025])

        tokenize = find_metric(exported, 'ninfer_request_preparation_tokenize_seconds',
                               metric_type='histogram')
        self.assertAlmostEqual(tokenize.datapoint['sum'], 0.0031, places=6)
        self.assertEqual(
            find_metric(exported, 'ninfer_request_preparation_acquisition_seconds',
                        metric_type='histogram').datapoint['count'], 1)

    def test_counts_requests_started(self):
        recorder = self._recorder()
        self._append(_request_start(), _request_start(ts_ms=_TS_MS + 5))
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_started_total').datapoint['total'], 2)

    def test_media_and_vision_counters(self):
        recorder = self._recorder()
        self._append(_request_start_with_media())
        self._append(_request_start_with_media(ts_ms=_TS_MS + 5))
        recorder.on_tick()

        expected = {
            'ninfer_media_items_total': 4,
            'ninfer_media_bytes_total': 8388608,
            'ninfer_raw_patches_total': 3072,
            'ninfer_vision_tokens_total': 24576,
            'ninfer_patch_bytes_total': 16777216,
            'ninfer_media_preprocess_built_patch_bytes_total': 8388608,
            'ninfer_media_preprocess_reused_patch_bytes_total': 8388608,
            'ninfer_media_cache_hits_total': 2,
            'ninfer_media_cache_misses_total': 2,
            'ninfer_media_singleflight_waits_total': None,
        }
        for name, total in expected.items():
            metric = find_metric(self._exported(), name)
            if total is None:
                # Zero is not a measurement: the series is never created.
                self.assertIsNone(metric, name)
            else:
                self.assertEqual(metric.datapoint['total'], total, name)

    def test_media_preprocess_histogram(self):
        recorder = self._recorder()
        self._append(_request_start_with_media())
        recorder.on_tick()
        metric = find_metric(self._exported(),
                             'ninfer_request_preparation_media_preprocess_seconds',
                             metric_type='histogram')
        self.assertAlmostEqual(metric.datapoint['sum'], 0.31, places=6)


class NInferRecorderRequestDoneTest(NInferRecorderTestBase):
    def _histogram(self, name='ninfer_request_duration_seconds'):
        return find_metric(self._exported(), name, metric_type='histogram')

    def test_latency_histograms_from_timings_seconds(self):
        recorder = self._recorder()
        self._append(_request_done())
        recorder.on_tick()

        expected = {
            'ninfer_request_duration_seconds': 1.256,
            'ninfer_request_prepare_seconds': 0.0125,
            'ninfer_request_ttft_seconds': 0.184,
            'ninfer_request_prefill_seconds': 0.121,
            'ninfer_request_decode_seconds': 1.06,
        }
        for name, total in expected.items():
            metric = self._histogram(name)
            self.assertIsNotNone(metric, name)
            self.assertAlmostEqual(metric.datapoint['sum'], total, places=6, msg=name)
            self.assertAlmostEqual(metric.datapoint['min'], total, places=6, msg=name)
            self.assertAlmostEqual(metric.datapoint['max'], total, places=6, msg=name)

    def test_zero_duration_phase_is_not_measured(self):
        # timings_seconds.vision is 0.0 when the engine skipped the phase.
        recorder = self._recorder()
        self._append(_request_done())
        recorder.on_tick()
        self.assertIsNone(self._histogram('ninfer_request_vision_seconds'))

    def test_engine_timing_histograms_use_the_fine_bucket_set(self):
        recorder = self._recorder()
        self._append(_request_done())
        recorder.on_tick()

        expected = {
            'ninfer_request_queue_wait_seconds': 0.0043,
            'ninfer_request_host_exposed_seconds': 0.0065,
            'ninfer_request_device_wait_seconds': 0.0412,
            'ninfer_request_decode_host_seconds': 0.0044,
            'ninfer_request_decode_device_wait_seconds': 0.0331,
        }
        for name, total in expected.items():
            metric = self._histogram(name)
            self.assertIsNotNone(metric, name)
            self.assertAlmostEqual(metric.datapoint['sum'], total, places=6, msg=name)

        # 4.3ms is below the request-level 5ms floor but has an engine bucket.
        queue_wait = self._histogram('ninfer_request_queue_wait_seconds')
        self.assertEqual(queue_wait.datapoint['bins'], [0.005])
        self.assertEqual(queue_wait.datapoint['counts'], [1])
        self.assertIsNone(self._histogram('ninfer_request_duration_seconds')
                          and None)
        self.assertEqual(self._histogram('ninfer_request_duration_seconds')
                         .datapoint['bins'], [2.5])

    def test_histogram_accumulates_across_requests(self):
        recorder = self._recorder()
        self._append(_request_done())
        self._append(_request_done(ts_ms=_TS_MS + 100))
        self._append(_request_done(ts_ms=_TS_MS + 200))
        recorder.on_tick()

        metric = self._histogram()
        self.assertEqual(metric.datapoint['count'], 3)
        self.assertAlmostEqual(metric.datapoint['sum'], 3.768, places=5)
        self.assertEqual(metric.datapoint['bins'], [2.5])
        self.assertEqual(metric.datapoint['counts'], [3])

    def test_token_counters(self):
        recorder = self._recorder()
        self._append(_request_done())
        self._append(_request_done(ts_ms=_TS_MS + 100))
        recorder.on_tick()

        expected = {
            'ninfer_prompt_tokens_total': 3072,
            'ninfer_completion_tokens_total': 256,
            'ninfer_computed_prefill_tokens_total': 2560,
            'ninfer_prefix_cache_hit_tokens_total': 512,
            'ninfer_model_thinking_tokens_total': 128,
            'ninfer_thinking_control_tokens_total': 16,
        }
        for name, total in expected.items():
            self.assertEqual(
                find_metric(self._exported(), name).datapoint['total'], total, name)

    def test_finish_reason_and_protocol_series(self):
        recorder = self._recorder()
        self._append(_request_done())
        self._append(_with(_request_done(ts_ms=_TS_MS + 100),
                           result={**_request_done()['result'],
                                   'finish_reason': 'output_limit'}))
        self._append(_with(_request_done(ts_ms=_TS_MS + 200),
                           request=_request(protocol='anthropic_messages')))
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_completed').datapoint['total'], 3)
        # The third record keeps the default result, so stop_token is two.
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_completed_by_finish_reason',
                        {'finish_reason': 'stop_token'}).datapoint['total'], 2)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_completed_by_finish_reason',
                        {'finish_reason': 'output_limit'}).datapoint['total'], 1)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_completed_by_protocol',
                        {'protocol': 'openai_chat_completions'}).datapoint['total'],
            2)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_completed_by_protocol',
                        {'protocol': 'anthropic_messages'}).datapoint['total'], 1)

    def test_speculative_counters_and_gauge(self):
        recorder = self._recorder()
        self._append(_request_done())
        self._append(_request_done(ts_ms=_TS_MS + 100))
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_speculative_rounds_total').datapoint['total'], 256)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_speculative_drafted_tokens_total'
                        ).datapoint['total'], 768)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_speculative_accepted_tokens_total'
                        ).datapoint['total'], 512)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_speculative_fallback_steps_total'
                        ).datapoint['total'], 4)
        gauge = find_metric(self._exported(), 'ninfer_speculative_accepted_per_position',
                            metric_type='gauge')
        self.assertAlmostEqual(gauge.datapoint['value'], 0.666, places=6)

    def test_engine_units_and_boolean_flags(self):
        recorder = self._recorder()
        self._append(_request_done())
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_engine_prefill_units_total').datapoint['total'], 2)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_engine_control_units_total').datapoint['total'], 1)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_requests_thinking_total').datapoint['total'], 1)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_thinking_control_applied_total'
                        ).datapoint['total'], 1)
        # stream: false and the tool-call diagnostics are all zero/false.
        self.assertIsNone(find_metric(self._exported(), 'ninfer_requests_streamed_total'))
        self.assertIsNone(find_metric(self._exported(), 'ninfer_tool_calls_total'))

    def test_materialization_gauges_and_counters(self):
        recorder = self._recorder()
        self._append(_request_done())
        recorder.on_tick()

        exported = self._exported()
        # Nanoseconds in the artifact, seconds in the store.
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_materialization_planning_seconds',
                        metric_type='gauge').datapoint['value'], 0.00045, places=8)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_materialization_search_seconds',
                        metric_type='gauge').datapoint['value'], 0.0018, places=8)
        self.assertEqual(
            find_metric(exported, 'ninfer_materialization_search_renewals_total'
                        ).datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported,
                        'ninfer_materialization_incumbent_improvements_total'
                        ).datapoint['total'], 2)
        self.assertEqual(
            find_metric(exported, 'ninfer_materialization_search_work_total'
                        ).datapoint['total'], 512)
        # budget_exhausted and selected_maximal_fallback are both false.
        self.assertIsNone(find_metric(
            exported, 'ninfer_materialization_budget_exhausted_total'))
        self.assertIsNone(find_metric(
            exported, 'ninfer_materialization_maximal_fallback_total'))

    def test_tool_call_parse_counters(self):
        recorder = self._recorder()
        record = _request_done()
        record['result']['tool_call_count'] = 2
        record['result']['tool_call_parse'] = {
            'marker_seen': True, 'structured_call_count': 2,
            'empty_arguments_omitted': 1, 'schema_mismatch_arguments': 1,
            'duplicate_parameters_repaired': 3, 'fallback_reason': 'schema_mismatch'}
        self._append(record)
        recorder.on_tick()

        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_tool_calls_total').datapoint['total'], 2)
        self.assertEqual(
            find_metric(exported, 'ninfer_tool_call_parse_structured_total'
                        ).datapoint['total'], 2)
        self.assertEqual(
            find_metric(exported,
                        'ninfer_tool_call_parse_empty_arguments_omitted_total'
                        ).datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_tool_call_parse_schema_mismatch_total'
                        ).datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported,
                        'ninfer_tool_call_parse_duplicate_parameters_repaired_total'
                        ).datapoint['total'], 3)
        self.assertEqual(
            find_metric(exported, 'ninfer_tool_call_parse_fallback_total',
                        {'fallback_reason': 'schema_mismatch'}).datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_tool_call_parse_marker_seen_total'
                        ).datapoint['total'], 1)

    def test_no_request_id_tag_anywhere(self):
        recorder = self._recorder()
        self._append(_request_done())
        recorder.on_tick()
        for metric in self._exported():
            self.assertNotIn('request_id', str(metric.tags))
            self.assertNotIn('1', metric.tags.get('tag', 'x'))
        self.assertNotIn('request_id', str(self.watcher.tags()))


class NInferRecorderErrorTest(NInferRecorderTestBase):
    def _entries(self):
        """NInfer's own log entries — the store also holds the watcher's."""
        return [e for e in self.watcher.log_store().export()
                if e['tags'].get('scope.name') == 'workload']

    def test_request_error_becomes_a_counter_and_an_error_entry(self):
        recorder = self._recorder()
        self._append(_request_error())
        recorder.on_tick()

        metric = find_metric(self._exported(), 'ninfer_request_errors')
        self.assertEqual(metric.type, 'counter')
        # `request_error` carries only a message, so the counter is untagged.
        self.assertEqual(metric.tags, {})
        self.assertEqual(metric.datapoint['total'], 1)

        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['level'], 'error')
        self.assertEqual(entries[0]['ts'], _TS_NS)
        self.assertIn('illegal memory access', entries[0]['message'])
        self.assertEqual(entries[0]['tags']['event'], 'request_error')
        self.assertEqual(entries[0]['tags']['scope.name'], 'workload')
        self.assertEqual(entries[0]['tags']['process.pid'], '100')

    def test_error_without_message_still_records(self):
        recorder = self._recorder()
        self._append(_event('request_error', request=_request(), error={}))
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_request_errors'
                        ).datapoint['total'], 1)
        self.assertEqual(len(self._entries()), 1)

    def test_rejection_is_a_warning_with_structured_counters(self):
        recorder = self._recorder()
        self._append(_request_rejected())
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_rejected',
                        {'reason': 'context_length_exceeded'}
                        ).datapoint['total'], 1)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_rejected_by_status',
                        {'status': '400'}).datapoint['total'], 1)
        self.assertIsNone(find_metric(self._exported(), 'ninfer_request_errors'))

        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['level'], 'warning')
        self.assertIn('context_length_exceeded', entries[0]['message'])
        self.assertIn('prepare', entries[0]['message'])

    def test_rejection_falls_back_to_the_error_type(self):
        recorder = self._recorder()
        self._append(_request_rejected(code=None, status='503',
                                       error_type='overloaded_error',
                                       message='server is busy'))
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_requests_rejected',
                        {'reason': 'overloaded_error'}).datapoint['total'], 1)

    def test_error_accumulates(self):
        recorder = self._recorder()
        for index in range(4):
            self._append(_request_error(f'failure {index}'))
        recorder.on_tick()
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_request_errors').datapoint['total'], 4)

    def test_long_message_is_truncated_not_dropped(self):
        recorder = self._recorder()
        self._append(_request_error('x' * 5000))
        recorder.on_tick()
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0]['message']), 1024)

    def test_rejection_reason_cardinality_is_capped(self):
        recorder = self._recorder()
        for index in range(MAX_TAG_VALUES + 10):
            self._append(_request_rejected(code=f'code_{index}'))
        recorder.on_tick()

        series = [m for m in self._exported() if m.name == 'ninfer_requests_rejected']
        self.assertLessEqual(len(series), MAX_TAG_VALUES + 1)  # + 'other'
        self.assertEqual(sum(m.datapoint['total'] for m in series),
                         MAX_TAG_VALUES + 10)

    def test_log_entries_are_rate_capped_per_tick(self):
        recorder = self._recorder()
        for index in range(100):
            self._append(_request_error(f'failure {index}'))
        recorder.on_tick()
        # The counter saw all of them...
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_request_errors').datapoint['total'], 100)
        # ...but the log ring is not flooded in one tick.
        self.assertLessEqual(len(self._entries()), 20)

        # The next tick starts a fresh budget.
        for index in range(100, 110):
            self._append(_request_error(f'failure {index}'))
        recorder.on_tick()
        self.assertLessEqual(len(self._entries()), 40)


class NInferRecorderThroughputTest(NInferRecorderTestBase):
    def test_token_deltas_become_cumulative_counters(self):
        recorder = self._recorder()
        self._append(_throughput())
        self._append(_throughput(ts_ms=_TS_MS + 2000,
                                 tokens={'computed_prefill': 512,
                                         'committed_decode': 128}))
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(), 'ninfer_throughput_prefill_tokens_total'
                        ).datapoint['total'], 2048)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_throughput_decode_tokens_total'
                        ).datapoint['total'], 512)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_decode_rounds_total').datapoint['total'], 768)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_decode_row_rounds_total').datapoint['total'], 4608)

    def test_scheduler_gauges_are_current_not_cumulative(self):
        recorder = self._recorder()
        self._append(_throughput())
        self._append(_throughput(ts_ms=_TS_MS + 2000,
                                 scheduler={'running': 2, 'prefilling': 0,
                                            'decode_ready': 2, 'waiting': 0,
                                            'materializing': 0, 'capture_pending': 0,
                                            'terminal_pending': 1}))
        recorder.on_tick()

        expected = {
            'ninfer_requests_running': 2.0,
            'ninfer_requests_prefilling': 0.0,
            'ninfer_requests_decode_ready': 2.0,
            'ninfer_requests_waiting': 0.0,
            'ninfer_requests_materializing': 0.0,
            'ninfer_requests_capture_pending': 0.0,
            'ninfer_requests_terminal_pending': 1.0,
        }
        for name, value in expected.items():
            metric = find_metric(self._exported(), name, metric_type='gauge')
            self.assertIsNotNone(metric, name)
            self.assertEqual(metric.datapoint['value'], value, name)

    def test_engine_reported_rates_and_batch_size(self):
        recorder = self._recorder()
        self._append(_throughput())
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(), 'ninfer_prefill_tokens_per_second',
                        metric_type='gauge').datapoint['value'], 1536.0)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_decode_tokens_per_second',
                        metric_type='gauge').datapoint['value'], 384.0)
        self.assertEqual(
            find_metric(self._exported(), 'ninfer_decode_batch_average_size',
                        metric_type='gauge').datapoint['value'], 6.0)
        self.assertAlmostEqual(
            find_metric(self._exported(),
                        'ninfer_decode_host_microseconds_per_round',
                        metric_type='gauge').datapoint['value'], 18.5, places=6)
        self.assertAlmostEqual(
            find_metric(self._exported(),
                        'ninfer_decode_host_microseconds_per_row_round',
                        metric_type='gauge').datapoint['value'], 3.08, places=6)
        self.assertAlmostEqual(
            find_metric(self._exported(),
                        'ninfer_decode_device_wait_microseconds_per_round',
                        metric_type='gauge').datapoint['value'], 291.9, places=6)

    def test_host_work_cumulative_seconds_and_rates(self):
        recorder = self._recorder()
        self._append(_throughput())
        self._append(_throughput(ts_ms=_TS_MS + 2000,
                                 host_work={
                                     'elapsed_seconds': {
                                         'engine_boundary': 0.002, 'program_submit': 0.0,
                                         'program_post': 0.0, 'engine_commit_output': 0.0,
                                         'engine_maintenance': 0.0, 'total': 0.002},
                                     'device_wait_seconds': 0.1,
                                     'work_class_seconds': {}, 'detail_subset_seconds': {},
                                     'detail_invocations': {}, 'units': {},
                                     'decode_host_microseconds_per_round': None,
                                     'decode_host_microseconds_per_row_round': None,
                                     'decode_device_wait_microseconds_per_round': None,
                                     'detail_microseconds_per_invocation': {}}))
        recorder.on_tick()

        exported = self._exported()
        # engine_boundary: 0.0041s over the first interval + 0.002s over the
        # second.
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_total',
                        {'work': 'engine_boundary'}).datapoint['total'],
            0.0061, places=6)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_total',
                        {'work': 'total'}).datapoint['total'],
            0.0244, places=6)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_total',
                        {'work': 'device_wait'}).datapoint['total'],
            0.2536, places=6)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_total',
                        {'work': 'decode_host'}).datapoint['total'],
            0.0071, places=6)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_total',
                        {'work': 'admission_policy'}).datapoint['total'],
            0.0012, places=6)

        # Rates: the second interval's device wait of 0.1s over 1s.
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_per_second',
                        {'work': 'engine_boundary'},
                        metric_type='gauge').datapoint['value'], 0.002, places=6)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_host_work_seconds_per_second',
                        {'work': 'device_wait'},
                        metric_type='gauge').datapoint['value'], 0.1, places=6)

    def test_host_work_units_and_invocations(self):
        recorder = self._recorder()
        self._append(_throughput())
        recorder.on_tick()

        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_host_work_prefill_units_total'
                        ).datapoint['total'], 6)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_host_work_control_units_total'
                        ).datapoint['total'], 128)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_host_work_invocations_total',
                        {'detail': 'admission_policy'}).datapoint['total'], 384)
        self.assertAlmostEqual(
            find_metric(self._exported(), 'ninfer_host_work_microseconds_per_invocation',
                        {'detail': 'admission_policy'},
                        metric_type='gauge').datapoint['value'], 3.12, places=6)

    def test_context_cache_counters(self):
        recorder = self._recorder()
        self._append(_throughput())
        self._append(_throughput(ts_ms=_TS_MS + 2000))
        recorder.on_tick()

        exported = self._exported()
        for name, total in (
                ('ninfer_context_cache_captures_completed_total', 4),
                ('ninfer_context_cache_captures_skipped_total', 2),
                ('ninfer_context_cache_salvaged_total', 2),
                ('ninfer_context_cache_reused_prompt_tokens_total', 2560),
                ('ninfer_context_cache_state_moves_total', 24),
                ('ninfer_context_cache_state_forks_total', 6),
                ('ninfer_context_cache_state_restores_total', 2),
                ('ninfer_context_cache_spill_pages_total', 12),
                ('ninfer_context_cache_pressure_searches_total', 6),
                ('ninfer_context_cache_historical_fork_hits_total', 10),
                ('ninfer_context_cache_transfer_seconds_total', 0.028)):
            metric = find_metric(exported, name)
            self.assertIsNotNone(metric, name)
            self.assertAlmostEqual(metric.datapoint['total'], total, places=6,
                                   msg=name)

        # Zero deltas create no series.
        self.assertIsNone(find_metric(
            exported, 'ninfer_context_cache_captures_aborted_total'))
        self.assertIsNone(find_metric(
            exported, 'ninfer_context_cache_checkpoints_dropped_total'))

    def test_context_cache_tagged_counters(self):
        recorder = self._recorder()
        self._append(_throughput())
        self._append(_throughput(ts_ms=_TS_MS + 2000))
        recorder.on_tick()

        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_selections_total',
                        {'path': 'shared_stable_prefix'}).datapoint['total'], 22)
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_state_transfers_total',
                        {'direction': 'h2d'}).datapoint['total'], 6)
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_state_transfer_bytes_total',
                        {'direction': 'h2d'}).datapoint['total'], 16777216)
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_main_kv_pages_total',
                        {'direction': 'd2h'}).datapoint['total'], 16)
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_backend_kv_pages_total',
                        {'direction': 'd2d'}).datapoint['total'], 2)
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_owners_degraded_total',
                        {'owner': 'shared'}).datapoint['total'], 4)
        self.assertEqual(
            find_metric(exported, 'ninfer_context_cache_owners_evicted_total',
                        {'owner': 'shared'}).datapoint['total'], 2)
        # A zero delta is not a series.
        self.assertIsNone(find_metric(
            exported, 'ninfer_context_cache_owners_evicted_total',
            {'owner': 'private'}))

    def test_context_cache_occupancy_gauges(self):
        recorder = self._recorder()
        self._append(_throughput())
        recorder.on_tick()

        exported = self._exported()
        expected = {
            'ninfer_context_cache_occupancy_device_state_slots': 4.0,
            'ninfer_context_cache_occupancy_host_state_slots': 8.0,
            'ninfer_context_cache_occupancy_device_main_kv_pages': 2048.0,
            'ninfer_context_cache_occupancy_device_main_kv_lease_pages': 2100.0,
            'ninfer_context_cache_occupancy_device_backend_kv_pages': 128.0,
            'ninfer_context_cache_occupancy_device_backend_kv_lease_pages': 130.0,
            'ninfer_context_cache_occupancy_host_kv_bytes': 1073741824.0,
            'ninfer_context_cache_occupancy_shared_active_references': 3.0,
            'ninfer_context_cache_frontier_tokens': 4096.0,
        }
        for name, value in expected.items():
            metric = find_metric(exported, name, metric_type='gauge')
            self.assertIsNotNone(metric, name)
            self.assertEqual(metric.datapoint['value'], value, name)

    def test_missing_interval_publishes_no_host_work_rates(self):
        recorder = self._recorder()
        record = _throughput()
        del record['interval_seconds']
        self._append(record)
        recorder.on_tick()

        # The cumulative seconds stand on their own...
        self.assertAlmostEqual(
            find_metric(self._exported(), 'ninfer_host_work_seconds_total',
                        {'work': 'total'}).datapoint['total'], 0.0224, places=6)
        # ...but a rate without a window would be a guess.
        self.assertIsNone(find_metric(
            self._exported(), 'ninfer_host_work_seconds_per_second'))

    def test_nullable_average_size_is_skipped(self):
        recorder = self._recorder()
        record = _throughput()
        record['decode_batch'] = {'rounds': 0, 'row_rounds': 0, 'average_size': None}
        self._append(record)
        recorder.on_tick()
        self.assertIsNone(find_metric(
            self._exported(), 'ninfer_decode_batch_average_size'))

    def test_throughput_tokens_stay_separate_from_request_token_totals(self):
        recorder = self._recorder()
        self._append(_request_done(), _throughput())
        recorder.on_tick()
        # The same work, counted once each by the source that measured it.
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_prompt_tokens_total').datapoint['total'], 1536)
        self.assertEqual(
            find_metric(self._exported(),
                        'ninfer_throughput_prefill_tokens_total'
                        ).datapoint['total'], 1536)


class NInferRecorderFullFixtureTest(NInferRecorderTestBase):
    def test_a_run_from_start_to_finish(self):
        recorder = self._recorder()
        self._append(_server_start())
        self._append(_request_start())
        self._append(_request_start_with_media(ts_ms=_TS_MS + 10))
        self._append(_request_done(ts_ms=_TS_MS + 100))
        recorder.on_tick()

        self._append(_throughput(ts_ms=_TS_MS + 1000))
        recorder.on_tick()

        self._append(_request_rejected(ts_ms=_TS_MS + 1200))
        self._append(_request_error(ts_ms=_TS_MS + 1300))
        recorder.on_tick()

        self._append(_with(_request_done(ts_ms=_TS_MS + 2000),
                           result={**_request_done()['result'],
                                   'finish_reason': 'output_limit'}))
        recorder.finalize()

        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_server_starts').datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_requests_started_total'
                        ).datapoint['total'], 2)
        self.assertEqual(
            find_metric(exported, 'ninfer_requests_completed'
                        ).datapoint['total'], 2)
        self.assertEqual(
            find_metric(exported, 'ninfer_requests_rejected',
                        {'reason': 'context_length_exceeded'}).datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_request_errors').datapoint['total'], 1)
        self.assertEqual(
            find_metric(exported, 'ninfer_throughput_decode_tokens_total'
                        ).datapoint['total'], 384)
        self.assertEqual(
            find_metric(exported, 'ninfer_prompt_tokens_total'
                        ).datapoint['total'], 3072)
        self.assertEqual(
            find_metric(exported, 'ninfer_completion_tokens_total'
                        ).datapoint['total'], 256)
        self.assertEqual(
            find_metric(exported, 'ninfer_vision_tokens_total'
                        ).datapoint['total'], 12288)

        histogram = find_metric(exported, 'ninfer_request_duration_seconds',
                                metric_type='histogram')
        self.assertEqual(histogram.datapoint['count'], 2)
        self.assertAlmostEqual(histogram.datapoint['sum'], 2.512, places=5)
        self.assertEqual(histogram.datapoint['ts'], (_TS_MS + 2000) * 1_000_000)

        levels = [e['level'] for e in self.watcher.log_store().export()
                  if e['tags'].get('scope.name') == 'workload']
        self.assertEqual(levels, ['warning', 'error'])

        self.assertEqual(self.watcher.get_tag('ninfer.model'), 'qwen3.6-27b')
        resource = next(r for r in self.watcher.resource_store().export()
                        if r['kind'] == 'ninfer_server')
        self.assertEqual(resource['attributes']['port'], '8080')
