"""Fixture-driven tests for the ninfer_bench report recorder.

The fixtures mirror the JSON `bench/inference` actually writes (report schema
15, `artifact_type: ninfer_bench_report`): the `environment`/`artifact`/`load`/
`memory`/`config` blocks built from the engine's own summaries, and `tests`
with per-repetition `timings` and `speculative` blocks.
"""

import json
import os
import shutil
import tempfile
import unittest

import graphsignal.watcher
from graphsignal.recorders.ninfer_bench_recorder import (
    ARTIFACT_TYPE, SCHEMA_VERSION, NinferBenchRecorder, command_basename,
    is_ninfer_bench, parse_report_path)
from graphsignal.signals.routes import build_payload
from test.test_utils import configure_test_watcher, find_metric

_TS_NS = 1_712_345_678_901_000_000


def _report(tests=None, **overrides):
    report = {
        'schema_version': SCHEMA_VERSION,
        'artifact_type': ARTIFACT_TYPE,
        'tool': 'ninfer_bench',
        'command': 'build/bench/ninfer_bench --weights m.ninfer -o json',
        'environment': {'gpu_name': 'NVIDIA GB10', 'device_id': 0,
                        'cuda_runtime_version': '13.0',
                        'cuda_driver_version': '580.65'},
        'artifact': {'path': 'models/qwen3_8.ninfer', 'file_size_bytes': 5306779136},
        'load': {'architecture': 'qwen3_moe', 'load_seconds': 41.5,
                 'upload_seconds': 0.79},
        'memory': {
            'max_context': 4096, 'kv_capacity': 4096, 'kv_cache': 'nvfp4',
            'kv_capacity_mode': 'explicit', 'kv_payload_bytes': 123683840,
            'weights': {'used_bytes': 5306779136, 'capacity_bytes': 5306779136},
            'sequence': {'used_bytes': 1024, 'capacity_bytes': 2048},
            'workspace': {'used_bytes': 0, 'capacity_bytes': 889044992},
        },
        'config': {'max_context': 4096, 'speculative_backend': 'mtp',
                   'draft_tokens': 7, 'use_cuda_graph': True,
                   'repetitions': 3, 'corpus_tokens': 65536},
        'tests': tests if tests is not None else [_test()],
    }
    report.update(overrides)
    return report


def _rep(prefill, decode, total, prepare=1e-5, vision=0.0,
         drafted=0, accepted=0, rounds=0, fallback=0):
    return {
        'generated_output_tokens': 128,
        'decode_output_tokens': 128 if decode else None,
        'decode_engine_tokens': 128 if decode else None,
        'timings': {'prepare_seconds': prepare, 'vision_seconds': vision,
                    'prefill_seconds': prefill, 'decode_seconds': decode,
                    'total_seconds': total},
        'speculative': {'backend': 'mtp', 'enabled': True, 'draft_window': 7,
                        'rounds': rounds, 'drafted_tokens': drafted,
                        'accepted_tokens': accepted,
                        'fallback_steps': fallback,
                        'acceptance_rate': None, 'acceptance_length': None,
                        'accepted_per_position': []},
    }


def _test(label='pp2048+tg128', kind='pp+tg', n_prompt=2048, n_gen=128,
          reps=None, speculative=None, **extra):
    test = {
        'label': label,
        'kind': kind,
        'n_prompt': n_prompt,
        'n_gen': n_gen,
        'requested_output_tokens': n_gen,
        'prefill_tok_s_mean': 1772.82,
        'prefill_tok_s_stddev': 1.5,
        'decode_output_tok_s_mean': 101.5,
        'decode_output_tok_s_stddev': 0.2,
        'decode_engine_tok_s_mean': 101.5,
        'decode_engine_tok_s_stddev': 0.2,
        'prepare_seconds_mean': 1e-5,
        'prefill_seconds_mean': 1.15,
        'decode_seconds_mean': 1.26,
        'total_seconds_mean': 2.41,
        'workspace_peak_bytes': 889044992,
        'workspace_allocator_peak_bytes': 889044992,
        'speculative': speculative if speculative is not None else {
            'backend': 'mtp', 'enabled': True, 'draft_window': 7, 'rounds': 54,
            'drafted_tokens': 330, 'accepted_tokens': 330, 'fallback_steps': 0,
            'acceptance_rate': 1.0, 'acceptance_length': 7.11,
            'accepted_per_position': []},
        'reps': reps if reps is not None else [
            _rep(1.152, 1.258, 2.411, drafted=110, accepted=110, rounds=18),
            _rep(1.155, 1.261, 2.417, drafted=110, accepted=110, rounds=18),
            _rep(1.157, 1.265, 2.423, drafted=110, accepted=110, rounds=18),
        ],
    }
    test.update(extra)
    return test


class CommandParsingTest(unittest.TestCase):
    def test_command_basename(self):
        self.assertEqual(command_basename(['/a/b/ninfer_bench']), 'ninfer_bench')
        self.assertEqual(command_basename('build/bench/ninfer_bench -r 3'),
                         'ninfer_bench')
        # The platform executable suffix is stripped, not treated as the name.
        self.assertEqual(command_basename(['/opt/ninfer_bench.exe']),
                         'ninfer_bench.exe')
        self.assertIsNone(command_basename(None))
        self.assertIsNone(command_basename([]))

    def test_matches_both_spellings(self):
        self.assertTrue(is_ninfer_bench(['/usr/bin/ninfer_bench']))
        self.assertTrue(is_ninfer_bench(['/usr/bin/ninfer-bench']))
        self.assertTrue(is_ninfer_bench(['ninfer_bench.exe']))
        self.assertFalse(is_ninfer_bench(['/usr/bin/ninfer-serve']))
        self.assertFalse(is_ninfer_bench(['python', 'app.py']))
        self.assertFalse(is_ninfer_bench(None))

    def test_parses_report_path(self):
        self.assertEqual(
            parse_report_path(['ninfer_bench', '-o', 'json',
                               '--output-file', 'out.json']),
            ('out.json', True))
        self.assertEqual(
            parse_report_path(['ninfer_bench', '--output=json',
                               '--output-file=out.json']),
            ('out.json', True))

    def test_reports_non_json_and_missing_path(self):
        # Nothing to import when the harness prints to stdout, or when the
        # user asked for a table.
        self.assertEqual(parse_report_path(['ninfer_bench', '-o', 'json']),
                         (None, True))
        self.assertEqual(
            parse_report_path(['ninfer_bench', '--output-file', 'o.txt']),
            ('o.txt', False))
        self.assertEqual(parse_report_path(['ninfer_bench']), (None, False))
        self.assertEqual(parse_report_path(None), (None, False))


class NinferBenchRecorderTestBase(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'report.json')

    def tearDown(self):
        graphsignal.watcher.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, report):
        with open(self.path, 'w') as f:
            json.dump(report, f)

    def _recorder(self, args=None):
        if args is None:
            args = ['/usr/bin/ninfer_bench', '-o', 'json',
                    '--output-file', self.path]
        recorder = NinferBenchRecorder(root_pid=100, pid=100, args=args)
        recorder.setup()
        return recorder

    def _exported(self):
        self.watcher.tick(block=True, force=True)
        return self.watcher.metric_store().export()


class SetupTest(NinferBenchRecorderTestBase):
    def test_disabled_for_other_commands(self):
        recorder = NinferBenchRecorder(
            root_pid=100, pid=100,
            args=['/usr/bin/ninfer-serve', '-o', 'json',
                  '--output-file', self.path])
        recorder.setup()
        self._write(_report())
        recorder.finalize()
        self.assertEqual([m for m in self._exported()
                          if m.name.startswith('ninfer_')], [])

    def test_root_pid_only(self):
        recorder = NinferBenchRecorder(
            root_pid=100, pid=101,
            args=['/usr/bin/ninfer_bench', '-o', 'json',
                  '--output-file', self.path])
        recorder.setup()
        self._write(_report())
        recorder.finalize()
        self.assertEqual([m for m in self._exported()
                          if m.name.startswith('ninfer_')], [])

    def test_disabled_without_a_json_report_file(self):
        for args in (['/usr/bin/ninfer_bench'],
                     ['/usr/bin/ninfer_bench', '-o', 'table',
                      '--output-file', self.path]):
            recorder = self._recorder(args)
            self._write(_report())
            recorder.finalize()
            self.assertEqual([m for m in self._exported()
                              if m.name.startswith('ninfer_')], [])
            # Reset between attempts: one recorder per command line.
            self.watcher.set_counter('reset_marker', 1, self.watcher.start_ns())


class ImportTest(NinferBenchRecorderTestBase):
    def test_imports_nothing_until_the_report_exists(self):
        recorder = self._recorder()
        recorder.finalize()
        self.assertEqual([m for m in self._exported()
                          if m.name.startswith('ninfer_')], [])

        # The harness writes the report as its last act; finalize() then lands.
        self._write(_report())
        recorder.finalize()
        self.assertTrue(self._exported())

    def test_startup_gauges(self):
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_model_load_seconds',
                        metric_type='gauge').datapoint['value'], 41.5)
        self.assertEqual(
            find_metric(exported, 'ninfer_memory_kv_payload_bytes',
                        metric_type='gauge').datapoint['value'], 123683840.0)
        # The token-count capacity, named as the serve recorder names it.
        self.assertEqual(
            find_metric(exported, 'ninfer_engine_kv_capacity_tokens',
                        metric_type='gauge').datapoint['value'], 4096.0)
        self.assertIsNone(
            find_metric(exported, 'ninfer_memory_kv_capacity_bytes'))

    def test_run_tags(self):
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        tags = {k: v for k, v in self.watcher.tags().items()
                if k.startswith('bench.')}
        self.assertEqual(tags['bench.speculative_backend'], 'mtp')
        self.assertEqual(tags['bench.kv_cache'], 'nvfp4')
        self.assertEqual(tags['bench.gpu_name'], 'NVIDIA GB10')
        self.assertEqual(tags['bench.repetitions'], '3')
        # A config flag is tagged, spelled lowercase rather than dropped.
        self.assertEqual(tags['bench.cuda_graph'], 'true')

    def test_per_test_histograms_from_the_measured_reps(self):
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        exported = self._exported()
        decode = find_metric(exported, 'ninfer_request_decode_seconds',
                             tags={'test': 'pp2048+tg128'},
                             metric_type='histogram')
        dp = decode.datapoint
        # count/sum/min/max are the measured repetitions, not the report's
        # own mean-and-stddev summary.
        self.assertEqual(dp['count'], 3)
        self.assertAlmostEqual(dp['min'], 1.258, places=6)
        self.assertAlmostEqual(dp['max'], 1.265, places=6)
        self.assertAlmostEqual(dp['sum'], 1.258 + 1.261 + 1.265, places=6)
        # No bins were measured, so quantiles must read as not-measured
        # rather than as a fabricated value.
        self.assertNotIn('bins', dp)
        self.assertNotIn('counts', dp)

        prefill = find_metric(exported, 'ninfer_request_prefill_seconds',
                              tags={'test': 'pp2048+tg128'},
                              metric_type='histogram')
        self.assertAlmostEqual(prefill.datapoint['min'], 1.152, places=6)

    def test_test_label_keeps_series_apart(self):
        recorder = self._recorder()
        self._write(_report(tests=[_test(label='pp512', kind='pp', n_prompt=512),
                                   _test(label='pp16384', kind='pp',
                                         n_prompt=16384)]))
        recorder.finalize()
        exported = self._exported()
        for label in ('pp512', 'pp16384'):
            metric = find_metric(exported, 'ninfer_request_prefill_seconds',
                                 tags={'test': label}, metric_type='histogram')
            self.assertIsNotNone(metric, msg=f'missing series for {label}')
        tagged = [m for m in exported
                  if m.name == 'ninfer_request_prefill_seconds']
        self.assertEqual({tuple(sorted(m.tags.items())) for m in tagged},
                         {(('kind', 'pp'), ('test', 'pp512')),
                          (('kind', 'pp'), ('test', 'pp16384'))})

    def test_throughput_rates_come_from_the_report(self):
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_throughput_prefill_tokens_per_second',
                        metric_type='gauge').datapoint['value'], 1772.82)
        self.assertEqual(
            find_metric(exported, 'ninfer_throughput_decode_tokens_per_second',
                        metric_type='gauge').datapoint['value'], 101.5)
        self.assertEqual(
            find_metric(exported, 'ninfer_throughput_decode_tokens_total',
                        metric_type='counter').datapoint['total'], 128)

    def test_unmeasured_rate_is_not_reported_as_zero(self):
        # A prefill-only test has no decode rate: the report says null, and
        # null must stay 'not measured' instead of becoming a measured 0.0.
        recorder = self._recorder()
        self._write(_report(tests=[_test(
            label='pp512', kind='pp', n_gen=1, n_prompt=512,
            decode_output_tok_s_mean=None, decode_engine_tok_s_mean=None,
            prefill_tok_s_mean=900.0,
            reps=[_rep(0.5, 0.0, 0.5)])]))
        recorder.finalize()
        exported = self._exported()
        self.assertIsNone(find_metric(
            exported, 'ninfer_throughput_decode_tokens_per_second',
            metric_type='gauge'))
        self.assertIsNotNone(find_metric(
            exported, 'ninfer_throughput_prefill_tokens_per_second',
            metric_type='gauge'))
        # The prefill timing is still measured, at zero.
        decode = find_metric(exported, 'ninfer_request_decode_seconds',
                             tags={'test': 'pp512'}, metric_type='histogram')
        self.assertEqual(decode.datapoint['max'], 0.0)

    def test_speculative_counters_sum_the_reps(self):
        recorder = self._recorder()
        self._write(_report(tests=[_test(reps=[
            _rep(1.0, 1.0, 2.0, drafted=10, accepted=7, rounds=3, fallback=1),
            _rep(1.0, 1.0, 2.0, drafted=20, accepted=15, rounds=4, fallback=0),
        ], speculative={'backend': 'mtp', 'enabled': True, 'draft_window': 7,
                        'rounds': 7, 'drafted_tokens': 30, 'accepted_tokens': 22,
                        'fallback_steps': 1, 'acceptance_rate': 0.733,
                        'acceptance_length': 3.14, 'accepted_per_position': []})]))
        recorder.finalize()
        exported = self._exported()
        tags = {'test': 'pp2048+tg128', 'backend': 'mtp'}
        for name, total in (('ninfer_speculative_drafted_tokens_total', 30),
                            ('ninfer_speculative_accepted_tokens_total', 22),
                            ('ninfer_speculative_rounds_total', 7),
                            ('ninfer_speculative_fallback_steps_total', 1)):
            self.assertEqual(
                find_metric(exported, name, tags=tags,
                            metric_type='counter').datapoint['total'],
                total, msg=name)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_speculative_acceptance_rate',
                        tags=tags,
                        metric_type='gauge').datapoint['value'], 0.733)
        self.assertAlmostEqual(
            find_metric(exported, 'ninfer_speculative_acceptance_length',
                        tags=tags,
                        metric_type='gauge').datapoint['value'], 3.14)

    def test_workspace_peaks_are_per_test(self):
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_memory_workspace_peak_bytes',
                        tags={'test': 'pp2048+tg128'},
                        metric_type='gauge').datapoint['value'], 889044992.0)

    def test_imports_once(self):
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        first = self._exported()
        # Rewrite with a different value: a completed import must not re-read.
        self._write(_report(load={'architecture': 'qwen3_moe', 'load_seconds': 99.0,
                                  'upload_seconds': 0.79}))
        recorder.finalize()
        recorder.on_tick()
        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_model_load_seconds',
                        metric_type='gauge').datapoint['value'], 41.5)
        self.assertTrue(first)

    def test_partial_file_is_left_for_a_later_attempt(self):
        recorder = self._recorder()
        # A half-written report parses as nothing; the import must wait.
        with open(self.path, 'w') as f:
            f.write('{"schema_version": 15, "artifact_ty')
        recorder.finalize()
        self.assertEqual([m for m in self._exported()
                          if m.name.startswith('ninfer_')], [])
        self._write(_report())
        recorder.finalize()
        self.assertTrue(self._exported())

    def test_foreign_artifact_is_rejected(self):
        recorder = self._recorder()
        self._write({'artifact_type': 'something_else', 'schema_version': 15,
                     'tests': [_test()]})
        recorder.finalize()
        self.assertEqual([m for m in self._exported()
                          if m.name.startswith('ninfer_')], [])

    def test_newer_schema_still_imports_recognized_fields(self):
        recorder = self._recorder()
        self._write(_report(schema_version=SCHEMA_VERSION + 3))
        recorder.finalize()
        exported = self._exported()
        self.assertEqual(
            find_metric(exported, 'ninfer_model_load_seconds',
                        metric_type='gauge').datapoint['value'], 41.5)

    def test_missing_sections_do_not_break_the_import(self):
        recorder = self._recorder()
        report = _report()
        del report['memory']
        del report['load']
        self._write(report)
        recorder.finalize()
        exported = self._exported()
        self.assertIsNone(find_metric(exported, 'ninfer_model_load_seconds'))
        # The per-test series still land.
        self.assertIsNotNone(find_metric(
            exported, 'ninfer_request_prefill_seconds',
            tags={'test': 'pp2048+tg128'}, metric_type='histogram'))

    def test_malformed_test_entries_are_skipped(self):
        recorder = self._recorder()
        # A non-dict, a dict with no label, and one with an empty label are all
        # skipped rather than recorded under a blank test tag.
        self._write(_report(tests=[
            'not-a-dict', {}, {'label': '', 'reps': [_rep(1.0, 1.0, 2.0)]},
            _test(),
        ]))
        recorder.finalize()
        exported = self._exported()
        labels = {m.tags.get('test') for m in exported
                  if m.name == 'ninfer_request_prefill_seconds'}
        self.assertEqual(labels, {'pp2048+tg128'})

    def test_payload_reports_histograms_without_quantiles(self):
        # The payload's own view: exact totals, quantiles null because no bins
        # were measured. This is the whole reason for not inventing bins.
        recorder = self._recorder()
        self._write(_report())
        recorder.finalize()
        payload = build_payload()
        record = next(r for r in payload['metrics']
                      if r['name'] == 'ninfer_request_decode_seconds')
        stats = record['stats']
        self.assertEqual(stats['count'], 3)
        self.assertIsNone(stats['p50'])
        self.assertIsNone(stats['p95'])
        self.assertIsNotNone(stats['min'])
        self.assertIsNotNone(stats['max'])


if __name__ == '__main__':
    unittest.main()
