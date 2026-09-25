import threading
import time
import unittest
from unittest.mock import patch

import graphsignal.watcher
from graphsignal.recorders.prometheus_recorder import (
    PrometheusRecorder, _buckets_to_bins, build_metrics_endpoint,
    format_metrics_host, normalize_metrics_path)
from test.test_utils import configure_test_watcher, find_metric

_GAUGE_BODY = """# HELP vllm_num_requests_running Number of running requests.
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model_name="m"} %s
"""

_COUNTER_BODY = """# HELP vllm_request_success Count of successful requests.
# TYPE vllm_request_success counter
vllm_request_success{model_name="m"} %s
"""

_HISTOGRAM_BODY = """# HELP vllm_e2e_latency Request latency.
# TYPE vllm_e2e_latency histogram
vllm_e2e_latency_bucket{le="0.5",model_name="m"} %(b1)s
vllm_e2e_latency_bucket{le="1.0",model_name="m"} %(b2)s
vllm_e2e_latency_bucket{le="+Inf",model_name="m"} %(count)s
vllm_e2e_latency_count{model_name="m"} %(count)s
vllm_e2e_latency_sum{model_name="m"} %(sum)s
"""


def _recorder(**kwargs):
    kwargs.setdefault('root_pid', 123)
    kwargs.setdefault('pid', 123)
    recorder = PrometheusRecorder(**kwargs)
    recorder.setup()
    # Skip the initial detection delay so the first on_tick scrapes.
    recorder._next_detect_ts = 0
    return recorder


class EndpointBuildingTest(unittest.TestCase):
    def test_normalize_metrics_path(self):
        self.assertEqual(normalize_metrics_path(None), '/metrics')
        self.assertEqual(normalize_metrics_path('/prometheus/metrics'),
                         '/prometheus/metrics')
        self.assertEqual(normalize_metrics_path('prometheus/metrics'),
                         '/prometheus/metrics')

    def test_build_metrics_endpoint(self):
        self.assertEqual(build_metrics_endpoint(8000),
                         'http://127.0.0.1:8000/metrics')
        self.assertEqual(
            build_metrics_endpoint(8000, metrics_path='/prometheus/metrics',
                                   metrics_host='localhost'),
            'http://localhost:8000/prometheus/metrics')

    def test_format_metrics_host_ipv6(self):
        self.assertEqual(format_metrics_host('::1'), '[::1]')


class BucketsToBinsTest(unittest.TestCase):
    def test_cumulative_buckets_become_noncumulative_bins(self):
        bins, counts = _buckets_to_bins([('0.5', 4), ('1.0', 7), ('2.0', 9)])
        self.assertEqual(bins, [0.5, 1.0, 2.0])
        self.assertEqual(counts, [4, 3, 2])

    def test_inf_remainder_folded_into_last_finite_bound(self):
        # The last finite bucket has a zero delta (omitted), so the +Inf
        # remainder lands in a new bin at the largest finite bound.
        bins, counts = _buckets_to_bins([('0.5', 4), ('1.0', 4), ('+Inf', 10)])
        self.assertEqual(bins, [0.5, 1.0])
        self.assertEqual(counts, [4, 6])

    def test_inf_remainder_merged_into_emitted_last_bound(self):
        bins, counts = _buckets_to_bins([('0.5', 4), ('+Inf', 10)])
        self.assertEqual(bins, [0.5])
        self.assertEqual(counts, [10])

    def test_zero_delta_buckets_omitted(self):
        bins, counts = _buckets_to_bins(
            [('0.5', 4), ('1.0', 4), ('2.0', 6), ('+Inf', 6)])
        self.assertEqual(bins, [0.5, 2.0])
        self.assertEqual(counts, [4, 2])

    def test_unusable_buckets_return_none(self):
        self.assertEqual(_buckets_to_bins(None), (None, None))
        self.assertEqual(_buckets_to_bins([]), (None, None))
        # Only a +Inf bucket: no finite bounds to place counts on.
        self.assertEqual(_buckets_to_bins([('+Inf', 5)]), (None, None))
        # Unparsable bounds and non-finite values are skipped.
        self.assertEqual(_buckets_to_bins([('nope', 5)]), (None, None))
        self.assertEqual(_buckets_to_bins([('0.5', float('nan'))]),
                         (None, None))
        # All-zero cumulative values produce no bins.
        self.assertEqual(_buckets_to_bins([('0.5', 0), ('+Inf', 0)]),
                         (None, None))


class PrometheusRecorderScrapeTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _tick(self, recorder, body):
        with patch.object(recorder, '_fetch_metrics', return_value=body) as fetch_m:
            recorder.on_tick()
            if recorder._fetch_thread is not None:
                recorder._fetch_thread.join(timeout=1)
            recorder.on_tick()
        return fetch_m

    def test_no_metrics_port_never_fetches(self):
        recorder = _recorder(metrics_port=None)
        self.assertIsNone(recorder._endpoint)
        with patch.object(recorder, '_fetch_metrics') as fetch_m:
            recorder.on_tick()
        fetch_m.assert_not_called()

    def test_fetches_only_configured_endpoint(self):
        recorder = _recorder(metrics_port=8000)
        self.assertEqual(recorder._endpoint, 'http://127.0.0.1:8000/metrics')
        fetch_m = self._tick(recorder, _GAUGE_BODY % '3.0')
        fetch_m.assert_called_once_with('http://127.0.0.1:8000/metrics')

    def test_fetches_configured_path_and_host(self):
        recorder = _recorder(metrics_port=8000,
                             metrics_path='/prometheus/metrics',
                             metrics_host='localhost')
        fetch_m = self._tick(recorder, _GAUGE_BODY % '3.0')
        fetch_m.assert_called_once_with(
            'http://localhost:8000/prometheus/metrics')

    def test_gauges_overwrite_datapoint(self):
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, _GAUGE_BODY % '3.0')
        self._tick(recorder, _GAUGE_BODY % '5.0')

        metric = find_metric(self.watcher.metric_store().export(),
                             'vllm_num_requests_running', {'model_name': 'm'})
        self.assertEqual(metric.type, 'gauge')
        self.assertEqual(metric.datapoint['value'], 5.0)

    def test_counters_pass_through_raw_cumulative_totals(self):
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, _COUNTER_BODY % '100.0')

        metric = find_metric(self.watcher.metric_store().export(),
                             'vllm_request_success', {'model_name': 'm'})
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 100.0)

        # Second scrape: the raw total is emitted again (no delta state);
        # the store keeps the latest snapshot.
        self._tick(recorder, _COUNTER_BODY % '150.0')
        metric = find_metric(self.watcher.metric_store().export(),
                             'vllm_request_success', {'model_name': 'm'})
        self.assertEqual(metric.datapoint['total'], 150.0)

    def test_histogram_family_emits_one_histogram_with_bins_and_totals(self):
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, _HISTOGRAM_BODY % {
            'b1': '4', 'b2': '7', 'count': '10', 'sum': '25.5'})

        exported = self.watcher.metric_store().export()
        histogram = find_metric(exported, 'vllm_e2e_latency',
                                {'model_name': 'm'}, metric_type='histogram')
        self.assertIsNotNone(histogram)
        # Cumulative le buckets converted: 0.5→4, 1.0→3, +Inf remainder 3
        # folded into the last finite bound.
        self.assertEqual(histogram.datapoint['bins'], [0.5, 1.0])
        self.assertEqual(histogram.datapoint['counts'], [4, 6])
        # The exact totals from _count/_sum ride on the same datapoint.
        self.assertEqual(histogram.datapoint['count'], 10)
        self.assertEqual(histogram.datapoint['sum'], 25.5)
        # Prometheus exposes no extremes.
        self.assertNotIn('min', histogram.datapoint)
        self.assertNotIn('max', histogram.datapoint)
        # One metric, not one per half.
        self.assertEqual(
            len([m for m in exported if m.name == 'vllm_e2e_latency']), 1)

        # Every scrape re-emits raw cumulative values; latest snapshots kept.
        self._tick(recorder, _HISTOGRAM_BODY % {
            'b1': '6', 'b2': '10', 'count': '15', 'sum': '40.0'})
        exported = self.watcher.metric_store().export()
        histogram = find_metric(exported, 'vllm_e2e_latency',
                                {'model_name': 'm'}, metric_type='histogram')
        self.assertEqual(histogram.datapoint['bins'], [0.5, 1.0])
        self.assertEqual(histogram.datapoint['counts'], [6, 9])
        self.assertEqual(histogram.datapoint['count'], 15)
        self.assertEqual(histogram.datapoint['sum'], 40.0)

    def test_summary_family_emits_a_histogram_without_bins(self):
        body = """# HELP rpc_duration_seconds RPC duration.
# TYPE rpc_duration_seconds summary
rpc_duration_seconds{quantile="0.5"} 0.05
rpc_duration_seconds_count 12
rpc_duration_seconds_sum 1.5
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        exported = self.watcher.metric_store().export()
        histogram = find_metric(exported, 'rpc_duration_seconds',
                                metric_type='histogram')
        self.assertIsNotNone(histogram)
        self.assertEqual(histogram.datapoint['count'], 12)
        # Fractional seconds survive: a truncated sum would read 1.
        self.assertEqual(histogram.datapoint['sum'], 1.5)
        self.assertNotIn('bins', histogram.datapoint)

    def test_skips_gauge_histogram_bucket_labels(self):
        body = """# HELP sglang:routing_key_running_req_count Distribution of routing keys.
# TYPE sglang:routing_key_running_req_count gauge
sglang:routing_key_running_req_count{gt="0",le="1",model_name="m"} 2.0
sglang:routing_key_running_req_count{gt="1",le="2",model_name="m"} 1.0
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="m"} 12.5
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        exported = self.watcher.metric_store().export()
        self.assertIsNone(
            find_metric(exported, 'sglang:routing_key_running_req_count'))
        metric = find_metric(exported, 'sglang:gen_throughput')
        self.assertEqual(metric.datapoint['value'], 12.5)
        self.assertNotIn('gt', metric.tags)
        self.assertNotIn('le', metric.tags)

    def test_strips_pid_label(self):
        body = """# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{pid="1973",model_name="m"} 4.0
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        metric = find_metric(self.watcher.metric_store().export(),
                             'sglang:num_running_reqs')
        self.assertNotIn('pid', metric.tags)
        self.assertEqual(metric.tags.get('model_name'), 'm')

    def test_skips_non_finite_gauge_values(self):
        body = """# HELP sglang:fwd_occupancy Forward pass GPU occupancy percentage.
# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy NaN
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput 12.5
"""
        recorder = _recorder(metrics_port=8000)
        self._tick(recorder, body)

        exported = self.watcher.metric_store().export()
        self.assertIsNone(find_metric(exported, 'sglang:fwd_occupancy'))
        self.assertIsNotNone(find_metric(exported, 'sglang:gen_throughput'))

    def test_non_prometheus_body_is_not_emitted(self):
        recorder = _recorder(metrics_port=8000)
        with patch.object(recorder, '_fetch_metrics',
                          return_value='<html>nope</html>'), \
             patch.object(recorder, '_parse_and_emit') as parse_m:
            recorder.on_tick()
            if recorder._fetch_thread is not None:
                recorder._fetch_thread.join(timeout=1)
            recorder.on_tick()
        parse_m.assert_not_called()
        self.assertFalse(recorder._verified)

    def test_fetch_failure_backs_off_without_raising(self):
        recorder = _recorder(metrics_port=8000)
        with patch.object(recorder, '_fetch_metrics',
                          side_effect=OSError('refused')):
            recorder.on_tick()  # must not raise
            if recorder._fetch_thread is not None:
                recorder._fetch_thread.join(timeout=1)
            recorder.on_tick()  # consume the background failure
        self.assertFalse(recorder._verified)

    def test_slow_fetch_does_not_block_tick_or_start_overlap(self):
        recorder = _recorder(metrics_port=8000)
        started = threading.Event()
        release = threading.Event()

        def slow_fetch(url):
            started.set()
            release.wait(timeout=2)
            return _GAUGE_BODY % '1.0'

        with patch.object(recorder, '_fetch_metrics', side_effect=slow_fetch) as fetch_m:
            start = time.monotonic()
            recorder.on_tick()
            self.assertLess(time.monotonic() - start, 0.1)
            self.assertTrue(started.wait(timeout=1))
            recorder.on_tick()
            fetch_m.assert_called_once()

            release.set()
            recorder._fetch_thread.join(timeout=1)
            recorder.on_tick()


if __name__ == '__main__':
    unittest.main()
