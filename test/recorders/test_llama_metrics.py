import unittest
from unittest.mock import patch

import graphsignal.watcher
from graphsignal.recorders.prometheus_recorder import PrometheusRecorder
from test.test_utils import configure_test_watcher, find_metric

# Verbatim-shaped llama.cpp /metrics output from
# tools/server/server-task.cpp: server_task_result_metrics::to_metrics().
_LLAMA_METRICS = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 17
# HELP llamacpp:prompt_tokens_cached_total Number of prompt tokens reused from the cache
# TYPE llamacpp:prompt_tokens_cached_total counter
llamacpp:prompt_tokens_cached_total 3
# HELP llamacpp:prompt_seconds_total Total time spent processing prompts
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 0.125
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 16
# HELP llamacpp:n_decode_total Total number of llama_decode() calls
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 15
# HELP llamacpp:spec_decode_num_accepted_tokens_total Speculative accepted tokens
# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
llamacpp:spec_decode_num_accepted_tokens_total 4
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 54.0
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 49.0
# HELP llamacpp:requests_processing Number of requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# HELP llamacpp:requests_deferred Number of requests deferred
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
# HELP llamacpp:n_busy_slots_per_decode Average number of busy slots per llama_decode() call
# TYPE llamacpp:n_busy_slots_per_decode gauge
llamacpp:n_busy_slots_per_decode 1.0
# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position
# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 2
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 2
"""


class LlamaMetricsTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def _scrape(self, body=_LLAMA_METRICS):
        recorder = PrometheusRecorder(root_pid=1, pid=1, metrics_port=8080)
        recorder.setup()
        recorder._next_detect_ts = 0
        with patch.object(recorder, '_fetch_metrics', return_value=body):
            recorder.on_tick()
            recorder._fetch_thread.join(timeout=1)
            recorder.on_tick()
        return self.watcher.metric_store().export()

    def test_llama_cpp_metrics_are_parsed_with_bounded_tags(self):
        metrics = self._scrape()
        # prometheus_client normalizes a counter family's exposition name by
        # removing the conventional `_total` suffix from the family name.
        expected_counters = {
            'llamacpp:prompt_tokens': 17.0,
            'llamacpp:prompt_tokens_cached': 3.0,
            'llamacpp:prompt_seconds': 0.125,
            'llamacpp:tokens_predicted': 16.0,
            'llamacpp:n_decode': 15.0,
            'llamacpp:spec_decode_num_accepted_tokens': 4.0,
        }
        for name, value in expected_counters.items():
            metric = find_metric(metrics, name, metric_type='counter')
            self.assertIsNotNone(metric, name)
            self.assertEqual(metric.datapoint['total'], value)

        for name, value in {
            'llamacpp:prompt_tokens_seconds': 54.0,
            'llamacpp:predicted_tokens_seconds': 49.0,
            'llamacpp:requests_processing': 1.0,
            'llamacpp:requests_deferred': 0.0,
            'llamacpp:n_busy_slots_per_decode': 1.0,
        }.items():
            metric = find_metric(metrics, name, metric_type='gauge')
            self.assertIsNotNone(metric, name)
            self.assertEqual(metric.datapoint['value'], value)

        # The per-position series is aggregated into one untagged metric rather
        # than creating an unbounded number of position-labeled metrics.
        per_position = [m for m in metrics
                        if m.name == 'llamacpp:spec_decode_num_accepted_tokens_per_pos']
        self.assertEqual(len(per_position), 1)
        self.assertNotIn('position', per_position[0].tags)
        self.assertEqual(per_position[0].datapoint['total'], 4.0)


if __name__ == '__main__':
    unittest.main()
