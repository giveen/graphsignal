import time
import types
import unittest
from unittest.mock import MagicMock

import xxhash

from graphsignal.collector.collector import Collector
from graphsignal.proto import signals_pb2
from graphsignal.signals.logs import LogStore
from graphsignal.signals.metrics import MetricStore
from graphsignal.signals.resources import ResourceStore


def _tags_dict(proto):
    return {t.key: t.value for t in proto.tags}


class CollectorTest(unittest.TestCase):
    """The collector snapshots the stores non-destructively and owns all delta
    state; the stores themselves are never reset."""

    def setUp(self):
        self.metric_store = MetricStore()
        self.metric_store._last_cleanup_ts = time.time_ns()
        self.log_store = LogStore()
        self.resource_store = ResourceStore()
        self.watcher = types.SimpleNamespace(
            tags=lambda: {'instance.id': 'r1', 'host.name': 'h1'},
            metric_store=lambda: self.metric_store,
            log_store=lambda: self.log_store,
            resource_store=lambda: self.resource_store)

        self.collector = Collector('k1', api_base='http://127.0.0.1:1')
        self.uploader = MagicMock()
        self.collector._uploader = self.uploader

    def _uploaded_metrics(self):
        return [call.args[0] for call in self.uploader.upload_metric.call_args_list]

    def test_gauge_uploaded_only_when_ts_advances(self):
        self.metric_store.set_gauge('g1', 2.0, measurement_ts=20)

        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        proto = protos[0]
        self.assertEqual(proto.type, signals_pb2.Metric.MetricType.GAUGE_METRIC)
        self.assertEqual(len(proto.datapoints), 1)
        self.assertEqual(proto.datapoints[0].gauge, 2.0)
        self.assertEqual(proto.datapoints[0].measurement_ts, 20)
        self.uploader.flush.assert_called_once()

        # Unchanged ts: the same snapshot is not re-uploaded.
        self.uploader.upload_metric.reset_mock()
        self.collector.on_tick(self.watcher)
        self.uploader.upload_metric.assert_not_called()

        # ts advanced: the new snapshot goes out.
        self.metric_store.set_gauge('g1', 3.0, measurement_ts=30)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].datapoints[0].gauge, 3.0)
        self.assertEqual(protos[0].datapoints[0].measurement_ts, 30)

    def test_counter_delta_uploaded(self):
        self.metric_store.set_counter('c1', 100, measurement_ts=10)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].type,
                         signals_pb2.Metric.MetricType.COUNTER_METRIC)
        self.assertEqual(protos[0].datapoints[0].total, 100)

        # Increase: only the delta is uploaded.
        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_counter('c1', 150, measurement_ts=20)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].datapoints[0].total, 50)
        self.assertEqual(protos[0].datapoints[0].measurement_ts, 20)

        # Unchanged total: nothing uploaded.
        self.uploader.upload_metric.reset_mock()
        self.collector.on_tick(self.watcher)
        self.uploader.upload_metric.assert_not_called()

    def test_counter_reset_uploads_full_new_total(self):
        self.metric_store.set_counter('c1', 150, measurement_ts=10)
        self.collector.on_tick(self.watcher)

        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_counter('c1', 30, measurement_ts=20)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].datapoints[0].total, 30)

    def test_summary_delta_uploaded(self):
        self.metric_store.set_summary(
            's1', count=10, sum_val=100.0, sum2_val=1200.0, measurement_ts=10)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].type,
                         signals_pb2.Metric.MetricType.SUMMARY_METRIC)
        dp = protos[0].datapoints[0]
        self.assertEqual(dp.summary.count, 10)
        self.assertEqual(dp.summary.sum, 100.0)
        self.assertEqual(dp.summary.sum2, 1200.0)
        self.assertEqual(dp.measurement_ts, 10)

        # Count unchanged: nothing uploaded.
        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_summary(
            's1', count=10, sum_val=100.0, sum2_val=1200.0, measurement_ts=20)
        self.collector.on_tick(self.watcher)
        self.uploader.upload_metric.assert_not_called()

        # Count grew: only the count/sum/sum2 deltas go out.
        self.metric_store.set_summary(
            's1', count=15, sum_val=160.0, sum2_val=2000.0, measurement_ts=30)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        dp = protos[0].datapoints[0]
        self.assertEqual(dp.summary.count, 5)
        self.assertEqual(dp.summary.sum, 60.0)
        self.assertEqual(dp.summary.sum2, 800.0)
        self.assertEqual(dp.measurement_ts, 30)

    def test_summary_count_reset_uploads_full_current_values(self):
        self.metric_store.set_summary(
            's1', count=15, sum_val=160.0, sum2_val=2000.0, measurement_ts=10)
        self.collector.on_tick(self.watcher)

        # Writer restarted: count dropped, full current values are uploaded.
        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_summary(
            's1', count=3, sum_val=10.0, sum2_val=40.0, measurement_ts=20)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        dp = protos[0].datapoints[0]
        self.assertEqual(dp.summary.count, 3)
        self.assertEqual(dp.summary.sum, 10.0)
        self.assertEqual(dp.summary.sum2, 40.0)

    def test_histogram_bin_deltas_uploaded(self):
        self.metric_store.set_histogram(
            'h1', bins=[5, 20], counts=[1, 2], measurement_ts=10)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].type,
                         signals_pb2.Metric.MetricType.HISTOGRAM_METRIC)
        dp = protos[0].datapoints[0]
        self.assertEqual(list(dp.histogram.bins), [5, 20])
        self.assertEqual(list(dp.histogram.counts), [1, 2])
        self.assertEqual(dp.measurement_ts, 10)

        # Counts unchanged: nothing uploaded.
        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_histogram(
            'h1', bins=[5, 20], counts=[1, 2], measurement_ts=20)
        self.collector.on_tick(self.watcher)
        self.uploader.upload_metric.assert_not_called()

        # Counts grew: only bins with positive count deltas go out.
        self.metric_store.set_histogram(
            'h1', bins=[5, 20], counts=[2, 2], measurement_ts=30)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        dp = protos[0].datapoints[0]
        self.assertEqual(list(dp.histogram.bins), [5])
        self.assertEqual(list(dp.histogram.counts), [1])
        self.assertEqual(dp.measurement_ts, 30)

    def test_histogram_count_reset_uploads_full_current_bins(self):
        self.metric_store.set_histogram(
            'h1', bins=[5, 20], counts=[4, 6], measurement_ts=10)
        self.collector.on_tick(self.watcher)

        # Writer restarted: total count dropped, full current bins are uploaded.
        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_histogram(
            'h1', bins=[5, 20], counts=[1, 2], measurement_ts=20)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        dp = protos[0].datapoints[0]
        self.assertEqual(list(dp.histogram.bins), [5, 20])
        self.assertEqual(list(dp.histogram.counts), [1, 2])

    def test_profile_frames_uploaded_with_frame_ids(self):
        self.metric_store.set_profile(
            'p1', frames={'a': 100.0, 'b': 7.0}, samples={'a': 10, 'b': 1},
            measurement_ts=10)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        proto = protos[0]
        self.assertEqual(proto.type,
                         signals_pb2.Metric.MetricType.PROFILE_METRIC)

        id_a = xxhash.xxh64(b'a').intdigest()
        id_b = xxhash.xxh64(b'b').intdigest()
        self.assertEqual({f.name: f.frame_id for f in proto.frames},
                         {'a': id_a, 'b': id_b})
        dp = proto.datapoints[0]
        self.assertEqual(dict(zip(dp.profile.frame_ids, dp.profile.values)),
                         {id_a: 100, id_b: 7})
        self.assertEqual(dict(zip(dp.profile.frame_ids, dp.profile.samples)),
                         {id_a: 10, id_b: 1})
        self.assertEqual(dp.measurement_ts, 10)

        # Frames unchanged: nothing uploaded.
        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_profile(
            'p1', frames={'a': 100.0, 'b': 7.0}, samples={'a': 10, 'b': 1},
            measurement_ts=20)
        self.collector.on_tick(self.watcher)
        self.uploader.upload_metric.assert_not_called()

        # One frame grew: its value and sample deltas go out together.
        self.metric_store.set_profile(
            'p1', frames={'a': 150.0, 'b': 7.0}, samples={'a': 14, 'b': 1},
            measurement_ts=30)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        proto = protos[0]
        self.assertEqual({f.name: f.frame_id for f in proto.frames},
                         {'a': id_a})
        dp = proto.datapoints[0]
        self.assertEqual(list(dp.profile.frame_ids), [id_a])
        self.assertEqual(list(dp.profile.values), [50])
        self.assertEqual(list(dp.profile.samples), [4])
        self.assertEqual(dp.measurement_ts, 30)

    def test_profile_frame_value_decrease_uploads_full_current_frames(self):
        # Reset semantics (as for counters/summaries/histograms):
        # a decreased frame value means the writer restarted, so the full
        # current frames must be re-uploaded. Not implemented yet in
        # Collector._collect_profile, which drops decreases silently.
        self.metric_store.set_profile(
            'p1', frames={'a': 100.0, 'b': 7.0}, samples={'a': 10, 'b': 1},
            measurement_ts=10)
        self.collector.on_tick(self.watcher)

        self.uploader.upload_metric.reset_mock()
        self.metric_store.set_profile(
            'p1', frames={'a': 20.0, 'b': 3.0}, samples={'a': 2, 'b': 1},
            measurement_ts=20)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        dp = protos[0].datapoints[0]
        id_a = xxhash.xxh64(b'a').intdigest()
        id_b = xxhash.xxh64(b'b').intdigest()
        self.assertEqual(dict(zip(dp.profile.frame_ids, dp.profile.values)),
                         {id_a: 20, id_b: 3})
        self.assertEqual(dict(zip(dp.profile.frame_ids, dp.profile.samples)),
                         {id_a: 2, id_b: 1})

    def test_state_keyed_by_type_same_name_does_not_collide(self):
        self.metric_store.set_summary(
            'm1', count=2, sum_val=4.0, measurement_ts=10)
        self.metric_store.set_histogram(
            'm1', bins=[1], counts=[2], measurement_ts=10)
        self.collector.on_tick(self.watcher)
        types_uploaded = sorted(p.type for p in self._uploaded_metrics())
        self.assertEqual(types_uploaded, sorted([
            signals_pb2.Metric.MetricType.SUMMARY_METRIC,
            signals_pb2.Metric.MetricType.HISTOGRAM_METRIC]))

        # Neither changed: nothing uploaded.
        self.uploader.upload_metric.reset_mock()
        self.collector.on_tick(self.watcher)
        self.uploader.upload_metric.assert_not_called()

        # Only the histogram grew: only it goes out.
        self.metric_store.set_histogram(
            'm1', bins=[1], counts=[3], measurement_ts=20)
        self.collector.on_tick(self.watcher)
        protos = self._uploaded_metrics()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].type,
                         signals_pb2.Metric.MetricType.HISTOGRAM_METRIC)
        dp = protos[0].datapoints[0]
        self.assertEqual(list(dp.histogram.counts), [1])

    def test_logs_uploaded_incrementally(self):
        self.log_store.log_message(message='m1', level='error', timestamp_ns=10,
                                   tags={'t1': '1'})
        self.log_store.log_message(message='m2', level='info', timestamp_ns=20,
                                   tags={'t1': '1'})
        self.collector.on_tick(self.watcher)
        self.assertEqual(self.uploader.upload_log_batch.call_count, 1)
        batch = self.uploader.upload_log_batch.call_args.args[0]
        self.assertEqual([e.message for e in batch.log_entries], ['m1', 'm2'])
        self.assertEqual(batch.log_entries[0].level,
                         signals_pb2.LogEntry.LogLevel.ERROR_LEVEL)
        self.assertEqual(_tags_dict(batch),
                         {'instance.id': 'r1', 'host.name': 'h1', 't1': '1'})

        # No new entries: nothing uploaded.
        self.uploader.upload_log_batch.reset_mock()
        self.collector.on_tick(self.watcher)
        self.uploader.upload_log_batch.assert_not_called()

        # Only entries newer than the last uploaded ts go out.
        self.log_store.log_message(message='m3', level='error', timestamp_ns=30)
        self.collector.on_tick(self.watcher)
        self.assertEqual(self.uploader.upload_log_batch.call_count, 1)
        batch = self.uploader.upload_log_batch.call_args.args[0]
        self.assertEqual([e.message for e in batch.log_entries], ['m3'])

    def test_resource_reuploaded_only_when_last_seen_advances(self):
        self.resource_store.update_resource(
            'process', tags={'process.pid': '1'}, attributes={'a': '1'},
            first_seen_ts=10, last_seen_ts=20)
        self.collector.on_tick(self.watcher)
        self.assertEqual(self.uploader.upload_resource.call_count, 1)
        proto = self.uploader.upload_resource.call_args.args[0]
        self.assertEqual(proto.kind, 'process')
        self.assertEqual(proto.first_seen_ts, 10)
        self.assertEqual(proto.last_seen_ts, 20)
        self.assertEqual([(a.name, a.value) for a in proto.attributes],
                         [('a', '1')])

        # last_seen_ts unchanged: not re-uploaded.
        self.uploader.upload_resource.reset_mock()
        self.collector.on_tick(self.watcher)
        self.uploader.upload_resource.assert_not_called()

        # last_seen_ts advanced: re-uploaded.
        self.resource_store.update_resource(
            'process', tags={'process.pid': '1'}, attributes={'a': '1'},
            first_seen_ts=10, last_seen_ts=30)
        self.collector.on_tick(self.watcher)
        self.assertEqual(self.uploader.upload_resource.call_count, 1)

    def test_on_tick_does_not_reset_stores(self):
        self.metric_store.set_gauge('g1', 1.0, measurement_ts=10)
        self.metric_store.set_counter('c1', 100, measurement_ts=10)
        self.log_store.log_message(message='m1', timestamp_ns=10)
        self.resource_store.update_resource('process',
                                            first_seen_ts=1, last_seen_ts=2)

        self.collector.on_tick(self.watcher)

        exported = {m.name: m for m in self.metric_store.export()}
        self.assertEqual(exported['g1'].datapoint, {'ts': 10, 'value': 1.0})
        self.assertEqual(exported['c1'].datapoint, {'ts': 10, 'total': 100})
        self.assertEqual(len(self.log_store.export()), 1)
        self.assertEqual(len(self.resource_store.export()), 1)

    def test_metric_tags_merged_with_watcher_tags(self):
        self.metric_store.set_gauge('g1', 1.0, measurement_ts=10,
                                    tags={'process.pid': '42'})
        self.collector.on_tick(self.watcher)

        proto = self.uploader.upload_metric.call_args.args[0]
        self.assertEqual(_tags_dict(proto), {
            'instance.id': 'r1', 'host.name': 'h1', 'process.pid': '42'})

    def test_setup_and_shutdown_delegate_to_uploader(self):
        self.collector.setup()
        self.uploader.setup.assert_called_once()
        self.collector.shutdown()
        self.uploader.flush.assert_called_once()


if __name__ == '__main__':
    unittest.main()
