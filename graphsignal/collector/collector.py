"""Production feedback loop (optional addon).

Active only when an API key is provided. On every watcher tick the collector
snapshots the signal stores (non-destructive `export()`), computes what is new
since its last upload, converts it to protobuf, and uploads it to Graphsignal.
The stores themselves are never reset — the collector owns all delta state.
"""

import logging

import xxhash

from graphsignal.collector.signal_uploader import SignalUploader
from graphsignal.proto import signals_pb2
from graphsignal.signals import metrics as metrics_module

logger = logging.getLogger('graphsignal')

_LOG_LEVEL_MAP = {
    'debug': signals_pb2.LogEntry.LogLevel.DEBUG_LEVEL,
    'info': signals_pb2.LogEntry.LogLevel.INFO_LEVEL,
    'warning': signals_pb2.LogEntry.LogLevel.WARNING_LEVEL,
    'error': signals_pb2.LogEntry.LogLevel.ERROR_LEVEL,
    'critical': signals_pb2.LogEntry.LogLevel.CRITICAL_LEVEL,
}


class Collector:
    def __init__(self, api_key, api_base=None):
        self._uploader = SignalUploader(api_key, api_base=api_base)
        # Delta state per metric series key; stores are never reset, so the
        # collector remembers what it already uploaded.
        self._metric_state = {}
        self._last_log_ts = 0
        self._resource_state = {}

    def setup(self):
        self._uploader.setup()

    def uploader(self):
        return self._uploader

    def on_tick(self, watcher):
        try:
            global_tags = watcher.tags()
            self._collect_metrics(watcher.metric_store(), global_tags)
            self._collect_logs(watcher.log_store(), global_tags)
            self._collect_resources(watcher.resource_store(), global_tags)
            self._uploader.flush()
        except Exception:
            logger.error('Error collecting signals for upload', exc_info=True)

    def shutdown(self):
        self._uploader.flush(timeout=SignalUploader.SHUTDOWN_TIMEOUT_SEC)

    def _collect_metrics(self, metric_store, global_tags):
        snapshot = metric_store.export()
        seen_keys = set()

        for metric in snapshot:
            if metric.datapoint is None:
                continue
            key = (metric.name, metric.type, frozenset(metric.tags.items()))
            seen_keys.add(key)

            if metric.type == metrics_module.GAUGE:
                self._collect_gauge(key, metric, global_tags)
            elif metric.type == metrics_module.COUNTER:
                self._collect_counter(key, metric, global_tags)
            elif metric.type == metrics_module.HISTOGRAM:
                self._collect_histogram(key, metric, global_tags)
            elif metric.type == metrics_module.PROFILE:
                self._collect_profile(key, metric, global_tags)

        expired = [key for key in self._metric_state if key not in seen_keys]
        for key in expired:
            del self._metric_state[key]

    def _collect_gauge(self, key, metric, global_tags):
        dp = metric.datapoint
        ts = dp.get('ts') or 0
        last_ts = self._metric_state.get(key, 0)
        if ts <= last_ts:
            return
        self._metric_state[key] = ts

        proto = self._proto_metric(
            signals_pb2.Metric.MetricType.GAUGE_METRIC, metric, global_tags)
        proto_dp = proto.datapoints.add()
        proto_dp.gauge = dp['value']
        proto_dp.measurement_ts = ts
        self._uploader.upload_metric(proto)

    def _collect_counter(self, key, metric, global_tags):
        dp = metric.datapoint
        total = dp.get('total') or 0
        last_total = self._metric_state.get(key)
        if last_total is not None and total >= last_total:
            delta = total - last_total
        else:
            delta = total
        self._metric_state[key] = total
        if delta == 0:
            return

        proto = self._proto_metric(
            signals_pb2.Metric.MetricType.COUNTER_METRIC, metric, global_tags)
        proto_dp = proto.datapoints.add()
        proto_dp.total = delta
        proto_dp.measurement_ts = dp.get('ts') or 0
        self._uploader.upload_metric(proto)

    def _collect_histogram(self, key, metric, global_tags):
        """Bin deltas and aggregate deltas, from whichever halves are present.

        Bins and count/sum are deltaed independently: a source can report
        either or both, and a bins-only tick must still upload when its buckets
        moved. `min`/`max` are the exception — they are lifetime extremes of a
        cumulative instrument, and the difference of two of them is not the
        interval's extreme, so the current values are sent as they stand and
        the platform folds them with min/max rather than a sum.
        """
        dp = metric.datapoint
        bins = dp.get('bins') or []
        counts = dp.get('counts') or []
        current = dict(zip(bins, counts))
        bin_total = sum(counts)
        count = dp.get('count')
        sum_val = dp.get('sum')

        last = self._metric_state.get(key)
        last_bins = last['bins'] if last is not None else None
        if last_bins is not None and bin_total >= last['bin_total']:
            delta_counts = {
                bin_value: bin_count - last_bins.get(bin_value, 0)
                for bin_value, bin_count in current.items()
                if bin_count > last_bins.get(bin_value, 0)}
        else:
            delta_counts = dict(current)

        d_count = d_sum = None
        if count is not None and sum_val is not None:
            last_count = last['count'] if last is not None else None
            last_sum = last['sum'] if last is not None else None
            if last_count is not None and last_sum is not None and count >= last_count:
                d_count = count - last_count
                d_sum = sum_val - last_sum
            else:
                d_count, d_sum = count, sum_val

        self._metric_state[key] = {
            'bin_total': bin_total, 'bins': current,
            'count': count, 'sum': sum_val}
        if not delta_counts and not d_count:
            return

        proto = self._proto_metric(
            signals_pb2.Metric.MetricType.HISTOGRAM_METRIC, metric, global_tags)
        proto_dp = proto.datapoints.add()
        for bin_value in sorted(delta_counts):
            proto_dp.histogram.bins.append(bin_value)
            proto_dp.histogram.counts.append(delta_counts[bin_value])
        if d_count is not None:
            proto_dp.histogram.count = int(d_count)
            proto_dp.histogram.sum = d_sum
        if dp.get('min') is not None:
            proto_dp.histogram.min = dp['min']
        if dp.get('max') is not None:
            proto_dp.histogram.max = dp['max']
        proto_dp.measurement_ts = dp.get('ts') or 0
        self._uploader.upload_metric(proto)

    def _collect_profile(self, key, metric, global_tags):
        # Frames carry string names everywhere up to this point; they are
        # encoded to integer frame ids only here, at upload time.
        dp = metric.datapoint
        frames = dp.get('frames') or {}
        samples = dp.get('samples') or {}

        last = self._metric_state.get(key)
        last_frames = last['frames'] if last is not None else None
        last_samples = last['samples'] if last is not None else {}
        reset = last_frames is not None and any(
            value < last_frames.get(frame_name, 0)
            for frame_name, value in frames.items())
        if last_frames is not None and not reset:
            delta_frames = {
                frame_name: value - last_frames.get(frame_name, 0)
                for frame_name, value in frames.items()
                if value > last_frames.get(frame_name, 0)}
            delta_samples = {
                frame_name: max(samples.get(frame_name, 0)
                                - last_samples.get(frame_name, 0), 0)
                for frame_name in delta_frames}
        else:
            delta_frames = dict(frames)
            delta_samples = dict(samples)
        self._metric_state[key] = {'frames': dict(frames), 'samples': dict(samples)}
        if not delta_frames:
            return

        proto = self._proto_metric(
            signals_pb2.Metric.MetricType.PROFILE_METRIC, metric, global_tags)
        proto_dp = proto.datapoints.add()
        for frame_name in sorted(delta_frames):
            frame_id = xxhash.xxh64(frame_name.encode('utf-8')).intdigest()
            frame = proto.frames.add()
            frame.frame_id = frame_id
            frame.name = frame_name
            proto_dp.profile.frame_ids.append(frame_id)
            proto_dp.profile.values.append(delta_frames[frame_name])
            proto_dp.profile.samples.append(delta_samples.get(frame_name, 0))
        proto_dp.measurement_ts = dp.get('ts') or 0
        self._uploader.upload_metric(proto)

    def _proto_metric(self, metric_type, metric, global_tags):
        proto = signals_pb2.Metric()
        proto.type = metric_type
        proto.name = metric.name
        all_tags = dict(global_tags)
        all_tags.update(metric.tags)
        for tag_key, tag_value in all_tags.items():
            tag = proto.tags.add()
            tag.key = str(tag_key)[:50]
            tag.value = str(tag_value)[:250]
        return proto

    def _collect_logs(self, log_store, global_tags):
        entries = [e for e in log_store.export() if e['ts'] > self._last_log_ts]
        if not entries:
            return
        self._last_log_ts = max(e['ts'] for e in entries)

        batches = {}
        for entry in entries:
            all_tags = dict(global_tags)
            all_tags.update(entry['tags'])
            batch_key = frozenset(all_tags.items())
            batch = batches.get(batch_key)
            if batch is None:
                batch = signals_pb2.LogBatch()
                for tag_key, tag_value in all_tags.items():
                    tag = batch.tags.add()
                    tag.key = str(tag_key)[:50]
                    tag.value = str(tag_value)[:250]
                batches[batch_key] = batch
            proto_entry = batch.log_entries.add()
            proto_entry.level = _LOG_LEVEL_MAP.get(
                entry['level'], signals_pb2.LogEntry.LogLevel.INFO_LEVEL)
            proto_entry.message = entry['message'] or ''
            proto_entry.exception = entry['exception'] or ''
            proto_entry.log_ts = entry['ts']

        for batch in batches.values():
            self._uploader.upload_log_batch(batch)

    def _collect_resources(self, resource_store, global_tags):
        for resource in resource_store.export():
            key = (resource['kind'], frozenset(resource['tags'].items()))
            if self._resource_state.get(key) == resource['last_seen_ts']:
                continue
            self._resource_state[key] = resource['last_seen_ts']

            proto = signals_pb2.Resource()
            proto.kind = resource['kind']
            all_tags = dict(global_tags)
            all_tags.update(resource['tags'])
            for tag_key, tag_value in all_tags.items():
                tag = proto.tags.add()
                tag.key = str(tag_key)[:50]
                tag.value = str(tag_value)[:250]
            for attr_name, attr_value in resource['attributes'].items():
                attr = proto.attributes.add()
                attr.name = attr_name
                attr.value = attr_value
            proto.first_seen_ts = resource['first_seen_ts']
            proto.last_seen_ts = resource['last_seen_ts']
            self._uploader.upload_resource(proto)
