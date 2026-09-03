import logging
import threading
import time

logger = logging.getLogger('graphsignal')

GAUGE = 'gauge'
COUNTER = 'counter'
SUMMARY = 'summary'
HISTOGRAM = 'histogram'
PROFILE = 'profile'

# Metrics not updated for this long are removed entirely.
METRIC_EXPIRY_NS = 600 * 1_000_000_000
# Cleanup runs at most this often, checked on set_* calls.
CLEANUP_INTERVAL_NS = 60 * 1_000_000_000
MAX_METRICS = 5000
MAX_PROFILE_FRAMES = 250

MAX_TAG_KEY_LEN = 50
MAX_TAG_VALUE_LEN = 250


class Metric:
    __slots__ = ('name', 'type', 'tags', 'datapoint')

    def __init__(self, name, metric_type, tags=None):
        self.name = name
        self.type = metric_type
        self.tags = tags or {}
        # The latest snapshot, overwritten on every set_*:
        # gauge:     {'ts', 'value'}
        # counter:   {'ts', 'total'}                      (cumulative)
        # summary:   {'ts', 'count', 'sum', 'sum2'}       (cumulative;
        #            None fields = not measured)
        # histogram: {'ts', 'bins', 'counts'}             (cumulative)
        # profile:   {'ts', 'frames', 'samples'}          (cumulative value and
        #            sample count per frame name, at most MAX_PROFILE_FRAMES)
        self.datapoint = None


class MetricStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {}
        self._last_cleanup_ts = 0
        self._last_cap_log_ts = 0

    def set_gauge(self, name, value, measurement_ts, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if value is None:
            raise ValueError('Gauge value cannot be None')

        with self._lock:
            metric = self._get_metric(GAUGE, name, tags=tags)
            if metric is None:
                return
            metric.datapoint = {'ts': measurement_ts, 'value': value}
            self._maybe_cleanup(time.time_ns())

    def set_counter(self, name, total, measurement_ts, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if total is None:
            raise ValueError('Counter total cannot be None')

        with self._lock:
            metric = self._get_metric(COUNTER, name, tags=tags)
            if metric is None:
                return
            metric.datapoint = {'ts': measurement_ts, 'total': total}
            self._maybe_cleanup(time.time_ns())

    def set_summary(self, name, count, sum_val, sum2_val=None,
                    measurement_ts=None, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if count is None:
            raise ValueError('Summary count cannot be None')
        if sum_val is None:
            raise ValueError('Summary sum cannot be None')

        with self._lock:
            metric = self._get_metric(SUMMARY, name, tags=tags)
            if metric is None:
                return
            metric.datapoint = {
                'ts': measurement_ts,
                'count': count,
                'sum': sum_val,
                'sum2': sum2_val,
            }
            self._maybe_cleanup(time.time_ns())

    def set_histogram(self, name, bins, counts, measurement_ts=None,
                      tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if not bins or not counts or len(bins) != len(counts):
            raise ValueError('Histogram requires bins and counts of equal length')

        with self._lock:
            metric = self._get_metric(HISTOGRAM, name, tags=tags)
            if metric is None:
                return
            metric.datapoint = {
                'ts': measurement_ts,
                'bins': list(bins),
                'counts': list(counts),
            }
            self._maybe_cleanup(time.time_ns())

    def set_profile(self, name, frames, samples=None, measurement_ts=None,
                    tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if frames is None:
            raise ValueError('Profile frames cannot be None')

        # Frame values and sample counts are unsigned 64-bit cumulative
        # counters end to end — the probes record them as uint64 and the
        # upload proto declares uint64 — so anything float is truncated here,
        # at the door.
        frames = {frame_name: int(value) for frame_name, value in frames.items()}
        if len(frames) > MAX_PROFILE_FRAMES:
            top = sorted(frames.items(), key=lambda kv: kv[1], reverse=True)
            frames = dict(top[:MAX_PROFILE_FRAMES])
        samples = {frame_name: int(count)
                   for frame_name, count in (samples or {}).items()
                   if frame_name in frames}

        with self._lock:
            metric = self._get_metric(PROFILE, name, tags=tags)
            if metric is None:
                return
            metric.datapoint = {
                'ts': measurement_ts, 'frames': frames, 'samples': samples}
            self._maybe_cleanup(time.time_ns())

    def export(self):
        """Snapshot of all metrics; never resets the store."""
        with self._lock:
            snapshot = []
            for metric in self._metrics.values():
                copied = Metric(metric.name, metric.type, tags=dict(metric.tags))
                if metric.datapoint is not None:
                    copied.datapoint = dict(metric.datapoint)
                    for per_frame in ('frames', 'samples'):
                        if copied.datapoint.get(per_frame) is not None:
                            copied.datapoint[per_frame] = dict(copied.datapoint[per_frame])
                snapshot.append(copied)
            return snapshot

    def clear(self):
        with self._lock:
            self._metrics.clear()

    def _get_metric(self, metric_type, name, tags=None):
        norm_tags = {}
        if tags:
            for key, value in tags.items():
                norm_tags[str(key)[:MAX_TAG_KEY_LEN]] = str(value)[:MAX_TAG_VALUE_LEN]

        metric_key = (name, metric_type, frozenset(norm_tags.items()))
        metric = self._metrics.get(metric_key)
        if metric is None:
            if len(self._metrics) >= MAX_METRICS:
                now = time.time_ns()
                if now - self._last_cap_log_ts >= CLEANUP_INTERVAL_NS:
                    self._last_cap_log_ts = now
                    logger.debug('Max metrics reached (%d), dropping new metric: %s',
                                 MAX_METRICS, name)
                return None
            metric = Metric(name, metric_type, tags=norm_tags)
            self._metrics[metric_key] = metric
        return metric

    def _maybe_cleanup(self, now):
        if now - self._last_cleanup_ts < CLEANUP_INTERVAL_NS:
            return
        self._last_cleanup_ts = now

        expiry_cutoff = now - METRIC_EXPIRY_NS

        expired_keys = []
        for metric_key, metric in self._metrics.items():
            last_ts = metric.datapoint.get('ts') if metric.datapoint else None
            if last_ts is None or last_ts < expiry_cutoff:
                expired_keys.append(metric_key)
        for metric_key in expired_keys:
            del self._metrics[metric_key]
