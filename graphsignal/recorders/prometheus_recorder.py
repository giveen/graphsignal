import logging
import math
import time
from typing import Optional

try:
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

import graphsignal
import graphsignal.watcher
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

INITIAL_DETECT_DELAY_SEC = 2.0
MAX_DETECT_DELAY_SEC = 60.0
DEFAULT_METRICS_PATH = '/metrics'
DEFAULT_METRICS_HOST = '127.0.0.1'

# Prometheus labels used only for exposition (buckets, multiprocess); not useful
# as Graphsignal metric tags and they explode cardinality if kept.
_STRIP_LABELS = frozenset({'le', 'quantile', 'gt', 'pid'})


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def normalize_metrics_path(path: Optional[str]) -> str:
    normalized = (path or DEFAULT_METRICS_PATH).strip()
    if not normalized.startswith('/'):
        normalized = '/' + normalized
    return normalized


def format_metrics_host(host: Optional[str]) -> str:
    host = (host or DEFAULT_METRICS_HOST).strip()
    if ':' in host and not host.startswith('['):
        return f'[{host}]'
    return host


def build_metrics_endpoint(metrics_port: int, metrics_path: Optional[str] = None,
                           metrics_host: Optional[str] = None) -> str:
    path = normalize_metrics_path(metrics_path)
    host = format_metrics_host(metrics_host)
    return f'http://{host}:{int(metrics_port)}{path}'


class PrometheusRecorder(BaseRecorder):
    """Scrapes a single, known Prometheus metrics HTTP endpoint.

    The scrape port is resolved by the launcher (from `--metrics-port` or the
    engine's serving port) and passed in explicitly. The path defaults to
    `/metrics` (vLLM, SGLang); TensorRT-LLM uses `/prometheus/metrics`.
    We never enumerate or probe a process's other listening sockets —
    blindly connecting to them can corrupt internal IPC channels (e.g.
    TensorRT-LLM's ZeroMQ queues).
    """

    def __init__(self, root_pid=None, pid=None, args=None, metrics_port=None,
                 metrics_path: Optional[str] = None,
                 metrics_host: Optional[str] = None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._endpoint: Optional[str] = None
        if metrics_port is not None:
            self._endpoint = build_metrics_endpoint(
                metrics_port, metrics_path=metrics_path, metrics_host=metrics_host)
        self._verified: bool = False
        self._next_detect_ts: float = 0.0
        self._detect_delay_sec: float = INITIAL_DETECT_DELAY_SEC

    def setup(self):
        # Scraping is lazy; the first on_tick waits the initial delay since the
        # server may not be listening yet right after launch.
        self._next_detect_ts = time.time() + INITIAL_DETECT_DELAY_SEC

    def on_tick(self):
        if not PROMETHEUS_AVAILABLE or not HTTP_AVAILABLE:
            return
        if self._endpoint is None:
            return

        if not self._verified and time.time() < self._next_detect_ts:
            return

        try:
            body = self._fetch_metrics(self._endpoint)
        except Exception as exc:
            logger.debug('Failed to fetch %s: %s', self._endpoint, exc)
            # Server may still be starting up; back off and retry the same port.
            self._verified = False
            self._detect_delay_sec = min(self._detect_delay_sec * 2, MAX_DETECT_DELAY_SEC)
            self._next_detect_ts = time.time() + self._detect_delay_sec
            return

        if not self._verified:
            if not _looks_like_prometheus(body):
                self._detect_delay_sec = min(self._detect_delay_sec * 2, MAX_DETECT_DELAY_SEC)
                self._next_detect_ts = time.time() + self._detect_delay_sec
                return
            self._verified = True
            logger.debug('Prometheus /metrics endpoint confirmed: %s', self._endpoint)

        try:
            self._parse_and_emit(body)
        except Exception as exc:
            logger.error('Failed to parse Prometheus metrics: %s', exc, exc_info=True)

    @staticmethod
    def _fetch_metrics(url: str, timeout: float = 2.0) -> str:
        req = urllib.request.Request(url, headers={'Accept': 'text/plain'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            charset = resp.headers.get_content_charset() or 'utf-8'
            return data.decode(charset, errors='replace')

    def _parse_and_emit(self, body: str) -> None:
        # Scraped values pass through as-is: gauges and counters as the raw
        # cumulative values, histogram and summary families as one histogram
        # each — exact `_count`/`_sum`, plus `le` buckets converted to
        # non-cumulative bins where the family has them.
        watcher = graphsignal.watcher.watcher()
        now_ns = time.time_ns()

        for family in text_string_to_metric_families(body):
            name = family.name
            mtype = family.type

            sample_groups = {}
            bucket_groups = {}
            for sample in family.samples:
                # SGLang GaugeHistogram emits one gauge per bucket (gt/le labels).
                # These are heatmap buckets, not scalar gauges — skip them.
                if 'gt' in sample.labels:
                    continue
                labels = {k: v for k, v in sample.labels.items() if k not in _STRIP_LABELS}
                group_key = frozenset(labels.items())
                if sample.name == f'{name}_bucket' and 'le' in sample.labels:
                    bucket_groups.setdefault(group_key, []).append(
                        (sample.labels['le'], sample.value))
                    continue
                sample_groups.setdefault(group_key, {})[sample.name] = sample

            for group_key, sample_map in sample_groups.items():
                tags = dict(group_key)

                if mtype == 'gauge':
                    s = sample_map.get(name)
                    if s is not None and _is_finite_number(s.value):
                        watcher.set_gauge(name=name, tags=tags, value=s.value, measurement_ts=now_ns)

                elif mtype == 'counter':
                    s = sample_map.get(f'{name}_total') or sample_map.get(name)
                    if s is not None and _is_finite_number(s.value):
                        watcher.set_counter(name=name, tags=tags, total=s.value, measurement_ts=now_ns)

                elif mtype in ('histogram', 'summary'):
                    # Both families land as ONE histogram. A Prometheus summary
                    # has _count/_sum and quantiles we do not keep, so it
                    # arrives with no bins; a histogram adds its `le` buckets on
                    # top of the same totals. One name, one type, whichever the
                    # engine happens to expose.
                    c = sample_map.get(f'{name}_count')
                    su = sample_map.get(f'{name}_sum')
                    if c is not None and su is not None and _is_finite_number(c.value) and _is_finite_number(su.value):
                        bins, counts = _buckets_to_bins(bucket_groups.get(group_key))
                        watcher.set_histogram(name=name, tags=tags,
                                              bins=bins, counts=counts,
                                              count=int(c.value), sum_val=su.value,
                                              measurement_ts=now_ns)


def _buckets_to_bins(buckets):
    """Convert Prometheus cumulative `le` buckets to non-cumulative bins.

    Bin values are the buckets' upper bounds; the `+Inf` remainder is folded
    into a final bin at the largest finite bound. Returns (None, None) when
    no usable buckets exist.
    """
    if not buckets:
        return None, None

    finite = []
    inf_count = None
    for le, value in buckets:
        if not _is_finite_number(value):
            continue
        try:
            bound = float(le)
        except (TypeError, ValueError):
            continue
        if math.isinf(bound):
            inf_count = value
            continue
        finite.append((bound, value))

    if not finite:
        return None, None
    finite.sort(key=lambda b: b[0])

    bins = []
    counts = []
    previous_cumulative = 0.0
    for bound, cumulative in finite:
        delta = cumulative - previous_cumulative
        previous_cumulative = cumulative
        if delta > 0:
            bins.append(bound)
            counts.append(int(delta))
    if inf_count is not None:
        remainder = inf_count - previous_cumulative
        if remainder > 0:
            if bins and bins[-1] == finite[-1][0]:
                counts[-1] += int(remainder)
            else:
                bins.append(finite[-1][0])
                counts.append(int(remainder))

    if not bins:
        return None, None
    return bins, counts


def _looks_like_prometheus(body: str) -> bool:
    if not body:
        return False
    # OpenMetrics/Prometheus payloads begin with HELP/TYPE comments or metric samples.
    for line in body.splitlines():
        if not line:
            continue
        if line.startswith('# HELP') or line.startswith('# TYPE'):
            return True
        if line.startswith('#'):
            continue
        if ' ' in line and not line.startswith('<'):
            return True
        return False
    return False
