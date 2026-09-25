"""The watcher's local HTTP routes and the /signals payload.

`GET /signals` returns the payload built from the in-memory stores: metrics
(with per-type statistics), recent errors, resources, and the watcher's global
tags as a top-level `context` object. Values follow the convention:
null = not measured, 0 = measured zero.
"""

import http.server
import json
import logging
import math
import threading
import time

from graphsignal.signals import metrics as metrics_module
from graphsignal.version import __version__

logger = logging.getLogger('graphsignal')

DEFAULT_LISTEN_HOST = '127.0.0.1'
DEFAULT_LISTEN_PORT = 18259


def quantiles_from_bins(bins, counts, qs=(('p50', 0.5), ('p95', 0.95)),
                        lower=None, upper=None):
    """Nearest-rank quantiles over cumulative histogram bins.

    Quantile values are bin values, so resolution is bounded by the writer's
    bin grid. That grid extends past the observations at both ends, so a
    quantile can land below the measured min or above the measured max — the
    payload would then report p50 > max. `lower`/`upper` are the exact measured
    bounds, when the source reports them; a quantile outside them is clamped
    back onto the range the samples actually occupied.

    Returns None when bins are absent or empty.
    """
    if not bins or not counts or len(bins) != len(counts):
        return None
    total_count = sum(counts)
    if not total_count:
        return None

    quantiles = {}
    for name, q in qs:
        rank = max(1, math.ceil(q * total_count))
        cumulative = 0
        value = bins[-1]
        for bin_value, count in zip(bins, counts):
            cumulative += count
            if cumulative >= rank:
                value = bin_value
                break
        if lower is not None and value < lower:
            value = lower
        if upper is not None and value > upper:
            value = upper
        quantiles[name] = value
    return quantiles


def _gauge_stats(dp):
    return {'value': dp.get('value')}


def _counter_stats(dp):
    return {'total': dp.get('total')}


def _histogram_stats(dp):
    """count/sum/min/max exactly as measured; mean, p50, p95 derived.

    The exact totals win over the bins for the mean: a bin-weighted mean takes
    every observation at its bin's lower bound, and the writer's own sum did
    not. Quantiles have no such alternative — they only exist where there are
    bins, and are null for a source that reports totals alone. They are clamped
    to the measured min/max so a coarse bin grid cannot report a p50 above the
    maximum sample.
    """
    bins = dp.get('bins')
    counts = dp.get('counts')
    count = dp.get('count')
    total = dp.get('sum')
    minimum = dp.get('min')
    maximum = dp.get('max')
    mean = None
    if count is not None and total is not None:
        if count:
            mean = total / count
    elif bins and counts and len(bins) == len(counts):
        total_count = sum(counts)
        if total_count:
            mean = sum(b * c for b, c in zip(bins, counts)) / total_count
    quantiles = quantiles_from_bins(bins, counts, lower=minimum, upper=maximum) or {}
    return {
        'count': count,
        'sum': total,
        'min': minimum,
        'max': maximum,
        'mean': mean,
        'p50': quantiles.get('p50'),
        'p95': quantiles.get('p95'),
    }


def _profile_stats(dp):
    frames = dp.get('frames') or {}
    samples = dp.get('samples') or {}
    return {
        'frames': [
            {'name': name, 'value': value, 'samples': samples.get(name, 0)}
            for name, value in sorted(
                frames.items(), key=lambda kv: kv[1], reverse=True)],
    }


def build_payload():
    import graphsignal.watcher as gwatcher

    payload = {
        'profiler': {'version': __version__},
        'payload_ns': time.time_ns(),
    }

    if not gwatcher.is_configured():
        payload['start_ns'] = None
        payload['context'] = {}
        payload['metrics'] = []
        payload['errors'] = []
        payload['resources'] = []
        return payload

    watcher = gwatcher.watcher()

    # Counters are cumulative since the instance started; start_ns names the
    # accumulation base explicitly, like the platform's windowed snapshot does.
    payload['start_ns'] = watcher.start_ns()
    payload['context'] = watcher.tags()

    metric_records = []
    for metric in watcher.metric_store().export():
        if metric.datapoint is None:
            continue
        if metric.type == metrics_module.GAUGE:
            stats = _gauge_stats(metric.datapoint)
        elif metric.type == metrics_module.COUNTER:
            stats = _counter_stats(metric.datapoint)
        elif metric.type == metrics_module.HISTOGRAM:
            stats = _histogram_stats(metric.datapoint)
        else:
            stats = _profile_stats(metric.datapoint)
        metric_records.append({
            'name': metric.name,
            'type': metric.type,
            'tags': metric.tags,
            'stats': stats,
            'updated_ns': metric.datapoint.get('ts'),
        })
    payload['metrics'] = sorted(
        metric_records, key=lambda r: (r['name'], r['type'], sorted(r['tags'].items())))

    payload['errors'] = [
        {
            'level': entry.get('level'),
            'message': entry.get('message'),
            'exception': entry.get('exception'),
            'tags': entry.get('tags'),
            'log_ns': entry.get('ts'),
        }
        for entry in watcher.log_store().last_entries(min_level='warning', limit=10)]

    # The stores keep nanoseconds; resource lifetimes read fine at seconds
    # (the `_ts` suffix means seconds on the wire).
    payload['resources'] = [
        {**resource,
         'first_seen_ts': resource['first_seen_ts'] // 1_000_000_000,
         'last_seen_ts': resource['last_seen_ts'] // 1_000_000_000}
        for resource in watcher.resource_store().export()]

    return payload


class _SignalsRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/signals':
                self._respond_json(200, build_payload())
            elif self.path == '/health':
                self._respond_json(200, {'status': 'ok'})
            else:
                self._respond_json(404, {'error': 'not found'})
        except Exception:
            logger.error('Error handling signals request', exc_info=True)
            try:
                self._respond_json(500, {'error': 'internal error'})
            except Exception:
                pass

    def _respond_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.debug('signals endpoint: ' + format, *args)


class SignalsEndpoint:
    # A busy port is transient: the other profiler is usually a concurrent
    # `graphsignal-run` that exits within seconds. Retry with backoff so the
    # endpoint starts serving as soon as the port frees, rather than staying
    # dead for the rest of this instance's life.
    BIND_RETRY_INITIAL_DELAY = 0.1
    BIND_RETRY_MAX_DELAY = 2.0

    def __init__(self, host=DEFAULT_LISTEN_HOST, port=DEFAULT_LISTEN_PORT,
                 on_bind_event=None):
        self._host = str(host)
        self._port = int(port)
        self._on_bind_event = on_bind_event
        self._server = None
        self._serve_thread = None
        self._retry_thread = None
        self._reported_failure = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def host(self):
        return self._host

    def port(self):
        return self._port

    def is_running(self):
        with self._lock:
            return self._server is not None

    def setup(self):
        """Bind the endpoint, falling back to background retries.

        Never raises and never gives up: a port held by another instance is
        reported once, then retried until `shutdown()`. The first failure also
        goes to the `on_bind_event` callback, because the log store that would
        otherwise carry it is only readable over the endpoint that did not
        come up.
        """
        if self._bind_and_serve():
            return
        if self._stop_event.is_set():
            return
        self._retry_thread = threading.Thread(
            target=self._retry_bind, name='graphsignal-bind-retry', daemon=True)
        self._retry_thread.start()

    def _bind_and_serve(self):
        """Try once to bind and start serving. Returns True on success; a
        failure is reported once and the caller decides whether to retry."""
        try:
            server = http.server.ThreadingHTTPServer(
                (self._host, self._port), _SignalsRequestHandler)
        except OSError as exc:
            self._report_bind_failure(exc, first=not self._reported_failure)
            return False

        server.daemon_threads = True
        with self._lock:
            if self._stop_event.is_set():
                # Shut down raced the successful bind; drop the socket.
                server.server_close()
                return False
            self._server = server
        self._serve_thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05),
            name='graphsignal-signals', daemon=True)
        self._serve_thread.start()
        logger.debug('Signals endpoint listening on %s:%d', self._host, self._port)
        self._notify('bound', '%s:%d' % (self._host, self._port))
        return True

    def _report_bind_failure(self, exc, first):
        """Log and announce a failed bind. The first failure is an error; the
        retries behind it are debug only, so neither the log store nor the
        parent process gets one identical line per attempt."""
        detail = '%s:%d: %s' % (self._host, self._port, exc)
        if not first:
            logger.debug('Signals endpoint still cannot bind %s', detail)
            return
        self._reported_failure = True
        logger.error('Signals endpoint failed to bind %s; retrying', detail)
        self._notify('bind_failed', detail)

    def _notify(self, event, detail):
        """Tell the listener about a bind state change. Best effort: the
        callback is the only channel out of the profiler subprocess, so a
        failure here must not take the watcher down."""
        if self._on_bind_event is None:
            return
        try:
            self._on_bind_event(event, detail)
        except Exception:
            logger.debug('Signals endpoint bind listener failed', exc_info=True)

    def _retry_bind(self):
        delay = self.BIND_RETRY_INITIAL_DELAY
        while not self._stop_event.wait(delay):
            if self._bind_and_serve():
                return
            if self._stop_event.is_set():
                return
            delay = min(delay * 2, self.BIND_RETRY_MAX_DELAY)

    def shutdown(self):
        self._stop_event.set()
        if self._retry_thread is not None:
            self._retry_thread.join(timeout=2.0)
            self._retry_thread = None
        with self._lock:
            server = self._server
            self._server = None
        if server is None:
            return
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            logger.debug('Error shutting down signals endpoint', exc_info=True)
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=2.0)
            self._serve_thread = None
