import os
import socket
import time


def free_port() -> int:
    """Pick a currently-free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Wait for a background condition to become true (e.g. the /signals
    endpoint's bind retry loop picking up a freed port). True if it became
    true before the timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def clear_graphsignal_env() -> None:
    """Remove GRAPHSIGNAL_* env vars so tests see only explicit config."""
    for key in list(os.environ.keys()):
        if key.startswith('GRAPHSIGNAL_'):
            del os.environ[key]


def configure_test_watcher(**kwargs):
    """Configure the watcher for a test: explicit free signals port, no env
    leakage, auto tick disabled. Returns the watcher singleton."""
    import graphsignal.watcher

    clear_graphsignal_env()
    # No test may phone home. The watcher fires a version check the moment it
    # sees its target, so every test that configures one would otherwise reach
    # api.graphsignal.com; the version check's own tests call
    # start_version_check directly against a local server.
    os.environ['GRAPHSIGNAL_DISABLE_VERSION_CHECK'] = '1'
    kwargs.setdefault('listen_port', free_port())
    kwargs.setdefault('debug_mode', True)
    graphsignal.watcher.configure(**kwargs)
    watcher = graphsignal.watcher.watcher()
    watcher._auto_tick = False
    # Suppress the metric store's timestamp-based cleanup so tests may use
    # small synthetic measurement_ts values.
    watcher.metric_store()._last_cleanup_ts = time.time_ns()
    return watcher


def shutdown_test_watcher() -> None:
    import graphsignal.watcher
    graphsignal.watcher.shutdown()


def find_metric(exported, name, tags=None, metric_type=None):
    """Find a metric in a MetricStore.export() snapshot by name (and tags
    subset and/or type, when given)."""
    for metric in exported:
        if metric.name != name:
            continue
        if metric_type is not None and metric.type != metric_type:
            continue
        if tags is not None and any(
                metric.tags.get(k) != v for k, v in tags.items()):
            continue
        return metric
    return None
